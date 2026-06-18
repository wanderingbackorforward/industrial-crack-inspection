"""Train crack segmentation model."""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_crasam, build_unet, build_deeplabv3plus
from data.dataset import CrackSegmentationDataset
from data.transforms import build_segmentation_transform
from utils import load_config, segmentation_metrics


MODEL_BUILDERS = {
    "crasam": build_crasam,
    "unet": build_unet,
    "deeplabv3plus": build_deeplabv3plus,
}


def train_one_epoch(model, optimizer, loader, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="train"):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with autocast(enabled=scaler is not None):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), masks.float())
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    total_miou = 0.0
    total_f1 = 0.0
    count = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        m = segmentation_metrics(logits.squeeze(1), masks, threshold=threshold)
        total_miou += m["miou"]
        total_f1 += m["f1"]
        count += 1
    return {"miou": total_miou / count, "f1": total_f1 / count}


def main(args):
    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_ds = CrackSegmentationDataset(
        cfg["data"]["train_root"], transforms=build_segmentation_transform(cfg, is_train=True)
    )
    val_ds = CrackSegmentationDataset(
        cfg["data"]["val_root"], transforms=build_segmentation_transform(cfg, is_train=False)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model_name = cfg["model"]["name"]
    model = MODEL_BUILDERS[model_name](cfg).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = GradScaler() if cfg["training"].get("mixed_precision", False) else None

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    best_miou = -1.0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        metrics = evaluate(model, val_loader, device, threshold=cfg["metrics"].get("threshold", 0.5))
        scheduler.step()
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, miou={metrics['miou']:.4f}, f1={metrics['f1']:.4f}")
        if cfg["checkpoint"].get("save_best", True) and metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)

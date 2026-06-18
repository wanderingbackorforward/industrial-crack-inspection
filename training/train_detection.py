"""Train Faster R-CNN crack detector."""
import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_faster_rcnn
from data.dataset import CrackDetectionDataset
from data.transforms import build_detection_transform
from utils import load_config, detection_metrics


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, device, iou_thresholds=None):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            preds.extend([{k: v.cpu() for k, v in o.items()} for o in outputs])
            gts.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    return detection_metrics(preds, gts, iou_thresholds=iou_thresholds)


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = cfg["training"].get("optimizer", "sgd").lower()
    lr = cfg["training"]["lr"]
    if opt_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=cfg["training"].get("momentum", 0.9),
            weight_decay=cfg["training"].get("weight_decay", 0.0005),
        )
    return torch.optim.AdamW(params, lr=lr, weight_decay=cfg["training"].get("weight_decay", 0.01))


def build_scheduler(optimizer, cfg):
    sched = cfg["training"].get("lr_scheduler", "step")
    if sched == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["training"]["lr_decay_epochs"],
            gamma=cfg["training"].get("lr_decay_gamma", 0.33),
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])


def main(args):
    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_ds = CrackDetectionDataset(
        cfg["data"]["train_root"], transforms=build_detection_transform(cfg, is_train=True)
    )
    val_ds = CrackDetectionDataset(
        cfg["data"]["val_root"], transforms=build_detection_transform(cfg, is_train=False)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    model = build_faster_rcnn(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    best_map = -1.0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        metrics = evaluate(model, val_loader, device, iou_thresholds=cfg["metrics"].get("iou_thresholds"))
        map_score = metrics.get("map", metrics.get("map_50", 0.0))
        scheduler.step()
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f}, "
            f"mAP={map_score:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        if cfg["checkpoint"].get("save_best", True) and map_score > best_map:
            best_map = map_score
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args)

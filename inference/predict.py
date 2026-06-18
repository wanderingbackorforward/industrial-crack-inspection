"""End-to-end inference: detection + segmentation + parameter extraction."""
import argparse
import os
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_faster_rcnn, build_crasam
from data.transforms import Normalize
from utils import load_config, extract_crack_params, box_counting_dimension


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_image(path, size=1024):
    img = Image.open(path).convert("RGB")
    orig_size = img.size
    img = img.resize((size, size))
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    normalize = Normalize()
    tensor, _ = normalize(tensor, None)
    return tensor.unsqueeze(0), orig_size


def detect(model, image_tensor, device, score_thresh=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor.squeeze(0).to(device)])
    out = outputs[0]
    keep = out["scores"] > score_thresh
    return {
        "boxes": out["boxes"][keep].cpu().numpy(),
        "labels": out["labels"][keep].cpu().numpy(),
        "scores": out["scores"][keep].cpu().numpy(),
    }


def segment(model, image_tensor, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        prob = logits.sigmoid().squeeze().cpu().numpy()
    return (prob > threshold).astype(np.uint8), prob


def visualize(image_path, boxes, mask, output_path, pixel_size_mm=1.0):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, x2 = x1 * W / 1024, x2 * W / 1024
        y1, y2 = y1 * H / 1024, y2 * H / 1024
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
        axes[1].add_patch(rect)
    axes[1].set_title(f"Detection ({len(boxes)} cracks)")
    axes[1].axis("off")

    axes[2].imshow(img, alpha=0.6)
    axes[2].imshow(mask, alpha=0.4, cmap="hot")
    axes[2].set_title("Segmentation")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    det_cfg = load_config(args.det_config)
    seg_cfg = load_config(args.seg_config)

    det_model = build_faster_rcnn(det_cfg).to(device)
    det_model.load_state_dict(torch.load(args.det_weights, map_location=device))

    seg_model = build_crasam(seg_cfg).to(device)
    seg_model.load_state_dict(torch.load(args.seg_weights, map_location=device))

    image_tensor, orig_size = load_image(args.image, size=det_cfg["data"]["image_size"])

    detections = detect(det_model, image_tensor, device)
    mask, _ = segment(seg_model, image_tensor, device)

    params = extract_crack_params(mask, pixel_size_mm=args.pixel_size_mm)
    fractal_dim = box_counting_dimension(mask)

    report = {
        "num_cracks": int(len(detections["boxes"])),
        "detections": [
            {"box": box.tolist(), "label": int(label), "score": float(score)}
            for box, label, score in zip(
                detections["boxes"], detections["labels"], detections["scores"]
            )
        ],
        "segmentation": {
            "mask_pixels": int(mask.sum()),
            "crack_ratio": float(mask.sum() / mask.size),
        },
        "parameters": params,
        "fractal_dimension": float(fractal_dim),
    }

    os.makedirs(args.output, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    with open(os.path.join(args.output, f"{base}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Resize mask back to original size
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(orig_size)
    mask_img.save(os.path.join(args.output, f"{base}_mask.png"))

    visualize(args.image, detections["boxes"], mask, os.path.join(args.output, f"{base}_vis.png"), args.pixel_size_mm)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--det-config", default="configs/detection/faster_rcnn_resnet50.yaml")
    parser.add_argument("--det-weights", required=True)
    parser.add_argument("--seg-config", default="configs/segmentation/crasam.yaml")
    parser.add_argument("--seg-weights", required=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--pixel-size-mm", type=float, default=1.0)
    args = parser.parse_args()
    main(args)

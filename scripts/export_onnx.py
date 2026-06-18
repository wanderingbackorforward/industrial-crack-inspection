"""Export segmentation model to ONNX for edge deployment."""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.onnx
from models import build_crasam
from utils import load_config


def export(cfg_path, weights_path, output_path):
    cfg = load_config(cfg_path)
    model = build_crasam(cfg)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"])
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["mask_logits"],
        dynamic_axes={"image": {0: "batch"}, "mask_logits": {0: "batch"}},
        opset_version=11,
    )
    print(f"ONNX exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation/crasam.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="checkpoints/segmentation/crasam.onnx")
    args = parser.parse_args()
    export(args.config, args.weights, args.output)

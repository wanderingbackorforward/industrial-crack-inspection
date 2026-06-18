import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models import build_faster_rcnn, build_crasam


def test_faster_rcnn_forward():
    cfg = {
        "model": {
            "name": "faster_rcnn",
            "backbone": "resnet50_fpn",
            "num_classes": 5,
            "pretrained": False,
            "rpn": {
                "anchor_sizes": [16, 32, 64, 128, 256],
                "aspect_ratios": [0.5, 1.0, 2.0],
            },
            "box": {"score_thresh": 0.5, "nms_thresh": 0.5, "detections_per_img": 100},
        }
    }
    model = build_faster_rcnn(cfg)
    model.eval()
    x = torch.rand(1, 3, 512, 512)
    out = model(x)
    assert isinstance(out, list)
    assert "boxes" in out[0]


def test_crasam_forward():
    cfg = {
        "model": {
            "name": "crasam",
            "img_embed_dim": 64,
            "prompt_dim": 64,
            "num_points": 4,
            "surrogate_encoder": True,
        }
    }
    model = build_crasam(cfg)
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_crack_params():
    from utils import extract_crack_params, box_counting_dimension
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 20:80] = 1
    params = extract_crack_params(mask, pixel_size_mm=1.0)
    assert params["length_mm"] > 0
    assert params["mean_width_mm"] > 0
    dim = box_counting_dimension(mask)
    assert 0.0 <= dim <= 2.0

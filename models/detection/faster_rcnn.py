"""Faster R-CNN crack detector with configurable backbones."""
from typing import Dict, Any
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign


_BACKBONE_OUTPUT_CHANNELS = {
    "resnet50_fpn": 256,
    "resnet101_fpn": 256,
    "efficientnet_v2_s": 1280,
    "efficientnet_v2_m": 1280,
    "vgg16": 512,
}


def _build_resnet_fpn_backbone(name: str, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    if name == "resnet50_fpn":
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            "resnet50", weights=weights
        )
    elif name == "resnet101_fpn":
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            "resnet101", weights=weights
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return backbone, _BACKBONE_OUTPUT_CHANNELS[name]


def _build_efficientnet_v2_backbone(name: str, pretrained: bool = True):
    from torchvision.models import efficientnet_v2_s, efficientnet_v2_m
    weights = "DEFAULT" if pretrained else None
    if name == "efficientnet_v2_s":
        net = efficientnet_v2_s(weights=weights).features
    elif name == "efficientnet_v2_m":
        net = efficientnet_v2_m(weights=weights).features
    else:
        raise ValueError(f"Unknown backbone: {name}")
    net.out_channels = _BACKBONE_OUTPUT_CHANNELS[name]
    return net, _BACKBONE_OUTPUT_CHANNELS[name]


def _build_vgg16_backbone(pretrained: bool = True):
    from torchvision.models import vgg16_bn
    weights = "DEFAULT" if pretrained else None
    backbone = vgg16_bn(weights=weights).features
    backbone.out_channels = 512
    return backbone, 512


def build_faster_rcnn(cfg: Dict[str, Any]) -> FasterRCNN:
    backbone_name = cfg["model"]["backbone"]
    num_classes = cfg["model"]["num_classes"]
    pretrained = cfg["model"].get("pretrained", True)

    if backbone_name.startswith("resnet"):
        backbone, out_channels = _build_resnet_fpn_backbone(backbone_name, pretrained)
        box_roi_pool = None
    elif backbone_name.startswith("efficientnet"):
        backbone, out_channels = _build_efficientnet_v2_backbone(backbone_name, pretrained)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )
    elif backbone_name == "vgg16":
        backbone, out_channels = _build_vgg16_backbone(pretrained)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    anchor_cfg = cfg["model"].get("rpn", {})
    anchor_sizes = tuple(anchor_cfg.get("anchor_sizes", (32, 64, 128, 256, 512)))
    aspect_ratios = anchor_cfg.get(
        "aspect_ratios", (0.5, 1.0, 2.0)
    )
    # AnchorGenerator expects tuple of tuples
    aspect_ratios_per_anchor = (tuple(aspect_ratios),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        sizes=(anchor_sizes,) * len(anchor_sizes),
        aspect_ratios=aspect_ratios_per_anchor,
    )

    box_cfg = cfg["model"].get("box", {})
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_roi_pool=box_roi_pool,
        box_score_thresh=box_cfg.get("score_thresh", 0.5),
        box_nms_thresh=box_cfg.get("nms_thresh", 0.5),
        box_detections_per_img=box_cfg.get("detections_per_img", 300),
    )

    # Replace box predictor in case of custom num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == "__main__":
    cfg = {
        "model": {
            "name": "faster_rcnn",
            "backbone": "resnet50_fpn",
            "num_classes": 5,
            "pretrained": False,
            "rpn": {
                "anchor_sizes": [16, 32, 64, 128, 256],
                "aspect_ratios": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            },
            "box": {"score_thresh": 0.5, "nms_thresh": 0.5, "detections_per_img": 300},
        }
    }
    model = build_faster_rcnn(cfg)
    x = torch.rand(2, 3, 1024, 1024)
    model.eval()
    out = model(x)
    print("Boxes:", out[0]["boxes"].shape, "Labels:", out[0]["labels"].shape)

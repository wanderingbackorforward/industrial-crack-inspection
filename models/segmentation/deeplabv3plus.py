"""DeepLabV3+ segmentation baseline using segmentation-models-pytorch."""
from typing import Dict, Any


def build_deeplabv3plus(cfg: Dict[str, Any]):
    import segmentation_models_pytorch as smp
    encoder_name = cfg["model"].get("encoder", "resnet50")
    in_channels = cfg["model"].get("in_channels", 3)
    classes = cfg["model"].get("classes", 1)
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )

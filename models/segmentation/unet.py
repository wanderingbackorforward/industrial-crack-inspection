"""U-Net segmentation baseline using segmentation-models-pytorch."""
from typing import Dict, Any


def build_unet(cfg: Dict[str, Any]):
    import segmentation_models_pytorch as smp
    encoder_name = cfg["model"].get("encoder", "vgg13")
    in_channels = cfg["model"].get("in_channels", 3)
    classes = cfg["model"].get("classes", 1)
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )

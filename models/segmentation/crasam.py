"""CraSAM: a lightweight prompt adapter on top of SAM for crack segmentation.

This is a clean-room implementation for demonstration. It uses a tiny CNN prompt
encoder instead of SAM's full prompt encoder and fine-tunes only the mask decoder
and prompt adapter, keeping the SAM image encoder frozen.
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptAdapter(nn.Module):
    """Learnable prompt tokens conditioned on image embeddings."""

    def __init__(self, img_embed_dim: int, prompt_dim: int, num_points: int = 4):
        super().__init__()
        self.num_points = num_points
        # A small conv head that produces point prompts from image embedding.
        self.point_head = nn.Sequential(
            nn.Conv2d(img_embed_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_points, 1),
        )
        self.embed = nn.Sequential(
            nn.Linear(2, prompt_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_dim // 2, prompt_dim),
        )

    def forward(self, image_embedding: torch.Tensor):
        # image_embedding: (B, C, H, W)
        heatmap = self.point_head(image_embedding)  # (B, num_points, H, W)
        B, N, H, W = heatmap.shape
        flat = heatmap.view(B, N, -1)
        idx = flat.argmax(dim=-1)  # (B, N)
        y = (idx // W).float() / H
        x = (idx % W).float() / W
        coords = torch.stack([x, y], dim=-1)  # (B, N, 2)
        return self.embed(coords)  # (B, N, prompt_dim)


class MaskDecoderAdapter(nn.Module):
    """Simplified mask decoder for binary crack segmentation."""

    def __init__(self, img_embed_dim: int, prompt_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(img_embed_dim + prompt_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, image_embedding: torch.Tensor, prompt_embedding: torch.Tensor):
        B, C, H, W = image_embedding.shape
        prompt_map = prompt_embedding.mean(dim=1).view(B, -1, 1, 1).expand(B, -1, H, W)
        x = torch.cat([image_embedding, prompt_map], dim=1)
        logits = self.fusion(x)
        return logits


class CraSAM(nn.Module):
    """Crack-adapted SAM.

    image_encoder can be a real SAM ViT encoder or a simple ViT/CNN surrogate
    controlled by `surrogate` flag. For the surrogate version the model is fully
    trainable and does not require SAM weights.
    """

    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        img_embed_dim: int = 256,
        prompt_dim: int = 256,
        num_points: int = 4,
        freeze_image_encoder: bool = True,
        surrogate_encoder: bool = True,
    ):
        super().__init__()
        self.surrogate_encoder = surrogate_encoder
        if image_encoder is None:
            if surrogate_encoder:
                # A simple surrogate CNN encoder for training without SAM weights.
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, img_embed_dim, 3, padding=1),
                    nn.BatchNorm2d(img_embed_dim),
                    nn.ReLU(inplace=True),
                )
            else:
                raise ValueError("Provide a SAM image_encoder or set surrogate_encoder=True")
        else:
            self.image_encoder = image_encoder

        if freeze_image_encoder and not surrogate_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.prompt_adapter = PromptAdapter(img_embed_dim, prompt_dim, num_points)
        self.mask_decoder = MaskDecoderAdapter(img_embed_dim, prompt_dim)

    def forward(self, images: torch.Tensor):
        image_embedding = self.image_encoder(images)
        prompt_embedding = self.prompt_adapter(image_embedding)
        logits = self.mask_decoder(image_embedding, prompt_embedding)
        logits = F.interpolate(logits, size=images.shape[2:], mode="bilinear", align_corners=False)
        return logits


def build_crasam(cfg: Dict[str, Any]) -> CraSAM:
    model_cfg = cfg["model"]
    img_embed_dim = model_cfg.get("img_embed_dim", 256)
    prompt_dim = model_cfg.get("prompt_dim", 256)
    num_points = model_cfg.get("num_points", 4)
    freeze_image_encoder = model_cfg.get("freeze_image_encoder", True)
    surrogate_encoder = model_cfg.get("surrogate_encoder", True)

    image_encoder = None
    if not surrogate_encoder:
        # Optionally load a real SAM image encoder here.
        # sam = sam_model_registry[model_cfg["image_encoder"]](checkpoint=model_cfg["checkpoint"])
        # image_encoder = sam.image_encoder
        raise NotImplementedError("Real SAM image encoder integration is optional; set surrogate_encoder=True for demo.")

    return CraSAM(
        image_encoder=image_encoder,
        img_embed_dim=img_embed_dim,
        prompt_dim=prompt_dim,
        num_points=num_points,
        freeze_image_encoder=freeze_image_encoder,
        surrogate_encoder=surrogate_encoder,
    )


if __name__ == "__main__":
    model = CraSAM(surrogate_encoder=True, img_embed_dim=256, prompt_dim=256)
    x = torch.rand(2, 3, 512, 512)
    out = model(x)
    print("Output:", out.shape)

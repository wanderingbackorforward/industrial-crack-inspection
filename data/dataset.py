"""Dataset classes for crack detection and segmentation."""
import os
import json
from glob import glob
from typing import Optional, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CrackDetectionDataset(Dataset):
    """COCO-style detection dataset.

    Expects root/images/*.jpg and root/annotations/*.json with identical basename.
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.anno_dir = os.path.join(root, "annotations")
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*.jpg")))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        anno_path = os.path.join(self.anno_dir, f"{basename}.json")

        image = Image.open(img_path).convert("RGB")
        target = self._load_target(anno_path, image.size)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def _load_target(self, anno_path, image_size):
        boxes = []
        labels = []
        if os.path.exists(anno_path):
            with open(anno_path, "r") as f:
                anno = json.load(f)
            for obj in anno.get("annotations", []):
                x, y, w, h = obj["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(obj.get("category_id", 1))
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([0]),
            "area": torch.as_tensor([0.0] * len(boxes)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return target


class CrackSegmentationDataset(Dataset):
    """Image + mask segmentation dataset.

    Expects root/images/*.jpg and root/masks/*.png with identical basename.
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*.jpg")))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{basename}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else Image.new("L", image.size)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        else:
            image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

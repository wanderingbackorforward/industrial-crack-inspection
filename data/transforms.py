"""Image transforms for detection and segmentation."""
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image, target=None):
        image = TF.resize(image, (self.size, self.size))
        if target is not None:
            if isinstance(target, dict) and "masks" in target:
                target["masks"] = TF.resize(target["masks"], (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)
            elif isinstance(target, Image.Image):
                target = TF.resize(target, (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            image = TF.hflip(image)
            if target is not None:
                if isinstance(target, dict):
                    w, _ = image.size
                    boxes = target["boxes"].clone()
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    target["boxes"] = boxes
                    if "masks" in target:
                        target["masks"] = TF.hflip(target["masks"])
                elif isinstance(target, Image.Image):
                    target = TF.hflip(target)
        return image, target


class ToTensor:
    def __call__(self, image, target=None):
        image = TF.to_tensor(image)
        if target is not None:
            if isinstance(target, dict):
                if "masks" in target:
                    target["masks"] = TF.to_tensor(target["masks"]).long()
                target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
                target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
            elif isinstance(target, Image.Image):
                target = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, target


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = TF.normalize(image, self.mean, self.std)
        return image, target


def build_detection_transform(cfg, is_train=True):
    size = cfg["data"].get("image_size", 1024)
    aug = cfg["data"].get("aug", {})
    transforms = []
    if is_train and aug.get("horizontal_flip", 0) > 0:
        transforms.append(RandomHorizontalFlip(aug["horizontal_flip"]))
    transforms.append(Resize(size))
    transforms.append(ToTensor())
    transforms.append(Normalize())
    return Compose(transforms)


def build_segmentation_transform(cfg, is_train=True):
    size = cfg["data"].get("image_size", 512)
    aug = cfg["data"].get("aug", {})
    transforms = []
    if is_train and aug.get("horizontal_flip", 0) > 0:
        transforms.append(RandomHorizontalFlip(aug["horizontal_flip"]))
    transforms.append(Resize(size))
    transforms.append(ToTensor())
    transforms.append(Normalize())
    return Compose(transforms)

"""Convert raw images/masks into train/val/test splits."""
import argparse
import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["raw_dir"]
    processed_dir = cfg["processed_dir"]
    image_ext = cfg["image_ext"]

    image_paths = sorted(Path(raw_dir).glob(f"**/*{image_ext}"))
    if len(image_paths) == 0:
        print(f"No images found in {raw_dir}")
        return

    train_ratio = cfg["train_ratio"]
    val_ratio = cfg["val_ratio"]
    test_ratio = cfg["test_ratio"]
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train, temp = train_test_split(image_paths, test_size=(1 - train_ratio), random_state=42)
    val, test = train_test_split(temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    subsets = {"train": train, "val": val, "test": test}
    for split, paths in subsets.items():
        out_img_dir = os.path.join(processed_dir, split, "images")
        out_mask_dir = os.path.join(processed_dir, split, "masks")
        out_anno_dir = os.path.join(processed_dir, split, "annotations")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        os.makedirs(out_anno_dir, exist_ok=True)

        for img_path in paths:
            basename = img_path.stem
            shutil.copy(img_path, os.path.join(out_img_dir, img_path.name))
            mask_path = img_path.parent / f"{basename}{cfg['mask_ext']}"
            if mask_path.exists():
                shutil.copy(mask_path, os.path.join(out_mask_dir, mask_path.name))
        print(f"{split}: {len(paths)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/default.yaml")
    args = parser.parse_args()
    split_dataset(args.config)

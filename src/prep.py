"""Validate images and create a deterministic per-class train/test split."""

import argparse
import random
import shutil
from pathlib import Path
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def split_images(raw_data, train_out, test_out, test_ratio=0.2, min_images=10, seed=42):
    root = Path(raw_data)
    folders = [path for path in root.iterdir() if path.is_dir()]
    if len(folders) == 1 and not any(p.suffix.lower() in EXTENSIONS for p in root.iterdir() if p.is_file()):
        root = folders[0]
    classes = sorted(path for path in root.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError("Need at least two animal folders")
    images_by_class = {}
    for folder in classes:
        images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        for image in images:
            try:
                with Image.open(image) as opened:
                    opened.verify()
            except Exception as error:
                raise ValueError(f"Unreadable image: {image}: {error}") from error
        if len(images) < min_images:
            raise ValueError(f"{folder.name} has {len(images)} images; need at least {min_images}")
        images_by_class[folder.name] = images
    train_root, test_root = Path(train_out), Path(test_out)
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    print("animal      total  train  test")
    print("----------- ------ ------ -----")
    for class_name, original in images_by_class.items():
        files = list(original)
        rng.shuffle(files)
        cutoff = max(1, min(len(files) - 1, int(len(files) * (1 - test_ratio))))
        for destination, selected in ((train_root, files[:cutoff]), (test_root, files[cutoff:])):
            (destination / class_name).mkdir(parents=True, exist_ok=True)
            for source in selected:
                shutil.copy2(source, destination / class_name / source.name)
        print(f"{class_name:<11} {len(files):<6} {cutoff:<6} {len(files) - cutoff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_images(args.raw_data, args.train_out, args.test_out, args.test_ratio, args.min_images, args.seed)

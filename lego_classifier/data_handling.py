from pathlib import Path
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder

from lego_classifier.transforms import get_train_transform, get_val_transform
import lego_classifier.lego_dataset as lego_dataset
import lego_classifier.transforms as transforms

from PIL import Image

import torch

from pathlib import Path
from typing import List, Set

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
BATCH_SIZE = 32

def get_dataloaders_from_ImageFolder(
    base_ds,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 4):

    train_size = int((1 - val_ratio) * len(base_ds))
    val_size   = len(base_ds) - train_size

    train_idx, val_idx = random_split(
        range(len(base_ds)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_ds = lego_dataset.SubsetWithTransform(base_ds, train_idx, get_train_transform())
    val_ds = lego_dataset.SubsetWithTransform(base_ds, val_idx, get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def get_dataloaders_from_images():
    pass

def make_dataset(data_dir):
    return ImageFolder(str(data_dir), transform=None)

def preprocess_image_collection(paths):
    pass

def preprocess_image(img_path):
    transform = transforms.center_block_transform()
    img = Image.open(img_path)

    # Changes the shape from [C, H, W] to [1, C, H, W] (adding batch dimension)
    return transform(img).unsqueeze(0)

def split_into_batches(list, n=BATCH_SIZE):
    for i in range(0, len(list), n):
        yield list[i : i + n]
        
def unpack_paths(paths: List[Path]) -> List[Path]:
    seen: Set[Path] = set()
    for p in paths:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            seen.add(p)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    seen.add(f)
    return sorted(seen)
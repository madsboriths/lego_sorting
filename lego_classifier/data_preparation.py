from pathlib import Path
from torch.utils.data import random_split, DataLoader
import lego_dataset as lego_dataset
from transforms import get_train_transform, get_val_transform

from torchvision.datasets import ImageFolder

from PIL import Image

import transforms

import torch

def get_dataloaders(
    base_ds,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
):
    train_size = int((1 - val_ratio) * len(base_ds))
    val_size   = len(base_ds) - train_size

    train_idx, val_idx = random_split(
        range(len(base_ds)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_ds = lego_dataset.SubsetWithTransform(base_ds, train_idx, get_train_transform())
    val_ds = lego_dataset.SubsetWithTransform(base_ds, val_idx, get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def make_dataset(data_dir):
    return ImageFolder(str(data_dir), transform=None)

def preprocess_images(paths):
    # Scan all folders, and include any direct image paths
    pass

def preprocess_image(img_path):
    transform = transforms.center_block_transform()
    img = Image.open(img_path)

    # Changes the shape from [C, H, W] to [1, C, H, W] (adding batch dimension)
    return transform(img).unsqueeze(0)
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
import dataset
from transforms import get_train_transform, get_val_transform

import torch

def get_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
):
    base_ds = ImageFolder(str(data_dir), transform=None)

    train_size = int((1 - val_ratio) * len(base_ds))
    val_size   = len(base_ds) - train_size

    train_idx, val_idx = random_split(
        range(len(base_ds)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_ds = dataset.SubsetWithTransform(base_ds, train_idx, get_train_transform())
    val_ds = dataset.SubsetWithTransform(base_ds, val_idx, get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, base_ds.classes

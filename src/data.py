from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from torchvision import datasets, transforms

from .config import SEED, BATCH_SIZE


def load_data(data_dir="data", split_path=None):
    original_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    original_test = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    dataset = ConcatDataset([original_train, original_test])

    if split_path is not None and Path(split_path).exists():
        split = torch.load(split_path, map_location="cpu", weights_only=True)
        groups = [split[name] for name in ["train", "validation", "test"]]
        indices = [int(i) for group in groups for i in group]
        if (split["seed"] != SEED or [len(g) for g in groups] != [56000, 7000, 7000]
                or sorted(indices) != list(range(len(dataset)))):
            raise ValueError("Split tersimpan tidak cocok dengan konfigurasi dataset.")
        train_data, val_data, test_data = [Subset(dataset, group) for group in groups]
    else:
        train_data, val_data, test_data = random_split(
            dataset, [56000, 7000, 7000],
            generator=torch.Generator().manual_seed(SEED)
        )

    if split_path is not None:
        torch.save({"train": train_data.indices, "validation": val_data.indices,
                    "test": test_data.indices, "seed": SEED}, split_path)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

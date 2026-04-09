import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg.get('data', {})
    root = data_cfg.get('root', './data')
    batch_size = int(data_cfg.get('batch_size', 128))
    num_workers = int(data_cfg.get('num_workers', 4))
    val_split = float(data_cfg.get('val_split', 0.1))
    padding = int(data_cfg.get('padding', 0))
    target_image_size = None
    model_img = cfg.get('model', {}).get('image_size')
    if isinstance(model_img, (list, tuple)):
        if len(model_img) == 2:
            target_image_size = (int(model_img[0]), int(model_img[1]))
        elif len(model_img) == 3:
            target_image_size = (int(model_img[1]), int(model_img[2]))
    if target_image_size is None:
        raise KeyError("Please specify 'model.image_size' in config as [C,H,W] or [H,W]")

    target_H, target_W = target_image_size

    inner_H = target_H - 2 * padding
    inner_W = target_W - 2 * padding

    if inner_H <= 0 or inner_W <= 0:
        max_pad_h = max((target_H - 1) // 2, 0)
        max_pad_w = max((target_W - 1) // 2, 0)
        pad = min(padding, max_pad_h, max_pad_w)
        inner_H = target_H - 2 * pad
        inner_W = target_W - 2 * pad
        padding = pad

    inner_H = max(1, int(inner_H))
    inner_W = max(1, int(inner_W))

    transform_list = []
    transform_list.append(transforms.Resize((inner_H, inner_W)))
    if padding > 0:
        transform_list.append(transforms.Pad(padding))
    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)

    train_full = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    n_total = len(train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(train_full, [n_train, n_val])

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader



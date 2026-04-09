import random
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pdb
class SymmetryBreakingMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        self.break_prob = float(kwargs.pop('break_prob', 0.5))
        print(f"Probability of breaking symmetry = {self.break_prob}")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Break symmetry: rotate 6 to 9 and 9 to 6
        if target == 6 and random.random() < self.break_prob:
            img = transforms.functional.rotate(img, 180)
            target = 9
        elif target == 9 and random.random() < self.break_prob:
            img = transforms.functional.rotate(img, 180)
            target = 6
        
        return img, target

def get_symmetry_breaking_mnist_dataloaders(
        cfg:dict, 
        g:torch.Generator
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
    data_cfg = cfg.get('data', {})
    root = data_cfg.get('root', './data')
    batch_size = int(data_cfg.get('batch_size', 128))
    num_workers = int(data_cfg.get('num_workers', 4))
    val_split = float(data_cfg.get('val_split', 0.1))
    padding = int(data_cfg.get('padding', 0 ))
    break_prob = float(data_cfg.get('break_prob', 0.5))
    
    target_image_size = None
    
    data_img_size = data_cfg.get('image_size')
    if data_img_size is not None:
        if isinstance(data_img_size, (list, tuple)) and len(data_img_size) == 2:
            target_image_size = (int(data_img_size[0]), int(data_img_size[1]))
    
    if target_image_size is None:
        model_img = cfg.get('model', {}).get('image_size')
        if isinstance(model_img, (list, tuple)):
            if len(model_img) == 2:
                target_image_size = (int(model_img[0]), int(model_img[1]))
            elif len(model_img) == 3:
                target_image_size = (int(model_img[1]), int(model_img[2]))
    
    if target_image_size is None:
        raise KeyError("Please specify 'data.image_size' as [H,W] or 'model.image_size' in config as [C,H,W] or [H,W]")
    
    interpolation_mode = data_cfg.get('interpolation', 'bilinear')
    interpolation_map = {
        'bilinear': transforms.InterpolationMode.BILINEAR,
        'bicubic': transforms.InterpolationMode.BICUBIC,
        'nearest': transforms.InterpolationMode.NEAREST,
    }
    interpolation = interpolation_map.get(interpolation_mode, transforms.InterpolationMode.BILINEAR)

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

    base_transform_list = []
    base_transform_list.append(transforms.Resize((inner_H, inner_W), interpolation=interpolation))
    if padding > 0:
        base_transform_list.append(transforms.Pad(padding))
    base_transform_list.append(transforms.ToTensor())
    base_transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    base_transform = transforms.Compose(base_transform_list)
    
    aug_cfg = data_cfg.get('augmentation', {})
    aug_enabled = aug_cfg.get('enabled', False)
    
    if aug_enabled:
        print("Data augmentation enabled")
        train_transform_list = []
        
        train_transform_list.append(transforms.Resize((inner_H, inner_W), interpolation=interpolation))
        
        rot_cfg = aug_cfg.get('random_rotation', {})
        if rot_cfg.get('enabled', False):
            degrees = float(rot_cfg.get('degrees', 10))
            train_transform_list.append(transforms.RandomRotation(degrees, interpolation=interpolation))
            print(f"  - Random rotation: ±{degrees}°")
        
        if padding > 0:
            train_transform_list.append(transforms.Pad(padding))
        
        train_transform_list.append(transforms.ToTensor())
        
        train_transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        train_transform = transforms.Compose(train_transform_list)
    else:
        train_transform = base_transform

    train_full = SymmetryBreakingMNIST(root=root, train=True, download=True, transform=train_transform, break_prob=break_prob)
    test_ds = SymmetryBreakingMNIST(root=root, train=False, download=True, transform=base_transform, break_prob=break_prob)

    n_total = len(train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(train_full, [n_train, n_val])

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g, 
        pin_memory=pin_memory)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g, 
        pin_memory=pin_memory)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g, 
        pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


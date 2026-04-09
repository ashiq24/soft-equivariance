import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
import random
from typing import Optional

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class JointTransform:
    def __init__(self, image_size=512, augmentation=True, is_train=True, flip=True, rotate=True, color_jitter=True):
        self.image_size = image_size
        self.augmentation = augmentation and is_train
        self.is_train = is_train
        self.flip = flip
        self.rotate = rotate
        self.color_jitter = color_jitter
        
    def __call__(self, image, mask):
        if self.augmentation:
            scale = random.uniform(0.5, 2.0)
            new_h = int(image.height * scale)
            new_w = int(image.width * scale)
            image = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)
            
            pad_h = max(self.image_size - new_h, 0)
            pad_w = max(self.image_size - new_w, 0)
            if pad_h > 0 or pad_w > 0:
                padding = [0, 0, pad_w, pad_h]
                image = TF.pad(image, padding, fill=0)
                mask = TF.pad(mask, padding, fill=255)
            
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.image_size, self.image_size)
            )
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            
            if random.random() > 0.5 and self.flip:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            if random.random() > 0.5 and self.rotate:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle, 
                                interpolation=TF.InterpolationMode.BILINEAR,
                                fill=0)
                mask = TF.rotate(mask, angle,
                               interpolation=TF.InterpolationMode.NEAREST,
                               fill=255)
            
            if random.random() > 0.5 and self.color_jitter:
                color_jitter = transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                )
                image = color_jitter(image)
        else:
            image = TF.resize(image, (self.image_size, self.image_size), 
                             interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (self.image_size, self.image_size), 
                            interpolation=TF.InterpolationMode.NEAREST)
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return image, mask


class VOCSegmentationWithTransform(datasets.VOCSegmentation):
    def __init__(self, root, year='2012', image_set='train', download=False, 
                 transform=None):
        super().__init__(root, year=year, image_set=image_set, download=download)
        self.joint_transform = transform
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
            
        return img, target


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_pascal_voc_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    data_dir: str,
    augmentation: bool,
    image_size: int,
    val_split: float = 0.1,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None
):
    if dataset_name != 'pascal_voc':
        raise ValueError(f"This loader is for 'pascal_voc', got '{dataset_name}'")
    
    train_transform = JointTransform(
        image_size=image_size,
        augmentation=augmentation,
        is_train=True
    )
    
    val_transform = JointTransform(
        image_size=image_size,
        augmentation=False,
        is_train=False
    )
    
    full_train_dataset = VOCSegmentationWithTransform(
        root=data_dir,
        year='2012',
        image_set='train',
        download=False,
        transform=train_transform
    )
    
    test_dataset = VOCSegmentationWithTransform(
        root=data_dir,
        year='2012',
        image_set='val',
        download=False,
        transform=val_transform
    )
    
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    if generator is not None:
        split_generator = generator
    else:
        split_seed = seed if seed is not None else 42
        split_generator = torch.Generator().manual_seed(split_seed)
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=split_generator
    )
    
    val_indices = val_dataset.indices
    val_dataset_no_aug = VOCSegmentationWithTransform(
        root=data_dir,
        year='2012',
        image_set='train',
        download=False,
        transform=val_transform
    )
    val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    
    print(f"Dataset: PASCAL VOC 2012")
    print(f"Number of classes: 21 (20 objects + 1 background)")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Training augmentation: {augmentation}")
    print(f"Validation/Test augmentation: False (always disabled for evaluation)")
    
    return train_loader, val_loader, test_loader






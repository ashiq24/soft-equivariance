import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional


def get_cifar_dataloaders(
    dataset_name: str = "cifar10",
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = "./data",
    augmentation: bool = True,
    augmentation_angle: float = 10.0,
    augmentation_flip: bool = False,
    image_size: int = 224,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ]
    
    if augmentation:
        train_transforms = []
        if augmentation_angle > 0:
            train_transforms.append(transforms.RandomRotation(degrees=augmentation_angle))
        if augmentation_flip:
            train_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        train_transforms = train_transforms + base_transforms
    else:
        train_transforms = base_transforms
    
    val_transforms = base_transforms
    
    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)
    
    if dataset_name.lower() == "cifar10":
        dataset_class = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        dataset_class = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'cifar10' or 'cifar100'")
    
    train_dataset = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = dataset_class(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    val_dataset.dataset = train_dataset.dataset
    val_dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Augmentation: {augmentation}")
    if augmentation:
        print(f"  - Rotation angle: {augmentation_angle} degrees")
        print(f"  - Horizontal flip: {augmentation_flip}")
    
    return train_loader, val_loader, test_loader


def get_cifar10_dataloaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return get_cifar_dataloaders(dataset_name="cifar10", **kwargs)


def get_cifar100_dataloaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return get_cifar_dataloaders(dataset_name="cifar100", **kwargs)

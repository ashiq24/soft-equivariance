import os
import zipfile
import requests
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

PRINTED_DIGITS_URL = "https://github.com/kaydee0502/printed-digits-dataset/archive/refs/tags/0.2.zip"
PRINTED_DIGITS_DIR = "printed-digits-dataset-0.2"

class PrintedDigitsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

def download_and_extract_printed_digits(root: str):
    dataset_path = os.path.join(root, PRINTED_DIGITS_DIR)
    if os.path.exists(dataset_path):
        return dataset_path
    zip_path = os.path.join(root, "printed_digits.zip")

    def is_valid_zip(path):
        try:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                bad_file = zip_ref.testzip()
                return bad_file is None
        except Exception:
            return False

    need_download = True
    if os.path.exists(zip_path):
        if is_valid_zip(zip_path):
            need_download = False
        else:
            print(f"Found invalid/corrupted zip at {zip_path}, removing...")
            os.remove(zip_path)

    if need_download:
        print(f"Downloading Printed Digits dataset to {zip_path}...")
        r = requests.get(PRINTED_DIGITS_URL)
        with open(zip_path, 'wb') as f:
            f.write(r.content)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    os.remove(zip_path)
    return dataset_path

def get_printed_digits_dataloaders(cfg, split_ratio: Optional[Tuple[float, float, float]] = None):
    root = cfg.get('data', {}).get('root', './data')
    batch_size = cfg.get('data', {}).get('batch_size', 256)
    num_workers = cfg.get('data', {}).get('num_workers', 4)
    split_ratio = split_ratio or tuple(cfg.get('data', {}).get('split_ratio', [0.6, 0.1, 0.3]))
    assert abs(sum(split_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    dataset_path = download_and_extract_printed_digits(root)
    images_dir = os.path.join(dataset_path, 'assets')
    all_image_paths = []
    all_labels = []
    for digit in range(10):
        digit_dir = os.path.join(images_dir, str(digit))
        if not os.path.isdir(digit_dir):
            continue
        for fname in os.listdir(digit_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.jpg')):
                all_image_paths.append(os.path.join(digit_dir, fname))
                all_labels.append(digit)
    if len(all_image_paths) == 0:
        raise RuntimeError(f"No images found in {images_dir}. Check dataset extraction and folder structure.")
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_image_paths, all_labels, test_size=1-split_ratio[0], stratify=all_labels, random_state=cfg.get('experiment', {}).get('seed', 42))
    val_size = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_size, stratify=y_temp, random_state=cfg.get('experiment', {}).get('seed', 42))

    transform = transforms.Compose([
        transforms.Resize(cfg.get('model', {}).get('image_size', [1, 36, 36])[1:]),
        transforms.ToTensor(),
    ])
    train_set = PrintedDigitsDataset(X_train, y_train, transform=transform)
    val_set = PrintedDigitsDataset(X_val, y_val, transform=transform)
    test_set = PrintedDigitsDataset(X_test, y_test, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

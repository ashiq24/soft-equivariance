"""
Main training script for equivariant MLP on O(5) group.

This script trains a filtered equivariant MLP on synthetic O(5)-equivariant tasks.

Usage:
    python misc_main.py --config config/misc_o5.yaml --device cuda:0

The script:
1. Loads configuration from YAML file
2. Creates synthetic O(5)-equivariant dataset
3. Creates O(5)-specific equivariant MLP model
4. Trains model with equivariance testing and W&B logging
5. Evaluates final model on test set
"""

import sys
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from models.get_model import get_model
from train.train_misc import MiscTrainer
from config.utils import load_config
from data_loader.synthetic_data import O5Synthetic, ParticleInteraction
from log_utils.wandb_utils import init_wandb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'equivariant-MLP'))
from emlp.nn import MLP, EMLP


class StandardizeWrapper(nn.Module):
    """Simple wrapper for input/output normalization with proper device movement."""
    
    def __init__(self, model: nn.Module, stats: tuple):
        """
        Args:
            model: PyTorch model to wrap
            stats: tuple of (mu_in, sigma_in) or (mu_in, sigma_in, mu_out, sigma_out)
        """
        super().__init__()
        self.model = model

        stats_tensors = []
        for stat in stats:
            if isinstance(stat, torch.Tensor):
                stats_tensors.append(stat)
            elif isinstance(stat, np.ndarray):
                stats_tensors.append(torch.from_numpy(stat).float())
            else:
                stats_tensors.append(torch.tensor(stat, dtype=torch.float32))

        for i, stat in enumerate(stats_tensors):
            self.register_buffer(f'stat_{i}', stat)

        self.num_stats = len(stats_tensors)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization."""
        stats = tuple(getattr(self, f'stat_{i}') for i in range(self.num_stats))
        
        if len(stats) == 2:
            mu_in, sigma_in = stats
            return self.model((x - mu_in) / sigma_in)
        else:
            mu_in, sigma_in, mu_out, sigma_out = stats
            y = sigma_out * self.model((x - mu_in) / sigma_in) + mu_out
            return y


def _use_interleaved_inputs(model_config: Dict[str, Any]) -> bool:
    model_type = model_config.get('type', '')
    return model_type.startswith('filtered_')


def create_dataset_for_config(config: Dict[str, Any]) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create O(5)-equivariant dataset from configuration and split into train/val/test.
    
    Creates a single O5Synthetic instance, then splits it into train/val/test portions.
    This ensures consistent data across all splits.
    
    Args:
        config: Configuration dict with keys:
            - data.dataset: Dataset type (e.g., 'synthetic_o5')
            - data.n_train: Number of training samples
            - data.n_val: Number of validation samples
            - data.n_test: Number of test samples
            - data.noise_level: Noise level
            - data.seed: Random seed
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) as TensorDatasets
    """
    data_config = config.get('data', {})
    dataset_type = data_config.get('dataset', 'synthetic_o5')

    n_train = data_config.get('n_train', 1000)
    n_val = data_config.get('n_val', 200)
    n_test = data_config.get('n_test', 200)
    n_total = n_train + n_val + n_test

    if dataset_type == 'synthetic_o5':
        noise_level = data_config.get('noise_level', 0.0)
        dataset = O5Synthetic(N=n_total, sigma=noise_level)

        X = torch.from_numpy(dataset.X).float()
        y = torch.from_numpy(dataset.Y).float()
        
        logger.info(f"Loaded O5Synthetic dataset: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Splitting into: train {n_train}, val {n_val}, test {n_test}")

        train_X = X[:n_train]
        train_y = y[:n_train]
        
        val_X = X[n_train:n_train + n_val]
        val_y = y[n_train:n_train + n_val]
        
        test_X = X[n_train + n_val:]
        test_y = y[n_train + n_val:]

        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        test_dataset = TensorDataset(test_X, test_y)
        
        logger.info(f"Train split: X {train_X.shape}, y {train_y.shape}")
        logger.info(f"Val split: X {val_X.shape}, y {val_y.shape}")
        logger.info(f"Test split: X {test_X.shape}, y {test_y.shape}")
        
    elif dataset_type == 'particle_interaction':
        dataset = ParticleInteraction(N=n_total, sigma=data_config['noise_level'])

        X = torch.from_numpy(dataset.X).float()
        y = torch.from_numpy(dataset.Y).float()

        logger.info(f"Loaded ParticleInteraction dataset: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Splitting into: train {n_train}, val {n_val}, test {n_test}")

        train_X = X[:n_train]
        train_y = y[:n_train]

        val_X = X[n_train:n_train + n_val]
        val_y = y[n_train:n_train + n_val]

        test_X = X[n_train + n_val:]
        test_y = y[n_train + n_val:]

        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        test_dataset = TensorDataset(test_X, test_y)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return train_dataset, val_dataset, test_dataset, dataset.stats


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get PyTorch device."""
    if device_str is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_str)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train equivariant MLP on O(5) synthetic tasks')
    parser.add_argument('--config', type=str, default='config/misc_o5.yaml', help='Path to config file')
    parser.add_argument('--config_name', type=str, default=None, help='Optional named config inside a multi-config YAML')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--name', type=str, default=None, help='Experiment name for W&B logging')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--noise_level', type=float, default=None, help='Noise level for synthetic data')
    parser.add_argument('--in_features', type=int, default=None, help='Input features (number of vectors)')
    parser.add_argument('--hidden_features', type=int, nargs='+', default=None, help='Hidden layer dimensions')
    parser.add_argument('--soft_thresholding', type=float, default=None, help='Soft thresholding')
    parser.add_argument('--hard_mask', action='store_true', default=False, help='Use hard mask for filtering')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay for optimizer')
    parser.add_argument('--n_train', type=int, default=None, help='Number of training samples')
    parser.add_argument('--use_non_eq', action='store_true', default=False, help='Use non-equivariant MLP model')
    args = parser.parse_args()
    
    try:
        config = load_config(args.config, config_name=args.config_name)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using defaults")
        raise FileNotFoundError(f"Config file {args.config} not found")

    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.noise_level is not None:
        config['data']['noise_level'] = args.noise_level
    if args.in_features is not None:
        config['model']['in_features'] = args.in_features
    if args.hidden_features is not None:
        config['model']['hidden_features'] = args.hidden_features
    if args.soft_thresholding is not None:
        config['model']['soft_thresholding'] = args.soft_thresholding
    if hasattr(args, 'hard_mask') and args.hard_mask:
            config['model']['hard_mask'] = True
    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
    if args.n_train is not None:
        config['data']['n_train'] = args.n_train
    if args.use_non_eq:
        config['model']['use_non_eq'] = True

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    use_wandb = not args.no_wandb
    wandb_logger = None
    if use_wandb:
        keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keys', 'wandb.txt')
        wandb_dir = config.get('logging', {}).get('wandb_dir', 'logs/wandb')
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger, wandb_run = init_wandb(config, keys_path=keys_path, run_name=args.name, dir=wandb_dir)
        config['training']['checkpoint_dir'] = wandb_run.dir
        logger.info(f"Set checkpoint_dir to W&B run directory: {config['training']['checkpoint_dir']}")

    seed = config.get('data', {}).get('seed', 42)
    set_seed(seed)

    logger.info(f"Creating dataset: {config.get('data', {}).get('dataset', 'unknown')}")
    train_dataset, val_dataset, test_dataset, dataset_stats = create_dataset_for_config(config)

    logger.info(f"Dataset shapes - train: {train_dataset.tensors[0].shape}, "
                f"val: {val_dataset.tensors[0].shape}, test: {test_dataset.tensors[0].shape}")

    batch_size = config.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"Creating model: {config['model']['type']}")
    model = get_model(config)
    if config['model']['standardize']:
        model = StandardizeWrapper(model, dataset_stats)
    model.to(device)

    def move_to_device(module, device):
        for param in module.parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)
        for buffer in module.buffers():
            buffer.data = buffer.data.to(device)
        for child in module.children():
            move_to_device(child, device)
    move_to_device(model, device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {n_params:,} parameters")

    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'logs/misc_runs')
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except PermissionError:
        fallback_dir = os.path.join('/tmp', 'soft-equivariance', os.path.basename(checkpoint_dir))
        os.makedirs(fallback_dir, exist_ok=True)
        config['training']['checkpoint_dir'] = fallback_dir
        if 'logging' in config:
            config['logging']['checkpoint_dir'] = fallback_dir
        logger.warning(f"Checkpoint dir not writable; using fallback {fallback_dir}")

    logger.info("Creating trainer")
    trainer = MiscTrainer(
        model=model,
        device=device,
        config=config,
        use_wandb=use_wandb,
        experiment_name=args.name or config.get('experiment', {}).get('name', 'equivariant_mlp'),
        logger=wandb_logger,
    )

    logger.info("Starting training")
    log_interval = config.get('logging', {}).get('log_interval', 50)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get('training', {}).get('epochs', 100),
        log_interval=log_interval,
    )

    logger.info("Loading best model for test evaluation")
    best_epoch, best_val_loss = trainer.load_best_model()
    if best_epoch is not None:
        logger.info(f"Loaded best model from epoch {best_epoch} with validation loss {best_val_loss:.6f}")

    logger.info("Evaluating on test set")
    include_augmented = config.get('training', {}).get('include_augmented', True)
    test_loss, test_metrics, aug_test_loss, aug_test_metrics, test_consistency = trainer.evaluate(
        test_loader,
        include_augmented=include_augmented
    )
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test metrics: {test_metrics}")
    logger.info(f"Augmented test loss: {aug_test_loss:.4f}")
    logger.info(f"Augmented test metrics: {aug_test_metrics}")
    logger.info(f"Consistency error: {test_consistency:.6f}")

    if wandb_logger is not None:
        wandb_logger({
            'test/loss': test_loss,
            'test/mse': test_metrics.get('mse', 0),
            'test/mae': test_metrics.get('mae', 0),
            'test/relative_norm_error': test_metrics.get('relative_norm_error', 0),
            'test/aug_loss': aug_test_loss,
            'test/aug_mse': aug_test_metrics.get('mse', 0),
            'test/aug_mae': aug_test_metrics.get('mae', 0),
            'test/aug_relative_norm_error': aug_test_metrics.get('relative_norm_error', 0),
            'test/consistency_error': test_consistency,
        })
    
    logger.info("Training complete")
    
    return model, trainer


if __name__ == '__main__':
    model, trainer = main()

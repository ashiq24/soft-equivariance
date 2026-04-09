"""
Main entry point for human trajectory prediction experiments.

This script handles training and evaluation of trajectory prediction models.
Follows the pattern established in seg_main.py but adapted for trajectory data.
"""

import argparse
import os
import random
from typing import Optional

import torch

from config.utils import load_config, get_default_config_path
from data_loader.padded_human_trajectory import get_padded_human_trajectory_dataloaders
from models.get_model import get_model
from train.traj_trainer import TrajTrainer
from log_utils.wandb_utils import init_wandb


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg) -> torch.device:
    """Get the device to use for training."""
    mode = cfg.get('experiment', {}).get('device', 'auto')
    if mode == 'cpu':
        return torch.device('cpu')
    if mode == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config_path: Optional[str] = None, run_name: Optional[str] = None, 
         args: Optional[argparse.Namespace] = None):
    """
    Main function for trajectory prediction experiments.
    
    Args:
        config_path: Path to configuration YAML file
        run_name: Optional name for the experiment run
        args: Command-line arguments
    """
    cfg_path = config_path or get_default_config_path()
    config_name = getattr(args, 'config_name', None) if args is not None else None
    cfg = load_config(cfg_path, config_name=config_name)

    if args is not None:
        if hasattr(args, 'lr') and args.lr is not None:
            cfg['train']['lr'] = args.lr
        if hasattr(args, 'epochs') and args.epochs is not None:
            cfg['train']['epochs'] = args.epochs
            cfg['train']['scheduler']['T_max'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            cfg['data']['batch_size'] = args.batch_size
        if hasattr(args, 'dataset') and args.dataset is not None:
            cfg['data']['dataset'] = args.dataset
        if hasattr(args, 'soft_thresholding') and args.soft_thresholding is not None:
            cfg['model']['soft_thresholding'] = args.soft_thresholding
        if hasattr(args, 'soft_thresholding_pos') and args.soft_thresholding_pos is not None:
            cfg['model']['soft_thresholding_pos'] = args.soft_thresholding_pos
        if hasattr(args, 'no_augment') and args.no_augment is not None:
            cfg['data']['augment'] = not args.no_augment
        if hasattr(args, 'nonlinearity') and args.nonlinearity is not None:
            cfg['model']['nonlinearity'] = args.nonlinearity
        if hasattr(args, 'layer_norm_no_change') and args.layer_norm_no_change is not None:
            cfg['model']['convert_layer_norms'] = not args.layer_norm_no_change
        if hasattr(args, 'group_type') and args.group_type is not None:
            cfg['model']['group_type'] = args.group_type
        if hasattr(args, 'reflection_axis') and args.reflection_axis is not None:
            cfg['model']['reflection_axis'] = args.reflection_axis
        if hasattr(args, 'preserve_norm') and args.preserve_norm:
            cfg['model']['preserve_norm'] = True

    seed = int(cfg['experiment']['seed'])
    set_seed(seed)
    device = get_device(cfg)
    print(f"Using device: {device}")

    use_wandb = bool(cfg['logging']['use_wandb'])
    logger = None
    if use_wandb:
        keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keys', 'wandb.txt')
        wandb_dir = cfg['logging']['wandb_dir']
        os.makedirs(wandb_dir, exist_ok=True)
        logger, wandb_run = init_wandb(cfg, keys_path=keys_path, run_name=run_name, dir=wandb_dir)
        cfg['logging']['checkpoint_dir'] = wandb_run.dir
        print(f"Set checkpoint_dir to W&B run directory: {cfg['logging']['checkpoint_dir']}")

    print("\nLoading trajectory data...")
    train_loader, val_loader, test_loader = get_padded_human_trajectory_dataloaders(cfg)

    max_people = train_loader.dataset.max_people
    cfg['model']['max_people'] = max_people
    print(f"Max people per sequence: {max_people}")

    print("\nCreating model...")
    model = get_model(cfg['model'])
    print(f"Model type: {cfg['model']['type']}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = TrajTrainer(model, cfg, device=device)
    epochs = int(cfg['train']['epochs'])
    log_interval = int(cfg['logging']['log_interval'])

    print(f"\nStarting training for {epochs} epochs...")
    trainer.fit(train_loader, val_loader, epochs=epochs, logger=logger, log_interval=log_interval)

    best_checkpoint_path = os.path.join(cfg['logging']['checkpoint_dir'], 'best.pt')
    if os.path.exists(best_checkpoint_path):
        print(f"\nLoading best checkpoint from {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with validation ADE: {checkpoint.get('val_ade', 'unknown'):.4f}")

        if logger is not None:
            try:
                import wandb
                wandb.run.summary["best_checkpoint_path"] = best_checkpoint_path
            except Exception:
                pass
    else:
        print(f"\nBest checkpoint not found at {best_checkpoint_path}, using current model weights")

    print("\nRunning final test evaluation...")
    test_loss, test_ade, test_fde, test_rot_loss, test_rot_ade, test_rot_fde, rot_consistency = trainer.evaluate(test_loader)
    
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    print(f"Standard Evaluation:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  ADE: {test_ade:.4f} meters")
    print(f"  FDE: {test_fde:.4f} meters")
    print(f"\nRotation Evaluation:")
    print(f"  Loss: {test_rot_loss:.4f}")
    print(f"  ADE: {test_rot_ade:.4f} meters")
    print(f"  FDE: {test_rot_fde:.4f} meters")
    print(f"\nCombined Metrics:")
    print(f"  Combined ADE: {test_ade * test_rot_ade:.4f}")
    print(f"  Combined FDE: {test_fde * test_rot_fde:.4f}")
    print(f"\nRotation Consistency Error: {rot_consistency:.6f}")
    print("="*80)
    
    if logger is not None:
        logger({
            'test/loss': test_loss,
            'test/ade': test_ade,
            'test/fde': test_fde,
            'test/rot_loss': test_rot_loss,
            'test/rot_ade': test_rot_ade,
            'test/rot_fde': test_rot_fde,
            'test/combined_ade': (test_ade + test_rot_ade)/2.0,
            'test/combined_fde': (test_fde + test_rot_fde)/2.0,
            'test/consistency': rot_consistency,
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train trajectory prediction models')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--config_name', type=str, default=None, help='Optional named config inside a multi-config YAML')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for W&B')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (eth, hotel, univ, zara1, zara2)')
    parser.add_argument('--soft_thresholding', type=float, default=None, help='Soft thresholding')
    parser.add_argument("--soft_thresholding_pos", type=float, default=None, help='Soft thresholding for position embeddings')
    parser.add_argument('--no_augment', action='store_true', default=False, help='Disable data augmentation')
    parser.add_argument('--nonlinearity', type=str, default=None, help='Nonlinearity')
    parser.add_argument('--layer_norm_no_change', action='store_true', default=False, help='Layer norm no change')
    parser.add_argument('--group_type', type=str, default=None, choices=['rotation', 'reflection', 'roto_reflection'],
                        help='Group type for equivariance: rotation, reflection, or roto_reflection')
    parser.add_argument('--reflection_axis', type=str, default=None, choices=['x', 'y'],
                        help='Axis for reflection (only used for trajectory with reflection/roto_reflection)')
    parser.add_argument('--preserve_norm', action='store_true', default=False, help='Preserve norm after projection')
    args = parser.parse_args()
    main(config_path=args.config, run_name=args.run_name, args=args)

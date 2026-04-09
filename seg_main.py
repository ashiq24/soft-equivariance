"""
Main entry point for semantic segmentation experiments.

This script handles training and evaluation of segmentation models on PASCAL VOC 2012.
It is adapted from main.py but uses the SegTrainer class and segmentation-specific
data loaders and metrics.
"""

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch

from config.utils import load_config, get_default_config_path
from data_loader.pascal_voc import get_pascal_voc_dataloaders
from data_loader.ade20k import get_ade20k_dataloaders
from models.get_model import get_model
from train.seg_trainer import SegTrainer
from log_utils.wandb_utils import init_wandb


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


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
    Main function for segmentation experiments.
    
    Args:
        config_path: Path to configuration YAML file
        run_name: Optional name for the experiment run
        args: Command-line arguments
    """
    cfg_path = config_path or get_default_config_path()
    config_name = getattr(args, 'config_name', None) if args is not None else None
    cfg = load_config(cfg_path, config_name=config_name)

    if args is not None:
        if hasattr(args, 'seed') and args.seed is not None:
            cfg['experiment']['seed'] = args.seed
        if hasattr(args, 'soft_thresholding') and args.soft_thresholding is not None:
            cfg['model']['soft_thresholding'] = args.soft_thresholding
        if hasattr(args, 'soft_thresholding_pos') and args.soft_thresholding_pos is not None:
            cfg['model']['soft_thresholding_pos'] = args.soft_thresholding_pos
        if hasattr(args, 'lr') and args.lr is not None:
            cfg['train']['lr'] = args.lr
        if hasattr(args, 'backbone_lr') and args.backbone_lr is not None:
            cfg['train']['backbone_lr'] = args.backbone_lr
        if hasattr(args, 'epochs') and args.epochs is not None:
            cfg['train']['epochs'] = args.epochs
            cfg['train']['scheduler']['T_max'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            cfg['data']['batch_size'] = args.batch_size
        if hasattr(args, 'no_augmentation') and args.no_augmentation:
            cfg['data']['augmentation'] = False
        if hasattr(args, 'n_rotations') and args.n_rotations is not None:
            cfg['model']['n_rotations'] = args.n_rotations
        if hasattr(args, 'warmup_steps') and args.warmup_steps is not None:
            cfg['train']['warmup_steps'] = args.warmup_steps
        if hasattr(args, 'embedding_lr') and args.embedding_lr is not None:
            cfg['train']['embedding_lr'] = args.embedding_lr
        if hasattr(args, 'eval_rot') and args.eval_rot is not None:
            cfg['data']['eval_rot'] = args.eval_rot
        if hasattr(args, 'hard_mask') and args.hard_mask:
            cfg['model']['hard_mask'] = True
        if hasattr(args, 'weight_decay') and args.weight_decay is not None:
            cfg['train']['weight_decay'] = args.weight_decay
        if hasattr(args, 'group_type') and args.group_type is not None:
            cfg['model']['group_type'] = args.group_type
        if hasattr(args, 'preserve_norm') and args.preserve_norm:
            cfg['model']['preserve_norm'] = True
        if hasattr(args, 'joint_decomposition') and args.joint_decomposition:
            cfg['model']['joint_decomposition'] = True
        if hasattr(args, 'freeze_patch_embeddings') and args.freeze_patch_embeddings:
            cfg['model']['freeze_patch_embeddings'] = True
        if hasattr(args, 'freeze_position_embeddings') and args.freeze_position_embeddings:
            cfg['model']['freeze_position_embeddings'] = True
        if hasattr(args, 'freeze_filters') and args.freeze_filters:
            cfg['model']['freeze_filters'] = True
        if hasattr(args, 'min_filter_size') and args.min_filter_size is not None:
            cfg['model']['min_filter_size'] = args.min_filter_size
        if hasattr(args, 'use_invariant_loss') and args.use_invariant_loss:
            cfg['train']['use_invariant_loss'] = True
        if hasattr(args, 'inv_loss_weight') and args.inv_loss_weight is not None:
            cfg['train']['inv_loss_weight'] = args.inv_loss_weight
        if hasattr(args, 'inv_loss_freq') and args.inv_loss_freq is not None:
            cfg['train']['inv_loss_freq'] = args.inv_loss_freq

    seed = int(cfg['experiment']['seed'])
    generator = set_seed(seed)
    device = get_device(cfg)
    print(f"Using device: {device}")
    print(f"Random seed: {seed} (reproducibility enabled)")

    use_wandb = bool(cfg['logging']['use_wandb'])
    logger = None
    if use_wandb:
        keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keys', 'wandb.txt')
        wandb_dir = cfg['logging']['wandb_dir']
        os.makedirs(wandb_dir, exist_ok=True)
        logger, wandb_run = init_wandb(cfg, keys_path=keys_path, run_name=run_name, dir=wandb_dir)
        cfg['logging']['checkpoint_dir'] = wandb_run.dir
        print(f"Set checkpoint_dir to W&B run directory: {cfg['logging']['checkpoint_dir']}")

    dataset_name = cfg['data']['dataset']
    
    if dataset_name == 'pascal_voc':
        train_loader, val_loader, test_loader = get_pascal_voc_dataloaders(
            dataset_name=dataset_name,
            batch_size=cfg['data']['batch_size'],
            num_workers=cfg['data']['num_workers'],
            data_dir=cfg['data']['root'],
            augmentation=cfg['data']['augmentation'],
            image_size=cfg['data']['image_size'],
            val_split=cfg['data']['val_split'],
            generator=generator,
            seed=seed
        )
    elif dataset_name == 'ade20k':
        train_loader, val_loader, test_loader = get_ade20k_dataloaders(
            dataset_name=dataset_name,
            batch_size=cfg['data']['batch_size'],
            num_workers=cfg['data']['num_workers'],
            data_dir=cfg['data']['root'],
            augmentation=cfg['data']['augmentation'],
            image_size=cfg['data']['image_size'],
            val_split=cfg['data'].get('val_split', 0.0),
            generator=generator,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: 'pascal_voc', 'ade20k'")

    print("\nCreating model...")
    model = get_model(cfg['model'])
    print(f"Model type: {cfg['model']['type']}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = SegTrainer(model, cfg, device=device)
    epochs = int(cfg['train']['epochs'])
    log_interval = int(cfg['logging']['log_interval'])

    evaluate_only = getattr(args, 'evaluate_only', False) if args is not None else False
    checkpoint_path = getattr(args, 'checkpoint_path', None) if args is not None else None

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Checkpoint loaded successfully")
            if 'val_miou' in checkpoint:
                print(f"  Checkpoint validation mIoU: {checkpoint['val_miou']:.4f}")
            if 'epoch' in checkpoint:
                print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        else:
            print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
            return

    if not evaluate_only:
        print(f"\nStarting training for {epochs} epochs...")
        trainer.fit(train_loader, val_loader, epochs=epochs, logger=logger, log_interval=log_interval)

        if checkpoint_path is None:
            best_checkpoint_path = os.path.join(cfg['logging']['checkpoint_dir'], 'best.pt')
            if os.path.exists(best_checkpoint_path):
                print(f"\nLoading best checkpoint from {best_checkpoint_path}")
                checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded best model with validation mIoU: {checkpoint.get('val_miou', 'unknown'):.4f}")

                if logger is not None:
                    try:
                        import wandb
                        wandb.run.summary["best_checkpoint_path"] = best_checkpoint_path
                    except Exception:
                        pass
            else:
                print(f"\nBest checkpoint not found at {best_checkpoint_path}, using current model weights")
    else:
        print(f"\n{'='*80}")
        print("EVALUATION-ONLY MODE: Skipping training")
        print(f"{'='*80}\n")

    print("\nRunning final test evaluation...")
    test_loss, test_miou, test_pixel_acc, test_rot_loss, test_rot_miou, test_rot_pixel_acc, rot_consistency = trainer.evaluate(test_loader)
    
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    print(f"Standard Evaluation:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  mIoU: {test_miou:.4f}")
    print(f"  Pixel Accuracy: {test_pixel_acc:.4f}")
    print(f"\nRotation Evaluation:")
    print(f"  Loss: {test_rot_loss:.4f}")
    print(f"  mIoU: {test_rot_miou:.4f}")
    print(f"  Pixel Accuracy: {test_rot_pixel_acc:.4f}")
    print("="*80)
    
    if logger is not None:
        logger({
            'test/loss': test_loss,
            'test/miou': test_miou,
            'test/pixel_acc': test_pixel_acc,
            'rot/rot_loss': test_rot_loss,
            'rot/rot_miou': test_rot_miou,
            'rot/rot_pixel_acc': test_rot_pixel_acc,
            'test/combined_miou': (test_miou * test_rot_miou)**0.5,
            'test/consistency': rot_consistency,
        })
    
    print("\nExperiment completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train semantic segmentation models')
    parser.add_argument('--config', type=str, default='config/segmentation.yaml', help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--config_name', type=str, default='segformer_base', help='Named config inside the YAML file')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for W&B')
    parser.add_argument('--soft_thresholding', type=float, default=None, help='Soft thresholding value (overrides config)')
    parser.add_argument('--soft_thresholding_pos', type=float, default=None, help='Soft thresholding for position embeddings')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--backbone_lr', type=float, default=None, help='Backbone learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--no_augmentation', action='store_true', default=False, help='Disable data augmentation')
    parser.add_argument('--n_rotations', type=int, default=None, help='Rotation group for weight filtering')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup steps (overrides config)')
    parser.add_argument('--embedding_lr', type=float, default=None, help='Embedding learning rate (overrides config)')
    parser.add_argument('--evaluate_only', action='store_true', default=False, help='Skip training and only evaluate')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for evaluation (or resume training)')
    parser.add_argument('--eval_rot', type=float, default=None, help='Rotation for evaluation (overrides config)')
    parser.add_argument('--hard_mask', action='store_true', default=False, help='Use hard mask for filtering')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (overrides config)')
    parser.add_argument('--group_type', type=str, default=None, choices=['rotation', 'reflection', 'roto_reflection'],
                        help='Group type for equivariance: rotation, reflection, or roto_reflection')
    parser.add_argument('--preserve_norm', action='store_true', default=False, help='Preserve norm after projection')
    parser.add_argument('--joint_decomposition', action='store_true', default=False, help='Joint decomposition')
    parser.add_argument('--freeze_patch_embeddings', action='store_true', default=False,
                        help='Freeze patch embedding weights (prevents gradient updates)')
    parser.add_argument('--freeze_position_embeddings', action='store_true', default=False,
                        help='Freeze position embedding weights (prevents gradient updates)')
    parser.add_argument('--freeze_filters', action='store_true', default=False,
                        help='Freeze filtered convolution weights (prevents gradient updates)')
    parser.add_argument('--min_filter_size', type=int, default=None,
                        help='Minimum filter size to freeze (only freeze conv with kernel_size >= this value)')
    parser.add_argument('--use_invariant_loss', action='store_true', default=False, help='Enable invariant loss for training')
    parser.add_argument('--inv_loss_weight', type=float, default=None, help='Weight for invariant loss (overrides config)')
    parser.add_argument('--inv_loss_freq', type=int, default=None, help='Frequency for invariant loss calculation (every N batches, overrides config)')
    args = parser.parse_args()
    main(config_path=args.config, run_name=args.run_name, args=args)








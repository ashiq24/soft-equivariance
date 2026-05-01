import argparse
import os
import random
from typing import Optional

import torch

from config.utils import load_config, get_default_config_path

from data_loader.mnist import get_mnist_dataloaders
from data_loader.printed_digits import get_printed_digits_dataloaders
from data_loader.cifar import get_cifar_dataloaders
from models.get_model import get_model
from train.trainer import Trainer
from log_utils.wandb_utils import init_wandb


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg) -> torch.device:
    mode = cfg.get('experiment', {}).get('device', 'auto')
    if mode == 'cpu':
        return torch.device('cpu')
    if mode == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config_path: Optional[str] = None, run_name: Optional[str] = None, args: Optional[argparse.Namespace] = None):
    cfg_path = config_path or get_default_config_path()
    config_name = getattr(args, 'config_name', None) if args is not None else None
    cfg = load_config(cfg_path, config_name=config_name)
    
    if args is not None:
        if hasattr(args, 'soft_thresholding') and args.soft_thresholding is not None:
            cfg['model']['soft_thresholding'] = args.soft_thresholding
        if hasattr(args, 'seed') and args.seed is not None:
            cfg['experiment']['seed'] = args.seed
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
        if hasattr(args, 'eval_rot') and args.eval_rot is not None:
            cfg['data']['eval_rot'] = args.eval_rot
        if hasattr(args, 'hard_mask') and args.hard_mask:
            cfg['model']['hard_mask'] = True
        if hasattr(args, 'n_rotations') and args.n_rotations is not None:
            cfg['model']['n_rotations'] = args.n_rotations
        if hasattr(args, 'weight_decay') and args.weight_decay is not None:
            cfg['train']['weight_decay'] = args.weight_decay
        if hasattr(args, 'preserve_norm') and args.preserve_norm:
            cfg['model']['preserve_norm'] = True
        if hasattr(args, 'group_type') and args.group_type is not None:
            cfg['model']['group_type'] = args.group_type
            if 'canonicalization' in cfg['model']:
                cfg['model']['canonicalization']['group_type'] = args.group_type
        if hasattr(args, 'residual_strength') and args.residual_strength is not None:
            cfg['model']['residual_strength'] = args.residual_strength
        if hasattr(args, 'use_regular_representation') and args.use_regular_representation:
            cfg['model']['use_regular_representation'] = True
        if hasattr(args, 'use_invariant_loss') and args.use_invariant_loss:
            cfg['train']['use_invariant_loss'] = True
        if hasattr(args, 'inv_loss_weight') and args.inv_loss_weight is not None:
            cfg['train']['inv_loss_weight'] = args.inv_loss_weight
        if hasattr(args, 'inv_loss_freq') and args.inv_loss_freq is not None:
            cfg['train']['inv_loss_freq'] = args.inv_loss_freq
    

    seed = int(cfg.get('experiment', {}).get('seed', 42))
    set_seed(seed)
    device = get_device(cfg)

    use_wandb = bool(cfg.get('logging', {}).get('use_wandb', False))
    logger = None
    if use_wandb:
        keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keys', 'wandb.txt')
        wandb_dir = cfg['logging']['wandb_dir']
        os.makedirs(wandb_dir, exist_ok=True)
        logger, wandb_run = init_wandb(cfg, keys_path=keys_path, run_name=run_name, dir=wandb_dir)
        cfg['logging']['checkpoint_dir'] = wandb_run.dir 
        print(f"Set checkpoint_dir to W&B run directory: {cfg['logging']['checkpoint_dir']}")
        


    dataset_name = cfg.get('data', {}).get('dataset', 'mnist')
    if dataset_name == 'mnist':
        train_loader, val_loader, test_loader = get_mnist_dataloaders(cfg)
    elif dataset_name == 'printed_digits':
        train_loader, val_loader, test_loader = get_printed_digits_dataloaders(cfg)
    elif dataset_name in ['cifar10', 'cifar100']:
        train_loader, val_loader, test_loader = get_cifar_dataloaders(
            dataset_name=dataset_name,
            batch_size=cfg.get('data', {}).get('batch_size', 32),
            num_workers=cfg.get('data', {}).get('num_workers', 4),
            data_dir=cfg.get('data', {}).get('data_dir', './data'),
            augmentation=cfg.get('data', {}).get('augmentation', True),
            augmentation_angle=cfg.get('data', {}).get('augmentation_angle', 10.0),
            augmentation_flip=cfg.get('data', {}).get('augmentation_flip', False),
            image_size=cfg.get('data', {}).get('image_size', 224)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model = get_model(cfg.get('model', {}))

    trainer = Trainer(model, cfg, device=device)
    epochs = int(cfg.get('train', {}).get('epochs', 1))
    log_interval = int(cfg.get('logging', {}).get('log_interval', 50))
    
    evaluate_only = getattr(args, 'evaluate_only', False) if args is not None else False
    checkpoint_path = getattr(args, 'checkpoint_path', None) if args is not None else None
    
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Checkpoint loaded successfully")
            if 'val_acc' in checkpoint:
                print(f"  Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
            if 'epoch' in checkpoint:
                print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        else:
            print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
            return
    
    if not evaluate_only:
        print(f"\nStarting training for {epochs} epochs...")
        trainer.fit(train_loader, val_loader, epochs=epochs, logger=logger, log_interval=log_interval)
        
        if checkpoint_path is None:
            best_checkpoint_path = os.path.join(cfg.get('logging', {}).get('checkpoint_dir', './weights'), 'best.pt')
        else:
            best_checkpoint_path = checkpoint_path
    else:
        print(f"\n{'='*80}")
        print("EVALUATION-ONLY MODE: Skipping training")
        print(f"{'='*80}\n")
        best_checkpoint_path = checkpoint_path
    
    best_checkpoint_path = best_checkpoint_path or os.path.join(cfg.get('logging', {}).get('checkpoint_dir', './weights'), 'best.pt')
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best checkpoint from {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_acc = checkpoint.get('val_acc', None)
        print(f"✓ Loaded best model with validation accuracy: {val_acc:.4f}" if val_acc is not None else "✓ Loaded best model (val_acc not in checkpoint)")
        
        if logger is not None:
            try:
                import wandb
                wandb.run.summary["best_checkpoint_path"] = best_checkpoint_path
            except Exception:
                pass
    else:
        print(f"Best checkpoint not found at {best_checkpoint_path}, using current model weights")
    

    test_loss, test_acc, test_topk_acc, test_cm, test_true, test_pred, test_rot_loss, test_rot_acc, test_rot_topk_acc, test_rot_cm, test_rot_true, test_rot_pred, rot_consistency = trainer.evaluate(test_loader)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | Test top-{trainer.topk} acc: {test_topk_acc:.4f}")
    print(f"Test rotation loss: {test_rot_loss:.4f} | Test rotation acc: {test_rot_acc:.4f} | Test rotation top-{trainer.topk} acc: {test_rot_topk_acc:.4f}")
    if test_cm is not None:
        print("Test confusion matrix (rows=true, cols=pred):")
        print(test_cm)
    if logger is not None:
        logger({
            'test/loss': test_loss,
            'test/acc': test_acc,
            'test/topk_acc': test_topk_acc,
            'rot/rot_loss': test_rot_loss,
            'rot/rot_acc': test_rot_acc,
            'rot/rot_topk_acc': test_rot_topk_acc,
            'test/combined_acc': (test_acc * test_rot_acc)**0.5,
            'test/combined_topk_acc': (test_topk_acc * test_rot_topk_acc)**0.5,
            'test/consistency': rot_consistency,
        })
        
        if test_cm is not None and test_true is not None and test_pred is not None:
            try:
                import wandb  # type: ignore
                class_names = [str(i) for i in range(10)]  # MNIST digits 0-9
                logger({'test/confusion_matrix': wandb.plot.confusion_matrix(y_true=test_true, preds=test_pred, class_names=class_names)})
            except Exception:
                pass
        if test_rot_cm is not None and test_rot_true is not None and test_rot_pred is not None:
            try:
                import wandb  # type: ignore
                class_names = [str(i) for i in range(10)]  # MNIST digits 0-9
                logger({'rot_test/rot_confusion_matrix': wandb.plot.confusion_matrix(y_true=test_rot_true, preds=test_rot_pred, class_names=class_names)})
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--config_name', type=str, default=None, help='Optional named config inside a multi-config YAML')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for W&B')
    parser.add_argument('--soft_thresholding', type=float, default=None, help='Soft thresholding')
    parser.add_argument('--soft_thresholding_pos', type=float, default=None, help='Soft thresholding for position embeddings')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--backbone_lr', type=float, default=None, help='Backbone learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--evaluate_only', action='store_true', default=False, help='Skip training and only evaluate')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for evaluation (or resume training)')
    parser.add_argument('--eval_rot', type=float, default=None, help='Evaluate rotation prediction task only')
    parser.add_argument('--n_rotations', type=int, default=None, help='Number of rotations for filter')
    parser.add_argument('--hard_mask', action='store_true', default=False, help='Use hard mask for filtering')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--preserve_norm', action='store_true', default=False, help='Preserve norm after projection')
    parser.add_argument('--group_type', type=str, default=None, choices=['rotation', 'reflection', 'roto_reflection'],
                        help='Group type for equivariance: rotation, reflection, or roto_reflection')
    parser.add_argument('--residual_strength', type=float, default=None, help='Residual mixing strength for residual_cnn')
    parser.add_argument('--use_regular_representation', action='store_true', default=False,
                        help='Use regular representation in equivariant branch for residual_cnn')
    parser.add_argument('--use_invariant_loss', action='store_true', default=False, help='Enable invariant loss for training')
    parser.add_argument('--inv_loss_weight', type=float, default=None, help='Weight for invariant loss (overrides config)')
    parser.add_argument('--inv_loss_freq', type=int, default=None, help='Frequency for invariant loss calculation (every N batches, overrides config)')
    args = parser.parse_args()
    main(config_path=args.config, run_name=args.run_name, args=args)



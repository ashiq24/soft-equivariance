"""
Trainer for semantic segmentation tasks.

This module provides a specialized trainer for semantic segmentation that handles:
- Segmentation-specific loss functions (CrossEntropyLoss with ignore_index)
- mIoU metric calculation
- Logit upsampling to match label resolution
- Rotational evaluation for segmentation masks
"""

from typing import Dict, Optional
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
from tqdm import tqdm

from utils.metrics import SegmentationMetrics
from utils.consistency import test_on_augmented, get_eq_error


class SegTrainer:
    """
    Trainer class for semantic segmentation models.
    """
    def __init__(self, model: nn.Module, cfg: Dict, device: Optional[torch.device] = None):
        self.model = model
        self.cfg = cfg
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(self.device)

        train_cfg = cfg.get('train', {})
        lr = float(train_cfg['lr'])
        weight_decay = float(train_cfg['weight_decay'] if train_cfg['weight_decay'] is not None else 0.0)
        backbone_lr = train_cfg.get('backbone_lr', None)
        embedding_lr = train_cfg.get('embedding_lr', None)
        self.max_grad_norm = train_cfg.get('max_grad_norm', None)

        self.ignore_index = 255
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        param_groups = self._create_param_groups(model, lr, backbone_lr, embedding_lr, weight_decay)
        
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = None
        self.num_classes = cfg.get('model', {}).get('num_labels', 21)

        self.use_invariant_loss = train_cfg.get('use_invariant_loss', False)
        self.inv_loss_weight = float(train_cfg.get('inv_loss_weight', 0.0))
        self.inv_loss_freq = int(train_cfg.get('inv_loss_freq', 1))

        ckpt_dir = cfg['logging']['checkpoint_dir']
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_miou = -1.0

    @staticmethod
    def _is_embedding_param(name: str) -> bool:
        """
        Check if parameter belongs to embedding layers.
        
        Embedding parameters include:
        - patch_embeddings.projection (Conv2d that patchifies images)
        - position_embeddings (learnable positional encoding)
        - cls_token (class token for ViT/DINOv2)
        """
        embedding_keywords = [
            'embeddings.patch_embeddings.projection',
            'embeddings.position_embeddings',
            'embeddings.cls_token'
        ]
        return any(keyword in name for keyword in embedding_keywords)
    
    @staticmethod
    def _is_head_param(name: str) -> bool:
        """
        Check if parameter belongs to task-specific head.
        
        Head parameters include:
        - canon_net (canonicalization network)
        - classifier, decode_head, head (but NOT attention heads)
        """
        if 'canon_net' in name.lower():
            return True
        
        if any(kw in name.lower() for kw in ['classifier', 'decode_head', 'head']):
            if 'attention' not in name.lower() and 'attn' not in name.lower():
                return True
        
        return False

    def _create_param_groups(self, model: nn.Module, lr: float, backbone_lr: Optional[float],
                            embedding_lr: Optional[float], weight_decay: float):
        """
        Create parameter groups with differential learning rates for backbone, embeddings, and head.
        
        Simplified logic:
        1. Classify all parameters into 3 groups
        2. Determine effective LR for each group
        3. Freeze parameters with lr=0.0
        4. Build parameter groups for trainable params
        
        Args:
            model: The model to create parameter groups for
            lr: Learning rate for head parameters
            backbone_lr: Learning rate for backbone (None = use head lr)
            embedding_lr: Learning rate for embeddings (None = use backbone lr)
            weight_decay: Weight decay for all parameters
            
        Returns:
            List of parameter group dictionaries for optimizer
        """
        embedding_params = []
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if self._is_embedding_param(name):
                embedding_params.append(param)
            elif self._is_head_param(name):
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        embedding_count = sum(p.numel() for p in embedding_params)
        backbone_count = sum(p.numel() for p in backbone_params)
        head_count = sum(p.numel() for p in head_params)
        total_count = embedding_count + backbone_count + head_count
        
        effective_embedding_lr = embedding_lr if embedding_lr is not None else (backbone_lr if backbone_lr is not None else lr)
        effective_backbone_lr = backbone_lr if backbone_lr is not None else lr
        effective_head_lr = lr
        
        if effective_embedding_lr == 0.0:
            for param in embedding_params:
                param.requires_grad = False
        
        if effective_backbone_lr == 0.0:
            for param in backbone_params:
                param.requires_grad = False
        
        if effective_head_lr == 0.0:
            for param in head_params:
                param.requires_grad = False
        
        trainable_embedding = sum(p.numel() for p in embedding_params if p.requires_grad)
        trainable_backbone = sum(p.numel() for p in backbone_params if p.requires_grad)
        trainable_head = sum(p.numel() for p in head_params if p.requires_grad)
        trainable_total = trainable_embedding + trainable_backbone + trainable_head
        
        param_groups = []
        lr_to_params = {}
        
        if trainable_backbone > 0:
            if effective_backbone_lr not in lr_to_params:
                lr_to_params[effective_backbone_lr] = []
            lr_to_params[effective_backbone_lr].extend(backbone_params)
        
        if trainable_embedding > 0:
            if effective_embedding_lr not in lr_to_params:
                lr_to_params[effective_embedding_lr] = []
            lr_to_params[effective_embedding_lr].extend(embedding_params)
        
        if trainable_head > 0:
            if effective_head_lr not in lr_to_params:
                lr_to_params[effective_head_lr] = []
            lr_to_params[effective_head_lr].extend(head_params)
        
        for lr_val in sorted(lr_to_params.keys()):
            param_groups.append({
                'params': [p for p in lr_to_params[lr_val] if p.requires_grad],
                'lr': lr_val,
                'weight_decay': weight_decay
            })
        
        print(f"\n{'='*80}")
        if len(param_groups) == 1:
            if trainable_total == total_count:
                print(f"STANDARD TRAINING MODE")
            else:
                print(f"LINEAR PROBING MODE")
        elif len(param_groups) == 2:
            print(f"2-WAY DIFFERENTIAL LEARNING RATES")
        else:
            print(f"3-WAY DIFFERENTIAL LEARNING RATES")
        print(f"{'='*80}")
        
        print(f"Backbone: {backbone_count:,} params (lr={effective_backbone_lr if trainable_backbone > 0 else 0.0:.2e}, {'trainable' if trainable_backbone > 0 else 'frozen'})")
        print(f"Embeddings: {embedding_count:,} params (lr={effective_embedding_lr if trainable_embedding > 0 else 0.0:.2e}, {'trainable' if trainable_embedding > 0 else 'frozen'})")
        print(f"Head: {head_count:,} params (lr={effective_head_lr if trainable_head > 0 else 0.0:.2e}, {'trainable' if trainable_head > 0 else 'frozen'})")
        print(f"Total trainable: {trainable_total:,} / {total_count:,} ({100*trainable_total/total_count:.1f}%)")
        
        if len(param_groups) > 1:
            unique_lrs = sorted(lr_to_params.keys())
            print(f"Learning rates: {', '.join([f'{lr_val:.2e}' for lr_val in unique_lrs])}")
        
        print(f"{'='*80}\n")
        
        return param_groups

    def _setup_scheduler(self, epochs: int, steps_per_epoch: int):
        """
        Setup learning rate scheduler with optional warmup.
        
        Args:
            epochs: Total number of training epochs
            steps_per_epoch: Number of optimization steps per epoch
            
        Returns:
            Tuple of (scheduler, step_per_iter) where step_per_iter indicates
            whether to step the scheduler per iteration (True) or per epoch (False)
        """
        train_cfg = self.cfg.get('train', {})
        warmup_steps = train_cfg.get('warmup_steps', 0)
        scheduler_cfg = train_cfg.get('scheduler', {}).copy()
        
        if warmup_steps > 0:
            total_steps = epochs * steps_per_epoch
            
            def warmup_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0
            
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=warmup_lambda
            )
            
            if scheduler_cfg:
                scheduler_type = scheduler_cfg.pop('type', 'CosineAnnealingLR')
                
                if scheduler_type == 'CosineAnnealingLR':
                    scheduler_cfg['T_max'] = total_steps - warmup_steps
                    if 'eta_min' in scheduler_cfg:
                        scheduler_cfg['eta_min'] = float(scheduler_cfg['eta_min'])
                
                scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
                main_scheduler = scheduler_class(self.optimizer, **scheduler_cfg)
                
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                scheduler = warmup_scheduler
            
            return scheduler, True
        
        else:
            if scheduler_cfg:
                scheduler_type = scheduler_cfg.pop('type', 'CosineAnnealingLR')
                
                if scheduler_type == 'LinearLR' and 'total_iters' not in scheduler_cfg:
                    scheduler_cfg['total_iters'] = epochs
                
                if scheduler_type == 'CosineAnnealingLR' and 'eta_min' in scheduler_cfg:
                    scheduler_cfg['eta_min'] = float(scheduler_cfg['eta_min'])
                
                scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
                scheduler = scheduler_class(self.optimizer, **scheduler_cfg)
                
                return scheduler, False
            else:
                return None, False

    def train_one_epoch(self, loader: DataLoader, epoch: int, log_interval: int = 50, logger=None):
        """
        Train for one epoch.
        
        Args:
            loader: Training data loader
            epoch: Current epoch number
            log_interval: How often to log metrics
            logger: Optional W&B logger
            
        Returns:
            epoch_loss: Average loss for the epoch
            epoch_miou: Average mIoU for the epoch
        """
        self.model.train()
        running_loss = 0.0
        metrics = SegmentationMetrics(self.num_classes, ignore_index=self.ignore_index)
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(pixel_values=images, labels=masks)
            
            if isinstance(outputs, dict):
                loss = outputs.get('loss')
                logits = outputs.get('logits')
            else:
                logits = outputs
                loss = None
            
            if logits.shape[-2:] != masks.shape[-2:]:
                logits_upsampled = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                logits_upsampled = logits
            
            if loss is None:
                loss = self.criterion(logits_upsampled, masks)

            if self.use_invariant_loss and (batch_idx % self.inv_loss_freq == 0):
                eval_rot = self.cfg.get('data', {}).get('eval_rot', 30.0)
                group_type = self.cfg.get('model', {}).get('group_type', 'rotation')
                rotation_config = eval_rot if group_type in ['rotation', 'roto_reflection'] else None
                
                def model_wrapper(x):
                    outputs = self.model(pixel_values=x)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits')
                    else:
                        logits = outputs
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = F.interpolate(
                            logits,
                            size=masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    return logits
                
                inv_error = get_eq_error(
                    batch_inputs=images,
                    model=model_wrapper,
                    rotation_config=rotation_config,
                    task_type='segmentation',
                    batch_targets=masks,
                    device=self.device,
                    original_logits=logits_upsampled,
                    group_type=group_type
                )
                
                loss = loss + self.inv_loss_weight * inv_error

            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            if hasattr(self, 'step_scheduler_per_iter') and self.step_scheduler_per_iter:
                if self.scheduler is not None:
                    self.scheduler.step()

            running_loss += loss.item() * images.size(0)
            metrics.update(logits_upsampled.detach(), masks)

            if batch_idx % 50 == 0:
                current_miou, _ = metrics.get_miou()
            if len(self.optimizer.param_groups) == 3:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    miou=f"{current_miou:.4f}",
                    backbone_lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    embed_lr=f"{self.optimizer.param_groups[1]['lr']:.2e}",
                    head_lr=f"{self.optimizer.param_groups[2]['lr']:.2e}"
                )
            elif len(self.optimizer.param_groups) == 2:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    miou=f"{current_miou:.4f}",
                    lr1=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    lr2=f"{self.optimizer.param_groups[1]['lr']:.2e}"
                )
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss.item():.4f}", miou=f"{current_miou:.4f}", lr=f"{current_lr:.2e}")

            if logger is not None and (batch_idx + 1) % log_interval == 0:
                log_dict = {
                    'train/iter_loss': loss.item(),
                    'train/iter_miou': current_miou,
                    'train/iter': batch_idx + 1,
                    'train/epoch': epoch,
                    'train/lr': self.optimizer.param_groups[-1]['lr'],
                }
                if len(self.optimizer.param_groups) == 3:
                    log_dict['train/backbone_lr'] = self.optimizer.param_groups[0]['lr']
                    log_dict['train/embedding_lr'] = self.optimizer.param_groups[1]['lr']
                    log_dict['train/head_lr'] = self.optimizer.param_groups[2]['lr']
                elif len(self.optimizer.param_groups) == 2:
                    log_dict['train/lr1'] = self.optimizer.param_groups[0]['lr']
                    log_dict['train/lr2'] = self.optimizer.param_groups[1]['lr']
                logger(log_dict)

        epoch_loss = running_loss / len(loader.dataset)
        epoch_miou, _ = metrics.get_miou()
        return epoch_loss, epoch_miou

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, include_transformation: bool = True):
        """
        Evaluate the model on a dataset.
        
        Args:
            loader: Validation/test data loader
            include_transformation: Whether to also evaluate with group transformation augmentation
            
        Returns:
            Tuple of (loss, miou, pixel_acc, rot_loss, rot_miou, rot_pixel_acc)
        """
        self.model.eval()
        running_loss = 0.0
        metrics = SegmentationMetrics(self.num_classes, ignore_index=self.ignore_index)
        
        rot_running_loss = 0.0
        rot_metrics = SegmentationMetrics(self.num_classes, ignore_index=self.ignore_index)
        rot_evaluated = False
        rot_consistency_sum = 0.0
        rot_consistency_count = 0
        
        for images, masks in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(pixel_values=images)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits')
            else:
                logits = outputs
            
            if logits.shape[-2:] != masks.shape[-2:]:
                logits_upsampled = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                logits_upsampled = logits
            
            loss = self.criterion(logits_upsampled, masks)
            if hasattr(self.model, 'prior_loss') and self.model.prior_loss is not None:
                loss += self.model.prior_loss
                
            running_loss += loss.item() * images.size(0)
            metrics.update(logits_upsampled, masks)
            
            if include_transformation:
                rot_evaluated = True
                eval_rot = self.cfg['data']['eval_rot']
                group_type = self.cfg.get('model', {}).get('group_type', 'rotation')
                
                rotation_config = eval_rot if group_type in ['rotation', 'roto_reflection'] else None
                
                def model_wrapper(x):
                    outputs = self.model(pixel_values=x)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits')
                    else:
                        logits = outputs
                        logits = F.interpolate(
                            logits,
                            mode='bilinear',
                            align_corners=False
                        )
                    return logits
                
                aug_results = test_on_augmented(
                    batch_inputs=images,
                    model=model_wrapper,
                    task_type='segmentation',
                    criterion=self.criterion,
                    device=self.device,
                    return_consistency=True,
                    group_type=group_type
                )
                
                rot_logits_upsampled = aug_results['augmented_logits']
                rot_masks = aug_results['augmented_targets']
                rot_loss = aug_results.get('augmented_loss', self.criterion(rot_logits_upsampled, rot_masks))
                rot_consistency = aug_results.get('consistency', 0.0)
                
                rot_running_loss += rot_loss.item() * images.size(0)
                rot_metrics.update(rot_logits_upsampled, rot_masks)
                rot_consistency_sum += rot_consistency.item() * images.size(0)
                rot_consistency_count += images.size(0)
                del rot_logits_upsampled, rot_masks, rot_loss, rot_consistency, aug_results
            
            del loss, logits_upsampled, outputs, logits

        epoch_loss = running_loss / len(loader.dataset)
        epoch_miou, iou_per_class = metrics.get_miou()
        pixel_acc = metrics.get_pixel_accuracy()
        
        if rot_evaluated:
            rot_loss = rot_running_loss / len(loader.dataset)
            rot_miou, rot_iou_per_class = rot_metrics.get_miou()
            rot_pixel_acc = rot_metrics.get_pixel_accuracy()
            rot_consistency = rot_consistency_sum / max(rot_consistency_count, 1) if rot_consistency_count > 0 else 0.0
        else:
            rot_loss = 0.0
            rot_pixel_acc = 0.0
            rot_consistency = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Evaluation - Loss: {epoch_loss:.4f}, mIoU: {epoch_miou:.4f}, Pixel Acc: {pixel_acc:.4f}")
        if rot_evaluated:
            print(f"Rotation Eval - Loss: {rot_loss:.4f}, mIoU: {rot_miou:.4f}, Pixel Acc: {rot_pixel_acc:.4f}, Consistency: {rot_consistency:.4f}")
        
        return epoch_loss, epoch_miou, pixel_acc, rot_loss, rot_miou, rot_pixel_acc, rot_consistency

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
            epochs: int = 1, logger=None, log_interval: int = 50):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            logger: Optional W&B logger
            log_interval: How often to log metrics
            
        Returns:
            history: List of (train_loss, train_miou, val_loss, val_miou) tuples
        """
        self.scheduler, self.step_scheduler_per_iter = self._setup_scheduler(epochs, len(train_loader))
        
        history = []
        for epoch in range(1, epochs + 1):
            train_loss, train_miou = self.train_one_epoch(
                train_loader, epoch, log_interval=log_interval, logger=logger
            )
            
            val_loss, val_miou, val_pixel_acc = None, None, None
            val_rot_loss, val_rot_miou, val_rot_pixel_acc, val_rot_consistency = None, None, None, None
            
            if val_loader is not None:
                val_loss, val_miou, val_pixel_acc, val_rot_loss, val_rot_miou, val_rot_pixel_acc, val_rot_consistency = self.evaluate(val_loader, include_transformation=False)

            if val_loader is not None:
                print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} train_miou: {train_miou:.4f} | "
                      f"val_loss: {val_loss:.4f} val_miou: {val_miou:.4f} | "
                      f"val_rot_miou: {val_rot_miou:.4f} rot_consistency: {val_rot_consistency:.4f}")
            else:
                print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} train_miou: {train_miou:.4f}")

            if logger is not None:
                payload = {
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/miou': train_miou,
                    'train/lr': self.optimizer.param_groups[-1]['lr'],
                }
                if len(self.optimizer.param_groups) == 3:
                    payload['train/backbone_lr'] = self.optimizer.param_groups[0]['lr']
                    payload['train/embedding_lr'] = self.optimizer.param_groups[1]['lr']
                    payload['train/head_lr'] = self.optimizer.param_groups[2]['lr']
                elif len(self.optimizer.param_groups) == 2:
                    payload['train/lr1'] = self.optimizer.param_groups[0]['lr']
                    payload['train/lr2'] = self.optimizer.param_groups[1]['lr']
                
                if val_loader is not None:
                    payload.update({
                        'val/loss': val_loss,
                        'val/miou': val_miou,
                        'val/pixel_acc': val_pixel_acc,
                        'val/rot_loss': val_rot_loss,
                        'val/rot_miou': val_rot_miou,
                        'val/rot_pixel_acc': val_rot_pixel_acc,
                        'val/rot_consistency': val_rot_consistency,
                        'val/combined_miou': (val_miou * val_rot_miou)**0.5,
                    })
                logger(payload)

            history.append((train_loss, train_miou, val_loss, val_miou))
            
            if val_loader is not None:
                if val_miou is not None and val_miou > self.best_miou:
                    self.best_miou = val_miou
                    self._save_checkpoint('best.pt', epoch, train_loss, train_miou, val_loss, val_miou)
            
            self._save_checkpoint('last.pt', epoch, train_loss, train_miou, val_loss, val_miou)
            
            if self.scheduler is not None and not self.step_scheduler_per_iter:
                self.scheduler.step()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return history

    def _save_checkpoint(self, name: str, epoch: int, train_loss: float, train_miou: float, 
                        val_loss: Optional[float], val_miou: Optional[float]):
        """Save model checkpoint."""
        path = os.path.join(self.ckpt_dir, name)
        was_training = self.model.training
        self.model.eval()
        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'train_miou': train_miou,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'cfg': self.cfg,
        }
        torch.save(payload, path)
        if was_training:
            self.model.train()
        print(f"Saved checkpoint: {path}")








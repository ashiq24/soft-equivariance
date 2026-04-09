"""
Trajectory prediction trainer.

Handles training and evaluation of trajectory prediction models.
Follows the pattern established in seg_trainer.py but adapted for trajectory data.
"""

from typing import Dict, Optional
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
from tqdm import tqdm

from utils.trajectory_metrics import TrajectoryMetrics
from utils.trajectory_rotation import rotate_trajectory_2d
from utils.consistency import test_on_augmented


class TrajTrainer:
    """
    Trainer class for trajectory prediction models.
    """
    def __init__(self, model: nn.Module, cfg: Dict, device: Optional[torch.device] = None):
        """
        Initialize trajectory trainer.
        
        Args:
            model: Trajectory prediction model
            cfg: Configuration dictionary
            device: Device to use for training (default: auto-detect)
        """
        self.model = model
        self.cfg = cfg
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(self.device)

        train_cfg = cfg.get('train', {})
        lr = float(train_cfg['lr'])
        weight_decay = float(train_cfg['weight_decay'])

        self.criterion = nn.MSELoss(reduction='none')
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None

        self.obs_len = cfg.get('data', {}).get('obs_len', 8)
        self.pred_len = cfg.get('data', {}).get('pred_len', 12)

        ckpt_dir = cfg['logging']['checkpoint_dir']
        self.ckpt_dir = ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        elif not os.path.isdir(self.ckpt_dir):
            raise ValueError(f"Checkpoint directory path exists but is not a directory: {self.ckpt_dir}")
        self.best_ade = float('inf')

    def train_one_epoch(self, loader: DataLoader, epoch: int, log_interval: int = 10, logger=None):
        """
        Train for one epoch.
        
        Args:
            loader: Training data loader
            epoch: Current epoch number
            log_interval: How often to log metrics
            logger: Optional W&B logger
            
        Returns:
            epoch_loss: Average loss for the epoch
            epoch_ade: Average ADE for the epoch
            epoch_fde: Average FDE for the epoch
        """
        self.model.train()
        running_loss = 0.0
        metrics = TrajectoryMetrics()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            obs_traj = batch['obs_traj'].to(self.device)
            pred_traj_gt = batch['pred_traj'].to(self.device)
            validity_mask = batch['validity_mask'].to(self.device)
            
            pred_traj = self.model(obs_traj, validity_mask, pred_traj_gt)
            
            loss_per_element = self.criterion(pred_traj, pred_traj_gt)
            
            validity_mask_expanded = validity_mask.unsqueeze(2).unsqueeze(3)
            masked_loss = loss_per_element * validity_mask_expanded
            
            loss = masked_loss.sum() / (validity_mask.sum() * 2 * self.pred_len + 1e-8)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * obs_traj.size(0)
            
            metrics.update(pred_traj, pred_traj_gt, validity_mask)

            current_ade = metrics.get_ade()
            current_fde = metrics.get_fde()
            pbar.set_postfix(loss=f"{loss.item():.4f}", ade=f"{current_ade:.4f}", fde=f"{current_fde:.4f}")

            if logger is not None and (batch_idx + 1) % log_interval == 0:
                logger({
                    'train/iter_loss': loss.item(),
                    'train/iter_ade': current_ade,
                    'train/iter_fde': current_fde,
                    'train/iter': batch_idx + 1,
                    'train/epoch': epoch,
                })

        epoch_loss = running_loss / len(loader.dataset)
        epoch_ade = metrics.get_ade()
        epoch_fde = metrics.get_fde()
        return epoch_loss, epoch_ade, epoch_fde

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, include_transformation: bool = True):
        """
        Evaluate the model on a dataset.
        
        Args:
            loader: Validation/test data loader
            include_transformation: Whether to also evaluate with group transformation augmentation
            
        Returns:
            Tuple of (loss, ade, fde, rot_loss, rot_ade, rot_fde)
        """
        self.model.eval()
        running_loss = 0.0
        metrics = TrajectoryMetrics()
        
        rot_running_loss = 0.0
        rot_metrics = TrajectoryMetrics()
        rot_evaluated = False
        rot_consistency_sum = 0.0
        rot_consistency_count = 0
        
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            obs_traj = batch['obs_traj'].to(self.device)
            pred_traj_gt = batch['pred_traj'].to(self.device)
            validity_mask = batch['validity_mask'].to(self.device)
            
            pred_traj = self.model(obs_traj, validity_mask)
            
            loss_per_element = self.criterion(pred_traj, pred_traj_gt)
            validity_mask_expanded = validity_mask.unsqueeze(2).unsqueeze(3)
            masked_loss = loss_per_element * validity_mask_expanded
            loss = masked_loss.sum() / (validity_mask.sum() * 2 * self.pred_len + 1e-8)
            
            running_loss += loss.item() * obs_traj.size(0)
            
            metrics.update(pred_traj, pred_traj_gt, validity_mask)
            
            if include_transformation:
                rot_evaluated = True
                eval_rot = self.cfg['data']['eval_rot']
                group_type = self.cfg.get('model', {}).get('group_type', 'rotation')
                reflection_axis = self.cfg.get('model', {}).get('reflection_axis', 'x')
                
                rotation_config = eval_rot if group_type in ['rotation', 'roto_reflection'] else None
                
                aug_results = test_on_augmented(
                    batch_inputs=obs_traj,
                    model=self.model,
                    rotation_config=rotation_config,
                    task_type='trajectory',
                    batch_targets=pred_traj_gt,
                    validity_mask=validity_mask,
                    criterion=self.criterion,
                    device=self.device,
                    return_consistency=True,
                    original_outputs=pred_traj,
                    original_logits=pred_traj,
                    group_type=group_type,
                    reflection_axis=reflection_axis
                )
                
                rot_pred_traj = aug_results['augmented_logits']
                rot_pred_traj_gt = aug_results['augmented_targets']
                rot_consistency = aug_results.get('consistency', 0.0)
                
                rot_loss_per_element = self.criterion(rot_pred_traj, rot_pred_traj_gt)
                rot_masked_loss = rot_loss_per_element * validity_mask_expanded
                rot_loss_val = rot_masked_loss.sum() / (validity_mask.sum() * 2 * self.pred_len + 1e-8)
                
                rot_running_loss += rot_loss_val.item() * obs_traj.size(0)
                
                rot_metrics.update(rot_pred_traj, rot_pred_traj_gt, validity_mask)
                
                rot_consistency_sum += rot_consistency.item() * obs_traj.size(0)
                rot_consistency_count += obs_traj.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        epoch_ade = metrics.get_ade()
        epoch_fde = metrics.get_fde()
        
        if rot_evaluated:
            rot_loss = rot_running_loss / len(loader.dataset)
            rot_ade = rot_metrics.get_ade()
            rot_fde = rot_metrics.get_fde()
            rot_consistency = rot_consistency_sum / max(rot_consistency_count, 1) if rot_consistency_count > 0 else 0.0
        else:
            rot_loss = 0.0
            rot_ade = 0.0
            rot_fde = 0.0
            rot_consistency = 0.0

        return epoch_loss, epoch_ade, epoch_fde, rot_loss, rot_ade, rot_fde, rot_consistency

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
            epochs: int = 1, logger=None, log_interval: int = 10):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            logger: Optional W&B logger
            log_interval: How often to log batch metrics
            
        Returns:
            history: List of dictionaries with metrics per epoch
        """
        scheduler_cfg = self.cfg.get('train', {}).get('scheduler', {})
        if scheduler_cfg:
            scheduler_type = scheduler_cfg.get('type', 'StepLR')
            scheduler_params = {k: v for k, v in scheduler_cfg.items() if k != 'type'}
            if scheduler_type == 'StepLR' and 'step_size' not in scheduler_params:
                scheduler_params['step_size'] = max(epochs // 3, 1)
            
            if 'T_max' in scheduler_params:
                scheduler_params['T_max'] = int(scheduler_params['T_max'])
            if 'eta_min' in scheduler_params:
                scheduler_params['eta_min'] = float(scheduler_params['eta_min'])
            if 'step_size' in scheduler_params:
                scheduler_params['step_size'] = int(scheduler_params['step_size'])
            if 'gamma' in scheduler_params:
                scheduler_params['gamma'] = float(scheduler_params['gamma'])
            
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
            self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None
        
        history = []
        for epoch in range(1, epochs + 1):
            train_loss, train_ade, train_fde = self.train_one_epoch(
                train_loader, epoch, log_interval=log_interval, logger=logger
            )
            
            val_loss, val_ade, val_fde, val_rot_loss, val_rot_ade, val_rot_fde, val_rot_consistency = (None,) * 7
            if val_loader is not None:
                val_loss, val_ade, val_fde, val_rot_loss, val_rot_ade, val_rot_fde, val_rot_consistency = self.evaluate(val_loader)

            if val_loader is not None:
                combined_ade = val_ade * val_rot_ade
                combined_fde = val_fde * val_rot_fde
                print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} train_ade: {train_ade:.4f} train_fde: {train_fde:.4f} | "
                      f"val_loss: {val_loss:.4f} val_ade: {val_ade:.4f} val_fde: {val_fde:.4f} | "
                      f"val_rot_loss: {val_rot_loss:.4f} val_rot_ade: {val_rot_ade:.4f} val_rot_fde: {val_rot_fde:.4f} | "
                      f"rot_consistency: {val_rot_consistency:.4f} | "
                      f"combined_ade: {combined_ade:.4f} combined_fde: {combined_fde:.4f}")
            else:
                print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} train_ade: {train_ade:.4f} train_fde: {train_fde:.4f}")

            if logger is not None:
                payload = {
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/ade': train_ade,
                    'train/fde': train_fde,
                }
                if val_loader is not None:
                    payload.update({
                        'val/loss': val_loss,
                        'val/ade': val_ade,
                        'val/fde': val_fde,
                        'val/rot_loss': val_rot_loss,
                        'val/rot_ade': val_rot_ade,
                        'val/rot_fde': val_rot_fde,
                        'val/rot_consistency': val_rot_consistency,
                        'val/combined_ade': (val_ade + val_rot_ade)/2,
                        'val/combined_fde': (val_fde + val_rot_fde)/2,
                    })
                logger(payload)

            if val_loader is not None and val_ade < self.best_ade:
                self.best_ade = val_ade
                self._save_checkpoint('best.pt', epoch, train_loss, train_ade, train_fde, val_loss, val_ade, val_fde)
                print(f"Saved best model with validation ADE: {val_ade:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

        return history

    def _save_checkpoint(self, name: str, epoch: int, train_loss: float, train_ade: float, train_fde: float,
                        val_loss: Optional[float], val_ade: Optional[float], val_fde: Optional[float]):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.ckpt_dir, name)
        was_training = self.model.training
        self.model.eval()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'train_ade': train_ade,
            'train_fde': train_fde,
            'val_loss': val_loss,
            'val_ade': val_ade,
            'val_fde': val_fde,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        if was_training:
            self.model.train()

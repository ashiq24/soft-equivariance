"""
Generic trainer for equivariant MLPs on synthetic tasks.

This trainer works with any EMLP-supported group and uses a universal
equivariance testing method that works generically across all groups.

Key Features:
- Standard supervised learning training loop
- Universal equivariance testing via group augmentation
- Works with any group (O(5), SO(3), S(5), etc.)
- Optional W&B logging and checkpointing
"""

from math import e
import os
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Any
import logging
import numpy as np
import pdb
logger = logging.getLogger(__name__)


def relative_norm_error(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Compute relative norm error: ||a - b|| / (||a|| + ||b||)
    
    Args:
        predictions: Predicted values tensor
        targets: Target values tensor
        epsilon: Small value to prevent division by zero
    
    Returns:
        Mean relative norm error as a scalar
    """
    numerator = torch.norm(predictions - targets, dim=-1)
    denominator = torch.norm(predictions, dim=-1) + torch.norm(targets, dim=-1)
    return torch.mean(numerator / (denominator + epsilon)).item()


class MiscTrainer:
    """
    Generic trainer for equivariant MLPs on synthetic tasks.
    
    Supports:
    - Any EMLP group (O(n), SO(n), SE(n), S(n), etc.)
    - Custom representations via configuration
    - Universal group-agnostic equivariance testing
    - W&B logging and model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        use_wandb: bool = False,
        experiment_name: str = "equivariant_mlp",
        logger=None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Equivariant MLP model
            device: Device to use (torch.device)
            config: Configuration dictionary with keys:
                - training.learning_rate
                - training.weight_decay
                - training.checkpoint_dir
                - experiment.group (for equivariance testing)
            use_wandb: Whether to log to Weights & Biases
            experiment_name: Name for W&B experiment
            logger: Optional W&B logger function for custom logging
        """
        self.model = model
        self.device = device
        self.config = config
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name
        self.logger = logger
        
        training_config = config.get('training', {})
        self.lr = training_config.get('learning_rate', 1e-3)
        self.weight_decay = training_config.get('weight_decay', 1e-5)
        self.checkpoint_dir = training_config.get('checkpoint_dir', 'logs/misc')
        self.group = config['model']['group']
        
        self.scheduler_config = training_config.get('scheduler', {})
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.criterion = nn.MSELoss()
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = None
        
        self.best_val_metric = float('inf')
        self.best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        
        if self.use_wandb and self.logger is None:
            try:
                import wandb
                wandb.init(
                    project="soft-equivariance",
                    name=experiment_name,
                    config=config,
                    dir=self.checkpoint_dir,
                )
                logging.getLogger(__name__).info("W&B logging initialized")
            except ImportError:
                logging.getLogger(__name__).warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False
    
    def train_one_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        log_interval: int = 10,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            loader: Training data loader
            epoch: Current epoch number
            log_interval: How often to log metrics
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = len(loader)
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                X, y = batch
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            X = X.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            
            loss = self.criterion(y_pred, y)
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                if self.logger is not None:
                    log_dict = {
                        'train/iter_loss': loss.item(),
                        'train/iter': batch_idx + 1,
                        'train/epoch': epoch,
                        'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                    if len(self.optimizer.param_groups) > 1:
                        for i, pg in enumerate(self.optimizer.param_groups):
                            log_dict[f'train/lr_group_{i}'] = pg['lr']
                    else:
                        log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                    
                    self.logger(log_dict)
        
        epoch_loss = total_loss / n_batches
        
        return epoch_loss
    
    def evaluate(
        self,
        loader: DataLoader,
        compute_mse: bool = True,
        compute_mae: bool = True,
        include_augmented: bool = True,
    ) -> Tuple[float, Dict[str, float], float, Dict[str, float], float]:
        """
        Evaluate model on a dataset, including augmented (rotated) evaluation.
        
        Args:
            loader: Evaluation data loader
            compute_mse: Whether to compute MSE
            compute_mae: Whether to compute MAE
            include_augmented: Whether to evaluate on augmented data
        
        Returns:
            Tuple of (loss, metrics_dict, aug_loss, aug_metrics_dict, consistency_error)
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_X = []
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    X, y = batch
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")
                
                X = X.to(self.device)
                y = y.to(self.device)
                
                y_pred = self.model(X)
                
                loss = self.criterion(y_pred, y)
                batch_size = X.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                all_predictions.append(y_pred.cpu())
                all_targets.append(y.cpu())
                all_X.append(X.cpu())
        
        avg_loss = total_loss / total_samples if total_samples > 0 else -1.0
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_X = torch.cat(all_X, dim=0)
        
        metrics = {}
        if compute_mse:
            mse = torch.mean((all_predictions - all_targets) ** 2).item()
            metrics['mse'] = mse
        if compute_mae:
            mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
            metrics['mae'] = mae
        
        metrics['relative_norm_error'] = relative_norm_error(all_predictions, all_targets)
        
        aug_loss = -1.0
        aug_metrics = {'mse': -1.0, 'mae': -1.0, 'relative_norm_error': -1.0}
        consistency_error = -1.0
        
        if include_augmented:
            aug_loss, aug_metrics, consistency_error = self._evaluate_augmented(
                all_X, all_targets, compute_mse=compute_mse, compute_mae=compute_mae
            )
        
        return avg_loss, metrics, aug_loss, aug_metrics, consistency_error
    


    
    def _evaluate_augmented(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        compute_mse: bool = True,
        compute_mae: bool = True,
    ) -> Tuple[float, Dict[str, float], float]:
        """
        Evaluate on augmented (rotated) data using proper group representations.
        
        Args:
            X: Input data tensor
            y: Target data tensor
            n_augmentations: Number of random group elements to sample
            compute_mse: Whether to compute MSE
            compute_mae: Whether to compute MAE
        
        Returns:
            Tuple of (loss, metrics_dict, consistency_error)
        """
        self.model.eval()
        
        from emlp.reps import Scalar, Vector
        from emlp.groups import O, Lorentz
        
        group_name = self.config.get('model', {}).get('group', 'O(5)')
        group_key = group_name.strip().lower()
        if group_key in ['lorentz', 'o(1,3)', 'o13']:
            group = Lorentz()
        else:
            group = O(5)
        
        in_features = self.config.get('model', {}).get('in_features', 2)
        in_rep_str = self.config.get('model', {}).get('in_rep', 'V')
        out_rep_str = self.config.get('model', {}).get('out_rep', 'S')
        model_type = self.config.get('model', {}).get('type', '')
        interleaved_inputs = model_type.startswith('filtered_')

        def interleaved_to_grouped(x_flat: torch.Tensor, n_channels: int, rep_dim: int) -> torch.Tensor:
            return x_flat.view(-1, n_channels, rep_dim).transpose(1, 2).reshape(-1, n_channels * rep_dim)

        def grouped_to_interleaved(x_flat: torch.Tensor, n_channels: int, rep_dim: int) -> torch.Tensor:
            return x_flat.view(-1, rep_dim, n_channels).transpose(1, 2).reshape(-1, n_channels * rep_dim)
        
        if in_rep_str == 'V':
            in_rep = in_features * Vector
        else:
            in_rep = in_features * Scalar
        
        if out_rep_str == 'S':
            out_rep = Scalar
        else:
            out_rep = Vector
        
        aug_total_loss = 0.0
        all_aug_predictions = []
        all_aug_targets = []
        consistency_errors = []
        
        rep_dim = X.shape[1] // in_features if in_features > 0 else None
        if interleaved_inputs and (rep_dim is None or X.shape[1] % in_features != 0):
            raise ValueError("Interleaved inputs require X.shape[1] divisible by in_features.")

        with torch.no_grad():
            for i in range(len(X)):
                x_i = X[i:i+1]
                y_i = y[i:i+1]
                
                g = group.sample()
                
                rep_in_concrete = in_rep(group)
                rep_out_concrete = out_rep(group)
                
                rho_in_g = np.array(rep_in_concrete.rho_dense(g))
                rho_out_g = np.array(rep_out_concrete.rho_dense(g))
                
                x_device = x_i.to(self.device)
                y_device = y_i.to(self.device)
                
                rho_in_tensor = torch.tensor(rho_in_g, dtype=x_device.dtype, device=self.device)
                rho_out_tensor = torch.tensor(rho_out_g, dtype=x_device.dtype, device=self.device)
                
                y_pred = self.model(x_device)
                
                if interleaved_inputs:
                    x_grouped = interleaved_to_grouped(x_device, in_features, rep_dim)
                else:
                    x_grouped = x_device
                if len(x_grouped.shape) == 2:
                    gx_grouped = torch.matmul(x_grouped, rho_in_tensor.T)
                else:
                    gx_grouped = torch.matmul(rho_in_tensor, x_grouped.T).T
                gx = grouped_to_interleaved(gx_grouped, in_features, rep_dim) if interleaved_inputs else gx_grouped
                
                Yg_pred = self.model(gx)
                
                if out_rep_str == 'S':
                    g_y_pred = rho_out_g[0, 0] * y_pred
                    g_y = rho_out_g[0, 0] * y_device
                else:
                    if interleaved_inputs:
                        out_channels = y_device.shape[1] // rep_dim
                        y_pred_grouped = interleaved_to_grouped(y_pred, out_channels, rep_dim)
                        y_grouped = interleaved_to_grouped(y_device, out_channels, rep_dim)
                    else:
                        y_pred_grouped = y_pred
                        y_grouped = y_device
                    g_y_pred_grouped = torch.matmul(y_pred_grouped, rho_out_tensor.T)
                    g_y_grouped = torch.matmul(y_grouped, rho_out_tensor.T)
                    g_y_pred = grouped_to_interleaved(g_y_pred_grouped, out_channels, rep_dim) if interleaved_inputs else g_y_pred_grouped
                    g_y = grouped_to_interleaved(g_y_grouped, out_channels, rep_dim) if interleaved_inputs else g_y_grouped
                
                consistency_error = torch.norm(g_y_pred - Yg_pred, dim=-1).mean().item()
                denom = torch.norm(g_y_pred, dim=-1) + torch.norm(Yg_pred, dim=-1) + 1e-6
                consistency_error /= denom.mean().item()
                consistency_errors.append(consistency_error)
                
                aug_loss = self.criterion(Yg_pred, g_y)
                aug_total_loss += aug_loss.item()
                
                all_aug_predictions.append(Yg_pred.cpu())
                all_aug_targets.append(g_y.cpu())
        
        all_aug_predictions = torch.cat(all_aug_predictions, dim=0)
        all_aug_targets = torch.cat(all_aug_targets, dim=0)
        
        avg_aug_loss = aug_total_loss / len(X)
        
        aug_metrics = {}
        if compute_mse:
            mse = torch.mean((all_aug_predictions - all_aug_targets) ** 2).item()
            aug_metrics['mse'] = mse
        if compute_mae:
            mae = torch.mean(torch.abs(all_aug_predictions - all_aug_targets)).item()
            aug_metrics['mae'] = mae
        
        aug_metrics['relative_norm_error'] = relative_norm_error(all_aug_predictions, all_aug_targets)
        
        avg_consistency_error = np.mean(consistency_errors) 

        
        return avg_aug_loss, aug_metrics, avg_consistency_error
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        log_interval: int = 10,
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            log_interval: How often to log metrics
        """
        scheduler_type = self.scheduler_config.get('type', 'StepLR')
        if scheduler_type == 'StepLR':
            step_size = self.scheduler_config.get('step_size', 500)
            gamma = self.scheduler_config.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'CosineAnnealingLR':
            T_max = self.scheduler_config.get('T_max', epochs)
            eta_min = self.scheduler_config.get('eta_min', 1e-6)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch, log_interval=log_interval)
            logger.info(f"Epoch {epoch} - Train loss: {train_loss:.6f}")
            
            if val_loader is not None:
                training_cfg = self.config.get('training', {})
                include_augmented_cfg = training_cfg.get('include_augmented', True)
                aug_eval_every = training_cfg.get('aug_eval_every', 50)
                if include_augmented_cfg and (epoch % max(1, int(aug_eval_every)) == 0):
                    aug = True
                else:
                    aug = False
                val_loss, val_metrics, aug_val_loss, aug_val_metrics, val_consistency = self.evaluate(
                    val_loader,
                    include_augmented=aug
                )
                logger.info(f"Epoch {epoch} - Val loss: {val_loss:.6f}, metrics: {val_metrics}")
                logger.info(f"Epoch {epoch} - Aug Val loss: {aug_val_loss:.6f}, metrics: {aug_val_metrics}, consistency: {val_consistency:.6f}")
                
                val_relative_error = val_metrics.get('relative_norm_error', float('inf'))
                if val_relative_error < self.best_val_metric:
                    self.best_val_metric = val_relative_error
                    self._save_checkpoint(epoch, val_relative_error)
                    logger.info(f"Best model updated at epoch {epoch} with relative error: {val_relative_error:.6f}")
                
                if self.logger is not None:
                    log_dict = {
                        'train/loss': train_loss,
                        'val/loss': val_loss,
                        'val/mse': val_metrics.get('mse', 0),
                        'val/mae': val_metrics.get('mae', 0),
                        'val/relative_norm_error': val_metrics.get('relative_norm_error', 0),
                        'val/aug_loss': aug_val_loss,
                        'val/aug_mse': aug_val_metrics.get('mse', 0),
                        'val/aug_mae': aug_val_metrics.get('mae', 0),
                        'val/aug_relative_norm_error': aug_val_metrics.get('relative_norm_error', 0),
                        'val/consistency_error': val_consistency,
                        'epoch': epoch,
                    }
                    if len(self.optimizer.param_groups) > 1:
                        for i, pg in enumerate(self.optimizer.param_groups):
                            log_dict[f'train/lr_group_{i}'] = pg['lr']
                    else:
                        log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                    
                    self.logger(log_dict)
            else:
                if self.logger is not None:
                    log_dict = {
                        'train/loss': train_loss,
                        'epoch': epoch,
                    }
                    if len(self.optimizer.param_groups) > 1:
                        for i, pg in enumerate(self.optimizer.param_groups):
                            log_dict[f'train/lr_group_{i}'] = pg['lr']
                    else:
                        log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                    
                    self.logger(log_dict)
            
            self.scheduler.step()
        
        logger.info("Training complete")
    
    def _save_checkpoint(self, epoch: int, metric: float):
        """Save best model checkpoint only."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
        }
        
        torch.save(checkpoint, self.best_model_path)
    
    def load_best_model(self):
        """Load the best model checkpoint."""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            metric = checkpoint.get('metric', checkpoint.get('loss', 0.0))
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with relative error {metric:.6f}")
            return checkpoint['epoch'], metric
        else:
            logger.warning(f"Best model checkpoint not found at {self.best_model_path}")
            return None, None


__all__ = ['MiscTrainer']

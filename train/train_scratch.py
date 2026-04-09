from typing import Dict, Optional, Tuple, List
import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm
from utils.consistency import test_on_augmented

class ScratchTrainer:
    def __init__(self, model: nn.Module, cfg:Dict, device:Optional[torch.device] = None):

        self.model = model
        self.cfg = cfg
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(self.device)

        train_cfg = cfg.get('train', {})
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = self._create_optimizer(train_cfg)
        self.scheduler = None
        
        self.grad_clip_cfg = train_cfg.get('grad_clip', {})

        self.topk = int(cfg.get('data', {}).get('topk_accuracy', 5))

        ckpt_dir = cfg.get('logging', {}).get('checkpoint_dir', './weights_fcnn')
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_acc = -1.0

    def _create_optimizer(self, train_cfg: Dict) -> torch.optim.Optimizer:
        """
        Create optimizer from config.
        
        Supports:
        - AdamW (default)
        - Adam
        - SGD
        - RMSprop
        - Adagrad
        """
        opt_cfg = train_cfg.get('optimizer', {})
        
        if isinstance(opt_cfg, dict) and 'type' in opt_cfg:
            opt_type = opt_cfg.get('type', 'AdamW')
            lr = float(opt_cfg.get('lr', train_cfg.get('lr', 1e-3)))
            weight_decay = float(opt_cfg.get('weight_decay', train_cfg.get('weight_decay', 0.0)))
        else:
            opt_type = train_cfg.get('optimizer_type', 'AdamW')
            lr = float(train_cfg.get('lr', 1e-3))
            weight_decay = float(train_cfg.get('weight_decay', 0.0))
            opt_cfg = train_cfg
        
        params = self.model.parameters()
        
        if opt_type == 'AdamW':
            betas = opt_cfg.get('betas', [0.9, 0.999])
            eps = float(opt_cfg.get('eps', 1e-8))
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        
        elif opt_type == 'Adam':
            betas = opt_cfg.get('betas', [0.9, 0.999])
            eps = float(opt_cfg.get('eps', 1e-8))
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        
        elif opt_type == 'SGD':
            momentum = float(opt_cfg.get('momentum', 0.9))
            nesterov = bool(opt_cfg.get('nesterov', False))
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, 
                                   momentum=momentum, nesterov=nesterov)
        
        elif opt_type == 'RMSprop':
            alpha = float(opt_cfg.get('alpha', 0.99))
            eps = float(opt_cfg.get('eps', 1e-8))
            momentum = float(opt_cfg.get('momentum', 0.0))
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay,
                                       alpha=alpha, eps=eps, momentum=momentum)
        
        elif opt_type == 'Adagrad':
            lr_decay = float(opt_cfg.get('lr_decay', 0.0))
            eps = float(opt_cfg.get('eps', 1e-10))
            return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay,
                                       lr_decay=lr_decay, eps=eps)
        
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}. "
                           f"Supported: AdamW, Adam, SGD, RMSprop, Adagrad")

    def train_one_epoch(self, loader: DataLoader, epoch:int, log_interval:int =50, logger=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        topk_correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            
            if self.grad_clip_cfg.get('enabled', False):
                clip_type = self.grad_clip_cfg.get('type', 'norm')
                if clip_type == 'norm':
                    max_norm = float(self.grad_clip_cfg.get('max_norm', 1.0))
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                elif clip_type == 'value':
                    clip_value = float(self.grad_clip_cfg.get('clip_value', 1.0))
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value)
            
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()

            _, topk_preds = torch.topk(outputs, k=self.topk, dim=1)
            topk_correct += topk_preds.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)

            iter_acc = (preds == targets).float().mean().item()
            iter_topk_acc = topk_preds.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{iter_acc:.4f}", 
                topk_acc=f"{iter_topk_acc:.4f}"
            )

            if logger is not None and (batch_idx + 1) % log_interval == 0:
                log_dict = {
                    'train/iter_loss': loss.item(),
                    'train/iter_acc': (preds == targets).float().mean().item(),
                    'train/iter_topk_acc': iter_topk_acc,
                    'train/iter': batch_idx + 1,
                    'train/epoch': epoch,
                    'train/lr' : self.optimizer.param_groups[0]['lr']
                }
                logger(log_dict)

        epoch_acc = correct / max(total,1)
        epoch_topk_acc = topk_correct / max(total,1)
        return epoch_loss, epoch_acc, epoch_topk_acc
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, include_transformation: bool = True):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        topk_correct = 0
        total = 0

        cm = None 
        num_classes = None
        all_true = []
        all_pred = []

        rot_running_loss = 0.0
        rot_correct = 0
        rot_topk_correct = 0
        rot_total = 0
        rot_cm = None
        rot_all_true = []
        rot_all_pred = []
        rot_consistency_sum = 0.0
        rot_consistency_count = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()

            _, topk_preds = torch.topk(outputs, k=self.topk, dim=1)
            topk_correct += sum([targets[i] in topk_preds[i] for i in range(targets.size(0))])
            
            total += targets.size(0)

            if num_classes is None:
                num_classes = outputs.size(1)
                cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1
            all_true.extend(targets.view(-1).detach().cpu().tolist())
            all_pred.extend(preds.view(-1).detach().cpu().tolist())
            
            if include_transformation and images.dim() == 4:
                eval_rot = self.cfg.get('data', {}).get('eval_rot', 15.0)
                group_type = self.cfg.get('model', {}).get('group_type', 'rotation')
                
                rotation_config = eval_rot if group_type in ['rotation', 'roto_reflection'] else None
                
                aug_results = test_on_augmented(
                    batch_inputs=images,
                    model=self.model,
                    rotation_config=rotation_config,
                    task_type='classification',
                    batch_targets=targets,
                    criterion=self.criterion,
                    device=self.device,
                    return_consistency=True,
                    group_type=group_type
                )
                
                rot_outputs = aug_results['augmented_logits']
                rot_loss = aug_results.get('augmented_loss', self.criterion(rot_outputs, targets))
                rot_consistency = aug_results.get('consistency', 0.0)
                
                rot_running_loss += rot_loss.item() * images.size(0)
                _, rot_preds = torch.max(rot_outputs, dim=1)
                rot_correct += (rot_preds == targets).sum().item()
                
                _, rot_topk_preds = torch.topk(rot_outputs, k=self.topk, dim=1)
                rot_topk_correct += sum([targets[i] in rot_topk_preds[i] for i in range(targets.size(0))])
                
                rot_total += targets.size(0)
                
                rot_consistency_sum += rot_consistency.item() * images.size(0)
                rot_consistency_count += images.size(0)
                
                if num_classes is None:
                    num_classes = rot_outputs.size(1)
                if rot_cm is None:
                    rot_cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)
                for t, p in zip(targets.view(-1), rot_preds.view(-1)):
                    rot_cm[t.long(), p.long()] += 1
                rot_all_true.extend(targets.view(-1).detach().cpu().tolist())
                rot_all_pred.extend(rot_preds.view(-1).detach().cpu().tolist())

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        topk_acc = topk_correct / max(total, 1)
        
        rot_loss = rot_running_loss / max(rot_total, 1) if include_transformation and rot_total > 0 else 0.0
        rot_acc = rot_correct / max(rot_total, 1) if include_transformation and rot_total > 0 else 0.0
        rot_topk_acc = rot_topk_correct / max(rot_total, 1) if include_transformation and rot_total > 0 else 0.0
        rot_consistency = rot_consistency_sum / max(rot_consistency_count, 1) if rot_consistency_count > 0 else 0.0
        
        if cm is not None:
            print("Confusion matrix (rows=true, cols=pred):")
            print(cm.to('cpu'))
        
        if include_transformation and rot_total > 0:
            print(f"Rotation evaluation - Loss: {rot_loss:.4f}, Accuracy: {rot_acc:.4f}, Consistency: {rot_consistency:.4f}")
        
        return (
            epoch_loss,
            epoch_acc,
            topk_acc,
            (cm.to('cpu') if cm is not None else None),
            all_true,
            all_pred,
            rot_loss,
            rot_acc,
            rot_topk_acc,
            (rot_cm.to('cpu') if rot_cm is not None else None),
            rot_all_true if len(rot_all_true) > 0 else None,
            rot_all_pred if len(rot_all_pred) > 0 else None,
            rot_consistency,
        )
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 1, logger=None, log_interval: int = 50):
        scheduler_cfg = self.cfg.get('train', {}).get('scheduler', {})
        if scheduler_cfg:
            scheduler_cfg = scheduler_cfg.copy()
            scheduler_type = scheduler_cfg.pop('type', 'LinearLR')
            
            if scheduler_type == 'LinearLR' and 'total_iters' not in scheduler_cfg:
                scheduler_cfg['total_iters'] = epochs
            elif scheduler_type == 'CosineAnnealingLR' and 'T_max' not in scheduler_cfg:
                scheduler_cfg['T_max'] = epochs
            elif scheduler_type == 'OneCycleLR' and 'total_steps' not in scheduler_cfg:
                scheduler_cfg['total_steps'] = epochs * len(train_loader)
                scheduler_cfg['epochs'] = epochs
                scheduler_cfg['steps_per_epoch'] = len(train_loader)
            
            try:
                scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
                self.scheduler = scheduler_class(self.optimizer, **scheduler_cfg)
                self.scheduler_type = scheduler_type
                print(f"Using {scheduler_type} scheduler with config: {scheduler_cfg}")
            except AttributeError:
                print(f"Warning: Scheduler type '{scheduler_type}' not found. Proceeding without scheduler.")
                self.scheduler = None
                self.scheduler_type = None
        else:
            self.scheduler = None
            self.scheduler_type = None
        
        history = []
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_topk_acc = self.train_one_epoch(train_loader, epoch, log_interval=log_interval, logger=logger)
            val_loss, val_acc, val_topk_acc, val_cm, val_true, val_pred, val_rot_loss, val_rot_acc, val_rot_topk_acc, val_rot_cm, val_rot_true, val_rot_pred, val_rot_consistency = (None, None, None, None, None, None, None, None, None, None, None, None, None)
            if val_loader is not None:
                val_loss, val_acc, val_topk_acc, val_cm, val_true, val_pred, val_rot_loss, val_rot_acc, val_rot_topk_acc, val_rot_cm, val_rot_true, val_rot_pred, val_rot_consistency = self.evaluate(val_loader)

            if val_loader is not None:
                print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} train_topk_acc: {train_topk_acc:.4f} | "
                      f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_topk_acc: {val_topk_acc:.4f} | "
                      f"val_rot_loss: {val_rot_loss:.4f} val_rot_acc: {val_rot_acc:.4f} val_rot_topk_acc: {val_rot_topk_acc:.4f} | "
                      f"rot_consistency: {val_rot_consistency:.4f}")
            else:
                print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} train_topk_acc: {train_topk_acc:.4f}")

            if logger is not None:
                payload = {
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/acc': train_acc,
                    'train/topk_acc': train_topk_acc,
                }
 
                payload['train/lr'] = self.optimizer.param_groups[0]['lr']
                    
                if val_loader is not None:
                    payload.update({
                        'val/loss': val_loss,
                        'val/acc': val_acc,
                        'val/topk_acc': val_topk_acc,
                        'val/rot_loss': val_rot_loss,
                        'val/rot_acc': val_rot_acc,
                        'val/rot_topk_acc': val_rot_topk_acc,
                        'val/combined_acc': (val_acc * val_rot_acc)**0.5,
                        'val/combined_topk_acc': (val_topk_acc * val_rot_topk_acc)**0.5,
                        'val/consistency': val_rot_consistency,
                    })
                logger(payload)
                if val_cm is not None and val_true is not None and val_pred is not None:
                    try:
                        import wandb  # type: ignore
                        class_names = [str(i) for i in range(10)]
                        logger({'val/confusion_matrix': wandb.plot.confusion_matrix(y_true=val_true, preds=val_pred, class_names=class_names)})
                    except Exception:
                        pass
                if val_rot_cm is not None and val_rot_true is not None and val_rot_pred is not None:
                    try:
                        import wandb  # type: ignore
                        class_names = [str(i) for i in range(10)]
                        logger({'rot_val/rot_confusion_matrix': wandb.plot.confusion_matrix(y_true=val_rot_true, preds=val_rot_pred, class_names=class_names)})
                    except Exception:
                        pass

            history.append((train_loss, train_acc, val_loss, val_acc))
            if val_loader is not None:
                if val_acc is not None and val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self._save_checkpoint('best.pt', epoch, train_loss, train_acc, val_loss, val_acc)
            self._save_checkpoint('last.pt', epoch, train_loss, train_acc, val_loss, val_acc)
            
            if self.scheduler is not None:
                if self.scheduler_type == 'ReduceLROnPlateau':
                    metric = val_loss if val_loss is not None else train_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

        return history
    
    def _save_checkpoint(self, name: str, epoch: int, train_loss: float, train_acc: float, val_loss: Optional[float], val_acc: Optional[float]):
        path = os.path.join(self.ckpt_dir, name)
        was_training = self.model.training
        self.model.eval()
        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'cfg': self.cfg,
        }
        torch.save(payload, path)
        if was_training:
            self.model.train()
from typing import Dict, Optional
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import random
from tqdm import tqdm
import pdb
from utils.consistency import test_on_augmented, get_eq_error

class Trainer:
    def __init__(self, model: nn.Module, cfg: Dict, device: Optional[torch.device] = None):
        self.model = model
        self.cfg = cfg
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(self.device)

        train_cfg = cfg.get('train', {})
        lr = float(train_cfg['lr'])
        weight_decay = float(train_cfg['weight_decay'])
        backbone_lr = train_cfg.get('backbone_lr', None)

        self.criterion = nn.CrossEntropyLoss()
        
        param_groups = self._create_param_groups(model, lr, backbone_lr, weight_decay)
        
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = None
        self.topk = cfg.get('data', {}).get('topk_accuracy', 5)

        self.use_invariant_loss = train_cfg.get('use_invariant_loss', False)
        self.inv_loss_weight = float(train_cfg.get('inv_loss_weight', 0.0))
        self.inv_loss_freq = int(train_cfg.get('inv_loss_freq', 1))

        ckpt_dir = cfg['logging']['checkpoint_dir']
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_acc = -1.0

    def _create_param_groups(self, model: nn.Module, lr: float, backbone_lr: Optional[float], 
                            weight_decay: float):
        """
        Create parameter groups with differential learning rates for backbone and head.
        
        Args:
            model: The model to create parameter groups for
            lr: Learning rate for head parameters
            backbone_lr: Learning rate for backbone parameters (None = use same lr for all)
            weight_decay: Weight decay for all parameters
            
        Returns:
            List of parameter group dictionaries for optimizer
        """
        if backbone_lr is None:
            print(f"Using standard training: lr={lr} for all parameters")
            return [{'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay}]
        
        backbone_params = []
        head_params = []
        backbone_names = []
        head_names = []
        
        for name, param in model.named_parameters():
            if 'canon_net' in name.lower():
                head_params.append(param)
                head_names.append(name)
            elif any(keyword in name.lower() for keyword in ['classifier', 'decode_head', 'head']):
                if 'attention' not in name.lower() and 'attn' not in name.lower():
                    head_params.append(param)
                    head_names.append(name)
                else:
                    backbone_params.append(param)
                    backbone_names.append(name)
            else:
                backbone_params.append(param)
                backbone_names.append(name)
        
        backbone_count = sum(p.numel() for p in backbone_params)
        head_count = sum(p.numel() for p in head_params)
        canon_net_params_count = sum(p.numel() for name, p in zip(head_names, head_params) if 'canon_net' in name.lower())
        
        if backbone_lr == 0.0:
            for param in backbone_params:
                param.requires_grad = False
            
            trainable_count = sum(p.numel() for p in head_params if p.requires_grad)
            print(f"\n{'='*80}")
            print(f"LINEAR PROBING MODE: Backbone frozen, only training head")
            print(f"{'='*80}")
            print(f"Backbone parameters: {backbone_count:,} (frozen)")
            print(f"Head parameters: {head_count:,} (trainable)")
            if canon_net_params_count > 0:
                print(f"  → Canon_net parameters: {canon_net_params_count:,} (included in head)")
            print(f"Total trainable: {trainable_count:,}")
            print(f"Head learning rate: {lr}")
            print(f"{'='*80}\n")
            
            return [{'params': head_params, 'lr': lr, 'weight_decay': weight_decay}]
        else:
            print(f"\n{'='*80}")
            print(f"DIFFERENTIAL LEARNING RATES MODE")
            print(f"{'='*80}")
            print(f"Backbone parameters: {backbone_count:,} (lr={backbone_lr})")
            print(f"Head parameters: {head_count:,} (lr={lr})")
            if canon_net_params_count > 0:
                print(f"  → Canon_net parameters: {canon_net_params_count:,} (included in head)")
            print(f"Total trainable: {backbone_count + head_count:,}")
            print(f"LR ratio (head/backbone): {lr/backbone_lr:.2f}x")
            print(f"{'='*80}\n")
            
            return [
                {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
                {'params': head_params, 'lr': lr, 'weight_decay': weight_decay}
            ]

    def train_one_epoch(self, loader: DataLoader, epoch: int, log_interval: int = 50, logger=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        topk_correct = 0
        total = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            images_original = images.clone()
            
            if 'test' in self.model.__class__.__name__.lower() and images.dim() == 4:
                images = images.flatten(start_dim=1)

            outputs = self.model(images)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            loss = self.criterion(logits, targets)
            if hasattr(self.model, 'prior_loss') and self.model.prior_loss is not None:
                loss += self.model.prior_loss
            
            if self.use_invariant_loss and (batch_idx % self.inv_loss_freq == 0):
                eval_rot = self.cfg.get('data', {}).get('eval_rot', 90.0)
                group_type = self.cfg.get('model', {}).get('group_type', 'rotation')
                rotation_config = eval_rot if group_type in ['rotation', 'roto_reflection'] else None
                
                if 'test' in self.model.__class__.__name__.lower():
                    model_wrapper = lambda x: self.model(x.flatten(start_dim=1) if x.dim() == 4 else x)
                else:
                    model_wrapper = self.model
                
                inv_error = get_eq_error(
                    batch_inputs=images_original,
                    model=model_wrapper,
                    rotation_config=rotation_config,
                    task_type='classification',
                    device=self.device,
                    original_logits=logits,
                    group_type=group_type
                )
                
                loss = loss + self.inv_loss_weight * inv_error
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == targets).sum().item()
            
            _, topk_preds = torch.topk(logits, k=self.topk, dim=1)
            topk_correct += sum([targets[i] in topk_preds[i] for i in range(targets.size(0))])
            
            total += targets.size(0)

            iter_acc = (preds == targets).float().mean().item()
            iter_topk_acc = sum([targets[i] in topk_preds[i] for i in range(targets.size(0))]) / targets.size(0)
            
            if len(self.optimizer.param_groups) > 1:
                backbone_lr = self.optimizer.param_groups[0]['lr']
                head_lr = self.optimizer.param_groups[1]['lr']
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{iter_acc:.4f}", 
                               topk_acc=f"{iter_topk_acc:.4f}", 
                               backbone_lr=f"{backbone_lr:.2e}", head_lr=f"{head_lr:.2e}")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{iter_acc:.4f}", topk_acc=f"{iter_topk_acc:.4f}")

            if logger is not None and (batch_idx + 1) % log_interval == 0:
                log_dict = {
                    'train/iter_loss': loss.item(),
                    'train/iter_acc': (preds == targets).float().mean().item(),
                    'train/iter_topk_acc': iter_topk_acc,
                    'train/iter': batch_idx + 1,
                    'train/epoch': epoch,
                }
                if len(self.optimizer.param_groups) > 1:
                    log_dict['train/backbone_lr'] = self.optimizer.param_groups[0]['lr']
                    log_dict['train/head_lr'] = self.optimizer.param_groups[1]['lr']
                else:
                    log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                logger(log_dict)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        epoch_topk_acc = topk_correct / max(total, 1)
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
            
            if 'test' in self.model.__class__.__name__.lower() and images.dim() == 4:
                images_flat = images.flatten(start_dim=1)
            else:
                images_flat = images

            outputs = self.model(images_flat)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            loss = self.criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == targets).sum().item()
            
            _, topk_preds = torch.topk(logits, k=self.topk, dim=1)
            topk_correct += sum([targets[i] in topk_preds[i] for i in range(targets.size(0))])
            
            total += targets.size(0)

            if num_classes is None:
                num_classes = logits.size(1)
                cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1
            all_true.extend(targets.view(-1).detach().cpu().tolist())
            all_pred.extend(preds.view(-1).detach().cpu().tolist())
            
            if include_transformation and images.dim() == 4:
                eval_rot = self.cfg.get('data', {}).get('eval_rot', 90.0)
                group_type = self.cfg.get('model', {}).get('group_type', 'rotation')
                
                rotation_config = eval_rot if group_type in ['rotation', 'roto_reflection'] else None
                
                if 'test' in self.model.__class__.__name__.lower():
                    model_wrapper = lambda x: self.model(x.flatten(start_dim=1) if x.dim() == 4 else x)
                else:
                    model_wrapper = self.model
                
                aug_results = test_on_augmented(
                    batch_inputs=images,
                    model=model_wrapper,
                    rotation_config=rotation_config,
                    task_type='classification',
                    batch_targets=targets,
                    criterion=self.criterion,
                    device=self.device,
                    return_consistency=True,
                    group_type=group_type
                )
                
                rot_logits = aug_results['augmented_logits']
                rot_loss = aug_results.get('augmented_loss', self.criterion(rot_logits, targets))
                rot_consistency = aug_results.get('consistency', 0.0)
                
                rot_running_loss += rot_loss.item() * images.size(0)
                _, rot_preds = torch.max(rot_logits, dim=1)
                rot_correct += (rot_preds == targets).sum().item()
                
                _, rot_topk_preds = torch.topk(rot_logits, k=self.topk, dim=1)
                rot_topk_correct += sum([targets[i] in rot_topk_preds[i] for i in range(targets.size(0))])
                
                rot_total += targets.size(0)
                
                rot_consistency_sum += rot_consistency.item() * images.size(0)
                rot_consistency_count += images.size(0)
                
                if num_classes is None:
                    num_classes = rot_logits.size(1)
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
            scheduler_type = scheduler_cfg.pop('type', 'LinearLR')
            if scheduler_type == 'LinearLR' and 'total_iters' not in scheduler_cfg:
                scheduler_cfg['total_iters'] = epochs
            
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
            self.scheduler = scheduler_class(self.optimizer, **scheduler_cfg)
        else:
            self.scheduler = None
        
        best_val_acc = None
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
                if len(self.optimizer.param_groups) > 1:
                    payload['train/backbone_lr'] = self.optimizer.param_groups[0]['lr']
                    payload['train/head_lr'] = self.optimizer.param_groups[1]['lr']
                else:
                    payload['train/lr'] = self.optimizer.param_groups[0]['lr']
                    
                if val_loader is not None:
                    payload.update({
                        'val/loss': val_loss,
                        'val/acc': val_acc,
                        'val/topk_acc': val_topk_acc,
                        'val/rot_loss': val_rot_loss,
                        'val/rot_acc': val_rot_acc,
                        'val/rot_topk_acc': val_rot_topk_acc,
                        'val/rot_consistency': val_rot_consistency,
                        'val/combined_acc': (val_acc * val_rot_acc)**0.5,
                        'val/combined_topk_acc': (val_topk_acc * val_rot_topk_acc)**0.5,
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



"""
Trajectory prediction metrics following SGAN conventions.

This module implements Average Displacement Error (ADE) and Final Displacement Error (FDE)
for evaluating trajectory prediction models. Adapted from SGAN's losses.py to work with
padded dataloaders.
"""

import torch


def displacement_error(pred_traj, pred_traj_gt, validity_mask=None, mode='average'):
    """
    Calculate Average Displacement Error (ADE).
    
    Computes L2 distance between predicted and ground truth trajectories,
    averaged over all time steps and valid pedestrians.
    
    Args:
        pred_traj: Tensor of shape (batch, max_people, 2, pred_len). Predicted trajectories.
        pred_traj_gt: Tensor of shape (batch, max_people, 2, pred_len). Ground truth trajectories.
        validity_mask: Tensor of shape (batch, max_people). Mask for valid people (1=valid, 0=padding).
        mode: 'sum' returns total error, 'average' returns mean error, 'raw' returns per-person errors.
    
    Returns:
        Scalar tensor with displacement error (or tensor of errors if mode='raw').
    """
    batch_size, max_people, _, pred_len = pred_traj.shape
    
    # Calculate squared differences: (batch, max_people, 2, pred_len)
    diff = pred_traj - pred_traj_gt
    diff_squared = diff ** 2
    
    # Sum over x,y coordinates: (batch, max_people, pred_len)
    l2_squared = diff_squared.sum(dim=2)
    
    # Take sqrt to get L2 distance: (batch, max_people, pred_len)
    l2_dist = torch.sqrt(l2_squared)
    
    # Sum over time steps: (batch, max_people)
    displacement_per_person = l2_dist.sum(dim=2)
    
    # Apply validity mask if provided
    if validity_mask is not None:
        displacement_per_person = displacement_per_person * validity_mask
        num_valid = validity_mask.sum()
    else:
        num_valid = batch_size * max_people
    
    if mode == 'raw':
        return displacement_per_person
    elif mode == 'sum':
        return displacement_per_person.sum()
    elif mode == 'average':
        if num_valid > 0:
            return displacement_per_person.sum() / num_valid
        else:
            return torch.tensor(0.0, device=pred_traj.device)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'sum', 'average', 'raw'.")


def final_displacement_error(pred_pos, pred_pos_gt, validity_mask=None, mode='average'):
    """
    Calculate Final Displacement Error (FDE).
    
    Computes L2 distance between predicted and ground truth positions at the final time step.
    
    Args:
        pred_pos: Tensor of shape (batch, max_people, 2). Predicted final positions.
        pred_pos_gt: Tensor of shape (batch, max_people, 2). Ground truth final positions.
        validity_mask: Tensor of shape (batch, max_people). Mask for valid people (1=valid, 0=padding).
        mode: 'sum' returns total error, 'average' returns mean error, 'raw' returns per-person errors.
    
    Returns:
        Scalar tensor with final displacement error (or tensor of errors if mode='raw').
    """
    batch_size, max_people, _ = pred_pos.shape
    
    # Calculate squared differences: (batch, max_people, 2)
    diff = pred_pos - pred_pos_gt
    diff_squared = diff ** 2
    
    # Sum over x,y coordinates and take sqrt: (batch, max_people)
    l2_dist = torch.sqrt(diff_squared.sum(dim=2))
    
    # Apply validity mask if provided
    if validity_mask is not None:
        l2_dist = l2_dist * validity_mask
        num_valid = validity_mask.sum()
    else:
        num_valid = batch_size * max_people
    
    if mode == 'raw':
        return l2_dist
    elif mode == 'sum':
        return l2_dist.sum()
    elif mode == 'average':
        if num_valid > 0:
            return l2_dist.sum() / num_valid
        else:
            return torch.tensor(0.0, device=pred_pos.device)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'sum', 'average', 'raw'.")


class TrajectoryMetrics:
    """
    Accumulator for trajectory prediction metrics across batches.
    
    Similar to SegmentationMetrics pattern but for ADE and FDE.
    """
    
    def __init__(self):
        """Initialize metric accumulators."""
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_ade = 0.0
        self.total_fde = 0.0
        self.num_samples = 0
    
    def update(self, pred_traj, pred_traj_gt, validity_mask):
        """
        Update metrics with a new batch.
        
        Args:
            pred_traj: Tensor of shape (batch, max_people, 2, pred_len). Predicted trajectories.
            pred_traj_gt: Tensor of shape (batch, max_people, 2, pred_len). Ground truth trajectories.
            validity_mask: Tensor of shape (batch, max_people). Mask for valid people.
        """
        # Calculate ADE for this batch
        ade = displacement_error(pred_traj, pred_traj_gt, validity_mask, mode='sum')
        self.total_ade += ade.item()
        
        # Calculate FDE for this batch (use final time step)
        pred_final = pred_traj[:, :, :, -1]  # (batch, max_people, 2)
        gt_final = pred_traj_gt[:, :, :, -1]  # (batch, max_people, 2)
        fde = final_displacement_error(pred_final, gt_final, validity_mask, mode='sum')
        self.total_fde += fde.item()
        
        # Count number of valid samples
        if validity_mask is not None:
            self.num_samples += validity_mask.sum().item()
        else:
            batch_size, max_people = pred_traj.shape[0], pred_traj.shape[1]
            self.num_samples += batch_size * max_people
    
    def get_ade(self):
        """Get average ADE across all accumulated samples."""
        if self.num_samples > 0:
            return self.total_ade / self.num_samples
        else:
            return 0.0
    
    def get_fde(self):
        """Get average FDE across all accumulated samples."""
        if self.num_samples > 0:
            return self.total_fde / self.num_samples
        else:
            return 0.0
    
    def get_metrics(self):
        """Get all metrics as a dictionary."""
        return {
            'ade': self.get_ade(),
            'fde': self.get_fde(),
            'num_samples': self.num_samples
        }

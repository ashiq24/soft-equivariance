"""
Evaluation metrics for semantic segmentation.

This module provides implementations of standard metrics used in semantic
segmentation tasks, particularly Mean Intersection over Union (mIoU).
"""

import torch
import numpy as np


def calculate_miou(logits, labels, num_classes, ignore_index=255):
    """
    Calculate Mean Intersection over Union (mIoU) for semantic segmentation.
    
    Args:
        logits: Model output logits, shape [B, C, H, W]
        labels: Ground truth labels, shape [B, H, W]
        num_classes: Number of classes (including background)
        ignore_index: Label value to ignore (e.g., 255 for PASCAL VOC boundaries)
        
    Returns:
        miou: Mean IoU across all classes (float)
        iou_per_class: IoU for each class (numpy array of shape [num_classes])
    """
    # Get predictions by taking argmax over channel dimension
    preds = torch.argmax(logits, dim=1)  # Shape: [B, H, W]
    
    # Flatten predictions and labels
    preds = preds.view(-1)  # Shape: [B*H*W]
    labels = labels.view(-1)  # Shape: [B*H*W]
    
    # Create mask for valid pixels (not ignore_index)
    valid_mask = labels != ignore_index
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    
    # Initialize IoU storage
    iou_per_class = np.zeros(num_classes)
    valid_classes = 0
    
    # Calculate IoU for each class
    for cls in range(num_classes):
        # True positives: pixels correctly predicted as class cls
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        
        # Intersection: pixels that are both predicted and labeled as cls
        intersection = (pred_cls & label_cls).sum().item()
        
        # Union: pixels that are either predicted or labeled as cls
        union = (pred_cls | label_cls).sum().item()
        
        if union == 0:
            # Class not present in this batch, skip it
            iou_per_class[cls] = float('nan')
        else:
            iou_per_class[cls] = intersection / union
            valid_classes += 1
    
    # Calculate mean IoU (ignoring classes not present)
    if valid_classes > 0:
        miou = np.nanmean(iou_per_class)
    else:
        miou = 0.0
    
    return miou, iou_per_class


def calculate_pixel_accuracy(logits, labels, ignore_index=255):
    """
    Calculate pixel-wise accuracy for semantic segmentation.
    
    Args:
        logits: Model output logits, shape [B, C, H, W]
        labels: Ground truth labels, shape [B, H, W]
        ignore_index: Label value to ignore
        
    Returns:
        accuracy: Pixel accuracy (float)
    """
    # Get predictions
    preds = torch.argmax(logits, dim=1)  # Shape: [B, H, W]
    
    # Create mask for valid pixels
    valid_mask = labels != ignore_index
    
    # Calculate accuracy only on valid pixels
    correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
    total = valid_mask.sum().item()
    
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    return accuracy


class SegmentationMetrics:
    """
    Accumulator for segmentation metrics across multiple batches.
    """
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
        
    def reset(self):
        """Reset all accumulated metrics."""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.correct_pixels = 0
        self.total_pixels = 0
        
    def update(self, logits, labels):
        """
        Update metrics with a new batch.
        
        Args:
            logits: Model output logits, shape [B, C, H, W]
            labels: Ground truth labels, shape [B, H, W]
        """
        # Get predictions
        preds = torch.argmax(logits, dim=1)  # Shape: [B, H, W]
        
        # Flatten
        preds = preds.view(-1)
        labels = labels.view(-1)
        
        # Create mask for valid pixels 
        valid_mask = labels != self.ignore_index
        preds = preds[valid_mask]
        labels = labels[valid_mask]
        
        # Update pixel accuracy (transfer only scalar)
        self.correct_pixels += (preds == labels).sum().item()
        self.total_pixels += preds.numel()
        
        # Update IoU for each class 
        for cls in range(self.num_classes):
            pred_cls = (preds == cls)
            label_cls = (labels == cls)
            
            self.intersection[cls] += (pred_cls & label_cls).sum().item()
            self.union[cls] += (pred_cls | label_cls).sum().item()
            
    def get_miou(self):
        """
        Calculate mean IoU from accumulated statistics.
        
        Returns:
            miou: Mean IoU (float)
            iou_per_class: IoU for each class (numpy array)
        """
        iou_per_class = np.zeros(self.num_classes)
        valid_classes = 0
        
        for cls in range(self.num_classes):
            if self.union[cls] == 0:
                iou_per_class[cls] = float('nan')
            else:
                iou_per_class[cls] = self.intersection[cls] / self.union[cls]
                valid_classes += 1
        
        if valid_classes > 0:
            miou = np.nanmean(iou_per_class)
        else:
            miou = 0.0
            
        return miou, iou_per_class
    
    def get_pixel_accuracy(self):
        """
        Calculate pixel accuracy from accumulated statistics.
        
        Returns:
            accuracy: Pixel accuracy (float)
        """
        if self.total_pixels == 0:
            return 0.0
        return self.correct_pixels / self.total_pixels








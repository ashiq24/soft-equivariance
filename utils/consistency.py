"""
Unified augmentation testing and consistency metrics for soft equivariance experiments.

This module provides a centralized way to:
1. Apply augmentations (rotation, reflection, roto-reflection) to different data types
2. Calculate consistency metrics between original and augmented predictions
3. Support different task types (classification, segmentation, trajectory prediction)

Supported group types:
- "rotation": Rotation group (SO(2) for 2D images/vectors)
- "reflection": Reflection group (Z_2) - horizontal flip for images, axis flip for vectors
- "roto_reflection": Roto-reflection group - first reflect (with 50% probability), then rotate
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
from typing import Union, List, Tuple, Optional, Dict, Any
from .trajectory_rotation import (
    rotate_trajectory_2d,
    reflect_trajectory_2d,
    rotate_trajectory_3d,
    reflect_trajectory_3d
)

# Valid group types
VALID_GROUP_TYPES = ["rotation", "reflection", "roto_reflection"]


def test_on_augmented(
    batch_inputs: torch.Tensor,
    model: torch.nn.Module,
    rotation_config: Union[float, List[float], None] = None,
    task_type: str = 'classification',
    batch_targets: Optional[torch.Tensor] = None,
    validity_mask: Optional[torch.Tensor] = None,
    criterion: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    return_consistency: bool = True,
    original_outputs: Optional[torch.Tensor] = None,
    original_logits: Optional[torch.Tensor] = None,
    group_type: str = "rotation",
    reflection_axis: str = 'x',
    **kwargs
) -> Dict[str, Any]:
    """
    Unified function to test model on augmented inputs and calculate consistency.
    Supports rotation, reflection, and roto-reflection augmentations.
    
    Args:
        batch_inputs: Input batch (images for classification/segmentation, trajectories for trajectory prediction)
        model: The model to evaluate
        rotation_config: Either a float (range) or list of specific angles for rotation, or None for no rotation
        task_type: One of 'classification', 'segmentation', 'trajectory'
        batch_targets: Optional targets (labels for classification, masks for segmentation, future trajectories for trajectory)
        validity_mask: Optional mask for valid samples (used in trajectory prediction)
        criterion: Optional loss function for computing augmented loss
        device: Device to use (if None, uses input device)
        return_consistency: Whether to calculate and return consistency metrics
        original_outputs: Optional precomputed model outputs on original inputs (avoids recomputation)
        original_logits: Optional precomputed logits on original inputs (avoids recomputation)
        group_type: Type of group action - "rotation", "reflection", or "roto_reflection"
        reflection_axis: Axis for reflection ('x' or 'y' for 2D). Only used for trajectory task.
        **kwargs: Additional task-specific arguments (legacy support for 'reflection' parameter)
    
    Returns:
        Dictionary containing:
            - 'augmented_inputs': The augmented input batch
            - 'augmented_outputs': Model outputs on augmented inputs
            - 'augmented_targets': Augmented targets (if applicable)
            - 'rotation_angles': List of rotation angles used
            - 'reflection_flags': List of booleans indicating which samples were reflected
            - 'augmented_loss': Loss on augmented inputs (if criterion provided)
            - 'consistency': Consistency metric (if return_consistency=True)
            - 'original_outputs': Model outputs on original inputs (if needed for consistency)
    """
    # Handle legacy 'reflection' parameter
    if 'reflection' in kwargs:
        legacy_reflection = kwargs.pop('reflection')
        if legacy_reflection and group_type == "rotation":
            group_type = "roto_reflection"
    
    # Validate group_type
    if group_type not in VALID_GROUP_TYPES:
        raise ValueError(f"Invalid group_type: {group_type}. Must be one of {VALID_GROUP_TYPES}")
    
    if device is None:
        device = batch_inputs.device
    
    batch_size = batch_inputs.size(0)
    
    # Determine rotation angles for each sample (if rotation is enabled)
    rotation_angles = []
    if group_type in ["rotation", "roto_reflection"] and rotation_config is not None:
        for i in range(batch_size):
            if isinstance(rotation_config, (list, tuple)):
                # Pick one angle from the list
                angle = float(random.choice(rotation_config))
            else:
                # Random angle in range [-rotation_config, rotation_config]
                angle = random.uniform(-float(rotation_config), float(rotation_config))
            rotation_angles.append(angle)
    else:
        rotation_angles = [0.0] * batch_size  # No rotation
    
    # Determine reflection flags for each sample (50% probability for reflection/roto_reflection)
    reflection_flags = []
    if group_type in ["reflection", "roto_reflection"]:
        for i in range(batch_size):
            reflection_flags.append(random.random() < 0.5)
    else:
        reflection_flags = [False] * batch_size  # No reflection
    
    # Apply task-specific augmentation
    if task_type == 'classification':
        augmented_inputs, augmented_targets = _augment_classification(
            batch_inputs, batch_targets, rotation_angles, reflection_flags, group_type
        )
    elif task_type == 'segmentation':
        augmented_inputs, augmented_targets = _augment_segmentation(
            batch_inputs, batch_targets, rotation_angles, reflection_flags, group_type
        )
    elif task_type == 'trajectory':
        augmented_inputs, augmented_targets = _augment_trajectory(
            batch_inputs, batch_targets, rotation_angles, reflection_flags, 
            group_type, reflection_axis
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be one of 'classification', 'segmentation', 'trajectory'")
    
    with torch.no_grad():
        # Get outputs on augmented inputs
        if task_type == 'trajectory' and validity_mask is not None:
            augmented_outputs = model(augmented_inputs, validity_mask)
        else:
            augmented_outputs = model(augmented_inputs)
        
        # Extract logits if needed
        if hasattr(augmented_outputs, 'logits'):
            augmented_logits = augmented_outputs.logits
        elif isinstance(augmented_outputs, dict) and 'logits' in augmented_outputs:
            augmented_logits = augmented_outputs['logits']
        else:
            augmented_logits = augmented_outputs
        
        # Get original outputs if needed for consistency
        # Use precomputed values if provided, otherwise compute them
        if return_consistency:
            if original_outputs is None or original_logits is None:
                # Need to compute original outputs
                if task_type == 'trajectory' and validity_mask is not None:
                    original_outputs = model(batch_inputs, validity_mask)
                else:
                    original_outputs = model(batch_inputs)
                
                # Extract logits
                if hasattr(original_outputs, 'logits'):
                    original_logits = original_outputs.logits
                elif isinstance(original_outputs, dict) and 'logits' in original_outputs:
                    original_logits = original_outputs['logits']
                else:
                    original_logits = original_outputs
            # If precomputed values were provided, use them as-is
    
    # Build result dictionary
    result = {
        'augmented_inputs': augmented_inputs,
        'augmented_outputs': augmented_outputs,
        'augmented_logits': augmented_logits,
        'rotation_angles': rotation_angles,
        'reflection_flags': reflection_flags,
        'group_type': group_type,
    }
    
    if augmented_targets is not None:
        result['augmented_targets'] = augmented_targets
    
    # Calculate augmented loss if criterion provided
    if criterion is not None and augmented_targets is not None:
        if task_type == 'trajectory' and validity_mask is not None:
            # Special handling for trajectory loss with validity mask
            augmented_loss = _calculate_trajectory_loss(
                augmented_logits, augmented_targets, validity_mask, criterion
            )
        else:
            augmented_loss = criterion(augmented_logits, augmented_targets)
        result['augmented_loss'] = augmented_loss
    
    # Calculate consistency if requested
    if return_consistency:
        result['original_outputs'] = original_outputs
        result['original_logits'] = original_logits
        
        if task_type == 'classification':
            consistency = calculate_classification_consistency(
                original_logits, augmented_logits
            )
        elif task_type == 'segmentation':
            consistency = calculate_segmentation_consistency(
                original_logits, augmented_logits, rotation_angles, reflection_flags,
                batch_inputs.shape, group_type
            )
        elif task_type == 'trajectory':
            consistency = calculate_trajectory_consistency(
                original_logits, augmented_logits, rotation_angles, reflection_flags,
                validity_mask, group_type, reflection_axis
            )
        
        result['consistency'] = consistency
    
    return result


def _augment_classification(
    images: torch.Tensor,
    labels: Optional[torch.Tensor],
    rotation_angles: List[float],
    reflection_flags: List[bool],
    group_type: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply augmentation to classification inputs based on group type."""
    batch_size = images.size(0)
    augmented_images = torch.zeros_like(images)
    
    for i in range(batch_size):
        img = images[i]
        
        # For roto-reflection: first reflect, then rotate
        # For reflection: only reflect
        if group_type in ["reflection", "roto_reflection"] and reflection_flags[i]:
            img = TF.hflip(img)
        
        # Apply rotation if needed (for rotation and roto_reflection)
        if group_type in ["rotation", "roto_reflection"] and rotation_angles[i] != 0.0:
            img = TF.rotate(
                img,
                rotation_angles[i],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0
            )
        
        augmented_images[i] = img
    
    # Labels don't change for classification
    return augmented_images, labels


def _augment_segmentation(
    images: torch.Tensor,
    masks: Optional[torch.Tensor],
    rotation_angles: List[float],
    reflection_flags: List[bool],
    group_type: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply the requested group augmentation to segmentation images and masks.

    Masks are treated as integer label maps. For 2D masks, a temporary channel
    dimension is added before applying flips/rotations and removed afterward.
    Rotated masks use nearest-neighbor interpolation and fill out-of-bounds
    pixels with ``255`` so they can be ignored by the segmentation loss.
    """
    batch_size = images.size(0)
    augmented_images = torch.zeros_like(images)
    augmented_masks = None
    
    if masks is not None:
        augmented_masks = torch.zeros_like(masks)
    
    for i in range(batch_size):
        img = images[i]
        mask = masks[i] if masks is not None else None
        
        # For roto-reflection: first reflect, then rotate
        # For reflection: only reflect
        if group_type in ["reflection", "roto_reflection"] and reflection_flags[i]:
            img = TF.hflip(img)
            if mask is not None:
                # Add channel dimension for flip if needed
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).float()
                else:
                    mask = mask.float()
                mask = TF.hflip(mask)
                # Remove channel dimension if it was added
                if masks[i].dim() == 2:
                    mask = mask.squeeze(0)
                mask = mask.long()
        
        # Apply rotation if needed (for rotation and roto_reflection)
        if group_type in ["rotation", "roto_reflection"] and rotation_angles[i] != 0.0:
            img = TF.rotate(
                img,
                rotation_angles[i],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0
            )
            
            if mask is not None:
                # Add channel dimension for rotation if needed
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).float()
                else:
                    mask = mask.float()
                
                rotated_mask = TF.rotate(
                    mask,
                    rotation_angles[i],
                    interpolation=TF.InterpolationMode.NEAREST,
                    fill=255  # Use ignore_index for out-of-bounds pixels
                )
                
                # Remove channel dimension if it was added
                if masks[i].dim() == 2:
                    rotated_mask = rotated_mask.squeeze(0)
                
                mask = rotated_mask.long()
        
        augmented_images[i] = img
        if masks is not None:
            augmented_masks[i] = mask
    
    return augmented_images, augmented_masks


def _augment_trajectory(
    trajectories: torch.Tensor,
    future_trajectories: Optional[torch.Tensor],
    rotation_angles: List[float],
    reflection_flags: List[bool],
    group_type: str,
    reflection_axis: str = 'x'
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply augmentation to trajectory data based on group type."""
    batch_size = trajectories.size(0)
    vec_dim = trajectories.size(2)  # 2 for 2D, 3 for 3D
    
    # Select appropriate functions based on vector dimension
    if vec_dim == 2:
        rotate_fn = rotate_trajectory_2d
        reflect_fn = lambda t: reflect_trajectory_2d(t, axis=reflection_axis)
    elif vec_dim == 3:
        rotate_fn = lambda t, angle: rotate_trajectory_3d(t, angle, axis='z')
        reflect_fn = lambda t: reflect_trajectory_3d(t, axis=reflection_axis)
    else:
        raise ValueError(f"Unsupported vector dimension: {vec_dim}. Must be 2 or 3.")
    
    augmented_trajectories = trajectories.clone()
    augmented_future = future_trajectories.clone() if future_trajectories is not None else None
    
    for i in range(batch_size):
        # For roto-reflection: first reflect, then rotate
        # For reflection: only reflect
        if group_type in ["reflection", "roto_reflection"] and reflection_flags[i]:
            augmented_trajectories[i:i+1] = reflect_fn(augmented_trajectories[i:i+1])
            if augmented_future is not None:
                augmented_future[i:i+1] = reflect_fn(augmented_future[i:i+1])
        
        # Apply rotation if needed (for rotation and roto_reflection)
        if group_type in ["rotation", "roto_reflection"] and rotation_angles[i] != 0.0:
            augmented_trajectories[i:i+1] = rotate_fn(
                augmented_trajectories[i:i+1], rotation_angles[i]
            )
            if augmented_future is not None:
                augmented_future[i:i+1] = rotate_fn(
                    augmented_future[i:i+1], rotation_angles[i]
                )
    
    return augmented_trajectories, augmented_future


def _calculate_trajectory_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    validity_mask: torch.Tensor,
    criterion: torch.nn.Module
) -> torch.Tensor:
    """Calculate loss for trajectory prediction with validity mask."""
    # predictions/targets: (batch, max_people, 2, pred_len)
    # validity_mask: (batch, max_people)
    
    loss_per_element = criterion(predictions, targets)  # Should be MSE with reduction='none'
    
    # Expand validity mask to match loss dimensions
    validity_mask_expanded = validity_mask.unsqueeze(2).unsqueeze(3)  # (batch, max_people, 1, 1)
    masked_loss = loss_per_element * validity_mask_expanded
    
    # Average over valid elements
    pred_len = predictions.size(-1)
    loss = masked_loss.sum() / (validity_mask.sum() * 2 * pred_len + 1e-8)
    
    return loss


def calculate_classification_consistency(
    original_logits: torch.Tensor,
    augmented_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Calculate KL divergence consistency for classification.
    
    consistency = KL(M(x), M(aug(x)))
    
    Args:
        original_logits: Logits from original inputs (batch, num_classes)
        augmented_logits: Logits from augmented inputs (batch, num_classes)
        temperature: Temperature for softmax (default: 1.0)
    
    Returns:
        Average KL divergence over the batch
    """
    # Convert logits to probabilities
    original_probs = F.softmax(original_logits / temperature, dim=-1)
    augmented_probs = F.softmax(augmented_logits / temperature, dim=-1)
    
    # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
    # Using log_softmax for numerical stability
    original_log_probs = F.log_softmax(original_logits / temperature, dim=-1)
    augmented_log_probs = F.log_softmax(augmented_logits / temperature, dim=-1)
    
    kl_div = F.kl_div(augmented_log_probs, original_probs, reduction='none', log_target=False)
    
    # Sum over classes and average over batch
    consistency = kl_div.sum(dim=-1).mean()
    
    return consistency


def calculate_segmentation_consistency(
    original_logits: torch.Tensor,
    augmented_logits: torch.Tensor,
    rotation_angles: List[float],
    reflection_flags: List[bool],
    original_shape: torch.Size,
    group_type: str,
    ignore_index: int = 255
) -> torch.Tensor:
    """
    Calculate pixel-wise KL divergence consistency for segmentation.
    
    consistency = KL(aug(M(x)), M(aug(x))) / num_active_pixels
    
    Transforms original predictions to match augmented space before comparison.
    
    Args:
        original_logits: Logits from original inputs (batch, num_classes, H, W)
        augmented_logits: Logits from augmented inputs (batch, num_classes, H, W)
        rotation_angles: List of rotation angles used for each sample
        reflection_flags: List of booleans indicating which samples were reflected
        original_shape: Shape of original input for creating validity mask
        group_type: Type of group action
        ignore_index: Index to ignore in calculations (default: 255)
    
    Returns:
        Average KL divergence over active pixels
    """
    batch_size = original_logits.size(0)
    num_classes = original_logits.size(1)
    height, width = original_logits.shape[-2:]
    
    # Transform original predictions to align with augmented predictions
    transformed_original_logits = torch.zeros_like(original_logits)
    
    # Create validity mask by transforming a tensor of ones
    validity_masks = []
    
    for i in range(batch_size):
        transformed_logits = original_logits[i]  # (num_classes, H, W)
        
        # Apply same transformations as input augmentation (reflect first, then rotate)
        if group_type in ["reflection", "roto_reflection"] and reflection_flags[i]:
            transformed_logits = TF.hflip(transformed_logits)
        
        if group_type in ["rotation", "roto_reflection"] and rotation_angles[i] != 0.0:
            transformed_logits = TF.rotate(
                transformed_logits,
                rotation_angles[i],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0
            )
        
        transformed_original_logits[i] = transformed_logits
        
        # Create validity mask for pixels that remain in frame after transformation
        ones_mask = torch.ones(1, height, width, device=original_logits.device)
        
        if group_type in ["reflection", "roto_reflection"] and reflection_flags[i]:
            ones_mask = TF.hflip(ones_mask)
        
        if group_type in ["rotation", "roto_reflection"] and rotation_angles[i] != 0.0:
            ones_mask = TF.rotate(
                ones_mask,
                rotation_angles[i],
                interpolation=TF.InterpolationMode.NEAREST,
                fill=0.0
            )
        
        validity_masks.append(ones_mask.squeeze(0) > 0.5)
    
    validity_mask = torch.stack(validity_masks, dim=0)  # (batch, H, W)
    
    # Convert logits to probabilities
    transformed_original_probs = F.softmax(transformed_original_logits, dim=1)
    augmented_probs = F.softmax(augmented_logits, dim=1)
    
    # Calculate pixel-wise KL divergence
    # Reshape for KL div calculation: (batch*H*W, num_classes)
    transformed_original_probs_flat = transformed_original_probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
    augmented_log_probs_flat = F.log_softmax(
        augmented_logits.permute(0, 2, 3, 1).reshape(-1, num_classes), dim=-1
    )
    
    kl_div = F.kl_div(
        augmented_log_probs_flat,
        transformed_original_probs_flat,
        reduction='none',
        log_target=False
    ).sum(dim=-1)  # Sum over classes
    
    # Reshape back and apply validity mask
    kl_div = kl_div.view(batch_size, height, width)
    kl_div_masked = kl_div * validity_mask.float()
    
    # Average over valid pixels
    num_valid_pixels = validity_mask.sum()
    consistency = kl_div_masked.sum() / (num_valid_pixels + 1e-8)
    
    return consistency


def calculate_trajectory_consistency(
    original_predictions: torch.Tensor,
    augmented_predictions: torch.Tensor,
    rotation_angles: List[float],
    reflection_flags: List[bool],
    validity_mask: Optional[torch.Tensor] = None,
    group_type: str = "rotation",
    reflection_axis: str = 'x'
) -> torch.Tensor:
    """
    Calculate trajectory consistency metric.
    
    consistency = 1/T * ||aug[M(x)] - M(aug(x))||
    
    Transforms original predictions to match augmented space before comparison.
    
    Args:
        original_predictions: Predictions from original trajectories (batch, max_people, 2, pred_len)
        augmented_predictions: Predictions from augmented trajectories (batch, max_people, 2, pred_len)
        rotation_angles: List of rotation angles used for each sample
        reflection_flags: List of booleans indicating which samples were reflected
        validity_mask: Optional mask for valid people (batch, max_people)
        group_type: Type of group action
        reflection_axis: Axis for reflection ('x' or 'y')
    
    Returns:
        Average trajectory difference
    """
    batch_size = original_predictions.size(0)
    vec_dim = original_predictions.size(2)  # 2 for 2D, 3 for 3D
    
    # Select appropriate functions based on vector dimension
    if vec_dim == 2:
        rotate_fn = rotate_trajectory_2d
        reflect_fn = lambda t: reflect_trajectory_2d(t, axis=reflection_axis)
    elif vec_dim == 3:
        rotate_fn = lambda t, angle: rotate_trajectory_3d(t, angle, axis='z')
        reflect_fn = lambda t: reflect_trajectory_3d(t, axis=reflection_axis)
    else:
        raise ValueError(f"Unsupported vector dimension: {vec_dim}. Must be 2 or 3.")
    
    # Transform original predictions to match augmented space
    transformed_original_predictions = original_predictions.clone()
    
    for i in range(batch_size):
        # Apply same transformations as input augmentation (reflect first, then rotate)
        if group_type in ["reflection", "roto_reflection"] and reflection_flags[i]:
            transformed_original_predictions[i:i+1] = reflect_fn(
                transformed_original_predictions[i:i+1]
            )
        
        if group_type in ["rotation", "roto_reflection"] and rotation_angles[i] != 0.0:
            transformed_original_predictions[i:i+1] = rotate_fn(
                transformed_original_predictions[i:i+1], rotation_angles[i]
            )
    
    # Calculate L2 distance between trajectories
    diff = transformed_original_predictions - augmented_predictions  # (batch, max_people, 2, pred_len)
    l2_dist = torch.norm(diff, p=2, dim=2)  # (batch, max_people, pred_len)
    
    # Average over time steps
    mean_dist = l2_dist.mean(dim=-1)  # (batch, max_people)
    
    # Apply validity mask if provided
    if validity_mask is not None:
        mean_dist = mean_dist * validity_mask.float()
        consistency = mean_dist.sum() / (validity_mask.sum() + 1e-8)
    else:
        consistency = mean_dist.mean()
    
    return consistency



def get_eq_error(batch_inputs: torch.Tensor,
    model: torch.nn.Module,
    rotation_config: Union[float, List[float], None] = None,
    task_type: str = 'classification',
    batch_targets: Optional[torch.Tensor] = None,
    validity_mask: Optional[torch.Tensor] = None,
    criterion: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    return_consistency: bool = True,
    original_outputs: Optional[torch.Tensor] = None,
    original_logits: Optional[torch.Tensor] = None,
    group_type: str = "rotation",
    reflection_axis: str = 'x',):
    """
    Calculate equivariance error for training.
    
    This function samples a single group element (same for entire batch) and computes
    the error between transformed original predictions and augmented predictions.
    
    error = ||transform(M(x)) - M(transform(x))||
    
    Args:
        batch_inputs: Input batch (images for classification/segmentation, trajectories for trajectory prediction)
        model: The model to evaluate
        rotation_config: Either a float (range) or list of specific angles for rotation, or None for no rotation
        task_type: One of 'classification', 'segmentation', 'trajectory'
        batch_targets: Optional targets (not used for error calculation, kept for compatibility)
        validity_mask: Optional mask for valid samples (used in trajectory prediction)
        criterion: Optional loss function (not used, kept for compatibility)
        device: Device to use (if None, uses input device)
        return_consistency: Whether to return consistency metric (kept for compatibility, not used)
        original_outputs: Optional precomputed model outputs on original inputs
        original_logits: Optional precomputed logits on original inputs
        group_type: Type of group action - "rotation", "reflection", or "roto_reflection"
        reflection_axis: Axis for reflection ('x' or 'y' for 2D). Only used for trajectory task.
    
    Returns:
        Equivariance error (scalar tensor with gradients)
    """
    # Validate group_type
    if group_type not in VALID_GROUP_TYPES:
        raise ValueError(f"Invalid group_type: {group_type}. Must be one of {VALID_GROUP_TYPES}")
    
    if device is None:
        device = batch_inputs.device
    
    batch_size = batch_inputs.size(0)
    
    # Sample a single group element (same for entire batch)
    rotation_angle = 0.0
    if group_type in ["rotation", "roto_reflection"] and rotation_config is not None:
        if isinstance(rotation_config, (list, tuple)):
            # Pick one angle from the list
            rotation_angle = float(random.choice(rotation_config))
        else:
            # Random angle in range [-rotation_config, rotation_config]
            rotation_angle = random.uniform(-float(rotation_config), float(rotation_config))
    
    # Sample reflection flag (50% probability for reflection/roto_reflection)
    reflection_flag = False
    if group_type in ["reflection", "roto_reflection"]:
        reflection_flag = random.random() < 0.5
    
    # Apply transformation to batch input
    if task_type == 'classification':
        augmented_inputs, _ = _augment_classification(
            batch_inputs, None, [rotation_angle] * batch_size, [reflection_flag] * batch_size, group_type
        )
    elif task_type == 'segmentation':
        augmented_inputs, _ = _augment_segmentation(
            batch_inputs, batch_targets, [rotation_angle] * batch_size, [reflection_flag] * batch_size, group_type
        )
    elif task_type == 'trajectory':
        augmented_inputs, _ = _augment_trajectory(
            batch_inputs, batch_targets, [rotation_angle] * batch_size, [reflection_flag] * batch_size,
            group_type, reflection_axis
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be one of 'classification', 'segmentation', 'trajectory'")
    
    # Get output of model on transformed input (with gradients for training)
    with torch.no_grad():
        if task_type == 'trajectory' and validity_mask is not None:
            augmented_outputs = model(augmented_inputs, validity_mask)
        else:
            augmented_outputs = model(augmented_inputs)
    
        # Extract logits
        if hasattr(augmented_outputs, 'logits'):
            augmented_logits = augmented_outputs.logits
        elif isinstance(augmented_outputs, dict) and 'logits' in augmented_outputs:
            augmented_logits = augmented_outputs['logits']
        else:
            augmented_logits = augmented_outputs
        
        # detach the output
        augmented_logits = augmented_logits.detach()
    
    # Get original outputs if needed
    if original_logits is None:
        if task_type == 'trajectory' and validity_mask is not None:
            original_outputs = model(batch_inputs, validity_mask)
        else:
            original_outputs = model(batch_inputs)
        
        # Extract logits
        if hasattr(original_outputs, 'logits'):
            original_logits = original_outputs.logits
        elif isinstance(original_outputs, dict) and 'logits' in original_outputs:
            original_logits = original_outputs['logits']
        else:
            original_logits = original_outputs
    
    # Apply transformation on original logits (depending on task type)
    if task_type == 'classification':
        # For classification, logits are invariant (should not change)
        # Error is L2 distance between original and augmented logits
        error = torch.nn.functional.mse_loss(original_logits, augmented_logits)
    
    elif task_type == 'segmentation':
        # Transform original logits to match augmented space
        transformed_original_logits = original_logits.clone()
        
        # Apply same transformations as input augmentation (reflect first, then rotate)
        if group_type in ["reflection", "roto_reflection"] and reflection_flag:
            transformed_original_logits = TF.hflip(transformed_original_logits)
        
        if group_type in ["rotation", "roto_reflection"] and rotation_angle != 0.0:
            transformed_original_logits = TF.rotate(
                transformed_original_logits,
                rotation_angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0
            )
        
        # Create validity mask for pixels that remain in frame after transformation
        height, width = original_logits.shape[-2:]
        ones_mask = torch.ones(batch_size, 1, height, width, device=original_logits.device)
        
        if group_type in ["reflection", "roto_reflection"] and reflection_flag:
            ones_mask = TF.hflip(ones_mask)
        
        if group_type in ["rotation", "roto_reflection"] and rotation_angle != 0.0:
            ones_mask = TF.rotate(
                ones_mask,
                rotation_angle,
                interpolation=TF.InterpolationMode.NEAREST,
                fill=0.0
            )
        
        validity_mask_pixels = ones_mask.squeeze(1) > 0.5  # (batch, H, W)
        
        # Calculate MSE error over valid pixels
        diff = transformed_original_logits - augmented_logits  # (batch, num_classes, H, W)
        mse_per_pixel = (diff ** 2).mean(dim=1)  # Average over classes: (batch, H, W)
        mse_masked = mse_per_pixel * validity_mask_pixels.float()
        num_valid_pixels = validity_mask_pixels.sum()
        error = mse_masked.sum() / (num_valid_pixels + 1e-8)
    
    elif task_type == 'trajectory':
        # Transform original predictions to match augmented space
        vec_dim = original_logits.size(2)  # 2 for 2D, 3 for 3D
        
        # Select appropriate functions based on vector dimension
        if vec_dim == 2:
            rotate_fn = rotate_trajectory_2d
            reflect_fn = lambda t: reflect_trajectory_2d(t, axis=reflection_axis)
        elif vec_dim == 3:
            rotate_fn = lambda t, angle: rotate_trajectory_3d(t, angle, axis='z')
            reflect_fn = lambda t: reflect_trajectory_3d(t, axis=reflection_axis)
        else:
            raise ValueError(f"Unsupported vector dimension: {vec_dim}. Must be 2 or 3.")
        
        transformed_original_logits = original_logits.clone()
        
        # Apply same transformations as input augmentation (reflect first, then rotate)
        if group_type in ["reflection", "roto_reflection"] and reflection_flag:
            transformed_original_logits = reflect_fn(transformed_original_logits)
        
        if group_type in ["rotation", "roto_reflection"] and rotation_angle != 0.0:
            transformed_original_logits = rotate_fn(transformed_original_logits, rotation_angle)
        
        # Calculate L2 distance between trajectories
        diff = transformed_original_logits - augmented_logits  # (batch, max_people, 2, pred_len)
        l2_dist = torch.norm(diff, p=2, dim=2)  # (batch, max_people, pred_len)
        
        # Average over time steps
        mean_dist = l2_dist.mean(dim=-1)  # (batch, max_people)
        
        # Apply validity mask if provided
        if validity_mask is not None:
            mean_dist = mean_dist * validity_mask.float()
            error = mean_dist.sum() / (validity_mask.sum() + 1e-8)
        else:
            error = mean_dist.mean()
    
    return error 
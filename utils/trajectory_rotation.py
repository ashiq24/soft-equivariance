"""
Rotation and reflection utilities for 2D and 3D trajectories.

Provides functions to rotate and reflect trajectory tensors and handle coordinate transformations.
"""

import math
import torch


def rotate_trajectory_2d(trajectory, angle_degrees):
    """
    Rotate 2D trajectory by given angle.
    
    Applies 2D rotation matrix:
    [x']   [cos(θ)  -sin(θ)] [x]
    [y'] = [sin(θ)   cos(θ)] [y]
    
    Args:
        trajectory: Tensor of shape (batch, max_people, 2, seq_len) or (batch, max_people, 2)
                   where dim=2 contains [x, y] coordinates.
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise).
        
    Returns:
        rotated_trajectory: Rotated trajectory with same shape as input.
    """
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Handle different input shapes
    if trajectory.dim() == 4:  # (batch, max_people, 2, seq_len)
        batch, max_people, _, seq_len = trajectory.shape
        
        # Extract x and y coordinates
        x = trajectory[:, :, 0, :]  # (batch, max_people, seq_len)
        y = trajectory[:, :, 1, :]  # (batch, max_people, seq_len)
        
        # Apply rotation
        x_rot = cos_a * x - sin_a * y
        y_rot = sin_a * x + cos_a * y
        
        # Stack back
        rotated = torch.stack([x_rot, y_rot], dim=2)  # (batch, max_people, 2, seq_len)
        
    elif trajectory.dim() == 3:  # (batch, max_people, 2)
        batch, max_people, _ = trajectory.shape
        
        x = trajectory[:, :, 0]  # (batch, max_people)
        y = trajectory[:, :, 1]  # (batch, max_people)
        
        x_rot = cos_a * x - sin_a * y
        y_rot = sin_a * x + cos_a * y
        
        rotated = torch.stack([x_rot, y_rot], dim=2)  # (batch, max_people, 2)
    else:
        raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}. "
                        f"Expected (batch, max_people, 2, seq_len) or (batch, max_people, 2).")
    
    return rotated


def reflect_trajectory_2d(trajectory, axis='x'):
    """
    Reflect 2D trajectory along specified axis.
    
    For axis='x': Reflects across X-axis (flips Y coordinate)
        [x']   [1,  0] [x]
        [y'] = [0, -1] [y]
        
    For axis='y': Reflects across Y-axis (flips X coordinate)
        [x']   [-1, 0] [x]
        [y'] = [0,  1] [y]
    
    Args:
        trajectory: Tensor of shape (batch, max_people, 2, seq_len) or (batch, max_people, 2)
                   where dim=2 contains [x, y] coordinates.
        axis: 'x' to flip Y coordinate, 'y' to flip X coordinate.
        
    Returns:
        reflected_trajectory: Reflected trajectory with same shape as input.
    """
    # Handle different input shapes
    if trajectory.dim() == 4:  # (batch, max_people, 2, seq_len)
        x = trajectory[:, :, 0, :]  # (batch, max_people, seq_len)
        y = trajectory[:, :, 1, :]  # (batch, max_people, seq_len)
        
        if axis == 'x':
            # Reflect across X-axis: (x, y) -> (x, -y)
            x_ref = x
            y_ref = -y
        elif axis == 'y':
            # Reflect across Y-axis: (x, y) -> (-x, y)
            x_ref = -x
            y_ref = y
        else:
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
        
        reflected = torch.stack([x_ref, y_ref], dim=2)
        
    elif trajectory.dim() == 3:  # (batch, max_people, 2)
        x = trajectory[:, :, 0]
        y = trajectory[:, :, 1]
        
        if axis == 'x':
            x_ref = x
            y_ref = -y
        elif axis == 'y':
            x_ref = -x
            y_ref = y
        else:
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
        
        reflected = torch.stack([x_ref, y_ref], dim=2)
    else:
        raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}. "
                        f"Expected (batch, max_people, 2, seq_len) or (batch, max_people, 2).")
    
    return reflected


def rotate_trajectory_3d(trajectory, angle_degrees, axis='z'):
    """
    Rotate 3D trajectory by given angle around specified axis.
    
    Args:
        trajectory: Tensor of shape (batch, max_people, 3, seq_len) or (batch, max_people, 3)
                   where dim=2 contains [x, y, z] coordinates.
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise).
        axis: Rotation axis ('x', 'y', or 'z').
        
    Returns:
        rotated_trajectory: Rotated trajectory with same shape as input.
    """
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Handle different input shapes
    if trajectory.dim() == 4:  # (batch, max_people, 3, seq_len)
        x = trajectory[:, :, 0, :]
        y = trajectory[:, :, 1, :]
        z = trajectory[:, :, 2, :]
        
        if axis == 'z':
            x_rot = cos_a * x - sin_a * y
            y_rot = sin_a * x + cos_a * y
            z_rot = z
        elif axis == 'y':
            x_rot = cos_a * x + sin_a * z
            y_rot = y
            z_rot = -sin_a * x + cos_a * z
        elif axis == 'x':
            x_rot = x
            y_rot = cos_a * y - sin_a * z
            z_rot = sin_a * y + cos_a * z
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
        
        rotated = torch.stack([x_rot, y_rot, z_rot], dim=2)
        
    elif trajectory.dim() == 3:  # (batch, max_people, 3)
        x = trajectory[:, :, 0]
        y = trajectory[:, :, 1]
        z = trajectory[:, :, 2]
        
        if axis == 'z':
            x_rot = cos_a * x - sin_a * y
            y_rot = sin_a * x + cos_a * y
            z_rot = z
        elif axis == 'y':
            x_rot = cos_a * x + sin_a * z
            y_rot = y
            z_rot = -sin_a * x + cos_a * z
        elif axis == 'x':
            x_rot = x
            y_rot = cos_a * y - sin_a * z
            z_rot = sin_a * y + cos_a * z
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
        
        rotated = torch.stack([x_rot, y_rot, z_rot], dim=2)
    else:
        raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}. "
                        f"Expected (batch, max_people, 3, seq_len) or (batch, max_people, 3).")
    
    return rotated


def reflect_trajectory_3d(trajectory, axis='z'):
    """
    Reflect 3D trajectory across a plane perpendicular to specified axis.
    
    For axis='z': Reflects across XY-plane (flips Z coordinate)
    For axis='y': Reflects across XZ-plane (flips Y coordinate)
    For axis='x': Reflects across YZ-plane (flips X coordinate)
    
    Args:
        trajectory: Tensor of shape (batch, max_people, 3, seq_len) or (batch, max_people, 3)
                   where dim=2 contains [x, y, z] coordinates.
        axis: 'x', 'y', or 'z' - the axis perpendicular to the reflection plane.
        
    Returns:
        reflected_trajectory: Reflected trajectory with same shape as input.
    """
    if trajectory.dim() == 4:  # (batch, max_people, 3, seq_len)
        x = trajectory[:, :, 0, :]
        y = trajectory[:, :, 1, :]
        z = trajectory[:, :, 2, :]
        
        if axis == 'x':
            x_ref, y_ref, z_ref = -x, y, z
        elif axis == 'y':
            x_ref, y_ref, z_ref = x, -y, z
        elif axis == 'z':
            x_ref, y_ref, z_ref = x, y, -z
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
        
        reflected = torch.stack([x_ref, y_ref, z_ref], dim=2)
        
    elif trajectory.dim() == 3:  # (batch, max_people, 3)
        x = trajectory[:, :, 0]
        y = trajectory[:, :, 1]
        z = trajectory[:, :, 2]
        
        if axis == 'x':
            x_ref, y_ref, z_ref = -x, y, z
        elif axis == 'y':
            x_ref, y_ref, z_ref = x, -y, z
        elif axis == 'z':
            x_ref, y_ref, z_ref = x, y, -z
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
        
        reflected = torch.stack([x_ref, y_ref, z_ref], dim=2)
    else:
        raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}. "
                        f"Expected (batch, max_people, 3, seq_len) or (batch, max_people, 3).")
    
    return reflected

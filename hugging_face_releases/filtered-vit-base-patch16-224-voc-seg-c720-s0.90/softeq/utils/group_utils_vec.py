"""
Group utilities for group actions on vectors.
Implements rotation matrices and Lie algebra generators for 2D and 3D vectors.
"""

import torch
import math


def create_2d_rotation_matrix(angle_degrees: float) -> torch.Tensor:
    """
    Create a 2×2 rotation matrix for 2D vectors.
    
    Args:
        angle_degrees: Rotation angle in degrees
        
    Returns:
        (2, 2) rotation matrix: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ], dtype=torch.float32)
    
    return rotation_matrix


def create_3d_rotation_matrix_z(angle_degrees: float) -> torch.Tensor:
    """
    Create a 3×3 rotation matrix around Z-axis for 3D vectors.
    
    Args:
        angle_degrees: Rotation angle in degrees
        
    Returns:
        (3, 3) rotation matrix around Z-axis
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0.0],
        [sin_theta, cos_theta, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    return rotation_matrix


def create_3d_rotation_matrix_y(angle_degrees: float) -> torch.Tensor:
    """
    Create a 3×3 rotation matrix around Y-axis for 3D vectors.
    
    Args:
        angle_degrees: Rotation angle in degrees
        
    Returns:
        (3, 3) rotation matrix around Y-axis
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    rotation_matrix = torch.tensor([
        [cos_theta, 0.0, sin_theta],
        [0.0, 1.0, 0.0],
        [-sin_theta, 0.0, cos_theta]
    ], dtype=torch.float32)
    
    return rotation_matrix


def create_3d_rotation_matrix_x(angle_degrees: float) -> torch.Tensor:
    """
    Create a 3×3 rotation matrix around X-axis for 3D vectors.
    
    Args:
        angle_degrees: Rotation angle in degrees
        
    Returns:
        (3, 3) rotation matrix around X-axis
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    rotation_matrix = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, cos_theta, -sin_theta],
        [0.0, sin_theta, cos_theta]
    ], dtype=torch.float32)
    
    return rotation_matrix


def create_2d_lie_algebra_so2() -> torch.Tensor:
    """
    Create generator for SO(2) Lie algebra.
    
    This is the infinitesimal generator of 2D rotations,
    representing the derivative of rotation at θ=0.
    
    Returns:
        (2, 2) skew-symmetric matrix: [[0, -1], [1, 0]]
    """
    generator = torch.tensor([
        [0.0, -1.0],
        [1.0, 0.0]
    ], dtype=torch.float32)
    
    return generator


def create_3d_lie_algebra_so3() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create generators for SO(3) Lie algebra.
    
    Returns three infinitesimal generators for 3D rotations around X, Y, Z axes.
    These are the basis elements of the so(3) Lie algebra.
    
    Returns:
        Tuple of (L_x, L_y, L_z) where each is a (3, 3) skew-symmetric matrix
    """
    # Generator for rotation around X-axis
    L_x = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ], dtype=torch.float32)
    
    # Generator for rotation around Y-axis
    L_y = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    ], dtype=torch.float32)
    
    # Generator for rotation around Z-axis
    L_z = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=torch.float32)
    
    return L_x, L_y, L_z


def create_unit_rotation_action_matrix_vec(n_rotations: int, vec_dim: int, axis: str = 'z') -> torch.Tensor:
    """
    Create rotation action matrix for discrete rotations.
    
    Args:
        n_rotations: Number of discrete rotations (e.g., 4 for 90° steps)
        vec_dim: Dimension of vectors (2 or 3)
        axis: Rotation axis for 3D ('x', 'y', or 'z'), ignored for 2D
        
    Returns:
        (vec_dim, vec_dim) rotation matrix representing one unit rotation
        
    Raises:
        ValueError: If vec_dim is not 2 or 3, or axis is invalid
    """
    if vec_dim not in [2, 3]:
        raise ValueError(f"vec_dim must be 2 or 3, got {vec_dim}")
    
    angle_degrees = 360.0 / n_rotations
    
    if vec_dim == 2:
        return create_2d_rotation_matrix(angle_degrees)
    else:  # vec_dim == 3
        axis = axis.lower()
        if axis == 'z':
            return create_3d_rotation_matrix_z(angle_degrees)
        elif axis == 'y':
            return create_3d_rotation_matrix_y(angle_degrees)
        elif axis == 'x':
            return create_3d_rotation_matrix_x(angle_degrees)
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")


def create_lie_algebra_action_matrix_vec(vec_dim: int, axis: str = 'z') -> torch.Tensor:
    """
    Create infinitesimal generator matrix for continuous rotations.
    
    This is used when n_rotations = -1 (Lie algebra representation).
    
    Args:
        vec_dim: Dimension of vectors (2 or 3)
        axis: Rotation axis for 3D ('x', 'y', or 'z'), ignored for 2D
        
    Returns:
        (vec_dim, vec_dim) skew-symmetric generator matrix
        
    Raises:
        ValueError: If vec_dim is not 2 or 3, or axis is invalid
    """
    if vec_dim not in [2, 3]:
        raise ValueError(f"vec_dim must be 2 or 3, got {vec_dim}")
    
    if vec_dim == 2:
        return create_2d_lie_algebra_so2()
    else:  # vec_dim == 3
        L_x, L_y, L_z = create_3d_lie_algebra_so3()
        axis = axis.lower()
        if axis == 'z':
            return L_z
        elif axis == 'y':
            return L_y
        elif axis == 'x':
            return L_x
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")


# =============================================================================
# Reflection Matrices for Vectors
# =============================================================================

def create_2d_reflection_matrix_x() -> torch.Tensor:
    """
    Create a 2×2 reflection matrix along the X-axis for 2D vectors.
    
    Reflects vectors across the X-axis (flips Y coordinate).
    (x, y) -> (x, -y)
    
    Returns:
        (2, 2) reflection matrix: [[1, 0], [0, -1]]
    """
    reflection_matrix = torch.tensor([
        [1.0, 0.0],
        [0.0, -1.0]
    ], dtype=torch.float32)
    
    return reflection_matrix


def create_2d_reflection_matrix_y() -> torch.Tensor:
    """
    Create a 2×2 reflection matrix along the Y-axis for 2D vectors.
    
    Reflects vectors across the Y-axis (flips X coordinate).
    (x, y) -> (-x, y)
    
    Returns:
        (2, 2) reflection matrix: [[-1, 0], [0, 1]]
    """
    reflection_matrix = torch.tensor([
        [-1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float32)
    
    return reflection_matrix


def create_3d_reflection_matrix_x() -> torch.Tensor:
    """
    Create a 3×3 reflection matrix across the YZ-plane (perpendicular to X-axis) for 3D vectors.
    
    Reflects vectors across the YZ-plane (flips X coordinate).
    (x, y, z) -> (-x, y, z)
    
    Returns:
        (3, 3) reflection matrix: [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
    reflection_matrix = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    return reflection_matrix


def create_3d_reflection_matrix_y() -> torch.Tensor:
    """
    Create a 3×3 reflection matrix across the XZ-plane (perpendicular to Y-axis) for 3D vectors.
    
    Reflects vectors across the XZ-plane (flips Y coordinate).
    (x, y, z) -> (x, -y, z)
    
    Returns:
        (3, 3) reflection matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
    """
    reflection_matrix = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    return reflection_matrix


def create_3d_reflection_matrix_z() -> torch.Tensor:
    """
    Create a 3×3 reflection matrix across the XY-plane (perpendicular to Z-axis) for 3D vectors.
    
    Reflects vectors across the XY-plane (flips Z coordinate).
    (x, y, z) -> (x, y, -z)
    
    Returns:
        (3, 3) reflection matrix: [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    """
    reflection_matrix = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=torch.float32)
    
    return reflection_matrix


def create_reflection_action_matrix_vec(vec_dim: int, axis: str = 'x') -> torch.Tensor:
    """
    Create reflection action matrix for vectors.
    
    Args:
        vec_dim: Dimension of vectors (2 or 3)
        axis: Reflection axis ('x', 'y', or 'z')
            - For 2D: 'x' reflects across X-axis (flips Y), 'y' reflects across Y-axis (flips X)
            - For 3D: 'x' reflects across YZ-plane, 'y' across XZ-plane, 'z' across XY-plane
        
    Returns:
        (vec_dim, vec_dim) reflection matrix
        
    Raises:
        ValueError: If vec_dim is not 2 or 3, or axis is invalid
    """
    if vec_dim not in [2, 3]:
        raise ValueError(f"vec_dim must be 2 or 3, got {vec_dim}")
    
    axis = axis.lower()
    
    if vec_dim == 2:
        if axis == 'x':
            return create_2d_reflection_matrix_x()
        elif axis == 'y':
            return create_2d_reflection_matrix_y()
        else:
            raise ValueError(f"For 2D vectors, axis must be 'x' or 'y', got '{axis}'")
    else:  # vec_dim == 3
        if axis == 'x':
            return create_3d_reflection_matrix_x()
        elif axis == 'y':
            return create_3d_reflection_matrix_y()
        elif axis == 'z':
            return create_3d_reflection_matrix_z()
        else:
            raise ValueError(f"For 3D vectors, axis must be 'x', 'y', or 'z', got '{axis}'")


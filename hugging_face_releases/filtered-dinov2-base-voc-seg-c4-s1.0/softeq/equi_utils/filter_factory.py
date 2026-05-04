"""
Filter Factory Module

Centralized factory functions for creating invariant and equivariant filters
based on group type. This provides a single source of truth for filter creation
across all filtered models.

Supported group types:
- "rotation": Discrete rotation group (C_n)
- "roto_reflection": Roto-reflection group (D_n) - rotation + reflection

Usage:
    from softeq.equi_utils.filter_factory import get_invariant_filter, get_equivariant_filter
    
    # Create invariant filter for rotation group
    filter = get_invariant_filter(
        group_type="rotation",
        n_rotations=4,
        input_size=(1, 16, 16),
        soft_threshold=0.0,
        decomposition_method="schur"
    )
    
    # Create invariant filter for roto-reflection group
    filter = get_invariant_filter(
        group_type="roto_reflection",
        n_rotations=4,
        input_size=(1, 16, 16),
        soft_threshold=0.0,
        decomposition_method="svd"
    )
"""

from .rotation_filters import get_invariant_filter_rotation, get_equivariant_filter_rotation
from .roto_reflection_filters import get_invariant_filter_roto_reflection, get_equivariant_filter_roto_reflection


# Valid group types
VALID_GROUP_TYPES = ["rotation", "roto_reflection"]


def get_invariant_filter(group_type: str, n_rotations: int, input_size: tuple, 
                         soft_threshold: float, decomposition_method: str,
                         debug: bool = False, hard_mask: bool = False, preserve_norm: bool = False, joint_decomposition: bool = True):
    """
    Factory function to get the appropriate invariant filter based on group type.
    
    Invariant filters project weights to the invariant subspace of the specified group,
    meaning the filtered weights remain unchanged under group transformations.
    
    Args:
        group_type: Type of group action. Options:
            - "rotation": Discrete rotation group (C_n)
            - "roto_reflection": Roto-reflection group (D_n) - rotation + reflection
        n_rotations: Number of discrete rotations (e.g., 4 for 90° rotations)
        input_size: Input size tuple (channels, height, width)
        soft_threshold: Soft thresholding parameter (0.0 = strict, higher = softer)
        decomposition_method: Decomposition method ('schur' or 'svd')
        debug: Enable debug output
        hard_mask: Use hard masking instead of soft thresholding
        preserve_norm: If True, preserve the norm of weights after projection (default: False)
        joint_decomposition: If True, use a single joint decomposition for
            roto-reflection constraints instead of chaining per-generator projectors.
        
    Returns:
        InvariantProjector: Filter module that projects weights to invariant subspace
        
    Raises:
        ValueError: If group_type is not recognized
        
    Example:
        >>> filter = get_invariant_filter("rotation", 4, (1, 16, 16), 0.0, "schur")
        >>> smoothed_weights = filter.smooth(weights)
    """
    if group_type not in VALID_GROUP_TYPES:
        raise ValueError(f"Unknown group_type: '{group_type}'. Valid options: {VALID_GROUP_TYPES}")
    
    if group_type == "roto_reflection":
        return get_invariant_filter_roto_reflection(
            n_rotations=n_rotations,
            input_size=input_size,
            soft_threshold=soft_threshold,
            decomposition_method=decomposition_method,
            debug=debug,
            hard_mask=hard_mask,
            preserve_norm=preserve_norm,
            joint_decomposition=joint_decomposition
        )
    else:  # rotation (default)
        return get_invariant_filter_rotation(
            n_rotations=n_rotations,
            input_size=input_size,
            soft_threshold=soft_threshold,
            decomposition_method=decomposition_method,
            debug=debug,
            hard_mask=hard_mask,
            preserve_norm=preserve_norm
        )


def get_equivariant_filter(group_type: str, n_rotations: int, input_size: tuple,
                           output_size: tuple, soft_threshold: float,
                           apply_soft_mask: bool = False, debug: bool = False,
                           soft_threshold_reflection: float = None,
                           soft_threshold_rotation: float = None):
    """
    Factory function to get the appropriate equivariant filter based on group type.
    
    Equivariant filters project weights to the equivariant subspace of the specified group,
    meaning the output transforms consistently with the input under group transformations.
    
    Args:
        group_type: Type of group action. Options:
            - "rotation": Discrete rotation group (C_n)
            - "roto_reflection": Roto-reflection group (D_n) - rotation + reflection
        n_rotations: Number of discrete rotations (e.g., 4 for 90° rotations)
        input_size: Input size tuple (channels, height, width)
        output_size: Output size tuple (channels, height, width)
        soft_threshold: Soft thresholding parameter (0.0 = strict, higher = softer)
        apply_soft_mask: Forwarded to the selected backend; currently only some
            rotation/roto-reflection implementations use it.
        debug: Enable debug output
        soft_threshold_reflection: Separate soft threshold for reflection (roto_reflection only)
        soft_threshold_rotation: Separate soft threshold for rotation (roto_reflection only)
        
    Returns:
        EquivariantProjector or MultiGeneratorEquivariantProjector: Filter module
        
    Raises:
        ValueError: If group_type is not recognized
        
    Example:
        >>> filter = get_equivariant_filter("rotation", 4, (1, 7, 7), (1, 7, 7), 0.0)
        >>> equivariant_weights = filter.project(weights)
    """
    if group_type not in VALID_GROUP_TYPES:
        raise ValueError(f"Unknown group_type: '{group_type}'. Valid options: {VALID_GROUP_TYPES}")
    
    if group_type == "roto_reflection":
        return get_equivariant_filter_roto_reflection(
            n_rotations=n_rotations,
            input_size=input_size,
            output_size=output_size,
            soft_threshold=soft_threshold,
            soft_threshold_reflection=soft_threshold_reflection,
            soft_threshold_rotation=soft_threshold_rotation,
            apply_soft_mask=apply_soft_mask,
            debug=debug
        )
    else:  # rotation (default)
        return get_equivariant_filter_rotation(
            n_rotations=n_rotations,
            input_size=input_size,
            output_size=output_size,
            soft_threshold=soft_threshold,
            apply_soft_mask=apply_soft_mask,
            debug=debug
        )

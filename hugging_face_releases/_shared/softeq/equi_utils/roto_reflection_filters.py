from .equi_constraint_ref import DiscreteReflectionConstraints
from .equi_constraints import DiscreteRotationConstraints
from .reflection_constraints_vec import DiscreteReflectionConstraintsVec
from .rotation_constraints_vec import DiscreteRotationConstraintsVec
from .inv_projector import InvariantProjector
import torch
import numpy as np
from softeq.utils.misc import red_warn
from softeq.utils.block_utils import extract_blocks, get_block_norm
from softeq.utils.decompositions import svd_decomposition
import math
from .reflection_filters import get_equivariant_filter_reflection
from .rotation_filters import get_equivariant_filter_rotation
from .reflection_filters import get_invariant_filter_reflection
from .rotation_filters import get_invariant_filter_rotation
from .equi_projectors import MultiGeneratorEquivariantProjector
from .inv_projector import MultiGeneratorInvariantProjector

def get_invariant_filter_roto_reflection(n_rotations,
                                        input_size,
                                        soft_threshold=0.0,
                                        decomposition_method='svd',
                                        debug=False,
                                        hard=False,
                                        hard_mask=False,
                                        vector=False,
                                        axis='x',
                                        preserve_norm=False,
                                        joint_decomposition=False):
    """Create a roto-reflection invariant weight projector.

    For the image backend, the function either chains the rotation and reflection
    invariant projectors or computes a joint SVD over the stacked forward-difference
    matrices. The current vector backend path is intentionally unsupported and
    raises ``NotImplementedError`` because the joint decomposition logic is not
    implemented for vector constraints yet.

    Args:
        n_rotations: Number of discrete rotations.
        input_size: Input image size for images.
        soft_threshold: Parameter (between 0 and 1) controlling soft interpolation between strict and relaxed projection. 0 = perfect invariance, 1 = no projection.
        decomposition_method: Decomposition backend passed to the constraint objects.
        debug: Enable verbose projector diagnostics.
        hard: Enable hard masking when the spectrum has no exact null space.
        hard_mask: Start from a zero-valued mask before the hard/soft blend.
        vector: If True, request the vector backend. This currently raises.
        axis: Axis parameter reserved for the vector backend.
        preserve_norm: Renormalize projected weights to match the original norm.
        joint_decomposition: If True, build one joint projector from stacked
            generators; otherwise chain the separate rotation and reflection projectors.

    Returns:
        A projector module implementing the invariant filter.
    """
    if vector:
        ref_cons = DiscreteReflectionConstraintsVec(
            input_size=input_size,
            output_size=None,
            decomposition_method=decomposition_method,
            axis=axis
        )
        rot_cons = DiscreteRotationConstraintsVec(
            n_rotations=n_rotations,
            input_size=input_size,
            output_size=None,
            decomposition_method=decomposition_method,
            axis=axis
        )
        raise NotImplementedError("Vector constraints are not supported for joint decomposition")
    else:
        ref_cons = DiscreteReflectionConstraints(
            input_size=input_size,
            output_size=None,
            decomposition_method=decomposition_method
        )
        rot_cons = DiscreteRotationConstraints(
            n_rotations=n_rotations,
            input_size=input_size,
            output_size=None,
            decomposition_method=decomposition_method
        )


    
    
    if not joint_decomposition:
        # ref rotation filter 
        rot_filter = get_invariant_filter_rotation(n_rotations, input_size, soft_threshold, decomposition_method, debug, vector, hard, hard_mask, preserve_norm)
        ref_filter = get_invariant_filter_reflection(input_size, soft_threshold, decomposition_method, debug, hard, hard_mask, vector, axis, preserve_norm)
        filter = MultiGeneratorInvariantProjector([ref_filter, rot_filter])
        return filter

    
    # Get the forward difference matrices
    D_ref = ref_cons.forward_difference_matrix  # (d, d)
    D_rot = rot_cons.forward_difference_matrix   # (d, d)
    
    
    d = D_ref.shape[0]
    assert D_ref.shape == D_rot.shape, f"Forward difference matrices must have same shape: {D_ref.shape} vs {D_rot.shape}"
    
    
    # Get basis_l by doing SVD on [D_ref, D_rot] concatenated along the rows -> (d, 2d)
    # Get the left singular vectors
    stacked_horizontal = torch.cat([D_ref, D_rot], dim=1)  # (d, 2d)
    U_l, S_l, V_l = svd_decomposition(stacked_horizontal)
    # U_l has shape (d, d) - these are the left singular vectors
    basis_l = U_l
    
    # Get basis_r by doing SVD on [D_ref; D_rot] concatenated along the columns -> (2d, d)
    # Get the right singular vectors (V_r from SVD, which is already V^T)
    stacked_vertical = torch.cat([D_ref, D_rot], dim=0)  # (2d, d)
    U_r, S_r, V_r = svd_decomposition(stacked_vertical)
    # V_r has shape (d, d) - these are the right singular vectors (columns of V^T)
    basis_r = V_r
    
    # For values, take the sum of the singular values from the two SVDs
    # Pad the shorter one if needed to match lengths
    min_len = min(len(S_l), len(S_r))
    values = S_l[:min_len] + S_r[:min_len]
    
    if debug:
        print(f"Roto-reflection: basis_l shape: {basis_l.shape}, basis_r shape: {basis_r.shape}, values shape: {values.shape}")
        print(f"Singular values (first 10): {values[:min(10, len(values))]}")
    
    # Create the invariant projector (masking is done automatically in InvariantProjector)
    filter = InvariantProjector(basis_l, basis_r, values, softness=soft_threshold, debug=debug, hard=hard, hard_mask=hard_mask, preserve_norm=preserve_norm)
    
    return filter

def get_equivariant_filter_roto_reflection(n_rotations,
                                        input_size,
                                        output_size,
                                        soft_threshold=0.0,
                                        soft_threshold_reflection=None,
                                        soft_threshold_rotation=None,
                                        apply_soft_mask=False,
                                        debug=False,
                                        vector=False,
                                        axis='x'):
    """Create a roto-reflection equivariant weight projector.

    The function builds separate rotation and reflection equivariant projectors
    and applies them sequentially with ``MultiGeneratorEquivariantProjector``.

    Args:
        n_rotations: Number of discrete rotations.
        input_size: Input image size for images, or vector dimension for vectors.
        output_size: Output image size for images, or vector dimension for vectors.
        soft_threshold: Default soft threshold used when the per-generator values
            are not overridden.
        soft_threshold_reflection: Optional override for the reflection projector.
        soft_threshold_rotation: Optional override for the rotation projector.
        apply_soft_mask: Kept for API compatibility; forwarded to the subprojectors.
        debug: Enable verbose projector diagnostics.
        vector: If True, use the vector constraint backend.
        axis: Axis parameter for the vector backend.

    Returns:
        A projector module implementing the equivariant filter.
    """
    
    
    # Use separate softness values if provided, otherwise use the default
    soft_ref = soft_threshold_reflection if soft_threshold_reflection is not None else soft_threshold
    soft_rot = soft_threshold_rotation if soft_threshold_rotation is not None else soft_threshold
    
    # Get individual equivariant filters for reflection and rotation
    filter_reflection = get_equivariant_filter_reflection(
        input_size=input_size,
        output_size=output_size,
        soft_threshold=soft_ref,
        apply_soft_mask=apply_soft_mask,
        debug=debug,
        vector=vector,
        axis=axis
    )
    
    filter_rotation = get_equivariant_filter_rotation(
        n_rotations=n_rotations,
        input_size=input_size,
        output_size=output_size,
        soft_threshold=soft_rot,
        apply_soft_mask=apply_soft_mask,
        debug=debug,
        vector=vector
    )
    
    # Combine the filters using MultiGeneratorEquivariantProjector
    # This applies both filters sequentially to ensure equivariance to both transformations
    filter = MultiGeneratorEquivariantProjector([filter_reflection, filter_rotation])
    
    return filter
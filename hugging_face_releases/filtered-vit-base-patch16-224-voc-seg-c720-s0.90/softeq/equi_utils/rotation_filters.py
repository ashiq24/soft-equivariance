from .equi_constraints import DiscreteRotationConstraints
from .rotation_constraints_vec import DiscreteRotationConstraintsVec
from .equi_projectors import EquivariantProjectorviaSchur as EquivariantProjector
from .inv_projector import InvariantProjector
import torch
import numpy as np
from ..utils.misc import red_warn
from ..utils.block_utils import extract_blocks, get_block_norm
import math

def get_invariant_filter_rotation(n_rotations,
                                 input_size,
                                 soft_threshold=0.0,
                                 decomposition_method='svd',
                                 debug=False,
                                 vector=False,
                                 hard=False,
                                 hard_mask=False,
                                 preserve_norm=False):
    """Create a rotation invariant weight projector.

    ``n_rotations = -1`` switches the vector backend to the continuous Lie
    algebra generators instead of discrete rotations.

    Args:
        n_rotations: Number of discrete rotations, or ``-1`` for the continuous
            Lie-algebra formulation when ``vector=True``.
        input_size: Input image size for the image backend, or vector dimension
            for the vector backend.
        soft_threshold: Parameter (between 0 and 1) controlling soft interpolation between strict and relaxed projection. 0 = perfect invariance, 1 = no projection.
        decomposition_method: Decomposition backend used by the constraint object.
        debug: Enable verbose projector diagnostics.
        vector: If True, use the vector constraint backend instead of images. For 2D vectors.
        hard: Enable hard masking when the spectrum has no exact null space.
        hard_mask: Start from a zero-valued mask before the hard/soft blend.
        preserve_norm: Renormalize projected weights to match the original norm.

    Returns:
        A projector module implementing the invariant filter.
    """

    if vector:
        rot_cons = DiscreteRotationConstraintsVec(n_rotations,
                                            input_size,
                                            output_size=None,
                                            decomposition_method=decomposition_method)
    else:
        rot_cons = DiscreteRotationConstraints(n_rotations,
                                            input_size,
                                            output_size=None,
                                            decomposition_method=decomposition_method)
    rot_cons.create_invariant_basis()
    basis_l, values, basis_r = rot_cons.get_decomposition(rep='input', method=decomposition_method, return_original=False)
    
    filter = InvariantProjector(basis_l, basis_r, values, softness=soft_threshold, debug=debug, hard=hard, hard_mask=hard_mask, preserve_norm=preserve_norm)
    
    return filter


def get_equivariant_filter_rotation(n_rotations,
                                    input_size,
                                    output_size,
                                    soft_threshold=0.0,
                                    apply_soft_mask=False,
                                    debug=False,
                                    vector=False):
    """Create a rotation equivariant weight projector.

    Args:
        n_rotations: Number of discrete rotations, or ``-1`` for the continuous
            Lie-algebra formulation when ``vector=True``.
        input_size: Input image size for the image backend, or vector dimension
            for the vector backend.
        output_size: Output image size for the image backend, or vector dimension
            for the vector backend.
        soft_threshold: Parameter (between 0 and 1) controlling soft interpolation between strict and relaxed projection. 0 = perfect equivariance, 1 = no projection.
        apply_soft_mask: Accepted for API compatibility; the current image backend
            ignores it and applies the Schur-based projector directly.
        debug: Enable verbose projector diagnostics.
        vector: If True, use the vector constraint backend instead of images. For 2D vectors.

    Returns:
        A projector module implementing the equivariant filter.
    """
    
    red_warn("Forcing the channel size to 1")
    red_warn("Doing blockwise weight filtering")
    if vector:
        rot_cons = DiscreteRotationConstraintsVec(n_rotations,
                                            input_size,
                                            output_size=output_size,
                                            decomposition_method='schur')
    else:
        rot_cons = DiscreteRotationConstraints(n_rotations,
                                                (1, input_size[1], input_size[2]),
                                                output_size=(1, output_size[1], output_size[2]),
                                                decomposition_method='schur')
    left_x, sigma_x, right_x = rot_cons.get_decomposition(rep='input', method='schur', return_original=True)
    left_y, sigma_y, right_y = rot_cons.get_decomposition(rep='output', method='schur', return_original=True)

    filter = EquivariantProjector(left_y, left_x, sigma_y, sigma_x, softness=soft_threshold, debug=debug)
    return filter

    
 
### Equivariant filter
## Legacy code 
def _create_mask_from_blocks(sigma_y_blocks, sigma_x_blocks, soft_threshold=0.0, apply_soft_mask=False):
    """Build a block-compatibility mask from Schur block spectra.

    Exact block matches receive weight 1.0. When ``soft_threshold > 0``,
    mismatched blocks are downweighted according to the difference in block
    norms. ``apply_soft_mask`` is currently reserved for API compatibility and
    does not change the returned mask.
    """
    m = sum(block.shape[0] for block in sigma_y_blocks)
    n = sum(block.shape[0] for block in sigma_x_blocks)

    mask = torch.zeros(m, n)
    y_cursor = 0
    for T_I in sigma_y_blocks:
        t_dim = T_I.shape[0]
        x_cursor = 0
        for S_J in sigma_x_blocks:
            s_dim = S_J.shape[0]
            # Check if blocks have the same dimension and are element-wise identical.
            if t_dim == s_dim and torch.allclose(T_I, S_J, atol=1e-6):
                mask[y_cursor : y_cursor + t_dim, x_cursor : x_cursor + s_dim] = 1.0
            else:
                # absolute of the distance
                distance = abs(get_block_norm(T_I) - get_block_norm(S_J))
                # factor Exp ^ {-distance/ (soft_threshold+eps)}
                eps = 1e-8
                mask[y_cursor : y_cursor + t_dim, x_cursor : x_cursor + s_dim] = math.exp(-distance / (soft_threshold + eps))

            x_cursor += s_dim
        y_cursor += t_dim
    return mask

def get_projection_matrices(m, n, sigma_y_blocks, sigma_x_blocks, soft_threshold=0.0):
    """
    Build the blockwise sparse and structural projectors used by the legacy
    block-filter path.

    Args:
        m: Number of rows in the output representation.
        n: Number of columns in the input representation.
        sigma_y_blocks: Output Schur blocks.
        sigma_x_blocks: Input Schur blocks.
        soft_threshold: Interpolates the structure projector toward identity.

    Returns:
        A pair ``(P_sparse, P_struct)`` of sparse projection matrices.
    """
    # 1. Create the Sparsity Projector (P_sparse)
    # This is equivalent to the binary mask, but as a diagonal matrix
    # acting on the flattened mn-dimensional weight vector.
    mask = _create_mask_from_blocks(sigma_y_blocks, sigma_x_blocks) # Helper from before
    P_sparse = torch.diag(mask.flatten())

    # 2. Create the Structure Projector (P_struct)
    P_struct = torch.eye(m * n) # Start with identity
    
    y_cursor = 0
    for T_I in sigma_y_blocks:
        t_dim = T_I.shape[0]
        x_cursor = 0
        for S_J in sigma_x_blocks:
            s_dim = S_J.shape[0]
            
            if t_dim == 2 and s_dim == 2 and torch.equal(T_I, S_J):
                # This is an allowed 2x2 block. We need to modify P_struct
                # to enforce the alpha/beta structure for these 4 weights.
                
                # Get the flat indices for the 4 elements of this block
                k11_idx = (y_cursor + 0) * n + (x_cursor + 0)
                k12_idx = (y_cursor + 0) * n + (x_cursor + 1)
                k21_idx = (y_cursor + 1) * n + (x_cursor + 0)
                k22_idx = (y_cursor + 1) * n + (x_cursor + 1)
                
                # The projection for alpha = (k11+k22)/2
                P_struct[k11_idx, k11_idx] = 0.5
                P_struct[k11_idx, k22_idx] = 0.5
                P_struct[k22_idx, k11_idx] = 0.5
                P_struct[k22_idx, k22_idx] = 0.5
                
                # The projection for beta = (k12-k21)/2
                P_struct[k12_idx, k12_idx] = 0.5
                P_struct[k12_idx, k21_idx] = -0.5
                P_struct[k21_idx, k12_idx] = -0.5
                P_struct[k21_idx, k21_idx] = 0.5
            
            x_cursor += s_dim
        y_cursor += t_dim

    I = torch.eye(m * n)
    P_struct = soft_threshold * I  +  (1 - soft_threshold) * P_struct
    return P_sparse.to_sparse(), P_struct.to_sparse()
    
    
    
from .equi_constraint_ref import DiscreteReflectionConstraints
from .reflection_constraints_vec import DiscreteReflectionConstraintsVec
from .equi_projectors import EquivariantProjectorviaSchur as EquivariantProjector
from .inv_projector import InvariantProjector
import torch
import numpy as np
from ..utils.misc import red_warn
from ..utils.block_utils import extract_blocks, get_block_norm
import math

def get_invariant_filter_reflection(input_size,
                                    soft_threshold=0.0,
                                    decomposition_method='svd',
                                    debug=False,
                                    hard=False,
                                    hard_mask=False,
                                    vector=False,
                                    axis='x',
                                    preserve_norm=False):
    """
    Create an invariant filter for reflection (horizontal flip) along x-axis.
    
    Args:
        input_size (tuple or int): Image size as (channels, height, width) for images,
                                   or vector dimension (2 or 3) for vectors.
        soft_threshold (float): Soft thresholding parameter (0.0 = strict, higher = softer)
        decomposition_method (str): 'schur' or 'svd'
        debug (bool): Enable debug output
        hard (bool): Use hard masking
        hard_mask (bool): Use hard mask for smoothing
        vector (bool): If True, use vector constraints (DiscreteReflectionConstraintsVec)
        axis (str): Reflection axis for vectors ('x', 'y', or 'z'). Only used when vector=True.
        
    Returns:
        InvariantProjector: Filter module that projects weights to invariant subspace
    """
    if vector:
        ref_cons = DiscreteReflectionConstraintsVec(
            input_size=input_size,
            output_size=None,
            decomposition_method=decomposition_method,
            axis=axis
        )
    else:
        ref_cons = DiscreteReflectionConstraints(
            input_size=input_size,
            output_size=None,
            decomposition_method=decomposition_method
        )
    ref_cons.create_invariant_basis()
    basis_l, values, basis_r = ref_cons.get_decomposition(rep='input', method=decomposition_method, return_original=False)
    
    filter = InvariantProjector(basis_l, basis_r, values, softness=soft_threshold, debug=debug, hard=hard, hard_mask=hard_mask, preserve_norm=preserve_norm)
    
    return filter


def get_equivariant_filter_reflection(input_size,
                                      output_size,
                                      soft_threshold=0.0,
                                      apply_soft_mask=False,
                                      debug=False,
                                      vector=False,
                                      axis='x'):
    """
    Create an equivariant filter for reflection (horizontal flip) along x-axis.
    
    Args:
        input_size (tuple or int): Input image size as (channels, height, width) for images,
                                   or vector dimension (2 or 3) for vectors.
        output_size (tuple or int): Output image size as (channels, height, width) for images,
                                    or vector dimension (2 or 3) for vectors.
        soft_threshold (float): Soft thresholding parameter (0.0 = strict, higher = softer)
        apply_soft_mask (bool): Apply soft masking (currently not used, kept for API consistency)
        debug (bool): Enable debug output
        vector (bool): If True, use vector constraints (DiscreteReflectionConstraintsVec)
        axis (str): Reflection axis for vectors ('x', 'y', or 'z'). Only used when vector=True.
        
    Returns:
        EquivariantProjector: Filter module that projects weights to equivariant subspace
    """
    if vector:
        ref_cons = DiscreteReflectionConstraintsVec(
            input_size=input_size,
            output_size=output_size,
            decomposition_method='schur',
            axis=axis
        )
    else:
        red_warn("Forcing the channel size to 1")
        red_warn("Doing blockwise weight filtering")
        
        ref_cons = DiscreteReflectionConstraints(
            input_size=(1, input_size[1], input_size[2]),
            output_size=(1, output_size[1], output_size[2]),
            decomposition_method='schur'
        )
    left_x, sigma_x, right_x = ref_cons.get_decomposition(rep='input', method='schur', return_original=True)
    left_y, sigma_y, right_y = ref_cons.get_decomposition(rep='output', method='schur', return_original=True)

    filter = EquivariantProjector(left_y, left_x, sigma_y, sigma_x, softness=soft_threshold, debug=debug)
    return filter


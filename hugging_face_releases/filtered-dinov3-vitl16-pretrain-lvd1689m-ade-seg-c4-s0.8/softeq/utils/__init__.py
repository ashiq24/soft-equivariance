"""Utilities for softeq package."""

from .decompositions import schur_decomposition, svd_decomposition
from .group_utils import create_unit_rotation_action_matrix
from .misc import create_patterned_array
from .block_utils import extract_blocks, get_block_norm, _create_mask_from_blocks

__all__ = [
    "schur_decomposition",
    "svd_decomposition",
    "create_unit_rotation_action_matrix",
    "create_patterned_array",
    "extract_blocks",
    "get_block_norm",
    "_create_mask_from_blocks",
    "compute_equivariance_error",
]

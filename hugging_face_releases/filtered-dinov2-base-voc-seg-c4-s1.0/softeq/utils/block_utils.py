"""
Utilities for working with block-diagonal matrices in Schur decompositions.

This module provides functions for extracting and analyzing 1x1 and 2x2 blocks
from block-diagonal matrices (real Schur form).
"""

import torch
import math


def extract_blocks(sigma, tol=1e-8):
    """
    Extracts 1x1 and 2x2 blocks from a block-diagonal matrix (real Schur form).

    This function iterates through the diagonal and identifies 2x2 blocks by
    checking for non-zero elements on the sub-diagonal.

    Args:
        sigma (torch.Tensor): A square, block-diagonal matrix.
        tol (float): A small tolerance to check for non-zero elements. Default: 1e-8.

    Returns:
        list of torch.Tensor: A list of the extracted 1x1 and 2x2 blocks.
    
    Example:
        >>> sigma = torch.tensor([[1., 0., 0.], [0., 2., 1.], [0., -1., 2.]])
        >>> blocks = extract_blocks(sigma)
        >>> len(blocks)  # Returns 2: one 1x1 block and one 2x2 block
        2
    """
    blocks = []
    n = sigma.shape[0]
    i = 0
    while i < n:
        # Check for a 2x2 block
        # A 2x2 block is identified by a non-zero sub-diagonal element
        if i + 1 < n and torch.abs(sigma[i + 1, i]) > tol:
            block = sigma[i:i+2, i:i+2]
            blocks.append(block)
            i += 2  # Move past the 2x2 block
        # Otherwise, it's a 1x1 block
        else:
            block = sigma[i:i+1, i:i+1]
            blocks.append(block)
            i += 1  # Move past the 1x1 block
    return blocks


def get_block_norm(block):
    """
    Computes the Frobenius norm of a 1x1 or 2x2 block.
    
    For a 1x1 block, returns its absolute value.
    For a 2x2 block [[a, b], [c, d]], returns sqrt(a^2 + c^2).
    
    Args:
        block (torch.Tensor): A square matrix (1x1 or 2x2).
    
    Returns:
        float: The Frobenius norm of the block.
    
    Raises:
        ValueError: If block is not 1x1 or 2x2.
        
    Example:
        >>> block_1x1 = torch.tensor([[3.0]])
        >>> get_block_norm(block_1x1)
        3.0
        >>> block_2x2 = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        >>> get_block_norm(block_2x2)  # sqrt(3^2 + 5^2) = sqrt(34)
        5.830951690673828
    """
    if block.shape[0] == 1:
        return torch.abs(block[0, 0]).item()
    elif block.shape[0] == 2:
        return torch.sqrt(block[0, 0]**2 + block[1, 0]**2).item()
    else:
        raise ValueError("Block must be either 1x1 or 2x2.")


def _create_mask_from_blocks(sigma_y_blocks, sigma_x_blocks):
    """
    Generates a binary mask from lists of block matrices.
    
    Creates a mask where elements are 1.0 if the corresponding blocks in
    sigma_y and sigma_x have the same dimension and are element-wise identical.
    
    Args:
        sigma_y_blocks (list of torch.Tensor): List of output representation blocks.
        sigma_x_blocks (list of torch.Tensor): List of input representation blocks.
    
    Returns:
        torch.Tensor: Binary mask of shape (m, n) where m and n are the total
                      dimensions of sigma_y and sigma_x respectively.
    
    Example:
        >>> y_blocks = [torch.tensor([[1.0]]), torch.tensor([[2.0, 1.0], [-1.0, 2.0]])]
        >>> x_blocks = [torch.tensor([[1.0]]), torch.tensor([[2.0, 1.0], [-1.0, 2.0]])]
        >>> mask = _create_mask_from_blocks(y_blocks, x_blocks)
        >>> mask.shape
        torch.Size([3, 3])
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
            if t_dim == s_dim and torch.equal(T_I, S_J):
                mask[y_cursor:y_cursor+t_dim, x_cursor:x_cursor+s_dim] = 1.0
            x_cursor += s_dim
        y_cursor += t_dim
    return mask


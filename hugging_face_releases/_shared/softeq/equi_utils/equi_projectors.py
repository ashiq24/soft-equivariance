import torch
import torch.nn as nn
from softeq.utils.block_utils import extract_blocks, get_block_norm
import math
from softeq.utils.misc import _to_blocks, _from_blocks

def _create_mask_from_blocks(sigma_y_blocks, sigma_x_blocks, soft_threshold, debug=False, temperature=1.0):
    """Build a sparsity mask from Schur block spectra.

    Blocks with the same block size and element-wise identical values (checked
    with ``torch.allclose(..., atol=1e-6)``) receive a hard value of 1.0. When
    ``soft_threshold > 0``, mismatched blocks are downweighted smoothly using
    an exponential penalty based on the difference of their block norms and the
    provided ``temperature``.
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
                if debug:
                    print("Exact match found for blocks at y_cursor:", y_cursor, "x_cursor:", x_cursor)
                    print("T_I:", T_I)
                    print("S_J:", S_J)
            elif soft_threshold > 0.0:
                # absolute of the distance
                distance = abs(get_block_norm(T_I) - get_block_norm(S_J))
                if debug:
                    print("Exp distance:", math.exp(-temperature * distance / (soft_threshold + 1e-8)), 'distance tdim sdim:', distance, t_dim, s_dim)
                    # print blocks
                    print("T_I:", T_I)
                    print("S_J:", S_J)

                eps = 1e-8
                mask[y_cursor : y_cursor + t_dim, x_cursor : x_cursor + s_dim] = math.exp(-temperature * distance / (soft_threshold + eps))

            x_cursor += s_dim
        y_cursor += t_dim
    
    if debug:
        print("*^*"*5)
        print("Generated mask from blocks:")
        print(mask)
        print("*^*"*5)
    return mask



class EquivariantProjectorviaSVD(nn.Module):
    def __init__(self, invariant_projector, in_size, out_size, soft_thresold):
        super().__init__()
        self.invariant_projector = invariant_projector
        self.out_size = out_size
        self.in_size = in_size
        self.soft_thresold = soft_thresold

    def forward(self, W):
        if self.soft_thresold == 1.0:
            return W  # No projection needed
        # W is of shape [out_channels * out_size, in_channels * in_size]
        # We want to reshape to [out_channels * in_channels, out_size * in_size]
        
        original_shape = W.shape
        out_channels_x_out_size, in_channels_x_in_size = original_shape
        
        # Calculate individual dimensions
        out_channels = out_channels_x_out_size // self.out_size
        in_channels = in_channels_x_in_size // self.in_size
        
        # Reshape: [out_channels*out_size, in_channels*in_size] -> [out_channels, out_size, in_channels, in_size]
        W_4d = W.view(out_channels, self.out_size, in_channels, self.in_size)
        
        # Permute and reshape to: [out_channels*in_channels, out_size*in_size]
        W_reshaped = W_4d.permute(0, 2, 1, 3).contiguous()
        
        # transpose the last two dim: [out_channels, in_channels, in_size, out_size]
        W_reshaped = W_reshaped.transpose(-2, -1)
        
        # flatten the last two dims: [out_channels * in_channels, in_size * out_size]
        W_reshaped = W_reshaped.contiguous().view(out_channels * in_channels, self.in_size * self.out_size)
        
        # Apply invariant projector
        W_filtered = self.invariant_projector(W_reshaped)   
        
        # Reshape back: [out_channels*in_channels, in_size*out_size] -> [out_channels, in_channels, in_size, out_size]
        W_filtered_4d = W_filtered.view(out_channels, in_channels, self.in_size, self.out_size)
        
        # Transpose last two dims back: [out_channels, in_channels, out_size, in_size]
        W_filtered_4d = W_filtered_4d.transpose(-2, -1)
        
        # Permute back and reshape to original shape: [out_channels*out_size, in_channels*in_size]
        W_filtered = W_filtered_4d.permute(0, 2, 1, 3).contiguous().view(original_shape)
        
        return W_filtered

class EquivariantProjectorviaSchur(nn.Module):
    def __init__(self, U_y, U_x, sigma_y, sigma_x, softness=0.0, debug=False, temperature=1.0):
        super().__init__()
        
        if not (0.0 <= softness <= 1.0):
            raise ValueError("Softness parameter must be between 0.0 and 1.0")
        self.softness = softness
        self.debug = debug
                
        self.register_buffer('U_y', U_y)
        self.register_buffer('U_x', U_x)
        
        sigma_y_blocks = extract_blocks(sigma_y.to(torch.float32))
        sigma_x_blocks = extract_blocks(sigma_x.to(torch.float32))

        # 1. Pre-compute the sparsity mask
        sparsity_mask = _create_mask_from_blocks(sigma_y_blocks, sigma_x_blocks, soft_threshold=softness, debug=debug, temperature=temperature)
        self.register_buffer('sparsity_mask', sparsity_mask)
        
        # 2. Pre-compute indices for vectorized gather/scatter of 2x2 blocks
        block_row_indices = []
        block_col_indices = []
        y_cursor = 0
        for T_I in sigma_y_blocks:
            t_dim = T_I.shape[0]
            x_cursor = 0
            for S_J in sigma_x_blocks:
                s_dim = S_J.shape[0]
                if t_dim == 2 and s_dim == 2 and torch.allclose(T_I, S_J, atol=1e-6):
                    rows = [y_cursor, y_cursor, y_cursor + 1, y_cursor + 1]
                    cols = [x_cursor, x_cursor + 1, x_cursor, x_cursor + 1]
                    block_row_indices.extend(rows)
                    block_col_indices.extend(cols)
                x_cursor += s_dim
            y_cursor += t_dim
        
        self.register_buffer('block_rows', torch.LongTensor(block_row_indices))
        self.register_buffer('block_cols', torch.LongTensor(block_col_indices))
        if debug:
            print("==*=="*10)
            print("Y Sigma:", sigma_y)
            print("Y blocks:", sigma_y_blocks)
            print("==*=="*10)
            print("X Sigma:", sigma_x)
            print("X blocks:", sigma_x_blocks)
            print("==*=="*10)
            print("U_y:", U_y)
            print("U_x:", U_x)
            print("==*=="*10)
            print("Sparsity Mask:", sparsity_mask)
            print("Block row indices for 2x2 blocks:", block_row_indices)
            print("Block col indices for 2x2 blocks:", block_col_indices)
            print("==End=="*10)
            

    def forward(self, W):
        if self.softness == 1.0:
            return W  # No projection needed

        W = _to_blocks(W, W.shape[-2]//self.U_y.shape[0], W.shape[-1]//self.U_x.shape[0], self.U_y.shape[0], self.U_x.shape[0])
        # 1. Transform to Schur Basis and apply sparsity
        W_prime = self.U_y.T @ W @ self.U_x
        W_prime_proj = W_prime * self.sparsity_mask
        
        # If no 2x2 blocks exist, we can exit early.
        if self.block_rows.numel() == 0:
            return _from_blocks(self.U_y @ W_prime_proj @ self.U_x.T)

        # 2. Vectorized Gather of all relevant 2x2 blocks
        flat_blocks = W_prime_proj[..., self.block_rows, self.block_cols]
        num_blocks = len(self.block_rows) // 4
        gathered_blocks = flat_blocks.reshape(*W.shape[:-2], num_blocks, 2, 2)

        # 3. Calculate the EXACT projection on the small gathered tensor
        # this is the Symmetry imposition step, where we enforce the 2x2 blocks to be of the form [[a, b], [-b, a]]
        alpha = (gathered_blocks[..., 0, 0] + gathered_blocks[..., 1, 1]) / 2.0
        beta = (gathered_blocks[..., 0, 1] - gathered_blocks[..., 1, 0]) / 2.0
        row1 = torch.stack([alpha, beta], dim=-1)
        row2 = torch.stack([-beta, alpha], dim=-1)
        hard_projected_blocks = torch.stack([row1, row2], dim=-2)
        
        # 4. Interpolate between original and projected blocks for SOFTNESS
        soft_projected_blocks = (self.softness * gathered_blocks) + \
                                ((1 - self.softness) * hard_projected_blocks)

        # 5. Vectorized Scatter back into the main tensor
        projected_flat = soft_projected_blocks.reshape(*W.shape[:-2], -1)
        W_prime_proj[..., self.block_rows, self.block_cols] = projected_flat

        # 6. Transform back
        W = _from_blocks(self.U_y @ W_prime_proj @ self.U_x.T)
        if self.debug:
            print("==*=="*50)
            print("W prime proj:", W_prime_proj)
            print("==*=="*50)
            print("Weights", W)
            print("==End=="*50)
        
        return W


class MultiGeneratorEquivariantProjectorviaSVD(nn.Module):
    def __init__(self, invariant_projector, in_size, out_size, soft_thresold):
        super().__init__()
        self.invariant_projector = invariant_projector
        self.out_size = out_size
        self.in_size = in_size
        self.soft_thresold = soft_thresold

    def forward(self, W):
        if self.soft_thresold == 1.0:
            return W  # No projection needed
        # W is of shape [out_channels * out_size, in_channels * in_size]
        # We want to reshape to [out_channels * in_channels, out_size * in_size]
        
        original_shape = W.shape
        out_channels_x_out_size, in_channels_x_in_size = original_shape
        
        # Calculate individual dimensions
        out_channels = out_channels_x_out_size // self.out_size
        in_channels = in_channels_x_in_size // self.in_size
        
        # Reshape: [out_channels*out_size, in_channels*in_size] -> [out_channels, out_size, in_channels, in_size]
        W_4d = W.view(out_channels, self.out_size, in_channels, self.in_size)
        
        # Permute and reshape to: [out_channels*in_channels, out_size*in_size]
        W_reshaped = W_4d.permute(0, 2, 1, 3).contiguous()
        
        # transpose the last two dim: [out_channels, in_channels, in_size, out_size]
        W_reshaped = W_reshaped.transpose(-2, -1)
        
        # flatten the last two dims: [out_channels * in_channels, in_size * out_size]
        W_reshaped = W_reshaped.contiguous().view(out_channels * in_channels, self.in_size * self.out_size)
        
        # Apply invariant projector
        W_filtered = self.invariant_projector(W_reshaped)   
        
        # Reshape back: [out_channels*in_channels, in_size*out_size] -> [out_channels, in_channels, in_size, out_size]
        W_filtered_4d = W_filtered.view(out_channels, in_channels, self.in_size, self.out_size)
        
        # Transpose last two dims back: [out_channels, in_channels, out_size, in_size]
        W_filtered_4d = W_filtered_4d.transpose(-2, -1)
        
        # Permute back and reshape to original shape: [out_channels*out_size, in_channels*in_size]
        W_filtered = W_filtered_4d.permute(0, 2, 1, 3).contiguous().view(original_shape)
        
        return W_filtered

class MultiGeneratorEquivariantProjector(nn.Module):
    def __init__(self, EquivariantProjector_list):
        super().__init__()
        ## nn sequence of equivariant projectors
        self.EquivariantProjector_list = nn.ModuleList(EquivariantProjector_list)

    def forward(self, W):
        for _ in range(5):  # For better equivariance needs to be applied multiple times
            for module in self.EquivariantProjector_list:
                W = module(W)
        return W
    def smooth(self, x):
        for _ in range(5):  
            for module in self.EquivariantProjector_list:
                x = module.smooth(x)
        return x
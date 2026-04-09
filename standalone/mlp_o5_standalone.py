#!/usr/bin/env python3
"""
Standalone script for O(5) equivariant MLP using soft equivariance filters.

All softeq dependencies are inlined — only requires:
    pip install torch scipy numpy emlp

Usage:
    python mlp_o5_standalone.py

Architecture:
    Input (in_rep, per channel)
      -> FilteredLinear [in_rep -> hidden_reps[0]] + EQNonLin
      -> ...
      -> FilteredLinear [hidden_reps[-1] -> S]  (invariant)
      -> nn.Linear [S -> S]                     (scalar output)

The `hidden_reps` argument controls which representations are used.
Examples:
    hidden_reps=['V', 'V']        — all-vector hidden layers (5D per channel)
    hidden_reps=['V', 'V*V']      — V then V⊗V (25D tensor) hidden
    hidden_reps=['V*V', 'V*V']    — all-tensor hidden layers
    hidden_reps=['V.T**3']        — rank-3 tensor (125D, expensive)
"""

# ============================================================
# Standard / third-party imports
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import math
from typing import List, Optional

# EMLP is required (not inlined, as it's a complex library).
# Try installed package first, then fall back to external/ directory.
import os as _os, sys as _sys
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_ext_path = _os.path.join(_script_dir, '..', 'external', 'equivariant-MLP')
try:
    from emlp.groups import O
    from emlp.reps import V, Scalar
except ImportError:
    _sys.path.insert(0, _ext_path)
    try:
        from emlp.groups import O
        from emlp.reps import V, Scalar
    except ImportError:
        raise ImportError(
            "EMLP library is required. Install with:\n"
            "  pip install emlp\n"
            "or place the equivariant-MLP repo at external/equivariant-MLP/"
        )

try:
    import jax.numpy as jnp
except ImportError:
    raise ImportError("JAX is required by EMLP. Install with: pip install jax")


# ============================================================
# INLINED: softeq/utils/decompositions.py
# ============================================================

def canonicalize_schur(T, U):
    T_canon = T.copy()
    U_canon = U.copy()
    n = T.shape[0]
    i = 0
    P = np.array([[0., 1.], [1., 0.]])
    while i < n - 1:
        if abs(T_canon[i + 1, i]) > 1e-8:
            block = T_canon[i:i+2, i:i+2]
            b = block[0, 1]
            if b < 0:
                T_canon[i:i+2, i:i+2] = P.T @ block @ P
                U_canon[:, i:i+2] = U_canon[:, i:i+2] @ P
            i += 2
        else:
            i += 1
    return T_canon, U_canon


def schur_decomposition(matrix, return_original=False):
    if not isinstance(matrix, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("Input must be a square matrix")
    matrix_np = matrix.cpu().numpy()
    T_np, Z_np = scipy.linalg.schur(matrix_np, output='real')
    T_np, Z_np = canonicalize_schur(T_np, Z_np)
    if return_original:
        return torch.from_numpy(T_np).float(), torch.from_numpy(Z_np).float()
    n = T_np.shape[0]
    is_scalar = np.zeros(n, dtype=bool)
    is_scalar[:-1] = np.isclose(T_np[1:, :-1].diagonal(), 0, atol=1e-5)
    is_scalar[-1] = True
    scalar_diagonals = []
    scalar_index = []
    i = 0
    while i < n:
        if is_scalar[i]:
            scalar_diagonals.append(np.abs(T_np[i, i]))
            scalar_index.append(i)
            i += 1
        else:
            block_val = np.sqrt(T_np[i, i]**2 + T_np[i+1, i]**2)
            scalar_diagonals.extend([block_val, block_val])
            scalar_index.extend([i, i+1])
            i += 2
    sorted_indices = np.argsort(scalar_diagonals)
    T_sorted = np.array(scalar_diagonals)[sorted_indices]
    Z_sorted = Z_np[:, sorted_indices]
    return torch.from_numpy(T_sorted).float(), torch.from_numpy(Z_sorted).float()


def svd_decomposition(matrix):
    with torch.no_grad():
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        if matrix.dim() != 2:
            raise ValueError("Input must be a 2D matrix")
        device = matrix.device
        dtype = matrix.dtype
        try:
            U, S, V = torch.linalg.svd(matrix.to('cpu'), full_matrices=False)
        except RuntimeError as e:
            try:
                U, S, V = torch.svd(matrix.to('cpu'))
            except RuntimeError:
                raise RuntimeError(f"SVD decomposition failed: {e}")
        U = U.to(device=device, dtype=dtype)
        S = S.to(device=device, dtype=dtype)
        V = V.to(device=device, dtype=dtype).transpose(0, 1)
        U = torch.flip(U, dims=[1])
        S = torch.flip(S, dims=[0])
        V = torch.flip(V, dims=[1])
    return U, S, V


# ============================================================
# INLINED: softeq/utils/block_utils.py
# ============================================================

def extract_blocks(sigma, tol=1e-8):
    blocks = []
    n = sigma.shape[0]
    i = 0
    while i < n:
        if i + 1 < n and torch.abs(sigma[i + 1, i]) > tol:
            block = sigma[i:i+2, i:i+2]
            blocks.append(block)
            i += 2
        else:
            block = sigma[i:i+1, i:i+1]
            blocks.append(block)
            i += 1
    return blocks


def get_block_norm(block):
    if block.shape[0] == 1:
        return torch.abs(block[0, 0]).item()
    elif block.shape[0] == 2:
        return torch.sqrt(block[0, 0]**2 + block[1, 0]**2).item()
    else:
        raise ValueError("Block must be either 1x1 or 2x2.")


# ============================================================
# INLINED: softeq/utils/misc.py (_to_blocks / _from_blocks)
# ============================================================

def _to_blocks(M, c_out, c_in, o, inn):
    return M.view(c_out, o, c_in, inn).permute(0, 2, 1, 3)


def _from_blocks(blocks):
    c_out, c_in, o, inn = blocks.shape
    return blocks.permute(0, 2, 1, 3).contiguous().view(c_out * o, c_in * inn)


# ============================================================
# INLINED: utils/representation_utils.py (Lie algebra extraction)
# ============================================================

def _build_emlp_rep(rep_type: str, G):
    """Build EMLP representation from a type string."""
    import re
    rep_type = rep_type.strip()
    if rep_type in ['fundamental', 'fund']:
        rep_type = 'V'
    elif rep_type in ['scalar', 'trivial', 'S']:
        rep_type = 'Scalar'
    elif rep_type in ['dual', 'V*']:
        rep_type = 'V.T'
    rep_type = rep_type.replace('⊗', '*').replace('⊕', '+').replace('²', '**2').replace('³', '**3')
    rep_type = re.sub(r'V\*\*\*(\d+)', r'V.T**\1', rep_type)
    rep_type = re.sub(r'V\*($|[+)])', r'V.T\1', rep_type)
    namespace = {
        'V': V(G),
        'Scalar': Scalar,
        'T': lambda p, q=0: __import__('emlp').reps.T(p, q, G=G),
        '__builtins__': {},
    }
    try:
        return eval(rep_type, namespace)
    except Exception as e:
        raise ValueError(f"Failed to parse representation '{rep_type}': {e}")


def get_lie_algebra_rep(rep_type: str = 'V', group_name: str = 'O(5)',
                        include_discrete: bool = True) -> List[torch.Tensor]:
    """Extract Lie algebra generators for any EMLP representation."""
    G = O(5)  # Only O(5) is supported in this standalone
    rep = _build_emlp_rep(rep_type, G)
    lie_algebra = []
    for A in G.lie_algebra:
        drho_A = rep.drho(A)
        drho_A_dense = jnp.array(drho_A @ jnp.eye(rep.size()))
        lie_algebra.append(torch.FloatTensor(np.array(drho_A_dense)))
    if include_discrete and len(G.discrete_generators) > 0:
        for h in G.discrete_generators:
            rho_h = rep.rho(h)
            rho_h_dense = jnp.array(rho_h @ jnp.eye(rep.size()))
            rho_h_torch = torch.FloatTensor(np.array(rho_h_dense))
            lie_algebra.append(rho_h_torch - torch.eye(rep.size()))
    return lie_algebra


def get_rep_size(rep_type: str) -> int:
    """Get dimension of a representation."""
    G = O(5)
    rep = _build_emlp_rep(rep_type, G)
    return rep.size()


# ============================================================
# INLINED: softeq/equi_utils/inv_projector.py
# ============================================================

def _create_smoothing_mask(values, size, soft_threshold, dtype, debug=False, hard=False, hard_mask=False):
    values = torch.abs(values)
    zero_indices = torch.where(torch.abs(values) < 1e-5)[0]
    if len(zero_indices) == 0:
        last_zero_eig_index = 0 if not hard else -1
    else:
        last_zero_eig_index = zero_indices[-1]
    last_basis_idx = last_zero_eig_index + int((len(values) - len(zero_indices) - 1) * soft_threshold)
    print(f"  [InvariantProjector] last_basis_idx={last_basis_idx}, total_basis={len(values)}")
    assert (last_basis_idx >= 0 or hard) and last_basis_idx < size[-1]
    exp_values = torch.exp(-values / (soft_threshold + 1e-6)).to(dtype=dtype)
    if hard_mask:
        exp_values = torch.zeros_like(exp_values)
    mask = torch.diag(exp_values)
    if soft_threshold == 1.0:
        mask = torch.eye(size[-1], dtype=dtype, device=mask.device)
    elif last_basis_idx >= 0:
        mask[:last_basis_idx+1, :last_basis_idx+1] = torch.eye(last_basis_idx+1, dtype=dtype, device=mask.device)
    else:
        if hard and soft_threshold == 0.0:
            mask = torch.zeros_like(mask)
    return mask


class InvariantProjector(nn.Module):
    def __init__(self, U_w, U_x, values, softness=0.0, debug=False, hard=False, hard_mask=False, preserve_norm=False):
        super().__init__()
        if not (0.0 <= softness <= 1.0):
            raise ValueError("Softness parameter must be between 0.0 and 1.0")
        self.softness = softness
        self.preserve_norm = preserve_norm
        mask = _create_smoothing_mask(values, U_w.shape, softness, U_w.dtype, debug=debug, hard=hard, hard_mask=hard_mask)
        self.register_buffer('filter_w_T', (U_w @ mask @ U_w.transpose(0, 1)).t())
        self.register_buffer('filter_x', U_x @ mask @ U_x.transpose(0, 1))

    def forward(self, W):
        if self.softness == 1.0:
            return W
        in_channels = W.shape[1] // self.filter_w_T.shape[0]
        W = W.view(W.shape[0], in_channels, self.filter_w_T.shape[0])
        if self.preserve_norm:
            norm_before = torch.norm(W, dim=-1, keepdim=False)
        W = torch.einsum('fi,bci->bcf', self.filter_w_T, W)
        if self.preserve_norm:
            norm_after = torch.norm(W, dim=-1, keepdim=False)
            W = W * (norm_before / (norm_after + 1e-8)).unsqueeze(-1)
        return W.reshape(W.shape[0], -1)

    def smooth(self, x, data_last=False):
        if self.softness == 1.0:
            return x
        if x.dim() == 2:
            in_channels = x.shape[1] // self.filter_x.shape[1]
            x = x.view(x.shape[0], in_channels, self.filter_x.shape[1])
            x = torch.einsum('fi,bci->bcf', self.filter_x, x)
            x = x.reshape(x.shape[0], -1)
        elif x.dim() == 1:
            in_channels = x.shape[0] // self.filter_x.shape[1]
            x = x.view(in_channels, self.filter_x.shape[1])
            x = torch.einsum('fi,ci->cf', self.filter_x, x)
            x = x.reshape(-1)
        else:
            raise ValueError("Input tensor must be 1D or 2D for smooth()")
        return x


class MultiGeneratorInvariantProjector(nn.Module):
    def __init__(self, projector_list):
        super().__init__()
        self.projectors = nn.ModuleList(projector_list)

    def forward(self, W):
        for m in self.projectors:
            W = m(W)
        for m in self.projectors:
            W = m(W)
        return W

    def smooth(self, x):
        for m in self.projectors:
            x = m.smooth(x)
        for m in self.projectors:
            x = m.smooth(x)
        return x


# ============================================================
# INLINED: softeq/equi_utils/equi_projectors.py
# ============================================================

def _create_mask_from_blocks(sigma_y_blocks, sigma_x_blocks, soft_threshold, debug=False, temperature=1.0):
    m = sum(b.shape[0] for b in sigma_y_blocks)
    n = sum(b.shape[0] for b in sigma_x_blocks)
    mask = torch.zeros(m, n)
    y_cursor = 0
    for T_I in sigma_y_blocks:
        t_dim = T_I.shape[0]
        x_cursor = 0
        for S_J in sigma_x_blocks:
            s_dim = S_J.shape[0]
            if t_dim == s_dim and torch.allclose(T_I, S_J, atol=1e-6):
                mask[y_cursor:y_cursor+t_dim, x_cursor:x_cursor+s_dim] = 1.0
            elif soft_threshold > 0.0:
                distance = abs(get_block_norm(T_I) - get_block_norm(S_J))
                mask[y_cursor:y_cursor+t_dim, x_cursor:x_cursor+s_dim] = math.exp(
                    -temperature * distance / (soft_threshold + 1e-8)
                )
            x_cursor += s_dim
        y_cursor += t_dim
    return mask

   
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
        sparsity_mask = _create_mask_from_blocks(sigma_y_blocks, sigma_x_blocks,
                                                  soft_threshold=softness, debug=debug,
                                                  temperature=temperature)
        self.register_buffer('sparsity_mask', sparsity_mask)
        block_row_indices = []
        block_col_indices = []
        y_cursor = 0
        for T_I in sigma_y_blocks:
            t_dim = T_I.shape[0]
            x_cursor = 0
            for S_J in sigma_x_blocks:
                s_dim = S_J.shape[0]
                if t_dim == 2 and s_dim == 2 and torch.allclose(T_I, S_J, atol=1e-6):
                    rows = [y_cursor, y_cursor, y_cursor+1, y_cursor+1]
                    cols = [x_cursor, x_cursor+1, x_cursor, x_cursor+1]
                    block_row_indices.extend(rows)
                    block_col_indices.extend(cols)
                x_cursor += s_dim
            y_cursor += t_dim
        self.register_buffer('block_rows', torch.LongTensor(block_row_indices))
        self.register_buffer('block_cols', torch.LongTensor(block_col_indices))

    def forward(self, W):
        if self.softness == 1.0:
            return W
        W = _to_blocks(W, W.shape[-2]//self.U_y.shape[0], W.shape[-1]//self.U_x.shape[0],
                        self.U_y.shape[0], self.U_x.shape[0])
        W_prime = self.U_y.T @ W @ self.U_x
        W_prime_proj = W_prime * self.sparsity_mask
        if self.block_rows.numel() == 0:
            return _from_blocks(self.U_y @ W_prime_proj @ self.U_x.T)
        flat_blocks = W_prime_proj[..., self.block_rows, self.block_cols]
        num_blocks = len(self.block_rows) // 4
        gathered_blocks = flat_blocks.reshape(*W.shape[:-2], num_blocks, 2, 2)
        alpha = (gathered_blocks[..., 0, 0] + gathered_blocks[..., 1, 1]) / 2.0
        beta = (gathered_blocks[..., 0, 1] - gathered_blocks[..., 1, 0]) / 2.0
        row1 = torch.stack([alpha, beta], dim=-1)
        row2 = torch.stack([-beta, alpha], dim=-1)
        hard_projected_blocks = torch.stack([row1, row2], dim=-2)
        soft_projected_blocks = (self.softness * gathered_blocks) + ((1 - self.softness) * hard_projected_blocks)
        projected_flat = soft_projected_blocks.reshape(*W.shape[:-2], -1)
        W_prime_proj[..., self.block_rows, self.block_cols] = projected_flat
        return _from_blocks(self.U_y @ W_prime_proj @ self.U_x.T)


class MultiGeneratorEquivariantProjector(nn.Module):
    def __init__(self, projector_list):
        super().__init__()
        self.projectors = nn.ModuleList(projector_list)

    def forward(self, W):
        for _ in range(5):
            for m in self.projectors:
                W = m(W)
        return W


class MultiGeneratorEquivariantProjectorviaSVD(nn.Module):
    def __init__(self, invariant_projector, in_size, out_size, soft_threshold):
        super().__init__()
        self.invariant_projector = invariant_projector
        self.out_size = out_size
        self.in_size = in_size
        self.soft_threshold = soft_threshold

    def forward(self, W):
        if self.soft_threshold == 1.0:
            return W
        original_shape = W.shape
        out_channels = original_shape[0] // self.out_size
        in_channels = original_shape[1] // self.in_size
        W_4d = W.view(out_channels, self.out_size, in_channels, self.in_size)
        W_reshaped = W_4d.permute(0, 2, 1, 3).contiguous()
        W_reshaped = W_reshaped.transpose(-2, -1)
        W_reshaped = W_reshaped.contiguous().view(out_channels * in_channels, self.in_size * self.out_size)
        W_filtered = self.invariant_projector(W_reshaped)
        W_filtered_4d = W_filtered.view(out_channels, in_channels, self.in_size, self.out_size)
        W_filtered_4d = W_filtered_4d.transpose(-2, -1)
        return W_filtered_4d.permute(0, 2, 1, 3).contiguous().view(original_shape)


# ============================================================
# INLINED: softeq/equi_utils/o5_constraints_vec.py
# ============================================================

class O5ConstraintsVec:
    """O(5) constraints for any representation using EMLP."""

    def __init__(self, in_rep: str = 'V', out_rep: str = None,
                 decomposition_method: str = 'svd', use_reflection: bool = True):
        self.in_rep = in_rep
        self.out_rep = out_rep
        self.decomposition_method = decomposition_method
        self.use_reflection = use_reflection

        self.input_size = get_rep_size(in_rep)
        if out_rep is not None:
            self.output_size = get_rep_size(out_rep)
        else:
            self.output_size = None

        # Compute forward difference matrices for both reps
        self.forward_difference_matrices = get_lie_algebra_rep(
            in_rep, group_name='O(5)', include_discrete=use_reflection
        )

        if out_rep is not None and out_rep != in_rep:
            self.forward_difference_matrices_out = get_lie_algebra_rep(
                out_rep, group_name='O(5)', include_discrete=use_reflection
            )
        else:
            self.forward_difference_matrices_out = self.forward_difference_matrices

        self.invariant_basis = None
        self.invariant_scaling_values = None
        self.equivariant_basis = None
        self.equivariant_scaling_values = None

    def get_decomposition(self, rep='input', method=None, return_original=False):
        matrices = self.forward_difference_matrices if rep == 'input' else self.forward_difference_matrices_out
        decomp_method = method if method is not None else self.decomposition_method
        decompositions = []
        for matrix in matrices:
            if decomp_method == 'schur':
                s_values, right = schur_decomposition(matrix, return_original=return_original)
                left = right
            elif decomp_method == 'svd':
                left, s_values, right = svd_decomposition(matrix)
            else:
                raise ValueError(f"Decomposition method {decomp_method} not supported")
            decompositions.append((left, s_values, right))
        return decompositions

    def get_equivariant_condition_matrix(self, diagonal_only=False):
        matrices = []
        if diagonal_only:
            for i, matrix in enumerate(self.forward_difference_matrices):
                if i < len(self.forward_difference_matrices_out):
                    out_matrix = self.forward_difference_matrices_out[i]
                    C = (torch.kron(matrix.transpose(0, 1).contiguous(),
                                    torch.eye(self.output_size, device=matrix.device, dtype=matrix.dtype))
                         - torch.kron(torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype),
                                      out_matrix.contiguous()))
                    matrices.append(C)
        else:
            for matrix in self.forward_difference_matrices:
                for out_matrix in self.forward_difference_matrices_out:
                    C = (torch.kron(matrix.transpose(0, 1).contiguous(),
                                    torch.eye(self.output_size, device=matrix.device, dtype=matrix.dtype))
                         - torch.kron(torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype),
                                      out_matrix.contiguous()))
                    matrices.append(C)
        return matrices


# ============================================================
# INLINED: softeq/equi_utils/o5_filter.py
# ============================================================

def get_invariant_filter_o5(input_size: int = 5, soft_threshold: float = 0.0,
                             decomposition_method: str = 'svd', debug: bool = False,
                             hard: bool = False, hard_mask: bool = False,
                             preserve_norm: bool = False, use_reflection: bool = True,
                             joint_decomposition: bool = True, in_rep: str = 'V'):
    """Create an O(5) invariant filter."""
    constraints = O5ConstraintsVec(
        in_rep=in_rep, out_rep=None,
        decomposition_method=decomposition_method, use_reflection=use_reflection
    )

    if not joint_decomposition:
        decompositions = constraints.get_decomposition(rep='input', method=decomposition_method)
        projectors = []
        for basis_l, values, basis_r in decompositions:
            projectors.append(InvariantProjector(basis_l, basis_r, values,
                                                  softness=soft_threshold, debug=debug,
                                                  hard=hard, hard_mask=hard_mask,
                                                  preserve_norm=preserve_norm))
        return MultiGeneratorInvariantProjector(projectors)
    else:
        fwd = constraints.forward_difference_matrices
        stacked_h = torch.cat(fwd, dim=1)
        stacked_v = torch.cat(fwd, dim=0)
        U_l, S_l, _ = svd_decomposition(stacked_h)
        _, S_r, V_r = svd_decomposition(stacked_v)
        min_len = min(len(S_l), len(S_r))
        values = S_l[:min_len] + S_r[:min_len]
        return InvariantProjector(U_l, V_r, values, softness=soft_threshold, debug=debug,
                                   hard=hard, hard_mask=hard_mask, preserve_norm=preserve_norm)


def get_equivariant_filter_o5(input_size: int = 5, output_size: int = 5,
                               soft_threshold: float = 0.0, debug: bool = False,
                               use_reflection: bool = True, use_invariant_filter: bool = False,
                               hard: bool = False, hard_mask: bool = False,
                               in_rep: str = 'V', out_rep: str = 'V'):
    """Create an O(5) equivariant filter."""
    constraints = O5ConstraintsVec(
        in_rep=in_rep, out_rep=out_rep,
        decomposition_method='svd', use_reflection=use_reflection
    )

    if use_invariant_filter:
        condition_list = constraints.get_equivariant_condition_matrix(diagonal_only=True)
        stacked_condition = torch.cat(condition_list, dim=0)
        U, S, V_mat = svd_decomposition(stacked_condition)
        tolerance = 1e-5
        zero_indices = torch.where(S < tolerance)[0]
        if len(zero_indices) == 0:
            num_keep = int(V_mat.shape[1] * (1 - soft_threshold))
            null_space_basis = V_mat[:, -num_keep:]
        else:
            null_space_basis = V_mat[:, zero_indices]
        in_shape = get_rep_size(in_rep)
        out_shape = get_rep_size(out_rep)
        inv_filter = InvariantProjector(V_mat, V_mat, S[:V_mat.shape[1]],
                                         softness=soft_threshold, hard_mask=hard_mask,
                                         hard=hard, debug=debug)
        return MultiGeneratorEquivariantProjectorviaSVD(inv_filter, in_shape, out_shape, soft_threshold)
    else:
        decompositions_input = constraints.get_decomposition(rep='input', method='schur', return_original=True)
        decompositions_output = constraints.get_decomposition(rep='output', method='schur', return_original=True)
        projectors = []
        for i, ((left_x, sigma_x, right_x), (left_y, sigma_y, right_y)) in enumerate(
                zip(decompositions_input, decompositions_output)):
            print(f"  Creating EquivariantProjectorviaSchur for generator {i}")
            projectors.append(EquivariantProjectorviaSchur(
                left_y, left_x, sigma_y, sigma_x,
                softness=soft_threshold, debug=debug, temperature=0.01
            ))
        return MultiGeneratorEquivariantProjector(projectors)


# ============================================================
# INLINED: softeq/layers/flinear.py — FilteredLinear
# ============================================================

class FilteredLinear(nn.Module):
    """Linear layer with equivariant/invariant weight filtering."""

    def __init__(self, original_layer: nn.Linear, filter_eq: nn.Module, filter_inv: nn.Module = None):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.weight = nn.Parameter(original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        self.filter_eq = filter_eq
        self.filter_inv = filter_inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_2d = self.weight.view(self.weight.shape[0], -1)
        filtered_weight = self.filter_eq(weight_2d).view(self.weight.shape)
        if self.bias is not None and self.filter_inv is not None:
            self.bias.data = self.filter_inv.smooth(self.bias.data)
        return F.linear(x, filtered_weight, self.bias)


# ============================================================
# INLINED: utils/eq_nonlin.py — EQNonLin
# ============================================================

class EQNonLin(nn.Module):
    """
    Equivariant non-linearity: norm-based activation applied per-vector.
    f(v) = act(||v|| + bias) * v  (applied channel-wise)
    """
    def __init__(self, dim: int = 5, nonlinearity: str = 'relu',
                 per_channel_bias: bool = True, n_channels: int = 1):
        super().__init__()
        self.dim = dim
        if nonlinearity == 'relu':
            self.act = nn.ReLU()
        elif nonlinearity == 'gelu':
            self.act = nn.GELU()
        elif nonlinearity == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        self.bias = nn.Parameter(torch.zeros(n_channels if per_channel_bias else 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("EQNonLin expects 2D input (batch, channels*dim)")
        batch_size, total_dim = x.shape
        assert total_dim % self.dim == 0, "Input dim not divisible by vector dim"
        n_vectors = total_dim // self.dim
        x_r = x.view(batch_size, n_vectors, self.dim)
        norms = torch.norm(x_r, dim=-1, keepdim=True)  # (batch, n_vectors, 1)
        if self.bias.numel() == 1:
            bias = self.bias.view(1, 1, 1)
        elif self.bias.numel() == n_vectors:
            bias = self.bias.view(1, n_vectors, 1)
        else:
            raise ValueError(f"Bias channels ({self.bias.numel()}) != n_vectors ({n_vectors})")
        activated_norms = self.act(norms + bias)
        return (x_r * activated_norms).view(batch_size, total_dim)


# ============================================================
# O(5) Equivariant MLP (from models/filtered_o5.py)
# ============================================================

class O5EquivariantMLP(nn.Module):
    """
    O(5) equivariant MLP with soft-equivariance filter projection.

    Supports arbitrary representation paths via `hidden_reps`. Each entry is an EMLP
    representation string, e.g. 'V', 'V*V', 'V.T**2', 'V.T**3'.  The special token
    'S' means scalar (1D, no equivariance constraint on weight).

    Architecture:
        Input (in_rep, per channel)
          -> FilteredLinear [in_rep -> hidden_reps[0]] + EQNonLin
          -> FilteredLinear [hidden_reps[i] -> hidden_reps[i+1]] + EQNonLin ...
          -> FilteredLinear [hidden_reps[-1] -> 'S'] (invariant projection)
          -> nn.Linear [S -> S] (scalar mixing -> scalar output)

    Args:
        in_channels:   Number of input rep-sized vectors (input dim = in_channels * rep_dim(in_rep))
        in_rep:        Input representation string (default 'V', 5D for O(5))
        out_features:  Scalar output dimension
        hidden_channels: List of channel counts — one per hidden_reps entry
        hidden_reps:   List of representation strings for hidden layers.
                       Examples:
                         ['V', 'V*V']        — V → V → V*V → S  (tensor hidden)
                         ['V', 'V', 'V']     — V → V → V → V → S  (vector only)
                         ['V*V', 'V*V']      — V → V*V → V*V → S
                         ['V.T**2', 'V*V']   — V → T(2) → V*V → S
        filter_config: Dict with filter options (soft_thresholding, hard, hard_mask, ...)
    """

    def __init__(
        self,
        in_channels: int = 2,
        in_rep: str = 'V',
        out_features: int = 1,
        hidden_channels: List[int] = None,
        hidden_reps: List[str] = None,
        filter_config: Optional[dict] = None,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [4, 4]
        if hidden_reps is None:
            hidden_reps = ['V'] * len(hidden_channels)
        if len(hidden_channels) != len(hidden_reps):
            raise ValueError(
                f"hidden_channels (len={len(hidden_channels)}) and "
                f"hidden_reps (len={len(hidden_reps)}) must have the same length"
            )

        self.in_channels = in_channels
        self.in_rep = in_rep
        self.out_features = out_features
        self.hidden_channels = hidden_channels
        self.hidden_reps = hidden_reps

        cfg = filter_config or {}
        soft_thresh = cfg.get('soft_thresholding', 0.0)
        use_reflection = cfg.get('use_reflection', True)
        hard = cfg.get('hard', True)
        hard_mask = cfg.get('hard_mask', True)
        debug = cfg.get('debug', False)
        decomp_method = cfg.get('decomposition_method', 'svd')
        use_inv_filter = cfg.get('use_invariant_filter', False)

        # Cache rep sizes (computed via EMLP, may be slow for large reps)
        _rep_size_cache = {}
        def rep_dim(rep: str) -> int:
            if rep == 'S':
                return 1
            if rep not in _rep_size_cache:
                _rep_size_cache[rep] = get_rep_size(rep)
            return _rep_size_cache[rep]

        # Build layer specifications: (in_rep_str, out_rep_str, in_ch, out_ch)
        # Full path: in_rep -> hidden_reps[0] -> ... -> hidden_reps[-1] -> 'S' -> 'S'
        all_reps = [in_rep] + list(hidden_reps) + ['S']
        all_channels = [in_channels] + list(hidden_channels) + [hidden_channels[-1] * out_features, out_features]

        specs = []
        for i in range(len(all_reps) - 1):
            specs.append((all_reps[i], all_reps[i + 1], all_channels[i], all_channels[i + 1]))

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for in_rep_l, out_rep_l, in_ch, out_ch in specs:
            in_dim = in_ch * rep_dim(in_rep_l)
            out_dim = out_ch * rep_dim(out_rep_l)

            print(f"  Building layer: [{in_rep_l}({in_ch})] -> [{out_rep_l}({out_ch})]  "
                  f"dim {in_dim} -> {out_dim}")

            if in_rep_l == 'S' and out_rep_l == 'S':
                layer = nn.Linear(in_dim, out_dim)
                act = nn.Identity()

            elif out_rep_l == 'S':
                # Invariant projection: non-scalar rep -> scalar
                filt = get_invariant_filter_o5(
                    input_size=rep_dim(in_rep_l),
                    soft_threshold=soft_thresh,
                    decomposition_method=decomp_method,
                    debug=debug,
                    use_reflection=use_reflection,
                    hard=hard,
                    hard_mask=hard_mask,
                    in_rep=in_rep_l,
                )
                layer = FilteredLinear(nn.Linear(in_dim, out_dim), filter_eq=filt, filter_inv=None)
                act = nn.ReLU()

            else:
                # Equivariant layer: non-scalar rep -> non-scalar rep
                filt_eq = get_equivariant_filter_o5(
                    input_size=rep_dim(in_rep_l),
                    output_size=rep_dim(out_rep_l),
                    soft_threshold=soft_thresh,
                    debug=debug,
                    use_reflection=use_reflection,
                    use_invariant_filter=use_inv_filter,
                    hard=hard,
                    hard_mask=hard_mask,
                    in_rep=in_rep_l,
                    out_rep=out_rep_l,
                )
                filt_inv = get_invariant_filter_o5(
                    input_size=rep_dim(out_rep_l),
                    soft_threshold=soft_thresh,
                    decomposition_method=decomp_method,
                    debug=debug,
                    use_reflection=use_reflection,
                    hard=hard,
                    hard_mask=hard_mask,
                    in_rep=out_rep_l,
                )
                layer = FilteredLinear(nn.Linear(in_dim, out_dim), filter_eq=filt_eq, filter_inv=filt_inv)
                act = EQNonLin(dim=rep_dim(out_rep_l), nonlinearity='relu',
                                per_channel_bias=True, n_channels=out_ch)

            self.layers.append(layer)
            self.activations.append(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels * rep_dim(in_rep))
        Returns:
            (batch, out_features)
        """
        for layer, act in zip(self.layers, self.activations):
            x = layer(x)
            x = act(x)
        return x


# ============================================================
# Demo / main
# ============================================================

def test_o5_equivariance(model: nn.Module, in_channels: int, in_rep: str = 'V', n_samples: int = 8):
    """
    Verify O(5) equivariance: f(R @ x) ≈ f(x)  (for invariant output).
    Generates a random O(5) element and applies the group action rho(g) to the input.

    For non-fundamental reps (e.g. V*V) the group acts via the representation matrix
    rho(g) of size (rep_dim, rep_dim), not just the 5x5 rotation matrix.
    """
    print(f"\n--- O(5) Equivariance Test  (in_rep='{in_rep}') ---")
    model.eval()
    G = O(5)

    # Sample a random group element (5x5 orthogonal matrix in fundamental rep)
    g = G.samples(1)       # jax array, shape (1, 5, 5)
    g_np = np.array(g[0])  # (5, 5)
    g_jax = g[0]           # single element for rho computation

    # Get the representation matrix rho(g) for the chosen rep
    emlp_rep = _build_emlp_rep(in_rep, G)
    rho_g = emlp_rep.rho(g_jax)
    rho_g_dense = np.array(rho_g @ np.eye(emlp_rep.size()))  # materialise sparse
    R = torch.FloatTensor(rho_g_dense)  # (rep_dim, rep_dim)

    rep_dim_val = R.shape[0]
    total_dim = in_channels * rep_dim_val

    with torch.no_grad():
        x = torch.randn(n_samples, total_dim)

        # Apply group action channel-wise: x_rot[b, c, :] = R @ x[b, c, :]
        x_rotated = x.view(n_samples, in_channels, rep_dim_val) @ R.T
        x_rotated = x_rotated.view(n_samples, total_dim)

        y_orig = model(x)
        y_rot = model(x_rotated)

        err = (y_orig - y_rot).abs().mean().item()
        rel_err = err / (y_orig.abs().mean().item() + 1e-8)
        print(f"  |f(x) - f(Rx)|  = {err:.6f}")
        print(f"  relative error  = {rel_err:.6f}")
        print(f"  f(x)  mean/std  = {y_orig.mean().item():.4f} / {y_orig.std().item():.4f}")
        print(f"  Equivariance {'PASSED' if rel_err < 0.05 else 'FAILED (soft filter may allow small error)'}")
    return rel_err


def main():
    print("=" * 60)
    print("O(5) Equivariant MLP — Standalone Demo")
    print("=" * 60)

    # -------------------------------------------------------
    # Configuration
    # Representation examples:
    #   'V'        —  5D fundamental (O(5) vector)
    #   'V*V'      — 25D tensor product (T2)
    #   'V.T**2'   — same as V*V via EMLP notation
    #   'V.T**3'   — 125D rank-3 tensor (expensive!)
    # -------------------------------------------------------
    IN_CHANNELS = 2          # 2 input 5D vectors → input dim = 10
    IN_REP = 'V'             # input representation
    OUT_FEATURES = 1         # scalar output

    # Representation path through hidden layers.
    # Change this to explore different architectures, e.g.:
    #   ['V', 'V']        — all-vector hidden layers
    #   ['V', 'V*V']      — V then V*V (25D tensor) hidden
    #   ['V*V', 'V*V']    — all-tensor hidden layers
    HIDDEN_REPS = ['V', 'V*V']
    HIDDEN_CHANNELS = [4, 4]  # channel counts matching HIDDEN_REPS
    BATCH_SIZE = 16

    filter_config = {
        'soft_thresholding': 0.5,   # 0.0 = hard equivariance (exact); 1.0 = no filtering
        'hard': True,
        'hard_mask': True,
        'use_reflection': True,     # Full O(5) including reflection
        'decomposition_method': 'svd',
        'use_invariant_filter': False,  # Schur-based equivariant filter
        'debug': False,
    }

    in_dim = IN_CHANNELS * get_rep_size(IN_REP)
    print(f"\nConfig:")
    print(f"  Input : {IN_CHANNELS} x {IN_REP} → total {in_dim}D")
    print(f"  Hidden: reps={HIDDEN_REPS}, channels={HIDDEN_CHANNELS}")
    print(f"  Output: {OUT_FEATURES} scalar(s)")
    print(f"  Filter: soft_thresholding={filter_config['soft_thresholding']}")

    # -------------------------------------------------------
    # Build model
    # -------------------------------------------------------
    print("\nBuilding O(5) equivariant MLP...")
    model = O5EquivariantMLP(
        in_channels=IN_CHANNELS,
        in_rep=IN_REP,
        out_features=OUT_FEATURES,
        hidden_channels=HIDDEN_CHANNELS,
        hidden_reps=HIDDEN_REPS,
        filter_config=filter_config,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel built. Parameters: {n_params:,}")

    # -------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------
    print("\nRunning forward pass...")
    x = torch.randn(BATCH_SIZE, in_dim)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output sample: {y[:4].squeeze().tolist()}")

    # -------------------------------------------------------
    # Equivariance test
    # -------------------------------------------------------
    rel_err = test_o5_equivariance(model, IN_CHANNELS, in_rep=IN_REP)

    # -------------------------------------------------------
    # Quick training demo
    # -------------------------------------------------------
    print("\n--- Quick Training Demo (20 steps) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic O(5)-invariant target: ||x||^2 summed over channels
    def invariant_target(x, in_ch):
        # sum of squared norms of each 5D vector
        return x.view(x.shape[0], in_ch, 5).norm(dim=-1).sum(dim=-1, keepdim=True)

    for step in range(20):
        x_batch = torch.randn(32, in_dim)
        y_target = invariant_target(x_batch, IN_CHANNELS)
        y_pred = model(x_batch)
        loss = F.mse_loss(y_pred, y_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1:2d}  loss={loss.item():.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

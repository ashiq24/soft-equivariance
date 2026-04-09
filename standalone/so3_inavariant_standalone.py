#!/usr/bin/env python3
"""
Standalone script for SO(3) Deep Invariant MLP using soft equivariance filters.

All required components are inlined from the project so this file can run
without importing project modules.

Requirements:
    pip install torch scipy numpy

Usage:
    python so3_inavariant_standalone.py
"""

from abc import ABC, abstractmethod
import math

import numpy as np
import scipy
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init


# ============================================================
# INLINED: softeq/equi_utils/base_constraints.py
# ============================================================

class BaseConstraints(ABC):
    """Abstract base class for group symmetry constraints."""

    @abstractmethod
    def create_unit_group_action_matrix(self):
        ...

    def apply_unit_group_action(self, vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement apply_unit_group_action().")

    @abstractmethod
    def create_unit_group_action_forward_difference_matrix(self):
        ...

    def apply_forward_difference(self, vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement apply_forward_difference().")

    @abstractmethod
    def get_decomposition(self, rep: str = 'input', method: str = None, return_original: bool = False):
        ...

    def create_invariant_basis(self):
        raise NotImplementedError("Subclasses must implement create_invariant_basis().")

    def get_invariant_basis(self, soft_thresholding: float = 0.0, tolerance: float = 1e-4):
        raise NotImplementedError("Subclasses must implement get_invariant_basis().")

    def create_equivariant_basis(self):
        raise NotImplementedError("Subclasses must implement create_equivariant_basis().")

    def get_equivariant_basis(self, soft_thresholding: float = 0.0, tolerance: float = 1e-4):
        raise NotImplementedError("Subclasses must implement get_equivariant_basis().")

    def _select_basis(
        self,
        basis: torch.Tensor,
        scaling_values: torch.Tensor,
        soft_thresholding: float,
        tolerance: float,
        label: str
    ):
        raise NotImplementedError("Subclasses must implement _select_basis().")


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
            block = T_canon[i:i + 2, i:i + 2]
            b = block[0, 1]

            if b < 0:
                T_canon[i:i + 2, i:i + 2] = P.T @ block @ P
                U_canon[:, i:i + 2] = U_canon[:, i:i + 2] @ P

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
    i = 0
    while i < n:
        if is_scalar[i]:
            scalar_diagonals.append(np.abs(T_np[i, i]))
            i += 1
        else:
            block_val = np.sqrt(T_np[i, i] ** 2 + T_np[i + 1, i] ** 2)
            scalar_diagonals.extend([block_val, block_val])
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
        svd_device = 'cpu'

        try:
            U, S, V = torch.linalg.svd(matrix.to(svd_device), full_matrices=False)
        except RuntimeError as e:
            print("Using torch.svd instead of torch.linalg.svd")
            try:
                U, S, V = torch.svd(matrix.to(svd_device))
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
# INLINED: softeq/equi_utils/inv_projector.py
# ============================================================

def _create_smoothing_mask(values, size, soft_threshold, dtype, debug=False, hard=False, hard_mask=False):
    values = torch.abs(values)
    if debug:
        print("values:", values)
    zero_indices = torch.where(torch.abs(values) < 1e-5)[0]
    if len(zero_indices) == 0:
        last_zero_eig_index = 0 if not hard else -1
    else:
        last_zero_eig_index = zero_indices[-1]

    last_basis_idx = last_zero_eig_index + int((len(values) - len(zero_indices) - 1) * soft_threshold)

    print("last_basis_idx and total basis:", last_basis_idx, len(values))
    assert (last_basis_idx >= 0 or hard) and last_basis_idx < size[-1]

    exp_values = torch.exp(-values / (soft_threshold + 1e-6)).to(dtype=dtype)
    if hard_mask:
        exp_values = torch.zeros_like(exp_values)
    mask = torch.diag(exp_values)

    if soft_threshold == 1.0:
        mask = torch.eye(size[-1], dtype=dtype, device=mask.device)
    elif last_basis_idx >= 0:
        mask[:last_basis_idx + 1, :last_basis_idx + 1] = torch.eye(last_basis_idx + 1, dtype=dtype, device=mask.device)
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
        self.debug = debug
        self.preserve_norm = preserve_norm
        mask = _create_smoothing_mask(values, U_w.shape, softness, U_w.dtype, debug=debug, hard=hard, hard_mask=hard_mask)

        if self.debug:
            print("Values for smoothing mask:")
            print("---" * 20)
            print(values)
            print("---" * 20)
            print("Smoothing mask:")
            for i in mask:
                print(i)
            print("---" * 20)

        filter_w = U_w @ mask @ U_w.transpose(0, 1)
        self.register_buffer('filter_w_T', filter_w.t())
        filter_x = U_x @ mask @ U_x.transpose(0, 1)
        self.register_buffer('filter_x', filter_x)

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
            norm_ratio = norm_before / (norm_after + 1e-8)
            W = W * norm_ratio.unsqueeze(-1)

        return W.reshape(W.shape[0], -1)

    def smooth(self, x, data_last=False):
        if self.softness == 1.0:
            return x

        if x.dim() == 3:
            if data_last:
                in_channels = x.shape[2] // self.filter_x.shape[1]
                x = x.view(x.shape[0], x.shape[1], in_channels, self.filter_x.shape[1])
                if self.preserve_norm:
                    norm_before = torch.norm(x, dim=-1, keepdim=False)
                x = torch.einsum('fi,btci->btcf', self.filter_x, x)
                if self.preserve_norm:
                    norm_after = torch.norm(x, dim=-1, keepdim=False)
                    norm_ratio = norm_before / (norm_after + 1e-8)
                    x = x * norm_ratio.unsqueeze(-1)
                x = x.reshape(x.shape[0], x.shape[1], -1)
            else:
                if self.preserve_norm:
                    norm_before = torch.norm(x, dim=1, keepdim=False)
                x = torch.einsum('fi,bic->bfc', self.filter_x, x)
                if self.preserve_norm:
                    norm_after = torch.norm(x, dim=1, keepdim=False)
                    norm_ratio = norm_before / (norm_after + 1e-8)
                    x = x * norm_ratio.unsqueeze(1)
        elif x.dim() == 2:
            in_channels = x.shape[1] // self.filter_x.shape[1]
            x = x.view(x.shape[0], in_channels, self.filter_x.shape[1])
            if self.preserve_norm:
                norm_before = torch.norm(x, dim=-1, keepdim=False)
            x = torch.einsum('fi,bci->bcf', self.filter_x, x)
            if self.preserve_norm:
                norm_after = torch.norm(x, dim=-1, keepdim=False)
                norm_ratio = norm_before / (norm_after + 1e-8)
                x = x * norm_ratio.unsqueeze(-1)
            x = x.reshape(x.shape[0], -1)
        elif x.dim() == 1:
            in_channels = x.shape[0] // self.filter_x.shape[1]
            x = x.view(in_channels, self.filter_x.shape[1])
            if self.preserve_norm:
                norm_before = torch.norm(x, dim=-1, keepdim=False)
            x = torch.einsum('fi,ci->cf', self.filter_x, x)
            if self.preserve_norm:
                norm_after = torch.norm(x, dim=-1, keepdim=False)
                norm_ratio = norm_before / (norm_after + 1e-8)
                x = x * norm_ratio.unsqueeze(-1)
            x = x.reshape(-1)
        else:
            raise ValueError("Input tensor must be 1D, 2D, or 3D")

        return x


class MultiGeneratorInvariantProjector(nn.Module):
    def __init__(self, InvariantProjector_list):
        super().__init__()
        self.InvariantProjector_list = nn.ModuleList(InvariantProjector_list)

    def forward(self, W):
        for module in self.InvariantProjector_list:
            W = module(W)
        for module in self.InvariantProjector_list:
            W = module(W)
        return W

    def smooth(self, x):
        for module in self.InvariantProjector_list:
            x = module.smooth(x)
        for module in self.InvariantProjector_list:
            x = module.smooth(x)
        return x


# ============================================================
# INLINED: softeq/equi_utils/equi_projectors.py (SVD projector)
# ============================================================

class MultiGeneratorEquivariantProjectorviaSVD(nn.Module):
    def __init__(self, invariant_projector, in_size, out_size, soft_thresold):
        super().__init__()
        self.invariant_projector = invariant_projector
        self.out_size = out_size
        self.in_size = in_size
        self.soft_thresold = soft_thresold

    def forward(self, W):
        if self.soft_thresold == 1.0:
            return W

        original_shape = W.shape
        out_channels_x_out_size, in_channels_x_in_size = original_shape

        out_channels = out_channels_x_out_size // self.out_size
        in_channels = in_channels_x_in_size // self.in_size

        W_4d = W.view(out_channels, self.out_size, in_channels, self.in_size)
        W_reshaped = W_4d.permute(0, 2, 1, 3).contiguous()
        W_reshaped = W_reshaped.transpose(-2, -1)
        W_reshaped = W_reshaped.contiguous().view(out_channels * in_channels, self.in_size * self.out_size)

        W_filtered = self.invariant_projector(W_reshaped)

        W_filtered_4d = W_filtered.view(out_channels, in_channels, self.in_size, self.out_size)
        W_filtered_4d = W_filtered_4d.transpose(-2, -1)
        W_filtered = W_filtered_4d.permute(0, 2, 1, 3).contiguous().view(original_shape)

        return W_filtered


# ============================================================
# INLINED: softeq/layers/flinear.py
# ============================================================

class FLinear(nn.Module):
    """Filter based equivariant (or invariant) Linear Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        enforce_equivariance: bool = False,
        in_channels: int = None,
        out_channels: int = None,
        filter: nn.Module = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.enforce_equivariance = enforce_equivariance
        self.in_channels = in_channels if in_channels is not None else 1
        self.out_channels = out_channels if out_channels is not None else 1

        self.filter = filter

        if enforce_equivariance:
            weights_shape = (out_features * out_channels, in_features * in_channels)
        else:
            weights_shape = (out_features, in_features)

        self.weights = torch.nn.Parameter(torch.empty(weights_shape, device=device, dtype=dtype))
        if bias:
            if enforce_equivariance:
                self.bias = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
            else:
                self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        self._apply_filter()
        return nn.functional.linear(input, self.weights, self.bias)

    def _apply_filter(self):
        self.weights.data = self.filter(self.weights.data)


# ============================================================
# INLINED: utils/eq_nonlin.py
# ============================================================

class IdentityActivation(nn.Module):
    def forward(self, x):
        return x


class EQNonLin(nn.Module):
    """Norm-based equivariant nonlinearity."""

    def __init__(self, dim=2, nonlinearity='sigmoid', per_channel_bias=False, n_channels=1, layer_norm=False, vec_dim=None):
        super(EQNonLin, self).__init__()
        self.dim = dim
        self.nonlinearity = nonlinearity
        self.per_channel_bias = per_channel_bias
        self.n_channels = n_channels

        if nonlinearity == 'relu':
            self.act = nn.ReLU()
        elif nonlinearity == 'gelu':
            self.act = nn.GELU()
        elif nonlinearity == 'tanh':
            self.act = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.act = nn.Sigmoid()
        elif nonlinearity == 'identity':
            self.act = IdentityActivation()
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        if per_channel_bias and not layer_norm:
            self.bias = nn.Parameter(torch.zeros(n_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.dim() == 2:
            batch_size, total_dim = x.shape
            assert total_dim % self.dim == 0, "Input dimension mismatch"
            n_vectors = total_dim // self.dim
            x_reshaped = x.view(batch_size, n_vectors, self.dim)
            norms = torch.norm(x_reshaped, dim=-1, keepdim=True)

            if self.bias.numel() == 1:
                bias_tensor = self.bias.view(1, 1, 1)
            elif self.bias.numel() == n_vectors:
                bias_tensor = self.bias.view(1, n_vectors, 1)
            else:
                raise ValueError(
                    f"Bias channels ({self.bias.numel()}) do not match number of vectors ({n_vectors}). "
                    f"Set n_channels={n_vectors} or disable per_channel_bias."
                )
            norms = norms + bias_tensor
            activated_norms = self.act(norms)
            scaled_vectors = x_reshaped * activated_norms
            output = scaled_vectors.view(batch_size, total_dim)
            return output

        if x.dim() == 3:
            batch_size, seq_len, total_dim = x.shape
            assert total_dim % self.dim == 0, "Input dimension mismatch"
            n_vectors = total_dim // self.dim
            x_reshaped = x.view(batch_size, seq_len, n_vectors, self.dim)
            norms = torch.norm(x_reshaped, dim=-1, keepdim=True)

            if self.bias.numel() == 1:
                bias_tensor = self.bias.view(1, 1, 1, 1)
            elif self.bias.numel() == n_vectors:
                bias_tensor = self.bias.view(1, 1, n_vectors, 1)
            else:
                raise ValueError(
                    f"Bias channels ({self.bias.numel()}) do not match number of vectors ({n_vectors}). "
                    f"Set n_channels={n_vectors} or disable per_channel_bias."
                )

            norms = norms + bias_tensor
            activated_norms = self.act(norms)
            scaled_vectors = x_reshaped * activated_norms
            output = scaled_vectors.view(batch_size, seq_len, total_dim)
            return output

        raise ValueError("Unsupported input shape. Expected 2D (batch, n*dim) or 3D (batch, seq_len, n*dim)")


# ============================================================
# INLINED: utils/eq_layernorm.py
# ============================================================

class EQLayerNorm(nn.Module):
    """Equivariant layer norm for 2D/3D vectors."""

    def __init__(self, normalized_shape, dim=2, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None):
        super(EQLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.normalized_shape = normalized_shape
        assert normalized_shape[0] % dim == 0, "normalized_shape must be divisible by vector dimension"

    def forward(self, x):
        batch_size, seq_len, total_dim = x.shape
        assert total_dim % self.dim == 0, f"Input dimension {total_dim} must be divisible by vector dimension {self.dim}"
        n_vectors = total_dim // self.dim

        x_reshaped = x.view(batch_size, seq_len, n_vectors, self.dim)
        norms = torch.linalg.vector_norm(x_reshaped, dim=-1, keepdim=False)
        safe_norms = norms.unsqueeze(-1) + self.eps
        unit_vectors = x_reshaped / safe_norms
        output = unit_vectors.view(batch_size, seq_len, total_dim)
        return output


# ============================================================
# INLINED from notebooks/mlp_so3.ipynb
# ============================================================

class SO3Constraints(BaseConstraints):
    def __init__(self, n_vectors_in=1, n_vectors_out=1, decomposition_method='svd'):
        self.n_vectors_in = n_vectors_in
        self.n_vectors_out = n_vectors_out
        self.input_size = n_vectors_in * 3
        self.output_size = n_vectors_out * 3
        self.decomposition_method = decomposition_method
        self.create_unit_group_action_matrix()
        self.create_unit_group_action_forward_difference_matrix()

    def create_unit_group_action_matrix(self):
        L_x = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
        L_y = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=torch.float32)
        L_z = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)

        def block_diag(mat, n):
            if n == 1:
                return mat
            return torch.block_diag(*[mat for _ in range(n)])

        self.unit_group_action_matrices = [block_diag(m, self.n_vectors_in) for m in [L_x, L_y, L_z]]
        self.unit_group_action_matrix = self.unit_group_action_matrices[2]

        if self.output_size is not None:
            self.unit_group_action_matrices_out = [block_diag(m, self.n_vectors_out) for m in [L_x, L_y, L_z]]
            self.unit_group_action_matrix_out = self.unit_group_action_matrices_out[2]
        else:
            self.unit_group_action_matrices_out = None
            self.unit_group_action_matrix_out = None

    def create_unit_group_action_forward_difference_matrix(self):
        self.forward_difference_matrices = self.unit_group_action_matrices
        self.forward_difference_matrix = self.unit_group_action_matrix
        if self.output_size is not None:
            self.forward_difference_matrices_out = self.unit_group_action_matrices_out
            self.forward_difference_matrix_out = self.unit_group_action_matrix_out
        else:
            self.forward_difference_matrices_out = None
            self.forward_difference_matrix_out = None

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
            decompositions.append((left, s_values, right))
        return decompositions

    def get_equivariant_condition_matrix(self, diagonal_only=True):
        metrices = []
        if diagonal_only:
            for i, matrix in enumerate(self.forward_difference_matrices):
                out_matrix = self.forward_difference_matrices_out[i]
                d = matrix.shape[0]
                eye_in = torch.eye(d, device=matrix.device, dtype=matrix.dtype)
                eye_out = torch.eye(self.output_size, device=matrix.device, dtype=matrix.dtype)
                condition = torch.kron(matrix.T.contiguous(), eye_out) - torch.kron(eye_in, out_matrix.contiguous())
                metrices.append(condition)
        return metrices


def get_so3_invariant_filter(n_vectors_in=5, soft_threshold=0.1):
    so3 = SO3Constraints(n_vectors_in=n_vectors_in)
    stacked_h = torch.cat(so3.forward_difference_matrices, dim=1)
    stacked_v = torch.cat(so3.forward_difference_matrices, dim=0)
    U_l, S_l, _ = svd_decomposition(stacked_h)
    _, S_r, V_r = svd_decomposition(stacked_v)
    min_len = min(len(S_l), len(S_r))
    vals = S_l[:min_len] + S_r[:min_len]
    return InvariantProjector(U_l, V_r, vals, softness=soft_threshold, hard_mask=True, hard=True, debug=False)


def get_so3_equivariant_filter(n_vectors_in=5, n_vectors_out=1, soft_threshold=0.0):
    st = float(soft_threshold)
    constraints = SO3Constraints(n_vectors_in=n_vectors_in, n_vectors_out=n_vectors_out)
    condition_list = constraints.get_equivariant_condition_matrix(diagonal_only=True)
    stacked_condition = torch.cat(condition_list, dim=0)
    U, S, V = svd_decomposition(stacked_condition)
    inv_filter = InvariantProjector(V, V, S[: V.shape[1]], softness=st, hard_mask=True, hard=True, debug=False)
    return MultiGeneratorEquivariantProjectorviaSVD(inv_filter, constraints.input_size, constraints.output_size, st)


def create_so3_equivariant_layer(n_vectors_in=5, n_vectors_out=1, soft_threshold=0.0):
    filt = get_so3_equivariant_filter(n_vectors_in, n_vectors_out, soft_threshold)
    return FLinear(
        n_vectors_in * 3,
        n_vectors_out * 3,
        bias=False,
        enforce_equivariance=True,
        in_channels=1,
        out_channels=1,
        filter=filt,
    )


class DeepSO3InvariantModel(nn.Module):
    def __init__(self, n_vectors_in=5, hidden_vectors=[16, 16, 16], hidden_scalars=32, soft_threshold=0.0):
        """
        Multi-layer SO(3) invariant network.
        Sequence:
        Input
          -> [Equivariant Layer -> EQLayerNorm -> EQNonLin] x len(hidden_vectors)
          -> Invariant Layer -> ReLU
          -> Linear -> Scalar Output
        """
        super().__init__()
        self.n_vectors_in = n_vectors_in
        self.soft_threshold = soft_threshold

        self.eq_layers = nn.ModuleList()
        self.eq_norms = nn.ModuleList()
        self.eq_acts = nn.ModuleList()

        curr_vecs = n_vectors_in
        for out_vecs in hidden_vectors:
            layer = create_so3_equivariant_layer(
                n_vectors_in=curr_vecs,
                n_vectors_out=out_vecs,
                soft_threshold=soft_threshold,
            )
            self.eq_layers.append(layer)

            norm = EQLayerNorm([out_vecs * 3], dim=3)
            self.eq_norms.append(norm)

            act = EQNonLin(dim=3, nonlinearity='relu', per_channel_bias=True, n_channels=out_vecs)
            self.eq_acts.append(act)

            curr_vecs = out_vecs

        inv_filter = get_so3_invariant_filter(n_vectors_in=curr_vecs, soft_threshold=soft_threshold)
        self.inv_layer = FLinear(curr_vecs * 3, hidden_scalars, filter=inv_filter)
        self.inv_act = nn.ReLU()

        self.final_layer = nn.Linear(hidden_scalars, 1)

    def forward(self, x, return_stats=False):
        stats = {}
        if return_stats:
            stats['Input'] = {'mean': x.mean().item(), 'var': x.var().item(), 'std': x.std().item()}

        for i, (layer, norm, act) in enumerate(zip(self.eq_layers, self.eq_norms, self.eq_acts)):
            x = layer(x)
            if return_stats:
                stats[f'Eq_Linear_{i + 1}'] = {'mean': x.mean().item(), 'var': x.var().item(), 'std': x.std().item()}

            x_unsq = x.unsqueeze(1)
            x_normed = norm(x_unsq)
            x = x_normed.squeeze(1)
            if return_stats:
                stats[f'Eq_Norm_{i + 1}'] = {'mean': x.mean().item(), 'var': x.var().item(), 'std': x.std().item()}

            x = act(x)
            if return_stats:
                stats[f'Eq_Act_{i + 1}'] = {'mean': x.mean().item(), 'var': x.var().item(), 'std': x.std().item()}

        x = self.inv_layer(x)
        if return_stats:
            stats['Inv_Linear'] = {'mean': x.mean().item(), 'var': x.var().item(), 'std': x.std().item()}

        x = self.inv_act(x)
        if return_stats:
            stats['Inv_Act'] = {'mean': x.mean().item(), 'var': x.var().item(), 'std': x.std().item()}

        out = self.final_layer(x)
        if return_stats:
            stats['Final_Output'] = {'mean': out.mean().item(), 'var': out.var().item(), 'std': out.std().item()}
            return out, stats

        return out


# ============================================================
# Demo (no plotting)
# ============================================================

def get_rodrigues_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return torch.tensor(R, dtype=torch.float32)


def generate_vector_set(n_vectors=5, batch_size=1):
    phi = np.random.uniform(0, 2 * np.pi, (batch_size, n_vectors))
    theta = np.arccos(np.random.uniform(-1, 1, (batch_size, n_vectors)))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return torch.tensor(np.stack([x, y, z], axis=-1), dtype=torch.float32)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("SO(3) Deep Invariant MLP — Standalone Demo")
    print("=" * 60)

    print("Building Deep SO(3) soft-invariant model...")
    deep_model = DeepSO3InvariantModel(n_vectors_in=5, hidden_vectors=[16, 16, 16], soft_threshold=0.7)
    deep_model.eval()
    print("Model Architecture Built Successfully!\n---")
    print(deep_model)

    bs = 100
    test_pts = generate_vector_set(n_vectors=5, batch_size=bs)
    test_pts_flat = test_pts.view(bs, 15)

    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    R_test = get_rodrigues_rotation(axis, angle)
    rot_test_pts = (test_pts @ R_test.T).contiguous().view(bs, 15)

    out_original, layer_stats = deep_model(test_pts_flat, return_stats=True)
    out_rotated = deep_model(rot_test_pts)

    diff = (out_original - out_rotated).abs().max().item()
    print(f"\n[Test] Maximum difference between Deep f(x) and f(Rx) across {bs} batches: {diff:.6e}")

    print("\n" + "=" * 52)
    print("  HIDDEN FEATURE STATISTICS (Mean, Variance, Std)  ")
    print("=" * 52)
    print(f"{'Layer Name':<16} | {'Mean':>9} | {'Variance':>9} | {'Std Dev':>9}")
    print("-" * 52)
    for layer_name, stat in layer_stats.items():
        print(f"{layer_name:<16} | {stat['mean']:9.4f} | {stat['var']:9.4f} | {stat['std']:9.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()

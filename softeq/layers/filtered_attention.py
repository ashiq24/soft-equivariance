"""
Filtered Multi-Head Attention with Rotation Equivariance.

This module provides a multi-head attention layer that applies equivariant filters
to Q, K, V projection matrices to preserve rotation equivariance for vector inputs.
"""

import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor


class FilteredMultiheadAttention(nn.MultiheadAttention):
    """
    Multi-head attention with equivariant filtering on Q, K, V projections.
    
    Inherits from nn.MultiheadAttention to reuse forward logic.
    Applies filter_eq to q_proj_weight, k_proj_weight, v_proj_weight before
    computing attention.
    
    This enables rotation-equivariant self-attention for vector-valued inputs
    where each embedding is a concatenation of multiple 2D or 3D vectors.
    
    Args:
        embed_dim: Total dimension of the model (must equal num_channels * vector_dim)
        num_heads: Number of parallel attention heads
        filter_eq: Equivariant filter module to apply to Q, K, V weights
        filter_inv: Invariant filter for biases (stored but not used for now)
        dropout: Dropout probability on attention weights. Default: 0.0
        bias: Whether to add bias to input/output projections. Default: True
        add_bias_kv: Adds bias to key and value sequences. Default: False
        add_zero_attn: Adds new batch of zeros to key and value. Default: False
        kdim: Total number of features for keys. Default: None (uses embed_dim)
        vdim: Total number of features for values. Default: None (uses embed_dim)
        batch_first: If True, input/output are (batch, seq, feature). Default: False
        device: Device for parameters. Default: None
        dtype: Data type for parameters. Default: None
    
    Example:
        >>> # For 3 channels of 2D vectors: embed_dim = 3 * 2 = 6
        >>> filter_eq = get_equivariant_filter_rotation(
        ...     n_rotations=4, input_size=2, output_size=2, vector=True
        ... )
        >>> attn = FilteredMultiheadAttention(
        ...     embed_dim=6, num_heads=2, filter_eq=filter_eq
        ... )
        >>> seq = torch.randn(10, 4, 6)  # (seq_len, batch, embed_dim)
        >>> output, weights = attn(seq, seq, seq)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        filter_eq: nn.Module,
        filter_inv: Optional[nn.Module] = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize filtered multi-head attention."""
        # Initialize parent class
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        
        # Store filter modules
        self.filter_eq = filter_eq
        self.filter_inv = filter_inv
    
    def _apply_filters(self) -> None:
        """
        Apply equivariant filter to Q, K, V projection weights.
        
        Handles both fused in_proj_weight (for Q=K=V dimensions) and
        separate q_proj_weight, k_proj_weight, v_proj_weight.
        
        Also applies filter to output projection.
        """
        if self.in_proj_weight is not None:
            # Fused QKV projection: shape (3 * embed_dim, embed_dim)
            # Split into Q, K, V components
            embed_dim = self.embed_dim
            
            # Extract Q, K, V weight matrices
            q_weight = self.in_proj_weight[:embed_dim, :]
            k_weight = self.in_proj_weight[embed_dim:2*embed_dim, :]
            v_weight = self.in_proj_weight[2*embed_dim:, :]
            
            # Apply equivariant filter to each
            q_filtered = self.filter_eq(q_weight)
            k_filtered = self.filter_eq(k_weight)
            v_filtered = self.filter_eq(v_weight)
            
            # Reassemble fused weight matrix
            self.in_proj_weight.data = torch.cat([
                q_filtered, k_filtered, v_filtered
            ], dim=0)
        else:
            # Separate Q, K, V projections
            if self.q_proj_weight is not None:
                self.q_proj_weight.data = self.filter_eq(self.q_proj_weight.data)
            if self.k_proj_weight is not None:
                self.k_proj_weight.data = self.filter_eq(self.k_proj_weight.data)
            if self.v_proj_weight is not None:
                self.v_proj_weight.data = self.filter_eq(self.v_proj_weight.data)
        
        # smooth the bias
        if self.in_proj_bias is not None and self.filter_inv is not None:
            self.in_proj_bias.data = self.filter_inv.smooth(self.in_proj_bias.data)
            
        if self.bias_k is not None and self.filter_inv is not None:
            self.bias_k.data = self.filter_inv.smooth(self.bias_k.data)
        if self.bias_v is not None and self.filter_inv is not None:
            self.bias_v.data = self.filter_inv.smooth(self.bias_v.data)
        # Apply filter to output projection
        if hasattr(self, 'out_proj') and self.out_proj is not None:
            if hasattr(self.out_proj, 'weight'):
                self.out_proj.weight.data = self.filter_eq(self.out_proj.weight.data)
            # also filter output bias
            if self.out_proj.bias is not None and self.filter_inv is not None:
                self.out_proj.bias.data = self.filter_inv.smooth(self.out_proj.bias.data)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with filtered projections.
        
        Applies equivariant filters to Q, K, V weights before computing attention.
        
        Args:
            query: Query embeddings
                - Shape: (L, E) for unbatched, (L, N, E) or (N, L, E) for batched
                - L: target sequence length, N: batch size, E: embed_dim
            key: Key embeddings
                - Shape: (S, E) for unbatched, (S, N, E) or (N, S, E) for batched
                - S: source sequence length
            value: Value embeddings
                - Shape: (S, E) for unbatched, (S, N, E) or (N, S, E) for batched
            key_padding_mask: Mask for key padding
                - Shape: (N, S) or (S,) for unbatched
                - True indicates padding elements to ignore
            need_weights: Return attention weights. Default: True
            attn_mask: Attention mask
                - Shape: (L, S) or (N*num_heads, L, S)
                - Prevents attention to certain positions
            average_attn_weights: Average weights across heads. Default: True
            is_causal: Use causal mask. Default: False
        
        Returns:
            attn_output: Attention output
                - Shape matches input query shape
            attn_weights: Attention weights (if need_weights=True)
                - Shape: (N, L, S) if averaged, else (N, num_heads, L, S)
        """
        # Apply equivariant filters to projection weights
        self._apply_filters()
        
        # Call parent's forward method with all attention logic
        return super().forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

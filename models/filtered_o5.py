"""
Filtered Equivariant MLP for O(5) group.

This module provides a simple and efficient MLP architecture that respects
O(5) group equivariance using soft-equivariance filter projections.

Architecture:
    Input -> FLinear (O(5) equivariant) -> ReLU -> ... -> FLinear -> Output

Each FLinear layer projects weights to the O(5)-equivariant subspace.
"""

import sys
import os
import logging
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from softeq.equi_utils.o5_filter import get_invariant_filter_o5, get_equivariant_filter_o5
from utils.eq_nonlin import EQNonLin

# Try importing EMLP from installed package, fall back to external/ if not available
try:
    from emlp.nn.pytorch import EMLP, MLP, Standardize
    from emlp.reps import Scalar, Vector, T
    from emlp.groups import O
    from emlp.nn.pytorch import GatedNonlinearity
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'equivariant-MLP'))
    from emlp.nn.pytorch import EMLP, MLP, Standardize
    from emlp.reps import Scalar, Vector, T
    from emlp.groups import O
    from emlp.nn.pytorch import GatedNonlinearity
from softeq.layers.flinear import FLinear, FilteredLinear

logger = logging.getLogger(__name__)


class FilteredO5MLP(nn.Module):
    """
    Equivariant MLP for O(5) with per-layer soft-equivariant projection. This MLP is meant for invariant tasks so the last layer maps to scalars, but the hidden layers can use vector or tensor representations. 
    
    Architecture:
        Input -> [FilteredLinear (equivariant) -> EQNonLin]* -> Linear/FilteredLinear (invariant) -> Output
    
    Each linear map is projected by O(5)-compatible filters at forward time.
    
    Args:
        in_features: Number of input channels in the chosen input representation.
        out_features: Number of output channels in the chosen output representation.
        hidden_features: Hidden channel counts used to build intermediate layers.
        filter_config: Dictionary controlling filter construction and softness.
        in_rep: Input representation code (for example "V" or "S").
        out_rep: Output representation code.
        use_tensor_hidden: If True, uses the tensor representation T(2) in the hidden layers (V->V->T->...->S); else V->V->...->S.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int],
        filter_config: Optional[Dict[str, Any]] = None,
        in_rep: str = "V",
        out_rep: str = "S", # currently only supports invariant outputs - i.e., out_rep = "S". Need to update "layer_specs" construction in __init__ to support equivariant outputs.
        use_tensor_hidden: bool = True,
    ):
        super().__init__()
        
        if use_tensor_hidden and len(hidden_features) < 2:
            raise ValueError("hidden_features must have at least two entries for tensor architecture (V→V→T→T...→S)")

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.filter_config = filter_config or {}
        self.in_rep = in_rep
        self.out_rep = out_rep
        self.use_tensor_hidden = use_tensor_hidden

        # Representation dimensions
        self.group = get_emlp_group("O(5)")
        self.vector_rep = Vector(self.group)
        self.scalar_rep = Scalar(self.group)

        # Resolve representation dimensions; handle both attribute and callable .size
        def resolve_dim(rep_obj, default):
            attr = getattr(rep_obj, "size", default)
            return attr() if callable(attr) else attr

        self.vec_dim = resolve_dim(self.vector_rep, 5)
        self.tensor_dim = self.vec_dim * self.vec_dim  # Flattened 5x5 tensor representation for O(5)

        def rep_dim(rep: str) -> int:
            """Return dimension for representation code using EMLP."""
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from utils.representation_utils import get_representation_size
            
            try:
                return get_representation_size(rep, group_name='O(5)')
            except Exception as e:
                raise ValueError(f"Unsupported representation '{rep}' for O(5): {e}")
        
        def rep_to_emlp_string(rep: str) -> str:
            """Pass through representation string - EMLP native parsing handles all notation."""
            return rep
        
        # Build layer stack
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        cfg = self.filter_config
        soft_thresh = cfg.get('soft_thresholding', 0.0)
        use_reflection = cfg.get('use_reflection', True)
        hard = cfg.get('hard', True)
        hard_mask = cfg.get('hard_mask', True)
        debug = cfg.get('debug', False)
        decomp_method = cfg.get('decomposition_method', 'svd')
        use_inv_filter = cfg.get('use_invariant_filter', False)

        # Layer specification: (in_rep, out_rep, in_channels, out_channels)
        layer_specs = []
        if use_tensor_hidden:
            # Tensor architecture: V→V→T→T...→S
            layer_specs.append(("V", "V", in_features, hidden_features[0]))
            layer_specs.append(("V", "T", hidden_features[0], hidden_features[1]))
            for idx in range(1, len(hidden_features) - 1):
                layer_specs.append(("T", "T", hidden_features[idx], hidden_features[idx + 1]))
            layer_specs.append(("T", "S", hidden_features[-1], hidden_features[-1]*out_features))
            layer_specs.append(("S", "S", hidden_features[-1]*out_features, out_features))
        else:
            # Vector architecture: V→V→V...→S
            layer_specs.append(("V", "V", in_features, hidden_features[0]))
            for idx in range(len(hidden_features) - 1):
                layer_specs.append(("V", "V", hidden_features[idx], hidden_features[idx + 1]))
            layer_specs.append(("V", "S", hidden_features[-1], hidden_features[-1]*out_features))
            layer_specs.append(("S", "S", hidden_features[-1]*out_features, out_features))

        for in_rep_l, out_rep_l, in_ch, out_ch in layer_specs:
            in_dim = in_ch * rep_dim(in_rep_l)
            out_dim = out_ch * rep_dim(out_rep_l)

            # Select filter
            if in_rep_l == "S" and out_rep_l == "S":
                # Both scalar: no filter needed
                flayer = nn.Linear(in_dim, out_dim)
                non_lin = nn.Identity()
                drop_out = nn.Identity()
            elif out_rep_l == "S" and in_rep_l in ["V", "T"]:
                filt = get_invariant_filter_o5(
                    input_size=rep_dim(in_rep_l),
                    soft_threshold=soft_thresh,
                    decomposition_method=decomp_method,
                    debug=debug,
                    use_reflection=use_reflection,
                    hard=hard,
                    hard_mask=hard_mask,
                    in_rep=rep_to_emlp_string(in_rep_l),
                )
                flayer = FilteredLinear(nn.Linear(in_dim, out_dim), filter_eq=filt, filter_inv=None)
                non_lin = nn.ReLU()
                drop_out = nn.Dropout(p=0.0)
            else:
                filt_eq = get_equivariant_filter_o5(
                    input_size=rep_dim(in_rep_l),
                    output_size=rep_dim(out_rep_l),
                    soft_threshold=soft_thresh,
                    debug=debug,
                    use_reflection=use_reflection,
                    use_invariant_filter=use_inv_filter,
                    hard=hard,
                    hard_mask=hard_mask,
                    in_rep=rep_to_emlp_string(in_rep_l),
                    out_rep=rep_to_emlp_string(out_rep_l),
                )
                # Get invariant filter with input rep for filter_inv
                filt_inv = get_invariant_filter_o5(
                    input_size=rep_dim(out_rep_l),
                    soft_threshold=soft_thresh,
                    decomposition_method=decomp_method,
                    debug=debug,
                    use_reflection=use_reflection,
                    hard=hard,
                    hard_mask=hard_mask,
                    in_rep=rep_to_emlp_string(out_rep_l),
                )
                flayer = FilteredLinear(nn.Linear(in_dim, out_dim), filter_eq=filt_eq, filter_inv=filt_inv)
                non_lin = EQNonLin(dim=rep_dim(out_rep_l), nonlinearity='relu', per_channel_bias=True, n_channels=out_ch)
                drop_out = nn.Dropout(p=0.0)

            self.layers.append(flayer)
            self.activations.append(non_lin)
            # Keep dropout disabled unless the architecture is extended to use it explicitly.
            self.dropouts.append(drop_out)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        
        # Pass through layers with activations and dropouts
        for layer, activation, dropout in zip(self.layers, self.activations, self.dropouts):
            x = layer(x)
            x = activation(x)
            x = dropout(x)
            
        
        return x


def get_emlp_group(group_name: str = "O(5)") -> Any:
    """
    Get EMLP O(5) group instance.
    
    Args:
        group_name: Group name (default: "O(5)")
    
    Returns:
        EMLP O(5) group instance
    
    Raises:
        ValueError: If group not supported
    """
    try:
        from emlp import groups as emlp_groups
    except ImportError:
        raise ImportError("EMLP library not available")
    
    # Parse group name
    group_name = group_name.strip().upper()
    
    # For O(5)
    if group_name == "O(5)":
        return emlp_groups.O(5)
    
    raise ValueError(f"Unsupported group: {group_name}. Only O(5) is supported.")


def create_filtered_o5_mlp(model_config: Dict[str, Any]) -> nn.Module:
    """
    Create a filtered equivariant MLP for O(5) group.
    
    Args:
        model_config: Flat configuration dictionary with keys such as:
            - in_features: Input channel count for the chosen representation.
            - out_features: Output channel count.
            - hidden_features: List of hidden layer channel counts.
            - use_tensor_hidden: If True, uses the tensor path V->V->T->...->S.
            - in_rep: Input representation code.
            - out_rep: Output representation code.
            - soft_thresholding, hard_mask, use_reflection, decomposition_method: filter options.
    
    Returns:
        FilteredO5MLP instance
    
    Example:
        >>> config = {
        ...     'model': {
        ...         'in_features': 2,
        ...         'out_features': 1,
        ...         'hidden_features': [64, 64],
        ...     }
        ... }
        >>> model = create_filtered_o5_mlp(config)
    """
    
    in_features = model_config.get('in_features', 10)
    out_features = model_config.get('out_features', 1)
    hidden_features = model_config.get('hidden_features', [64, 64])
    in_rep = model_config.get('in_rep', 'V')
    out_rep = model_config.get('out_rep', 'S')
    use_tensor_hidden = model_config.get('use_tensor_hidden', True)
    filter_config = model_config
    
    # Create model
    model = FilteredO5MLP(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        filter_config=filter_config,
        in_rep=in_rep,
        out_rep=out_rep,
        use_tensor_hidden=use_tensor_hidden,
    )
    
    arch_type = "tensor (V→V→T→T...→S)" if use_tensor_hidden else "vector (V→V→V...→S)"
    logger.info(f"Created FilteredO5MLP [{arch_type}]: {in_features} -> {hidden_features} -> {out_features}")
    
    return model


__all__ = [
    'FilteredO5MLP',
    'create_filtered_o5_mlp',
    'get_emlp_group',
]

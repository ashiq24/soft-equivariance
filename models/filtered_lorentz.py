"""
Filtered Equivariant MLP for Lorentz (O(1,3)) group.

Uses soft-equivariant filters to project weights at each forward pass.
"""
import os
import sys
import logging
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn

from softeq.equi_utils.lorentz_filter import (
    get_invariant_filter_lorentz,
    get_equivariant_filter_lorentz,
)
from utils.eq_nonlin import EQNonLin
from softeq.layers.flinear import FilteredLinear

# Try importing EMLP from installed package, fall back to external/ if not available
try:
    from emlp.reps import Scalar, Vector
    from emlp.groups import Lorentz
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'equivariant-MLP'))
    try:
        from emlp.reps import Scalar, Vector
    except Exception:
        Scalar = None
        Vector = None
    from emlp.groups import Lorentz

logger = logging.getLogger(__name__)


class FilteredLorentzMLP(nn.Module):
    """
    Equivariant MLP for Lorentz (O(1,3)) group with filter-based weight projection.

    Architecture:
        Input -> [FilteredLinear -> EQNonLin]* -> FilteredLinear -> Output
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int],
        filter_config: Optional[Dict[str, Any]] = None,
        in_rep: str = "V",
        out_rep: str = "S",
        hidden_reps: Optional[List[str]] = None,
        use_tensor_hidden: bool = True,  # Deprecated, kept for backward compatibility
    ):
        """
        Initialize FilteredLorentzMLP with flexible representation structure.
        
        Args:
            in_features: Number of input channels
            out_features: Number of output channels
            hidden_features: List of channel counts for hidden layers
            filter_config: Configuration for equivariance filters
            in_rep: Input representation (e.g., 'V', 'V*', 'V⊗V*', 'T', 'S')
            out_rep: Output representation (e.g., 'V', 'V*', 'S')
            hidden_reps: List of representations for hidden layers.
                If None, uses legacy use_tensor_hidden behavior.
                Examples: ['V', 'V*', 'V⊗V*'], ['T', 'T', 'V'], ['V*', '(V*)²']
            use_tensor_hidden: (Deprecated) If True and hidden_reps=None, uses V->T->T...
                If False and hidden_reps=None, uses V->V->V...
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.filter_config = filter_config or {}
        self.in_rep = in_rep
        self.out_rep = out_rep
        self.use_tensor_hidden = use_tensor_hidden

        # Import representation utilities for getting sizes
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from utils.representation_utils import get_representation_size

        self.group = get_emlp_group("Lorentz")
        
        # Helper function to get representation dimension from EMLP
        def rep_dim(rep: str) -> int:
            """Get dimension of a representation using EMLP's size() method."""
            try:
                return get_representation_size(rep, group_name='Lorentz')
            except Exception as e:
                raise ValueError(f"Unsupported representation '{rep}' for Lorentz group: {e}")

        def rep_to_emlp_string(rep: str) -> str:
            """Pass through representation string - EMLP native parsing handles all notation."""
            return rep

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
        use_inv_filter = cfg.get('use_invariant_filter', True)

        # Build layer specifications based on hidden_reps or legacy behavior
        layer_specs = []
        
        if hidden_reps is not None:
            # New flexible representation path
            # Build: in_rep -> hidden_reps[0] -> hidden_reps[1] -> ... -> out_rep
            all_reps = [in_rep] + list(hidden_reps) + [out_rep]
            all_channels = [in_features] + list(hidden_features) + [out_features]
            
            for i in range(len(all_reps) - 1):
                in_rep_l = all_reps[i]
                out_rep_l = all_reps[i + 1]
                in_ch = all_channels[i]
                out_ch = all_channels[i + 1]
                layer_specs.append((in_rep_l, out_rep_l, in_ch, out_ch))
                
        else:
            raise ValueError("hidden_reps must be specified for FilteredLorentzMLP. "
                             "The use_tensor_hidden flag is deprecated.")

        for in_rep_l, out_rep_l, in_ch, out_ch in layer_specs:
            in_dim = in_ch * rep_dim(in_rep_l)
            out_dim = out_ch * rep_dim(out_rep_l)

            if in_rep_l == "S" and out_rep_l == "S":
                flayer = nn.Linear(in_dim, out_dim)
                non_lin = nn.Identity()
                drop_out = nn.Identity()
            elif out_rep_l == "S" and in_rep_l != "S":
                filt = get_invariant_filter_lorentz(
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
                if in_rep_l == out_rep_l and rep_dim(in_rep_l) > 64 :
                    print("[Filter_Log] Using invariant filter disabled for large same-rep layer")
                    use_inv_filter = False
                else: 
                    use_inv_filter = True
                    
                filt_eq = get_equivariant_filter_lorentz(
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
                filt_inv = get_invariant_filter_lorentz(
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
                non_lin = EQNonLin(dim=rep_dim(out_rep_l), nonlinearity='tanh', layer_norm=True, n_channels=out_ch)
                drop_out = nn.Dropout(p=0.0)

            self.layers.append(flayer)
            self.activations.append(non_lin)
            self.dropouts.append(drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, activation, dropout in zip(self.layers, self.activations, self.dropouts):
            x = layer(x)
            x = activation(x)
            x = dropout(x)
        return x


def get_emlp_group(group_name: str = "Lorentz"):
    group_name = group_name.strip().lower()
    if group_name in ["lorentz", "o(1,3)", "o13"]:
        return Lorentz()
    raise ValueError(f"Unsupported group: {group_name}. Only Lorentz is supported.")


def create_filtered_lorentz_mlp(model_config: Dict[str, Any]) -> nn.Module:
    """Create a FilteredLorentzMLP from configuration dictionary.
    
    Args:
        model_config: Configuration dictionary with keys:
            - in_rep: Input representation (default: 'V')
            - out_rep: Output representation (default: 'S')
            - hidden_reps: List of hidden representations (optional)
            - in_features: Number of input channels (optional, computed from input_dim)
            - out_features: Number of output channels (default: 1)
            - hidden_features: List of hidden channel counts (default: [64, 64])
            - use_tensor_hidden: (Deprecated) Use tensor path if hidden_reps not specified
            - input_dim: Total input dimension (optional)
            + filter configuration options
    
    Examples:
        # Using hidden_reps (recommended)
        config = {
            'in_rep': 'V',
            'hidden_reps': ['V*', 'V⊗V*', 'V'],
            'out_rep': 'S',
            'hidden_features': [16, 32, 16]
        }
        
        # Legacy tensor path
        config = {
            'in_rep': 'V',
            'out_rep': 'S',
            'use_tensor_hidden': True,
            'hidden_features': [64, 64]
        }
    """
    from utils.representation_utils import get_representation_size
    
    in_rep = model_config.get('in_rep', 'V')
    input_dim = model_config.get('input_dim', model_config.get('dim', None))
    hidden_reps = model_config.get('hidden_reps', None)
    
    # Get representation dimension from EMLP (never hardcode)
    try:
        rep_dim = get_representation_size(in_rep, group_name='Lorentz')
    except Exception as e:
        raise ValueError(f"Unsupported in_rep '{in_rep}' for Lorentz: {e}")

    if input_dim is not None:
        if input_dim % rep_dim != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by rep_dim ({rep_dim}) for in_rep='{in_rep}'"
            )
        in_features = input_dim // rep_dim
    else:
        in_features = model_config.get('in_features', 4)
    
    out_features = model_config.get('out_features', 1)
    hidden_features = model_config.get('hidden_features', [64, 64])
    out_rep = model_config.get('out_rep', 'S')
    use_tensor_hidden = model_config.get('use_tensor_hidden', False)
    filter_config = model_config

    model = FilteredLorentzMLP(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        filter_config=filter_config,
        in_rep=in_rep,
        out_rep=out_rep,
        hidden_reps=hidden_reps,
        use_tensor_hidden=use_tensor_hidden,
    )
    
    # Log architecture
    if hidden_reps is not None:
        rep_path = f"{in_rep} → {' → '.join(hidden_reps)} → {out_rep}"
        logger.info(f"Created FilteredLorentzMLP [custom reps]: {rep_path}")
        logger.info(f"  Channels: {in_features} → {hidden_features} → {out_features}")
    else:
        arch_type = "tensor (V→T→T...→S)" if use_tensor_hidden else "vector (V→V→V...→S)"
        logger.info(f"Created FilteredLorentzMLP [{arch_type}]: {in_features} → {hidden_features} → {out_features}")
    
    return model


__all__ = [
    'FilteredLorentzMLP',
    'create_filtered_lorentz_mlp',
    'get_emlp_group',
]

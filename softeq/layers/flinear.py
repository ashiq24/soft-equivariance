import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import math
from typing import Optional
import einops
from softeq.utils.misc import apply_filter_blockwise, project_fast_batched
from softeq.utils.misc import extract_blocks, exact_equivarinace_projection

class FLinear(nn.Module):
    """
    Filter based equivariant (or invariant) Linear Layer.
    """
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
                # Shared scalar bias for the equivariant case. For vector data bias need to be False
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
        

class FilteredLinear(nn.Module):
    """
    Wrapper that filters a Linear layer before applying it.

    The weight matrix is passed through the provided equivariant filter and
    then restored to the original shape before the linear transform is applied.
    
    Args:
        original_layer: Pretrained nn.Linear layer
        filter: Filter module (InvariantProjector or EquivariantProjector)
    """
    def __init__(self, original_layer: nn.Linear, filter_eq: nn.Module, filter_inv: nn.Module = None):
        super().__init__()
        # Copy linear parameters
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Copy pretrained weights as parameters
        self.weight = nn.Parameter(original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        
        # Store filter module
        self.filter_eq = filter_eq
        self.filter_inv = filter_inv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply filter to linear weights and perform linear transformation.
        
        Args:
            x: Input tensor (batch, in_features)
        
        Returns:
            Output tensor after filtered linear transformation
        """
        # Reshape linear weight to 2D for filtering
        # Original shape: (out_features, in_features)
        # Flatten to: (out_features, in_features)
        original_shape = self.weight.shape
        weight_2d = self.weight.view(original_shape[0], -1)
        
        # Apply filter
        filtered_weight_2d = self.filter_eq(weight_2d)
        
        # smooth the bias
        if self.bias is not None and self.filter_inv is not None:
            self.bias.data = self.filter_inv.smooth(self.bias.data)
        # Reshape back to linear weight shape
        filtered_weight = filtered_weight_2d.view(original_shape)
        
        return nn.functional.linear(x, filtered_weight, self.bias)
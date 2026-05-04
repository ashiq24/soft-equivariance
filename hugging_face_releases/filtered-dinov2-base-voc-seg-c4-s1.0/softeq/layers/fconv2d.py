import torch
import torch.nn as nn
import torch.nn.functional as F


class FilteredConv2d(nn.Module):
    """
    Wrapper that filters a Conv2d kernel before applying convolution.
    
    The convolution weights are flattened to a 2D matrix, passed through the
    provided filter module, and reshaped back to the original kernel layout.
    
    Args:
        original_layer: Pretrained nn.Conv2d layer
        filter: Filter module (InvariantProjector or EquivariantProjector)
    """
    def __init__(self, original_layer: nn.Conv2d, filter: nn.Module):
        super().__init__()
        # Copy conv parameters
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        
        # Copy pretrained weights as parameters
        self.weight = nn.Parameter(original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        
        # Store filter module
        self.filter = filter
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply filter to conv weights and perform convolution.
        
        Args:
            x: Input tensor (batch, in_channels, height, width)
        
        Returns:
            Output tensor after filtered convolution
        """
        # Reshape conv weight to 2D for filtering
        # Original shape: (out_channels, in_channels, kernel_h, kernel_w)
        # Flatten to: (out_channels, in_channels * kernel_h * kernel_w)
        original_shape = self.weight.shape
        weight_2d = self.weight.view(original_shape[0], -1)
        
        # Apply filter
        filtered_weight_2d = self.filter(weight_2d)
        
        # Reshape back to conv weight shape
        filtered_weight = filtered_weight_2d.view(original_shape)
        # Perform convolution with filtered weights
        return F.conv2d(
            x, filtered_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )
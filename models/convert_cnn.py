"""
Generic CNN conversion utility for soft equivariance filtering.

This module provides functions to convert any CNN model by replacing
Conv2d layers with FilteredConv2d layers. The conversion applies
soft equivariance constraints to enforce rotation/reflection invariance.

Usage:
    model = load_your_cnn_model()
    filter_config = {
        'n_rotations': 4,
        'soft_thresholding': 0.1,
        'decomposition_method': 'svd',
        'group_type': 'rotation',
    }
    filtered_model = convert_cnn_to_filtered(model, filter_config)
"""

import torch
import torch.nn as nn
from softeq.layers.fconv2d import FilteredConv2d
from softeq.equi_utils.filter_factory import get_invariant_filter


def convert_cnn_to_filtered(model: nn.Module, filter_config: dict) -> nn.Module:
    """
    Convert a CNN model by replacing Conv2d layers with FilteredConv2d.
    
    This function recursively traverses the model and replaces all Conv2d layers
    that have kernel_size > 1 with FilteredConv2d layers. The 1x1 convolutions
    are preserved as they are typically used for dimension matching.
    
    Args:
        model: Any PyTorch CNN model (nn.Module)
        filter_config: Dictionary with filter configuration:
            - n_rotations (int): Number of discrete rotations (e.g., 4 for 90° rotations)
            - soft_thresholding (float): Softness parameter (0.0 = strict, 1.0 = no filtering)
            - decomposition_method (str): 'svd' or 'schur'
            - group_type (str): 'rotation' or 'roto_reflection'
            - hard_mask (bool, optional): Use hard mask for smoothing. Default: False
            - preserve_norm (bool, optional): Preserve weight norms after projection. Default: False
            - joint_decomposition (bool, optional): Use joint decomposition. Default: True
            - min_kernel_size (int, optional): Minimum kernel size to filter. Default: 2
            - skip_first_conv (bool, optional): Skip the first conv layer. Default: False
    
    Returns:
        The same model with Conv2d layers replaced by FilteredConv2d layers.
        Note: The model is modified in-place and also returned.
    
    Example:
        >>> import torchvision.models as models
        >>> model = models.resnet18(pretrained=True)
        >>> filter_config = {
        ...     'n_rotations': 4,
        ...     'soft_thresholding': 0.1,
        ...     'decomposition_method': 'svd',
        ...     'group_type': 'rotation',
        ... }
        >>> filtered_model = convert_cnn_to_filtered(model, filter_config)
    """
    # Extract configuration
    n_rotations = filter_config['n_rotations']
    soft_thresholding = filter_config['soft_thresholding']
    decomposition_method = filter_config.get('decomposition_method', 'svd')
    group_type = filter_config.get('group_type', 'rotation')
    hard_mask = filter_config.get('hard_mask', False)
    preserve_norm = filter_config.get('preserve_norm', False)
    joint_decomposition = filter_config.get('joint_decomposition', True)
    min_kernel_size = filter_config.get('min_kernel_size', 2)
    skip_first_conv = filter_config.get('skip_first_conv', False)
    
    # Skip filtering entirely if soft_thresholding >= 1.0
    if soft_thresholding >= 1.0:
        print("Soft thresholding >= 1.0, skipping filtering")
        return model
    
    # Track how many layers we've filtered
    filtered_count = [0]  # Use list to allow modification in nested function
    first_conv_seen = [False]
    
    def _replace_conv_in_module(parent_module, name, child_module):
        """Replace a single Conv2d with FilteredConv2d."""
        kernel_size = child_module.kernel_size[0]
        
        # Skip 1x1 convolutions and convolutions below min_kernel_size
        if kernel_size < min_kernel_size:
            return False
        
        # Optionally skip first conv
        if skip_first_conv and not first_conv_seen[0]:
            first_conv_seen[0] = True
            print(f"  Skipping first conv: {name}")
            return False
        
        first_conv_seen[0] = True
        
        # Verify square kernel (or use minimum dimension)
        if kernel_size != child_module.kernel_size[1]:
            print(f"  Warning: Non-square kernel at {name}: {child_module.kernel_size}")
        
        # Create invariant filter
        filter_module = get_invariant_filter(
            group_type=group_type,
            n_rotations=n_rotations,
            input_size=(1, kernel_size, kernel_size),
            soft_threshold=soft_thresholding,
            decomposition_method=decomposition_method,
            debug=False,
            hard_mask=hard_mask,
            preserve_norm=preserve_norm,
            joint_decomposition=joint_decomposition
        )
        
        # Create FilteredConv2d wrapper
        filtered_conv = FilteredConv2d(
            original_layer=child_module,
            filter=filter_module
        )
        
        # Replace the layer
        setattr(parent_module, name, filtered_conv)
        filtered_count[0] += 1
        
        in_c = child_module.in_channels
        out_c = child_module.out_channels
        print(f"  Replaced: Conv2d({in_c}, {out_c}, kernel={kernel_size})")
        return True
    
    def _recursive_replace(module, prefix=""):
        """Recursively traverse and replace Conv2d layers."""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Conv2d):
                _replace_conv_in_module(module, name, child)
            elif isinstance(child, FilteredConv2d):
                # Already filtered, skip
                pass
            else:
                # Recurse into child modules
                _recursive_replace(child, full_name)
    
    print(f"Converting CNN with filter config:")
    print(f"  group_type={group_type}, n_rotations={n_rotations}")
    print(f"  soft_thresholding={soft_thresholding}, method={decomposition_method}")
    print(f"  min_kernel_size={min_kernel_size}, skip_first_conv={skip_first_conv}")
    
    _recursive_replace(model)
    
    print(f"Total Conv2d layers converted: {filtered_count[0]}")
    
    return model


def get_conv2d_layer_info(model: nn.Module) -> list:
    """
    Get information about all Conv2d layers in a model.
    
    Args:
        model: Any PyTorch model
    
    Returns:
        List of dicts with layer info: name, in_channels, out_channels, kernel_size, stride, padding
    """
    layers = []
    
    def _collect_layers(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Conv2d):
                layers.append({
                    'name': full_name,
                    'in_channels': child.in_channels,
                    'out_channels': child.out_channels,
                    'kernel_size': child.kernel_size,
                    'stride': child.stride,
                    'padding': child.padding,
                    'is_filtered': False,
                })
            elif isinstance(child, FilteredConv2d):
                layers.append({
                    'name': full_name,
                    'in_channels': child.in_channels,
                    'out_channels': child.out_channels,
                    'kernel_size': child.kernel_size,
                    'stride': child.stride,
                    'padding': child.padding,
                    'is_filtered': True,
                })
            else:
                _collect_layers(child, full_name)
    
    _collect_layers(model)
    return layers


def freeze_filtered_layers(model: nn.Module, min_kernel_size: int = 1) -> int:
    """
    Freeze the weights of FilteredConv2d layers.
    
    Args:
        model: Model with FilteredConv2d layers
        min_kernel_size: Only freeze layers with kernel_size >= this value
    
    Returns:
        Number of layers frozen
    """
    frozen_count = 0
    
    def _freeze_recursive(module):
        nonlocal frozen_count
        for name, child in module.named_children():
            if isinstance(child, FilteredConv2d):
                kernel_size = child.kernel_size[0]
                if kernel_size >= min_kernel_size:
                    child.weight.requires_grad = False
                    if child.bias is not None:
                        child.bias.requires_grad = False
                    frozen_count += 1
            else:
                _freeze_recursive(child)
    
    _freeze_recursive(model)
    print(f"Frozen {frozen_count} FilteredConv2d layers with kernel_size >= {min_kernel_size}")
    return frozen_count

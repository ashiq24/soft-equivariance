"""
Utility functions for custom ViT implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_structure(model: nn.Module, max_depth: int = 3):
    """
    Print the structure of a model.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth to print
    """
    def _print_module(module, name, depth):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        print(f"{indent}{name}: {module.__class__.__name__}")
        
        if depth < max_depth:
            for child_name, child_module in module.named_children():
                _print_module(child_module, child_name, depth + 1)
    
    _print_module(model, "Model", 0)


def get_layer_names(model: nn.Module, prefix: str = "") -> List[str]:
    """
    Get all layer names in a model.
    
    Args:
        model: PyTorch model
        prefix: Prefix for layer names
    
    Returns:
        List of layer names
    """
    names = []
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        names.append(full_name)
        names.extend(get_layer_names(module, full_name))
    return names


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specific layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Unfreeze specific layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                break


def get_learning_rates_by_layer(
    model: nn.Module,
    base_lr: float = 1e-4,
    layer_lr_decay: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    Create layer-wise learning rate schedule.
    
    Deeper layers get higher learning rates, earlier layers get lower rates.
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate
        layer_lr_decay: Decay factor for each layer
    
    Returns:
        List of parameter groups for an optimizer. Each group has the form
        {"params": [...], "lr": scaled_lr}, where scaled_lr is computed as
        base_lr * layer_lr_decay^(max_depth - depth).
    """
    # Get all layer names and depths
    layer_params = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine layer depth (number of dots in name)
        depth = name.count(".")
        
        if depth not in layer_params:
            layer_params[depth] = []
        layer_params[depth].append(param)
    
    # Create parameter groups with different learning rates
    param_groups = []
    max_depth = max(layer_params.keys())
    
    for depth in sorted(layer_params.keys()):
        # Deeper layers get higher learning rates
        lr = base_lr * (layer_lr_decay ** (max_depth - depth))
        param_groups.append({
            "params": layer_params[depth],
            "lr": lr,
        })
    
    return param_groups


def compute_rotation_equivariance_error(
    model: nn.Module,
    images: torch.Tensor,
    rotation_angles: List[int] = [90, 180, 270],
) -> Dict[str, float]:
    """
    Compute rotation equivariance error.
    
    Args:
        model: PyTorch model
        images: Input images (batch_size, channels, height, width)
        rotation_angles: List of rotation angles to test (in degrees)
    
    Returns:
        Dictionary with errors for each rotation angle
    """
    model.eval()
    errors = {}
    
    with torch.no_grad():
        # Get output for original images
        original_output = model(images)
        if hasattr(original_output, "logits"):
            original_output = original_output.logits
        
        # Test each rotation
        for angle in rotation_angles:
            # Rotate images
            k = angle // 90
            rotated_images = torch.rot90(images, k=k, dims=[2, 3])
            
            # Get output for rotated images
            rotated_output = model(rotated_images)
            if hasattr(rotated_output, "logits"):
                rotated_output = rotated_output.logits
            
            # Compute relative error
            error = torch.norm(original_output - rotated_output) / (torch.norm(original_output) + 1e-8)
            errors[f"rotation_{angle}"] = error.item()
    
    return errors


def initialize_custom_layer_from_pretrained(
    custom_layer: nn.Module,
    pretrained_weight: torch.Tensor,
    pretrained_bias: torch.Tensor = None,
):
    """
    Initialize custom layer to approximate pretrained weights.
    
    This uses least-squares to find coefficients that best approximate
    the pretrained weights using the custom layer's basis.
    
    Args:
        custom_layer: Custom layer (ELinear or FLinear)
        pretrained_weight: Pretrained weight matrix
        pretrained_bias: Optional pretrained bias
    """
    # Check if layer has coefficients (ELinear) or weights (FLinear)
    if hasattr(custom_layer, "coeffs") and hasattr(custom_layer, "basis"):
        # ELinear: solve for coefficients
        # weight = basis @ coeffs.T
        # We want: pretrained_weight ≈ basis @ coeffs.T
        # Solution: coeffs.T = basis^+ @ pretrained_weight
        # where basis^+ is the pseudoinverse
        
        basis = custom_layer.basis
        target = pretrained_weight.T  # Transpose to match dimensions
        
        # Solve least-squares: coeffs.T = pinv(basis) @ target
        coeffs_T = torch.linalg.lstsq(basis, target).solution
        
        # Update coefficients
        with torch.no_grad():
            custom_layer.coeffs.copy_(coeffs_T.T)
    
    elif hasattr(custom_layer, "weights"):
        # FLinear: directly copy weights (will be filtered during forward)
        with torch.no_grad():
            custom_layer.weights.copy_(pretrained_weight)
    
    # Copy bias if present
    if pretrained_bias is not None and custom_layer.bias is not None:
        with torch.no_grad():
            custom_layer.bias.copy_(pretrained_bias)


def visualize_attention_maps(
    model: nn.Module,
    images: torch.Tensor,
    layer_idx: int = -1,
    head_idx: int = 0,
) -> torch.Tensor:
    """
    Visualize attention maps from a specific layer and head.
    
    Args:
        model: PyTorch model
        images: Input images (batch_size, channels, height, width)
        layer_idx: Index of transformer layer (-1 for last layer)
        head_idx: Index of attention head
    
    Returns:
        Attention map (batch_size, num_patches, num_patches)
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(images, output_attentions=True, return_dict=True)
        
        # Get attention weights from specified layer
        attentions = outputs.attentions[layer_idx]  # (batch, num_heads, seq_len, seq_len)
        
        # Get specific head
        attention_map = attentions[:, head_idx, :, :]  # (batch, seq_len, seq_len)
    
    return attention_map


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs,
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> Tuple[int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Path to checkpoint
    
    Returns:
        Tuple of (epoch, loss)
    """
    # Use weights_only=False since this is our own trusted checkpoint
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded from {path}")
    print(f"Resuming from epoch {epoch}, loss {loss:.4f}")
    return epoch, loss

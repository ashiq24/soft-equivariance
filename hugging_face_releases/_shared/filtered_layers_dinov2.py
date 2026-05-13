"""
Filtered layer wrappers for DINOv2 models.

These functions monkey-patch DINOv2 embeddings to apply soft equivariance
filters to position embeddings and patch projections.

Supports both standard DINOv2 and DINOv2 with register tokens.
"""

import os
import sys
import importlib

import torch
from typing import Optional
from types import MethodType
import math


def _ensure_softeq():
    """Make the bundled softeq package importable."""
    _dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.isdir(os.path.join(_dir, "softeq")):
        if _dir not in sys.path:
            sys.path.insert(0, _dir)
        return

    if importlib.util.find_spec("softeq") is not None:
        return

    from huggingface_hub import snapshot_download
    parts = os.path.normpath(_dir).split(os.sep)
    idx = next((i for i, p in enumerate(parts) if p == "transformers_modules"), None)
    if idx is None or idx + 2 >= len(parts):
        raise ImportError("Cannot locate the bundled softeq package.")
    owner = parts[idx + 1]
    repo = parts[idx + 2].replace("_hyphen_", "-").replace("_dot_", ".")
    revision = parts[idx + 3] if idx + 3 < len(parts) else None
    snap = snapshot_download(
        f"{owner}/{repo}", revision=revision, allow_patterns=["softeq/**"],
    )
    sys.path.insert(0, snap)


_ensure_softeq()

FilteredConv2d = importlib.import_module("softeq.layers.fconv2d").FilteredConv2d
get_invariant_filter = importlib.import_module("softeq.equi_utils.filter_factory").get_invariant_filter


def custom_dinov2embeddings_forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
    """
    Forward pass for standard DINOv2 embeddings with filtered patch projection (including positional embeddings projection).

    Args:
        pixel_values: Input images of shape (B, C, H, W).
        bool_masked_pos: Optional patch mask used for masked-token replacement.

    Returns:
        Embedded token sequence with CLS token prepended and filtered positional encoding added.
    """
    batch_size, num_channels, height, width = pixel_values.shape
    target_dtype = self.patch_embeddings.projection.weight.dtype
    embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

    if bool_masked_pos is not None and self.use_mask_token:
        embeddings = torch.where(
            bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
        )

    # add the [CLS] token to the embedded patch tokens
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)


    self.position_embeddings.data[:, 1:, :] = self.filter_pos.smooth(
            self.position_embeddings.data[:, 1:, :]
        )
    
    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

    embeddings = self.dropout(embeddings)
    return embeddings


def custom_dinov2_with_registers_embeddings_forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
    """
    Forward pass for DINOv2 with register tokens and filtered patch projection (including positional embeddings projection).

    Args:
        pixel_values: Input images of shape (B, C, H, W).
        bool_masked_pos: Optional patch mask used for masked-token replacement.

    Returns:
        Embedded token sequence with CLS, register tokens, and patch tokens.
    """
    batch_size, num_channels, height, width = pixel_values.shape
    target_dtype = self.patch_embeddings.projection.weight.dtype
    embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

    if bool_masked_pos is not None:
        embeddings = torch.where(
            bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
        )

    # add the [CLS] token to the embedded patch tokens
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)

    # add positional encoding to each token
    # Apply filtering to positional embeddings - only to patch token positions
    if hasattr(self, 'position_embeddings'):
        # Only filter patch positions (skip CLS position)
        self.position_embeddings.data[:, 1:, :] = self.filter_pos.smooth(
            self.position_embeddings.data[:, 1:, :]
        )
    
    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

    # add register tokens
    embeddings = torch.cat(
        (embeddings[:, :1], self.register_tokens.expand(embeddings.shape[0], -1, -1), embeddings[:, 1:]), dim=1
    )

    embeddings = self.dropout(embeddings)
    return embeddings


def monkeypatch_dinov2embeddings(dinov2embeddings, filter_configs):
    """
    Monkey patch DINOv2 embeddings to apply filters.
    
    Supports both standard DINOv2 and DINOv2 with register tokens.
    - Standard DINOv2: No register tokens
    - DINOv2 with registers: Has register tokens (typically 4)
    
    Register tokens should NOT be filtered as they are learned representations.
    
    Args:
        dinov2embeddings: DINOv2 embeddings module.
        filter_configs: Dictionary with filter configuration, including:
            - n_rotations: Number of discrete rotations.
            - soft_thresholding: Softness for patch embedding filter.
            - soft_thresholding_pos: Softness for position embedding filter.
            - decomposition_method: 'svd' or 'schur'.
            - group_type: 'rotation' or 'roto_reflection'.
            - hard_mask: Use hard mask for smoothing.
            - preserve_norm: Preserve weight norms after projection.
            - joint_decomposition: Use joint decomposition for multi-generator groups.
    """
    
    # Determine DINOv2 variant
    is_register_variant = 'WithRegisters' in dinov2embeddings.__class__.__name__
    
    group_type = filter_configs.get("group_type", "rotation")
    
    # Filter patch embeddings (same for both variants)
    original_conv = dinov2embeddings.patch_embeddings.projection
    kernel_size = original_conv.kernel_size[0]  
    # assert square kernel
    assert kernel_size == original_conv.kernel_size[1], "Kernel size is not square"
    filter = get_invariant_filter(
            group_type=group_type,
            n_rotations=filter_configs["n_rotations"],
            input_size=(1, kernel_size , kernel_size),
            soft_threshold=filter_configs["soft_thresholding"],
            decomposition_method=filter_configs["decomposition_method"],
            debug=False,
            hard_mask=filter_configs.get("hard_mask", False),
            preserve_norm=filter_configs.get("preserve_norm", False),
            joint_decomposition=filter_configs.get("joint_decomposition", True)
        )
    
    dinov2embeddings.patch_embeddings.projection = FilteredConv2d(dinov2embeddings.patch_embeddings.projection, filter)
    
    # Handle positional embeddings
    if hasattr(dinov2embeddings, 'position_embeddings') and dinov2embeddings.position_embeddings is not None:
        # Filter positional embeddings for patch tokens only
        num_patches = dinov2embeddings.patch_embeddings.num_patches
        # Check if num_patches is a perfect square
        if math.sqrt(num_patches).is_integer():
            len_pos = int(math.sqrt(num_patches))
            filter_pos = get_invariant_filter(
                    group_type=group_type,
                    n_rotations=filter_configs["n_rotations"],
                    input_size=(1, len_pos, len_pos),
                    soft_threshold=filter_configs["soft_thresholding_pos"],
                    decomposition_method=filter_configs["decomposition_method"],
                    debug=False,
                    hard_mask=filter_configs.get("hard_mask", False),
                    preserve_norm=filter_configs.get("preserve_norm", False),
                    joint_decomposition=filter_configs.get("joint_decomposition", True)
                )
            dinov2embeddings.filter_pos = filter_pos

        else:
            print(f"Warning: DINOv2 num_patches ({num_patches}) is not a perfect square. Skipping positional embedding filtering.")
    else:
        print("DINOv2 model does not have traditional positional embeddings.")

    
    # Replace forward method with appropriate custom implementation
    if is_register_variant:
        dinov2embeddings.forward = MethodType(custom_dinov2_with_registers_embeddings_forward, dinov2embeddings)
    else:
        dinov2embeddings.forward = MethodType(custom_dinov2embeddings_forward, dinov2embeddings)
        print("Applied standard DINOv2 forward method")

"""
Filtered layer wrappers for ViT (Vision Transformer) models.

These functions monkey-patch ViT embeddings and outputs to apply
soft equivariance filters to position embeddings and patch projections.
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


def custom_vitembeddings_forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
    """
    Forward pass for ViT embeddings with filtered patch projection.

    Args:
        pixel_values: Input images of shape (B, C, H, W).
        bool_masked_pos: Optional patch mask used for masked-token replacement.
        interpolate_pos_encoding: If True, interpolate positional encodings to the input size.

    Returns:
        Embedded token sequence with CLS token prepended and filtered positional encoding added.
    """
    batch_size, num_channels, height, width = pixel_values.shape
    embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

    if bool_masked_pos is not None:
        seq_length = embeddings.shape[1]
        mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
        # replace masked visual tokens with mask_tokens
        mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
        embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

    # add the [CLS] token to the embedded patch tokens
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)

    # add positional encoding to each token
    self.position_embeddings.data[:,1:,:] = self.filter_pos.smooth(self.position_embeddings.data[:,1:,:])   
    
    if interpolate_pos_encoding:
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
    else:
        embeddings = embeddings + self.position_embeddings

    embeddings = self.dropout(embeddings)

    return embeddings


def monkeypatch_vitembeddings(vitembeddings, filter_configs):
    """
    Monkey patch ViT embeddings to apply soft equivariance filters.
    
    This replaces the patch projection with a FilteredConv2d and adds
    filtering to position embeddings.
    
    Args:
        vitembeddings: ViT embeddings module from HuggingFace transformers
        filter_configs: Dictionary with filter configuration:
            - n_rotations: Number of discrete rotations
            - soft_thresholding: Softness for patch embedding filter
            - soft_thresholding_pos: Softness for position embedding filter
            - decomposition_method: 'svd' or 'schur'
            - group_type: 'rotation' or 'roto_reflection'
            - hard_mask: Use hard mask for smoothing
            - preserve_norm: Preserve weight norms after projection
            - joint_decomposition: Use joint decomposition for multi-generator groups
    """
    original_conv = vitembeddings.patch_embeddings.projection
    kernel_size = original_conv.kernel_size[0]  
    # assert square kernel
    assert kernel_size == original_conv.kernel_size[1], "Kernel size is not square"
    
    group_type = filter_configs.get("group_type", "rotation")
    
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
    
    vitembeddings.patch_embeddings.projection = FilteredConv2d(vitembeddings.patch_embeddings.projection, filter)
    
    # filter for position embeddings
    num_patches = vitembeddings.patch_embeddings.num_patches
    # assert num_patches is a  perfect square
    assert math.sqrt(num_patches).is_integer(), "Number of patches is not a perfect square"
    len_pos = int(math.sqrt(num_patches))
    filter_pos = get_invariant_filter(
            group_type=group_type,
            n_rotations=filter_configs["n_rotations"],
            input_size=(1, len_pos, len_pos ),
            soft_threshold=filter_configs["soft_thresholding_pos"],
            decomposition_method=filter_configs["decomposition_method"],
            debug=False,
            hard_mask=filter_configs.get("hard_mask", False),
            preserve_norm=filter_configs.get("preserve_norm", False),
            joint_decomposition=filter_configs.get("joint_decomposition", True)
        )
    vitembeddings.filter_pos = filter_pos
    vitembeddings.forward = MethodType(custom_vitembeddings_forward, vitembeddings)


def custom_vitoutput_forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for ViT attention output with token-wise filtering.

    Args:
        hidden_states: Attention output of shape (B, N, D).
        input_tensor: Residual input tensor of shape (B, N, D).

    Returns:
        Residual output where non-CLS tokens are smoothed by the configured filter.
    """
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = hidden_states + input_tensor

    hidden_states[:,1:, :] = self.filter.smooth(hidden_states[:,1:, :])  # Do not filter CLS token

    return hidden_states


def monkeypatch_vitoutput(vitoutput, filter_configs, num_patches):
    """
    Monkey patch ViT output layer to apply soft equivariance filters.
    
    This adds filtering to the attention output layer, excluding the CLS token.
    
    Args:
        vitoutput: ViT output module from HuggingFace transformers.
        filter_configs: Dictionary with filter configuration.
        num_patches: Number of patches; must be a perfect square.
    """
    assert math.sqrt(num_patches).is_integer(), "Number of patches is not a perfect square"
    len_pos = int(math.sqrt(num_patches))
    
    group_type = filter_configs.get("group_type", "rotation")
    
    filter_pos = get_invariant_filter(
            group_type=group_type,
            n_rotations=filter_configs["n_rotations"],
            input_size=(1, len_pos, len_pos ),
            soft_threshold=filter_configs["soft_thresholding_attention_output"],
            decomposition_method=filter_configs["decomposition_method"],
            debug=False,
            hard_mask=filter_configs.get("hard_mask", False),
            preserve_norm=filter_configs.get("preserve_norm", False),
            joint_decomposition=filter_configs.get("joint_decomposition", True)
        )
    vitoutput.filter = filter_pos
    vitoutput.forward = MethodType(custom_vitoutput_forward, vitoutput)

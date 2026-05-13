"""
Filtered layer wrappers for DINOv3 models.

DINOv3 has NO learnable positional embedding. Position is encoded via RoPE
(Rotary Position Embedding) applied inside every attention layer. The RoPE
angles are computed from a 2D patch coordinate grid (normalized to [-1, +1]).

To introduce soft equivariance we:
  1. Filter the patch embedding Conv2d  -- same as DINOv2.
  2. Project the RoPE coordinate grid onto the rotation-invariant subspace
     of the group, via InvariantProjector.smooth(). This is done once per
     forward pass, AFTER the lru_cache returns the base grid and BEFORE the
     optional training-time augmentation (rescale/shift/jitter).

The monkey-patch replaces DINOv3ViTRopePositionEmbedding.forward() so that
the rest of the model (attention, MLP, etc.) is untouched.
"""

import math
import os
import sys
import importlib

import torch
from types import MethodType
from typing import Optional


def _ensure_softeq():
    """Make the bundled softeq package importable."""
    _dir = os.path.dirname(os.path.abspath(__file__))

    # If the model folder contains `softeq/`, import directly from it.
    if os.path.isdir(os.path.join(_dir, "softeq")):
        if _dir not in sys.path:
            sys.path.insert(0, _dir)
        return

    # If a real pip package `softeq` is installed, use it.
    if importlib.util.find_spec("softeq") is not None:
        return

    # Otherwise, download only the `softeq/**` subtree from the Hub.
    from huggingface_hub import snapshot_download

    parts = os.path.normpath(_dir).split(os.sep)
    idx = next((i for i, p in enumerate(parts) if p == "transformers_modules"), None)
    if idx is None or idx + 2 >= len(parts):
        raise ImportError("Cannot locate the bundled softeq package.")

    owner = parts[idx + 1]
    repo = parts[idx + 2].replace("_hyphen_", "-").replace("_dot_", ".")
    revision = parts[idx + 3] if idx + 3 < len(parts) else None

    snap = snapshot_download(
        f"{owner}/{repo}",
        revision=revision,
        allow_patterns=["softeq/**"],
    )
    if snap not in sys.path:
        sys.path.insert(0, snap)


_ensure_softeq()

FilteredConv2d = importlib.import_module("softeq.layers.fconv2d").FilteredConv2d
get_invariant_filter = importlib.import_module("softeq.equi_utils.filter_factory").get_invariant_filter

from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    get_patches_center_coordinates,
    augment_patches_center_coordinates,
)

try:
    from transformers.utils.generic import maybe_autocast  # type: ignore
except Exception:  # pragma: no cover
    try:
        from transformers.utils import maybe_autocast  # type: ignore
    except Exception:  # pragma: no cover
        from contextlib import contextmanager

        @contextmanager
        def maybe_autocast(*args, **kwargs):  # type: ignore
            yield


def custom_dinov3_rope_forward(
    self,
    pixel_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Drop-in replacement for DINOv3ViTRopePositionEmbedding.forward().
    """
    _, _, height, width = pixel_values.shape
    num_patches_h = height // self.config.patch_size
    num_patches_w = width // self.config.patch_size

    device = pixel_values.device
    device_type = (
        device.type
        if isinstance(device.type, str) and device.type != "mps"
        else "cpu"
    )

    with maybe_autocast(device_type=device_type, enabled=False):

        patch_coords = get_patches_center_coordinates(
            num_patches_h, num_patches_w,
            dtype=torch.float32, device=device,
        )

        if hasattr(self, "filter_coords"):
            patch_coords = self.filter_coords.smooth(patch_coords.unsqueeze(0)).squeeze(0)

        if self.training:
            patch_coords = augment_patches_center_coordinates(
                patch_coords,
                shift=self.config.pos_embed_shift,
                jitter=self.config.pos_embed_jitter,
                rescale=self.config.pos_embed_rescale,
            )

        angles = (
            2 * math.pi
            * patch_coords[:, :, None]
            * self.inv_freq[None, None, :]
        )
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

    dtype = pixel_values.dtype
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def monkeypatch_dinov3embeddings(
    dinov3_model,
    filter_configs: dict,
    image_size: Optional[int] = None,
) -> None:
    """
    Apply soft-equivariance filters to a DINOv3ViTModel (or backbone).

    Args:
        dinov3_model: A DINOv3ViTModel or DINOv3ViTBackbone instance with
                      .embeddings and .rope_embeddings.
        filter_configs: Filter parameters (mirrors SoftEqConfig._make_filter_config extras).
        image_size: Fine-tuning resolution; if None, uses dinov3_model.config.image_size.
    """
    embeddings = dinov3_model.embeddings
    rope_emb = dinov3_model.rope_embeddings
    config = dinov3_model.config
    group_type = filter_configs.get("group_type", "rotation")

    _image_size = image_size if image_size is not None else config.image_size
    if image_size is not None and image_size != config.image_size:
        print(
            f"  [monkeypatch] Fine-tuning image_size={image_size} "
            f"differs from config.image_size={config.image_size}. "
            f"Building coord filter for {_image_size}x{_image_size} grid."
        )

    if filter_configs.get("filter_patch_embeddings", True):
        print("Filtering DINOv3 patch embedding Conv2d...")

        original_conv = embeddings.patch_embeddings
        kernel_size = original_conv.kernel_size[0]
        assert kernel_size == original_conv.kernel_size[1], (
            "Patch embedding kernel must be square"
        )

        conv_filter = get_invariant_filter(
            group_type=group_type,
            n_rotations=filter_configs["n_rotations"],
            input_size=(1, kernel_size, kernel_size),
            soft_threshold=filter_configs["soft_thresholding"],
            decomposition_method=filter_configs["decomposition_method"],
            debug=False,
            hard_mask=filter_configs.get("hard_mask", False),
            preserve_norm=filter_configs.get("preserve_norm", False),
            joint_decomposition=filter_configs.get("joint_decomposition", True),
        )
        embeddings.patch_embeddings = FilteredConv2d(original_conv, conv_filter)
        print(f"  Conv2d kernel ({kernel_size}x{kernel_size}) wrapped with FilteredConv2d.")

    soft_pos = filter_configs.get("soft_thresholding_pos", 1.0)

    num_patches_h = _image_size // config.patch_size
    num_patches_w = _image_size // config.patch_size
    num_patches = num_patches_h * num_patches_w

    if not math.sqrt(num_patches).is_integer():
        print(
            f"Warning: num_patches ({num_patches}) is not a perfect square. "
            "Skipping RoPE coord filtering."
        )
    else:
        grid_size = int(math.sqrt(num_patches))
        print(
            f"Filtering DINOv3 RoPE coordinate grid "
            f"({grid_size}x{grid_size}, softness={soft_pos})..."
        )

        coord_filter = get_invariant_filter(
            group_type=group_type,
            n_rotations=filter_configs["n_rotations"],
            input_size=(1, grid_size, grid_size),
            soft_threshold=soft_pos,
            decomposition_method=filter_configs["decomposition_method"],
            debug=False,
            hard_mask=filter_configs.get("hard_mask", False),
            preserve_norm=filter_configs.get("preserve_norm", False),
            joint_decomposition=filter_configs.get("joint_decomposition", True),
        )

        rope_emb.filter_coords = coord_filter
        rope_emb.forward = MethodType(custom_dinov3_rope_forward, rope_emb)
        print("  RoPE forward patched — coord smoothing active.")

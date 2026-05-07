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
     optional training-time augmentation (rescale/shift/jitter). The augment-
     ation then operates on the already-isotropic coords, which preserves
     isotropy because rescale/shift are uniform operations.

The monkey-patch replaces DINOv3ViTRopePositionEmbedding.forward() so that
the rest of the model (attention, MLP, etc.) is untouched.
"""

import math
import torch
from types import MethodType
from typing import Optional

from softeq.layers.fconv2d import FilteredConv2d
from softeq.equi_utils.filter_factory import get_invariant_filter

# DINOv3 internals we need to call
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    get_patches_center_coordinates,
    augment_patches_center_coordinates,
)

# maybe_autocast moved across transformers versions.
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


# ---------------------------------------------------------------------------
# Patched RoPE forward
# ---------------------------------------------------------------------------

def custom_dinov3_rope_forward(
    self,
    pixel_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Drop-in replacement for DINOv3ViTRopePositionEmbedding.forward().

    Changes vs. the original:
      - After fetching the base grid from lru_cache, applies
        self.filter_coords.smooth() to project it onto the
        rotation-invariant subspace.
      - Training-time augmentation (rescale / shift / jitter) is applied
        AFTER the smoothing, so augmented grids remain isotropic.

    Everything else (float32 casting, angle computation, cos/sin output)
    is identical to the original implementation.
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

    with maybe_autocast(device_type=device_type, enabled=False):  # force float32

        # ── 1. Base grid from cache (H*W, 2) ──────────────────────────────
        patch_coords = get_patches_center_coordinates(
            num_patches_h, num_patches_w,
            dtype=torch.float32, device=device,
        )

        # ── 2. Project to isotropic subspace (soft equivariance) ──────────
        # filter_coords is an InvariantProjector built for the (H x W) grid.
        # smooth() is a cheap einsum: (H*W, H*W) @ (H*W, 2) → (H*W, 2)
        # softness == 1.0 is a no-op (identity), so this branch is safe even
        # when filtering is disabled.
        if hasattr(self, "filter_coords"):
            # InvariantProjector.smooth projects the *token axis* for 3D inputs:
            #   (B, tokens, channels) -> (B, tokens, channels)
            # Our coords are (tokens, 2), so add/remove a batch dim.
            patch_coords = self.filter_coords.smooth(patch_coords.unsqueeze(0)).squeeze(0)

        # ── 3. Training-time augmentation on the already-isotropic grid ───
        if self.training:
            patch_coords = augment_patches_center_coordinates(
                patch_coords,
                shift=self.config.pos_embed_shift,
                jitter=self.config.pos_embed_jitter,
                rescale=self.config.pos_embed_rescale,
            )

        # ── 4. Angles → cos / sin  (unchanged from original) ──────────────
        # (H*W, 2, head_dim/4)  →  flatten  →  (H*W, head_dim/2)  →  tile  →  (H*W, head_dim)
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def monkeypatch_dinov3embeddings(
    dinov3_model,
    filter_configs: dict,
    image_size: Optional[int] = None,
) -> None:
    """
    Apply soft-equivariance filters to a DINOv3ViTModel (or backbone).

    Two interventions:
      A) patch embedding Conv2d  → wrapped with FilteredConv2d
      B) RoPE coordinate grid   → projected via InvariantProjector.smooth()

    Args:
        dinov3_model:   A DINOv3ViTModel or DINOv3ViTBackbone instance.
                        Must have .embeddings and .rope_embeddings attributes.
        filter_configs: dict with keys:
            group_type            str   e.g. "rotation"
            n_rotations           int   e.g. 4
            soft_thresholding     float patch-conv softness  [0, 1]
            soft_thresholding_pos float coord-grid softness  [0, 1]
            decomposition_method  str   "svd" | "schur"
            hard_mask             bool
            preserve_norm         bool
            joint_decomposition   bool
        image_size:     The actual fine-tuning resolution (e.g. 512 for ADE20K).
                        If None, falls back to config.image_size (224 for most
                        pretrained checkpoints). MUST be set explicitly when
                        fine-tuning at a different resolution than pretraining,
                        otherwise the coord filter will be built for the wrong
                        grid size and mismatches at forward time.
    """
    embeddings    = dinov3_model.embeddings       # DINOv3ViTEmbeddings
    rope_emb      = dinov3_model.rope_embeddings  # DINOv3ViTRopePositionEmbedding
    config        = dinov3_model.config
    group_type    = filter_configs.get("group_type", "rotation")

    # image_size: explicit arg wins; fallback to config (pretrain resolution)
    _image_size = image_size if image_size is not None else config.image_size
    if image_size is not None and image_size != config.image_size:
        print(
            f"  [monkeypatch] Fine-tuning image_size={image_size} "
            f"differs from config.image_size={config.image_size}. "
            f"Building coord filter for {image_size}x{image_size} grid."
        )

    # ── A. Filter patch embedding Conv2d ────────────────────────────────────
    if filter_configs.get("filter_patch_embeddings", True):
        print("Filtering DINOv3 patch embedding Conv2d...")

        original_conv = embeddings.patch_embeddings   # nn.Conv2d
        kernel_size   = original_conv.kernel_size[0]
        assert kernel_size == original_conv.kernel_size[1], \
            "Patch embedding kernel must be square"

        conv_filter = get_invariant_filter(
            group_type          = group_type,
            n_rotations         = filter_configs["n_rotations"],
            input_size          = (1, kernel_size, kernel_size),
            soft_threshold      = filter_configs["soft_thresholding"],
            decomposition_method= filter_configs["decomposition_method"],
            debug               = False,
            hard_mask           = filter_configs.get("hard_mask", False),
            preserve_norm       = filter_configs.get("preserve_norm", False),
            joint_decomposition = filter_configs.get("joint_decomposition", True),
        )
        embeddings.patch_embeddings = FilteredConv2d(original_conv, conv_filter)
        print(f"  Conv2d kernel ({kernel_size}x{kernel_size}) wrapped with FilteredConv2d.")

    # ── B. Filter RoPE coordinate grid ──────────────────────────────────────
    soft_pos = filter_configs.get("soft_thresholding_pos", 1.0)

    # Use _image_size (fine-tuning resolution) NOT config.image_size
    num_patches_h = _image_size // config.patch_size
    num_patches_w = _image_size // config.patch_size
    num_patches   = num_patches_h * num_patches_w

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

        # The filter sees the grid as a (1, grid_size, grid_size) spatial signal.
        # Its filter_x matrix will be (num_patches, num_patches) and is applied
        # to the token axis of patch_coords: (H*W, 2).
        coord_filter = get_invariant_filter(
            group_type          = group_type,
            n_rotations         = filter_configs["n_rotations"],
            input_size          = (1, grid_size, grid_size),
            soft_threshold      = soft_pos,
            decomposition_method= filter_configs["decomposition_method"],
            debug               = False,
            hard_mask           = filter_configs.get("hard_mask", False),
            preserve_norm       = filter_configs.get("preserve_norm", False),
            joint_decomposition = filter_configs.get("joint_decomposition", True),
        )

        # Attach to rope_embeddings so the patched forward can find it
        rope_emb.filter_coords = coord_filter

        # Replace forward method
        rope_emb.forward = MethodType(custom_dinov3_rope_forward, rope_emb)
        print("  RoPE forward patched — coord smoothing active.")

"""
Filtered DINOv3 for Semantic Segmentation (linear probing style).

Loads the generic pretrained DINOv3 backbone (no classification head) and
adds a lightweight 1×1 Conv segmentation head, mirroring FilteredDino2Seg.

Soft-equivariance is applied to:
  - the patch embedding Conv2d
  - the RoPE coordinate grid (projected to rotation-invariant subspace)

Token layout in DINOv3:
    [CLS | REG_0 ... REG_{R-1} | patch_0 ... patch_{N-1}]
    where R = config.num_register_tokens (default 4 for most checkpoints).

The forward pass skips CLS + register tokens and reshapes the patch tokens
back to a (H/patch_size, W/patch_size) spatial grid before the classifier.
Logits are bilinearly upsampled to the input image resolution.

Usage:
    model = FilteredDinoV3Seg(
        pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        num_labels=21,
        soft_thresholding=0.3,
        soft_thresholding_pos=0.3,
    )
"""

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SemanticSegmenterOutput

from models.filtered_layers_dinov3 import monkeypatch_dinov3embeddings


class FilteredDinoV3Seg(nn.Module):
    """
    DINOv3 backbone with a linear segmentation head (1×1 Conv).

    Uses the generic *pretrain* checkpoint (no classification head) so the
    full patch token sequence is available for dense prediction.

    Args:
        pretrained_model_name:  HuggingFace model id for a pretrained DINOv3.
                                Default: "facebook/dinov3-vits16-pretrain-lvd1689m".
        num_labels:             Number of semantic classes.
        filter_patch_embeddings: Whether to filter the patch Conv2d.
        group_type:             Symmetry group, e.g. "rotation" (C_n).
        n_rotations:            Number of discrete rotations (4 = C4, 8 = C8).
        soft_thresholding:      Softness for Conv2d filter [0=exact, 1=no-op].
        soft_thresholding_pos:  Softness for RoPE coord filter [0=exact, 1=no-op].
        decomposition_method:   "schur" (default) or "svd".
        hard_mask:              Use hard zero mask beyond invariant subspace.
        preserve_norm:          Re-scale projected weights to original norm.
        joint_decomposition:    Joint decomp for multi-generator groups.
        ignore_index:           Class index ignored in cross-entropy loss.
        load_pretrained_weight: Load pretrained backbone weights.
        freeze_patch_embeddings: Freeze the (filtered) patch Conv2d.
        image_size:             Actual fine-tuning resolution. Used to build the
                                RoPE coord filter for the correct grid size.
                                Must match the images fed at training time.
                                Default 512 for ADE20K at 512×512.
    """

    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        num_labels: int = 21,
        filter_patch_embeddings: bool = True,
        group_type: str = "rotation",
        n_rotations: int = 4,
        soft_thresholding: float = 0.0,
        soft_thresholding_pos: float = 0.0,
        decomposition_method: str = "schur",
        hard_mask: bool = False,
        preserve_norm: bool = False,
        joint_decomposition: bool = True,
        ignore_index: int = 255,
        load_pretrained_weight: bool = True,
        freeze_patch_embeddings: bool = False,
        image_size: int = 512,
    ):
        super().__init__()
        self.image_size = image_size

        # ── Load generic pretrained backbone (DINOv3ViTModel) ────────────
        if load_pretrained_weight:
            print(f"Loading pretrained DINOv3 backbone: {pretrained_model_name}")
            self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        else:
            print("Initialising DINOv3 backbone from config only (random weights).")
            cfg = AutoConfig.from_pretrained(pretrained_model_name)
            self.backbone = AutoModel.from_config(cfg)

        self.config     = self.backbone.config
        self.num_labels = num_labels
        self.ignore_index = ignore_index

        hidden_size = self.config.hidden_size
        self.num_register_tokens = getattr(self.config, "num_register_tokens", 0)

        self.filter_config = {
            "filter_patch_embeddings": filter_patch_embeddings,
            "group_type":              group_type,
            "n_rotations":             n_rotations,
            "soft_thresholding":       soft_thresholding,
            "soft_thresholding_pos":   soft_thresholding_pos,
            "decomposition_method":    decomposition_method,
            "hard_mask":               hard_mask,
            "preserve_norm":           preserve_norm,
            "joint_decomposition":     joint_decomposition,
            "freeze_patch_embeddings": freeze_patch_embeddings,
            "image_size":              image_size,
        }

        # ── Linear segmentation head (1×1 Conv = linear per pixel) ───────
        self.classifier = nn.Conv2d(hidden_size, num_labels, kernel_size=1, bias=True)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # ── Apply filters to backbone ─────────────────────────────────────
        # DINOv3ViTModel has .embeddings and .rope_embeddings directly
        monkeypatch_dinov3embeddings(self.backbone, self.filter_config, image_size=image_size)

        # ── Optional weight freezing ──────────────────────────────────────
        if freeze_patch_embeddings:
            print("Freezing patch embedding weights...")
            patch_proj = self.backbone.embeddings.patch_embeddings
            if hasattr(patch_proj, "weight"):
                patch_proj.weight.requires_grad = False
            if hasattr(patch_proj, "bias") and patch_proj.bias is not None:
                patch_proj.bias.requires_grad = False

        print("FilteredDinoV3Seg created successfully.")
        print(f"  Hidden size:      {hidden_size}")
        print(f"  Num labels:       {num_labels}")
        print(f"  Register tokens:  {self.num_register_tokens}")

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else True

        batch_size, _, input_height, input_width = pixel_values.shape

        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # sequence layout: [CLS | REG_0 .. REG_{R-1} | patch_0 .. patch_{N-1}]
        sequence_output = outputs.last_hidden_state  # (B, 1 + R + N, hidden)

        # Skip CLS + register tokens → pure patch tokens
        num_skip = 1 + self.num_register_tokens
        patch_tokens = sequence_output[:, num_skip:, :]  # (B, N, hidden)

        num_patches = patch_tokens.shape[1]
        grid_h = input_height // self.config.patch_size
        grid_w = input_width  // self.config.patch_size

        if grid_h * grid_w != num_patches:
            raise ValueError(
                f"Expected {grid_h}*{grid_w}={grid_h*grid_w} patch tokens "
                f"but got {num_patches}. "
                f"(input {input_height}x{input_width}, patch_size={self.config.patch_size}, "
                f"registers={self.num_register_tokens})"
            )

        # (B, N, hidden) → (B, hidden, grid_h, grid_w)
        feature_map = (
            patch_tokens
            .permute(0, 2, 1)
            .reshape(batch_size, -1, grid_h, grid_w)
        )

        # (B, num_labels, grid_h, grid_w)
        logits_lowres = self.classifier(feature_map)

        # Upsample to label / input resolution
        target_size = labels.shape[-2:] if labels is not None else (input_height, input_width)
        logits = nn.functional.interpolate(
            logits_lowres,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions   if output_attentions   else None,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_filtered_dinov3_seg(config: Dict[str, Any]) -> FilteredDinoV3Seg:
    """
    Convenience factory that builds a FilteredDinoV3Seg from a config dict.

    Expected keys mirror the constructor arguments; see FilteredDinoV3Seg docstring.
    """
    return FilteredDinoV3Seg(
        pretrained_model_name  = config.get("pretrained_model",
                                            "facebook/dinov3-vits16-pretrain-lvd1689m"),
        num_labels             = config.get("num_labels", 21),
        filter_patch_embeddings= config.get("filter_patch_embeddings", True),
        group_type             = config.get("group_type", "rotation"),
        n_rotations            = config.get("n_rotations", 4),
        soft_thresholding      = config.get("soft_thresholding", 0.0),
        soft_thresholding_pos  = config.get("soft_thresholding_pos", 0.0),
        decomposition_method   = config.get("decomposition_method", "schur"),
        hard_mask              = config.get("hard_mask", False),
        preserve_norm          = config.get("preserve_norm", False),
        joint_decomposition    = config.get("joint_decomposition", True),
        ignore_index           = config.get("ignore_index", 255),
        load_pretrained_weight = config.get("load_pretrained_weight", True),
        freeze_patch_embeddings= config.get("freeze_patch_embeddings", False),
        image_size             = config.get("image_size", 512),
    )

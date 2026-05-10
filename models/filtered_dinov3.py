"""
Filtered DINOv3 for Image Classification.

Loads a pretrained DINOv3 backbone (no classification head) and adds a
lightweight linear classifier on top of the CLS token.

Soft-equivariance filters are applied to:
  - the patch embedding Conv2d
  - the RoPE coordinate grid (projected to rotation-invariant subspace)

Usage:
    model = FilteredDinoV3(
                pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        num_labels=1000,
        soft_thresholding=0.3,
        soft_thresholding_pos=0.3,
    )
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import ImageClassifierOutput

from models.filtered_layers_dinov3 import monkeypatch_dinov3embeddings


class FilteredDinoV3(nn.Module):
    """
    DINOv3 image-classification model with soft-equivariance filters.

    Wraps a DINOv3ViTModel backbone (AutoModel) and adds a linear head on top
    of the CLS token. The backbone's patch-embedding Conv2d and RoPE coordinate
    grid are projected onto the invariant subspace of the chosen symmetry group.

    Args:
        pretrained_model_name:  HuggingFace model id for a pretrained DINOv3.
                                Default is the ViT-S/16 pretrain checkpoint.
        num_labels:             Number of output classes (1000 for ImageNet).
        filter_patch_embeddings: Whether to filter the patch Conv2d.
        group_type:             Symmetry group, e.g. "rotation" (C_n).
        n_rotations:            Number of discrete rotations (4 = C4, 8 = C8).
        soft_thresholding:      Softness for Conv2d filter [0=exact, 1=no-op].
        soft_thresholding_pos:  Softness for RoPE coord filter [0=exact, 1=no-op].
        decomposition_method:   "schur" (default) or "svd".
        hard_mask:              Use hard zero mask beyond invariant subspace.
        preserve_norm:          Re-scale projected weights to original norm.
        joint_decomposition:    Joint decomp for multi-generator groups.
        load_pretrained_weight: Load pretrained backbone + head weights.
        freeze_patch_embeddings: Freeze the (filtered) patch Conv2d.
        image_size:             Fine-tuning resolution. Default 224 matches the
                                pretrain resolution of most DINOv3 checkpoints.
                                Change if fine-tuning classification at a
                                different crop size.
    """

    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        num_labels: int = 1000,
        filter_patch_embeddings: bool = True,
        group_type: str = "rotation",
        n_rotations: int = 4,
        soft_thresholding: float = 0.0,
        soft_thresholding_pos: float = 0.0,
        decomposition_method: str = "schur",
        hard_mask: bool = False,
        preserve_norm: bool = False,
        joint_decomposition: bool = True,
        load_pretrained_weight: bool = True,
        freeze_patch_embeddings: bool = False,
        image_size: int = 224,
    ):
        super().__init__()

        self.num_labels = num_labels

        # ── Load the pretrained backbone (no classifier head) ────────────
        if load_pretrained_weight:
            print(f"Loading pretrained DINOv3 backbone: {pretrained_model_name}")
            self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        else:
            print("Initialising DINOv3 backbone from config only (random weights).")
            cfg = AutoConfig.from_pretrained(pretrained_model_name)
            self.backbone = AutoModel.from_config(cfg)

        self.config = self.backbone.config
        hidden_size = self.config.hidden_size

        # ── Linear classifier head (random init) ─────────────────────────
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        self.loss_fct = nn.CrossEntropyLoss()

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

        # ── Apply filters to backbone ─────────────────────────────────────
        monkeypatch_dinov3embeddings(self.backbone, self.filter_config, image_size=image_size)

        # ── Optional weight freezing ──────────────────────────────────────
        if freeze_patch_embeddings:
            print("Freezing patch embedding weights...")
            patch_proj = self.backbone.embeddings.patch_embeddings
            if hasattr(patch_proj, "weight"):
                patch_proj.weight.requires_grad = False
            if hasattr(patch_proj, "bias") and patch_proj.bias is not None:
                patch_proj.bias.requires_grad = False

        print("FilteredDinoV3 (classification) created successfully.")

    # ------------------------------------------------------------------
    def _get_backbone(self):
        """Return the DINOv3 backbone module (AutoModel)."""
        return self.backbone

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else True

        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("Backbone output missing last_hidden_state")

        # CLS token is first token
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_filtered_dinov3(config: Dict[str, Any]) -> FilteredDinoV3:
    """
    Convenience factory that builds a FilteredDinoV3 from a config dict.

    Expected keys mirror the constructor arguments; see FilteredDinoV3 docstring.
    """
    return FilteredDinoV3(
        pretrained_model_name  = config.get("pretrained_model",
                                            "facebook/dinov3-vits16-pretrain-lvd1689m"),
        num_labels             = config.get("num_labels", 1000),
        filter_patch_embeddings= config.get("filter_patch_embeddings", True),
        group_type             = config.get("group_type", "rotation"),
        n_rotations            = config.get("n_rotations", 4),
        soft_thresholding      = config.get("soft_thresholding", 0.0),
        soft_thresholding_pos  = config.get("soft_thresholding_pos", 0.0),
        decomposition_method   = config.get("decomposition_method", "schur"),
        hard_mask              = config.get("hard_mask", False),
        preserve_norm          = config.get("preserve_norm", False),
        joint_decomposition    = config.get("joint_decomposition", True),
        load_pretrained_weight = config.get("load_pretrained_weight", True),
        freeze_patch_embeddings= config.get("freeze_patch_embeddings", False),
        image_size             = config.get("image_size", 224),
    )

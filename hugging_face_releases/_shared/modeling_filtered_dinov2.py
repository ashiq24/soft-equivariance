"""
Soft-Equivariant DINOv2 for Image Classification.

Supports both standard DINOv2 and DINOv2 with register tokens.
The variant is selected automatically from config.pretrained_model
(any name containing "reg" or "register" uses the register variant).

Architecture (state-dict compatible with training checkpoints):
* self.dinov2      – Dinov2ForImageClassification backbone
* The patch-embedding and positional-embedding layers are monkeypatched
  to apply soft-equivariant invariant projections.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

try:
    from transformers import (
        Dinov2ForImageClassification,
        Dinov2Config,
        Dinov2WithRegistersForImageClassification,
        Dinov2WithRegistersConfig,
    )
    REGISTERS_AVAILABLE = True
except ImportError:
    from transformers import Dinov2ForImageClassification, Dinov2Config
    REGISTERS_AVAILABLE = False

from configuration_softeq import SoftEqConfig
from filtered_layers_dinov2 import monkeypatch_dinov2embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Standard DINOv2 (no register tokens)
# ─────────────────────────────────────────────────────────────────────────────

class FilteredDinoV2(PreTrainedModel):
    """Soft-Equivariant DINOv2 for image classification (standard variant)."""

    config_class = SoftEqConfig

    def __init__(self, config: SoftEqConfig):
        super().__init__(config)

        print(f"Loading DINOv2 architecture config from: {config.pretrained_model}")
        backbone_cfg = Dinov2Config.from_pretrained(config.pretrained_model)
        backbone_cfg.num_labels = config.num_labels

        # Build full classification model (random init; weights from safetensors).
        self.dinov2 = Dinov2ForImageClassification(backbone_cfg)
        # Keep the same top-level attribute so state-dict keys match training.

        filter_config = config._make_filter_config()
        if config.filter_patch_embeddings:
            print("Applying soft-equivariant filter to DINOv2 patch embeddings...")
            monkeypatch_dinov2embeddings(self.dinov2.dinov2.embeddings, filter_config)

        if config.freeze_patch_embeddings:
            proj = self.dinov2.dinov2.embeddings.patch_embeddings.projection
            if hasattr(proj, "weight"):
                proj.weight.requires_grad = False
        if config.freeze_position_embeddings:
            emb = self.dinov2.dinov2.embeddings
            if hasattr(emb, "position_embeddings") and emb.position_embeddings is not None:
                emb.position_embeddings.requires_grad = False

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.dinov2(
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 with register tokens
# ─────────────────────────────────────────────────────────────────────────────

class FilteredDinoV2wRegister(PreTrainedModel):
    """Soft-Equivariant DINOv2 with register tokens for image classification."""

    config_class = SoftEqConfig

    def __init__(self, config: SoftEqConfig):
        if not REGISTERS_AVAILABLE:
            raise ImportError(
                "DINOv2 with registers is not available in this transformers version."
            )
        super().__init__(config)

        print(f"Loading DINOv2-with-registers architecture config from: {config.pretrained_model}")
        backbone_cfg = Dinov2WithRegistersConfig.from_pretrained(config.pretrained_model)
        backbone_cfg.num_labels = config.num_labels

        self.dinov2_reg = Dinov2WithRegistersForImageClassification(backbone_cfg)

        filter_config = config._make_filter_config()
        if config.filter_patch_embeddings:
            print("Applying soft-equivariant filter to DINOv2-reg patch embeddings...")
            monkeypatch_dinov2embeddings(
                self.dinov2_reg.dinov2_with_registers.embeddings, filter_config
            )

        if config.freeze_patch_embeddings:
            proj = self.dinov2_reg.dinov2_with_registers.embeddings.patch_embeddings.projection
            if hasattr(proj, "weight"):
                proj.weight.requires_grad = False
        if config.freeze_position_embeddings:
            emb = self.dinov2_reg.dinov2_with_registers.embeddings
            if hasattr(emb, "position_embeddings") and emb.position_embeddings is not None:
                emb.position_embeddings.requires_grad = False

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.dinov2_reg(
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_filtered_dinov2(config: SoftEqConfig):
    """Return the correct filtered DINOv2 class based on the pretrained model name."""
    name = config.pretrained_model.lower()
    if "reg" in name or "register" in name:
        return FilteredDinoV2wRegister(config)
    return FilteredDinoV2(config)

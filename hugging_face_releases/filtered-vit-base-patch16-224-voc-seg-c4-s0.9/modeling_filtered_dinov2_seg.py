"""
Soft-Equivariant DINOv2 for Semantic Segmentation.

Supports both standard DINOv2 and DINOv2 with register tokens.
The variant is selected automatically from config.pretrained_model.

Architecture (state-dict compatible with training checkpoints):
* self.dinov2      – Dinov2Model (or Dinov2WithRegistersModel) backbone
* self.classifier  – nn.Conv2d(hidden_size, num_labels, kernel_size=1) head
* Logits are bilinearly upsampled to input resolution.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Dinov2Config, Dinov2Model
from transformers.modeling_outputs import SemanticSegmenterOutput

try:
    from transformers import Dinov2WithRegistersModel, Dinov2WithRegistersConfig
    REGISTERS_AVAILABLE = True
except ImportError:
    REGISTERS_AVAILABLE = False

from .configuration_softeq import SoftEqConfig
from .filtered_layers_dinov2 import monkeypatch_dinov2embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Standard DINOv2 segmentation
# ─────────────────────────────────────────────────────────────────────────────

class FilteredDino2Seg(PreTrainedModel):
    """Soft-Equivariant DINOv2 for semantic segmentation (standard variant)."""

    config_class = SoftEqConfig

    def __init__(self, config: SoftEqConfig):
        super().__init__(config)

        print(f"Loading DINOv2 segmentation backbone config from: {config.pretrained_model}")
        backbone_cfg = Dinov2Config.from_pretrained(config.pretrained_model)

        self.dinov2 = Dinov2Model(backbone_cfg)
        self.ignore_index = config.ignore_index

        hidden_size = backbone_cfg.hidden_size
        self.classifier = nn.Conv2d(hidden_size, config.num_labels, kernel_size=1, bias=True)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        filter_config = config._make_filter_config()
        with torch.device('cpu'):
            if config.filter_patch_embeddings:
                print("Applying soft-equivariant filter to DINOv2 patch embeddings...")
                monkeypatch_dinov2embeddings(self.dinov2.embeddings, filter_config)

        if config.freeze_patch_embeddings:
            proj = self.dinov2.embeddings.patch_embeddings.projection
            if hasattr(proj, "weight"):
                proj.weight.requires_grad = False
        if config.freeze_position_embeddings:
            emb = self.dinov2.embeddings
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
        return_dict = return_dict if return_dict is not None else True

        batch_size, _, input_height, input_width = pixel_values.shape

        outputs = self.dinov2(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # Standard DINOv2: skip only the CLS token (index 0).
        sequence_output = outputs.last_hidden_state
        patch_tokens = sequence_output[:, 1:, :]
        num_patches = patch_tokens.shape[1]
        grid_size = int(math.isqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch tokens {num_patches} is not a perfect square.")

        feature_map = patch_tokens.permute(0, 2, 1).reshape(batch_size, -1, grid_size, grid_size)
        logits = self.classifier(feature_map)
        target_size = labels.shape[-2:] if labels is not None else (input_height, input_width)
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

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
            attentions=outputs.attentions if output_attentions else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 with register tokens segmentation
# ─────────────────────────────────────────────────────────────────────────────

class FilteredDino2wRegisterSeg(PreTrainedModel):
    """Soft-Equivariant DINOv2 with register tokens for semantic segmentation."""

    config_class = SoftEqConfig

    def __init__(self, config: SoftEqConfig):
        if not REGISTERS_AVAILABLE:
            raise ImportError(
                "DINOv2 with registers is not available in this transformers version."
            )
        super().__init__(config)

        print(f"Loading DINOv2-reg segmentation backbone config from: {config.pretrained_model}")
        backbone_cfg = Dinov2WithRegistersConfig.from_pretrained(config.pretrained_model)

        self.dinov2 = Dinov2WithRegistersModel(backbone_cfg)
        self.ignore_index = config.ignore_index

        hidden_size = backbone_cfg.hidden_size
        self.classifier = nn.Conv2d(hidden_size, config.num_labels, kernel_size=1, bias=True)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        filter_config = config._make_filter_config()
        with torch.device('cpu'):
            if config.filter_patch_embeddings:
                print("Applying soft-equivariant filter to DINOv2-reg patch embeddings...")
                monkeypatch_dinov2embeddings(self.dinov2.embeddings, filter_config)

        if config.freeze_patch_embeddings:
            proj = self.dinov2.embeddings.patch_embeddings.projection
            if hasattr(proj, "weight"):
                proj.weight.requires_grad = False
        if config.freeze_position_embeddings:
            emb = self.dinov2.embeddings
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
        return_dict = return_dict if return_dict is not None else True

        batch_size, _, input_height, input_width = pixel_values.shape

        outputs = self.dinov2(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # With registers: skip 1 CLS + num_register_tokens tokens.
        sequence_output = outputs.last_hidden_state
        num_registers = getattr(self.dinov2.config, "num_register_tokens", 0)
        patch_tokens = sequence_output[:, 1 + num_registers:, :]

        num_patches = patch_tokens.shape[1]
        grid_size = int(math.isqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"DINOv2 with registers: patch tokens {num_patches} is not a perfect square "
                f"(num_registers={num_registers})."
            )

        feature_map = patch_tokens.permute(0, 2, 1).reshape(batch_size, -1, grid_size, grid_size)
        logits = self.classifier(feature_map)
        target_size = (input_height, input_width)
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

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
            attentions=outputs.attentions if output_attentions else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_filtered_dino2_seg(config: SoftEqConfig):
    """Return the correct segmentation model class based on the pretrained model name."""
    name = config.pretrained_model.lower()
    if REGISTERS_AVAILABLE and ("reg" in name or "register" in name):
        return FilteredDino2wRegisterSeg(config)
    return FilteredDino2Seg(config)

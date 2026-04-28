"""
Soft-Equivariant ViT for Semantic Segmentation.

Architecture (state-dict compatible with training checkpoints):
* self.vit          – ViTModel backbone (no classification head)
* self.classifier   – nn.Conv2d(hidden_size, num_labels, kernel_size=1) segmentation head
* Logits are bilinearly upsampled to input resolution.

Filter buffers are stored in model.safetensors alongside learnable weights.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, ViTConfig, ViTModel
from transformers.modeling_outputs import SemanticSegmenterOutput

from configuration_softeq import SoftEqConfig
from filtered_layers_vit import monkeypatch_vitembeddings, monkeypatch_vitoutput


class FilteredViTSeg(PreTrainedModel):
    """
    Soft-Equivariant Vision Transformer for semantic segmentation.

    Uses a linear probing style head: the ViT patch tokens are reshaped to a
    spatial grid, classified by a 1x1 Conv, then bilinearly upsampled to the
    input resolution.
    """

    config_class = SoftEqConfig

    def __init__(self, config: SoftEqConfig):
        super().__init__(config)

        print(f"Loading ViT segmentation backbone config from: {config.pretrained_model}")
        vit_config = ViTConfig.from_pretrained(config.pretrained_model)

        # Build backbone (random init; weights from model.safetensors).
        self.vit = ViTModel(vit_config, add_pooling_layer=False)
        self.ignore_index = config.ignore_index

        hidden_size = vit_config.hidden_size
        self.classifier = nn.Conv2d(hidden_size, config.num_labels, kernel_size=1, bias=True)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        filter_config = config._make_filter_config()
        if config.filter_patch_embeddings:
            print("Applying soft-equivariant filter to patch embeddings...")
            monkeypatch_vitembeddings(self.vit.embeddings, filter_config)

        if config.filter_attention_output:
            num_patches = self.vit.embeddings.patch_embeddings.num_patches
            for layer_idx in range(vit_config.num_hidden_layers):
                if layer_idx in (config.attention_output_filter_list or []):
                    layer = self.vit.encoder.layer[layer_idx]
                    monkeypatch_vitoutput(
                        layer.attention.output, filter_config, num_patches
                    )

        if config.freeze_patch_embeddings:
            proj = self.vit.embeddings.patch_embeddings.projection
            if hasattr(proj, "weight"):
                proj.weight.requires_grad = False
        if config.freeze_position_embeddings:
            if hasattr(self.vit.embeddings, "position_embeddings"):
                self.vit.embeddings.position_embeddings.requires_grad = False

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, input_height, input_width = pixel_values.shape

        outputs = self.vit(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # Remove CLS token; reshape patches to spatial grid.
        sequence_output = outputs.last_hidden_state
        patch_tokens = sequence_output[:, 1:, :]           # [B, N, hidden]
        num_patches = patch_tokens.shape[1]
        grid_size = int(math.sqrt(num_patches))
        feature_map = patch_tokens.permute(0, 2, 1).reshape(batch_size, -1, grid_size, grid_size)

        logits = self.classifier(feature_map)              # [B, num_labels, gh, gw]
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

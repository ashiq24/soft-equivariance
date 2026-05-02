"""
Soft-Equivariant ViT for Image Classification.

This file is part of a self-contained HuggingFace model release.
The model can be loaded with::

    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(
        "your-username/filtered-vit-base-patch16-224-imagenet-c4-s0.0",
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        "your-username/filtered-vit-base-patch16-224-imagenet-c4-s0.0",
        trust_remote_code=True,
    )

Architecture
------------
* Backbone: google/vit-base-patch16-224 (or as specified in config.pretrained_model)
* The patch-embedding Conv2d is wrapped by FilteredConv2d, applying a
  soft-equivariant invariant projection at each forward pass.
* Positional embeddings are smoothed by a second invariant projector.
* All filter matrices are stored as non-trainable buffers and saved in
  model.safetensors alongside the regular weights.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, ViTConfig, ViTModel
from transformers.modeling_outputs import ImageClassifierOutput

from .configuration_softeq import SoftEqConfig
from .filtered_layers_vit import monkeypatch_vitembeddings, monkeypatch_vitoutput


class FilteredViT(PreTrainedModel):
    """
    Soft-Equivariant Vision Transformer for image classification.

    The architecture mirrors ViTForImageClassification (state-dict compatible):
    * self.vit          – ViTModel backbone
    * self.classifier   – nn.Linear head

    The patch-embedding and positional-embedding layers are monkeypatched during
    __init__ to apply invariant projections parameterised by the config.
    """

    config_class = SoftEqConfig

    def __init__(self, config: SoftEqConfig):
        super().__init__(config)

        # Download the backbone ViT architecture config (~3 KB, not weights).
        print(f"Loading ViT architecture config from: {config.pretrained_model}")
        vit_config = ViTConfig.from_pretrained(config.pretrained_model)
        vit_config.num_labels = config.num_labels

        # Build architecture (random init; weights loaded from model.safetensors).
        self.vit = ViTModel(vit_config, add_pooling_layer=False)
        self.classifier = nn.Linear(vit_config.hidden_size, config.num_labels)
        nn.init.zeros_(self.classifier.bias)

        # Apply soft-equivariant filters (creates FilteredConv2d + filter buffers).
        filter_config = config._make_filter_config()
        with torch.device('cpu'):
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

        # Optional weight freezing.
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

        outputs = self.vit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # CLS token is at position 0.
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.config.num_labels), labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

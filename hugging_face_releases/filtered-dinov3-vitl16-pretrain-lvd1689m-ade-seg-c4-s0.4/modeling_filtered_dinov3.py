"""
Soft-Equivariant DINOv3 for Image Classification (HF packaged release).

Training checkpoints use ``backbone`` + ``classifier`` keys (FilteredDinoV3).
"""

from typing import Optional

import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

try:
    from .configuration_softeq import SoftEqConfig
    from .filtered_layers_dinov3 import monkeypatch_dinov3embeddings
except ImportError:
    from configuration_softeq import SoftEqConfig
    from filtered_layers_dinov3 import monkeypatch_dinov3embeddings


class FilteredDinoV3(PreTrainedModel):
    """Soft-equivariant DINOv3 image classification."""

    config_class = SoftEqConfig

    def __init__(self, config: SoftEqConfig):
        super().__init__(config)

        filename = "backbone_config_dinov3-vitl16-pretrain-lvd1689m.json"
        backbone_cfg_path = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.exists(backbone_cfg_path):
            from huggingface_hub import snapshot_download

            parts = os.path.normpath(os.path.dirname(os.path.abspath(__file__))).split(os.sep)
            idx = next((i for i, p in enumerate(parts) if p == "transformers_modules"), None)
            if idx is None or idx + 2 >= len(parts):
                raise FileNotFoundError(
                    "Missing vendored DINOv3 backbone config and cannot infer repo to download it. "
                    f"Expected local file: {backbone_cfg_path}."
                )
            owner = parts[idx + 1]
            repo = parts[idx + 2].replace("_hyphen_", "-").replace("_dot_", ".")
            revision = parts[idx + 3] if idx + 3 < len(parts) else None

            snap = snapshot_download(
                f"{owner}/{repo}",
                revision=revision,
                allow_patterns=[filename],
            )
            backbone_cfg_path = os.path.join(snap, filename)

        print(f"Loading DINOv3 backbone config from: {backbone_cfg_path}")
        backbone_cfg = AutoConfig.from_pretrained(backbone_cfg_path)
        if getattr(backbone_cfg, "model_type", None) != "dinov3_vit":
            raise ValueError(
                f"Unexpected backbone config model_type={getattr(backbone_cfg, 'model_type', None)}; "
                "expected 'dinov3_vit'."
            )
        if getattr(backbone_cfg, "hidden_size", None) != 1024 or getattr(backbone_cfg, "patch_size", None) != 16:
            raise ValueError(
                "Backbone config does not look like dinov3 vit-large/16 "
                f"(hidden_size={getattr(backbone_cfg, 'hidden_size', None)}, patch_size={getattr(backbone_cfg, 'patch_size', None)})."
            )
        self.backbone = AutoModel.from_config(backbone_cfg)

        hidden_size = backbone_cfg.hidden_size
        self.classifier = nn.Linear(hidden_size, config.num_labels, bias=True)
        nn.init.normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        self.loss_fct = nn.CrossEntropyLoss()

        filter_cfg = config._make_filter_config()
        _img_sz = getattr(config, "image_size", None)
        with torch.device("cpu"):
            if config.filter_patch_embeddings:
                print("Applying soft-equivariant filters to DINOv3 backbone...")
                monkeypatch_dinov3embeddings(
                    self.backbone,
                    filter_cfg,
                    image_size=_img_sz,
                )

        if config.freeze_patch_embeddings:
            patch_proj = self.backbone.embeddings.patch_embeddings
            if hasattr(patch_proj, "weight"):
                patch_proj.weight.requires_grad = False
            if hasattr(patch_proj, "bias") and patch_proj.bias is not None:
                patch_proj.bias.requires_grad = False

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

        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

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

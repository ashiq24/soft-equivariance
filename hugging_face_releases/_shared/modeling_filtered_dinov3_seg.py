"""
Soft-Equivariant DINOv3 for Semantic Segmentation.

State-dict keys match training (models.filtered_dinov3_seg.FilteredDinoV3Seg):
  backbone.*    — AutoModel / DINOv3ViTModel
  classifier.* — 1×1 Conv head

Token layout: [CLS | REG_* | patches]. Patch grid uses input H×W and patch_size
from the backbone config (same as training; not sqrt(N) reordering).
"""

from typing import Optional

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput

try:
    from .configuration_softeq import SoftEqConfig
    from .filtered_layers_dinov3 import monkeypatch_dinov3embeddings
except ImportError:  # loaded as top-level module (sys.path folder / tooling smoke tests)
    from configuration_softeq import SoftEqConfig
    from filtered_layers_dinov3 import monkeypatch_dinov3embeddings


class FilteredDinoV3Seg(PreTrainedModel):
    """Soft-equivariant DINOv3 segmentation (packaged HF release)."""

    config_class = SoftEqConfig

    def __init__(self, config: SoftEqConfig):
        super().__init__(config)

        filename = "backbone_config_dinov3-vitl16-pretrain-lvd1689m.json"
        backbone_cfg_path = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.exists(backbone_cfg_path):
            from huggingface_hub import snapshot_download

            # Resolve <owner>/<repo> from the dynamic module cache path.
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

        self.ignore_index = config.ignore_index
        self.num_register_tokens = getattr(backbone_cfg, "num_register_tokens", 0)

        hidden_size = backbone_cfg.hidden_size
        self.classifier = nn.Conv2d(hidden_size, config.num_labels, kernel_size=1, bias=True)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

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

        batch_size, _, input_height, input_width = pixel_values.shape

        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        num_skip = 1 + self.num_register_tokens
        patch_tokens = sequence_output[:, num_skip:, :]

        patch_size = self.backbone.config.patch_size
        num_patches = patch_tokens.shape[1]
        grid_h = input_height // patch_size
        grid_w = input_width // patch_size

        if grid_h * grid_w != num_patches:
            raise ValueError(
                f"Expected {grid_h * grid_w} patch tokens; "
                f"got {num_patches} (input {input_height}x{input_width}, "
                f"patch_size={patch_size}, registers={self.num_register_tokens})"
            )

        feature_map = patch_tokens.permute(0, 2, 1).reshape(batch_size, -1, grid_h, grid_w)
        logits_lowres = self.classifier(feature_map)

        target_size = labels.shape[-2:] if labels is not None else (input_height, input_width)
        logits = F.interpolate(
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
            attentions=outputs.attentions if output_attentions else None,
        )

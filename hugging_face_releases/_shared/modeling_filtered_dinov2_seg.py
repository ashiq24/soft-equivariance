"""HuggingFace remote-code wrapper for filtered DINOv2 segmentation."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel

from configuration_softeq import SoftEqConfig
from models.filtered_dino2_seg import create_filtered_dino2_seg


class FilteredDinov2BaseSeg(PreTrainedModel):
    config_class = SoftEqConfig
    base_model_prefix = "model"

    def __init__(self, config: SoftEqConfig) -> None:
        super().__init__(config)
        model_kwargs = config.to_model_kwargs()
        model_kwargs["type"] = "filtered_dino2_seg"
        self.model = create_filtered_dino2_seg(model_kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

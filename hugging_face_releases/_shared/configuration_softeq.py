"""HuggingFace config helpers for filtered SoftEq models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from transformers import PretrainedConfig


MODEL_PARAM_KEYS = {
    "type",
    "pretrained_model",
    "num_labels",
    "filter_patch_embeddings",
    "filter_attention_qkv",
    "filter_attention_output",
    "filter_mlp",
    "group_type",
    "n_rotations",
    "soft_thresholding",
    "soft_thresholding_pos",
    "decomposition_method",
    "hard_mask",
    "preserve_norm",
    "joint_decomposition",
    "attention_output_filter_list",
    "soft_thresholding_attention_output",
    "ignore_index",
    "load_pretrained_weight",
    "freeze_patch_embeddings",
    "freeze_position_embeddings",
    "freeze_filters",
    "min_filter_size",
}


class SoftEqConfig(PretrainedConfig):
    """Config that intentionally persists only architecture/filter parameters."""

    model_type = "filtered-vit-base-patch16-224"

    def __init__(
        self,
        model_type: str = "filtered-vit-base-patch16-224",
        pretrained_model: str = "google/vit-base-patch16-224",
        num_labels: int = 1000,
        filter_patch_embeddings: bool = True,
        filter_attention_qkv: bool = False,
        filter_attention_output: bool = False,
        filter_mlp: bool = False,
        group_type: str = "rotation",
        n_rotations: int = 4,
        soft_thresholding: float = 0.0,
        soft_thresholding_pos: float = 0.0,
        decomposition_method: str = "schur",
        hard_mask: bool = False,
        preserve_norm: bool = False,
        joint_decomposition: bool = True,
        attention_output_filter_list: list[int] | None = None,
        soft_thresholding_attention_output: float = 0.1,
        ignore_index: int = 255,
        load_pretrained_weight: bool = False,
        freeze_patch_embeddings: bool = False,
        freeze_position_embeddings: bool = False,
        freeze_filters: bool = False,
        min_filter_size: int = 1,
        model_class: str = "",
        task: str = "classification",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.num_labels = num_labels
        self.filter_patch_embeddings = filter_patch_embeddings
        self.filter_attention_qkv = filter_attention_qkv
        self.filter_attention_output = filter_attention_output
        self.filter_mlp = filter_mlp
        self.group_type = group_type
        self.n_rotations = n_rotations
        self.soft_thresholding = soft_thresholding
        self.soft_thresholding_pos = soft_thresholding_pos
        self.decomposition_method = decomposition_method
        self.hard_mask = hard_mask
        self.preserve_norm = preserve_norm
        self.joint_decomposition = joint_decomposition
        self.attention_output_filter_list = attention_output_filter_list or []
        self.soft_thresholding_attention_output = soft_thresholding_attention_output
        self.ignore_index = ignore_index
        self.load_pretrained_weight = load_pretrained_weight
        self.freeze_patch_embeddings = freeze_patch_embeddings
        self.freeze_position_embeddings = freeze_position_embeddings
        self.freeze_filters = freeze_filters
        self.min_filter_size = min_filter_size
        self.model_class = model_class
        self.task = task

    @classmethod
    def from_model_cfg(cls, cfg: Dict[str, Any]) -> "SoftEqConfig":
        """Build a HF config from project config dict by filtering model keys only."""
        raw_model_cfg = deepcopy(cfg.get("model", cfg))
        filtered = {k: v for k, v in raw_model_cfg.items() if k in MODEL_PARAM_KEYS}
        if "pretrained_model" not in filtered:
            raise ValueError("model config must include `pretrained_model`")
        return cls(**filtered)

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Convert config to kwargs expected by model factory functions."""
        return {
            "pretrained_model": self.pretrained_model,
            "num_labels": self.num_labels,
            "filter_patch_embeddings": self.filter_patch_embeddings,
            "filter_attention_qkv": self.filter_attention_qkv,
            "filter_attention_output": self.filter_attention_output,
            "filter_mlp": self.filter_mlp,
            "group_type": self.group_type,
            "n_rotations": self.n_rotations,
            "soft_thresholding": self.soft_thresholding,
            "soft_thresholding_pos": self.soft_thresholding_pos,
            "decomposition_method": self.decomposition_method,
            "hard_mask": self.hard_mask,
            "preserve_norm": self.preserve_norm,
            "joint_decomposition": self.joint_decomposition,
            "attention_output_filter_list": self.attention_output_filter_list,
            "soft_thresholding_attention_output": self.soft_thresholding_attention_output,
            "ignore_index": self.ignore_index,
            "load_pretrained_weight": self.load_pretrained_weight,
            "freeze_patch_embeddings": self.freeze_patch_embeddings,
            "freeze_position_embeddings": self.freeze_position_embeddings,
            "freeze_filters": self.freeze_filters,
            "min_filter_size": self.min_filter_size,
        }

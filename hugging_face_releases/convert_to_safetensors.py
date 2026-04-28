#!/usr/bin/env python3
"""Convert project checkpoints (.pt) to HuggingFace-ready safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import save_file


MODEL_ONLY_KEYS = {
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


AUTO_MAP_BY_TYPE = {
    "filtered_vit": {
        "AutoModel": "modeling_filtered_vit.FilteredVitBasePatch16_224",
        "AutoConfig": "configuration_softeq.SoftEqConfig",
    },
    "filtered_dinov2": {
        "AutoModel": "modeling_filtered_dinov2.FilteredDinov2Base",
        "AutoConfig": "configuration_softeq.SoftEqConfig",
    },
    "filtered_vit_seg": {
        "AutoModel": "modeling_filtered_vit_seg.FilteredVitBasePatch16_224Seg",
        "AutoConfig": "configuration_softeq.SoftEqConfig",
    },
    "filtered_dino2_seg": {
        "AutoModel": "modeling_filtered_dinov2_seg.FilteredDinov2BaseSeg",
        "AutoConfig": "configuration_softeq.SoftEqConfig",
    },
}


MODEL_TYPE_NAME_BY_TYPE = {
    "filtered_vit": "filtered-vit-base-patch16-224",
    "filtered_dinov2": "filtered-dinov2-base",
    "filtered_vit_seg": "filtered-vit-base-patch16-224-seg",
    "filtered_dino2_seg": "filtered-dinov2-base-seg",
}


def _extract_model_cfg(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    cfg = checkpoint.get("cfg", {})
    if "model" in cfg:
        cfg = cfg["model"]
    model_cfg = {k: v for k, v in cfg.items() if k in MODEL_ONLY_KEYS}
    if not model_cfg:
        raise ValueError(
            "Could not find model config in checkpoint['cfg']. "
            "Pass --model-config-json to provide the model settings."
        )
    return model_cfg


def _load_model_cfg_arg(model_config_json: str | None) -> Dict[str, Any]:
    if not model_config_json:
        return {}
    model_cfg = json.loads(model_config_json)
    return {k: v for k, v in model_cfg.items() if k in MODEL_ONLY_KEYS or k == "type"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .pt checkpoint to model.safetensors + config.json")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output model directory")
    parser.add_argument(
        "--model-config-json",
        default=None,
        help="Optional JSON string with model-only config (used if checkpoint lacks cfg/model keys).",
    )
    parser.add_argument("--dataset-tag", default=None, help="Optional dataset tag for metadata only.")
    parser.add_argument("--task", choices=["classification", "segmentation"], default="classification")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError("Checkpoint does not contain `model_state_dict` or `state_dict`.")

    model_cfg = _load_model_cfg_arg(args.model_config_json)
    if not model_cfg:
        model_cfg = _extract_model_cfg(checkpoint)

    model_kind = model_cfg.get("type")
    if not model_kind:
        raise ValueError("Model config must include model `type`.")
    if model_kind not in AUTO_MAP_BY_TYPE:
        raise ValueError(f"Unsupported model type `{model_kind}` for HF release conversion.")

    safetensors_path = output_dir / "model.safetensors"
    save_file(state_dict, str(safetensors_path))

    config = {
        "model_type": MODEL_TYPE_NAME_BY_TYPE[model_kind],
        "task": args.task,
        "dataset_tag": args.dataset_tag,
        **model_cfg,
        "auto_map": AUTO_MAP_BY_TYPE[model_kind],
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saved: {safetensors_path}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()

"""
package_model.py
-----------------
Assemble a self-contained HuggingFace model folder from _shared/ code and a
converted model.safetensors file.

The resulting folder can be directly uploaded to HuggingFace Hub or loaded
locally with ``from_pretrained(..., trust_remote_code=True)``.

Output folder structure
-----------------------
<output_dir>/
  config.json                    # model + equivariance config (no training params)
  model.safetensors              # converted weights
  configuration_softeq.py       # SoftEqConfig class
  modeling_filtered_vit.py      # FilteredViT (classification only)
  modeling_filtered_dinov2.py   # FilteredDinoV2 / FilteredDinoV2wRegister (classification)
  modeling_filtered_vit_seg.py  # FilteredViTSeg (segmentation only)
  modeling_filtered_dinov2_seg.py  # FilteredDino2Seg / wRegister (segmentation)
  filtered_layers_vit.py        # ViT monkeypatch helpers
  filtered_layers_dinov2.py     # DINOv2 monkeypatch helpers
  softeq/                       # Soft-equivariance core (filters, projectors)

Naming convention
-----------------
Classification (ImageNet):
  filtered-{backbone}-imagenet-c{n_rotations}-s{soft_thresholding}
  e.g. filtered-vit-base-patch16-224-imagenet-c4-s0.0

Segmentation:
  filtered-{backbone}-{dataset}-seg-c{n_rotations}-s{soft_thresholding}
  e.g. filtered-dinov2-base-voc-seg-c4-s0.7

Usage
-----
    python package_model.py \
        --safetensors path/to/model.safetensors \
        --output_dir  hugging_face_releases/filtered-vit-base-patch16-224-imagenet-c4-s0.0 \
        --model_arch  filtered_vit \
        --pretrained_model google/vit-base-patch16-224 \
        --num_labels  1000 \
        --n_rotations 4 \
        --soft_thresholding 0.0 \
        --soft_thresholding_pos 0.0

    # DINOv2 classification:
    python package_model.py \
        --safetensors path/to/model.safetensors \
        --output_dir  hugging_face_releases/filtered-dinov2-base-imagenet-c4-s0.0 \
        --model_arch  filtered_dinov2 \
        --pretrained_model facebook/dinov2-base \
        --num_labels  1000 \
        --n_rotations 4 \
        --soft_thresholding 0.0

    # ViT segmentation (PASCAL VOC 21 classes):
    python package_model.py \
        --safetensors path/to/best.safetensors \
        --output_dir  hugging_face_releases/filtered-vit-base-patch16-224-voc-seg-c4-s0.0 \
        --model_arch  filtered_vit_seg \
        --pretrained_model google/vit-base-patch16-224 \
        --num_labels  21 \
        --n_rotations 4 \
        --soft_thresholding 0.0
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


SHARED_DIR = Path(__file__).parent / "_shared"

# Files that must be copied to every packaged model folder.
SHARED_FILES = [
    "configuration_softeq.py",
    "modeling_filtered_vit.py",
    "modeling_filtered_dinov2.py",
    "modeling_filtered_vit_seg.py",
    "modeling_filtered_dinov2_seg.py",
    "filtered_layers_vit.py",
    "filtered_layers_dinov2.py",
]

# Architecture → primary modeling file name
ARCH_TO_MODULE = {
    "filtered_vit":       "modeling_filtered_vit",
    "filtered_dinov2":    "modeling_filtered_dinov2",
    "filtered_vit_seg":   "modeling_filtered_vit_seg",
    "filtered_dino2_seg": "modeling_filtered_dinov2_seg",
}

# Architecture → HuggingFace class name in that module
ARCH_TO_CLASS = {
    "filtered_vit":       "FilteredViT",
    "filtered_dinov2":    "FilteredDinoV2",
    "filtered_vit_seg":   "FilteredViTSeg",
    "filtered_dino2_seg": "FilteredDino2Seg",
}


def build_config_dict(args) -> dict:
    """Build the config.json payload from command-line arguments."""
    return {
        "model_type": "soft_equivariant",
        "model_arch": args.model_arch,
        "pretrained_model": args.pretrained_model,
        "num_labels": args.num_labels,
        # ── Core equivariance parameters ────────────────────────────────────
        "n_rotations": args.n_rotations,
        "soft_thresholding": args.soft_thresholding,
        "soft_thresholding_pos": args.soft_thresholding_pos,
        "group_type": args.group_type,
        "decomposition_method": args.decomposition_method,
        # ── Filter flags ─────────────────────────────────────────────────────
        "filter_patch_embeddings": args.filter_patch_embeddings,
        "filter_attention_qkv": False,
        "filter_attention_output": False,
        "filter_mlp": False,
        "attention_output_filter_list": [],
        "soft_thresholding_attention_output": 0.1,
        # ── Architecture options ─────────────────────────────────────────────
        "preserve_norm": args.preserve_norm,
        "hard_mask": args.hard_mask,
        "joint_decomposition": args.joint_decomposition,
        "freeze_patch_embeddings": False,
        "freeze_position_embeddings": False,
        # ── Segmentation ─────────────────────────────────────────────────────
        "ignore_index": args.ignore_index,
        # ── HuggingFace auto-map ─────────────────────────────────────────────
        "auto_map": {
            "AutoConfig": "configuration_softeq.SoftEqConfig",
            "AutoModel":  f"{ARCH_TO_MODULE[args.model_arch]}.{ARCH_TO_CLASS[args.model_arch]}",
        },
        # ── Transformers metadata ────────────────────────────────────────────
        "transformers_version": _get_transformers_version(),
        "torch_dtype": "float32",
    }


def _get_transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unknown"


def package_model(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy model.safetensors
    if args.safetensors:
        src_st = Path(args.safetensors)
        if not src_st.exists():
            sys.exit(f"ERROR: safetensors file not found: {src_st}")
        dst_st = output_dir / "model.safetensors"
        if src_st.resolve() != dst_st.resolve():
            shutil.copy2(src_st, dst_st)
            print(f"Copied model.safetensors  ({dst_st.stat().st_size / 1024**2:.1f} MB)")

    # 2. Write config.json
    config_dict = build_config_dict(args)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Written  config.json")

    # 3. Copy shared Python files
    for fname in SHARED_FILES:
        src = SHARED_DIR / fname
        if not src.exists():
            print(f"WARNING: shared file not found, skipping: {src}")
            continue
        shutil.copy2(src, output_dir / fname)
        print(f"Copied   {fname}")

    # 4. Copy softeq/ package
    src_softeq = SHARED_DIR / "softeq"
    dst_softeq = output_dir / "softeq"
    if dst_softeq.exists():
        shutil.rmtree(dst_softeq)
    shutil.copytree(src_softeq, dst_softeq,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Copied   softeq/")

    print(f"\nModel folder ready: {output_dir}")
    print("\nTo upload to HuggingFace Hub:")
    print(f"  huggingface-cli upload <your-username>/<repo-name> {output_dir}")
    print("\nTo load locally:")
    print(f"  model = AutoModel.from_pretrained(\"{output_dir}\", trust_remote_code=True)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package a trained soft-equivariant model for HuggingFace release."
    )

    # Required
    parser.add_argument("--output_dir", required=True,
                        help="Destination folder for the packaged model.")
    parser.add_argument("--model_arch", required=True,
                        choices=["filtered_vit", "filtered_dinov2",
                                 "filtered_vit_seg", "filtered_dino2_seg"],
                        help="Model architecture variant.")
    parser.add_argument("--pretrained_model", required=True,
                        help="HuggingFace backbone id (e.g. google/vit-base-patch16-224).")
    parser.add_argument("--num_labels", type=int, required=True,
                        help="Number of output classes (1000 for ImageNet, 21 for PASCAL VOC).")

    # Core equivariance
    parser.add_argument("--n_rotations", type=int, default=4,
                        help="Number of discrete rotations (e.g. 4 for C4, 180 for C180).")
    parser.add_argument("--soft_thresholding", type=float, default=0.0,
                        help="Softness of patch-embedding filter [0..1].")
    parser.add_argument("--soft_thresholding_pos", type=float, default=0.0,
                        help="Softness of positional-embedding filter [0..1].")
    parser.add_argument("--group_type", default="rotation",
                        choices=["rotation", "roto_reflection"],
                        help="Symmetry group type.")
    parser.add_argument("--decomposition_method", default="schur",
                        choices=["schur", "svd"],
                        help="Basis decomposition algorithm.")

    # Filter flags
    parser.add_argument("--filter_patch_embeddings", action="store_true", default=True)
    parser.add_argument("--no_filter_patch_embeddings",
                        dest="filter_patch_embeddings", action="store_false")

    # Architecture options
    parser.add_argument("--preserve_norm", action="store_true", default=False)
    parser.add_argument("--hard_mask", action="store_true", default=False)
    parser.add_argument("--joint_decomposition", action="store_true", default=True)
    parser.add_argument("--no_joint_decomposition",
                        dest="joint_decomposition", action="store_false")
    parser.add_argument("--ignore_index", type=int, default=255,
                        help="Ignored label index for segmentation loss.")

    # Weights
    parser.add_argument("--safetensors", default=None,
                        help="Path to model.safetensors. If omitted, only config/code are copied.")

    args = parser.parse_args()

    if args.model_arch not in ARCH_TO_MODULE:
        sys.exit(f"ERROR: Unknown model_arch \"{args.model_arch}\"")

    package_model(args)

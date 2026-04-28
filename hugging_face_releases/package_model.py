#!/usr/bin/env python3
"""Assemble a self-contained HuggingFace model release folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SHARED_MODELING_BY_TYPE = {
    "filtered_vit": "modeling_filtered_vit.py",
    "filtered_dinov2": "modeling_filtered_dinov2.py",
    "filtered_vit_seg": "modeling_filtered_vit_seg.py",
    "filtered_dino2_seg": "modeling_filtered_dinov2_seg.py",
}


PROJECT_MODEL_FILES = [
    "models/__init__.py",
    "models/filtered_vit.py",
    "models/filtered_dino2.py",
    "models/filtered_vit_seg.py",
    "models/filtered_dino2_seg.py",
    "models/filtered_layers.py",
    "models/filtered_layers_vit.py",
    "models/filtered_layers_dinov2.py",
]


def copy_file(rel_path: str, src_root: Path, dst_root: Path) -> None:
    src = src_root / rel_path
    dst = dst_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_model_card(output_dir: Path, repo_name: str, model_type: str) -> None:
    text = (
        f"# {repo_name}\n\n"
        "Filtered soft-equivariant release for the CVPR public codebase.\n\n"
        "## Usage\n\n"
        "```python\n"
        "from transformers import AutoModel\n\n"
        f'model = AutoModel.from_pretrained("{repo_name}", trust_remote_code=True)\n'
        "```\n\n"
        "The release config includes only model/filter/projection parameters and intentionally excludes "
        "training hyperparameters.\n"
        f"\nModel family: `{model_type}`\n"
    )
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package self-contained HF model folder")
    parser.add_argument("--repo-root", required=True, help="Project root path")
    parser.add_argument("--variant-name", required=True, help="Release folder name")
    parser.add_argument("--source-dir", required=True, help="Folder containing config.json and model.safetensors")
    parser.add_argument("--write-readme", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source_dir = Path(args.source_dir).resolve()
    release_root = repo_root / "hugging_face_releases"
    output_dir = release_root / args.variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = source_dir / "config.json"
    model_path = source_dir / "model.safetensors"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError("source-dir must contain both config.json and model.safetensors")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("type")
    if model_type not in SHARED_MODELING_BY_TYPE:
        raise ValueError(f"Unsupported model type in config.json: {model_type}")

    shutil.copy2(config_path, output_dir / "config.json")
    shutil.copy2(model_path, output_dir / "model.safetensors")

    shared_root = release_root / "_shared"
    shared_files = [
        "configuration_softeq.py",
        SHARED_MODELING_BY_TYPE[model_type],
    ]
    for filename in shared_files:
        shutil.copy2(shared_root / filename, output_dir / filename)

    for rel_path in PROJECT_MODEL_FILES:
        copy_file(rel_path, repo_root, output_dir)

    shared_softeq_root = shared_root / "softeq"
    if not shared_softeq_root.exists():
        raise FileNotFoundError("Expected curated softeq files in hugging_face_releases/_shared/softeq")
    shutil.copytree(shared_softeq_root, output_dir / "softeq", dirs_exist_ok=True)

    if args.write_readme:
        write_model_card(output_dir, args.variant_name, model_type)

    print(f"Packaged model folder: {output_dir}")


if __name__ == "__main__":
    main()

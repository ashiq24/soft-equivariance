"""
convert_to_safetensors.py
--------------------------
Convert a .pt training checkpoint to model.safetensors.

Extracts only the model state dict (no optimizer state, no training metadata)
and writes it as a safetensors file, which is the preferred format for
HuggingFace model releases.

Supported checkpoint formats
-----------------------------
* Segmentation trainer:  checkpoint["model_state_dict"]
* timm CheckpointSaver:  checkpoint["state_dict"] or checkpoint["model"]
* Raw state dict:        checkpoint (top-level dict of tensors)

Usage
-----
    python convert_to_safetensors.py \
        --checkpoint path/to/best.pt \
        --output_dir path/to/output_folder

    # Optionally verify key counts:
    python convert_to_safetensors.py \
        --checkpoint best.pt \
        --output_dir ./out \
        --print_keys
"""

import argparse
import os

import torch
from safetensors.torch import save_file, load_file


def extract_state_dict(checkpoint_path: str) -> dict:
    """Load a .pt checkpoint and return the model state dict."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        # Try common key names in order of priority.
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                print(f"  Found state dict under key: \"{key}\"")
                state_dict = ckpt[key]
                break
        else:
            # Assume the checkpoint IS the state dict (all values are tensors).
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                print("  Checkpoint appears to be a raw state dict.")
                state_dict = ckpt
            else:
                available = [k for k in ckpt.keys() if not isinstance(ckpt[k], torch.Tensor)]
                raise KeyError(
                    f"Cannot find model weights. Top-level keys: {list(ckpt.keys())}\n"
                    f"Non-tensor keys: {available}"
                )
    else:
        raise TypeError(f"Expected a dict checkpoint, got {type(ckpt)}")

    # Ensure all values are contiguous float tensors (safetensors requirement).
    clean = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.contiguous()
    print(f"  State dict has {len(clean)} parameter/buffer tensors.")
    return clean


def convert(checkpoint_path: str, output_dir: str, print_keys: bool = False):
    """Convert a .pt checkpoint to model.safetensors in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    state_dict = extract_state_dict(checkpoint_path)

    if print_keys:
        print("\nState dict keys:")
        for k in sorted(state_dict.keys()):
            print(f"  {k:80s}  {tuple(state_dict[k].shape)}")

    out_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")


def verify(safetensors_path: str):
    """Quick sanity check: load the safetensors file and print key count."""
    tensors = load_file(safetensors_path)
    print(f"Verification: loaded {len(tensors)} tensors from {safetensors_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt checkpoint to model.safetensors")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--output_dir", required=True, help="Directory to write model.safetensors")
    parser.add_argument(
        "--print_keys", action="store_true", default=False,
        help="Print all state dict keys and shapes",
    )
    parser.add_argument(
        "--verify", action="store_true", default=False,
        help="Verify the output safetensors file after writing",
    )
    args = parser.parse_args()

    convert(args.checkpoint, args.output_dir, args.print_keys)

    if args.verify:
        verify(os.path.join(args.output_dir, "model.safetensors"))

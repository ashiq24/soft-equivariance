"""
test_release.py
----------------
Verify that a packaged HuggingFace model produces numerically identical outputs
to the original training-codebase model loaded from a .pt checkpoint.

Run this BEFORE uploading any model to HuggingFace Hub.

What it checks
--------------
1. Both models load without errors.
2. State-dict key sets match (same parameter names, same shapes).
3. Forward pass outputs (logits) are identical to within floating-point precision
   (max absolute difference < 1e-4 for fp32 models).
4. Optionally: tests on a real image from disk.

Usage
-----
    # Classification — random input
    python hugging_face_releases/test_release.py \\
        --model_arch  filtered_vit \\
        --checkpoint  path/to/best.pt \\
        --hf_dir      hugging_face_releases/filtered-vit-base-patch16-224-imagenet-c4-s0.0 \\
        --pretrained_model google/vit-base-patch16-224 \\
        --num_labels  1000 \\
        --n_rotations 4 \\
        --soft_thresholding 0.0 \\
        --soft_thresholding_pos 0.0

    # Segmentation — real image
    python hugging_face_releases/test_release.py \\
        --model_arch  filtered_vit_seg \\
        --checkpoint  path/to/best.pt \\
        --hf_dir      hugging_face_releases/filtered-vit-base-patch16-224-voc-seg-c4-s0.0 \\
        --pretrained_model google/vit-base-patch16-224 \\
        --num_labels  21 \\
        --n_rotations 4 \\
        --soft_thresholding 0.0 \\
        --image path/to/test_image.jpg

The script must be run from the repository root so that `models/` and `softeq/`
are importable.
"""

import argparse
import os
import sys
import json

import torch
import torch.nn.functional as F

# ── Make the repo root importable ────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Make the _shared/ directory importable for the HF model ──────────────────
SHARED_DIR = os.path.join(os.path.dirname(__file__), "_shared")
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Original model loading (from training codebase + .pt checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def _load_state_dict_from_pt(checkpoint_path: str) -> dict:
    """Extract model state dict from a .pt checkpoint (same logic as convert script)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        raise KeyError(
            f"Cannot find model weights. Top-level keys: {list(ckpt.keys())}"
        )
    raise TypeError(f"Expected dict checkpoint, got {type(ckpt)}")


def load_original_model(args):
    """
    Instantiate a model from the training codebase and load .pt weights.

    The model is built with load_pretrained_weight=False so the backbone weights
    come entirely from the checkpoint, not re-downloaded from HF Hub.
    """
    arch = args.model_arch
    common = dict(
        pretrained_model_name=args.pretrained_model,
        num_labels=args.num_labels,
        n_rotations=args.n_rotations,
        soft_thresholding=args.soft_thresholding,
        soft_thresholding_pos=args.soft_thresholding_pos,
        group_type=args.group_type,
        decomposition_method=args.decomposition_method,
        preserve_norm=args.preserve_norm,
        hard_mask=args.hard_mask,
        joint_decomposition=args.joint_decomposition,
        load_pretrained_weight=False,   # weights come from .pt checkpoint below
    )

    print("\n── Loading ORIGINAL model (codebase) ──────────────────────────────")
    if arch == "filtered_vit":
        from models.filtered_vit import FilteredViT
        model = FilteredViT(**common)

    elif arch == "filtered_dinov2":
        from models.filtered_dino2 import create_filtered_dinov2
        cfg = {
            "pretrained_model": args.pretrained_model,
            "num_labels": args.num_labels,
            "n_rotations": args.n_rotations,
            "soft_thresholding": args.soft_thresholding,
            "soft_thresholding_pos": args.soft_thresholding_pos,
            "group_type": args.group_type,
            "decomposition_method": args.decomposition_method,
            "preserve_norm": args.preserve_norm,
            "hard_mask": args.hard_mask,
            "joint_decomposition": args.joint_decomposition,
            "load_pretrained_weight": False,
            "filter_patch_embeddings": True,
        }
        model = create_filtered_dinov2(cfg)

    elif arch == "filtered_vit_seg":
        from models.filtered_vit_seg import FilteredViTSeg
        model = FilteredViTSeg(
            pretrained_model_name=args.pretrained_model,
            num_labels=args.num_labels,
            n_rotations=args.n_rotations,
            soft_thresholding=args.soft_thresholding,
            soft_thresholding_pos=args.soft_thresholding_pos,
            group_type=args.group_type,
            decomposition_method=args.decomposition_method,
            preserve_norm=args.preserve_norm,
            hard_mask=args.hard_mask,
            joint_decomposition=args.joint_decomposition,
            load_pretrained_weight=False,
            ignore_index=args.ignore_index,
        )

    elif arch == "filtered_dino2_seg":
        from models.filtered_dino2_seg import create_filtered_dino2_seg
        cfg = {
            "pretrained_model": args.pretrained_model,
            "num_labels": args.num_labels,
            "n_rotations": args.n_rotations,
            "soft_thresholding": args.soft_thresholding,
            "soft_thresholding_pos": args.soft_thresholding_pos,
            "group_type": args.group_type,
            "decomposition_method": args.decomposition_method,
            "preserve_norm": args.preserve_norm,
            "hard_mask": args.hard_mask,
            "joint_decomposition": args.joint_decomposition,
            "load_pretrained_weight": False,
            "filter_patch_embeddings": True,
            "ignore_index": args.ignore_index,
        }
        model = create_filtered_dino2_seg(cfg)

    else:
        raise ValueError(f"Unknown model_arch: {arch}")

    # Load .pt weights
    print(f"Loading .pt checkpoint: {args.checkpoint}")
    state_dict = _load_state_dict_from_pt(args.checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING — missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  WARNING — unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"  Loaded {len(state_dict)} tensors from checkpoint.")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace model loading (from packaged folder with model.safetensors)
# ─────────────────────────────────────────────────────────────────────────────

def load_hf_model(hf_dir: str):
    """Load the packaged HF model using AutoModel with trust_remote_code=True."""
    # Add the model directory to sys.path so local imports resolve.
    if hf_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(hf_dir))

    from transformers import AutoModel, AutoConfig

    print("\n── Loading HF model (packaged folder) ─────────────────────────────")
    print(f"Folder: {hf_dir}")
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_dir, trust_remote_code=True, config=config)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# State-dict comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_state_dicts(orig_sd: dict, hf_sd: dict) -> bool:
    """Compare key sets and tensor shapes between two state dicts."""
    orig_keys = set(orig_sd.keys())
    hf_keys   = set(hf_sd.keys())

    only_orig = orig_keys - hf_keys
    only_hf   = hf_keys   - orig_keys
    shared    = orig_keys & hf_keys

    print(f"\n── State-dict comparison ───────────────────────────────────────────")
    print(f"  Original keys : {len(orig_keys)}")
    print(f"  HF model keys : {len(hf_keys)}")
    print(f"  Shared keys   : {len(shared)}")

    ok = True
    if only_orig:
        print(f"  Keys only in ORIGINAL ({len(only_orig)}): {sorted(only_orig)[:5]}")
        ok = False
    if only_hf:
        print(f"  Keys only in HF MODEL ({len(only_hf)}): {sorted(only_hf)[:5]}")
        ok = False

    shape_mismatches = []
    for k in sorted(shared):
        if orig_sd[k].shape != hf_sd[k].shape:
            shape_mismatches.append((k, orig_sd[k].shape, hf_sd[k].shape))
    if shape_mismatches:
        print(f"  Shape mismatches ({len(shape_mismatches)}):")
        for k, s1, s2 in shape_mismatches[:5]:
            print(f"    {k}: orig={s1}  hf={s2}")
        ok = False

    if ok:
        print("  ✓ Key sets and shapes match.")
    return ok


def compare_tensor_values(orig_sd: dict, hf_sd: dict) -> float:
    """Return the max absolute difference across all shared tensors."""
    shared = set(orig_sd.keys()) & set(hf_sd.keys())
    max_diff = 0.0
    worst_key = ""
    for k in sorted(shared):
        diff = (orig_sd[k].float() - hf_sd[k].float()).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            worst_key = k
    if worst_key:
        print(f"  Max weight diff: {max_diff:.4e}  (key: {worst_key})")
    return max_diff


# ─────────────────────────────────────────────────────────────────────────────
# Forward-pass comparison
# ─────────────────────────────────────────────────────────────────────────────

def build_input(args) -> torch.Tensor:
    """Return a [1, 3, H, W] input tensor."""
    if args.image:
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(args.image).convert("RGB")
            size = 224 if "seg" not in args.model_arch else 512
            transform = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = transform(img).unsqueeze(0)
            print(f"Using real image: {args.image}  (resized to {size}×{size})")
            return tensor
        except Exception as e:
            print(f"WARNING: could not load image ({e}), falling back to random input.")

    # Reproducible random input
    torch.manual_seed(42)
    h = w = 224 if "seg" not in args.model_arch else 512
    tensor = torch.randn(1, 3, h, w)
    print(f"Using random input: shape {list(tensor.shape)}  (seed=42)")
    return tensor


def run_forward(model, pixel_values: torch.Tensor):
    """Run a model forward pass and return the logits tensor."""
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    # Both HF PreTrainedModel and original models return an object with .logits
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 68)
    print("  Soft-Equivariant Model Release Verification")
    print(f"  arch={args.model_arch}  n_rot={args.n_rotations}"
          f"  s={args.soft_thresholding}  s_pos={args.soft_thresholding_pos}")
    print("=" * 68)

    # ── 1. Load both models ───────────────────────────────────────────────────
    orig_model = load_original_model(args)
    hf_model   = load_hf_model(args.hf_dir)

    # ── 2. Compare state dicts ────────────────────────────────────────────────
    print("\n── Comparing state dicts ───────────────────────────────────────────")
    orig_sd = orig_model.state_dict()
    hf_sd   = hf_model.state_dict()
    keys_ok = compare_state_dicts(orig_sd, hf_sd)
    weight_diff = compare_tensor_values(orig_sd, hf_sd)

    # ── 3. Compare forward-pass outputs ──────────────────────────────────────
    print("\n── Comparing forward-pass outputs ──────────────────────────────────")
    pixel_values = build_input(args)
    orig_logits  = run_forward(orig_model, pixel_values)
    hf_logits    = run_forward(hf_model,   pixel_values)

    max_logit_diff = (orig_logits.float() - hf_logits.float()).abs().max().item()
    mean_logit_diff = (orig_logits.float() - hf_logits.float()).abs().mean().item()

    print(f"  Original logits  : shape={list(orig_logits.shape)}  "
          f"min={orig_logits.min():.4f}  max={orig_logits.max():.4f}")
    print(f"  HF model logits  : shape={list(hf_logits.shape)}  "
          f"min={hf_logits.min():.4f}  max={hf_logits.max():.4f}")
    print(f"  Max absolute diff: {max_logit_diff:.4e}")
    print(f"  Mean absolute diff: {mean_logit_diff:.4e}")

    # Top-1 predictions
    if orig_logits.dim() == 2:
        orig_top1 = orig_logits.argmax(dim=-1).item()
        hf_top1   = hf_logits.argmax(dim=-1).item()
        print(f"  Original top-1   : class {orig_top1}")
        print(f"  HF model  top-1  : class {hf_top1}")

    # ── 4. Pass / Fail ────────────────────────────────────────────────────────
    THRESHOLD = 1e-4
    print("\n" + "=" * 68)
    passed = max_logit_diff < THRESHOLD and keys_ok

    if passed:
        print(f"  ✓ Max absolute logit difference: {max_logit_diff:.4e}  "
              f"[PASS — outputs are identical]")
    else:
        if not keys_ok:
            print("  ✗ FAIL — state-dict key mismatch (see above)")
        if max_logit_diff >= THRESHOLD:
            print(f"  ✗ FAIL — logit difference {max_logit_diff:.4e} exceeds threshold {THRESHOLD:.0e}")
        print("  DO NOT upload this model to HuggingFace.")
        sys.exit(1)

    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify HF packaged model matches original .pt checkpoint."
    )

    # Required
    parser.add_argument("--model_arch", required=True,
                        choices=["filtered_vit", "filtered_dinov2",
                                 "filtered_vit_seg", "filtered_dino2_seg"],
                        help="Model architecture variant.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the original .pt training checkpoint.")
    parser.add_argument("--hf_dir", required=True,
                        help="Path to the packaged HF model folder (output of package_model.py).")
    parser.add_argument("--pretrained_model", required=True,
                        help="HuggingFace backbone id (must match training config).")
    parser.add_argument("--num_labels", type=int, required=True,
                        help="Number of output classes.")

    # Equivariance parameters (must match training config exactly)
    parser.add_argument("--n_rotations", type=int, default=4)
    parser.add_argument("--soft_thresholding", type=float, default=0.0)
    parser.add_argument("--soft_thresholding_pos", type=float, default=0.0)
    parser.add_argument("--group_type", default="rotation",
                        choices=["rotation", "roto_reflection"])
    parser.add_argument("--decomposition_method", default="schur",
                        choices=["schur", "svd"])
    parser.add_argument("--preserve_norm", action="store_true", default=False)
    parser.add_argument("--hard_mask", action="store_true", default=False)
    parser.add_argument("--joint_decomposition", action="store_true", default=True)
    parser.add_argument("--no_joint_decomposition",
                        dest="joint_decomposition", action="store_false")
    parser.add_argument("--ignore_index", type=int, default=255)

    # Optional real image
    parser.add_argument("--image", default=None,
                        help="Path to a real image (jpg/png) for testing instead of random noise.")

    args = parser.parse_args()

    # Validate paths
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.hf_dir):
        sys.exit(f"ERROR: HF model folder not found: {args.hf_dir}")
    if not os.path.isfile(os.path.join(args.hf_dir, "model.safetensors")):
        sys.exit(f"ERROR: model.safetensors not found in {args.hf_dir}. Run convert_to_safetensors.py first.")
    if not os.path.isfile(os.path.join(args.hf_dir, "config.json")):
        sys.exit(f"ERROR: config.json not found in {args.hf_dir}. Run package_model.py first.")

    main(args)

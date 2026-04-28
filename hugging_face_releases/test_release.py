"""
test_release.py
----------------
Verify that a packaged HuggingFace model produces numerically identical outputs
to the original training-codebase model loaded from a .pt checkpoint.

The original model is instantiated from the same YAML config used for training
(via --config and --config_name, exactly like the training scripts use), so
there is no risk of mismatched architecture parameters.

Run this BEFORE uploading any model to HuggingFace Hub.

What it checks
--------------
1. Both models load without errors.
2. State-dict key sets and shapes match.
3. Forward-pass outputs (logits) are numerically identical within floating-point
   precision (max absolute difference < 1e-4).
4. Optionally: real image test with --image.

Usage (must be run from repository root)
-----------------------------------------
    # ViT classification — use the exact config + config_name from training
    python hugging_face_releases/test_release.py \\
        --config      config/imagenet_configs.yaml \\
        --config_name imagenet1k_vit \\
        --checkpoint  path/to/best.pt \\
        --hf_dir      hugging_face_releases/filtered-vit-base-patch16-224-imagenet-c4-s0.0

    # Override soft_thresholding at test time (same as training scripts allow)
    python hugging_face_releases/test_release.py \\
        --config      config/imagenet_configs.yaml \\
        --config_name imagenet1k_vit_c18 \\
        --soft_thresholding 0.7 \\
        --soft_thresholding_pos 0.7 \\
        --checkpoint  path/to/best.pt \\
        --hf_dir      hugging_face_releases/filtered-vit-base-patch16-224-imagenet-c180-s0.7

    # DINOv2 classification
    python hugging_face_releases/test_release.py \\
        --config      config/imagenet_configs.yaml \\
        --config_name imagenet1k_dinov2_c18 \\
        --checkpoint  path/to/best.pt \\
        --hf_dir      hugging_face_releases/filtered-dinov2-base-imagenet-c4-s0.0

    # ViT segmentation (PASCAL VOC)
    python hugging_face_releases/test_release.py \\
        --config      config/segmentation.yaml \\
        --config_name vit_pascal_voc \\
        --checkpoint  path/to/best.pt \\
        --hf_dir      hugging_face_releases/filtered-vit-base-patch16-224-voc-seg-c4-s0.0

    # DINOv2 segmentation with real image
    python hugging_face_releases/test_release.py \\
        --config      config/segmentation.yaml \\
        --config_name dinov2_pascal_voc \\
        --checkpoint  path/to/best.pt \\
        --hf_dir      hugging_face_releases/filtered-dinov2-base-voc-seg-c4-s0.0 \\
        --image       path/to/test_image.jpg

A passing run prints:
    ✓ Max absolute logit difference: 0.0000e+00  [PASS — outputs are identical]
"""

import argparse
import os
import sys

import torch

# ── Add repo root to sys.path so training-codebase imports work ───────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Add _shared/ so HF model local imports resolve ───────────────────────────
SHARED_DIR = os.path.join(os.path.dirname(__file__), "_shared")
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Config loading (mirrors training scripts)
# ─────────────────────────────────────────────────────────────────────────────

def load_training_config(config_path: str, config_name: str, args) -> dict:
    """
    Load a YAML config using the same load_config() function used by the
    training scripts, then apply any CLI overrides.
    """
    from config.utils import load_config
    cfg = load_config(config_path, config_name=config_name)

    # Apply the same CLI overrides supported by seg_main.py / main_imagenet.py
    model_cfg = cfg.setdefault("model", {})
    if args.soft_thresholding is not None:
        model_cfg["soft_thresholding"] = args.soft_thresholding
    if args.soft_thresholding_pos is not None:
        model_cfg["soft_thresholding_pos"] = args.soft_thresholding_pos
    if args.n_rotations is not None:
        model_cfg["n_rotations"] = args.n_rotations

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Original model loading (.pt checkpoint + training codebase)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_state_dict(checkpoint_path: str) -> dict:
    """Load a .pt checkpoint and return the model state dict."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                print(f"  Found state dict under key: \"{key}\"")
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            print("  Checkpoint is a raw state dict.")
            return ckpt
        raise KeyError(
            f"Cannot find model weights. Top-level keys: {list(ckpt.keys())}"
        )
    raise TypeError(f"Expected dict checkpoint, got {type(ckpt)}")


def load_original_model(cfg: dict, checkpoint_path: str):
    """
    Instantiate the model using get_model() (same as training) and load
    the .pt checkpoint weights on top.

    load_pretrained_weight is forced to False so the backbone is NOT
    re-downloaded from HuggingFace Hub — weights come entirely from the
    .pt checkpoint.
    """
    from models.get_model import get_model

    # Force no backbone download; weights come from the checkpoint.
    cfg["model"]["load_pretrained_weight"] = False

    print("\n── Loading ORIGINAL model (codebase + .pt checkpoint) ─────────────")
    print(f"  model type : {cfg['model'].get('type')}")
    print(f"  backbone   : {cfg['model'].get('pretrained_model')}")
    print(f"  num_labels : {cfg['model'].get('num_labels')}")
    print(f"  n_rotations: {cfg['model'].get('n_rotations')}")
    print(f"  soft       : {cfg['model'].get('soft_thresholding')}")
    print(f"  soft_pos   : {cfg['model'].get('soft_thresholding_pos')}")

    model = get_model(cfg)

    print(f"\n  Loading checkpoint: {checkpoint_path}")
    state_dict = _extract_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING — missing keys  ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  WARNING — unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"  Loaded {len(state_dict)} tensors.")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace model loading (packaged folder with model.safetensors)
# ─────────────────────────────────────────────────────────────────────────────

def load_hf_model(hf_dir: str):
    """Load the packaged HF model using AutoModel with trust_remote_code=True."""
    abs_dir = os.path.abspath(hf_dir)
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)

    from transformers import AutoModel, AutoConfig

    print("\n── Loading HF model (packaged folder) ─────────────────────────────")
    print(f"  Folder: {abs_dir}")

    config = AutoConfig.from_pretrained(abs_dir, trust_remote_code=True)
    model  = AutoModel.from_pretrained(abs_dir, trust_remote_code=True, config=config)
    model.eval()

    print(f"  model_arch : {config.model_arch}")
    print(f"  backbone   : {config.pretrained_model}")
    print(f"  num_labels : {config.num_labels}")
    print(f"  n_rotations: {config.n_rotations}")
    print(f"  soft       : {config.soft_thresholding}")
    print(f"  soft_pos   : {config.soft_thresholding_pos}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# State-dict comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_state_dicts(orig_sd: dict, hf_sd: dict) -> bool:
    orig_keys = set(orig_sd.keys())
    hf_keys   = set(hf_sd.keys())
    only_orig = orig_keys - hf_keys
    only_hf   = hf_keys   - orig_keys
    shared    = orig_keys & hf_keys

    print(f"\n── State-dict key comparison ───────────────────────────────────────")
    print(f"  Original keys : {len(orig_keys)}")
    print(f"  HF model keys : {len(hf_keys)}")
    print(f"  Shared keys   : {len(shared)}")

    ok = True
    if only_orig:
        print(f"  Keys only in ORIGINAL ({len(only_orig)}): "
              f"{sorted(only_orig)[:5]}{'...' if len(only_orig) > 5 else ''}")
        ok = False
    if only_hf:
        print(f"  Keys only in HF MODEL ({len(only_hf)}): "
              f"{sorted(only_hf)[:5]}{'...' if len(only_hf) > 5 else ''}")
        ok = False

    shape_mismatches = [
        (k, orig_sd[k].shape, hf_sd[k].shape)
        for k in sorted(shared)
        if orig_sd[k].shape != hf_sd[k].shape
    ]
    if shape_mismatches:
        print(f"  Shape mismatches ({len(shape_mismatches)}):")
        for k, s1, s2 in shape_mismatches[:5]:
            print(f"    {k}: original={s1}  hf={s2}")
        ok = False

    if ok:
        print("  ✓ Key sets and shapes match.")
    return ok


def compare_weight_values(orig_sd: dict, hf_sd: dict) -> float:
    """Return max absolute weight difference across all shared keys."""
    shared = set(orig_sd.keys()) & set(hf_sd.keys())
    max_diff, worst_key = 0.0, ""
    for k in sorted(shared):
        d = (orig_sd[k].float() - hf_sd[k].float()).abs().max().item()
        if d > max_diff:
            max_diff, worst_key = d, k
    if worst_key:
        print(f"  Max weight diff : {max_diff:.4e}  (key: {worst_key})")
    return max_diff


# ─────────────────────────────────────────────────────────────────────────────
# Forward-pass comparison
# ─────────────────────────────────────────────────────────────────────────────

def build_input(model_type: str, image_path: str = None) -> torch.Tensor:
    """Return a [1, 3, H, W] input tensor (real image or reproducible random)."""
    is_seg = "seg" in model_type
    size = 512 if is_seg else 224

    if image_path:
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(image_path).convert("RGB")
            t = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            tensor = t(img).unsqueeze(0)
            print(f"  Using real image : {image_path}  (resized to {size}×{size})")
            return tensor
        except Exception as e:
            print(f"  WARNING: could not load image ({e}), falling back to random.")

    torch.manual_seed(42)
    tensor = torch.randn(1, 3, size, size)
    print(f"  Using random input: shape {list(tensor.shape)}  (seed=42)")
    return tensor


def run_forward(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and return the logits tensor."""
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load training config ──────────────────────────────────────────────────
    cfg = load_training_config(args.config, args.config_name, args)
    model_type = cfg["model"].get("type", "unknown")

    print("=" * 68)
    print("  Soft-Equivariant Model Release Verification")
    print(f"  config     : {args.config}  [{args.config_name}]")
    print(f"  model type : {model_type}")
    print(f"  n_rot={cfg['model'].get('n_rotations')}  "
          f"soft={cfg['model'].get('soft_thresholding')}  "
          f"soft_pos={cfg['model'].get('soft_thresholding_pos')}")
    print("=" * 68)

    # ── Load both models ──────────────────────────────────────────────────────
    orig_model = load_original_model(cfg, args.checkpoint)
    hf_model   = load_hf_model(args.hf_dir)

    # ── Compare state dicts ───────────────────────────────────────────────────
    print("\n── Weight comparison ───────────────────────────────────────────────")
    orig_sd   = orig_model.state_dict()
    hf_sd     = hf_model.state_dict()
    keys_ok   = compare_state_dicts(orig_sd, hf_sd)
    weight_diff = compare_weight_values(orig_sd, hf_sd)

    # ── Forward-pass comparison ───────────────────────────────────────────────
    print("\n── Forward-pass comparison ─────────────────────────────────────────")
    pixel_values = build_input(model_type, args.image)
    orig_logits  = run_forward(orig_model, pixel_values)
    hf_logits    = run_forward(hf_model,   pixel_values)

    max_diff  = (orig_logits.float() - hf_logits.float()).abs().max().item()
    mean_diff = (orig_logits.float() - hf_logits.float()).abs().mean().item()

    print(f"  Original — shape={list(orig_logits.shape)}"
          f"  min={orig_logits.min():.4f}  max={orig_logits.max():.4f}")
    print(f"  HF model — shape={list(hf_logits.shape)}"
          f"  min={hf_logits.min():.4f}  max={hf_logits.max():.4f}")
    print(f"  Max absolute diff : {max_diff:.4e}")
    print(f"  Mean absolute diff: {mean_diff:.4e}")

    if orig_logits.dim() == 2:   # classification
        print(f"  Original top-1    : class {orig_logits.argmax(dim=-1).item()}")
        print(f"  HF model  top-1   : class {hf_logits.argmax(dim=-1).item()}")

    # ── Pass / Fail ───────────────────────────────────────────────────────────
    THRESHOLD = 1e-4
    passed = keys_ok and max_diff < THRESHOLD

    print("\n" + "=" * 68)
    if passed:
        print(f"  ✓ Max absolute logit difference: {max_diff:.4e}  "
              f"[PASS — outputs are identical]")
    else:
        if not keys_ok:
            print("  ✗ FAIL — state-dict key/shape mismatch (see above)")
        if max_diff >= THRESHOLD:
            print(f"  ✗ FAIL — logit diff {max_diff:.4e} exceeds threshold {THRESHOLD:.0e}")
        print("  DO NOT upload this model to HuggingFace.")
        print("=" * 68)
        sys.exit(1)

    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify HF packaged model matches original .pt checkpoint.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Config (mirrors training scripts)
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file used for training.\n"
             "e.g. config/imagenet_configs.yaml  or  config/segmentation.yaml",
    )
    parser.add_argument(
        "--config_name", required=True,
        help="Named configuration within the YAML file.\n"
             "e.g. imagenet1k_vit  or  vit_pascal_voc",
    )

    # Weights and packaged model folder
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pt checkpoint produced by training.",
    )
    parser.add_argument(
        "--hf_dir", required=True,
        help="Path to the packaged HF model folder (output of package_model.py).",
    )

    # Optional config overrides (same as training scripts support)
    parser.add_argument("--soft_thresholding", type=float, default=None,
                        help="Override model.soft_thresholding from config.")
    parser.add_argument("--soft_thresholding_pos", type=float, default=None,
                        help="Override model.soft_thresholding_pos from config.")
    parser.add_argument("--n_rotations", type=int, default=None,
                        help="Override model.n_rotations from config.")

    # Optional real image
    parser.add_argument("--image", default=None,
                        help="Path to a real image (jpg/png) to use instead of random noise.")

    args = parser.parse_args()

    # Validate paths
    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config file not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.hf_dir):
        sys.exit(f"ERROR: HF model folder not found: {args.hf_dir}")
    if not os.path.isfile(os.path.join(args.hf_dir, "model.safetensors")):
        sys.exit(f"ERROR: model.safetensors not found in {args.hf_dir}.\n"
                 f"       Run convert_to_safetensors.py first.")
    if not os.path.isfile(os.path.join(args.hf_dir, "config.json")):
        sys.exit(f"ERROR: config.json not found in {args.hf_dir}.\n"
                 f"       Run package_model.py first.")

    main(args)

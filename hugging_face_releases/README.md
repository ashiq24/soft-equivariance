# HuggingFace Release — Soft-Equivariant ViT and DINOv2

This folder contains everything needed to convert trained checkpoints, package them as self-contained HuggingFace model repositories, and verify correctness before upload.

---

## Folder structure

```
hugging_face_releases/
  README.md                          ← this file
  convert_to_safetensors.py          ← Step 1: .pt → model.safetensors
  package_model.py                   ← Step 2: assemble self-contained HF folder
  test_release.py                    ← Step 3: verify HF model ≡ original .pt model
  _shared/                           ← source files copied into every model folder
    configuration_softeq.py          ← SoftEqConfig (PretrainedConfig subclass)
    modeling_filtered_vit.py         ← FilteredViT for classification
    modeling_filtered_dinov2.py      ← FilteredDinoV2 / FilteredDinoV2wRegister
    modeling_filtered_vit_seg.py     ← FilteredViTSeg for segmentation
    modeling_filtered_dinov2_seg.py  ← FilteredDino2Seg / FilteredDino2wRegisterSeg
    filtered_layers_vit.py           ← ViT monkeypatch helpers
    filtered_layers_dinov2.py        ← DINOv2 monkeypatch helpers
    softeq/                          ← core filter/projector library
```

Each packaged model folder (e.g. `filtered-vit-base-patch16-224-imagenet-c4-s0.0/`) is
produced by Step 2 and contains a flat copy of all the above plus `config.json` and
`model.safetensors`. It can be uploaded directly to HuggingFace Hub.

---

## Step 1 — Convert `.pt` checkpoint to `model.safetensors`

```bash
python hugging_face_releases/convert_to_safetensors.py \
    --checkpoint  path/to/training_output/best.pt \
    --output_dir  /tmp/converted_vit_c4_s0 \
    --print_keys   # optional: inspect all state-dict key names
```

The script handles all checkpoint formats saved by the training pipelines:

| Format | Key |
|--------|-----|
| Segmentation trainer (`seg_main.py`) | `model_state_dict` |
| timm CheckpointSaver (`main_imagenet.py`) | `state_dict` or `model` |
| Raw state dict | top-level tensors |

---

## Step 2 — Package into a self-contained model folder

Run `package_model.py` with the **exact parameters the model was trained with**.
Wrong parameters produce wrong filter buffers and degraded performance.

### ViT — ImageNet classification

```bash
# C4, soft=0.0
python hugging_face_releases/package_model.py \
    --safetensors /tmp/converted_vit_c4_s0/model.safetensors \
    --output_dir  hugging_face_releases/filtered-vit-base-patch16-224-imagenet-c4-s0.0 \
    --model_arch  filtered_vit \
    --pretrained_model google/vit-base-patch16-224 \
    --num_labels  1000 \
    --n_rotations 4 \
    --soft_thresholding 0.0 \
    --soft_thresholding_pos 0.0

# C4, soft=0.7
python hugging_face_releases/package_model.py \
    --safetensors /tmp/converted_vit_c4_s07/model.safetensors \
    --output_dir  hugging_face_releases/filtered-vit-base-patch16-224-imagenet-c4-s0.7 \
    --model_arch  filtered_vit \
    --pretrained_model google/vit-base-patch16-224 \
    --num_labels  1000 \
    --n_rotations 4 \
    --soft_thresholding 0.7 \
    --soft_thresholding_pos 0.7

# C180, soft=0.0
python hugging_face_releases/package_model.py \
    --safetensors /tmp/converted_vit_c180_s0/model.safetensors \
    --output_dir  hugging_face_releases/filtered-vit-base-patch16-224-imagenet-c180-s0.0 \
    --model_arch  filtered_vit \
    --pretrained_model google/vit-base-patch16-224 \
    --num_labels  1000 \
    --n_rotations 180 \
    --soft_thresholding 0.0 \
    --soft_thresholding_pos 0.0
```

### DINOv2 — ImageNet classification

```bash
# Standard DINOv2, C4, soft=0.0
python hugging_face_releases/package_model.py \
    --safetensors /tmp/converted_dino_c4_s0/model.safetensors \
    --output_dir  hugging_face_releases/filtered-dinov2-base-imagenet-c4-s0.0 \
    --model_arch  filtered_dinov2 \
    --pretrained_model facebook/dinov2-base \
    --num_labels  1000 \
    --n_rotations 4 \
    --soft_thresholding 0.0 \
    --soft_thresholding_pos 0.0

# DINOv2 with registers, C4, soft=0.7
python hugging_face_releases/package_model.py \
    --safetensors /tmp/converted_dino_reg_c4_s07/model.safetensors \
    --output_dir  hugging_face_releases/filtered-dinov2-base-reg-imagenet-c4-s0.7 \
    --model_arch  filtered_dinov2 \
    --pretrained_model facebook/dinov2-base-reg \
    --num_labels  1000 \
    --n_rotations 4 \
    --soft_thresholding 0.7 \
    --soft_thresholding_pos 0.7
```

> **Note:** The model code auto-detects register variants from the pretrained model name
> (any name containing `"reg"` or `"register"`).

### ViT / DINOv2 — Segmentation (PASCAL VOC, 21 classes)

```bash
# ViT seg, C4, soft=0.0, VOC (21 classes)
python hugging_face_releases/package_model.py \
    --safetensors /tmp/converted_vit_seg_c4_s0/model.safetensors \
    --output_dir  hugging_face_releases/filtered-vit-base-patch16-224-voc-seg-c4-s0.0 \
    --model_arch  filtered_vit_seg \
    --pretrained_model google/vit-base-patch16-224 \
    --num_labels  21 \
    --n_rotations 4 \
    --soft_thresholding 0.0

# DINOv2 seg, C4, soft=0.0, VOC (21 classes)
python hugging_face_releases/package_model.py \
    --safetensors /tmp/converted_dino_seg_c4_s0/model.safetensors \
    --output_dir  hugging_face_releases/filtered-dinov2-base-voc-seg-c4-s0.0 \
    --model_arch  filtered_dino2_seg \
    --pretrained_model facebook/dinov2-base \
    --num_labels  21 \
    --n_rotations 4 \
    --soft_thresholding 0.0
```

---

## Step 3 — Verify: HF model output must match original `.pt` model

**Always run this before uploading.** The test script uses the same YAML config
that was used for training (`--config` / `--config_name`), so there is no risk of
accidentally using wrong architecture parameters. It checks state-dict keys,
weight values, and forward-pass logits. Max allowed difference is 1e-4.

```bash
# ViT classification — use the exact config + config_name from training
python hugging_face_releases/test_release.py \
    --config      config/imagenet_configs.yaml \
    --config_name imagenet1k_vit \
    --checkpoint  path/to/best.pt \
    --hf_dir      hugging_face_releases/filtered-vit-base-patch16-224-imagenet-c4-s0.0

# DINOv2 classification (C18 config, override softness at test time)
python hugging_face_releases/test_release.py \
    --config      config/imagenet_configs.yaml \
    --config_name imagenet1k_dinov2_c18 \
    --soft_thresholding 0.7 \
    --soft_thresholding_pos 0.7 \
    --checkpoint  path/to/best.pt \
    --hf_dir      hugging_face_releases/filtered-dinov2-base-imagenet-c4-s0.7

# ViT segmentation (PASCAL VOC)
python hugging_face_releases/test_release.py \
    --config      config/segmentation.yaml \
    --config_name vit_pascal_voc \
    --checkpoint  path/to/best.pt \
    --hf_dir      hugging_face_releases/filtered-vit-base-patch16-224-voc-seg-c4-s0.0

# DINOv2 segmentation — test on a real image instead of random noise
python hugging_face_releases/test_release.py \
    --config      config/segmentation.yaml \
    --config_name dinov2_pascal_voc \
    --checkpoint  path/to/best.pt \
    --hf_dir      hugging_face_releases/filtered-dinov2-base-voc-seg-c4-s0.0 \
    --image       path/to/test_image.jpg
```

The `--soft_thresholding`, `--soft_thresholding_pos`, and `--n_rotations` flags
override the corresponding values from the config, the same way training scripts
allow overriding them from the command line.

A passing run prints:

```
✓ Max absolute logit difference: 0.0000e+00  [PASS — outputs are identical]
```

A failing run prints the diff and exits with code 1.

---

## Step 4 — Upload to HuggingFace Hub

```bash
# Log in once
huggingface-cli login

# Create a new repo (do once per model)
huggingface-cli repo create filtered-vit-base-patch16-224-imagenet-c4-s0.0 --type model

# Upload the packaged folder
huggingface-cli upload \
    your-username/filtered-vit-base-patch16-224-imagenet-c4-s0.0 \
    hugging_face_releases/filtered-vit-base-patch16-224-imagenet-c4-s0.0 \
    --repo-type model
```

---

## Using a released model

```python
from transformers import AutoModel, AutoConfig

# Load config and model (trust_remote_code required for custom classes)
config = AutoConfig.from_pretrained(
    "your-username/filtered-vit-base-patch16-224-imagenet-c4-s0.0",
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    "your-username/filtered-vit-base-patch16-224-imagenet-c4-s0.0",
    trust_remote_code=True,
)
model.eval()

import torch
pixel_values = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(pixel_values)
logits = output.logits         # [1, 1000]
pred   = logits.argmax(dim=-1) # predicted ImageNet class index
```

---

## Naming convention

| Backbone | Task | n_rotations | soft | Folder / repo name |
|---|---|---|---|---|
| `google/vit-base-patch16-224` | ImageNet | 4 | 0.0 | `filtered-vit-base-patch16-224-imagenet-c4-s0.0` |
| `google/vit-base-patch16-224` | ImageNet | 4 | 0.7 | `filtered-vit-base-patch16-224-imagenet-c4-s0.7` |
| `google/vit-base-patch16-224` | ImageNet | 4 | 0.95 | `filtered-vit-base-patch16-224-imagenet-c4-s0.95` |
| `google/vit-base-patch16-224` | ImageNet | 180 | 0.0 | `filtered-vit-base-patch16-224-imagenet-c180-s0.0` |
| `facebook/dinov2-base` | ImageNet | 4 | 0.0 | `filtered-dinov2-base-imagenet-c4-s0.0` |
| `facebook/dinov2-base-reg` | ImageNet | 4 | 0.7 | `filtered-dinov2-base-reg-imagenet-c4-s0.7` |
| `google/vit-base-patch16-224` | VOC seg | 4 | 0.0 | `filtered-vit-base-patch16-224-voc-seg-c4-s0.0` |
| `facebook/dinov2-base` | VOC seg | 4 | 0.0 | `filtered-dinov2-base-voc-seg-c4-s0.0` |

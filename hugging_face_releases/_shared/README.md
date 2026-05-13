# Tunable Soft Equivariance with Guarantees

**Paper**: [Tunable Soft Equivariance with Guarantees](https://arxiv.org/abs/2603.26657)  
**Authors**: Md Ashiqur Rahman, Lim Jun Hao, Jeremiah Jiang, Teck-Yian Lim, Raymond A. Yeh

---

## Overview

This repository hosts soft-equivariant vision models introduced in our paper. These models incorporate a **tunable equivariant projection** into pre-trained Vision Transformers (ViT) and DINOv2 backbones. The projection filters patch embeddings and positional embeddings toward a desired symmetry group (e.g., discrete rotations), with a softness parameter that controls how strictly equivariance is enforced — from exact equivariance (`soft_thresholding=0`) to the original unconstrained model (`soft_thresholding=1`).

**Key properties:**
- Tunable equivariance: interpolate smoothly between equivariant and unconstrained models
- Theoretical guarantees on equivariance error as a function of the softness parameter
- Compatible with any pre-trained ViT or DINOv2 backbone
- Supports rotation and roto-reflection symmetry groups
- Applied to image classification (ImageNet) and semantic segmentation (PASCAL VOC / ADE20K)

---

## Available Model Variants

| Variant | Backbone | Task |
|---|---|---|
| `filtered_vit` | ViT | Image Classification |
| `filtered_dinov2` | DINOv2 | Image Classification |
| `filtered_vit_seg` | ViT | Semantic Segmentation |
| `filtered_dino2_seg` | DINOv2 | Semantic Segmentation |

---

## Usage

> **Note**: All models require `trust_remote_code=True` because they use custom model classes.

### Image Classification (ViT backbone)

```python
from transformers import AutoModel, AutoConfig
from PIL import Image
import torch

model_id = "ashiq24/filtered-vit-base-patch16-224-imagenet-c4-s0.0"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model  = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()

# Prepare input: (1, 3, 224, 224) float tensor, values in [0, 1]
pixel_values = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values)

logits = outputs.logits          # (1, num_labels)
pred   = logits.argmax(dim=-1)   # predicted class index
```

---

### Image Classification (DINOv2 backbone)

```python
from transformers import AutoModel, AutoConfig
import torch

model_id = "ashiq24/filtered-dinov2-base-imagenet-c4-s0.0"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model  = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()

pixel_values = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values)

logits = outputs.logits          # (1, num_labels)
pred   = logits.argmax(dim=-1)
```

---

### Semantic Segmentation (ViT backbone)

```python
from transformers import AutoModel, AutoConfig
import torch
import torch.nn.functional as F

model_id = "ashiq24/filtered-vit-base-patch16-224-voc-seg-c720-s0.9"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model  = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()

# Input size must match model training resolution (e.g., 224×224)
pixel_values = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values)

# outputs.logits shape: (1, num_labels, H, W) — already upsampled to input resolution
seg_map = outputs.logits.argmax(dim=1)   # (1, H, W) predicted label per pixel
```

---

### Semantic Segmentation (DINOv2 backbone)

```python
from transformers import AutoModel, AutoConfig
import torch

model_id = "ashiq24/filtered-dinov2-base-voc-seg-c720-s0.9"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model  = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()

pixel_values = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values)

seg_map = outputs.logits.argmax(dim=1)   # (1, H, W)
```

---

## Configuration Parameters

The `SoftEqConfig` class stores all architectural parameters. Key fields:

| Parameter | Type | Description |
|---|---|---|
| `n_rotations` | `int` | Size of the discrete rotation group (e.g., `4` for C4, `720` for near-continuous) |
| `soft_thresholding` | `float` | Softness of the patch-embedding filter in `[0, 1]`; `0` = strict equivariance, `1` = no filter |
| `soft_thresholding_pos` | `float` | Softness of the positional-embedding filter in `[0, 1]` |
| `group_type` | `str` | Symmetry group: `"rotation"` or `"roto_reflection"` |
| `hard_mask` | `bool` | Use a hard (step-function) mask instead of exponential damping |
| `model_arch` | `str` | Architecture variant (see table above) |
| `pretrained_model` | `str` | HuggingFace identifier of the base backbone |
| `num_labels` | `int` | Number of output classes |

---

## Citation

If you use these models in your research, please cite:

```bibtex
@article{rahman2026tunable,
  title={Tunable Soft Equivariance with Guarantees},
  author={Rahman, Md Ashiqur and Hao, Lim Jun and Jiang, Jeremiah and Lim, Teck-Yian and Yeh, Raymond A},
  journal={arXiv preprint arXiv:2603.26657},
  year={2026}
}
```

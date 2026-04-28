## 2026-04-28 00:20 UTC-4 - HuggingFace Release Infrastructure

### 1) What have I changed?
- Created a new branch `huggingface-release`.
- Updated `.gitignore` to ignore `*.safetensors` (and verified `.cursor/` and `.claude/` were already ignored).
- Added `safetensors` and `huggingface_hub` to `requirements.txt`.
- Added HuggingFace release scaffolding under `hugging_face_releases/`:
  - `_shared/configuration_softeq.py` for model-only config handling.
  - `_shared/modeling_filtered_vit.py`
  - `_shared/modeling_filtered_dinov2.py`
  - `_shared/modeling_filtered_vit_seg.py`
  - `_shared/modeling_filtered_dinov2_seg.py`
  - `_shared/softeq/` with required filter/projection-related modules.
  - `convert_to_safetensors.py` for `.pt -> .safetensors` conversion and model-only `config.json` generation.
  - `package_model.py` for assembling self-contained HuggingFace release folders using professional naming and remote-code files.

### 2) Why have I changed these?
- To make model releases reproducible and easy to consume via HuggingFace while preserving the exact filter/projection defaults (`n_rotations`, `soft_thresholding`, `soft_thresholding_pos`, and related architecture flags).
- To ensure `config.json` stores only model/filter/projection parameters and excludes training hyperparameters that can confuse downstream usage.
- To keep large release artifacts out of git history and prepare a dedicated release branch workflow.

### 3) Expected behavior (1 line)
- You can now convert checkpoints to `model.safetensors` and package self-contained HuggingFace model folders that load with the correct default equivariance/filter settings without copying training configs.

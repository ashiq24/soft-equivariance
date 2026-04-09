"""
Filtered DINOv2 for Semantic Segmentation (linear probing style).

This module mirrors FilteredViTSeg but with a DINOv2 backbone:
- Loads a pretrained DINOv2 backbone without a classification head
- Optionally applies filtering to the patch embedding layer
- Adds a simple 1x1 Conv (linear per-pixel) classifier
- Rearranges patch tokens back to a spatial grid
- Upsamples logits to the input resolution with bilinear interpolation
- Computes CrossEntropy loss with ignore_index if labels are provided
"""

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2Config
try:
    from transformers import Dinov2WithRegistersModel, Dinov2WithRegistersConfig
    REGISTERS_AVAILABLE = True
except ImportError:
    REGISTERS_AVAILABLE = False
from transformers.modeling_outputs import SemanticSegmenterOutput

from models.filtered_layers import monkeypatch_dinov2embeddings


class FilteredDino2Seg(nn.Module):
    """
    DINOv2 backbone with a linear segmentation head (1x1 conv).

    Args:
        pretrained_model_name: HuggingFace model id (e.g., "facebook/dinov2-base").
        num_labels: Number of semantic classes.
        filter_patch_embeddings: If True, wraps patch projection with FilteredConv2d.
        filter_attention_qkv: Compatibility flag for attention Q/K/V filtering; currently a no-op.
        filter_attention_output: Compatibility flag for attention output filtering; currently a no-op.
        filter_mlp: Compatibility flag for MLP filtering; currently a no-op.
        group_type: Group used to build invariant filters for patch/position embeddings.
        n_rotations: Number of rotations for rotation-based filters.
        soft_thresholding: Soft threshold for patch embedding filtering, in [0, 1]. 0 means exact equivariance, 1 means no filtering.
        soft_thresholding_pos: Soft threshold for positional embedding, in [0, 1].
        decomposition_method: Decomposition algorithm used by filter construction. ('svd' or 'schur')
        hard_mask: Whether to use a hard mask in the filter projector.
        preserve_norm: Whether to preserve weight norms during projection.
        joint_decomposition: Whether to use joint decomposition for multi-generator groups.
        attention_output_filter_list: Retained for config parity; unused here.
        soft_thresholding_attention_output: Retained for config parity; unused here.
        ignore_index: Label id ignored by cross-entropy loss.
        load_pretrained_weight: If True, loads pretrained backbone weights.
        freeze_patch_embeddings: If True, freezes the patch embedding projection.
        freeze_position_embeddings: If True, freezes positional embeddings.
    """
    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov2-base",
        num_labels: int = 21,
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
        attention_output_filter_list: Optional[list] = None,
        soft_thresholding_attention_output: float = 0.1,
        ignore_index: int = 255,
        load_pretrained_weight: bool = True,
        freeze_patch_embeddings: bool = False,
        freeze_position_embeddings: bool = False,
    ):
        super().__init__()

        # Load config first
        print(f"Loading DINOv2 backbone config: {pretrained_model_name}")
        config = Dinov2Config.from_pretrained(pretrained_model_name)
        
        # Create model from config (random initialization)
        self.dinov2 = Dinov2Model(config)
        self.config = self.dinov2.config
        
        # Load pretrained weights if requested
        if load_pretrained_weight:
            print("Loading pretrained weights...")
            pretrained_model = Dinov2Model.from_pretrained(pretrained_model_name)
            self.dinov2.load_state_dict(pretrained_model.state_dict(), strict=False)
        else:
            print("Skipping pretrained weights - using random initialization")

        self.num_labels = num_labels
        self.ignore_index = ignore_index

        self.filter_config = {
            "filter_patch_embeddings": filter_patch_embeddings,
            "filter_attention_qkv": filter_attention_qkv,
            "filter_attention_output": filter_attention_output,
            "filter_mlp": filter_mlp,
            "group_type": group_type,
            "n_rotations": n_rotations,
            "soft_thresholding": soft_thresholding,
            "soft_thresholding_pos": soft_thresholding_pos,
            "decomposition_method": decomposition_method,
            "attention_output_filter_list": attention_output_filter_list if attention_output_filter_list is not None else [],
            "soft_thresholding_attention_output": soft_thresholding_attention_output,
            "hard_mask": hard_mask,
            "preserve_norm": preserve_norm,
            "joint_decomposition": joint_decomposition,
            "freeze_patch_embeddings": freeze_patch_embeddings,
            "freeze_position_embeddings": freeze_position_embeddings,
        }

        hidden_size = self.config.hidden_size

        # Linear probing head as 1x1 Conv
        self.classifier = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=num_labels,
            kernel_size=1,
            bias=True,
        )
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self._apply_filters()
        
        # Freeze weights if requested
        self._freeze_weights()

        print("FilteredDino2Seg created successfully!")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num labels: {num_labels}")

    def _apply_filters(self):
        config = self.filter_config
        if config["filter_patch_embeddings"]:
            print("Filtering DINOv2 patch embeddings...")
            monkeypatch_dinov2embeddings(self.dinov2.embeddings, config)
    
    def _freeze_weights(self):
        """Freeze weights based on configuration."""
        config = self.filter_config
        
        # Freeze patch embeddings
        if config.get('freeze_patch_embeddings', False):
            print("Freezing patch embedding weights...")
            # After monkeypatching, patch_embeddings.projection is a FilteredConv2d
            patch_proj = self.dinov2.embeddings.patch_embeddings.projection
            if hasattr(patch_proj, 'weight'):
                patch_proj.weight.requires_grad = False
                if patch_proj.bias is not None:
                    patch_proj.bias.requires_grad = False
        
        # Freeze position embeddings
        if config.get('freeze_position_embeddings', False):
            print("Freezing position embedding weights...")
            if hasattr(self.dinov2.embeddings, 'position_embeddings') and self.dinov2.embeddings.position_embeddings is not None:
                self.dinov2.embeddings.position_embeddings.requires_grad = False

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else True

        batch_size, _, input_height, input_width = pixel_values.shape

        outputs = self.dinov2(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # last_hidden_state: [B, 1 + N, hidden] for standard (CLS + patches)
        sequence_output = outputs.last_hidden_state

        # Skip CLS only (no registers)
        patch_tokens = sequence_output[:, 1:, :]

        num_patches = patch_tokens.shape[1]
        grid_size = int(math.isqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"DINOv2 non-register: patch tokens {num_patches} is not a perfect square")

        # [B, num_patches, hidden] -> [B, hidden, gh, gw]
        patch_tokens = patch_tokens.permute(0, 2, 1)
        feature_map = patch_tokens.reshape(batch_size, -1, grid_size, grid_size)

        logits_lowres = self.classifier(feature_map)

        # Upsample to input resolution
        logits = nn.functional.interpolate(
            logits_lowres,
            size=(input_height, input_width) if labels is None else labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )


class FilteredDino2wRegisterSeg(nn.Module):
    """
    DINOv2-with-registers backbone with a linear segmentation head.
    Uses config.num_register_tokens to skip the correct number of register tokens.

    Args:
        pretrained_model_name: HuggingFace model id for the register-token variant.
        num_labels: Number of semantic classes.
        filter_patch_embeddings: If True, wraps patch projection with FilteredConv2d.
        filter_attention_qkv: Compatibility flag for attention Q/K/V filtering; currently a no-op.
        filter_attention_output: Compatibility flag for attention output filtering; currently a no-op.
        filter_mlp: Compatibility flag for MLP filtering; currently a no-op.
        group_type: Group used to build invariant filters for patch/position embeddings.
        n_rotations: Number of rotations for rotation-based filters.
        soft_thresholding: Soft threshold for patch embedding filtering, in [0, 1].
        soft_thresholding_pos: Soft threshold for positional embedding, in [0, 1].
        decomposition_method: Decomposition algorithm used by filter construction.
        attention_output_filter_list: Retained for config parity; unused here.
        soft_thresholding_attention_output: Retained for config parity; unused here.
        hard_mask: Whether to use a hard mask in the filter projector.
        preserve_norm: Whether to preserve weight norms during projection.
        joint_decomposition: Whether to use joint decomposition for multigen groups.
        ignore_index: Label id ignored by cross-entropy loss.
        load_pretrained_weight: If True, loads pretrained backbone weights.
        freeze_patch_embeddings: If True, freezes the patch embedding projection.
        freeze_position_embeddings: If True, freezes positional embeddings.
    """
    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov2-base-reg",
        num_labels: int = 21,
        filter_patch_embeddings: bool = True,
        filter_attention_qkv: bool = False,    # currently a no-op
        filter_attention_output: bool = False, # compatibility flag, currently a no-op
        filter_mlp: bool = False,
        group_type: str = "rotation",
        n_rotations: int = 4,
        soft_thresholding: float = 0.0,
        soft_thresholding_pos: float = 0.0,
        decomposition_method: str = "schur",
        attention_output_filter_list: Optional[list] = None,
        soft_thresholding_attention_output: float = 0.1, # not used, retained for config parity
        hard_mask: bool = False,
        preserve_norm: bool = False,
        joint_decomposition: bool = True,
        ignore_index: int = 255,
        load_pretrained_weight: bool = True,
        freeze_patch_embeddings: bool = False,
        freeze_position_embeddings: bool = False,
    ):
        super().__init__()

        if not REGISTERS_AVAILABLE:
            raise ImportError("DINOv2 with registers model is not available in this transformers version")

        # Load config first
        print(f"Loading DINOv2-with-registers backbone config: {pretrained_model_name}")
        config = Dinov2WithRegistersConfig.from_pretrained(pretrained_model_name)
        
        # Create model from config (random initialization)
        self.dinov2 = Dinov2WithRegistersModel(config)
        self.config = self.dinov2.config
        
        # Load pretrained weights if requested
        if load_pretrained_weight:
            print("Loading pretrained weights...")
            pretrained_model = Dinov2WithRegistersModel.from_pretrained(pretrained_model_name)
            self.dinov2.load_state_dict(pretrained_model.state_dict(), strict=False)
        else:
            print("Skipping pretrained weights - using random initialization")

        self.num_labels = num_labels
        self.ignore_index = ignore_index

        self.filter_config = {
            "filter_patch_embeddings": filter_patch_embeddings,
            "filter_attention_qkv": filter_attention_qkv,
            "filter_attention_output": filter_attention_output,
            "filter_mlp": filter_mlp,
            "group_type": group_type,
            "n_rotations": n_rotations,
            "soft_thresholding": soft_thresholding,
            "soft_thresholding_pos": soft_thresholding_pos,
            "decomposition_method": decomposition_method,
            "attention_output_filter_list": attention_output_filter_list if attention_output_filter_list is not None else [],
            "soft_thresholding_attention_output": soft_thresholding_attention_output,
            "hard_mask": hard_mask,
            "preserve_norm": preserve_norm,
            "joint_decomposition": joint_decomposition,
            "freeze_patch_embeddings": freeze_patch_embeddings,
            "freeze_position_embeddings": freeze_position_embeddings,
        }

        hidden_size = self.config.hidden_size

        self.classifier = nn.Conv2d(hidden_size, num_labels, kernel_size=1, bias=True)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self._apply_filters()
        
        # Freeze weights if requested
        self._freeze_weights()

        print("FilteredDino2wRegisterSeg created successfully!")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num labels: {num_labels}")

    def _apply_filters(self):
        config = self.filter_config
        if config["filter_patch_embeddings"]:
            print("Filtering DINOv2-with-registers patch embeddings...")
            # Dinov2WithRegistersModel also uses embeddings attribute
            monkeypatch_dinov2embeddings(self.dinov2.embeddings, config)
    
    def _freeze_weights(self):
        """Freeze weights based on configuration."""
        config = self.filter_config
        
        # Freeze patch embeddings
        if config.get('freeze_patch_embeddings', False):
            print("Freezing patch embedding weights...")
            # After monkeypatching, patch_embeddings.projection is a FilteredConv2d
            patch_proj = self.dinov2.embeddings.patch_embeddings.projection
            if hasattr(patch_proj, 'weight'):
                patch_proj.weight.requires_grad = False
        
        # Freeze position embeddings
        if config.get('freeze_position_embeddings', False):
            print("Freezing position embedding weights...")
            if hasattr(self.dinov2.embeddings, 'position_embeddings') and self.dinov2.embeddings.position_embeddings is not None:
                self.dinov2.embeddings.position_embeddings.requires_grad = False

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else True

        batch_size, _, input_height, input_width = pixel_values.shape

        outputs = self.dinov2(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state  # [B, 1 + R + N, hidden]
        num_registers = getattr(self.config, "num_register_tokens", 0)
        # Skip 1 CLS + R register tokens
        patch_tokens = sequence_output[:, 1 + num_registers:, :]

        num_patches = patch_tokens.shape[1]
        grid_size = int(math.isqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"DINOv2 with registers: patch tokens {num_patches} is not a perfect square (registers={num_registers})"
            )

        patch_tokens = patch_tokens.permute(0, 2, 1)
        feature_map = patch_tokens.reshape(batch_size, -1, grid_size, grid_size)

        logits_lowres = self.classifier(feature_map)
        logits = nn.functional.interpolate(
            logits_lowres,
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        )

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

def create_filtered_dino2_seg(config: Dict[str, Any]) -> FilteredDino2Seg:
    model_config = config

    # Choose register vs non-register variant based on pretrained name or availability flag
    pretrained_model_name = model_config['pretrained_model']
    is_register_variant = "reg" in pretrained_model_name.lower() or "register" in pretrained_model_name.lower()
    
    common_kwargs = {
        "pretrained_model_name": pretrained_model_name,
        "num_labels": model_config.get("num_labels", 21),
        "filter_patch_embeddings": model_config.get("filter_patch_embeddings", True),
        "filter_attention_qkv": model_config.get("filter_attention_qkv", False),
        "filter_attention_output": model_config.get("filter_attention_output", False),
        "filter_mlp": model_config.get("filter_mlp", False),
        "group_type": model_config.get("group_type", "rotation"),
        "n_rotations": model_config.get("n_rotations", 4),
        "soft_thresholding": model_config.get("soft_thresholding", 0.0),
        "soft_thresholding_pos": model_config.get("soft_thresholding_pos", 0.0),
        "decomposition_method": model_config.get("decomposition_method", "schur"),
        "preserve_norm": model_config['preserve_norm'],
        "joint_decomposition": model_config.get("joint_decomposition", True),
        "attention_output_filter_list": model_config.get("attention_output_filter_list", []),
        "soft_thresholding_attention_output": model_config.get("soft_thresholding_attention_output", 0.1),
        "ignore_index": model_config.get("ignore_index", 255),
        "hard_mask": model_config.get("hard_mask", False),
        "load_pretrained_weight": model_config.get("load_pretrained_weight", True),
        "freeze_patch_embeddings": model_config.get("freeze_patch_embeddings", False),
        "freeze_position_embeddings": model_config.get("freeze_position_embeddings", False),
    }

    if REGISTERS_AVAILABLE and is_register_variant:
        model = FilteredDino2wRegisterSeg(**common_kwargs)
    else:
        model = FilteredDino2Seg(**common_kwargs)
    
    return model
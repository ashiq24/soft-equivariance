"""
Filtered DINOv2 implementation.

This module loads a pretrained DINOv2 from HuggingFace and wraps specified
layers with filters that modify the pretrained weights during forward pass.
Supports both standard DINOv2 and DINOv2 with register tokens.
"""

import torch
import torch.nn as nn
from transformers import Dinov2ForImageClassification, Dinov2Config
from typing import Optional, Dict, Any

# Import both DINOv2 variants
try:
    from transformers import Dinov2WithRegistersForImageClassification, Dinov2WithRegistersConfig
    REGISTERS_AVAILABLE = True
except ImportError:
    REGISTERS_AVAILABLE = False
    print("Warning: DINOv2 with registers not available in this transformers version")

from models.filtered_layers import monkeypatch_dinov2embeddings
from softeq.equi_utils.rotation_filters import get_invariant_filter_rotation, get_equivariant_filter_rotation


class FilteredDinoV2(Dinov2ForImageClassification):
    """
    Standard DINOv2 with filtered weights.
    
    Inherits from Dinov2ForImageClassification and adds filtering capabilities.
    The filters modify the pretrained weights during forward pass.
    
    Args:
        pretrained_model_name: HuggingFace model identifier.
        num_labels: Number of output classes.
        filter_patch_embeddings: Whether to filter the patch embedding convolution.
        filter_attention_qkv: Configuration flag for Q/K/V filtering; currently a no-op.
        filter_attention_output: Configuration flag for attention output filtering; currently a no-op.
        filter_mlp: Configuration flag for MLP filtering; currently a no-op.
        group_type: Group used to build invariant filters.
        n_rotations: Number of discrete rotations used by the filter factory.
        soft_thresholding: Softness for patch embedding filtering, in [0, 1]. 0 means exact equivariance, 1 means no filtering.
        soft_thresholding_pos: Softness for positional embedding filtering, in [0, 1]. 0 means exact equivariance, 1 means no filtering.
        decomposition_method: Decomposition method passed to the filter factory. ('svd' or 'schur')
        hard_mask: Whether to use a hard mask in the filter projector.
        preserve_norm: Whether to preserve weight norms during projection.
        joint_decomposition: Whether to use joint decomposition for multigen groups. 
        load_pretrained_weight: If True, loads pretrained backbone weights.
        freeze_patch_embeddings: If True, freezes the patch embedding projection.
        freeze_position_embeddings: If True, freezes positional embeddings.
    """
    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov2-base",
        num_labels: int = 1000,
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
        load_pretrained_weight: bool = True,
        freeze_patch_embeddings: bool = False,
        freeze_position_embeddings: bool = False,
    ):
        # Load the pretrained model first
        model = Dinov2ForImageClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Initialize parent with the loaded model's config
        super().__init__(model.config)
        
        # Copy the pretrained weights if requested
        if load_pretrained_weight:
            print("Loading pretrained weights...")
            self.load_state_dict(model.state_dict(), strict=False)
        else:
            print("Skipping pretrained weights - using random initialization")
        
        # Store filter configuration
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
            'hard_mask': hard_mask,
            'preserve_norm': preserve_norm,
            'joint_decomposition': joint_decomposition,
            'freeze_patch_embeddings': freeze_patch_embeddings,
            'freeze_position_embeddings': freeze_position_embeddings,
        }
        
        # Override layers with filtered versions
        self._apply_filters()
        
        # Freeze weights if requested
        self._freeze_weights()
        
        print(f"Filtered DINOv2 standard created successfully!")
    
    def _apply_filters(self):
        """Apply filters to specified layers in the model."""
        config = self.filter_config
        
        # 1. Filter patch embeddings (Conv2d)
        if config["filter_patch_embeddings"]:
            print("Filtering DINOv2 patch embeddings...")
            self._filter_patch_embeddings()
        
        # 2. Filter attention and MLP layers in each transformer block
        # For DINOv2, most operations are token-wise and don't need spatial filtering
        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer = self.dinov2.encoder.layer[layer_idx]
            
            # Filter attention Q, K, V
            if config["filter_attention_qkv"]:
                self._filter_attention_qkv(layer, layer_idx)
            
            # Filter attention output
            if config["filter_attention_output"]:
                self._filter_attention_output(layer, layer_idx)
            
            # Filter MLP
            if config["filter_mlp"]:
                self._filter_mlp(layer, layer_idx)
    
    def _filter_patch_embeddings(self):
        """Filter patch embedding Conv2d layer for DINOv2."""
        config = self.filter_config
        monkeypatch_dinov2embeddings(self.dinov2.embeddings, config)
    
    def _filter_attention_qkv(self, layer, layer_idx):
        """Placeholder for Q/K/V filtering; currently a no-op for token-wise DINOv2 blocks."""
        pass
    
    def _filter_attention_output(self, layer, layer_idx):
        """Placeholder for attention-output filtering; currently a no-op."""
        pass
    
    def _filter_mlp(self, layer, layer_idx):
        """Placeholder for MLP filtering; currently a no-op."""
        pass
    
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


_DinoV2RegisterBase = Dinov2WithRegistersForImageClassification if REGISTERS_AVAILABLE else object


class FilteredDinoV2wRegister(_DinoV2RegisterBase):
    """
    DINOv2 with register tokens and filtered weights.
    
    Inherits from Dinov2WithRegistersForImageClassification and adds filtering capabilities.
    The filters modify the pretrained weights during forward pass.
    
    Args:
        pretrained_model_name: HuggingFace model identifier.
        num_labels: Number of output classes.
        filter_patch_embeddings: Whether to filter the patch embedding convolution.
        filter_attention_qkv: Configuration flag for Q/K/V filtering; currently a no-op.
        filter_attention_output: Configuration flag for attention output filtering; currently a no-op.
        filter_mlp: Configuration flag for MLP filtering; currently a no-op.
        group_type: Group used to build invariant filters.
        n_rotations: Number of discrete rotations used by the filter factory.
        soft_thresholding: Softness for patch embedding filtering, in [0, 1].
        soft_thresholding_pos: Softness for positional embedding filtering, in [0, 1].
        decomposition_method: Decomposition method passed to the filter factory.
        hard_mask: Whether to use a hard mask in the filter projector.
        preserve_norm: Whether to preserve weight norms during projection.
        joint_decomposition: Whether to use joint decomposition for multigen groups.
        load_pretrained_weight: If True, loads pretrained backbone weights.
        freeze_patch_embeddings: If True, freezes the patch embedding projection.
        freeze_position_embeddings: If True, freezes positional embeddings.
    """
    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov2-base-reg",
        num_labels: int = 1000,
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
        freeze_patch_embeddings: bool = False,
        freeze_position_embeddings: bool = False,
        preserve_norm: bool = False,
        joint_decomposition: bool = True,
        load_pretrained_weight: bool = True,
    ):
        if not REGISTERS_AVAILABLE:
            raise ImportError("DINOv2 with registers not available in this transformers version")
        
        # Load the pretrained model first
        model = Dinov2WithRegistersForImageClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Initialize parent with the loaded model's config
        super().__init__(model.config)
        
        # Copy the pretrained weights if requested
        if load_pretrained_weight:
            print("Loading pretrained weights...")
            self.load_state_dict(model.state_dict(), strict=False)
        else:
            print("Skipping pretrained weights - using random initialization")
        
        # Store filter configuration
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
            "hard_mask": hard_mask,
            "preserve_norm": preserve_norm,
            "freeze_patch_embeddings": freeze_patch_embeddings,
            "freeze_position_embeddings": freeze_position_embeddings,
            "joint_decomposition": joint_decomposition
        }
        
        # Override layers with filtered versions
        if self.filter_config['soft_thresholding'] < 1.0 or self.filter_config['soft_thresholding_pos'] < 1.0:
            self._apply_filters()
        
        # Freeze weights if requested
        self._freeze_weights()
        
        print(f"Filtered DINOv2 with registers created successfully!")
    
    def _apply_filters(self):
        """Apply filters to specified layers in the model."""
        config = self.filter_config
        
        # 1. Filter patch embeddings (Conv2d)
        if config["filter_patch_embeddings"]:
            print("Filtering DINOv2 patch embeddings...")
            self._filter_patch_embeddings()
        
        # 2. Filter attention and MLP layers in each transformer block
        # For DINOv2, most operations are token-wise and don't need spatial filtering
        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer = self.dinov2_with_registers.encoder.layer[layer_idx]
            
            # Filter attention Q, K, V
            if config["filter_attention_qkv"]:
                print(f"Filtering DINOv2 attention Q/K/V in layer {layer_idx}...")
                self._filter_attention_qkv(layer, layer_idx)
            
            # Filter attention output
            if config["filter_attention_output"]:
                print(f"Filtering DINOv2 attention output in layer {layer_idx}...")
                self._filter_attention_output(layer, layer_idx)
            
            # Filter MLP
            if config["filter_mlp"]:
                print(f"Filtering DINOv2 MLP in layer {layer_idx}...")
                self._filter_mlp(layer, layer_idx)
    
    def _filter_patch_embeddings(self):
        """Filter patch embedding Conv2d layer for DINOv2."""
        config = self.filter_config
        monkeypatch_dinov2embeddings(self.dinov2_with_registers.embeddings, config)
    
    def _filter_attention_qkv(self, layer, layer_idx):
        """Placeholder for Q/K/V filtering; currently a no-op for token-wise DINOv2 blocks."""
        pass
    
    def _filter_attention_output(self, layer, layer_idx):
        """Placeholder for attention-output filtering; currently a no-op."""
        pass
    
    def _filter_mlp(self, layer, layer_idx):
        """Placeholder for MLP filtering; currently a no-op."""
        pass
    
    def _freeze_weights(self):
        """Freeze weights based on configuration."""
        config = self.filter_config
        
        # Freeze patch embeddings
        if config.get('freeze_patch_embeddings', False):
            print("Freezing patch embedding weights...")
            # After monkeypatching, patch_embeddings.projection is a FilteredConv2d
            patch_proj = self.dinov2_with_registers.embeddings.patch_embeddings.projection
            if hasattr(patch_proj, 'weight'):
                patch_proj.weight.requires_grad = False
                if patch_proj.bias is not None:
                    patch_proj.bias.requires_grad = False
        
        # Freeze position embeddings
        if config.get('freeze_position_embeddings', False):
            print("Freezing position embedding weights...")
            if hasattr(self.dinov2_with_registers.embeddings, 'position_embeddings') and self.dinov2_with_registers.embeddings.position_embeddings is not None:
                self.dinov2_with_registers.embeddings.position_embeddings.requires_grad = False


def create_filtered_dinov2(config: Dict[str, Any]):
    """
    Factory function to create a filtered DINOv2 from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        FilteredDinoV2 or FilteredDinoV2wRegister model
    """
    model_config = config
    pretrained_model_name = model_config['pretrained_model']
    
    # Determine if this is a register variant
    is_register_variant = 'reg' in pretrained_model_name.lower() or 'register' in pretrained_model_name.lower()
    
    # Select appropriate class
    if is_register_variant:
        model = FilteredDinoV2wRegister(
            pretrained_model_name=pretrained_model_name,
            num_labels=model_config['num_labels'],
            filter_patch_embeddings=model_config['filter_patch_embeddings'],
            filter_attention_qkv=model_config['filter_attention_qkv'],
            filter_attention_output=model_config['filter_attention_output'],
            filter_mlp=model_config['filter_mlp'],
            group_type=model_config.get('group_type', 'rotation'),
            n_rotations=model_config['n_rotations'],
            soft_thresholding=model_config['soft_thresholding'],
            soft_thresholding_pos=model_config['soft_thresholding_pos'],
            decomposition_method=model_config['decomposition_method'],
            hard_mask=model_config.get('hard_mask', False),
            preserve_norm=model_config['preserve_norm'],
            joint_decomposition=model_config.get('joint_decomposition', True),
            load_pretrained_weight=model_config.get('load_pretrained_weight', True),
            freeze_patch_embeddings=model_config.get('freeze_patch_embeddings', False),
            freeze_position_embeddings=model_config.get('freeze_position_embeddings', False),
        )
    else:
        model = FilteredDinoV2(
            pretrained_model_name=pretrained_model_name,
            num_labels=model_config['num_labels'],
            filter_patch_embeddings=model_config['filter_patch_embeddings'],
            filter_attention_qkv=model_config['filter_attention_qkv'],
            filter_attention_output=model_config['filter_attention_output'],
            filter_mlp=model_config['filter_mlp'],
            group_type=model_config.get('group_type', 'rotation'),
            n_rotations=model_config['n_rotations'],
            soft_thresholding=model_config['soft_thresholding'],
            soft_thresholding_pos=model_config['soft_thresholding_pos'],
            decomposition_method=model_config['decomposition_method'],
            hard_mask=model_config.get('hard_mask', False),
            preserve_norm=model_config['preserve_norm'],
            joint_decomposition=model_config.get('joint_decomposition', True),
            load_pretrained_weight=model_config.get('load_pretrained_weight', True),
            freeze_patch_embeddings=model_config.get('freeze_patch_embeddings', False),
            freeze_position_embeddings=model_config.get('freeze_position_embeddings', False),
        )
    
    return model
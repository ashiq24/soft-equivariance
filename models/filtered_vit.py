"""
Filtered Vision Transformer implementation.

This module loads a pretrained ViT from HuggingFace and wraps specified
layers with filters that modify the pretrained weights during forward pass.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
from typing import Optional, Dict, Any

from models.filtered_layers import monkeypatch_vitembeddings, monkeypatch_vitoutput
from softeq.equi_utils.rotation_filters import get_invariant_filter_rotation, get_equivariant_filter_rotation


class FilteredViT(ViTForImageClassification):
    """
    Vision Transformer with soft-equivariant filtering hooks.
    
    This wrapper keeps the HuggingFace ViT architecture, optionally monkey-patching
    patch embeddings and selected attention output projections. Q/K/V and MLP
    filtering flags are accepted for configuration compatibility, but currently
    implemented as no-ops in this module.
    
    Args:
        pretrained_model_name: HuggingFace model identifier.
        num_labels: Number of output classes.
        filter_patch_embeddings: Whether to filter the patch embedding convolution.
        filter_attention_qkv: Configuration flag for Q/K/V filtering; currently a no-op.
        filter_attention_output: Whether to filter attention output projection.
        filter_mlp: Configuration flag for MLP filtering; currently a no-op.
        group_type: Group used to build invariant filters.
        n_rotations: Number of discrete rotations used by the filter factory.
        soft_thresholding: Softness for patch embedding filtering, in [0, 1], where 0 implies exact equivariant projection and 1 implies no projection.
        soft_thresholding_pos: Softness for positional embedding filtering, in [0, 1], where 0 is exact equivariant projection and 1 is no projection.
        decomposition_method: Decomposition method passed to the filter factory.
        hard_mask: Whether to use a hard mask in the filter projector.
        preserve_norm: Whether to preserve weight norms during projection.
        attention_output_filter_list: Layer indices where attention output filtering is applied.
        soft_thresholding_attention_output: Softness for attention output filtering.
        joint_decomposition: Whether to use joint decomposition for multi-generator groups.
        load_pretrained_weight: If True, loads pretrained backbone weights.
        freeze_patch_embeddings: If True, freezes the patch embedding projection.
        freeze_position_embeddings: If True, freezes positional embeddings.
    """
    def __init__(
        self,
        pretrained_model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 1000,
        filter_patch_embeddings: bool = True,
        filter_attention_qkv: bool = True,
        filter_attention_output: bool = True,
        filter_mlp: bool = True,
        group_type: str = "rotation",
        n_rotations: int = 4,
        soft_thresholding: float = 0.0,
        soft_thresholding_pos: float = 0.0,
        decomposition_method: str = "schur",
        hard_mask: bool = False,
        preserve_norm: bool = False,
        attention_output_filter_list: Optional[list] = None,
        soft_thresholding_attention_output: float = 0.1,
        joint_decomposition: bool = True,
        load_pretrained_weight: bool = True,
        freeze_patch_embeddings: bool = False,
        freeze_position_embeddings: bool = False,
    ):
        # Load pretrained model to get config (and optionally weights)
        print(f"Loading pretrained model: {pretrained_model_name}")
        pretrained_model = ViTForImageClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Initialize parent class with pretrained config
        super().__init__(pretrained_model.config)
        
        # Copy pretrained weights if requested
        if load_pretrained_weight:
            print("Loading pretrained weights...")
            self.load_state_dict(pretrained_model.state_dict(), strict=False)
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
            'attention_output_filter_list': attention_output_filter_list if attention_output_filter_list is not None else [],
            'soft_thresholding_attention_output': soft_thresholding_attention_output,
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
        
    def _apply_filters(self):
        """Apply filters to specified layers in the model."""
        config = self.filter_config
        
        # 1. Filter patch embeddings (Conv2d)
        if config["filter_patch_embeddings"]:
            print("Filtering patch embeddings...")
            self._filter_patch_embeddings()
        
        # 2. Filter attention and MLP layers in each transformer block
        
        # For ViT the following loop mostly keeps token-wise blocks unchanged.
        # and the position embeddings are added only in the patch embedding layer
        # For other models, the following loop might have to filter the attention and MLP layers
        
        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer = self.vit.encoder.layer[layer_idx]
            
            # Filter attention Q, K, V
            if config["filter_attention_qkv"]:
                self._filter_attention_qkv()
            
            # Filter attention output
            if config["filter_attention_output"] and layer_idx in config['attention_output_filter_list']:
                print(f"Filtering attention output in layer {layer_idx}...")
                self._filter_attention_output(layer)
            
            # Filter MLP
            if config["filter_mlp"]:
                self._filter_mlp()
    
    def _filter_patch_embeddings(self):
        """Filter patch embedding Conv2d layer."""
        config = self.filter_config
        monkeypatch_vitembeddings(self.vit.embeddings, config)
    
    def _filter_attention_qkv(self):
        """Placeholder for Q/K/V filtering; currently a no-op for token-wise ViT blocks."""
        return
    
    def _filter_attention_output(self, layer):
        """Filter attention output projection."""
        # not need to impose anything as operations are token-wise
        
        num_patches = self.vit.embeddings.patch_embeddings.num_patches
        monkeypatch_vitoutput(layer.attention.output, self.filter_config, num_patches)
    
    def _filter_mlp(self):
        """Placeholder for MLP filtering; currently a no-op for token-wise ViT blocks."""
        return
    
    def _freeze_weights(self):
        """Freeze weights based on configuration."""
        config = self.filter_config
        
        # Freeze patch embeddings
        if config.get('freeze_patch_embeddings', False):
            print("Freezing patch embedding weights...")
            # After monkeypatching, patch_embeddings.projection is a FilteredConv2d
            patch_proj = self.vit.embeddings.patch_embeddings.projection
            if hasattr(patch_proj, 'weight'):
                patch_proj.weight.requires_grad = False
        
        # Freeze position embeddings
        if config.get('freeze_position_embeddings', False):
            print("Freezing position embedding weights...")
            if hasattr(self.vit.embeddings, 'position_embeddings'):
                self.vit.embeddings.position_embeddings.requires_grad = False
        
def create_filtered_vit(config: Dict[str, Any]) -> FilteredViT:
    """
    Factory function to create a filtered ViT from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        FilteredViT model
    """
    model_config = config
    
    return FilteredViT(
        pretrained_model_name=model_config['pretrained_model'],
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
        preserve_norm=model_config['preserve_norm'],
        joint_decomposition=model_config.get('joint_decomposition', True),
        load_pretrained_weight=model_config.get('load_pretrained_weight', True),
        freeze_patch_embeddings=model_config.get('freeze_patch_embeddings', False),
        freeze_position_embeddings=model_config.get('freeze_position_embeddings', False),
    )
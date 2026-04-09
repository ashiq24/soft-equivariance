"""
Filtered Vision Transformer for Semantic Segmentation.

This module implements a linear probing style segmentation model that:
1. Uses a pretrained ViT backbone with optional filtering
2. Adds a simple 1x1 conv (linear) classifier on top of the final features
3. Rearranges patch tokens back to spatial grid
4. Upsamples logits to match input image size using bilinear interpolation
5. Computes segmentation loss if labels are provided

This is similar to linear probing but for dense prediction tasks.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional, Dict, Any
import math

from models.filtered_layers import monkeypatch_vitembeddings, monkeypatch_vitoutput
from softeq.equi_utils.rotation_filters import get_invariant_filter_rotation, get_equivariant_filter_rotation


class FilteredViTSeg(nn.Module):
    """
    Vision Transformer with filtered weights for semantic segmentation.
    
    This model:
    1. Loads a pretrained ViT backbone (without classification head)
    2. Applies optional filtering to patch embeddings and attention layers
    3. Uses a simple 1x1 conv classifier on the final feature map
    4. Upsamples predictions to match input resolution
    
    Args:
        pretrained_model_name: HuggingFace model identifier.
        num_labels: Number of segmentation classes.
        filter_patch_embeddings: Whether to filter the patch embedding convolution.
        filter_attention_qkv: Configuration flag for Q/K/V filtering; currently a no-op.
        filter_attention_output: Whether to filter attention output projection.
        filter_mlp: Configuration flag for MLP filtering; currently a no-op.
        group_type: Group used to build invariant filters.
        n_rotations: Number of discrete rotations used by the filter factory.
        soft_thresholding: Softness for patch embedding filtering, in [0, 1] (where 0 is hard projection and 1 is no projection).
        soft_thresholding_pos: Softness for positional embedding filtering, in [0, 1] (where 0 is hard projection and 1 is no projection).
        decomposition_method: Decomposition method passed to the filter factory.
        hard_mask: Whether to use a hard mask in the filter projector.
        preserve_norm: Whether to preserve weight norms during projection.
        joint_decomposition: Whether to use joint decomposition for multi-generator groups.
        attention_output_filter_list: Layer indices where attention output filtering is applied.
        soft_thresholding_attention_output: Softness for attention output filtering.
        ignore_index: Index ignored by the loss function.
        load_pretrained_weight: If True, loads pretrained backbone weights.
        freeze_patch_embeddings: If True, freezes the patch embedding projection.
        freeze_position_embeddings: If True, freezes positional embeddings.
    """
    def __init__(
        self,
        pretrained_model_name: str = "google/vit-base-patch16-224",
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
        print(f"Loading ViT backbone config: {pretrained_model_name}")
        config = ViTConfig.from_pretrained(pretrained_model_name)
        
        # Create model from config (random initialization)
        self.vit = ViTModel(config)
        self.config = self.vit.config
        
        # Load pretrained weights if requested
        if load_pretrained_weight:
            print("Loading pretrained weights...")
            pretrained_model = ViTModel.from_pretrained(pretrained_model_name)
            self.vit.load_state_dict(pretrained_model.state_dict(), strict=False)
        else:
            print("Skipping pretrained weights - using random initialization")
        
        # Store configuration
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        
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
        
        # Get feature dimension from ViT
        hidden_size = self.config.hidden_size
        
        # Create 1x1 conv classifier (equivalent to linear layer per pixel)
        # Input: [B, hidden_size, H, W], Output: [B, num_labels, H, W]
        self.classifier = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=num_labels,
            kernel_size=1,
            bias=True
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        # Initialize classifier with small weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        # Apply filters to ViT backbone
        self._apply_filters()
        
        # Freeze weights if requested
        self._freeze_weights()
        
        print(f"FilteredViTSeg created successfully!")
        print(f"  Backbone: {pretrained_model_name}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num labels: {num_labels}")
        print(f"  Patch size: {self.config.patch_size}")
    
    def _apply_filters(self):
        """Apply filters to specified layers in the ViT backbone."""
        config = self.filter_config
        
        # 1. Filter patch embeddings (Conv2d)
        if config["filter_patch_embeddings"]:
            print("Filtering patch embeddings...")
            self._filter_patch_embeddings()
        
        # 2. Filter attention and MLP layers in each transformer block
        # For ViT, most operations are token-wise and don't need spatial filtering
        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer = self.vit.encoder.layer[layer_idx]
            
            # Filter attention Q, K, V (usually not needed for ViT)
            if config["filter_attention_qkv"]:
                self._filter_attention_qkv()
            
            # Filter attention output (only for specified layers)
            if config["filter_attention_output"] and layer_idx in config['attention_output_filter_list']:
                print(f"Filtering attention output in layer {layer_idx}...")
                self._filter_attention_output(layer)
            
            # Filter MLP (usually not needed for ViT)
            if config["filter_mlp"]:
                self._filter_mlp()
    
    def _filter_patch_embeddings(self):
        """Filter patch embedding Conv2d layer."""
        config = self.filter_config
        monkeypatch_vitembeddings(self.vit.embeddings, config)
    
    def _filter_attention_qkv(self):
        """Filter Q, K, V projections in attention layer."""
        # Not needed for ViT as operations are token-wise
        return
    
    def _filter_attention_output(self, layer):
        """Filter attention output projection."""
        # Not needed for standard ViT, but can be used for specific layers
        num_patches = self.vit.embeddings.patch_embeddings.num_patches
        monkeypatch_vitoutput(layer.attention.output, self.filter_config, num_patches)
    
    def _filter_mlp(self):
        """Filter MLP (feed-forward) layers."""
        # Not needed for ViT as operations are token-wise
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
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass for semantic segmentation.
        
        Args:
            pixel_values: Input images [B, 3, H, W]
            labels: Optional ground truth segmentation masks [B, H, W]
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            return_dict: Whether to return a dict
            
        Returns:
            SemanticSegmenterOutput with:
                - loss: Segmentation loss (if labels provided)
                - logits: Segmentation logits [B, num_labels, H, W] after bilinear upsampling
                - hidden_states: Intermediate features (if requested)
                - attentions: Attention weights (if requested)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get input image size
        batch_size, _, input_height, input_width = pixel_values.shape
        
        # Forward pass through ViT backbone
        outputs = self.vit(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        
        # Get final hidden states [B, num_patches + 1, hidden_size]
        # (+1 for CLS token)
        sequence_output = outputs.last_hidden_state
        
        # Remove CLS token - we only use patch tokens for segmentation
        # [B, num_patches, hidden_size]
        patch_tokens = sequence_output[:, 1:, :]
        
        # Calculate grid size from number of patches
        num_patches = patch_tokens.shape[1]
        patch_size = self.config.patch_size
        grid_size = int(math.sqrt(num_patches))
        
        # Rearrange patch tokens to spatial grid
        # [B, num_patches, hidden_size] -> [B, hidden_size, grid_h, grid_w]
        patch_tokens = patch_tokens.permute(0, 2, 1)  # [B, hidden_size, num_patches]
        feature_map = patch_tokens.reshape(batch_size, -1, grid_size, grid_size)
        
        # Apply 1x1 conv classifier
        # [B, hidden_size, grid_h, grid_w] -> [B, num_labels, grid_h, grid_w]
        logits = self.classifier(feature_map)
        
        # Upsample logits to match input/label resolution.
        logits = nn.functional.interpolate(
            logits,
            size=(input_height, input_width) if labels is None else labels.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:        
            # Compute cross-entropy loss with ignore_index
            
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


def create_filtered_vit_seg(config: Dict[str, Any]) -> FilteredViTSeg:
    """
    Factory function to create a FilteredViTSeg from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        FilteredViTSeg model
    """
    model_config = config
    
    return FilteredViTSeg(
        pretrained_model_name=model_config['pretrained_model'],
        num_labels=model_config['num_labels'],
        filter_patch_embeddings=model_config.get('filter_patch_embeddings', True),
        filter_attention_qkv=model_config.get('filter_attention_qkv', False),
        filter_attention_output=model_config.get('filter_attention_output', False),
        filter_mlp=model_config.get('filter_mlp', False),
        group_type=model_config.get('group_type', 'rotation'),
        n_rotations=model_config.get('n_rotations', 4),
        soft_thresholding=model_config.get('soft_thresholding', 0.0),
        soft_thresholding_pos=model_config.get('soft_thresholding_pos', 0.0),
        decomposition_method=model_config.get('decomposition_method', 'schur'),
        preserve_norm=model_config['preserve_norm'],
        joint_decomposition=model_config.get('joint_decomposition', True),
        attention_output_filter_list=model_config.get('attention_output_filter_list', []),
        soft_thresholding_attention_output=model_config.get('soft_thresholding_attention_output', 0.1),
        ignore_index=model_config.get('ignore_index', 255),
        load_pretrained_weight=model_config.get('load_pretrained_weight', True),
        freeze_patch_embeddings=model_config.get('freeze_patch_embeddings', False),
        freeze_position_embeddings=model_config.get('freeze_position_embeddings', False),
    )


"""
Filtered Autoregressive Transformer for 2D Trajectory Prediction.

This module implements a group-equivariant version of the autoregressive transformer
by applying equivariant filters to all Linear and MultiheadAttention layers.

Supported group types:
- "rotation": Rotation group (SO(2) for 2D, SO(3) for 3D)
- "reflection": Reflection group (Z_2)
- "roto_reflection": Roto-reflection group (rotation + reflection)

Architecture:
- Inherits from AutoregressiveTransformer
- Replaces nn.Linear with FilteredLinear (using both filter_eq and filter_inv)
- Replaces nn.MultiheadAttention with FilteredMultiheadAttention (using filter_eq only)
- Preserves LayerNorm (already equivariant)
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoregressive_transformer import AutoregressiveTransformer
from softeq.layers.flinear import FilteredLinear
from softeq.layers.filtered_attention import FilteredMultiheadAttention
from softeq.equi_utils.rotation_filters import (
    get_equivariant_filter_rotation,
    get_invariant_filter_rotation
)
from softeq.equi_utils.reflection_filters import (
    get_equivariant_filter_reflection,
    get_invariant_filter_reflection
)
from softeq.equi_utils.roto_reflection_filters import (
    get_equivariant_filter_roto_reflection,
    get_invariant_filter_roto_reflection
)
from utils.eq_layernorm import EQLayerNorm

# Valid group types
VALID_GROUP_TYPES = ["rotation", "reflection", "roto_reflection"]


class FilteredAutoregressiveTransformer(AutoregressiveTransformer):
    """
    Autoregressive transformer with filtered layers for group equivariance.
    
    This model applies equivariant filters to all Linear and MultiheadAttention layers,
    enabling group-equivariant trajectory prediction for 2D coordinates.
    
    Args:
        max_people: Maximum number of people in a scene.
        obs_len: Number of observed timesteps.
        pred_len: Number of timesteps to autoregressively predict.
        d_model: Token embedding size.
        num_heads: Number of attention heads (must divide d_model).
        num_layers: Number of decoder layers.
        dim_feedforward: Hidden size of each decoder FFN block.
        dropout: Dropout probability used inside decoder layers.
        group_type: Symmetry group for filters: "rotation", "reflection", or "roto_reflection".
        n_rotations: Number of discrete rotations used by rotation-based filters.
        reflection_axis: Reflection axis used when group_type includes reflections.
        soft_thresholding: Softness for equivariant filtering of linear/attention weights.
        soft_thresholding_pos: Softness for invariant filtering of positional embeddings.
        convert_layer_norms: If True, replaces nn.LayerNorm with EQLayerNorm.
        nonlinearity: Decoder activation ('gelu', 'relu', 'identity', or 'eq_nonlin').
        hard: If True, uses hard thresholding in invariant filters.
    """
    
    def __init__(
        self,
        max_people: int,
        obs_len: int,
        pred_len: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        group_type: str = "rotation",
        n_rotations: int = 4,
        reflection_axis: str = 'x',
        soft_thresholding: float = 0.0,
        soft_thresholding_pos: float = None,
        convert_layer_norms: bool = False,
        nonlinearity: str = 'gelu',
        hard: bool = False  # If True, use hard thresholding for invariant filter
    ):
        """Initialize filtered autoregressive transformer."""
        # Validate group_type
        if group_type not in VALID_GROUP_TYPES:
            raise ValueError(f"Invalid group_type: {group_type}. Must be one of {VALID_GROUP_TYPES}")
        
        # Initialize parent class (creates unfiltered transformer)
        super().__init__(
            max_people=max_people,
            obs_len=obs_len,
            pred_len=pred_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            nonlinearity=nonlinearity
        )
        if soft_thresholding_pos is None:
            soft_thresholding_pos = soft_thresholding
            
        # Store whether to convert LayerNorms to Identity
        self.convert_layer_norms = convert_layer_norms
        
        # Store filter configuration
        self.filter_config = {
            'group_type': group_type,
            'n_rotations': n_rotations,
            'reflection_axis': reflection_axis,
            'soft_thresholding': soft_thresholding,
            'soft_thresholding_pos': soft_thresholding_pos,
            'dim': 2,  # 2D trajectories
            'max_people': max_people,
        }
        
        print(f"\n{'='*80}")
        print(f"Creating Filtered Autoregressive Transformer")
        print(f"{'='*80}")
        print(f"Architecture: max_people={max_people}, obs_len={obs_len}, pred_len={pred_len}")
        print(f"Model: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}")
        print(f"Group: type={group_type}, n_rotations={n_rotations}, reflection_axis={reflection_axis}")
        print(f"Filter: soft_thresholding={soft_thresholding}")
        print(f"Options: nonlinearity={nonlinearity}, convert_layer_norms={convert_layer_norms}")
        print(f"{'='*80}\n")
        
        # Create filters based on group_type
        self._create_filters(
            group_type=group_type,
            n_rotations=n_rotations,
            reflection_axis=reflection_axis,
            soft_thresholding=soft_thresholding,
            soft_thresholding_pos=soft_thresholding_pos,
            hard=hard
        )
        
        # Apply filters to all Linear and MultiheadAttention modules
        self._apply_filters_to_all_modules()
        
        print(f"\n{'='*80}")
        print("Filtered Autoregressive Transformer initialized successfully!")
        print(f"{'='*80}\n")
    
    def _create_filters(
        self,
        group_type: str,
        n_rotations: int,
        reflection_axis: str,
        soft_thresholding: float,
        soft_thresholding_pos: float,
        hard: bool
    ):
        """
        Create equivariant and invariant filters based on group type.
        
        Args:
            group_type: Type of group action ("rotation", "reflection", "roto_reflection")
            n_rotations: Number of discrete rotations
            reflection_axis: Axis for reflection ('x' or 'y' for 2D)
            soft_thresholding: Soft thresholding for equivariant filter
            soft_thresholding_pos: Soft thresholding for invariant filter
            hard: Use hard thresholding for invariant filter
        """
        if group_type == "rotation":
            print("Creating ROTATION equivariant filter for 2D trajectory vectors...")
            self.filter_eq = get_equivariant_filter_rotation(
                n_rotations=n_rotations,
                input_size=2,  # 2D vectors (x, y)
                output_size=2,
                soft_threshold=soft_thresholding,
                vector=True,
                debug=False
            )
            print("✓ Rotation equivariant filter created")
            
            print("Creating ROTATION invariant filter for biases...")
            self.filter_inv = get_invariant_filter_rotation(
                n_rotations=n_rotations,
                input_size=2,  # 2D vectors
                soft_threshold=soft_thresholding_pos,
                vector=True,
                debug=False,
                hard=hard
            )
            print("✓ Rotation invariant filter created\n")
            
        elif group_type == "reflection":
            print(f"Creating REFLECTION equivariant filter (axis={reflection_axis}) for 2D trajectory vectors...")
            self.filter_eq = get_equivariant_filter_reflection(
                input_size=2,  # 2D vectors (x, y)
                output_size=2,
                soft_threshold=soft_thresholding,
                vector=True,
                axis=reflection_axis,
                debug=False
            )
            print("✓ Reflection equivariant filter created")
            
            print(f"Creating REFLECTION invariant filter (axis={reflection_axis}) for biases...")
            self.filter_inv = get_invariant_filter_reflection(
                input_size=2,  # 2D vectors
                soft_threshold=soft_thresholding_pos,
                vector=True,
                axis=reflection_axis,
                debug=False,
                hard=hard
            )
            print("✓ Reflection invariant filter created\n")
            
        elif group_type == "roto_reflection":
            print(f"Creating ROTO-REFLECTION equivariant filter (n_rotations={n_rotations}, axis={reflection_axis})...")
            self.filter_eq = get_equivariant_filter_roto_reflection(
                n_rotations=n_rotations,
                input_size=2,  # 2D vectors (x, y)
                output_size=2,
                soft_threshold=soft_thresholding,
                vector=True,
                axis=reflection_axis,
                debug=False
            )
            print("✓ Roto-reflection equivariant filter created")
            
            print(f"Creating ROTO-REFLECTION invariant filter (n_rotations={n_rotations}, axis={reflection_axis})...")
            self.filter_inv = get_invariant_filter_roto_reflection(
                n_rotations=n_rotations,
                input_size=2,  # 2D vectors
                soft_threshold=soft_thresholding_pos,
                vector=True,
                axis=reflection_axis,
                debug=False,
                hard=hard
            )
            print("✓ Roto-reflection invariant filter created\n")
        else:
            raise ValueError(f"Unknown group_type: {group_type}")

    
    def _apply_filters_to_all_modules(self):
        """
        Traverse all modules and replace:
        1. nn.Linear → FilteredLinear (with filter_eq and filter_inv)
        2. nn.MultiheadAttention → FilteredMultiheadAttention (with filter_eq only)
        
        Skip:
        - nn.Dropout (no parameters)
        - Other activation functions
        """
        print("Applying filters to all Linear and MultiheadAttention modules...")
        print("-" * 80)
        
        # Initialize counters
        counts = {'linear': 0, 'attention': 0}
        
        # Recursively replace modules
        counts = self._replace_modules_recursive(
            self, "model", self.filter_eq, self.filter_inv, counts
        )
        
        print("-" * 80)
        print(f"✓ Replaced {counts['linear']} Linear layers with FilteredLinear")
        print(f"✓ Replaced {counts['attention']} MultiheadAttention modules with FilteredMultiheadAttention")
        if counts.get('layernorm', 0) > 0:
            print(f"✓ Converted {counts['layernorm']} LayerNorm layers to Identity")
        
        return counts
    
    def _replace_modules_recursive(self, module, module_name, filter_eq, filter_inv, counts):
        """
        Recursively traverse module tree and replace Linear/MultiheadAttention.
        
        Args:
            module: Current module to examine
            module_name: Name/path of current module
            filter_eq: Equivariant filter
            filter_inv: Invariant filter
            counts: Dictionary tracking replacement counts
            
        Returns:
            Updated counts dictionary
        """
        # Get list of children (create list to avoid modification during iteration)
 
        children_list = list(module.named_children())
        
        for child_name, child_module in children_list:
            child_path = f"{module_name}.{child_name}"
            
            # Check if this is a Linear layer
            if isinstance(child_module, nn.Linear) and not isinstance(child_module, FilteredLinear):
                in_features = child_module.in_features
                out_features = child_module.out_features
                has_bias = child_module.bias is not None
                
                print(f"  [Linear] {child_path}")
                print(f"    Shape: ({in_features} → {out_features}), bias={has_bias}")
                
                # Create FilteredLinear wrapper
                filtered_linear = FilteredLinear(
                    original_layer=child_module,
                    filter_eq=filter_eq,
                    filter_inv=filter_inv,
                )
                
                # Replace the layer
                setattr(module, child_name, filtered_linear)
                counts['linear'] += 1
                print(f"    → Replaced with FilteredLinear")
            
            # Check if this is a MultiheadAttention module
            elif isinstance(child_module, nn.MultiheadAttention) and not isinstance(child_module, FilteredMultiheadAttention):
                embed_dim = child_module.embed_dim
                num_heads = child_module.num_heads
                dropout = child_module.dropout
                batch_first = child_module.batch_first
                
                print(f"  [MultiheadAttention] {child_path}")
                print(f"    Config: embed_dim={embed_dim}, heads={num_heads}, dropout={dropout}, batch_first={batch_first}")
                
                # Create FilteredMultiheadAttention wrapper
                filtered_mha = FilteredMultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    filter_eq=filter_eq,
                    filter_inv=filter_inv,  # Don't use invariant filter for attention
                    dropout=dropout,
                    batch_first=batch_first
                )
                
                # Copy weights from original module
                try:
                    filtered_mha.load_state_dict(child_module.state_dict(), strict=False)
                    print(f"    → Copied weights from original")
                except Exception as e:
                    print(f"    → Warning: Could not copy weights: {e}")
                
                # Replace the module
                setattr(module, child_name, filtered_mha)
                counts['attention'] += 1
                print(f"    → Replaced with FilteredMultiheadAttention")
            
            # Handle LayerNorm (convert to Identity if requested)
            elif isinstance(child_module, nn.LayerNorm):
                if self.convert_layer_norms:
                    print(f"  [LayerNorm] {child_path}")
                    print(f"    → Replaced with Identity (no-op)")
                    new_layer_norm = EQLayerNorm(
                        normalized_shape=child_module.normalized_shape,
                        dim=2,
                        eps=child_module.eps,
                        elementwise_affine=child_module.elementwise_affine
                    )
                    setattr(module, child_name, new_layer_norm)
                    counts['layernorm'] = counts.get('layernorm', 0) + 1
                else:
                    print(f"  [LayerNorm] {child_path} - SKIPPED (already equivariant)")
            
            # Recursively process children (unless it's a leaf layer we just replaced)
            if not isinstance(child_module, (FilteredLinear, FilteredMultiheadAttention)):
                counts = self._replace_modules_recursive(
                    child_module, child_path, filter_eq, filter_inv, counts
                )
        
        return counts
    
    def forward(self, obs_traj, validity_mask, gt_traj=None):
        """
        Forward pass with positional-embedding smoothing before decoding.
        
        Args:
            obs_traj: Observed trajectories with shape (B, P, 2, T_obs).
            validity_mask: Person validity mask with shape (B, P).
                Passed through to the parent forward for API compatibility.
            gt_traj: Optional teacher-forcing targets with shape (B, P, 2, T_pred).
        
        Returns:
            Predicted future trajectories with shape (B, P, 2, T_pred).
        """
        self.pos_encoder.data = self.filter_inv.smooth(self.pos_encoder.data, data_last=True)
        return super().forward(obs_traj, validity_mask, gt_traj)


def create_filtered_autoregressive_transformer(config):
    """
    Factory function to create a FilteredAutoregressiveTransformer model.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        FilteredAutoregressiveTransformer model instance
    """
    model = FilteredAutoregressiveTransformer(
        max_people=config.get('max_people', 10),
        obs_len=config.get('obs_len', 8),
        pred_len=config.get('pred_len', 12),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        group_type=config.get('group_type', 'rotation'),
        n_rotations=config.get('n_rotations', 4),
        reflection_axis=config.get('reflection_axis', 'x'),
        soft_thresholding=config.get('soft_thresholding', config.get('soft_threshold', 0.0)),
        soft_thresholding_pos=config.get('soft_thresholding_pos', None),
        convert_layer_norms=config.get('convert_layer_norms', False),
        nonlinearity=config.get('nonlinearity', 'gelu'),
        hard=config.get('hard', False)
    )
    return model


if __name__ == "__main__":
    """Quick test of the filtered model with different group types."""
    print("\n" + "="*80)
    print("TESTING FILTERED AUTOREGRESSIVE TRANSFORMER")
    print("="*80 + "\n")
    
    # Test configuration (small model for quick testing)
    batch_size = 2
    max_people = 5
    obs_len = 4
    pred_len = 6
    d_model = 64
    num_heads = 2
    num_layers = 2
    
    # Create dummy data
    obs_traj = torch.randn(batch_size, max_people, 2, obs_len)
    validity_mask = torch.ones(batch_size, max_people)
    gt_traj = torch.randn(batch_size, max_people, 2, pred_len)
    
    # Test 1: Rotation group
    print("\n" + "="*80)
    print("TEST 1: ROTATION group with GELU activation")
    print("="*80)
    model1 = FilteredAutoregressiveTransformer(
        max_people=max_people,
        obs_len=obs_len,
        pred_len=pred_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=128,
        dropout=0.1,
        group_type='rotation',
        n_rotations=4,
        soft_thresholding=0.0,
        convert_layer_norms=False,
        nonlinearity='gelu'
    )
    model1.eval()
    with torch.no_grad():
        pred1 = model1(obs_traj, validity_mask)
    print(f"✓ Rotation model output shape: {pred1.shape}")
    
    # Test 2: Reflection group
    print("\n" + "="*80)
    print("TEST 2: REFLECTION group (axis=x)")
    print("="*80)
    model2 = FilteredAutoregressiveTransformer(
        max_people=max_people,
        obs_len=obs_len,
        pred_len=pred_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=128,
        dropout=0.1,
        group_type='reflection',
        reflection_axis='x',
        soft_thresholding=0.0,
        convert_layer_norms=False,
        nonlinearity='gelu'
    )
    model2.eval()
    with torch.no_grad():
        pred2 = model2(obs_traj, validity_mask)
    print(f"✓ Reflection model output shape: {pred2.shape}")
    
    # Test 3: Roto-reflection group
    print("\n" + "="*80)
    print("TEST 3: ROTO-REFLECTION group (n_rotations=4, axis=x)")
    print("="*80)
    model3 = FilteredAutoregressiveTransformer(
        max_people=max_people,
        obs_len=obs_len,
        pred_len=pred_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=128,
        dropout=0.1,
        group_type='roto_reflection',
        n_rotations=4,
        reflection_axis='x',
        soft_thresholding=0.0,
        convert_layer_norms=False,
        nonlinearity='gelu'
    )
    model3.eval()
    with torch.no_grad():
        pred3 = model3(obs_traj, validity_mask)
    print(f"✓ Roto-reflection model output shape: {pred3.shape}")
    
    # Test 4: Model with identity nonlinearity and LayerNorms converted
    print("\n" + "="*80)
    print("TEST 4: Rotation with identity activation and converted LayerNorms")
    print("="*80)
    model4 = FilteredAutoregressiveTransformer(
        max_people=max_people,
        obs_len=obs_len,
        pred_len=pred_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=128,
        dropout=0.1,
        group_type='rotation',
        n_rotations=4,
        soft_thresholding=0.0,
        convert_layer_norms=True,
        nonlinearity='identity'
    )
    
    total_params = sum(p.numel() for p in model4.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test training mode
    print("\nTesting training mode (with teacher forcing)...")
    model4.train()
    pred_train = model4(obs_traj, validity_mask, gt_traj)
    print(f"✓ Training output shape: {pred_train.shape}")
    assert pred_train.shape == (batch_size, max_people, 2, pred_len), "Training output shape mismatch!"
    
    # Test inference mode
    print("\nTesting inference mode (autoregressive)...")
    model4.eval()
    with torch.no_grad():
        pred_eval = model4(obs_traj, validity_mask)
    print(f"✓ Inference output shape: {pred_eval.shape}")
    assert pred_eval.shape == (batch_size, max_people, 2, pred_len), "Inference output shape mismatch!"
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80 + "\n")

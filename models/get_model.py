from models.test_models import TestModel, TestEqModel, TestEqModelResidual
from models.residualCnn import ResidualCnn
from models.filtered_vit import create_filtered_vit
from models.filtered_segformer import create_filtered_segformer
from models.filtered_dino2 import create_filtered_dinov2
from models.filtered_vit_seg import create_filtered_vit_seg
from models.filtered_dino2_seg import create_filtered_dino2_seg
from models.filtered_resnet import create_filtered_resnet
from models.filtered_o5 import create_filtered_o5_mlp
from models.filtered_lorentz import create_filtered_lorentz_mlp
from models.emlp_models import create_emlp_o5_mlp, create_emlp_lorentz_mlp
from softeq.equi_utils.equi_constraints import DiscreteRotationConstraints
from canonicalization.network import CanonicalizationNetwork
from canonicalization.wrapper import ClassificationCanonicalizationWrapper, SegmentationCanonicalizationWrapper
import torch

def _create_canonicalization_network(canon_config, model_config=None):
    """
    Create canonicalization network from config.
    
    Args:
        canon_config: Dictionary with canonicalization parameters
            - n_rotations: Number of rotations (default: 4)
            - in_channels: Input channels (default: 3)
            - mid_channels: Middle layer channels (default: [16, 16])
            - out_channels: Output channels (default: 1)
            - group_type: One of "rotation", "reflection", "roto_reflection"
            - use_reflection: (Deprecated) If True and group_type is None, uses roto_reflection
        model_config: Optional parent model config to inherit group_type from
        
    Returns:
        CanonicalizationNetwork instance
    """
    # Determine group_type with priority:
    # 1. Explicit group_type in canon_config
    # 2. group_type from parent model_config
    # 3. Backward compatibility: use_reflection parameter
    # 4. Default to "rotation"
    group_type = canon_config.get('group_type', None)
    
    if group_type is None and model_config is not None:
        # Inherit from parent model config
        group_type = model_config.get('group_type', None)
    
    if group_type is None:
        # Backward compatibility: use_reflection=True -> roto_reflection
        use_reflection = canon_config.get('use_reflection', None)
        if use_reflection is not None:
            group_type = "roto_reflection" if use_reflection else "rotation"
        else:
            # Default to rotation
            group_type = "rotation"
    
    # Get n_rotations with fallback to parent model config
    n_rotations = canon_config.get('n_rotations', None)
    if n_rotations is None and model_config is not None:
        n_rotations = model_config.get('n_rotations', 4)
    if n_rotations is None:
        n_rotations = 4
    
    model = CanonicalizationNetwork(
        n_rotations=n_rotations,
        in_channels=canon_config.get('in_channels', 3),
        mid_channels=canon_config.get('mid_channels', [16, 16]),
        out_channels=canon_config.get('out_channels', 1),
        group_type=group_type,
    )
    # Forward pass to initialize weights of the equivariant layers
    # Ensure model is on CPU for initialization
    model = model.cpu()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Canonicalization network created with {n_params} trainable parameters.")
    with torch.no_grad():
        dummy_input = torch.randn(1, model.in_channels, 32, 32)
        _ = model(dummy_input)
    # Count parameters after initialization
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Canonicalization network created with {n_params} trainable parameters.")
    return model


def get_model(config):
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary containing model parameters.
               For 'filtered_o5', expects full config with 'model' section.
               For other models, expects model-specific config.
    
    Returns:
        Model instance
    """
    # Handle case where config is the full config dict (has 'model' key) vs model-specific config
    if isinstance(config, dict) and 'model' in config and 'type' not in config:
        # Full config dict passed - extract model config
        model_config = config.get('model', {})
        model_type = model_config.get('type', 'test_model')
    else:
        # Model-specific config passed directly
        model_config = config
        model_type = model_config.get('type', 'test_model')
    
    # Regular models (no canonicalization)
    if model_type == 'filtered_vit':
        return create_filtered_vit(model_config)
    elif model_type == 'filtered_dinov2':
        return create_filtered_dinov2(model_config)
    elif model_type == 'filtered_resnet':
        return create_filtered_resnet(model_config)
    elif model_type == 'filtered_o5':
        # O(5)-specific equivariant MLP - pass full config with 'model' key
        return create_filtered_o5_mlp(model_config)
    elif model_type == 'filtered_lorentz':
        return create_filtered_lorentz_mlp(model_config)
    elif model_type == 'emlp_o5':
        return create_emlp_o5_mlp(model_config)
    elif model_type == 'emlp_lorentz':
        return create_emlp_lorentz_mlp(model_config)
    elif model_type == 'filtered_vit_seg':
        return create_filtered_vit_seg(model_config)
    elif model_type == 'filtered_dino2_seg':
        return create_filtered_dino2_seg(model_config)
    elif model_type == 'filtered_segformer':
        return create_filtered_segformer(model_config)
    
    
    # Canonicalization baselines (wrapped models)
    elif model_type == 'filtered_vit_canon':
        base_model = create_filtered_vit(config)
        canon_net = _create_canonicalization_network(config.get('canonicalization', {}), config)
        return ClassificationCanonicalizationWrapper(base_model, canon_net)
    elif model_type == 'filtered_dinov2_canon':
        base_model = create_filtered_dinov2(config)
        canon_net = _create_canonicalization_network(config.get('canonicalization', {}), config)
        return ClassificationCanonicalizationWrapper(base_model, canon_net)
    elif model_type == 'filtered_resnet_canon':
        base_model = create_filtered_resnet(config)
        canon_net = _create_canonicalization_network(config.get('canonicalization', {}), config)
        return ClassificationCanonicalizationWrapper(base_model, canon_net)
    elif model_type == 'filtered_vit_seg_canon':
        base_model = create_filtered_vit_seg(config)
        canon_net = _create_canonicalization_network(config.get('canonicalization', {}), config)
        return SegmentationCanonicalizationWrapper(base_model, canon_net)
    elif model_type == 'filtered_dinov2_seg_canon': 
        base_model = create_filtered_dino2_seg(config)
        canon_net = _create_canonicalization_network(config.get('canonicalization', {}), config)
        return SegmentationCanonicalizationWrapper(base_model, canon_net)
    elif model_type == 'filtered_segformer_canon':
        base_model = create_filtered_segformer(config)
        canon_net = _create_canonicalization_network(config.get('canonicalization', {}), config)
        return SegmentationCanonicalizationWrapper(base_model, canon_net)
    elif model_type == 'residual_cnn':
        return ResidualCnn(
            num_layers=config['num_layers'],
            residual_strength=config['residual_strength'],
            num_classes=config['num_labels'],
            in_channels=config.get('in_channels', 3),
            base_channels=config.get('base_channels', 64),
            n_rotations=config.get('n_rotations', 4),
            group_type=config.get('group_type', 'rotation'),
            use_regular_representation=config.get('use_regular_representation', False),
        )
    
    # Trajectory prediction models
    elif model_type == 'autoregressive_transformer':
        # Create AutoregressiveTransformer for trajectory prediction
        from models.autoregressive_transformer import AutoregressiveTransformer
        return AutoregressiveTransformer(
            max_people=config['max_people'],
            obs_len=config['obs_len'],
            pred_len=config['pred_len'],
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 4),
            dim_feedforward=config.get('dim_feedforward', 1024),
            dropout=config.get('dropout', 0.1),
            nonlinearity=config['nonlinearity']
        )
    elif model_type == 'filtered_autoregressive_transformer':
        # Create FilteredAutoregressiveTransformer for equivariant trajectory prediction
        from models.filtered_autoreg_transformer import FilteredAutoregressiveTransformer
        return FilteredAutoregressiveTransformer(
            max_people=config['max_people'],
            obs_len=config['obs_len'],
            pred_len=config['pred_len'],
            d_model=config.get('d_model', 64),
            num_heads=config.get('num_heads', 2),
            num_layers=config.get('num_layers', 2),
            dim_feedforward=config.get('dim_feedforward', 128),
            dropout=config.get('dropout', 0.1),
            n_rotations=config['n_rotations'],
            soft_thresholding=config['soft_thresholding'],
            soft_thresholding_pos=config['soft_thresholding_pos'],
            convert_layer_norms=config['convert_layer_norms'],
            nonlinearity=config['nonlinearity'],
            hard=config.get('hard', True)
        )
    elif model_type == 'fully_connected_eq':
        model = TestEqModel(
                            nlayers=config['nlayers'],
                            input_size=config['image_size'],
                            feature_size=config['feature_size'],
                            n_rotations=config.get('n_rotations'),
                            nclasses=config['nclasses'],
                            soft_thresholding=config['soft_thresholding'], 
                            soft_thresholding_reflection=config.get('soft_thresholding_reflection'),
                            soft_thresholding_rotation=config.get('soft_thresholding_rotation'),
                            decomposition_method=config['decomposition_method'], 
                            enforce_type=config['enforce_type'],
                            hard_mask=config.get('hard_mask', True),
                            reflection=config.get('reflection', False)
        )
        return model
    elif model_type == 'fully_connected_eq_residual':
        model = TestEqModelResidual(
                            nlayers=config['nlayers'],
                            input_size=config['image_size'],
                            feature_size=config['feature_size'],
                            n_rotations=config.get('n_rotations'),
                            nclasses=config['nclasses'],
                            soft_thresholding=config['soft_thresholding'],
                            soft_thresholding_reflection=config.get('soft_thresholding_reflection'),
                            soft_thresholding_rotation=config.get('soft_thresholding_rotation'),
                            decomposition_method=config['decomposition_method'],
                            reflection=config.get('reflection', False)
        )
        return model
    elif model_type == 'test_model':
        # Create TestModel (original functionality)
        nlayers = config['nlayers']
        feature_size = config['feature_size']
        nclasses = config['nclasses']
        image_size = config['image_size']
        output_size = config.get('output_size', image_size)
        soft_thresholding = config['soft_thresholding']
        enforce_equivariance = config['enforce_equivariance']
        decomposition_method = config['decomposition_method']
        enforce_type = config['enforce_type']

        if not enforce_equivariance:
            model = TestModel(nlayers=nlayers, 
                            input_size=image_size,
                            feature_size=feature_size,
                            n_rotations=config['n_rotations'],
                            nclasses=nclasses,
                            soft_thresholding=soft_thresholding,
                            decomposition_method=decomposition_method,
                            enforce_type=enforce_type)
        else:
            raise NotImplementedError("Please use 'fully_connected_eq' model type for equivariant TestEqModel.")
        return model

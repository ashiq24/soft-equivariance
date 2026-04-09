"""
Filtered ResNet Model for Image Classification.

This module implements an image classification model that combines:
1. ResNet backbone (Residual Networks)
2. Classification head
3. FilteredConv2d layers applied to all Conv2d layers with kernel_size > 1

The model loads pre-trained weights from HuggingFace and automatically
replaces all convolutional layers with kernel size > 1 with filtered versions.

ResNet Architecture Details:
1. Stem (embeddings): 7x7 convolution + maxpool
2. 4 stages with bottleneck/basic blocks
3. Each bottleneck block: 1x1 conv -> 3x3 conv -> 1x1 conv + residual
4. Shortcut connections use 1x1 conv when dimensions don't match
5. Global average pooling + classification head
"""

import torch
import torch.nn as nn
from transformers import ResNetForImageClassification
from softeq.layers.fconv2d import FilteredConv2d
from softeq.equi_utils.filter_factory import get_invariant_filter


class FilteredResNet(ResNetForImageClassification):
    """
    ResNet model with filtered convolutions applied to all Conv2d layers with kernel_size > 1.
    
    This model:
    1. Loads a pre-trained ResNet model from HuggingFace
    2. Replaces the final classifier to match the target number of classes
    3. Automatically replaces Conv2d layers with kernel_size > 1 with FilteredConv2d layers
    4. Skips 1x1 convolutions (used for dimension matching in residual connections)
    """
    
    def __init__(self, filter_config):
        """
        Initialize the filtered ResNet model.
        
        Args:
            filter_config: Configuration dictionary containing:
                - pretrained_model: HuggingFace model name (e.g., "microsoft/resnet-50")
                - num_labels: Number of classification classes (e.g., 10 for CIFAR-10)
                - group_type: Group used to build invariant filters.
                - n_rotations: Number of rotations for the filter (default: 4)
                - soft_thresholding: Soft thresholding value for the filter (default: 0.0)
                - decomposition_method: Decomposition method for the filter (default: 'schur')
                - hard_mask: Whether to use a hard mask in filter construction.
                - preserve_norm: Whether to preserve weight norms after projection.
                - joint_decomposition: Whether to use joint decomposition for multi-generator groups.
                - load_pretrained_weight: Whether to load pretrained weights (default: True)
                - freeze_filters: Whether to freeze filtered convolution weights.
                - min_filter_size: Minimum kernel size to apply filtering.
        """
        
        # Load pre-trained ResNet model
        pretrained_model = filter_config.get('pretrained_model', 'microsoft/resnet-50')
        load_pretrained_weight = filter_config.get('load_pretrained_weight', True)
        print(f"Loading pre-trained model: {pretrained_model}")
        
        try:
            resnet = ResNetForImageClassification.from_pretrained(pretrained_model)
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Attempting to load with explicit config...")
            from transformers import ResNetConfig
            model_config = ResNetConfig.from_pretrained(pretrained_model)
            resnet = ResNetForImageClassification(model_config)
        
        # Initialize parent class with config
        super().__init__(resnet.config)
        
        # Copy pretrained weights if requested
        if load_pretrained_weight:
            print("Loading pretrained weights...")
            self.load_state_dict(resnet.state_dict(), strict=False)
        else:
            print("Skipping pretrained weights - using random initialization")
        
        # Store filter config
        self.filter_config = filter_config
        
        # Get number of classes
        num_labels = filter_config['num_labels']
        
        # Replace the classifier head to match target number of classes
        # The pre-trained model is for ImageNet (1000 classes), we need to adapt it
        if hasattr(self, 'classifier') and hasattr(self.classifier, '1') and num_labels != self.classifier[1].out_features:
            # ResNet classifier is a Sequential with flatten and linear layer
            original_classifier = self.classifier[1]  # The Linear layer
            in_features = original_classifier.in_features
            
            self.classifier[1] = nn.Linear(in_features, num_labels)
            print(f"Replaced classifier: {original_classifier.out_features} -> {num_labels} classes")
        else:
            print("Warning: Could not find classifier to replace or number of labels matches pre-trained model.")
        
        # Apply filtering to all Conv2d layers with kernel_size > 1
        if filter_config['soft_thresholding'] < 1.0:
            self._apply_filters_to_all_conv_layers()
        
        # Freeze filtered convolution weights if requested
        self._freeze_filtered_convs()
    
    def _apply_filters_to_all_conv_layers(self):
        """
        Iterate through all layers and replace Conv2d layers with kernel_size > 1 
        with FilteredConv2d layers.
        
        Strategy:
        - Filter Conv2d layers with kernel_size > 1
        - Skip 1x1 convolutions (used for dimension matching)
        - Optionally skip small 3x3 filters with high soft thresholding
        """
        filter_config = self.filter_config
        group_type = filter_config.get('group_type', 'rotation')
        n_rotations = filter_config['n_rotations']
        soft_thresholding = filter_config['soft_thresholding']
        decomposition_method = filter_config['decomposition_method']
        hard_mask = filter_config.get('hard_mask', False)
        preserve_norm = filter_config['preserve_norm']
        joint_decomposition = filter_config.get('joint_decomposition', True)
        
        print(f"Applying invariant filters to all Conv2d layers with kernel_size > 1")
        print(f"Filter config: group_type={group_type}, n_rotations={n_rotations}, soft_threshold={soft_thresholding}, method={decomposition_method}")
        
        # Counter for filtered layers
        filtered_count = 0
        
        # Recursively find and replace Conv2d layers
        filtered_count += self._replace_conv_layers_recursive(
            self, "resnet", group_type, n_rotations, soft_thresholding, decomposition_method, hard_mask, preserve_norm, joint_decomposition
        )
        
        print(f"Total Conv2d layers filtered: {filtered_count}")
    
    def _replace_conv_layers_recursive(self, module, module_name, group_type, n_rotations, soft_thresholding, decomposition_method, hard_mask, preserve_norm, joint_decomposition):
        """
        Recursively traverse the module tree and replace Conv2d layers with kernel_size > 1.
        
        Args:
            module: Current module to examine
            module_name: Name/path of the current module
            group_type: Type of group action - "rotation" or "roto_reflection"
            n_rotations: Number of rotations for the filter
            soft_thresholding: Soft thresholding value
            decomposition_method: Decomposition method
            
        Returns:
            int: Number of layers that were filtered
        """
        filtered_count = 0
        
        # Get all child modules
        for child_name, child_module in module.named_children():
            child_path = f"{module_name}.{child_name}"
            
            # Check if this is a Conv2d layer
            if isinstance(child_module, nn.Conv2d):
                kernel_size = child_module.kernel_size[0]  # Assuming square kernels
                
                # Verify square kernel
                if kernel_size != child_module.kernel_size[1]:
                    print(f"  Warning: Non-square kernel at {child_path}: {child_module.kernel_size}")
                    # Continue anyway, use the first dimension
                
                in_channels = child_module.in_channels
                out_channels = child_module.out_channels
                stride = child_module.stride
                padding = child_module.padding
                dilation = child_module.dilation
                
                # Decision: Only filter convolutions with kernel_size > 1
                # This skips 1x1 convolutions used for dimension matching
                if kernel_size > 1:
                    # Optional: Skip small filters with high soft thresholding
                    if kernel_size <= 3 and soft_thresholding > 0.8:
                        print(f"  Skipping: {child_path} - Conv2d({in_channels}, {out_channels}, "
                              f"kernel_size={kernel_size}) (small filter with high soft threshold {soft_thresholding})")
                    else:
                        # This layer should be filtered
                        print(f"  Filtering: {child_path} - Conv2d({in_channels}, {out_channels}, "
                              f"kernel_size={kernel_size}), stride={stride}, padding={padding}, dilation={dilation}")
                        
                        # Create invariant filter based on group type
                        filter_module = get_invariant_filter(
                            group_type=group_type,
                            n_rotations=n_rotations,
                            input_size=(1, kernel_size, kernel_size),
                            soft_threshold=soft_thresholding,
                            decomposition_method=decomposition_method,
                            debug=False,
                            hard_mask=hard_mask,
                            preserve_norm=preserve_norm,
                            joint_decomposition=joint_decomposition
                        )
                        
                        # Create FilteredConv2d wrapper
                        filtered_conv = FilteredConv2d(
                            original_layer=child_module,
                            filter=filter_module
                        )
                        
                        # Replace the layer
                        setattr(module, child_name, filtered_conv)
                        filtered_count += 1
                        print(f"    -> Replaced with FilteredConv2d")
                else:
                    print(f"  Skipping: {child_path} - Conv2d({in_channels}, {out_channels}, "
                          f"kernel_size={kernel_size}) (1x1 convolution for dimension matching)")
            
            # Recursively process child modules
            filtered_count += self._replace_conv_layers_recursive(
                child_module, child_path, group_type, n_rotations, soft_thresholding, decomposition_method, hard_mask, 
                preserve_norm, joint_decomposition
            )
        
        return filtered_count
    
    def _freeze_filtered_convs(self):
        """
        Freeze FilteredConv2d layers based on min_filter_size configuration.
        Only freezes layers where kernel_size >= min_filter_size.
        """
        freeze_filters = self.filter_config.get('freeze_filters', False)
        min_filter_size = self.filter_config.get('min_filter_size', 1)
        
        if not freeze_filters:
            return
        
        print(f"Freezing filtered convolution weights with kernel_size >= {min_filter_size}...")
        frozen_count = 0
        
        # Recursively find and freeze FilteredConv2d layers
        frozen_count = self._freeze_filtered_convs_recursive(self, "resnet", min_filter_size)
        
        print(f"Total filtered Conv2d layers frozen: {frozen_count}")
    
    def _freeze_filtered_convs_recursive(self, module, module_name, min_filter_size):
        """
        Recursively traverse the module tree and freeze FilteredConv2d layers.
        
        Args:
            module: Current module to examine
            module_name: Name/path of the current module
            min_filter_size: Minimum kernel size to freeze
            
        Returns:
            int: Number of layers that were frozen
        """
        frozen_count = 0
        
        for child_name, child_module in module.named_children():
            child_path = f"{module_name}.{child_name}"
            
            # Check if this is a FilteredConv2d layer
            if isinstance(child_module, FilteredConv2d):
                kernel_size = child_module.kernel_size[0]  # Assuming square kernels
                
                # Only freeze if kernel_size >= min_filter_size
                if kernel_size >= min_filter_size:
                    print(f"  Freezing: {child_path} - FilteredConv2d(kernel_size={kernel_size})")
                    child_module.weight.requires_grad = False
                    if child_module.bias is not None:
                        child_module.bias.requires_grad = False
                    frozen_count += 1
            
            # Recursively process child modules
            frozen_count += self._freeze_filtered_convs_recursive(child_module, child_path, min_filter_size)
        
        return frozen_count


def create_filtered_resnet(config):
    """
    Factory function to create a FilteredResNet model.
    
    Args:
        config: Configuration dictionary containing:
            - pretrained_model: HuggingFace model name
            - num_labels: Number of output classes
            - n_rotations: Number of rotations for filter
            - soft_thresholding: Soft thresholding value
            - decomposition_method: Decomposition method for filter
        
    Returns:
        FilteredResNet model instance
    """
    model = FilteredResNet(config)
    return model

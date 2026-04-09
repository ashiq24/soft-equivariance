import torch
import torch.nn as nn
from typing import List, Optional, Union
from softeq.layers.fconv2d import FilteredConv2d
from softeq.equi_utils.rotation_filters import get_invariant_filter_rotation

class SimpleCNN(nn.Module):
    def __init__(
        self, 
        num_classes: int = 10,
        in_channels: int = 1,
        channels: Optional[List[int]] = None,
        kernel_sizes: Optional[Union[int, List[int]]] = None,
        pool_positions: Optional[List[int]] = None,
        use_pooling: bool = True
    ):
        """
        Configurable Simple CNN architecture.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (default: 1 for grayscale)
            channels: List of channel sizes for each conv layer (default: [32, 64, 128])
            kernel_sizes: Single int or list of kernel sizes for each layer (default: 5 for all)
            pool_positions: List of layer indices after which to apply MaxPool2d (default: [1])
            use_pooling: Whether to use pooling at all (default: True)
        """
        super().__init__()
        
        # Set default values
        if channels is None:
            channels = [32, 64, 128]
        
        num_layers = len(channels)
        
        # Handle kernel_sizes
        if kernel_sizes is None:
            kernel_sizes = [5] * num_layers
        elif isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        
        # Validate inputs
        if len(kernel_sizes) != num_layers:
            raise ValueError(f"Length of kernel_sizes ({len(kernel_sizes)}) must match length of channels ({num_layers})")
        
        # Handle pool_positions
        if pool_positions is None:
            pool_positions = [1] if use_pooling and num_layers > 1 else []
        
        # Validate pool_positions
        for pos in pool_positions:
            if pos < 0 or pos >= num_layers:
                raise ValueError(f"Invalid pool position {pos}. Must be in range [0, {num_layers-1}]")
        
        # Build the feature extractor
        layers = []
        prev_channels = in_channels
        
        for i, (out_ch, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            # Calculate padding to maintain spatial dimensions
            padding = kernel_size // 2
            
            # Add convolutional layer
            layers.append(nn.Conv2d(prev_channels, out_ch, kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            
            # Add pooling if specified for this position
            if use_pooling and i in pool_positions:
                layers.append(nn.MaxPool2d(2))
            
            prev_channels = out_ch
        
        # Add adaptive pooling to get fixed size output
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], num_classes)
        
        # Store config for inspection
        self.config = {
            'num_classes': num_classes,
            'in_channels': in_channels,
            'channels': channels,
            'kernel_sizes': kernel_sizes,
            'pool_positions': pool_positions,
            'use_pooling': use_pooling
        }

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
class FilteredCNN(SimpleCNN):
    def __init__(self, filter_config:dict):
        self.filter_config = filter_config
        num_classes = int(self.filter_config['num_classes'])
        
        # Extract architecture config if provided
        arch_config = self.filter_config.get('architecture', {})
        in_channels = arch_config.get('in_channels', 1)
        channels = arch_config.get('channels', None)
        kernel_sizes = arch_config.get('kernel_sizes', None)
        pool_positions = arch_config.get('pool_positions', None)
        use_pooling = arch_config.get('use_pooling', True)
        hard_mask = self.filter_config.get('hard_mask', False)
        
        # Initialize parent with architecture config
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
            pool_positions=pool_positions,
            use_pooling=use_pooling
        )
        
        # Apply filters if soft_thresholding < 1.0
        if self.filter_config['soft_thresholding'] < 1.0:
            self._apply_filters_to_all_conv_layers()

    def _apply_filters_to_all_conv_layers(self):
        filter_config = self.filter_config
        n_rotations = filter_config['n_rotations']
        soft_thresholding = filter_config['soft_thresholding']
        decomposition_method = filter_config['decomposition_method']
        hard_mask = filter_config.get('hard_mask')

        print("Applying invariant filters to all Conv2d layers with kernel_size > 1")
        print(f"Filter config: n_rotations={n_rotations}, soft_threshold={soft_thresholding}, method={decomposition_method}")
        
        # Counter for filtered layers
        filtered_count = 0
        
        # Recursively find and replace Conv2d layers
        filtered_count += self._replace_conv_layers_recursive(
            self, "CNN", n_rotations, soft_thresholding, decomposition_method, hard_mask
        )
        
        print(f"Total Conv2d layers filtered: {filtered_count}")

    def _replace_conv_layers_recursive(self, module, module_name, n_rotations, soft_thresholding, decomposition_method, hard_mask):
        """
        Recursively traverse the module tree and replace Conv2d layers with kernel_size > 1.
        
        Args:
            module: Current module to examine
            module_name: Name/path of the current module
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
                    if kernel_size <= 3 and soft_thresholding > 0.95:
                        print(f"  Skipping: {child_path} - Conv2d({in_channels}, {out_channels}, "
                              f"kernel_size={kernel_size}) (small filter with high soft threshold {soft_thresholding})")
                    else:
                        # This layer should be filtered
                        print(f"  Filtering: {child_path} - Conv2d({in_channels}, {out_channels}, "
                              f"kernel_size={kernel_size}), stride={stride}, padding={padding}, dilation={dilation}")
                        
                        # Create invariant filter
                        filter_module = get_invariant_filter_rotation(
                            n_rotations=n_rotations,
                            input_size=(1, kernel_size, kernel_size),
                            soft_threshold=soft_thresholding,
                            decomposition_method=decomposition_method,
                            hard_mask=hard_mask,
                            debug=False
                        )
                        
                        # Create FilteredConv2d wrapper
                        filtered_conv = FilteredConv2d(
                            original_layer=child_module,
                            filter=filter_module
                        )
                        
                        # Replace the layer
                        setattr(module, child_name, filtered_conv)
                        filtered_count += 1
                        print("    -> Replaced with FilteredConv2d")
                else:
                    print(f"  Skipping: {child_path} - Conv2d({in_channels}, {out_channels}, "
                          f"kernel_size={kernel_size}) (1x1 convolution for dimension matching)")
            
            # Recursively process child modules
            filtered_count += self._replace_conv_layers_recursive(
                child_module, child_path, n_rotations, soft_thresholding, decomposition_method, hard_mask
            )
        
        return filtered_count
    
def create_filtered_cnn(config):
    model = FilteredCNN(config)
    return model
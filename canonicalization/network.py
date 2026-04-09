
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
from e2cnn import gspaces
from e2cnn import nn as enn
import math

# Valid group types
VALID_GROUP_TYPES = ["rotation", "reflection", "roto_reflection"]


class CanonicalizationNetwork(nn.Module):
    def __init__(
        self, 
        n_rotations: int = 8, 
        in_channels: int = 3, 
        mid_channels=None, 
        out_channels: int = 1, 
        group_type: str = "roto_reflection",
        reflection_axis: float = math.pi / 2  # Default: vertical reflection axis (pi/2)
    ):
        """
        Canonicalization network that learns to map inputs to a canonical pose.
        
        The network is equivariant to the specified group and outputs fibre features
        that can be used to determine the transformation needed to canonicalize the input.
        
        Args:
            n_rotations (int): Number of discrete rotations in the group.
                - For "rotation": C_n group with n rotations (order = n)
                - For "reflection": Ignored (order = 2)
                - For "roto_reflection": D_n group with n rotations + reflection (order = 2n)
            in_channels (int): Number of input channels (e.g., 3 for RGB)
            mid_channels (list[int]): List of channels for each intermediate layer
            out_channels (int): Number of output channels for the final feature map
            group_type (str): Type of group action. One of:
                - "rotation": Cyclic group C_n (rotations only)
                - "reflection": Reflection group (flip only, order 2)
                - "roto_reflection": Dihedral group D_n (rotations + reflections)
            reflection_axis (float): Axis of reflection in radians (default: pi/2 for vertical)
                - 0: horizontal reflection (flip across y-axis)
                - pi/2: vertical reflection (flip across x-axis)
        """
        super().__init__()
        
        if group_type not in VALID_GROUP_TYPES:
            raise ValueError(f"group_type must be one of {VALID_GROUP_TYPES}, got {group_type}")
        
        if mid_channels is None:
            mid_channels = [16, 16]
        if len(mid_channels) < 1:
            raise ValueError("mid_channels must be a list with at least one value")
        
        self.n_rotations = n_rotations
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.group_type = group_type
        self.reflection_axis = reflection_axis

        # Choose group based on group_type
        if group_type == "rotation":
            # Cyclic group C_n (rotations only)
            self.r2_act = gspaces.Rot2dOnR2(N=n_rotations)
            self.use_reflection = False
        elif group_type == "reflection":
            # Reflection group (flip only)
            self.r2_act = gspaces.Flip2dOnR2(axis=reflection_axis)
            self.use_reflection = True
        elif group_type == "roto_reflection":
            # Dihedral group D_n (rotations + reflections)
            if n_rotations < 2:
                raise ValueError(f"n_rotations must be >= 2 for roto_reflection, got {n_rotations}")
            self.r2_act = gspaces.FlipRot2dOnR2(N=n_rotations, axis=reflection_axis)
            self.use_reflection = True
        
        self.num_group_elements = self.r2_act.fibergroup.order()

        # Field types for each layer
        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * in_channels)
        mid_types = [enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * c) for c in mid_channels]
        self.out_type = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * out_channels)

        # Build equivariant network
        modules = []
        # Input: Trivial -> Regular
        modules.append(enn.R2Conv(self.in_type, mid_types[0], kernel_size=7, padding=3, bias=False))
        modules.append(enn.ReLU(mid_types[0], inplace=True))
        # Intermediate layers
        for i in range(1, len(mid_channels)):
            modules.append(enn.R2Conv(mid_types[i-1], mid_types[i], kernel_size=3, padding=1, bias=False))
            modules.append(enn.ReLU(mid_types[i], inplace=True))
        # Output: Regular -> Regular
        modules.append(enn.R2Conv(mid_types[-1], self.out_type, kernel_size=3, padding=1, bias=False))
        self.canonicalization_network = enn.SequentialModule(*modules)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the equivariant network.
        
        Args:
            images: Input images of shape (B, C, H, W)
            
        Returns:
            fibre_features: Features for each group element, shape (B, num_group_elements)
        """
        x_geom = enn.GeometricTensor(images, self.in_type)
        feature_map_geom = self.canonicalization_network(x_geom)
        feature_map = feature_map_geom.tensor  # (B, out_channels * num_group_elements, H, W)
        B, _, H, W = feature_map.shape
        feature_map = feature_map.view(B, self.out_channels, self.num_group_elements, H, W)
        # Average over channel and spatial dims
        fibre_features = feature_map.mean(dim=(1, 3, 4))  # (B, num_group_elements)
        return fibre_features

    def get_canonicalized_images(self, images, fibre_features):
        """
        Canonicalize images based on the fibre features.
        
        The canonicalization applies the inverse of the detected transformation
        to bring the image to a canonical pose.
        
        Args:
            images: Input images of shape (B, C, H, W)
            fibre_features: Features for each group element, shape (B, num_group_elements)
            
        Returns:
            canonicalized: Canonicalized images (B, C, H, W)
            angles: Rotation angles applied (in degrees) (B,)
            reflect_indicator: Reflection indicator (B,) or (B, 1, 1, 1) - 1 if reflected, 0 otherwise
        """
        if self.group_type == "rotation":
            return self._canonicalize_rotation(images, fibre_features)
        elif self.group_type == "reflection":
            return self._canonicalize_reflection(images, fibre_features)
        elif self.group_type == "roto_reflection":
            return self._canonicalize_roto_reflection(images, fibre_features)
    
    def _canonicalize_rotation(self, images, fibre_features):
        """Canonicalization for rotation group (C_n)."""
        num_group_elements = fibre_features.shape[-1]
        
        fibre_features_one_hot = F.one_hot(
            torch.argmax(fibre_features, dim=-1),
            num_group_elements
        ).float()
        fibre_features_soft = F.softmax(fibre_features, dim=-1)
        
        # Reference angles for each rotation
        ref_angles = torch.linspace(0., 360., num_group_elements + 1, device=images.device)[:num_group_elements]
        
        # Straight-through estimator for angle selection
        angles = torch.sum((
            fibre_features_one_hot + fibre_features_soft - fibre_features_soft.detach()
        ) * ref_angles, dim=-1)
        
        # Apply inverse rotation to canonicalize
        canonicalized = K.geometry.rotate(images, -angles)
        
        # No reflection for rotation-only group
        reflect_indicator = torch.zeros(images.size(0), device=images.device)
        
        return canonicalized, angles, reflect_indicator
    
    def _canonicalize_reflection(self, images, fibre_features):
        """Canonicalization for reflection group (order 2)."""
        # fibre_features shape: (B, 2) - [identity, reflection]
        fibre_features_one_hot = F.one_hot(
            torch.argmax(fibre_features, dim=-1),
            2
        ).float()
        fibre_features_soft = F.softmax(fibre_features, dim=-1)
        
        # Reflection indicator: 0 for identity, 1 for reflection
        reflect_one_hot = torch.tensor([0., 1.], device=images.device)
        reflect_indicator = torch.sum((
            fibre_features_one_hot + fibre_features_soft - fibre_features_soft.detach()
        ) * reflect_one_hot, dim=-1)
        
        # Apply reflection based on axis
        images_reflected = self._apply_reflection(images)
        
        reflect_indicator_expanded = reflect_indicator[:, None, None, None]
        canonicalized = (1 - reflect_indicator_expanded) * images + reflect_indicator_expanded * images_reflected
        
        # No rotation for reflection-only group
        angles = torch.zeros(images.size(0), device=images.device)
        
        return canonicalized, angles, reflect_indicator
    
    def _canonicalize_roto_reflection(self, images, fibre_features):
        """
        Canonicalization for roto-reflection group (D_n).
        
        Group elements are ordered as: [e, r, r², ..., r^{n-1}, s, sr, sr², ..., sr^{n-1}]
        where:
        - r^k is rotation by k * 360/n degrees
        - s is reflection
        - sr^k means: first reflect, then rotate by k * 360/n degrees
        
        To canonicalize (bring to identity), we apply the inverse:
        - For r^k: apply r^{-k} (rotate by -k * angle)
        - For sr^k: apply (sr^k)^{-1} = r^{-k}s (first rotate by -k * angle, then reflect)
        """
        num_group_elements = fibre_features.shape[-1]
        num_rotations = num_group_elements // 2
        
        fibre_features_one_hot = F.one_hot(
            torch.argmax(fibre_features, dim=-1),
            num_group_elements
        ).float()
        fibre_features_soft = F.softmax(fibre_features, dim=-1)
        
        # Reference angles for each element
        # Elements 0 to n-1: rotations r^0, r^1, ..., r^{n-1}
        # Elements n to 2n-1: roto-reflections s, sr, sr^2, ..., sr^{n-1}
        ref_angles = torch.linspace(0., 360., num_rotations + 1, device=images.device)[:num_rotations]
        ref_angles = torch.cat([ref_angles, ref_angles], dim=0)  # Same angles for reflected elements
        
        # Straight-through estimator for angle selection
        angles = torch.sum((
            fibre_features_one_hot + fibre_features_soft - fibre_features_soft.detach()
        ) * ref_angles, dim=-1)
        
        # Reflection indicator: 0 for first half (pure rotations), 1 for second half (roto-reflections)
        reflect_one_hot = torch.cat([
            torch.zeros(num_rotations, device=images.device),
            torch.ones(num_rotations, device=images.device)
        ], dim=0)
        reflect_indicator = torch.sum((
            fibre_features_one_hot + fibre_features_soft - fibre_features_soft.detach()
        ) * reflect_one_hot, dim=-1)
        
        # To canonicalize sr^k: apply inverse (sr^k)^{-1} = r^{-k}s
        # Step 1: Apply inverse rotation r^{-k} first
        images_rotated = K.geometry.rotate(images, -angles)
        
        # Step 2: Then apply reflection s (if this was a roto-reflection element)
        images_reflected = self._apply_reflection(images_rotated)
        reflect_indicator_expanded = reflect_indicator[:, None, None, None]
        canonicalized = (1 - reflect_indicator_expanded) * images_rotated + reflect_indicator_expanded * images_reflected
        
        return canonicalized, angles, reflect_indicator
    
    def _apply_reflection(self, images):
        """
        Apply reflection based on the reflection axis.
        
        Args:
            images: Input images (B, C, H, W)
            
        Returns:
            Reflected images
        """
        # axis = 0: horizontal flip (reflect across y-axis, flip left-right)
        # axis = pi/2: vertical flip (reflect across x-axis, flip up-down)
        # For simplicity, we use hflip for axis close to pi/2 and vflip for axis close to 0
        # More general reflections would require rotation + flip + inverse rotation
        
        if abs(self.reflection_axis - math.pi / 2) < 0.1:
            # Vertical reflection axis -> horizontal flip
            return K.geometry.hflip(images)
        elif abs(self.reflection_axis) < 0.1:
            # Horizontal reflection axis -> vertical flip
            return K.geometry.vflip(images)
        else:
            # General reflection: rotate to align axis, flip, rotate back
            axis_degrees = math.degrees(self.reflection_axis)
            rotated = K.geometry.rotate(images, torch.tensor([axis_degrees - 90], device=images.device).expand(images.size(0)))
            flipped = K.geometry.hflip(rotated)
            return K.geometry.rotate(flipped, torch.tensor([-(axis_degrees - 90)], device=images.device).expand(images.size(0)))

    def apply_group_action(self, output, angles, reflect_indicator):
        """
        Apply the group action to transform output back to original image orientation.
        
        For segmentation, after canonicalizing the input and getting predictions in
        canonical pose, this applies the forward transformation to align the output
        with the original image orientation.
        
        For element r^k (pure rotation): applies rotation by k * angle
        For element sr^k (roto-reflection): applies reflection then rotation by k * angle
        
        Args:
            output: Output tensor (B, C, H, W) - e.g., segmentation logits in canonical pose
            angles: Rotation angles in degrees (from get_canonicalized_images)
            reflect_indicator: Reflection indicator (1 if roto-reflection, 0 if pure rotation)
            
        Returns:
            Transformed output aligned with original image orientation
        """
        B = output.shape[0]
        
        if self.group_type == "rotation":
            # Just rotate back
            reverted = K.geometry.rotate(output, angles)
        elif self.group_type == "reflection":
            # Just reflect back (reflection is its own inverse)
            if reflect_indicator is not None and reflect_indicator.shape[0] == B:
                reflect_indicator = reflect_indicator.view(B, 1, 1, 1)
                output_reflected = self._apply_reflection(output)
                reverted = (1 - reflect_indicator) * output + reflect_indicator * output_reflected
            else:
                reverted = output
        elif self.group_type == "roto_reflection":
            # To revert canonicalization, apply the forward transformation sr^k
            # sr^k means: first reflect s, then rotate r^k
            # Step 1: Apply reflection first (if this was a roto-reflection element)
            if reflect_indicator is not None and reflect_indicator.shape[0] == B:
                reflect_indicator = reflect_indicator.view(B, 1, 1, 1)
                output_reflected = self._apply_reflection(output)
                output = (1 - reflect_indicator) * output + reflect_indicator * output_reflected
            
            # Step 2: Then rotate
            reverted = K.geometry.rotate(output, angles)
        
        return reverted


# Backward compatibility alias
def create_canonicalization_network(
    n_rotations: int = 8,
    in_channels: int = 3,
    mid_channels=None,
    out_channels: int = 1,
    use_reflection: bool = True,
    group_type: str = None,
    reflection_axis: float = math.pi / 2
):
    """
    Factory function for creating CanonicalizationNetwork.
    
    Provides backward compatibility with the old use_reflection parameter.
    
    Args:
        n_rotations: Number of rotations
        in_channels: Input channels
        mid_channels: Middle layer channels
        out_channels: Output channels
        use_reflection: (Deprecated) If True and group_type is None, uses roto_reflection
        group_type: Explicit group type (overrides use_reflection)
        reflection_axis: Reflection axis in radians
        
    Returns:
        CanonicalizationNetwork instance
    """
    if group_type is None:
        # Backward compatibility
        group_type = "roto_reflection" if use_reflection else "rotation"
    
    return CanonicalizationNetwork(
        n_rotations=n_rotations,
        in_channels=in_channels,
        mid_channels=mid_channels,
        out_channels=out_channels,
        group_type=group_type,
        reflection_axis=reflection_axis
    )

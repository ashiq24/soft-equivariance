#!/usr/bin/env python3
"""
Standalone script for applying soft equivariance filters to DINOv2.

All softeq dependencies are inlined — only requires:
    pip install torch transformers pillow

Usage:
    python dinov2_soft_equivariance_standalone.py
"""

# ============================================================
# Standard / third-party imports
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from types import MethodType
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
from io import BytesIO

# ============================================================
# INLINED: softeq/utils/decompositions.py
# ============================================================

def svd_decomposition(matrix):
    with torch.no_grad():
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        if matrix.dim() != 2:
            raise ValueError("Input must be a 2D matrix")
        device = matrix.device
        dtype = matrix.dtype
        try:
            U, S, V = torch.linalg.svd(matrix.to('cpu'), full_matrices=False)
        except RuntimeError as e:
            try:
                U, S, V = torch.svd(matrix.to('cpu'))
            except RuntimeError:
                raise RuntimeError(f"SVD decomposition failed: {e}")
        U = U.to(device=device, dtype=dtype)
        S = S.to(device=device, dtype=dtype)
        V = V.to(device=device, dtype=dtype).transpose(0, 1)
        U = torch.flip(U, dims=[1])
        S = torch.flip(S, dims=[0])
        V = torch.flip(V, dims=[1])
    return U, S, V


# ============================================================
# INLINED: softeq/utils/group_utils.py
# ============================================================

def create_horizontal_flip_matrix(image_size):
    n = image_size
    flip_matrix = torch.zeros(n * n, n * n)
    for row in range(n):
        for col in range(n):
            orig_idx = row * n + col
            flipped_idx = row * n + (n - 1 - col)
            flip_matrix[flipped_idx, orig_idx] = 1.0
    return flip_matrix


def create_unit_reflection_action_matrix(image_size):
    channels, h, w = image_size
    n = h
    total_pixels = n * n
    total_elements = channels * total_pixels
    single = create_horizontal_flip_matrix(n)
    R = torch.zeros(total_elements, total_elements)
    for c in range(channels):
        s, e = c * total_pixels, (c + 1) * total_pixels
        R[s:e, s:e] = single
    return R


def create_unit_rotation_action_matrix(n_rotations, image_size, padding_mode='circular'):
    channels, h, w = image_size
    total_pixels = h * w
    total_elements = channels * total_pixels
    angle_rad = math.radians(-360.0 / n_rotations)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    theta = torch.tensor([[cos_a, sin_a, 0], [-sin_a, cos_a, 0]], dtype=torch.float32).unsqueeze(0)
    identity_batch = torch.eye(total_pixels).reshape(total_pixels, 1, h, w)
    if padding_mode == 'circular':
        pad = max(h, w) // 2
        identity_padded = F.pad(identity_batch, (pad, pad, pad, pad), mode='circular')
        theta_exp = theta.expand(total_pixels, -1, -1)
        grid = F.affine_grid(theta_exp, identity_padded.shape, align_corners=False)
        rotated_padded = F.grid_sample(identity_padded, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        rotated_batch = rotated_padded[:, :, pad:pad+h, pad:pad+w]
    else:
        grid = F.affine_grid(theta.expand(total_pixels, -1, -1), identity_batch.shape, align_corners=False)
        rotated_batch = F.grid_sample(identity_batch, grid, mode='bilinear', padding_mode=padding_mode, align_corners=False)
    single_channel_matrix = rotated_batch.reshape(total_pixels, total_pixels).T
    rotation_matrix = torch.zeros(total_elements, total_elements)
    for c in range(channels):
        s, e = c * total_pixels, (c + 1) * total_pixels
        rotation_matrix[s:e, s:e] = single_channel_matrix
    return rotation_matrix


# ============================================================
# INLINED: softeq/equi_utils/inv_projector.py
# ============================================================

def _create_smoothing_mask(values, size, soft_threshold, dtype, debug=False, hard=False, hard_mask=False):
    values = torch.abs(values)
    zero_indices = torch.where(torch.abs(values) < 1e-5)[0]
    if len(zero_indices) == 0:
        last_zero_eig_index = 0 if not hard else -1
    else:
        last_zero_eig_index = zero_indices[-1]
    last_basis_idx = last_zero_eig_index + int((len(values) - len(zero_indices) - 1) * soft_threshold)
    print("last_basis_idx and total basis:", last_basis_idx, len(values))
    assert (last_basis_idx >= 0 or hard) and last_basis_idx < size[-1]
    exp_values = torch.exp(-values / (soft_threshold + 1e-6)).to(dtype=dtype)
    if hard_mask:
        exp_values = torch.zeros_like(exp_values)
    mask = torch.diag(exp_values)
    if soft_threshold == 1.0:
        mask = torch.eye(size[-1], dtype=dtype, device=mask.device)
    elif last_basis_idx >= 0:
        mask[:last_basis_idx+1, :last_basis_idx+1] = torch.eye(last_basis_idx+1, dtype=dtype, device=mask.device)
    else:
        if hard and soft_threshold == 0.0:
            mask = torch.zeros_like(mask)
    return mask


class InvariantProjector(nn.Module):
    def __init__(self, U_w, U_x, values, softness=0.0, debug=False, hard=False, hard_mask=False, preserve_norm=False):
        super().__init__()
        if not (0.0 <= softness <= 1.0):
            raise ValueError("Softness parameter must be between 0.0 and 1.0")
        self.softness = softness
        self.preserve_norm = preserve_norm
        mask = _create_smoothing_mask(values, U_w.shape, softness, U_w.dtype, debug=debug, hard=hard, hard_mask=hard_mask)
        self.register_buffer('filter_w_T', (U_w @ mask @ U_w.transpose(0, 1)).t())
        self.register_buffer('filter_x', U_x @ mask @ U_x.transpose(0, 1))

    def forward(self, W):
        if self.softness == 1.0:
            return W
        in_channels = W.shape[1] // self.filter_w_T.shape[0]
        W = W.view(W.shape[0], in_channels, self.filter_w_T.shape[0])
        if self.preserve_norm:
            norm_before = torch.norm(W, dim=-1, keepdim=False)
        W = torch.einsum('fi,bci->bcf', self.filter_w_T, W)
        if self.preserve_norm:
            norm_after = torch.norm(W, dim=-1, keepdim=False)
            W = W * (norm_before / (norm_after + 1e-8)).unsqueeze(-1)
        return W.reshape(W.shape[0], -1)

    def smooth(self, x, data_last=False):
        if self.softness == 1.0:
            return x
        if x.dim() == 3:
            if data_last:
                in_channels = x.shape[2] // self.filter_x.shape[1]
                x = x.view(x.shape[0], x.shape[1], in_channels, self.filter_x.shape[1])
                if self.preserve_norm:
                    norm_before = torch.norm(x, dim=-1, keepdim=False)
                x = torch.einsum('fi,btci->btcf', self.filter_x, x)
                if self.preserve_norm:
                    x = x * (norm_before / (torch.norm(x, dim=-1, keepdim=False) + 1e-8)).unsqueeze(-1)
                x = x.reshape(x.shape[0], x.shape[1], -1)
            else:
                if self.preserve_norm:
                    norm_before = torch.norm(x, dim=1, keepdim=False)
                x = torch.einsum('fi,bic->bfc', self.filter_x, x)
                if self.preserve_norm:
                    x = x * (norm_before / (torch.norm(x, dim=1, keepdim=False) + 1e-8)).unsqueeze(1)
        elif x.dim() == 2:
            in_channels = x.shape[1] // self.filter_x.shape[1]
            x = x.view(x.shape[0], in_channels, self.filter_x.shape[1])
            if self.preserve_norm:
                norm_before = torch.norm(x, dim=-1, keepdim=False)
            x = torch.einsum('fi,bci->bcf', self.filter_x, x)
            if self.preserve_norm:
                x = x * (norm_before / (torch.norm(x, dim=-1, keepdim=False) + 1e-8)).unsqueeze(-1)
            x = x.reshape(x.shape[0], -1)
        elif x.dim() == 1:
            in_channels = x.shape[0] // self.filter_x.shape[1]
            x = x.view(in_channels, self.filter_x.shape[1])
            if self.preserve_norm:
                norm_before = torch.norm(x, dim=-1, keepdim=False)
            x = torch.einsum('fi,ci->cf', self.filter_x, x)
            if self.preserve_norm:
                x = x * (norm_before / (torch.norm(x, dim=-1, keepdim=False) + 1e-8)).unsqueeze(-1)
            x = x.reshape(-1)
        else:
            raise ValueError("Input tensor must be 1D, 2D, or 3D")
        return x


class MultiGeneratorInvariantProjector(nn.Module):
    def __init__(self, InvariantProjector_list):
        super().__init__()
        self.InvariantProjector_list = nn.ModuleList(InvariantProjector_list)

    def forward(self, W):
        for module in self.InvariantProjector_list:
            W = module(W)
        for module in self.InvariantProjector_list:
            W = module(W)
        return W

    def smooth(self, x):
        for module in self.InvariantProjector_list:
            x = module.smooth(x)
        for module in self.InvariantProjector_list:
            x = module.smooth(x)
        return x


# ============================================================
# INLINED: softeq/equi_utils/equi_constraints.py
# ============================================================

class DiscreteRotationConstraints:
    def __init__(self, n_rotations, input_size):
        self.n_rotations = n_rotations
        self.size = input_size
        assert self.size[-1] == self.size[-2]
        self.unit_group_action_matrix = create_unit_rotation_action_matrix(self.n_rotations, self.size)
        self.forward_difference_matrix = (self.unit_group_action_matrix -
            torch.eye(self.unit_group_action_matrix.shape[0])) / (2*math.pi/self.n_rotations)

    def create_invariant_basis(self):
        with torch.no_grad():
            U, S, _ = svd_decomposition(self.forward_difference_matrix)
            self.invariant_basis = U.to(self.forward_difference_matrix.device, dtype=self.forward_difference_matrix.dtype)
            self.invariant_scaling_values = S.to(self.forward_difference_matrix.device, dtype=self.forward_difference_matrix.dtype)


# ============================================================
# INLINED: softeq/equi_utils/equi_constraint_ref.py
# ============================================================

class DiscreteReflectionConstraints:
    def __init__(self, input_size):
        self.size = input_size
        assert self.size[-1] == self.size[-2]
        self.unit_group_action_matrix = create_unit_reflection_action_matrix(self.size)
        self.forward_difference_matrix = (self.unit_group_action_matrix -
            torch.eye(self.unit_group_action_matrix.shape[0])) / math.pi

    def create_invariant_basis(self):
        with torch.no_grad():
            U, S, _ = svd_decomposition(self.forward_difference_matrix)
            self.invariant_basis = U.to(self.forward_difference_matrix.device, dtype=self.forward_difference_matrix.dtype)
            self.invariant_scaling_values = S.to(self.forward_difference_matrix.device, dtype=self.forward_difference_matrix.dtype)


# ============================================================
# INLINED: filter functions
# ============================================================

def get_invariant_filter_rotation(n_rotations, input_size, soft_threshold=0.0,
                                   debug=False, hard=False, hard_mask=False, preserve_norm=False):
    rot_cons = DiscreteRotationConstraints(n_rotations, input_size)
    rot_cons.create_invariant_basis()
    basis = rot_cons.invariant_basis
    values = rot_cons.invariant_scaling_values
    return InvariantProjector(basis, basis, values, softness=soft_threshold, debug=debug,
                              hard=hard, hard_mask=hard_mask, preserve_norm=preserve_norm)


def get_invariant_filter_reflection(input_size, soft_threshold=0.0, debug=False,
                                     hard=False, hard_mask=False, preserve_norm=False):
    ref_cons = DiscreteReflectionConstraints(input_size=input_size)
    ref_cons.create_invariant_basis()
    basis = ref_cons.invariant_basis
    values = ref_cons.invariant_scaling_values
    return InvariantProjector(basis, basis, values, softness=soft_threshold, debug=debug,
                              hard=hard, hard_mask=hard_mask, preserve_norm=preserve_norm)


def get_invariant_filter_roto_reflection(n_rotations, input_size, soft_threshold=0.0,
                                          debug=False, hard=False, hard_mask=False,
                                          preserve_norm=False, joint_decomposition=False):
    ref_cons = DiscreteReflectionConstraints(input_size=input_size)
    rot_cons = DiscreteRotationConstraints(n_rotations, input_size)
    if not joint_decomposition:
        rot_filter = get_invariant_filter_rotation(n_rotations, input_size, soft_threshold,
                                                   debug, hard, hard_mask, preserve_norm)
        ref_filter = get_invariant_filter_reflection(input_size, soft_threshold,
                                                     debug, hard, hard_mask, preserve_norm)
        return MultiGeneratorInvariantProjector([ref_filter, rot_filter])
    D_ref = ref_cons.forward_difference_matrix
    D_rot = rot_cons.forward_difference_matrix
    assert D_ref.shape == D_rot.shape
    U_l, S_l, _ = svd_decomposition(torch.cat([D_ref, D_rot], dim=1))
    _, S_r, V_r = svd_decomposition(torch.cat([D_ref, D_rot], dim=0))
    min_len = min(len(S_l), len(S_r))
    values = S_l[:min_len] + S_r[:min_len]
    return InvariantProjector(U_l, V_r, values, softness=soft_threshold, debug=debug,
                              hard=hard, hard_mask=hard_mask, preserve_norm=preserve_norm)


def get_invariant_filter(group_type, n_rotations, input_size, soft_threshold,
                         debug=False, hard_mask=False, preserve_norm=False, joint_decomposition=True):
    if group_type == "roto_reflection":
        return get_invariant_filter_roto_reflection(
            n_rotations=n_rotations, input_size=input_size, soft_threshold=soft_threshold,
            debug=debug, hard_mask=hard_mask, preserve_norm=preserve_norm,
            joint_decomposition=joint_decomposition
        )
    else:
        return get_invariant_filter_rotation(
            n_rotations=n_rotations, input_size=input_size, soft_threshold=soft_threshold,
            debug=debug, hard_mask=hard_mask, preserve_norm=preserve_norm
        )


# ============================================================
# INLINED: softeq/layers/fconv2d.py
# ============================================================

class FilteredConv2d(nn.Module):
    def __init__(self, original_layer: nn.Conv2d, filter: nn.Module):
        super().__init__()
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        self.weight = nn.Parameter(original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        self.filter = filter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = self.weight.shape
        weight_2d = self.weight.view(original_shape[0], -1)
        filtered_weight = self.filter(weight_2d).view(original_shape)
        return F.conv2d(x, filtered_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# ============================================================
# INLINED: models/filtered_layers_dinov2.py
# ============================================================

def custom_dinov2embeddings_forward(self, pixel_values, bool_masked_pos=None):
    batch_size, num_channels, height, width = pixel_values.shape
    target_dtype = self.patch_embeddings.projection.weight.dtype
    embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

    if bool_masked_pos is not None and self.use_mask_token:
        embeddings = torch.where(
            bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
        )

    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)

    self.position_embeddings.data[:, 1:, :] = self.filter_pos.smooth(
        self.position_embeddings.data[:, 1:, :]
    )

    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
    embeddings = self.dropout(embeddings)
    return embeddings


def custom_dinov2_with_registers_embeddings_forward(self, pixel_values, bool_masked_pos=None):
    batch_size, num_channels, height, width = pixel_values.shape
    target_dtype = self.patch_embeddings.projection.weight.dtype
    embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

    if bool_masked_pos is not None:
        embeddings = torch.where(
            bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
        )

    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)

    if hasattr(self, 'position_embeddings'):
        self.position_embeddings.data[:, 1:, :] = self.filter_pos.smooth(
            self.position_embeddings.data[:, 1:, :]
        )

    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
    embeddings = torch.cat(
        (embeddings[:, :1], self.register_tokens.expand(embeddings.shape[0], -1, -1), embeddings[:, 1:]), dim=1
    )
    embeddings = self.dropout(embeddings)
    return embeddings


def monkeypatch_dinov2embeddings(dinov2embeddings, filter_configs):
    is_register_variant = 'WithRegisters' in dinov2embeddings.__class__.__name__
    group_type = filter_configs.get("group_type", "rotation")

    original_conv = dinov2embeddings.patch_embeddings.projection
    kernel_size = original_conv.kernel_size[0]
    assert kernel_size == original_conv.kernel_size[1], "Kernel size is not square"

    filter = get_invariant_filter(
        group_type=group_type,
        n_rotations=filter_configs["n_rotations"],
        input_size=(1, kernel_size, kernel_size),
        soft_threshold=filter_configs["soft_thresholding"],
        debug=False,
        hard_mask=filter_configs.get("hard_mask", False),
        preserve_norm=filter_configs.get("preserve_norm", False),
        joint_decomposition=filter_configs.get("joint_decomposition", True)
    )

    dinov2embeddings.patch_embeddings.projection = FilteredConv2d(
        dinov2embeddings.patch_embeddings.projection, filter
    )

    if hasattr(dinov2embeddings, 'position_embeddings') and dinov2embeddings.position_embeddings is not None:
        num_patches = dinov2embeddings.patch_embeddings.num_patches
        if math.sqrt(num_patches).is_integer():
            len_pos = int(math.sqrt(num_patches))
            filter_pos = get_invariant_filter(
                group_type=group_type,
                n_rotations=filter_configs["n_rotations"],
                input_size=(1, len_pos, len_pos),
                soft_threshold=filter_configs["soft_thresholding_pos"],
                debug=False,
                hard_mask=filter_configs.get("hard_mask", False),
                preserve_norm=filter_configs.get("preserve_norm", False),
                joint_decomposition=filter_configs.get("joint_decomposition", True)
            )
            dinov2embeddings.filter_pos = filter_pos
        else:
            print(f"Warning: DINOv2 num_patches ({num_patches}) is not a perfect square. Skipping positional embedding filtering.")
    else:
        print("DINOv2 model does not have traditional positional embeddings.")

    if is_register_variant:
        dinov2embeddings.forward = MethodType(custom_dinov2_with_registers_embeddings_forward, dinov2embeddings)
    else:
        dinov2embeddings.forward = MethodType(custom_dinov2embeddings_forward, dinov2embeddings)
        print("Applied standard DINOv2 forward method")


# ============================================================
# Original dinov2_soft_equivariance.py main logic (local imports removed)
# ============================================================

def main():
    print("=" * 60)
    print("Soft Equivariance for DINOv2")
    print("=" * 60)

    print("\n1. Loading pretrained DINOv2 model...")
    model_name = "facebook/dinov2-base"
    model = AutoModel.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)

    print(f"   - Model: {model_name}")
    print(f"   - Model type: {model.__class__.__name__}")

    is_register = 'WithRegisters' in model.embeddings.__class__.__name__
    print(f"   - Has register tokens: {is_register}")

    filter_config = {
        "n_rotations": 4,
        "soft_thresholding": 0.2,
        "soft_thresholding_pos": 0.2,
        "group_type": "rotation",
        "hard_mask": True,
        "preserve_norm": False,
        "joint_decomposition": True,
    }

    print(f"\n2. Applying soft equivariance filters...")
    print(f"   - n_rotations: {filter_config['n_rotations']}")
    print(f"   - soft_thresholding: {filter_config['soft_thresholding']}")

    monkeypatch_dinov2embeddings(model.embeddings, filter_config)
    print("   - Applied filters to patch embeddings and position embeddings")
    print("\n   - This model is not ready to be trained or be used as a part of VLM or VLA for finetuning")
    
    print("\n3. Loading sample image...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        print(f"   - Downloaded image: {image.size}")
        # make the image square by cropping the center
        min_dim = min(image.size)
        left = (image.width - min_dim) // 2
        top = (image.height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        image = image.crop((left, top, right, bottom))
        print(f"   - Cropped to square: {image.size}")
    except Exception as e:
        print(f"   - Could not download image: {e}")
        print("   - Using random tensor instead")
        image = None

    print("\n4. Running inference...")
    model.eval()
    with torch.no_grad():
        if image is not None:
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
        else:
            inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
            outputs = model(**inputs)

        features = outputs.last_hidden_state
        print(f"   - Feature shape: {features.shape}")
        cls_features = features[:, 0]
        print(f"   - CLS feature shape: {cls_features.shape}")
        print(f"   - CLS feature norm: {cls_features.norm().item():.4f}")

    print("\n5. Testing rotation consistency (feature similarity)...")
    if image is not None:
        features_list = []
        for angle in [0, 90, 180, 270]:
            rotated = image.rotate(-angle)
            inputs = processor(images=rotated, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                cls_feat = outputs.last_hidden_state[:, 0]
                features_list.append((angle, cls_feat))

        print("   Cosine similarity matrix:")
        print("          ", end="")
        for angle, _ in features_list:
            print(f"  {angle:>5}°", end="")
        print()

        for i, (angle_i, feat_i) in enumerate(features_list):
            print(f"   {angle_i:>5}°", end="")
            for j, (angle_j, feat_j) in enumerate(features_list):
                sim = torch.nn.functional.cosine_similarity(feat_i, feat_j).item()
                print(f"  {sim:>6.3f}", end="")
            print()

        avg_sim = 0
        count = 0
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                avg_sim += torch.nn.functional.cosine_similarity(
                    features_list[i][1], features_list[j][1]
                ).item()
                count += 1
        avg_sim /= count

        print(f"\n   Average pairwise similarity: {avg_sim:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

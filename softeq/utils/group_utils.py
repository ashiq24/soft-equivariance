import torch
import torchvision.transforms.functional as TF 
import math
import torch.nn.functional as F


def create_circular_shift_matrix(n):
    """Creates a circular shift matrix of size n x n."""
    matrix = torch.eye(n)
    shifted_matrix = torch.roll(matrix, shifts=1, dims=0)
    return shifted_matrix

def create_circular_difference_matrix(n):
    """Creates a circular difference matrix of size n x n."""
    matrix = torch.eye(n)
    shifted_matrix = torch.roll(matrix, shifts=1, dims=0)
    diff_matrix = matrix - shifted_matrix
    return diff_matrix

###
# Horizontal Flip for flattened image as matrix multiplication
###

def create_horizontal_flip_matrix(image_size: int) -> torch.Tensor:
    """Creates a horizontal flip matrix for an image of size image_size x image_size."""
    # The image is assumed to be flattened row-wise (C order)
    n = image_size
    flip_matrix = torch.zeros(n * n, n * n)
    for row in range(n):
        for col in range(n):
            orig_idx = row * n + col
            flipped_idx = row * n + (n - 1 - col)
            flip_matrix[flipped_idx, orig_idx] = 1.0
    return flip_matrix

def create_horizontal_flip_difference_matrix(image_size: int) -> torch.Tensor:
    """Creates a horizontal flip difference matrix for an image of size image_size x image_size."""
    flip_matrix = create_horizontal_flip_matrix(image_size)
    identity_matrix = torch.eye(image_size * image_size)
    diff_matrix = identity_matrix - flip_matrix
    return diff_matrix

def create_unit_reflection_action_matrix(image_size):
    """
    Create a unit group action matrix for horizontal reflection (reflection along x-axis).
    
    Args:
        image_size (tuple): Image size as (channels, height, width)
    
    Returns:
        torch.Tensor: Reflection matrix of shape (total_elements, total_elements)
    """
    channels, h, w = image_size
    
    # Only square images are supported (consistent with rotation constraints)
    assert h == w, "Only square images are supported"
    n = h
    
    total_pixels = n * n
    total_elements = channels * total_pixels
    
    # Create single-channel reflection matrix
    single_channel_reflection = create_horizontal_flip_matrix(n)
    
    # Create block diagonal matrix for multiple channels
    # Each channel is reflected independently
    reflection_matrix = torch.zeros(total_elements, total_elements)
    for c in range(channels):
        start_idx = c * total_pixels
        end_idx = (c + 1) * total_pixels
        reflection_matrix[start_idx:end_idx, start_idx:end_idx] = single_channel_reflection
    
    return reflection_matrix



def create_unit_rotation_action_matrix(n_rotations, image_size, padding_mode='circular'):
    """
    Create a flattened rotation action matrix for an image tensor.

    The matrix is built by rotating basis images with ``affine_grid`` and
    ``grid_sample``. When ``padding_mode='circular'``, the basis images are
    circularly padded before sampling and then cropped back to the original
    size. For any other padding mode, the value is passed directly to
    ``grid_sample``.

    Args:
        n_rotations: Number of rotations.
        image_size: Image size as (channels, height, width).
        padding_mode: One of ``'circular'``, ``'zeros'``, ``'border'``, or
            ``'reflection'``.
    Returns:
        torch.Tensor: Rotation matrix of shape (total_elements, total_elements).
    """
    channels, h, w = image_size
    total_pixels = h * w
    total_elements = channels * total_pixels
    angle_degrees = 360.0 / n_rotations

    angle_rad = math.radians(-angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    theta = torch.tensor([[cos_a, sin_a, 0],
                          [-sin_a, cos_a, 0]], dtype=torch.float32).unsqueeze(0)

    identity_batch = torch.eye(total_pixels).reshape(total_pixels, 1, h, w)

    if padding_mode == 'circular':
        # Manually pad circularly then sample with zeros
        pad = max(h, w) // 2
        identity_padded = F.pad(identity_batch, (pad, pad, pad, pad), mode='circular')

        # Adjust grid for padded size
        theta_exp = theta.expand(total_pixels, -1, -1)
        grid = F.affine_grid(theta_exp,
                             identity_padded.shape,
                             align_corners=False)

        rotated_padded = F.grid_sample(identity_padded,
                                       grid,
                                       mode='bilinear',
                                       padding_mode='zeros',
                                       align_corners=False)

        # Crop back to original size
        rotated_batch = rotated_padded[:, :, pad:pad+h, pad:pad+w]

    else:
        # 'reflection' or 'border' - directly supported
        grid = F.affine_grid(theta.expand(total_pixels, -1, -1),
                             identity_batch.shape,
                             align_corners=False)

        rotated_batch = F.grid_sample(identity_batch,
                                      grid,
                                      mode='bilinear',
                                      padding_mode=padding_mode,
                                      align_corners=False)

    single_channel_matrix = rotated_batch.reshape(total_pixels, total_pixels).T


    rotation_matrix = torch.zeros(total_elements, total_elements)
    for c in range(channels):
        start_idx = c * total_pixels
        end_idx = (c + 1) * total_pixels
        rotation_matrix[start_idx:end_idx, start_idx:end_idx] = single_channel_matrix

    return rotation_matrix




import torch
import numpy as np
from .block_utils import extract_blocks, _create_mask_from_blocks

def red_warn(srt):
    print("\033[91m {}\033[00m".format(srt))
    
def get_loss(signal, weight, circular_shift_matrix, signal_size):
    inv_error = []
    result = torch.dot(signal.squeeze(0), weight.squeeze(0))
    rotating_signal = signal.clone()
    for i in range(signal_size):
        rotating_signal = torch.matmul(rotating_signal, circular_shift_matrix)
        inv_error.append((result - torch.dot(weight.squeeze(0), rotating_signal.squeeze(0)))**2)
    return inv_error



def create_patterned_array(size, channels=1, pattern='checkerboard'):
    """
    Create a 2D array (size x size) with 1 or 3 channels and a given pattern.
    Supported patterns: 'checkerboard', 'sine_cosine', 'rectangle', 'triangle',
    'gradient_noise', 'diagonal_stripes', 'random_rectangles'
    Returns: torch.Tensor of shape (channels, size, size)
    """
    arr = torch.zeros(channels, size, size)
    for c in range(channels):
        if pattern == 'checkerboard':
            arr[c] = torch.tensor((np.indices((size, size)).sum(axis=0) + c) % 2, dtype=torch.float32)
        elif pattern == 'sine_cosine':
            x = np.arange(size)
            y = np.arange(size)
            X, Y = np.meshgrid(x, y)
            arr[c] = torch.tensor(np.sin((X + c*2) / 4.0) * np.cos((Y + c*2) / 4.0), dtype=torch.float32)
        elif pattern == 'rectangle':
            arr[c][:, :] = 0
            arr[c][size//4:size*3//4, size//4:size*3//4] = 1.0
        elif pattern == 'triangle':
            for i in range(size):
                arr[c][i, :i+1] = 1.0
        elif pattern == 'gradient_noise':
            x = torch.linspace(0, 1, size)
            y = torch.linspace(0, 1, size)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            arr[c] = grid_x + 0.3 * torch.rand(size, size)
        elif pattern == 'diagonal_stripes':
            img = torch.zeros(size, size)
            for i in range(size):
                thickness = torch.randint(1, 4, (1,)).item()
                img[i, max(0, i-thickness):min(size, i+thickness)] = 1.0
            img += 0.2 * torch.rand(size, size)
            arr[c] = img
        elif pattern == 'random_rectangles':
            img = torch.zeros(size, size)
            for _ in range(5):
                x0 = torch.randint(0, size-5, (1,)).item()
                y0 = torch.randint(0, size-5, (1,)).item()
                w = torch.randint(2, 8, (1,)).item()
                h = torch.randint(2, 8, (1,)).item()
                img[x0:x0+w, y0:y0+h] = torch.rand(1).item()
            arr[c] = img
        else:
            arr[c] = torch.rand(size, size)
    return arr

def animate_weights(weights_list, error_list, filename, interval):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax1.imshow(weights_list[frame], cmap='viridis')
        ax1.set_title(f'Weights at step {frame}')
        ax2.plot(error_list[:frame+1])
        ax2.set_title('Error')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Error')

    ani = FuncAnimation(fig, update, frames=len(weights_list), repeat=False, interval=interval)
    ani.save(filename, writer='ffmpeg')
    plt.close(fig)
    print(f'Animation saved as {filename}')
    
    
### block multiplication
import torch

def _to_blocks(M, c_out, c_in, o, inn):
    # M: (c_out*o, c_in*inn)  -> blocks shape (c_out, c_in, o, inn)
    # assumes row-major contiguous storage
    return M.view(c_out, o, c_in, inn).permute(0, 2, 1, 3)

def _from_blocks(blocks):
    # blocks: (c_out, c_in, o, inn) -> (c_out*o, c_in*inn)
    c_out, c_in, o, inn = blocks.shape
    return blocks.permute(0, 2, 1, 3).contiguous().view(c_out * o, c_in * inn)


def apply_filter_blockwise(weights, filter, c_out, c_in, o, inn):
    """
    Apply a filter to the weights using block-wise left multiplication.
    weights: (out_features, in_features) or (out_features, in_features, c_out, c_in)
    filter: list of 3 matrices [F, G, H]
        F: (o, o) - left multiplication
        G: (c_out, c_in) - elementwise multiplication
        H: (inn, inn) - right multiplication
    c_out: number of output channels
    c_in: number of input channels
    o: output feature dimension per channel
    inn: input feature dimension per channel
    returns: filtered weights of the same shape as input weights
    """
    weight_blocks = _to_blocks(weights, c_out, c_in, o, inn)
    # Project along the output-feature basis.
    filtered_blocks = torch.einsum('ao,ijop->ijap', filter[0].transpose(0, 1), weight_blocks)
    # Project along the input-feature basis.
    filtered_blocks = torch.einsum('ijop,pb->ijob', filtered_blocks, filter[2])
    # Apply the channel-wise sparsity mask.
    filtered_blocks = filtered_blocks * filter[1].unsqueeze(0).unsqueeze(0)
    
    # Map back to the output basis.
    filtered_blocks = torch.einsum('ao,ijop->ijap', filter[0], filtered_blocks)
    # Map back to the input basis.
    filtered_blocks = torch.einsum('ijop,pb->ijob', filtered_blocks, filter[2].transpose(0, 1))
    # rearrange back to original shape
    filtered_weights = _from_blocks(filtered_blocks)
    return filtered_weights

# ### 
# Method for explicit equivariance projection
# ###

def exact_equivarinace_projection(W, U_y, U_x, sigma_y_blocks, sigma_x_blocks):
    """
    Projects an arbitrary batched weight tensor W onto the closest exactly
    equivariant matrix for each item in the batch. This operation is fully
    differentiable.

    Args:
        W (torch.Tensor): The arbitrary batched weight tensor of shape (..., m, n).
        U_y (torch.Tensor): The output basis matrix (m x m).
        U_x (torch.Tensor): The input basis matrix (n x n).
        sigma_y_blocks (list): List of blocks for the output representation.
        sigma_x_blocks (list): List of blocks for the input representation.

    Returns:
        torch.Tensor: The projected, exactly equivariant weight tensor of shape (..., m, n).
    """
    # Transform to the Schur basis; torch.matmul handles batching automatically.
    W = _to_blocks(W, W.shape[0] // U_y.shape[0], W.shape[1] // U_x.shape[0], U_y.shape[0], U_x.shape[0])
    
    W_prime = U_y.T @ W @ U_x

    # Enforce sparsity and block structure across the batch.
    W_prime_proj = torch.zeros_like(W_prime)
    
    y_cursor = 0
    for T_I in sigma_y_blocks:
        t_dim = T_I.shape[0]
        x_cursor = 0
        for S_J in sigma_x_blocks:
            s_dim = S_J.shape[0]
            
            # Only matching block sizes and exact block values are projected.
            if t_dim == s_dim and torch.equal(T_I, S_J):
                # Use ellipsis '...' to slice across all batch dimensions
                block = W_prime[..., y_cursor:y_cursor+t_dim, x_cursor:x_cursor+s_dim]
                
                if t_dim == 1:
                    # 1x1 blocks are unconstrained scalars, just copy them over
                    W_prime_proj[..., y_cursor:y_cursor+t_dim, x_cursor:x_cursor+s_dim] = block
                elif t_dim == 2:
                    # For 2x2 blocks, project onto the canonical equivariant subspace.
                    alpha = (block[..., 0, 0] + block[..., 1, 1]) / 2.0
                    beta = (block[..., 0, 1] - block[..., 1, 0]) / 2.0
                    
                    # Reconstruct the ideal equivariant blocks for the whole batch
                    # alpha and beta have shape (...)
                    row1 = torch.stack([alpha, beta], dim=-1)   # Shape (..., 2)
                    row2 = torch.stack([-beta, alpha], dim=-1)  # Shape (..., 2)
                    proj_block = torch.stack([row1, row2], dim=-2) # Shape (..., 2, 2)
                    
                    W_prime_proj[..., y_cursor:y_cursor+2, x_cursor:x_cursor+2] = proj_block
            
            x_cursor += s_dim
        y_cursor += t_dim
    # Transform back to the standard basis.
    W_proj = U_y @ W_prime_proj @ U_x.T
    
    W_proj = _from_blocks(W_proj)
    
    return W_proj



####
# Fast batched projection
####

def project_fast_batched(W, U_y, U_x, P_sparse, P_struct):
    """
    A fast, vectorized projection using pre-computed projection matrices.
    """
    original_shape = W.shape
    m, n = original_shape[-2], original_shape[-1]
    
    # Transform to Schur basis
    W_prime = U_y.T @ W @ U_x
    
    # Flatten the last two dimensions for projection
    W_prime_flat = W_prime.reshape(-1, m * n) # Shape: (batch_size, m*n)
    
    # Apply projectors: P_struct is applied first, then P_sparse
    # (w @ P_sparse.T) @ P_struct.T is equivalent to P_struct @ P_sparse @ w
    W_proj_flat = W_prime_flat @ P_sparse.T @ P_struct.T

    # Reshape back to (..., m, n)
    W_prime_proj = W_proj_flat.reshape(original_shape)
    
    # Transform back to standard basis
    W_proj = U_y @ W_prime_proj @ U_x.T
    
    return W_proj
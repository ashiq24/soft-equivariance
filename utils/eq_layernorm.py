# Equivariant layer norm for 2D or 3D vectors.
import torch
import torch.nn as nn


class EQLayerNorm(nn.Module):
    """
    Normalize each vector independently to unit length.

    The input is interpreted as ``(batch, seq_len, n_vectors * dim)`` and is
    reshaped into ``(batch, seq_len, n_vectors, dim)``. Each vector is divided by
    its Euclidean norm, with ``eps`` added for numerical stability.

    The extra constructor arguments mirror ``torch.nn.LayerNorm`` for API
    compatibility, but only ``normalized_shape``, ``dim``, and ``eps`` are used.
    """
    def __init__(self, normalized_shape, dim=2, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None):
        super(EQLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.normalized_shape = normalized_shape
        assert normalized_shape[0] % dim == 0, "normalized_shape must be divisible by vector dimension"
        
    
    def forward(self, x):
        """
        Apply per-vector normalization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_vectors * dim)
        
        Returns:
            Normalized tensor of same shape as input
        """
        batch_size, seq_len, total_dim = x.shape
        assert total_dim % self.dim == 0, f"Input dimension {total_dim} must be divisible by vector dimension {self.dim}"
        n_vectors = total_dim // self.dim
        
        # Reshape to (batch, seq_len, n_vectors, dim)
        x_reshaped = x.view(batch_size, seq_len, n_vectors, self.dim)
        
        # Compute norms for each vector: (batch, seq_len, n_vectors)
        norms = torch.linalg.vector_norm(x_reshaped, dim=-1, keepdim=False)

        # Avoid division by zero when normalizing vectors
        safe_norms = norms.unsqueeze(-1) + self.eps  # (batch, seq_len, n_vectors, 1)
        
        # Normalize each vector to unit length, then scale by normalized norms
        # (batch, seq_len, n_vectors, dim)
        unit_vectors = x_reshaped / safe_norms
        
        # Reshape back to (batch, seq_len, n_vectors * dim)
        output = unit_vectors.view(batch_size, seq_len, total_dim)
        
        return output

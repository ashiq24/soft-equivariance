import torch.nn as nn
import torch
import numpy as np


class IdentityActivation(nn.Module):
    """Identity activation function as a proper nn.Module."""
    def forward(self, x):
        return x


class SmoothStepActivation(nn.Module):
    """Smooth step activation function as a proper nn.Module."""
    def forward(self, x):
        return torch.log(0.5 * torch.exp(x) + 0.5)


class EQNonLin(nn.Module):
    """
    Apply a norm-based nonlinearity to each vector independently.

    The input is reshaped from ``(batch, n_vectors * dim)`` or
    ``(batch, seq_len, n_vectors * dim)`` into explicit vector blocks. The norm
    of each vector is passed through the selected activation, then used to scale
    the original vectors.

    Args:
        dim: Dimensionality of each vector block.
        nonlinearity: Activation used on the vector norms.
        per_channel_bias: If True, use a separate bias per vector channel when
            ``layer_norm`` is False.
        n_channels: Number of vector channels when ``per_channel_bias`` is True.
        layer_norm: Retained for API compatibility; when True, forces a scalar
            bias.
        vec_dim: Unused compatibility argument.
    """
    def __init__(self, dim=2, nonlinearity='sigmoid', per_channel_bias=False, n_channels=1, layer_norm = False, vec_dim=None):
        super(EQNonLin, self).__init__()
        self.dim = dim
        self.nonlinearity = nonlinearity
        self.per_channel_bias = per_channel_bias
        self.n_channels = n_channels
        
        if nonlinearity == 'relu':
            self.act = nn.ReLU()
        elif nonlinearity == 'gelu':
            self.act = nn.GELU()
        elif nonlinearity == 'tanh':
            self.act = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.act = nn.Sigmoid()
        elif nonlinearity == 'identity':
            self.act = IdentityActivation()  # Identity function (no activation)
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        if per_channel_bias and not layer_norm:
            self.bias = nn.Parameter(torch.zeros(n_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Handle 2D input: (batch, n_vectors * dim)
        if x.dim() == 2:
            batch_size, total_dim = x.shape
            assert total_dim % self.dim == 0, "Input dimension mismatch"
            n_vectors = total_dim // self.dim
            x_reshaped = x.view(batch_size, n_vectors, self.dim)
            norms = torch.norm(x_reshaped, dim=-1, keepdim=True)  # (batch, n_vectors, 1)
            # Bias broadcasting
           
            if self.bias.numel() == 1:
                bias_tensor = self.bias.view(1, 1, 1)
            elif self.bias.numel() == n_vectors:
                bias_tensor = self.bias.view(1, n_vectors, 1)
            else:
                raise ValueError(
                    f"Bias channels ({self.bias.numel()}) do not match number of vectors ({n_vectors}). "
                    f"Set n_channels={n_vectors} or disable per_channel_bias."
                )
            norms = norms + bias_tensor
            activated_norms = self.act(norms)
            scaled_vectors = x_reshaped * activated_norms
            output = scaled_vectors.view(batch_size, total_dim)
            return output

        # Handle 3D input: (batch, seq_len, n_vectors * dim)
        if x.dim() == 3:
            batch_size, seq_len, total_dim = x.shape
            assert total_dim % self.dim == 0, "Input dimension mismatch"
            n_vectors = total_dim // self.dim
            x_reshaped = x.view(batch_size, seq_len, n_vectors, self.dim)
            norms = torch.norm(x_reshaped, dim=-1, keepdim=True)  # (batch, seq_len, n_vectors, 1)

            # Bias broadcasting
            if self.bias.numel() == 1:
                bias_tensor = self.bias.view(1, 1, 1, 1)
            elif self.bias.numel() == n_vectors:
                bias_tensor = self.bias.view(1, 1, n_vectors, 1)
            else:
                raise ValueError(
                    f"Bias channels ({self.bias.numel()}) do not match number of vectors ({n_vectors}). "
                    f"Set n_channels={n_vectors} or disable per_channel_bias."
                )

            norms = norms + bias_tensor
            activated_norms = self.act(norms)
            scaled_vectors = x_reshaped * activated_norms
            output = scaled_vectors.view(batch_size, seq_len, total_dim)
            return output

        raise ValueError("Unsupported input shape. Expected 2D (batch, n*dim) or 3D (batch, seq_len, n*dim)")


class EqComNonLin(nn.Module):
    """
    Magnitude-gated equivariant nonlinearity with a residual connection.
    
    Features:
    - Magnitude-only processing (perfectly equivariant)
    - Direction preserved unchanged
    - Optional cross-vector magnitude interactions
    
    Mathematical form before the residual is approximately
    ``g(||v||) * v``, with ``g`` learned from rotation-invariant magnitude
    features. The final output is ``v + g(||v||) * v`` up to the epsilon clamp
    used for numerical stability.
    
    Args:
        dim: Dimension of each vector
        magnitude_hidden: Hidden dimension for magnitude processing
        nonlinearity: Base activation function
        eps: Small constant for numerical stability
        use_cross_magnitude: Whether to use cross-vector magnitude interactions
    """
    def __init__(self, dim=2, magnitude_hidden=32, nonlinearity='gelu', eps=1e-6, 
                 use_cross_magnitude=False):
        super(EqComNonLin, self).__init__()
        self.dim = dim
        self.magnitude_hidden = magnitude_hidden
        self.eps = eps
        self.use_cross_magnitude = use_cross_magnitude
        
        # Base activation
        if nonlinearity == 'relu':
            self.act = nn.ReLU()
        elif nonlinearity == 'gelu':
            self.act = nn.GELU()
        elif nonlinearity == 'tanh':
            self.act = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.act = nn.Sigmoid()
        elif nonlinearity == 'swish':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        
        # Magnitude processing network - processes only rotation-invariant quantities
        if use_cross_magnitude:
            # Input: [current_magnitude, mean_magnitude, max_magnitude]
            # These are all rotation-invariant features
            self.magnitude_net = nn.Sequential(
                nn.Linear(3, magnitude_hidden),
                self.act,
                nn.Linear(magnitude_hidden, magnitude_hidden // 2),
                self.act,
                nn.Linear(magnitude_hidden // 2, 1),
                nn.Sigmoid()  # Gate output between 0 and 1
            )
        else:
            # Simple magnitude processing
            self.magnitude_net = nn.Sequential(
                nn.Linear(1, magnitude_hidden),
                self.act,
                nn.Linear(magnitude_hidden, 1),
                nn.Sigmoid()  # Gate output between 0 and 1
            )
        
        # Learnable magnitude bias
        self.magnitude_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Handle 2D input: (batch, n_vectors * dim)
        if x.dim() == 2:
            batch_size, total_dim = x.shape
            assert total_dim % self.dim == 0, "Input dimension mismatch"
            n_vectors = total_dim // self.dim
            x_reshaped = x.view(batch_size, n_vectors, self.dim)
            
            # Decompose into magnitude and direction
            norms = torch.norm(x_reshaped, dim=-1, keepdim=True)  # (batch, n_vectors, 1)
            safe_norms = torch.clamp(norms, min=self.eps)
            directions = x_reshaped / safe_norms  # (batch, n_vectors, dim)
            
            # Process magnitudes using only rotation-invariant features
            if self.use_cross_magnitude:
                # Cross-vector magnitude statistics (all rotation-invariant)
                mean_magnitude = torch.mean(norms, dim=1, keepdim=True)  # (batch, 1, 1)
                max_magnitude = torch.max(norms, dim=1, keepdim=True)[0]  # (batch, 1, 1)
                
                # Broadcast to all vectors
                mean_magnitude = mean_magnitude.expand_as(norms)
                max_magnitude = max_magnitude.expand_as(norms)
                
                # Concatenate invariant features
                magnitude_features = torch.cat([
                    norms + self.magnitude_bias,
                    mean_magnitude,
                    max_magnitude
                ], dim=-1)  # (batch, n_vectors, 3)
                
                magnitude_gates = self.magnitude_net(magnitude_features)  # (batch, n_vectors, 1)
            else:
                # Simple magnitude processing
                magnitude_gates = self.magnitude_net(norms + self.magnitude_bias)  # (batch, n_vectors, 1)
            
            # Magnitude gating keeps the direction unchanged up to the residual path.
            output_vectors = magnitude_gates * safe_norms * directions
            
            # Residual connection for training stability
            output = x_reshaped + output_vectors
            return output.view(batch_size, total_dim)

        # Handle 3D input: (batch, seq_len, n_vectors * dim)  
        elif x.dim() == 3:
            batch_size, seq_len, total_dim = x.shape
            assert total_dim % self.dim == 0, "Input dimension mismatch"
            n_vectors = total_dim // self.dim
            x_reshaped = x.view(batch_size, seq_len, n_vectors, self.dim)
            
            # Decompose into magnitude and direction
            norms = torch.norm(x_reshaped, dim=-1, keepdim=True)  # (batch, seq_len, n_vectors, 1)
            safe_norms = torch.clamp(norms, min=self.eps)
            directions = x_reshaped / safe_norms  # (batch, seq_len, n_vectors, dim)
            
            # Process magnitudes using only rotation-invariant features
            if self.use_cross_magnitude:
                # Cross-vector magnitude statistics (all rotation-invariant)
                mean_magnitude = torch.mean(norms, dim=2, keepdim=True)  # (batch, seq_len, 1, 1)
                max_magnitude = torch.max(norms, dim=2, keepdim=True)[0]  # (batch, seq_len, 1, 1)
                
                # Broadcast to all vectors
                mean_magnitude = mean_magnitude.expand_as(norms)
                max_magnitude = max_magnitude.expand_as(norms)
                
                # Concatenate invariant features
                magnitude_features = torch.cat([
                    norms + self.magnitude_bias,
                    mean_magnitude,
                    max_magnitude
                ], dim=-1)  # (batch, seq_len, n_vectors, 3)
                
                magnitude_gates = self.magnitude_net(magnitude_features)  # (batch, seq_len, n_vectors, 1)
            else:
                # Simple magnitude processing
                magnitude_gates = self.magnitude_net(norms + self.magnitude_bias)  # (batch, seq_len, n_vectors, 1)
            
            # Magnitude gating keeps the direction unchanged up to the residual path.
            output_vectors = magnitude_gates * safe_norms * directions
            
            # Residual connection for training stability
            output = x_reshaped + output_vectors
            return output.view(batch_size, seq_len, total_dim)

        else:
            raise ValueError("Unsupported input shape. Expected 2D (batch, n*dim) or 3D (batch, seq_len, n*dim)")


class TFNNonLin(nn.Module):
    """
    Tensor Field Network inspired equivariant non-linearity.
    
    Features:
    - Multi-type tensor processing (scalar L=0, vector L=1, tensor L=2)
    - Gating between different tensor types
    - Spherical harmonic inspired operations
    - Complex self-interaction mechanisms
    
    Based on "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds"
    
    Args:
        dim: Dimension of each vector (2 or 3)
        hidden_dim: Hidden dimension for internal processing
        use_scalar_gating: Whether to use scalar-to-vector gating
        use_vector_to_scalar: Whether to use vector-to-scalar reduction
        use_tensor_interactions: Whether to include L=2 tensor interactions (3D only)
        nonlinearity: Base activation function
        eps: Small constant for numerical stability
    """
    def __init__(self, dim=2, hidden_dim=64, use_scalar_gating=True, use_vector_to_scalar=True, 
                 use_tensor_interactions=False, nonlinearity='ssp', eps=1e-6):
        super(TFNNonLin, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.use_scalar_gating = use_scalar_gating
        self.use_vector_to_scalar = use_vector_to_scalar
        self.use_tensor_interactions = use_tensor_interactions and (dim == 3)  # Only for 3D
        self.eps = eps
        
        # Base activation - using shifted softplus like TFN
        if nonlinearity == 'ssp':  # Shifted softplus
            self.act = SmoothStepActivation()
        elif nonlinearity == 'relu':
            self.act = nn.ReLU()
        elif nonlinearity == 'gelu':
            self.act = nn.GELU()
        elif nonlinearity == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        
        # Scalar (L=0) processing networks
        self.scalar_self_interaction = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Vector magnitude processing for rotation equivariant nonlinearity
        self.vector_magnitude_net = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Scalar-to-vector gating (like TFN's 0 x 1 -> 1 operation)
        if use_scalar_gating:
            self.scalar_to_vector_gate = nn.Sequential(
                nn.Linear(1, hidden_dim // 2),
                self.act,
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Gate values
            )
        
        # Vector-to-scalar reduction (like TFN's 1 x 1 -> 0 operation)
        if use_vector_to_scalar:
            self.vector_to_scalar_net = nn.Sequential(
                nn.Linear(dim, hidden_dim // 2),
                self.act,
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Tensor L=2 interactions for 3D (following TFN's spherical harmonics)
        if self.use_tensor_interactions:
            # L=2 has 5 components for 3D
            self.tensor_l2_net = nn.Sequential(
                nn.Linear(5, hidden_dim // 2),
                self.act,
                nn.Linear(hidden_dim // 2, 1)
            )
            
        # Learnable biases for rotation equivariant nonlinearity
        self.scalar_bias = nn.Parameter(torch.zeros(1))
        self.vector_bias = nn.Parameter(torch.zeros(1))
        
    def _compute_l2_spherical_harmonics(self, vectors):
        """
        Compute L=2 spherical harmonic-like features from 3D vectors.
        Following TFN's Y_2 function.
        
        Args:
            vectors: (*, 3) tensor of 3D vectors
            
        Returns:
            (*, 5) tensor of L=2 features
        """
        x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
        r2 = torch.clamp(torch.sum(vectors**2, dim=-1), min=self.eps)
        
        # TFN's Y_2 spherical harmonic components
        l2_features = torch.stack([
            x * y / r2,                                                    # xy
            y * z / r2,                                                    # yz
            (-x**2 - y**2 + 2*z**2) / (2 * np.sqrt(3) * r2),            # z^2 component
            z * x / r2,                                                    # zx
            (x**2 - y**2) / (2 * r2)                                       # x^2-y^2
        ], dim=-1)
        
        return l2_features
    
    def _rotation_equivariant_nonlinearity(self, vectors, bias):
        """
        Apply rotation-equivariant nonlinearity to vectors.
        Following TFN's approach: f(v) = g(||v||) * v/||v||
        """
        # Compute norms
        norms = torch.norm(vectors, dim=-1, keepdim=True)
        safe_norms = torch.clamp(norms, min=self.eps)
        
        # Apply nonlinearity to norms with bias
        gated_norms = self.act(self.vector_magnitude_net(safe_norms + bias))
        
        # Scale vectors by gated norms
        normalized_vectors = vectors / safe_norms
        return gated_norms * normalized_vectors
    
    def _process_vectors(self, x_reshaped):
        """Process vectors with TFN-inspired operations."""
        batch_shape = x_reshaped.shape[:-2]  # All dims except n_vectors and dim
        n_vectors = x_reshaped.shape[-2]
        
        # Extract scalars (L=0) - use vector magnitudes as scalar features
        vector_norms = torch.norm(x_reshaped, dim=-1, keepdim=True)  # (..., n_vectors, 1)
        
        # Process scalars through self-interaction
        processed_scalars = self.scalar_self_interaction(vector_norms + self.scalar_bias)
        processed_scalars = self.act(processed_scalars)
        
        # Apply rotation-equivariant nonlinearity to vectors (L=1)
        processed_vectors = self._rotation_equivariant_nonlinearity(x_reshaped, self.vector_bias)
        
        # Scalar-to-vector gating (TFN's 0 x 1 -> 1)
        if self.use_scalar_gating:
            scalar_gates = self.scalar_to_vector_gate(processed_scalars)  # (..., n_vectors, 1)
            processed_vectors = processed_vectors * scalar_gates
        
        # Vector-to-scalar interactions (TFN's 1 x 1 -> 0)
        if self.use_vector_to_scalar:
            # Flatten for processing
            flat_vectors = x_reshaped.view(-1, self.dim)
            vector_to_scalar = self.vector_to_scalar_net(flat_vectors)
            vector_to_scalar = vector_to_scalar.view(*batch_shape, n_vectors, 1)
            
            # Add to processed scalars
            processed_scalars = processed_scalars + vector_to_scalar
        
        # L=2 tensor interactions for 3D
        if self.use_tensor_interactions and self.dim == 3:
            # Compute L=2 spherical harmonic features
            l2_features = self._compute_l2_spherical_harmonics(x_reshaped)  # (..., n_vectors, 5)
            
            # Process L=2 features
            flat_l2 = l2_features.view(-1, 5)
            l2_contribution = self.tensor_l2_net(flat_l2)
            l2_contribution = l2_contribution.view(*batch_shape, n_vectors, 1)
            
            # Add L=2 contribution to scalars
            processed_scalars = processed_scalars + l2_contribution
        
        # Combine scalar and vector contributions
        # Use processed scalars to gate the processed vectors
        final_scalar_gates = torch.sigmoid(processed_scalars)
        output_vectors = processed_vectors * final_scalar_gates
        
        # Residual connection for training stability
        return x_reshaped + output_vectors
    
    def forward(self, x):
        # Handle 2D input: (batch, n_vectors * dim)
        if x.dim() == 2:
            batch_size, total_dim = x.shape
            assert total_dim % self.dim == 0, "Input dimension mismatch"
            n_vectors = total_dim // self.dim
            x_reshaped = x.view(batch_size, n_vectors, self.dim)
            
            processed = self._process_vectors(x_reshaped)
            return processed.view(batch_size, total_dim)

        # Handle 3D input: (batch, seq_len, n_vectors * dim)  
        elif x.dim() == 3:
            batch_size, seq_len, total_dim = x.shape
            assert total_dim % self.dim == 0, "Input dimension mismatch"
            n_vectors = total_dim // self.dim
            x_reshaped = x.view(batch_size, seq_len, n_vectors, self.dim)
            
            processed = self._process_vectors(x_reshaped)
            return processed.view(batch_size, seq_len, total_dim)

        else:
            raise ValueError("Unsupported input shape. Expected 2D (batch, n*dim) or 3D (batch, seq_len, n*dim)")
    

class IrrepNonLin(nn.Module):
    """
    Irreducible Representation Equivariant Non-linearity.
    
    Based on SE(3)-Transformer's GNormSE3, this is the most appropriate non-linearity 
    for single tensor types like T(1,2) for O(5) groups.
    
    Key insight: For any irreducible representation, we can decompose into:
    - Norm: ||feature||₂ (invariant under group action)  
    - Phase: feature/||feature|| (transforms equivariantly)
    
    The non-linearity applies learned functions to norms while preserving phases.
    
    Mathematical form: f(x) = φ(||x||) * (x/||x||)
    where φ is a learnable scalar function.
    
    Args:
        irrep_dim: Dimension of the irreducible representation
        hidden_dim: Hidden dimension for norm processing network
        num_layers: Number of layers in the norm processing network
        nonlinearity: Activation function for internal processing
        eps: Small constant for numerical stability
        use_bias: Whether to add learnable bias to norms
        use_residual: Whether to include residual connections
    """
    def __init__(self, irrep_dim=None, dim=None, hidden_dim=64, num_layers=2, nonlinearity='relu', 
                 eps=1e-12, use_bias=True, use_residual=True):
        super(IrrepNonLin, self).__init__()
        
        # Handle both 'dim' and 'irrep_dim' for compatibility
        if irrep_dim is not None:
            self.irrep_dim = irrep_dim
        elif dim is not None:
            self.irrep_dim = dim
        else:
            raise ValueError("Must provide either 'irrep_dim' or 'dim'")
            
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eps = eps
        self.use_bias = use_bias
        self.use_residual = use_residual
        
        # Base activation for internal processing
        if nonlinearity == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif nonlinearity == 'gelu':
            self.act = nn.GELU()
        elif nonlinearity == 'swish':
            self.act = nn.SiLU()
        elif nonlinearity == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        
        # Norm processing network: maps from norms to scaling factors
        # This is the key component that makes it work for any irrep
        self.norm_net = self._build_norm_net()
        
        # Optional bias for numerical stability
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.bias = 0.0
    
    def _build_norm_net(self):
        """Build the norm processing network f: R⁺ → R⁺."""
        if self.num_layers == 0:
            # Just batch norm + activation if no hidden layers
            return nn.Sequential(
                nn.BatchNorm1d(1),
                self.act
            )
        
        layers = []
        
        # Input layer
        layers.append(nn.BatchNorm1d(1))
        layers.append(self.act)
        layers.append(nn.Linear(1, self.hidden_dim))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(self.act) 
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # Output layer (no bias on final layer for stability)
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(self.act)
        layers.append(nn.Linear(self.hidden_dim, 1, bias=False))
        
        return nn.Sequential(*layers)
    
    def _process_irreps(self, x_reshaped):
        """
        Apply irrep-equivariant non-linearity.
        
        Args:
            x_reshaped: (..., n_irreps, irrep_dim) tensor
            
        Returns:
            Processed tensor of same shape
        """
        # Compute norms for each irrep: ||x||₂
        norms = torch.norm(x_reshaped, dim=-1, keepdim=True)  # (..., n_irreps, 1)
        safe_norms = torch.clamp(norms, min=self.eps)
        
        # Compute normalized features (phases): x/||x||
        phases = x_reshaped / safe_norms  # (..., n_irreps, irrep_dim)
        
        # Process norms through learnable function
        # Flatten for batch processing
        original_shape = norms.shape
        flat_norms = norms.view(-1, 1)  # (batch_size * n_irreps, 1)
        
        # Add bias for numerical stability
        flat_norms_biased = flat_norms + self.bias
        
        # Apply norm network: φ(||x|| + bias)
        processed_norms = self.norm_net(flat_norms_biased)  # (batch_size * n_irreps, 1)
        
        # Reshape back to original structure
        processed_norms = processed_norms.view(original_shape)  # (..., n_irreps, 1)
        
        # Apply processed norms to phases: φ(||x||) * (x/||x||)
        output = processed_norms * phases  # (..., n_irreps, irrep_dim)
        
        # Optional residual connection for training stability
        if self.use_residual:
            output = output + x_reshaped
            
        return output
    
    def forward(self, x):
        # Handle 2D input: (batch, n_irreps * irrep_dim)
        if x.dim() == 2:
            batch_size, total_dim = x.shape
            assert total_dim % self.irrep_dim == 0, f"Input dimension {total_dim} not divisible by irrep_dim {self.irrep_dim}"
            n_irreps = total_dim // self.irrep_dim
            x_reshaped = x.view(batch_size, n_irreps, self.irrep_dim)
            
            processed = self._process_irreps(x_reshaped)
            return processed.view(batch_size, total_dim)

        # Handle 3D input: (batch, seq_len, n_irreps * irrep_dim)  
        elif x.dim() == 3:
            batch_size, seq_len, total_dim = x.shape
            assert total_dim % self.irrep_dim == 0, f"Input dimension {total_dim} not divisible by irrep_dim {self.irrep_dim}"
            n_irreps = total_dim // self.irrep_dim
            x_reshaped = x.view(batch_size, seq_len, n_irreps, self.irrep_dim)
            
            processed = self._process_irreps(x_reshaped)
            return processed.view(batch_size, seq_len, total_dim)

        else:
            raise ValueError("Unsupported input shape. Expected 2D (batch, n*irrep_dim) or 3D (batch, seq_len, n*irrep_dim)")

    
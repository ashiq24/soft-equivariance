"""
Autoregressive Transformer for 2D Human Trajectory Prediction.

Uses PyTorch's nn.TransformerDecoder for efficient implementation.
"""

import torch
import torch.nn as nn
from utils.eq_nonlin import EQNonLin

class AutoregressiveTransformer(nn.Module):
    """
    Autoregressive transformer for trajectory prediction using PyTorch nn.TransformerDecoder.
    
    This model uses a decoder-only architecture with causal masking, similar to GPT-style models.
    During training, it uses teacher forcing with the full sequence (observed + ground truth).
    During inference, it generates predictions autoregressively one timestep at a time.
    
    Args:
        max_people: Maximum number of people in a scene
        obs_len: Number of observed timesteps
        pred_len: Number of predicted timesteps
        d_model: Embedding dimension (default: 512)
        num_heads: Number of attention heads (default: 8, must divide d_model evenly)
        num_layers: Number of transformer decoder layers (default: 6)
        dim_feedforward: Feed-forward network hidden dimension (default: 2048, typically 4 * d_model)
        dropout: Dropout rate (default: 0.1)
        nonlinearity: Activation function (default: 'gelu')
            - 'gelu': Gaussian Error Linear Unit
            - 'relu': Rectified Linear Unit
            - 'identity': No activation (linear, for equivariance testing)
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
        nonlinearity: str = 'gelu'  # 'relu' or 'gelu'
    ):
        super().__init__()
        self.max_people = max_people
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Validate that num_heads divides d_model evenly
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        # Input projection: (max_people * 2) -> d_model
        # Each timestep has max_people * 2 coordinates (x, y for each person)
        self.input_projection = nn.Linear(max_people * 2, d_model)
        
        # Learnable positional encoding (simpler and often more effective than sinusoidal)
        # Shape: (1, obs_len + pred_len, d_model)
        # Initialized with small random values
        self.pos_encoder = nn.Parameter(torch.randn(1, obs_len + pred_len, d_model) * 0.02)
        
        # Handle activation function
        # Support 'identity' as a no-op activation for strict equivariance testing
        if nonlinearity == 'identity':
            activation_fn = lambda x: x  # Identity function (no activation)
        elif nonlinearity == 'eq_nonlin':
            if d_model % 2 != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by 2 for eq_nonlin activation")
            # Create a ModuleList to hold EQNonLin instances (one per layer)
            self.eq_nonlin_layers = nn.ModuleList([
                EQNonLin(dim=2, nonlinearity='sigmoid') for _ in range(num_layers)
            ])
            # We'll handle this specially in the forward pass
            activation_fn = 'eq_nonlin'  # String marker
        else:
            activation_fn = nonlinearity  # 'relu', 'gelu', etc.
        
        # PyTorch TransformerDecoder (uses TransformerDecoderLayer internally)
        # This includes multi-head attention, layer norm, feedforward, residual connections
        if nonlinearity == 'eq_nonlin':
            # Build custom decoder with separate EQNonLin per layer
            self.decoder_layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    bias=True,
                    dropout=dropout,
                    activation=self.eq_nonlin_layers[i],  # Pass the module instance
                    batch_first=True,
                    norm_first=False
                ) for i in range(num_layers)
            ])
            self.transformer_decoder = None  # Won't use this
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                bias=True,
                dropout=dropout,
                activation=activation_fn,
                batch_first=True,
                norm_first=False
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.decoder_layers = None
        
        # Output projection: d_model -> (max_people * 2)
        self.output_projection = nn.Linear(d_model, max_people * 2)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        # Input and output projections
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, obs_traj, validity_mask, gt_traj=None):
        """
        Forward pass with support for both training (teacher forcing) and inference (autoregressive).
        
        Args:
            obs_traj: (batch, max_people, 2, obs_len) - Observed trajectories
            validity_mask: (batch, max_people) - Binary mask indicating valid people
            gt_traj: (batch, max_people, 2, pred_len) - Ground truth trajectories (only during training)
            
        Returns:
            pred_traj: (batch, max_people, 2, pred_len) - Predicted trajectories
        """
        batch_size = obs_traj.size(0)
        device = obs_traj.device
        
        # Reshape input: (batch, max_people, 2, obs_len) -> (batch, obs_len, max_people * 2)
        # This flattens the spatial coordinates while keeping time as the sequence dimension
        obs_traj_reshaped = obs_traj.permute(0, 3, 1, 2).reshape(batch_size, self.obs_len, -1)
        
        if self.training and gt_traj is not None:
            # ===== TRAINING MODE: Teacher forcing with causal masking =====
            # Reshape ground truth: (batch, max_people, 2, pred_len) -> (batch, pred_len, max_people * 2)
            gt_traj_reshaped = gt_traj.permute(0, 3, 1, 2).reshape(batch_size, self.pred_len, -1)
            
            # Concatenate observed + ground truth: (batch, obs_len + pred_len, max_people * 2)
            full_seq = torch.cat([obs_traj_reshaped, gt_traj_reshaped], dim=1)
            
            # Project to embedding space and add positional encoding
            full_seq_embedded = self.input_projection(full_seq) + self.pos_encoder[:, :full_seq.size(1), :]
            
            # Generate causal mask for transformer (upper triangular)
            # This ensures each position can only attend to itself and previous positions
            seq_len = full_seq.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            
            # Memory is empty (decoder-only architecture, no encoder)
            memory = torch.zeros(batch_size, 0, self.d_model, device=device)
            
            # Pass through transformer decoder
            if self.decoder_layers is not None:
                # Custom decoder with EQNonLin
                output = full_seq_embedded
                for layer in self.decoder_layers:
                    output = layer(output, memory, tgt_mask=causal_mask)
            else:
                output = self.transformer_decoder(full_seq_embedded, memory, tgt_mask=causal_mask)
            
            # Extract predicted timesteps and project to output space
            pred_output = output[:, self.obs_len:, :]  # (batch, pred_len, d_model)
            pred_traj_reshaped = self.output_projection(pred_output)  # (batch, pred_len, max_people * 2)
            
        else:
            # ===== INFERENCE MODE: Autoregressive generation =====
            # Start with observed trajectory
            current_seq = obs_traj_reshaped.clone()  # (batch, obs_len, max_people * 2)
            pred_list = []
            
            for t in range(self.pred_len):
                seq_len = current_seq.size(1)
                
                # Project and add positional encoding
                seq_embedded = self.input_projection(current_seq) + self.pos_encoder[:, :seq_len, :]
                
                # Generate causal mask for current sequence length
                causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
                
                # Memory is empty (decoder-only)
                memory = torch.zeros(batch_size, 0, self.d_model, device=device)
                
                # Transformer forward pass
                if self.decoder_layers is not None:
                    # Custom decoder with EQNonLin
                    output = seq_embedded
                    for layer in self.decoder_layers:
                        output = layer(output, memory, tgt_mask=causal_mask)
                else:
                    output = self.transformer_decoder(seq_embedded, memory, tgt_mask=causal_mask)
                
                # Get last timestep prediction
                next_step = self.output_projection(output[:, -1:, :])  # (batch, 1, max_people * 2)
                pred_list.append(next_step)
                
                # Append to sequence for next iteration
                current_seq = torch.cat([current_seq, next_step], dim=1)
            
            # Concatenate predictions
            pred_traj_reshaped = torch.cat(pred_list, dim=1)  # (batch, pred_len, max_people * 2)
        
        # Reshape output: (batch, pred_len, max_people * 2) -> (batch, max_people, 2, pred_len)
        pred_traj = pred_traj_reshaped.reshape(batch_size, self.pred_len, self.max_people, 2)
        pred_traj = pred_traj.permute(0, 2, 3, 1)  # (batch, max_people, 2, pred_len)
        
        return pred_traj

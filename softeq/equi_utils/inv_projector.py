import torch
import torch.nn as nn


def _create_smoothing_mask(values, size, soft_threshold, dtype, debug=False, hard=False, hard_mask=False):
    """Build a diagonal smoothing mask from basis values.

    The mask is diagonal. Entries up to the selected basis index are set to 1,
    and the remaining diagonal entries are exponentially damped with
    ``exp(-|value| / (soft_threshold + eps))`` when hard mask is false, else
    the entries are set to 0.
    When ``soft_threshold == 1.0``, the mask becomes the identity.

    Args:
        values: 1D tensor of singular/eigen values.
        size: Shape of the basis matrix used to size the output mask.
        soft_threshold: Interpolation factor controlling how aggressively the
            non-zero spectrum is suppressed.
        dtype: Output dtype for the mask.
        debug: If True, prints the sorted values and chosen index.
        hard: If True, allows the no-solution branch to collapse to zeros.
        hard_mask: If True, set the entries to 0 to the mask beyond the selected index (cutoff index).

    Returns:
        A square diagonal mask tensor of shape ``(size[-1], size[-1])``.
    """
    # get last index zero eigenvalues
    values = torch.abs(values)
    if debug:
        print("values:", values)
    zero_indices= torch.where(torch.abs(values) < 1e-5)[0]
    if len(zero_indices) == 0:
        last_zero_eig_index = 0 if not hard else -1
    else:
        last_zero_eig_index = zero_indices[-1]
    # Keep the exact zero subspace and then extend into the non-zero spectrum.
    # The explicit soft_threshold == 1.0 branch below handles the full identity case.
    last_basis_idx = last_zero_eig_index + int((len(values) - len(zero_indices) - 1)* soft_threshold)

    print("last_basis_idx and total basis:", last_basis_idx, len(values))
    assert (last_basis_idx >= 0 or hard) and last_basis_idx < size[-1]
    
    exp_values = torch.exp(-values/ (soft_threshold + 1e-6)).to(dtype=dtype)
    if hard_mask:
        exp_values = torch.zeros_like(exp_values)
    # create a diagonal matrix from exp_values  
    mask = torch.diag(exp_values)

    if soft_threshold == 1.0:
        mask = torch.eye(size[-1], dtype=dtype, device=mask.device)
    elif last_basis_idx >= 0:
        mask[:last_basis_idx+1, :last_basis_idx+1] = torch.eye(last_basis_idx+1, dtype=dtype, device=mask.device)
    else:
        # no solution case
        if hard and soft_threshold == 0.0:
            mask = torch.zeros_like(mask)

    return mask

class InvariantProjector(nn.Module):
    def __init__(self, U_w, U_x, values, softness=0.0, debug=False, hard=False, hard_mask=False, preserve_norm=False):
        super().__init__()
        
        if not (0.0 <= softness <= 1.0):
            raise ValueError("Softness parameter must be between 0.0 and 1.0")
        self.softness = softness
        self.debug = debug
        self.preserve_norm = preserve_norm
        mask = _create_smoothing_mask(values, U_w.shape, softness, U_w.dtype, debug=debug, hard=hard, hard_mask=hard_mask)

        if self.debug:
            print("Values for smoothing mask:")
            print("---"*20)
            print(values)
            print("---"*20)
            print("Smoothing mask:")
            for i in mask:
                print(i)
            print("---"*20)
        
        filter_w = U_w @ mask @ U_w.transpose(0,1)
        self.register_buffer('filter_w_T', filter_w.t())
        filter_x = U_x @ mask @ U_x.transpose(0,1)
        self.register_buffer('filter_x', filter_x)

    def forward(self, W):
        if self.softness == 1.0:
            return W  # No smoothing applied
        in_channels = W.shape[1] // self.filter_w_T.shape[0]
        W = W.view(W.shape[0], in_channels, self.filter_w_T.shape[0])
        
        # Calculate norm along last dimension before projection (if preserving norm)
        if self.preserve_norm:
            norm_before = torch.norm(W, dim=-1, keepdim=False)  # [b, c]
        
        # Apply projection
        W = torch.einsum('fi,bci->bcf', self.filter_w_T, W)
        
        # Scale to match original norm (if preserving norm)
        if self.preserve_norm:
            norm_after = torch.norm(W, dim=-1, keepdim=False)  # [b, c]
            norm_ratio = norm_before / (norm_after + 1e-8)  # [b, c]
            W = W * norm_ratio.unsqueeze(-1)  # [b, c, f]
        
        return W.reshape(W.shape[0], -1)

    def smooth(self, x, data_last=False):
        """Project activations / positional embeddings onto the filtered basis.
        This projection funtion work on data (rather than weights in forward).
        
        Supports 1D, 2D, and 3D tensors. For 3D tensors, ``data_last=False``
        treats the layout as ``(batch, tokens, channels)`` and projects the token
        axis, while ``data_last=True`` treats the layout as
        ``(batch, tokens, channels * basis_dim)`` and reshapes before applying the
        filter.

        Args:
            x: Tensor to smooth.
            data_last: Whether the projected basis dimension is stored in the last
                axis of a 3D tensor.

        Returns:
            Tensor with the same rank as the input, filtered along the basis axis.
        """
        if self.softness == 1.0:
            return x  # No smoothing applied
        
        if x.dim() == 3:
            # (batch, n_tokens, channels)
            if data_last:
                # (batch, seq_len, channels*vectors)
                in_channels = x.shape[2] // self.filter_x.shape[1]
                x = x.view(x.shape[0], x.shape[1], in_channels, self.filter_x.shape[1])
                
                # Calculate norm along last dimension before projection (if preserving norm)
                if self.preserve_norm:
                    norm_before = torch.norm(x, dim=-1, keepdim=False)  # [b, t, c]
                
                # Apply projection
                x = torch.einsum('fi,btci->btcf', self.filter_x, x)
                
                # Scale to match original norm (if preserving norm)
                if self.preserve_norm:
                    norm_after = torch.norm(x, dim=-1, keepdim=False)  # [b, t, c]
                    norm_ratio = norm_before / (norm_after + 1e-8)  # [b, t, c]
                    x = x * norm_ratio.unsqueeze(-1)  # [b, t, c, f]
                
                x = x.reshape(x.shape[0], x.shape[1], -1)
            else:
                # Input: [b, i, c] where i is tokens, c is channels
                # Projection transforms i dimension to f dimension: [b, i, c] -> [b, f, c]
                # Calculate norm along the dimension being projected (dim=1, tokens) before projection
                if self.preserve_norm:
                    norm_before = torch.norm(x, dim=1, keepdim=False)  # [b, c]
                
                # Apply projection
                x = torch.einsum('fi,bic->bfc', self.filter_x, x)
                
                # Calculate norm along the projected dimension (dim=1, filters) after projection
                if self.preserve_norm:
                    norm_after = torch.norm(x, dim=1, keepdim=False)  # [b, c]
                    norm_ratio = norm_before / (norm_after + 1e-8)  # [b, c]
                    x = x * norm_ratio.unsqueeze(1)  # [b, f, c] - scale along f dimension
        elif x.dim() == 2:
            # shape: batch, channels * vectors
            in_channels = x.shape[1] // self.filter_x.shape[1]
            x = x.view(x.shape[0], in_channels, self.filter_x.shape[1])
            
            # Calculate norm along last dimension before projection (if preserving norm)
            if self.preserve_norm:
                norm_before = torch.norm(x, dim=-1, keepdim=False)  # [b, c]
            
            # Apply projection
            x = torch.einsum('fi,bci->bcf', self.filter_x, x)
            
            # Scale to match original norm (if preserving norm)
            if self.preserve_norm:
                norm_after = torch.norm(x, dim=-1, keepdim=False)  # [b, c]
                norm_ratio = norm_before / (norm_after + 1e-8)  # [b, c]
                x = x * norm_ratio.unsqueeze(-1)  # [b, c, f]
            
            x = x.reshape(x.shape[0], -1)
        elif x.dim() == 1:
            # shape: channels * vectors
            in_channels = x.shape[0] // self.filter_x.shape[1]
            x = x.view(in_channels, self.filter_x.shape[1])
            
            # Calculate norm along last dimension before projection (if preserving norm)
            if self.preserve_norm:
                norm_before = torch.norm(x, dim=-1, keepdim=False)  # [c]
            
            # Apply projection
            x = torch.einsum('fi,ci->cf', self.filter_x, x)
            
            # Scale to match original norm (if preserving norm)
            if self.preserve_norm:
                norm_after = torch.norm(x, dim=-1, keepdim=False)  # [c]
                norm_ratio = norm_before / (norm_after + 1e-8)  # [c]
                x = x * norm_ratio.unsqueeze(-1)  # [c, f]
            
            x = x.reshape(-1)
        else:
            raise ValueError("Input tensor must be 1D, 2D, or 3D")

        return x




class MultiGeneratorInvariantProjector(nn.Module):
    def __init__(self, InvariantProjector_list):
        super().__init__()
        ## nn sequence of invariant projectors
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

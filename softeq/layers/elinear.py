import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import math
from typing import Optional
import einops


     

class ELinear(nn.Module):
    """
    Equivariant (or invariant) Linear Layer.
    The default behavior is to enforce invariance.
    If enforce_equivariance is True, the layer will enforce equivariance.
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): If True, the layer will have a bias term
        device (torch.device): Device to use
        dtype (torch.dtype): Dtype to use
        enforce_equivariance (bool): If True, the layer will enforce equivariance
        in_channels (int): Number of input channels (optional)
        out_channels (int): Number of output channels (optional)
        basis (Tensor): Basis vectors, shape (dim, n_basis)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        enforce_equivariance: bool = False,
        in_channels: int = None,
        out_channels: int = None,
        basis: Tensor = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels if in_channels is not None else 1
        self.out_channels = out_channels if out_channels is not None else 1
        self.enforce_equivariance = enforce_equivariance

        if bias:
            if enforce_equivariance:
                # Shared scalar bias for the equivariant case when working with images or volumetric data.
                # For vector data bias need to be False 
                self.bias = nn.Parameter(torch.zeros(1, **self.factory_kwargs))
            else:
                self.bias = nn.Parameter(torch.zeros(out_features, **self.factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if basis is None:
            raise ValueError("ELinear requires a basis tensor; got None")
        self.register_buffer("basis", basis)

        if not enforce_equivariance:
            assert in_features == basis.shape[0], "In features must be equal to the number of basis vectors"
            coeff_shape = (out_features, basis.shape[1])
        else:
            assert in_features * out_features == basis.shape[0], "In features * out_features must be equal to the number of basis vectors"
            coeff_shape = (self.in_channels * self.out_channels, basis.shape[1])

        self.coeffs = nn.Parameter(torch.zeros(*coeff_shape, **self.factory_kwargs))
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        weight = self.basis @ self.coeffs.transpose(0, 1)
        if self.enforce_equivariance:
            weight = weight.reshape(self.in_features, self.out_features, self.in_channels * self.out_channels)
            weight = weight.transpose(0, 1)
            if self.in_channels > 1 or self.out_channels > 1:
                # weight currently has shape (o, i, c_in * c_out)
                # We want a 2D weight matrix with shape (o * c_out, i * c_in)
                # Mapping: M[o*c_out + out_c, i*c_in + in_c] = weight[o, i, in_c * c_out + out_c]
                weight = einops.rearrange(weight, 'o i (c_in c_out) -> (c_out o) (c_in i)', c_in=self.in_channels, c_out=self.out_channels)
            else:
                weight = weight.squeeze(-1)
        else:
            weight = weight.transpose(0, 1)
        return torch.nn.functional.linear(input, weight, self.bias)

    def reset_parameters(self) -> None:
        """
        Mimic torch.nn.Linear.reset_parameters for coeffs/bias.
        Weight in nn.Linear is initialized with kaiming_uniform_(a=sqrt(5)),
        and bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        Here we apply the same to coeffs; bias uses fan_in = in_features.
        """
        init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))
        self.coeffs.data = (self.coeffs.data.shape[-1] / self.in_features) * self.coeffs.data.clone().detach()
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)
"""
Discrete reflection constraints for 2D and 3D vectors.
Implements reflection invariant and equivariant basis computation for vector data.
"""

import torch
import math
from ..utils.decompositions import schur_decomposition, svd_decomposition
from ..utils.group_utils_vec import create_reflection_action_matrix_vec


class DiscreteReflectionConstraintsVec:
    """
    Discrete reflection constraints for 2D and 3D vectors.
    
    Reflection is an involution (applying it twice gives identity), so there's no
    "number of reflections" parameter like rotations. The reflection is defined by
    the axis parameter.
    
    Args:
        input_size (int): Dimension of input vectors (2 for 2D, 3 for 3D).
        output_size (int or None): Dimension of output vectors (2 for 2D, 3 for 3D). 
                                   If None, same as input_size.
        decomposition_method (str): Method for decomposition ('svd' or 'schur').
        axis (str): Reflection axis:
            - For 2D: 'x' reflects across X-axis (flips Y), 'y' reflects across Y-axis (flips X)
            - For 3D: 'x' reflects across YZ-plane (flips X), 'y' across XZ-plane (flips Y), 
                      'z' across XY-plane (flips Z)
            Default: 'x'
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = None,
        decomposition_method: str = 'svd',
        axis: str = 'x'
    ):
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.decomposition_method = decomposition_method
        self.axis = axis.lower()
        
        # Validate input
        if self.input_size not in [2, 3]:
            raise ValueError(f"input_size must be 2 or 3, got {input_size}")
        if self.output_size not in [2, 3]:
            raise ValueError(f"output_size must be 2 or 3, got {output_size}")
        
        # Validate axis
        valid_axes_2d = ['x', 'y']
        valid_axes_3d = ['x', 'y', 'z']
        if self.input_size == 2 and self.axis not in valid_axes_2d:
            raise ValueError(f"For 2D vectors, axis must be one of {valid_axes_2d}, got '{axis}'")
        if self.input_size == 3 and self.axis not in valid_axes_3d:
            raise ValueError(f"For 3D vectors, axis must be one of {valid_axes_3d}, got '{axis}'")
        
        # Initialize basis storage
        self.invariant_basis = None
        self.invariant_scaling_values = None
        self.equivariant_basis = None
        self.equivariant_scaling_values = None
        
        # Create group action matrices
        self.create_unit_group_action_matrix()
        self.create_unit_group_action_forward_difference_matrix()
    
    def create_unit_group_action_matrix(self):
        """
        Create the reflection matrices for input and output.
        Sets self.unit_group_action_matrix and self.unit_group_action_matrix_out.
        """
        # Create reflection matrices for input
        self.unit_group_action_matrix = create_reflection_action_matrix_vec(
            self.input_size, axis=self.axis
        )
        
        # Create reflection matrix for output (if different size)
        if self.output_size != self.input_size:
            self.unit_group_action_matrix_out = create_reflection_action_matrix_vec(
                self.output_size, axis=self.axis
            )
        else:
            self.unit_group_action_matrix_out = self.unit_group_action_matrix
    
    def create_unit_group_action_forward_difference_matrix(self):
        """
        Create forward difference matrix (R - I) for reflections.
        
        For reflection, R² = I, so the eigenvalues of R are ±1.
        The null space of (R - I) contains the +1 eigenspace (invariant vectors).
        """
        # For input
        self.forward_difference_matrix = (
            self.unit_group_action_matrix - 
            torch.eye(
                self.input_size, 
                device=self.unit_group_action_matrix.device,
                dtype=self.unit_group_action_matrix.dtype
            )
        )
        # normalize forward difference matrix by the discrete angle in radians of the reflections (pi)
        self.forward_difference_matrix = self.forward_difference_matrix / math.pi
        
        # For output
        if self.output_size != self.input_size:
            self.forward_difference_matrix_out = (
                self.unit_group_action_matrix_out - 
                torch.eye(
                    self.output_size,
                    device=self.unit_group_action_matrix_out.device,
                    dtype=self.unit_group_action_matrix_out.dtype
                )
            )
            # normalize forward difference matrix by the discrete angle in radians of the reflections (pi)
            self.forward_difference_matrix_out = self.forward_difference_matrix_out / math.pi
        else:
            self.forward_difference_matrix_out = self.forward_difference_matrix
    
    def apply_unit_group_action(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Apply the unit group action (reflection) to vectors.
        
        Args:
            vectors: Input vectors with shape:
                    - (batch_size, vec_dim) for single vectors
                    - (batch_size, num_points, vec_dim) for multiple vectors per sample
                    
        Returns:
            Reflected vectors with same shape as input
        """
        # Handle different input shapes
        original_shape = vectors.shape
        
        if len(original_shape) == 2:
            # (batch_size, vec_dim)
            # vectors: (B, D), reflection: (D, D) -> result: (B, D)
            reflected = vectors @ self.unit_group_action_matrix.T
        elif len(original_shape) == 3:
            # (batch_size, num_points, vec_dim)
            # vectors: (B, N, D), reflection: (D, D) -> result: (B, N, D)
            reflected = vectors @ self.unit_group_action_matrix.T
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        return reflected
    
    def apply_forward_difference(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Apply forward difference (R - I) to vectors.
        
        Args:
            vectors: Input vectors with shape:
                    - (batch_size, vec_dim) for single vectors
                    - (batch_size, num_points, vec_dim) for multiple vectors per sample
                    
        Returns:
            Transformed vectors with same shape as input
        """
        # Handle different input shapes
        original_shape = vectors.shape
        
        if len(original_shape) == 2:
            # (batch_size, vec_dim)
            result = vectors @ self.forward_difference_matrix.T
        elif len(original_shape) == 3:
            # (batch_size, num_points, vec_dim)
            result = vectors @ self.forward_difference_matrix.T
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        return result
    
    def get_decomposition(self, rep: str = 'input', method: str = None, return_original: bool = False):
        """
        Get the decomposition of the input or output representation.

        Args:
            rep (str): The representation to decompose ('input' or 'output').
            method (str): The decomposition method to use ('svd' or 'schur'). 
                         If None, uses self.decomposition_method.
            return_original (bool): If True, returns the original matrix for 'schur' method.

        Returns:
            (left_basis, scaling_values, right_basis): Tuple of decomposition components
        """
        # Select the matrix to decompose
        if rep == 'input':
            matrix = self.forward_difference_matrix
        elif rep == 'output':
            if self.output_size == self.input_size:
                matrix = self.forward_difference_matrix
            else:
                matrix = self.forward_difference_matrix_out
        else:
            raise ValueError(f"rep must be 'input' or 'output', got '{rep}'")
        
        # Use specified method or default
        decomp_method = method if method is not None else self.decomposition_method
        
        # Perform decomposition
        if decomp_method == 'schur':
            s_values, right = schur_decomposition(matrix, return_original=return_original)
            left = right
        elif decomp_method == 'svd':
            left, s_values, right = svd_decomposition(matrix)
        else:
            raise ValueError(f"Decomposition method {decomp_method} not supported")
        
        return left, s_values, right
    
    def create_invariant_basis(self):
        """
        Compute and store the invariant basis for the unit group action.
        The invariant basis spans the null space of the forward difference matrix,
        corresponding to features that don't change under reflection.
        
        For reflection along axis 'x' in 2D:
            - Invariant subspace: vectors along X-axis (y=0)
            - Anti-invariant subspace: vectors along Y-axis (x=0)
        """
        with torch.no_grad():
            if self.decomposition_method == 'schur':
                T, Z = schur_decomposition(self.forward_difference_matrix)
            elif self.decomposition_method == 'svd':
                U, S, V = svd_decomposition(self.forward_difference_matrix)
                T, Z = S, U
            else:
                raise ValueError(f"Decomposition method {self.decomposition_method} not supported")
            
            self.invariant_basis = Z.to(
                self.forward_difference_matrix.device,
                dtype=self.forward_difference_matrix.dtype
            )
            self.invariant_scaling_values = T.to(
                self.forward_difference_matrix.device,
                dtype=self.forward_difference_matrix.dtype
            )
    
    def get_invariant_basis(self, soft_thresholding: float = 0.0, tolerance: float = 1e-4):
        """
        Return the invariant basis for the unit group action.
        
        Args:
            soft_thresholding (float): Fraction of nonzero eigenvalues to keep (0 to 1).
            tolerance (float): Threshold for considering eigenvalues as zero.
            
        Returns:
            (basis, scaling_values): Filtered basis vectors and their scaling values
        """
        if self.invariant_basis is None:
            self.create_invariant_basis()
        
        return self._select_basis(
            basis=self.invariant_basis,
            scaling_values=self.invariant_scaling_values,
            soft_thresholding=soft_thresholding,
            tolerance=tolerance,
            label="reflection_invariant"
        )
    
    def create_equivariant_basis(self):
        """
        Compute and store the equivariant basis for the unit group action.
        The equivariant basis characterizes linear maps f: R^m -> R^n such that
        R_out @ f(v) = f(R_in @ v) for all reflections R.
        
        This is the null space of: kron(R_in^T, I_out) - kron(I_in, R_out)
        """
        if self.output_size == self.input_size:
            # For same input/output dimensions, use the same matrix
            condition_matrix_input = self.forward_difference_matrix
            condition_matrix_output = self.forward_difference_matrix_out
        else:
            condition_matrix_input = self.forward_difference_matrix
            condition_matrix_output = self.forward_difference_matrix_out
        
        with torch.no_grad():
            # Kronecker product condition: kron(R_in^T, I_out) - kron(I_in, R_out)
            condition_matrix = torch.kron(
                condition_matrix_input.transpose(0, 1).contiguous(),
                torch.eye(
                    condition_matrix_output.shape[0],
                    device=condition_matrix_output.device,
                    dtype=condition_matrix_output.dtype
                )
            ) - torch.kron(
                torch.eye(
                    condition_matrix_input.shape[0],
                    device=condition_matrix_input.device,
                    dtype=condition_matrix_input.dtype
                ),
                condition_matrix_output.contiguous()
            )
            
            if self.decomposition_method == 'schur':
                T, Z = schur_decomposition(condition_matrix)
            elif self.decomposition_method == 'svd':
                U, S, V = svd_decomposition(condition_matrix)
                T, Z = S, U
            else:
                raise ValueError(f"Decomposition method {self.decomposition_method} not supported")
            
            self.equivariant_basis = Z.to(
                self.forward_difference_matrix.device,
                dtype=self.forward_difference_matrix.dtype
            )
            self.equivariant_scaling_values = T.to(
                self.forward_difference_matrix.device,
                dtype=self.forward_difference_matrix.dtype
            )
    
    def get_equivariant_basis(self, soft_thresholding: float = 0.0, tolerance: float = 1e-4):
        """
        Return the equivariant basis for the unit group action.
        
        Args:
            soft_thresholding (float): Fraction of nonzero eigenvalues to keep (0 to 1).
            tolerance (float): Threshold for considering eigenvalues as zero.
            
        Returns:
            (basis, scaling_values): Filtered basis vectors and their scaling values
        """
        if self.equivariant_basis is None:
            self.create_equivariant_basis()
        
        return self._select_basis(
            basis=self.equivariant_basis,
            scaling_values=self.equivariant_scaling_values,
            soft_thresholding=soft_thresholding,
            tolerance=tolerance,
            label="reflection_equivariant"
        )
    
    def _select_basis(
        self,
        basis: torch.Tensor,
        scaling_values: torch.Tensor,
        soft_thresholding: float,
        tolerance: float,
        label: str
    ):
        """
        Select basis vectors up to a threshold determined by near-zero scaling values
        and optional soft-thresholding of the remaining components.
        
        Args:
            basis (Tensor): Basis matrix.
            scaling_values (Tensor): Associated scaling values (eigenvalues or singular values).
            soft_thresholding (float): Fraction of nonzero eigenvalues to keep (0 to 1).
            tolerance (float): Threshold for considering eigenvalues as zero.
            label (str): Label for debug prints.
            
        Returns:
            (basis, scaling_values): Filtered basis and scaling values
        """
        # Find indices where scaling values are approximately zero
        zero_idx = torch.where(scaling_values <= tolerance)[0]
        
        if zero_idx.numel() > 0:
            last_zero_index = zero_idx[-1]
        else:
            last_zero_index = torch.tensor(0, device=scaling_values.device)
        
        # Apply soft thresholding if specified
        if soft_thresholding > 0:
            remaining = scaling_values.shape[0] - int(last_zero_index)
            extra = int(soft_thresholding * remaining)
            threshold = int(last_zero_index) + extra
        else:
            threshold = int(last_zero_index)
        
        return basis[:, :threshold].clone().detach(), scaling_values[:threshold].clone().detach()

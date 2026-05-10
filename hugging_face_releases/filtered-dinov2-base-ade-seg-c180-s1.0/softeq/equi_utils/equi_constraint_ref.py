import torch
import math

from ..utils.decompositions import schur_decomposition, svd_decomposition
from ..utils.group_utils import create_unit_reflection_action_matrix

class DiscreteReflectionConstraints:
    """
    Discrete reflection constraints for 2D images (reflection along x-axis / horizontal flip).
    
    Args:
        input_size (tuple): (channels, height, width) for input.
        output_size (tuple or None): (channels, height, width) for output, or None.
        decomposition_method (str): 'schur' or 'svd'.
    """
    def __init__(self, input_size, output_size, decomposition_method='schur'):
        
        self.size = input_size
        self.output_size = output_size
        self.decomposition_method = decomposition_method
        
        # Only square images are supported
        assert self.size[-1] == self.size[-2], "Only square images are supported"
        if self.output_size is not None:
            assert self.output_size[-1] == self.output_size[-2], "Only square images are supported"

        self.create_unit_group_action_matrix()
        self.create_unit_group_action_forward_difference_matrix()
        
    def create_unit_group_action_matrix(self):
        """
        Create the reflection matrix for a unit group action (input and output if specified).
        Reflection along x-axis (horizontal flip).
        """
        self.unit_group_action_matrix = create_unit_reflection_action_matrix(self.size)
        if self.output_size is not None:
            self.unit_group_action_matrix_out = create_unit_reflection_action_matrix(
                self.output_size
            ).to(self.unit_group_action_matrix.device, dtype=self.unit_group_action_matrix.dtype)

    def apply_unit_group_action(self, image):
        """
        Apply the unit group action (horizontal reflection) to the image.
        Args:
            image (Tensor): (batch_size, channels, height, width)
        Returns:
            Tensor: Reflected image.
        """
        return self.apply_linear_operator(image, self.unit_group_action_matrix)
    
    def create_unit_group_action_forward_difference_matrix(self):
        """
        Create forward difference matrix for a unit group action (input and output if specified).
        Forward difference = R - I, where R is the reflection matrix.
        The null space of (R - I) contains invariant vectors (vectors v such that Rv = v).
        """
        self.forward_difference_matrix = self.unit_group_action_matrix - torch.eye(
            self.unit_group_action_matrix.shape[0], device=self.unit_group_action_matrix.device, dtype=self.unit_group_action_matrix.dtype
        )
        # normalize forward difference matrix by the discrete angle in radians of the reflections (pi)
        self.forward_difference_matrix = self.forward_difference_matrix / math.pi
        if self.output_size is not None:
            self.forward_difference_matrix_out = self.unit_group_action_matrix_out - torch.eye(
                self.unit_group_action_matrix_out.shape[0], device=self.unit_group_action_matrix_out.device, dtype=self.unit_group_action_matrix_out.dtype
            )
            # normalize forward difference matrix by the discrete angle in radians of the reflections (pi)
            self.forward_difference_matrix_out = self.forward_difference_matrix_out / math.pi
    
    def apply_forward_difference(self, image):
        """
        Apply forward difference to the image.
        Args:
            image (Tensor): (batch_size, channels, height, width)
        Returns:
            Tensor: Transformed image.
        """
        return self.apply_linear_operator(image, self.forward_difference_matrix)

    def apply_linear_operator(self, image, operator):
        """
        Apply a linear operator to the image.
        Args:
            image (Tensor): (batch_size, channels, height, width)
            operator (Tensor): (channels*height*width, channels*height*width)
        Returns:
            Tensor: Transformed image.
        """
        assert image.shape[2] == self.size[-2] and image.shape[3] == self.size[-1], "Image size does not match the size of the image"
        batch_size, channels, height, width = image.shape
        if operator.shape[0] != channels * height * width:
            raise ValueError("Operator has incorrect shape")
        image_flat = image.flatten(start_dim=1)
        
        # Apply linear operator: (channels*height*width, channels*height*width) @ (channels*height*width, batch_size)
        # Result: (channels*height*width, batch_size)
        linear_op_flat = operator @ image_flat.transpose(0, 1)
        # Transpose back to (batch_size, channels*height*width)
        linear_op_flat = linear_op_flat.transpose(0, 1)
        
        # Reshape back to original format
        linear_op_image = linear_op_flat.reshape(batch_size, channels, height, width)
        
        return linear_op_image

    def get_decomposition(self, rep, method, return_original=False):
        """
        Get the decomposition of a representation matrix.
        Args:
            rep (str): Decomposition of representation matrix, either 'input' or 'output'.
            method (str): 'schur' or 'svd'.
            return_original (bool): If True, returns the original schur matrix for 'schur' method instead of returning the list of scaling values.
        Returns:
            (Tensor, Tensor, Tensor): (left, scaling_values, right)
        """
        if rep == 'input':
            rep = self.forward_difference_matrix
        elif rep == 'output':
            assert self.output_size is not None, "Output size must be provided to get output decomposition"
            rep = self.forward_difference_matrix_out
        else:
            raise ValueError(f"rep must be 'input' or 'output', got {rep}")
            
        if method == 'schur':
            s_values, right = schur_decomposition(rep, return_original=return_original)
            left = right
        elif method == 'svd':
            left, s_values, right = svd_decomposition(rep)
        else:
            raise ValueError(f"Decomposition method {method} not supported")
        return left, s_values, right

    def create_invariant_basis(self):
        """
        Compute and store the invariant basis for the unit group action.
        Invariant vectors are those that remain unchanged under reflection.
        """
        with torch.no_grad():
            if self.decomposition_method == 'schur':
                T, Z = schur_decomposition(self.forward_difference_matrix)
            elif self.decomposition_method == 'svd':
                U, S, V = svd_decomposition(self.forward_difference_matrix)
                T, Z = S, U
            else:
                raise ValueError(f"Decomposition method {self.decomposition_method} not supported")
            self.invariant_basis = Z.to(self.forward_difference_matrix.device, dtype=self.forward_difference_matrix.dtype)
            self.invariant_scaling_values = T.to(self.forward_difference_matrix.device, dtype=self.forward_difference_matrix.dtype)
    
    def get_invariant_basis(self, soft_thresholding=0.0, tolerance=1e-4):
        """
        Return the invariant basis for the unit group action.
        Args:
            soft_thresholding (float): Fraction of nonzero eigenvalues to keep.
            tolerance (float): Threshold for zero eigenvalues.
        Returns:
            (Tensor, Tensor): (basis, scaling_values)
        """
        return self._select_basis(
            basis=self.invariant_basis,
            scaling_values=self.invariant_scaling_values,
            soft_thresholding=soft_thresholding,
            tolerance=tolerance,
            label="invariant"
        )

    def get_equivariant_basis(self, soft_thresholding=0.0, tolerance=1e-4):
        """
        Return the equivariant basis for the unit group action.
        Args:
            soft_thresholding (float): Fraction of nonzero eigenvalues to keep.
            tolerance (float): Threshold for zero eigenvalues.
        Returns:
            (Tensor, Tensor): (basis, scaling_values)
        """
        return self._select_basis(
            basis=self.equivariant_basis,
            scaling_values=self.equivariant_scaling_values,
            soft_thresholding=soft_thresholding,
            tolerance=tolerance,
            label="equivariant"
        )

    def _select_basis(self, basis, scaling_values, soft_thresholding: float, tolerance: float, label: str):
        """
        Select basis vectors up to a threshold determined by near-zero scaling values
        and optional soft-thresholding of the remaining components.
        Args:
            basis (Tensor): Basis matrix.
            scaling_values (Tensor): Associated scaling values.
            soft_thresholding (float): Fraction of nonzero eigenvalues to keep.
            tolerance (float): Threshold for zero eigenvalues.
            label (str): Label for debug prints.
        Returns:
            (Tensor, Tensor): (basis, scaling_values)
        """
        zero_idx = torch.where(scaling_values <= tolerance)[0]
        last_zero_index = zero_idx[-1] if zero_idx.numel() > 0 else torch.tensor(0, device=scaling_values.device)
        print(f"[{label}] Last zero index: {last_zero_index}")
        if soft_thresholding > 0:
            remaining = scaling_values.shape[0] - int(last_zero_index)
            extra = int(soft_thresholding * remaining)
            threshold = int(last_zero_index) + extra
        else:
            threshold = int(last_zero_index)
        print(f"[{label}] Number of basis vectors taken: {threshold} out of {basis.shape[1]}")
        return basis[:, :threshold].clone().detach(), scaling_values[:threshold].clone().detach()
    
    def create_equivariant_basis(self):
        """
        Compute and store the equivariant basis for the unit group action.
        Equivariant vectors transform consistently under reflection.
        """
        assert self.output_size is not None, "Output size must be provided to create equivariant basis"
        with torch.no_grad():
            condition_matrix = torch.kron(
                self.forward_difference_matrix.transpose(0, 1).contiguous(),
                torch.eye(self.forward_difference_matrix_out.shape[0])
            ) - torch.kron(
                torch.eye(self.forward_difference_matrix.shape[0]),
                self.forward_difference_matrix_out.contiguous()
            )
            if self.decomposition_method == 'schur':
                T, Z = schur_decomposition(condition_matrix)
            elif self.decomposition_method == 'svd':
                U, S, V = svd_decomposition(condition_matrix)
                T, Z = S, U
            else:
                raise ValueError(f"Decomposition method {self.decomposition_method} not supported")
            self.equivariant_basis = Z.to(self.forward_difference_matrix.device, dtype=self.forward_difference_matrix.dtype)
            self.equivariant_scaling_values = T.to(self.forward_difference_matrix.device, dtype=self.forward_difference_matrix.dtype)
        


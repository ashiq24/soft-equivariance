"""
Abstract base class for group symmetry constraints.

This module defines the BaseConstraints interface that new constraint classes
should inherit from. Existing constraint classes (DiscreteRotationConstraints,
DiscreteReflectionConstraints, O5ConstraintsVec, LorentzConstraintsVec) are
already compatible with this interface but do not inherit from it for
backward compatibility.
"""

from abc import ABC, abstractmethod
import torch


class BaseConstraints(ABC):
    """Abstract base class for group symmetry constraints.
    
    Subclasses implement constraints for a specific symmetry group
    (e.g., discrete rotations, reflections, O(5), Lorentz, SO(3)).
    
    Attributes set by subclasses in __init__:
        forward_difference_matrix: (d, d) tensor for single-generator groups
        forward_difference_matrices: list of (d, d) tensors for multi-generator groups
        forward_difference_matrix_out: (d_out, d_out) tensor (if output_size provided)
        unit_group_action_matrix: (d, d) group action on input
        unit_group_action_matrix_out: (d_out, d_out) group action on output
    """

    # --- Group actions ---
    @abstractmethod
    def create_unit_group_action_matrix(self):
        """Build and store the group action matrix(ces).
        
        For single-generator groups (rotations, reflections):
            Sets self.unit_group_action_matrix and self.unit_group_action_matrix_out.
        
        For multi-generator groups (O(5), Lorentz, SO(3)):
            Sets self.unit_group_action_matrices (list) and
            self.unit_group_action_matrices_out (list or None for invariant layers).
        """
        ...

    def apply_unit_group_action(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the unit group action to input vectors/images.
        
        Args:
            vectors: Input tensor. Shape depends on the constraint type:
                - (batch_size, vec_dim) for vectors
                - (batch_size, num_points, vec_dim) for point clouds
                - (batch_size, channels, height, width) for images
        
        Returns:
            Transformed tensor with same shape as input.
        """
        raise NotImplementedError("Subclasses must implement apply_unit_group_action().")

    @abstractmethod
    def create_unit_group_action_forward_difference_matrix(self):
        """Build and store forward difference matrix(ces).
        
        For discrete groups: computes (R - I) normalized by the angle.
        For continuous groups (Lie algebras): uses the generator directly.
        
        For single-generator groups:
            Sets self.forward_difference_matrix and self.forward_difference_matrix_out.
        
        For multi-generator groups:
            Sets self.forward_difference_matrices (list) and
            self.forward_difference_matrices_out (list).
        """
        ...

    def apply_forward_difference(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the forward difference operator to input vectors/images.
        
        For multi-generator groups, this typically applies the first generator.
        
        Args:
            vectors: Input tensor with same shape conventions as apply_unit_group_action.
        
        Returns:
            Transformed tensor with same shape as input.
        """
        raise NotImplementedError("Subclasses must implement apply_forward_difference().")

    # --- Decomposition ---
    @abstractmethod
    def get_decomposition(self, rep: str = 'input', method: str = None, return_original: bool = False):
        """Decompose (Schur/SVD) the forward difference matrix.
        
        Args:
            rep: Which representation to decompose ('input' or 'output').
            method: Decomposition method ('svd' or 'schur'). If None, uses
                   self.decomposition_method.
            return_original: If True, returns the original matrix for 'schur' method.
        
        Returns:
            For single-generator groups:
                (left_basis, scaling_values, right_basis) tuple
            For multi-generator groups:
                List of (left_basis, scaling_values, right_basis) tuples,
                one for each generator.
        """
        ...

    # --- Invariant basis ---
    def create_invariant_basis(self):
        """Compute and store the invariant basis and scaling values.
        
        The invariant basis spans the null space of the forward difference matrix,
        corresponding to features that don't change under the group action.
        
        For single-generator groups:
            Sets self.invariant_basis and self.invariant_scaling_values.
        
        For multi-generator groups:
            Sets self.invariant_basis (list) and self.invariant_scaling_values (list).
        """
        raise NotImplementedError("Subclasses must implement create_invariant_basis().")

    def get_invariant_basis(self, soft_thresholding: float = 0.0, tolerance: float = 1e-4):
        """Return (basis, scaling_values) for the invariant subspace.
        
        Args:
            soft_thresholding: Fraction of nonzero eigenvalues to keep (0 to 1).
                              0.0 = strict invariance (only null space).
            tolerance: Threshold for considering eigenvalues as zero.
        
        Returns:
            For single-generator groups:
                (basis, scaling_values) tuple where basis has shape (d, n_basis).
            For multi-generator groups:
                List of (basis, scaling_values) tuples, one for each generator.
        """
        raise NotImplementedError("Subclasses must implement get_invariant_basis().")

    # --- Equivariant basis ---
    def create_equivariant_basis(self):
        """Compute and store the equivariant basis and scaling values.
        
        The equivariant basis characterizes linear maps f: R^m -> R^n such that
        R_out @ f(v) = f(R_in @ v) for all group elements R.
        
        This is the null space of: kron(R_in^T, I_out) - kron(I_in, R_out)
        
        For multi-generator groups, this must account for equivariance under
        ALL generators simultaneously.
        """
        raise NotImplementedError("Subclasses must implement create_equivariant_basis().")

    def get_equivariant_basis(self, soft_thresholding: float = 0.0, tolerance: float = 1e-4):
        """Return (basis, scaling_values) for the equivariant subspace.
        
        The basis can be passed to ELinear for explicit equivariant layers.
        
        Args:
            soft_thresholding: Fraction of nonzero eigenvalues to keep (0 to 1).
            tolerance: Threshold for considering eigenvalues as zero.
        
        Returns:
            (basis, scaling_values) tuple where basis has shape 
            (in_features * out_features, n_basis).
        """
        raise NotImplementedError("Subclasses must implement get_equivariant_basis().")

    # --- Basis selection (shared logic, may be overridden) ---
    def _select_basis(
        self,
        basis: torch.Tensor,
        scaling_values: torch.Tensor,
        soft_thresholding: float,
        tolerance: float,
        label: str
    ):
        """Select basis vectors based on scaling values and soft thresholding.
        
        This is typically shared logic across constraint classes. The default
        implementation selects vectors corresponding to near-zero scaling values
        (the null space) plus optionally some additional vectors controlled by
        soft_thresholding.
        
        Args:
            basis: Full basis matrix with shape (d, n_total).
            scaling_values: Scaling values (eigenvalues or singular values).
            soft_thresholding: Fraction of nonzero values to additionally include.
            tolerance: Threshold for considering values as zero.
            label: Debug label for logging.
        
        Returns:
            (filtered_basis, filtered_scaling_values) tuple.
        """
        raise NotImplementedError("Subclasses must implement _select_basis().")

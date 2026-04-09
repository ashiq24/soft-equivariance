"""
O(5) constraints for 5D vectors using EMLP library.
Implements O(5)-invariant and equivariant basis computation for vector data.
"""

import torch
import numpy as np
import sys
import os

from softeq.utils.decompositions import schur_decomposition, svd_decomposition

# Try importing EMLP from installed package, fall back to external/ if not available
try:
    from emlp.groups import O
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'equivariant-MLP'))
    from emlp.groups import O


class O5ConstraintsVec:
    """
    O(5) constraints for 5D vectors using EMLP library.
    
    O(5) = SO(5) ∪ reflection, where:
    - SO(5) has Lie algebra generators (10 generators for 5D rotations)
    - O(5) adds a reflection generator
    
    Args:
        input_size (int): Dimension of input vectors (must be 5 for O(5))
        output_size (int or None): Dimension of output:
            - None or 1: For invariant layers (maps 5D vectors to scalars)
            - 5: For equivariant layers (maps 5D vectors to 5D vectors)
            If None, treated as invariant (only input constraints computed).
        decomposition_method (str): Method for decomposition ('svd' or 'schur').
        use_reflection (bool): If True, include reflection generator. If False, only SO(5).
    """
    
    def __init__(
        self,
        input_size: int = 5,
        output_size: int = None,
        decomposition_method: str = 'svd',
        use_reflection: bool = True,
        in_rep=None,
        out_rep=None
    ):
        """
        Initialize O(5) constraints with support for different representations.
        
        Args:
            input_size (int): Dimension of input vectors (auto-computed from in_rep if provided)
            output_size (int or None): Dimension of output
            decomposition_method (str): 'svd' or 'schur'
            use_reflection (bool): Include reflection generator
            in_rep (str or callable, optional): Input representation
                - If None: uses fundamental representation (default)
                - If str: Supported string formats:
                    * 'fundamental' or 'V': Fundamental representation (5D)
                    * 'V*' or 'dual': Dual representation (5D)
                    * 'V*V' or 'V⊗V': Tensor product V⊗V (25D)
                    * 'V*V*' or 'V⊗V*': Mixed tensor V⊗V* (25D)
                    * 'V**V*' or 'V*⊗V*': Dual tensor (V*)⊗(V*) (25D)
                    * 'V**2' or 'V²': Symmetric square V² (25D)
                    * 'V***2' or '(V*)²': Dual squared (V*)² (25D)
                    * 'V*V*V': Triple product V⊗V⊗V (125D)
                    * 'V*V**2' or 'V⊗(V*)²': V⊗(V*)² (125D)
                    * 'V**2*V*' or '(V*)²⊗V': (V*)²⊗V (125D)
                    * 'V+V' or 'V⊕V': Direct sum V⊕V (10D)
                    * 'V+V*' or 'V⊕V*': Direct sum V⊕V* (10D)
                    * 'T(p,q)': Rank (p,q) tensor (e.g., 'T(2,1)' = V⊗V⊗V*)
                    * 'scalar' or 'S': Scalar/trivial representation (1D)
                - If callable: custom representation builder function
                    Example: lambda G: V(G) * V(G).T  # For V⊗V*
            out_rep (str or callable, optional): Output representation
                - If None: preserves the explicit output_size argument (or None if omitted)
                - Same format options as in_rep
        
        Examples:
            # Fundamental representation (default)
            constraint = O5ConstraintsVec(input_size=5, output_size=1)
            
            # Dual representation V* -> Scalar
            constraint = O5ConstraintsVec(in_rep='V*', out_rep='scalar')
            
            # Mixed tensor V⊗V* -> Scalar
            constraint = O5ConstraintsVec(in_rep='V⊗V*', out_rep='scalar')
            
            # Dual squared (V*)² -> Scalar
            constraint = O5ConstraintsVec(in_rep='(V*)²', out_rep='scalar')
            
            # V⊗(V*)² -> Scalar
            constraint = O5ConstraintsVec(in_rep='V*V**2', out_rep='scalar')
            
            # Equivariant: V -> V
            constraint = O5ConstraintsVec(in_rep='V', out_rep='V')
            
            # Custom representation
            constraint = O5ConstraintsVec(
                in_rep=lambda G: V(G) * V(G).T * V(G).T,  # V⊗(V*)²
                out_rep='scalar'
            )
        """
        # Import representation utilities
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from utils.representation_utils import get_lie_algebra_rep, get_representation_size
        
        # Store representation types for later use
        self.in_rep = in_rep if in_rep is not None else 'fundamental'
        self.out_rep = out_rep
        
        # Compute input size from representation if in_rep is specified
        if in_rep is not None:
            computed_input_size = get_representation_size(
                representation_type=in_rep if isinstance(in_rep, str) else None,
                representation_builder=in_rep if callable(in_rep) else None
            )
            self.input_size = computed_input_size
        else:
            # Allow any input_size when no specific representation is given
            # This handles cases like multi-channel representations (10 channels × 5D = 50D)
            self.input_size = input_size
        
        # Compute output size from representation if out_rep is specified
        computed_output_size = get_representation_size(
                representation_type=out_rep if isinstance(out_rep, str) else None,
                representation_builder=out_rep if callable(out_rep) else None
            )
        if out_rep is not None:
            # If user provided output_size, validate it matches the representation
            if output_size is not None and output_size != computed_output_size:
                raise ValueError(
                    f"Provided output_size ({output_size}) does not match computed size "
                    f"from out_rep '{out_rep}' ({computed_output_size})"
                )
            self.output_size = computed_output_size
        else:
            self.output_size = computed_output_size
        
        self.decomposition_method = decomposition_method
        self.use_reflection = use_reflection
        
        # Relaxed validation: allow any output_size when using custom representations
        # Legacy behavior: validate only if using default fundamental representation
        if self.output_size is not None:
            if in_rep is None and out_rep is None:
                # Legacy validation for default fundamental representation
                if self.output_size != self.input_size and self.output_size not in [1, None]:
                    raise ValueError(
                        f"O(5) output_size must be 1 (invariant/scalars), {self.input_size} (equivariant/V), "
                        f"or None (same as input). Got {self.output_size}. "
                        f"For tensor representations, specify out_rep parameter (e.g., 'V*V' for 25D)"
                    )
        
        # Initialize basis storage (lists for multiple generators)
        self.invariant_basis = None  # List of bases, one per generator
        self.invariant_scaling_values = None  # List of scaling values, one per generator
        self.equivariant_basis = None
        self.equivariant_scaling_values = None
        
        # Create group and get generators using representation utilities
        self.create_unit_group_action_matrix()
        self.create_unit_group_action_forward_difference_matrix()
    
    def create_unit_group_action_matrix(self):
        """Create group action matrices from EMLP O(5) group."""
        # Create O(5) group using EMLP
        G = O(5)
        
        # Get all Lie algebra generators (for SO(5))
        # SO(5) has 10 generators (5*4/2 = 10 for antisymmetric matrices)
        lie_algebra = G.lie_algebra  # Shape: (10, 5, 5) - 10 generators
        
        # Store all generators as a list
        self.unit_group_action_matrices = [
            torch.FloatTensor(np.array(gen)) for gen in lie_algebra
        ]  # List of 10 (5, 5) tensors
        
        # Keep first generator for backward compatibility (deprecated)
        self.unit_group_action_matrix = self.unit_group_action_matrices[0]
        
        # If using reflection, store reflection generator
        if self.use_reflection and len(G.discrete_generators) > 0:
            self.reflection_matrix = torch.FloatTensor(np.array(G.discrete_generators[0]))
        else:
            self.reflection_matrix = None
        
        # For output: only create if output_size matches input (equivariance)
        # For output_size=1 or None (invariance), we don't need output group action
        if self.output_size == self.input_size:
            self.unit_group_action_matrices_out = self.unit_group_action_matrices
            self.unit_group_action_matrix_out = self.unit_group_action_matrix
        else:
            # For invariance (scalar output), no group action on output
            self.unit_group_action_matrices_out = None
            self.unit_group_action_matrix_out = None
    
    def create_unit_group_action_forward_difference_matrix(self):
        """
        Create forward difference matrices using representation utilities.
        Supports fundamental, tensor products, and other EMLP representations.
        """
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from utils.representation_utils import get_lie_algebra_rep
        
        # Extract Lie algebra for input representation
        # This automatically handles tensor products, direct sums, etc.
        self.forward_difference_matrices = get_lie_algebra_rep(
            representation_type=self.in_rep if isinstance(self.in_rep, str) else None,
            representation_builder=self.in_rep if callable(self.in_rep) else None,
            group_name='O(5)',
            include_discrete=self.use_reflection
        )
        
        # Extract Lie algebra for output representation (if different)
        if self.out_rep is not None and self.out_rep != self.in_rep:
            self.forward_difference_matrices_out = get_lie_algebra_rep(
                representation_type=self.out_rep if isinstance(self.out_rep, str) else None,
                representation_builder=self.out_rep if callable(self.out_rep) else None,
                group_name='O(5)',
                include_discrete=self.use_reflection
            )
        else:
            # Same representation for output
            self.forward_difference_matrices_out = self.forward_difference_matrices
    
    def apply_unit_group_action(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply O(5) group action to vectors."""
        original_shape = vectors.shape
        
        if len(original_shape) == 2:
            rotated = vectors @ self.unit_group_action_matrix.T
        elif len(original_shape) == 3:
            rotated = vectors @ self.unit_group_action_matrix.T
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        return rotated
    
    def apply_forward_difference(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Apply forward difference to vectors using all generators.
        
        For multi-generator groups, this applies the first generator by default.
        For full equivariance checking, use apply_forward_difference_all().
        """
        original_shape = vectors.shape
        
        # Use first generator for compatibility
        matrix = self.forward_difference_matrices[0]
        
        if len(original_shape) == 2:
            result = vectors @ matrix.T
        elif len(original_shape) == 3:
            result = vectors @ matrix.T
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        return result
    
    def get_decomposition(self, rep: str = 'input', method: str = None, return_original: bool = False):
        """
        Get decomposition of input or output representation.
        
        Returns a list of decompositions, one for each generator.
        Each element is (basis_l, values, basis_r).
        """
        if rep == 'input':
            matrices = self.forward_difference_matrices
        elif rep == 'output':
            if self.forward_difference_matrices_out is None:
                raise ValueError("Output decomposition not available for invariant layers (output_size != 5)")
            matrices = self.forward_difference_matrices_out
        else:
            raise ValueError(f"rep must be 'input' or 'output', got '{rep}'")
        
        decomp_method = method if method is not None else self.decomposition_method
        
        decompositions = []
        for i, matrix in enumerate(matrices):
            if decomp_method == 'schur':
                s_values, right = schur_decomposition(matrix, return_original=return_original)
                left = right
            elif decomp_method == 'svd':
                left, s_values, right = svd_decomposition(matrix)
            else:
                raise ValueError(f"Decomposition method {decomp_method} not supported")
            
            decompositions.append((left, s_values, right))
        
        return decompositions
    
    def create_invariant_basis(self):
        """Compute invariant basis using joint decomposition of all generators.
        
        For multi-generator groups like O(5), we stack all forward difference matrices
        and decompose jointly to find the common invariant subspace.
        
        Following o5_filter.py logic:
        - Stack horizontally: [D1, D2, ..., Dn] → (d, d*n)
        - SVD gives U as left basis (used for projecting weights)
        - Stack vertically: [D1; D2; ...; Dn] → (d*n, d)
        - SVD gives V as right basis
        """
        with torch.no_grad():
            # Stack all forward difference matrices horizontally: (d, d*n)
            stacked_horizontal = torch.cat(self.forward_difference_matrices, dim=1)
            
            # Stack all forward difference matrices vertically: (d*n, d)
            stacked_vertical = torch.cat(self.forward_difference_matrices, dim=0)
            
            if self.decomposition_method == 'schur':
                # For non-square matrices, schur doesn't apply directly
                # Fall back to SVD-like behavior
                T_l, Z_l = schur_decomposition(stacked_horizontal @ stacked_horizontal.T)
                self.invariant_basis = Z_l
                self.invariant_scaling_values = T_l
            elif self.decomposition_method == 'svd':
                # SVD on horizontal stack: get left singular vectors U as basis
                U_l, S_l, V_l = svd_decomposition(stacked_horizontal)
                
                # SVD on vertical stack: get right singular vectors V as basis
                U_r, S_r, V_r = svd_decomposition(stacked_vertical)
                
                # Store basis and combined scaling values
                # For ELinear invariant mode, we use the left basis (like o5_filter)
                self.invariant_basis = U_l
                
                # Combine singular values from both decompositions
                min_len = min(len(S_l), len(S_r))
                self.invariant_scaling_values = S_l[:min_len] + S_r[:min_len]
            else:
                raise ValueError(f"Decomposition method {self.decomposition_method} not supported")
            
            # Ensure correct device/dtype
            ref_matrix = self.forward_difference_matrices[0]
            self.invariant_basis = self.invariant_basis.to(ref_matrix.device, dtype=ref_matrix.dtype)
            self.invariant_scaling_values = self.invariant_scaling_values.to(ref_matrix.device, dtype=ref_matrix.dtype)
    
    def get_invariant_basis(self, soft_thresholding: float = 0.0, tolerance: float = 1e-4):
        """
        Return invariant basis with optional soft thresholding.
        
        For multi-generator groups, returns the joint invariant basis from
        the combined decomposition of all generators.
        
        Returns:
            (basis, scaling_values) tuple
        """
        if self.invariant_basis is None:
            self.create_invariant_basis()
        
        return self._select_basis(
            basis=self.invariant_basis,
            scaling_values=self.invariant_scaling_values,
            soft_thresholding=soft_thresholding,
            tolerance=tolerance,
            label="O5_invariant"
        )
    
    def get_equivariant_condition_matrix(self, diagonal_only=False):
        matrices = []
        if diagonal_only:
            # Only use diagonal pairs: same generator index for input and output
            for i, matrix in enumerate(self.forward_difference_matrices):
                if i < len(self.forward_difference_matrices_out):  # Ensure we don't go out of bounds
                    out_matrix = self.forward_difference_matrices_out[i]
                    condition_matrix = torch.kron(
                        matrix.transpose(0, 1).contiguous(),
                        torch.eye(
                            self.output_size,
                            device=matrix.device,
                            dtype=matrix.dtype
                        )
                    ) - torch.kron(
                        torch.eye(
                            matrix.shape[0],
                            device=matrix.device,
                            dtype=matrix.dtype
                        ),
                        out_matrix.contiguous()
                    )
                    matrices.append(condition_matrix)
        else:
            # Original quadratic approach: all combinations
            for i, matrix in enumerate(self.forward_difference_matrices):
                for j, out_matrix in enumerate(self.forward_difference_matrices_out):
                    condition_matrix = torch.kron(
                        matrix.transpose(0, 1).contiguous(),
                        torch.eye(
                            self.output_size,
                            device=matrix.device,
                            dtype=matrix.dtype
                        )
                    ) - torch.kron(
                        torch.eye(
                            matrix.shape[0],
                            device=matrix.device,
                            dtype=matrix.dtype
                        ),
                        self.forward_difference_matrices_out[j].contiguous()
                    )
                    matrices.append(condition_matrix)
        return matrices
    
    def create_equivariant_basis(self):
        """
        Compute equivariant basis for O(5) using all generators.
        
        The equivariant basis is the null space of the joint condition matrix
        built from all generator pairs. This ensures equivariance under all
        group transformations, not just one.
        
        Following o5_filter.py logic:
        - Stack condition matrices vertically: [C1; C2; ...; Cn]
        - The null space is in right singular vectors V corresponding to zero singular values
        """
        if self.output_size is None:
            raise ValueError("Equivariant basis requires output_size to be specified")
        if self.forward_difference_matrices_out is None:
            raise ValueError("Equivariant basis requires output representation (output_size must match a valid representation)")
        
        with torch.no_grad():
            # Build condition matrices for all generator pairs
            condition_matrices = self.get_equivariant_condition_matrix(diagonal_only=True)
            
            # Stack all condition matrices vertically to form a joint condition
            # The null space of this stacked matrix is equivariant under ALL generators
            joint_condition = torch.cat(condition_matrices, dim=0)
            
            # SVD: A = U @ S @ V^T
            # Null space of A (where A @ x = 0) is in right singular vectors V
            # corresponding to zero singular values
            U, S, V = svd_decomposition(joint_condition)
            
            # Store device/dtype from first forward difference matrix
            ref_matrix = self.forward_difference_matrices[0]
            self.equivariant_basis = V.to(ref_matrix.device, dtype=ref_matrix.dtype)
            self.equivariant_scaling_values = S.to(ref_matrix.device, dtype=ref_matrix.dtype)
    
    def get_equivariant_basis(self, soft_thresholding: float = 0.0, tolerance: float = 1e-4):
        """Return equivariant basis with optional soft thresholding."""
        if self.equivariant_basis is None:
            self.create_equivariant_basis()
        
        return self._select_basis(
            basis=self.equivariant_basis,
            scaling_values=self.equivariant_scaling_values,
            soft_thresholding=soft_thresholding,
            tolerance=tolerance,
            label="O5_equivariant"
        )
    
    def _select_basis(
        self,
        basis: torch.Tensor,
        scaling_values: torch.Tensor,
        soft_thresholding: float,
        tolerance: float,
        label: str
    ):
        """Select basis vectors based on scaling values and soft thresholding.
        
        The basis columns are ordered by ascending singular values. We select
        columns corresponding to near-zero singular values (the null space),
        plus optionally some fraction of the remaining columns (soft thresholding).
        """
        zero_idx = torch.where(torch.abs(scaling_values) <= tolerance)[0]
        
        if zero_idx.numel() > 0:
            # last_zero_index is the index of the last near-zero value
            # To include it in the slice [:threshold], we need threshold = index + 1
            num_zero = int(zero_idx[-1]) + 1
        else:
            num_zero = 0
        
        if soft_thresholding > 0:
            remaining = scaling_values.shape[0] - num_zero
            extra = int(soft_thresholding * remaining)
            threshold = num_zero + extra
        else:
            threshold = num_zero
        
        return basis[:, :threshold].clone().detach(), scaling_values[:threshold].clone().detach()

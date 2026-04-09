"""
Lorentz O(1,3) constraints for 4D vectors using EMLP library.
Implements Lorentz-invariant and equivariant basis computation for vector data.
"""

import torch
import numpy as np
import sys
import os

from softeq.utils.decompositions import schur_decomposition, svd_decomposition


class LorentzConstraintsVec:
    """
    Lorentz (O(1,3)) constraints for 4D vectors using EMLP library.

    Supports representation-aware sizing (fundamental, tensor products, sums, dual, etc.)
    via utils/representation_utils.get_lie_algebra_rep.

    Args:
        input_size (int): Dimension of input vectors (default: 4 for Lorentz)
        output_size (int or None): Dimension of output:
            - None or 1: invariant layers (maps to scalars)
            - input_size: equivariant layers (maps 4D vectors to 4D vectors)
        decomposition_method (str): Method for decomposition ('svd' or 'schur').
        use_reflection (bool): Include discrete generators (full Lorentz group).
        in_rep (str or callable, optional): Input representation type or builder.
            Supported string formats:
                * 'fundamental' or 'V': Fundamental representation (4D)
                * 'V*' or 'dual': Dual representation (4D)
                * 'V*V' or 'V⊗V': Tensor product V⊗V (16D)
                * 'V*V*' or 'V⊗V*': Mixed tensor V⊗V* (16D)
                * 'V**V*' or 'V*⊗V*': Dual tensor (V*)⊗(V*) (16D)
                * 'V**2' or 'V²': Symmetric square V² (16D)
                * 'V***2' or '(V*)²': Dual squared (V*)² (16D)
                * 'V*V*V': Triple product V⊗V⊗V (64D)
                * 'V*V**2' or 'V⊗(V*)²': V⊗(V*)² (64D)
                * 'V**2*V*' or '(V*)²⊗V': (V*)²⊗V (64D)
                * 'V+V' or 'V⊕V': Direct sum V⊕V (8D)
                * 'V+V*' or 'V⊕V*': Direct sum V⊕V* (8D)
                * 'T(p,q)': Rank (p,q) tensor (e.g., 'T(2,1)' = V⊗V⊗V*)
                * 'scalar' or 'S': Scalar/trivial representation (1D)
            Or callable: custom representation builder
        out_rep (str or callable, optional): Output representation type or builder.
            Same format options as in_rep.
    
    Examples:
        # Fundamental Lorentz representation
        constraint = LorentzConstraintsVec(input_size=4, output_size=1)
        
        # Dual representation V* -> Scalar
        constraint = LorentzConstraintsVec(in_rep='V*', out_rep='scalar')
        
        # Mixed tensor V⊗V* -> Scalar (16D -> 1D)
        constraint = LorentzConstraintsVec(in_rep='V⊗V*', out_rep='scalar')
        
        # Dual squared (V*)² -> Scalar (16D -> 1D)
        constraint = LorentzConstraintsVec(in_rep='(V*)²', out_rep='scalar')
        
        # Equivariant: V -> V (4D -> 4D)
        constraint = LorentzConstraintsVec(in_rep='V', out_rep='V')
    """

    def __init__(
        self,
        input_size: int = 4,
        output_size: int = None,
        decomposition_method: str = 'svd',
        use_reflection: bool = True,
        in_rep=None,
        out_rep=None
    ):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        try:
            from utils.representation_utils import get_lie_algebra_rep, get_representation_size
        except Exception:
            get_lie_algebra_rep = None
            get_representation_size = None

        self._get_lie_algebra_rep = get_lie_algebra_rep
        self._get_representation_size = get_representation_size
        self.in_rep = in_rep if in_rep is not None else 'fundamental'
        self.out_rep = out_rep

        if in_rep is not None and self._get_representation_size is not None:
            computed_input_size = get_representation_size(
                representation_type=in_rep if isinstance(in_rep, str) else None,
                representation_builder=in_rep if callable(in_rep) else None,
                group_name='Lorentz'
            )
            self.input_size = computed_input_size
        else:
            self.input_size = input_size

        if out_rep is not None and self._get_representation_size is not None:
            computed_output_size = get_representation_size(
                representation_type=out_rep if isinstance(out_rep, str) else None,
                representation_builder=out_rep if callable(out_rep) else None,
                group_name='Lorentz'
            )
            if output_size is not None and output_size != computed_output_size:
                raise ValueError(
                    f"Provided output_size ({output_size}) does not match computed size "
                    f"from out_rep '{out_rep}' ({computed_output_size})"
                )
            self.output_size = computed_output_size
        else:
            self.output_size = output_size

        self.decomposition_method = decomposition_method
        self.use_reflection = use_reflection

        # Legacy validation for default fundamental representation
        if self.output_size is not None:
            if in_rep is None and out_rep is None:
                if self.output_size != self.input_size and self.output_size not in [1, None]:
                    raise ValueError(
                        f"Lorentz output_size must be 1 (invariant/scalars), {self.input_size} (equivariant/V), "
                        f"or None (same as input). Got {self.output_size}. "
                        f"For tensor representations, specify out_rep parameter."
                    )

        self.invariant_basis = None
        self.invariant_scaling_values = None
        self.equivariant_basis = None
        self.equivariant_scaling_values = None

        self.create_unit_group_action_forward_difference_matrix()

    def create_unit_group_action_forward_difference_matrix(self):
        """Create forward difference matrices using representation utilities."""
        if self._get_lie_algebra_rep is not None:
            self.forward_difference_matrices = self._get_lie_algebra_rep(
                representation_type=self.in_rep if isinstance(self.in_rep, str) else None,
                representation_builder=self.in_rep if callable(self.in_rep) else None,
                group_name='Lorentz',
                include_discrete=self.use_reflection
            )

            if self.out_rep is not None and self.out_rep != self.in_rep:
                self.forward_difference_matrices_out = self._get_lie_algebra_rep(
                    representation_type=self.out_rep if isinstance(self.out_rep, str) else None,
                    representation_builder=self.out_rep if callable(self.out_rep) else None,
                    group_name='Lorentz',
                    include_discrete=self.use_reflection
                )
            else:
                self.forward_difference_matrices_out = self.forward_difference_matrices
        else:
            # Fallback: only support fundamental representation (Vector) without emlp.reps
            if self.in_rep not in [None, 'fundamental', 'V', 'fund']:
                raise ValueError("LorentzConstraintsVec fallback supports only fundamental representation.")
            # Add external emlp to path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'equivariant-MLP'))
            from emlp.groups import Lorentz
            G = Lorentz()
            self.forward_difference_matrices = [
                torch.FloatTensor(np.array(gen)) for gen in G.lie_algebra
            ]
            if self.use_reflection and len(G.discrete_generators) > 0:
                self.forward_difference_matrices.append(
                    torch.FloatTensor(np.array(G.discrete_generators[0]) - np.eye(4))
                )
            self.forward_difference_matrices_out = self.forward_difference_matrices

    def apply_unit_group_action(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply Lorentz group action to vectors using first generator."""
        original_shape = vectors.shape
        matrix = self.forward_difference_matrices[0]
        
        if len(original_shape) == 2:
            rotated = vectors @ matrix.T
        elif len(original_shape) == 3:
            rotated = vectors @ matrix.T
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        return rotated

    def apply_forward_difference(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Apply forward difference to vectors using first generator.
        
        For multi-generator groups, this applies the first generator by default.
        """
        original_shape = vectors.shape
        matrix = self.forward_difference_matrices[0]
        
        if len(original_shape) == 2:
            result = vectors @ matrix.T
        elif len(original_shape) == 3:
            result = vectors @ matrix.T
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        return result

    def get_decomposition(self, rep: str = 'input', method: str = None, return_original: bool = False):
        if rep == 'input':
            matrices = self.forward_difference_matrices
        elif rep == 'output':
            if self.forward_difference_matrices_out is None:
                raise ValueError("Output decomposition not available for invariant layers (output_size != input)")
            matrices = self.forward_difference_matrices_out
        else:
            raise ValueError(f"rep must be 'input' or 'output', got '{rep}'")

        decomp_method = method if method is not None else self.decomposition_method

        decompositions = []
        for matrix in matrices:
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
        
        For multi-generator groups like Lorentz, we stack all forward difference matrices
        and decompose jointly to find the common invariant subspace.
        
        Following the filter code logic:
        - Stack horizontally: [D1, D2, ..., Dn] → (d, d*n)
        - SVD gives U as left basis (used for projecting weights)
        """
        with torch.no_grad():
            # Stack all forward difference matrices horizontally: (d, d*n)
            stacked_horizontal = torch.cat(self.forward_difference_matrices, dim=1)
            
            # Stack all forward difference matrices vertically: (d*n, d)
            stacked_vertical = torch.cat(self.forward_difference_matrices, dim=0)
            
            if self.decomposition_method == 'schur':
                T_l, Z_l = schur_decomposition(stacked_horizontal @ stacked_horizontal.T)
                self.invariant_basis = Z_l
                self.invariant_scaling_values = T_l
            elif self.decomposition_method == 'svd':
                # SVD on horizontal stack: get left singular vectors U as basis
                U_l, S_l, V_l = svd_decomposition(stacked_horizontal)
                
                # SVD on vertical stack: get right singular vectors V as basis
                U_r, S_r, V_r = svd_decomposition(stacked_vertical)
                
                # Store basis and combined scaling values
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
        """Return invariant basis with optional soft thresholding."""
        if self.invariant_basis is None:
            self.create_invariant_basis()
        
        return self._select_basis(
            basis=self.invariant_basis,
            scaling_values=self.invariant_scaling_values,
            soft_thresholding=soft_thresholding,
            tolerance=tolerance,
            label="Lorentz_invariant"
        )

    def get_equivariant_condition_matrix(self, diagonal_only=False):
        matrices = []
        if diagonal_only:
            for i, matrix in enumerate(self.forward_difference_matrices):
                if i < len(self.forward_difference_matrices_out):
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
            for matrix in self.forward_difference_matrices:
                for out_matrix in self.forward_difference_matrices_out:
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
        return matrices

    def create_equivariant_basis(self):
        """
        Compute equivariant basis for Lorentz group using all generators.
        
        The equivariant basis is the null space of the joint condition matrix
        built from all generator pairs.
        
        Following the filter code logic:
        - Stack condition matrices vertically: [C1; C2; ...; Cn]
        - The null space is in right singular vectors V corresponding to zero singular values
        """
        if self.output_size is None:
            raise ValueError("Equivariant basis requires output_size to be specified")
        if self.forward_difference_matrices_out is None:
            raise ValueError("Equivariant basis requires output representation")
        
        with torch.no_grad():
            # Build condition matrices for all generator pairs
            condition_matrices = self.get_equivariant_condition_matrix(diagonal_only=True)
            
            # Stack all condition matrices vertically
            joint_condition = torch.cat(condition_matrices, dim=0)
            
            # SVD: A = U @ S @ V^T
            # Null space of A (where A @ x = 0) is in right singular vectors V
            U, S, V = svd_decomposition(joint_condition)
            
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
            label="Lorentz_equivariant"
        )

    def _select_basis(
        self,
        basis: torch.Tensor,
        scaling_values: torch.Tensor,
        soft_thresholding: float,
        tolerance: float,
        label: str
    ):
        """Select basis vectors based on scaling values and soft thresholding."""
        zero_idx = torch.where(torch.abs(scaling_values) <= tolerance)[0]
        if zero_idx.numel() > 0:
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

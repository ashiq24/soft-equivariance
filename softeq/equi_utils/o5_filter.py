"""Soft O(5) invariant and equivariant filters built
Project MLP weights onto subspace with bounded equivariance error under O(5) transformations.
"""

from .o5_constraints_vec import O5ConstraintsVec
from .equi_projectors import EquivariantProjectorviaSchur, MultiGeneratorEquivariantProjector, MultiGeneratorEquivariantProjectorviaSVD, EquivariantProjectorviaSVD
from .inv_projector import InvariantProjector, MultiGeneratorInvariantProjector
from softeq.utils.decompositions import svd_decomposition
import torch

def get_invariant_filter_o5(
    input_size: int = 5,
    soft_threshold: float = 0.0,
    decomposition_method: str = 'svd',
    debug: bool = False,
    hard: bool = True,
    hard_mask: bool = False,
    preserve_norm: bool = False,
    use_reflection: bool = True,
    joint_decomposition: bool = True,
    in_rep: str = 'V',
):
    """Create an O(5) invariant weight projector.

    The filter enforces soft ``f(Rx) = f(x)`` by projecting weights onto the common
    null space of the O(5) forward-difference matrices with softness/approximation controled by `soft_threshold`. If ``joint_decomposition``
    is True, the generators are concatenated and reduced with a single SVD; if
    False, each generator is decomposed separately and the resulting projectors
    are chained.

    Args:
        input_size: Representation size or default vector size.
        soft_threshold: Parameter (between 0 and 1) controlling soft interpolation between strict and relaxed projection. 0 = perfect invariance, 1= no projection.
        decomposition_method: Decomposition backend for the constraint object.
        debug: Enable verbose projector diagnostics.
        hard: Enable hard masking when the spectrum has no exact null space.
        hard_mask: Enable hard cuff off, else an exponential decay based masked is used. 
        preserve_norm: Renormalize projected weights to match the original norm.
        use_reflection: Include the discrete reflection generator in O(5).
        joint_decomposition: If True, build one joint projector from stacked
            generators; otherwise chain per-generator projectors.
        in_rep: Input representation descriptor or custom builder.

    Returns:
        A projector module implementing the invariant filter.
    """
    # Create O(5) constraints
    constraints = O5ConstraintsVec(
        input_size=input_size,
        output_size=None,  # None for invariance (only input constraints needed)
        decomposition_method=decomposition_method,
        use_reflection=use_reflection,
        in_rep=in_rep,
    )
    
    if not joint_decomposition:
        
        decompositions = constraints.get_decomposition(
            rep='input', 
            method=decomposition_method, 
            return_original=False
        )
        
        invariant_projectors = []
        for i, (basis_l, values, basis_r) in enumerate(decompositions):
            projector = InvariantProjector(
                basis_l, basis_r, values,
                softness=soft_threshold,
                debug=debug,
                hard=hard,
                hard_mask=hard_mask,
                preserve_norm=preserve_norm
            )
            invariant_projectors.append(projector)
        
        filter = MultiGeneratorInvariantProjector(invariant_projectors)
        return filter
    
    else:
        # Joint decomposition: concatenate all forward difference matrices
        # Get all forward difference matrices (list of (5, 5) tensors)
        forward_diff_matrices = constraints.forward_difference_matrices
        
        if debug:
            print(f"O5 Invariant: Concatenating {len(forward_diff_matrices)} forward difference matrices")
        
        # Concatenate horizontally: [D1, D2, ..., Dn] → (5, 5n)
        stacked_horizontal = torch.cat(forward_diff_matrices, dim=1)  # (5, 5*n_generators)
        
        # Concatenate vertically: [D1; D2; ...; Dn] → (5n, 5)
        stacked_vertical = torch.cat(forward_diff_matrices, dim=0)  # (5*n_generators, 5)
        
        if debug:
            print(f"  Horizontal stack shape: {stacked_horizontal.shape}")
            print(f"  Vertical stack shape: {stacked_vertical.shape}")
        
        # SVD on horizontal: get left singular vectors as basis_l
        U_l, S_l, V_l = svd_decomposition(stacked_horizontal)
        basis_l = U_l  # (5, 5)
        
        # SVD on vertical: get right singular vectors as basis_r
        U_r, S_r, V_r = svd_decomposition(stacked_vertical)
        basis_r = V_r  # (5, 5)
        
        # Combine singular values
        min_len = min(len(S_l), len(S_r))
        values = S_l[:min_len] + S_r[:min_len]
        
        if debug:
            print(f"  basis_l shape: {basis_l.shape}")
            print(f"  basis_r shape: {basis_r.shape}")
            print(f"  combined values shape: {values.shape}")
        
        # Create single unified projector
        filter = InvariantProjector(
            basis_l, basis_r, values,
            softness=soft_threshold,
            debug=debug,
            hard=hard,
            hard_mask=hard_mask,
            preserve_norm=preserve_norm
        )
        
        return filter
        
        
        


def get_equivariant_filter_o5(
    input_size: int = 5,
    output_size: int = 5,
    soft_threshold: float = 0.0,
    debug: bool = False,
    use_reflection: bool = True,
    use_invariant_filter: bool = False,
    hard: bool = False,
    hard_mask: bool = False,
    in_rep: str = 'V',
    out_rep: str = 'V'
):
    """Create an O(5) equivariant weight projector.

    The filter enforces ``f(Rx) = Rf(x)`` by projecting the weight tensor into
    the equivariant subspace. When ``use_invariant_filter`` is True, the code
    uses an SVD-based null-space formulation over the stacked condition matrix;
    otherwise it falls back to generator-wise Schur projectors which is more efficient for high-dimensional representations such as T(4,4).

    Args:
        input_size: Input representation size or default vector size.
        output_size: Output representation size or default vector size.
        soft_threshold: Parameter (between 0 and 1) controlling soft interpolation between strict and relaxed projection. 0 = perfect equivariance, 1 = no projection.
        debug: Enable verbose projector diagnostics.
        use_reflection: Include the discrete reflection generator in O(5).
        use_invariant_filter: Use the SVD-based null-space formulation.
        hard: Enable hard masking when using the invariant-filter path.
        hard_mask: Enable hard cuff off, else an exponential decay based masked is used.
        in_rep: Input representation descriptor or custom builder.
        out_rep: Output representation descriptor or custom builder.

    Returns:
        A projector module implementing the equivariant filter.
    """
    constraints = O5ConstraintsVec(
        input_size=input_size,
        output_size=output_size,
        decomposition_method='svd', 
        use_reflection=use_reflection,
        in_rep=in_rep,
        out_rep=out_rep
    )
    
    
    if use_invariant_filter:
        # Explicitly use the SVD-based null-space path to build the equivariant filter.
        condition_list = constraints.get_equivariant_condition_matrix(diagonal_only=True)
        # Joint approach: concatenate all condition matrices
        if debug:
            print(f"O5 Equivariant: Using diagonal-only generator pairing approach")
            print(f"Number of condition matrices: {len(condition_list)}")
        
        # Stack all condition matrices vertically: [C1; C2; ...; Cn] → (n*outshape, inshape)
        stacked_condition = torch.cat(condition_list, dim=0)  # Stack along rows
        
        if debug:
            print(f"  Stacked condition matrix shape: {stacked_condition.shape}")
        
        # Find the null space of the stacked condition matrix using SVD
        # For equivariance: we want vectors w such that stacked_condition @ w = 0
        U, S, V = svd_decomposition(stacked_condition)
        
        if debug:
            print(f"  SVD shapes: U={U.shape}, S={S.shape}, V={V.shape}")
            print(f"  Singular values: {S}")
        
        # Find null space: singular values close to zero give us the null space
        # The right singular vectors (V) corresponding to small singular values form the null space
        tolerance = 1e-5
        zero_indices = torch.where(S < tolerance)[0]
        
        if debug:
            print(f"  Zero singular values indices: {zero_indices}")
            print(f"  Number of null space vectors: {len(zero_indices)}")
        
        if len(zero_indices) == 0:
            # No exact null space found, use soft thresholding
            # Take vectors corresponding to smallest singular values
            num_keep = int(V.shape[1] * (1 - soft_threshold))
            if debug:
                print(f"  Using soft thresholding: keeping {num_keep} vectors out of {V.shape[1]}")
            null_space_basis = V[:, -num_keep:]  # Take last num_keep vectors (smallest singular values)
            projection_matrix = null_space_basis @ null_space_basis.T
        else:
            # Use exact null space
            null_space_basis = V[:, zero_indices]  # Null space vectors
            if debug:
                print(f"  Null space basis shape: {null_space_basis.shape}")
            # Create projection matrix onto null space
            projection_matrix = null_space_basis @ null_space_basis.T
        
        # Get shapes for multi-channel handling
        in_shape = input_size  # 5 for O(5)
        out_shape = output_size  # 5 for O(5)
        
        inv_filter = InvariantProjector(V, V, S[:V.shape[1]], softness=soft_threshold, hard_mask=hard_mask, hard=hard, debug=debug)
        
        filter = MultiGeneratorEquivariantProjectorviaSVD(inv_filter, in_shape, out_shape, soft_threshold)
        if debug:
            print(f"  Projection matrix shape: {projection_matrix.shape}")
            print(f"  Input shape per channel: {in_shape}")
            print(f"  Output shape per channel: {out_shape}")
        
        return filter
    else:
        # This uses Schur decomposition-based approach to create separate projectors for each generator
        # efficient to compute for larger representations such as V*V*V*V (T(4)) 
        decompositions_input = constraints.get_decomposition(
            rep='input', 
            method='schur', 
            return_original=True
        )
        decompositions_output = constraints.get_decomposition(
            rep='output', 
            method='schur', 
            return_original=True
        )
        
        equivariant_projectors = []        
        # Create projector for each combination of input and output decompositions
        for i, ((left_x, sigma_x, right_x), (left_y, sigma_y, right_y)) in enumerate(zip(decompositions_input, decompositions_output)):
            print(f"Creating Equivariant Projector for generator  {i}")
            projector = EquivariantProjectorviaSchur(
                left_y, left_x, sigma_y, sigma_x,
                softness=soft_threshold,
                debug=debug,
                temperature= 0.01 
            )
            equivariant_projectors.append(projector)
        
        filter = MultiGeneratorEquivariantProjector(equivariant_projectors)
        return filter


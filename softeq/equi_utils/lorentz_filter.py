"""Soft Lorentz (O(1,3)) invariant and equivariant filters.

These helpers mirror the O(5) pipeline but build their constraints from
``LorentzConstraintsVec`` and the Lorentz representation utilities.
"""
from .lorentz_constraints_vec import LorentzConstraintsVec
from .equi_projectors import (
    EquivariantProjectorviaSchur,
    MultiGeneratorEquivariantProjector,
    MultiGeneratorEquivariantProjectorviaSVD,
)
from .inv_projector import InvariantProjector, MultiGeneratorInvariantProjector
from softeq.utils.decompositions import svd_decomposition
import torch


def get_invariant_filter_lorentz(
    input_size: int = 4,
    soft_threshold: float = 0.0,
    decomposition_method: str = 'svd',
    debug: bool = False,
    hard: bool = False,
    hard_mask: bool = False,
    preserve_norm: bool = False,
    use_reflection: bool = True,
    joint_decomposition: bool = True,
    in_rep: str = 'V',
):
    """Create a soft Lorentz invariant weight projector i.e., approximately maintain f(Rx) = f(x).

    Args:
        input_size: Input representation size or default vector size.
        soft_threshold: Parameter (between 0 and 1) controlling soft interpolation between strict and relaxed projection. 0 = perfect invariance, 1 = no projection.
        decomposition_method: Decomposition backend for the constraint object.
        debug: Enable verbose projector diagnostics.
        hard: Enable hard masking when the spectrum has no exact null space.
        hard_mask: Enable hard cuff off, else an exponential decay based masked is used.
        preserve_norm: Renormalize projected weights to match the original norm.
        use_reflection: Include discrete Lorentz generators in the constraint set.
        joint_decomposition: If True, build one joint projector from stacked
            generators; otherwise chain per-generator projectors.
        in_rep: Input representation descriptor or custom builder.

    Returns:
        A projector module implementing the invariant filter.
    """
    constraints = LorentzConstraintsVec(
        input_size=input_size,
        output_size=None,
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
        for basis_l, values, basis_r in decompositions:
            projector = InvariantProjector(
                basis_l,
                basis_r,
                values,
                softness=soft_threshold,
                debug=debug,
                hard=hard,
                hard_mask=hard_mask,
                preserve_norm=preserve_norm
            )
            invariant_projectors.append(projector)
        return MultiGeneratorInvariantProjector(invariant_projectors)

    forward_diff_matrices = constraints.forward_difference_matrices
    stacked_horizontal = torch.cat(forward_diff_matrices, dim=1)
    stacked_vertical = torch.cat(forward_diff_matrices, dim=0)

    U_l, S_l, _ = svd_decomposition(stacked_horizontal)
    U_r, S_r, V_r = svd_decomposition(stacked_vertical)
    basis_l = U_l
    basis_r = V_r
    min_len = min(len(S_l), len(S_r))
    values = S_l[:min_len] + S_r[:min_len]

    return InvariantProjector(
        basis_l,
        basis_r,
        values,
        softness=soft_threshold,
        debug=debug,
        hard=hard,
        hard_mask=hard_mask,
        preserve_norm=preserve_norm
    )


def get_equivariant_filter_lorentz(
    input_size: int = 4,
    output_size: int = 4,
    soft_threshold: float = 0.0,
    debug: bool = False,
    use_reflection: bool = True,
    use_invariant_filter: bool = True,
    hard: bool = False,
    hard_mask: bool = False,
    in_rep: str = 'V',
    out_rep: str = 'V'
):
    """Create a Lorentz equivariant weight projector.

    Args:
        input_size: Input representation size or default vector size.
        output_size: Output representation size or default vector size.
        soft_threshold: Parameter (between 0 and 1) controlling soft interpolation between strict and relaxed projection. 0 = perfect equivariance, 1 = no projection.
        debug: Enable verbose projector diagnostics.
        use_reflection: Include discrete Lorentz generators in the constraint set.
        use_invariant_filter: If True, use the SVD-based null-space formulation;
            otherwise use the generator-wise Schur projectors.
        hard: Enable hard masking in the invariant-filter path.
        hard_mask: Enable hard cuff off, else an exponential decay based masked is used.
        in_rep: Input representation descriptor or custom builder.
        out_rep: Output representation descriptor or custom builder.

    Returns:
        A projector module implementing the equivariant filter.
    """
    constraints = LorentzConstraintsVec(
        input_size=input_size,
        output_size=output_size,
        decomposition_method='schur',
        use_reflection=use_reflection,
        in_rep=in_rep,
        out_rep=out_rep
    )

    if use_invariant_filter:
        condition_list = constraints.get_equivariant_condition_matrix(diagonal_only=True)
        stacked_condition = torch.cat(condition_list, dim=0)
        U, S, V = svd_decomposition(stacked_condition)
        tolerance = 1e-5
        zero_indices = torch.where(S < tolerance)[0]

        if len(zero_indices) == 0:
            num_keep = int(V.shape[1] * (1 - soft_threshold))
            null_space_basis = V[:, -num_keep:]
            projection_matrix = null_space_basis @ null_space_basis.T
        else:
            null_space_basis = V[:, zero_indices]
            projection_matrix = null_space_basis @ null_space_basis.T

        inv_filter = InvariantProjector(
            V,
            V,
            S[:V.shape[1]],
            softness=soft_threshold,
            hard_mask=hard_mask,
            hard=hard,
            debug=debug
        )
        in_shape = input_size
        out_shape = output_size
        filter = MultiGeneratorEquivariantProjectorviaSVD(inv_filter, in_shape, out_shape, soft_threshold)
        if debug:
            print(f"Lorentz Equivariant: stacked condition shape {stacked_condition.shape}")
            print(f"Projection matrix shape: {projection_matrix.shape}")
        return filter

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
    for left_x, sigma_x, _ in decompositions_input:
        for left_y, sigma_y, _ in decompositions_output:
            projector = EquivariantProjectorviaSchur(
                left_y,
                left_x,
                sigma_y,
                sigma_x,
                softness=soft_threshold,
                debug=debug
            )
            equivariant_projectors.append(projector)

    return MultiGeneratorEquivariantProjector(equivariant_projectors)

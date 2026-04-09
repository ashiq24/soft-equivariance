"""
EMLP Representation Utilities

Modular functions for extracting Lie algebras from EMLP representations.
Supports fundamental, tensor products, direct sums, and custom representations.
"""

import torch
import jax.numpy as jnp
import sys
import os
from typing import List, Optional, Callable

# Try importing EMLP from installed package, fall back to external/ if not available
try:
    from emlp.groups import O, Lorentz
    from emlp.reps import V, Scalar, T
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'equivariant-MLP'))
    from emlp.groups import O, Lorentz
    from emlp.reps import V, Scalar, T


def get_lie_algebra_rep(
    representation_type: Optional[str] = None,
    representation_builder: Optional[Callable] = None,
    group_name: str = 'O(5)',
    include_discrete: bool = True
) -> List[torch.Tensor]:
    """
    Extract Lie algebra generators for any EMLP representation.
    
    EMLP automatically handles representation composition:
    - Tensor products (V⊗V): Uses Kronecker sum for drho
    - Direct sums (V⊕V): Uses block diagonal for drho
    - Dual representations (V*): Transforms as -drho(A).T
    - Custom representations: Supported via representation_builder
    
    Args:
        representation_type (str, optional): Built-in representation type:
            Fundamental:
              - 'fundamental' or 'V': Fundamental representation
              - 'scalar' or 'S': Trivial/scalar representation
            
            Dual representations:
              - 'V*' or 'dual': Dual representation V*
              
            Tensor products:
              - 'V*V' or 'V⊗V': Tensor product V⊗V (25D for O(5), 16D for Lorentz)
              - 'V*V*V': Triple tensor product V⊗V⊗V
              - 'V*V*' or 'V⊗V*': Mixed tensor V⊗V* (25D)
              - 'V**V*' or 'V*⊗V*': Dual tensor (V*)⊗(V*)
              
            Symmetric/antisymmetric squares:
              - 'V**2' or 'V²': Symmetric square V²
              - 'V***2' or '(V*)²': Dual symmetric square (V*)²
              
            Complex combinations:
              - 'V*V**2' or 'V⊗(V*)²': V⊗(V*)²
              - 'V**2*V*' or '(V*)²⊗V': (V*)²⊗V
              
            Direct sums:
              - 'V+V' or 'V⊕V': Direct sum V⊕V (10D for O(5))
              - 'V+V*' or 'V⊕V*': Direct sum V⊕V*
              
            General tensors:
              - 'T(p,q)': Rank (p,q) tensor with p contravariant and q covariant indices
                Example: 'T(2,1)' = V⊗V⊗V* (rank (2,1) tensor)
            
            If None and representation_builder is None, defaults to 'fundamental'
        
        representation_builder (callable, optional): Custom function that takes group G
            and returns an EMLP representation. Example:
            lambda G: V(G) * V(G).T  # For V⊗V*
            lambda G: V(G).T * V(G).T * V(G).T  # For (V*)³
        
        group_name (str): Name of the group ('O(5)', 'SO(5)', 'Lorentz', 'O(3)', etc.)
        
        include_discrete (bool): Include discrete generators (reflection, etc.)
            Default: True
    
    Returns:
        List[torch.Tensor]: List of forward difference matrices (Lie algebra generators)
                           as PyTorch tensors
    
    Examples:
        # Fundamental representation (5D for O(5), 4D for Lorentz)
        lie_alg_v = get_lie_algebra_rep('fundamental', group_name='O(5)')
        # Returns: 11 tensors × (5,5) each [10 SO(5) + 1 reflection]
        
        # Dual representation (5D for O(5))
        lie_alg_v_dual = get_lie_algebra_rep('V*', group_name='O(5)')
        # Returns: 11 tensors × (5,5) each, transforming as -A.T
        
        # Tensor product V⊗V (25D for O(5))
        lie_alg_vv = get_lie_algebra_rep('V*V', group_name='O(5)')
        # Returns: 11 tensors × (25,25) each [Kronecker sum automatically applied]
        
        # Mixed tensor V⊗V* (25D for O(5))
        lie_alg_vv_dual = get_lie_algebra_rep('V⊗V*', group_name='O(5)')
        # Returns: 11 tensors × (25,25) each
        
        # Dual squared (V*)² (25D for O(5))
        lie_alg_v_dual_sq = get_lie_algebra_rep('V***2', group_name='O(5)')
        # Returns: 11 tensors × (25,25) each
        
        # Complex combination V⊗(V*)² (125D for O(5))
        lie_alg_complex = get_lie_algebra_rep('V*V**2', group_name='O(5)')
        # Returns: 11 tensors × (125,125) each
        
        # Rank (2,1) tensor T(2,1) = V⊗V⊗V* (125D for O(5))
        lie_alg_t21 = get_lie_algebra_rep('T(2,1)', group_name='O(5)')
        # Returns: 11 tensors × (125,125) each
        
        # Lorentz group dual representation
        lie_alg_lorentz_dual = get_lie_algebra_rep('V*', group_name='Lorentz')
        # Returns: 7 tensors × (4,4) each [6 Lorentz + 1 parity]
        
        # Custom representation
        def custom_rep(G):
            return V(G) * V(G).T * V(G).T  # V⊗(V*)²
        lie_alg_custom = get_lie_algebra_rep(representation_builder=custom_rep)
        # Returns: 11 tensors × (125,125) each
    """
    
    # Create group
    group_key = group_name.strip().lower()
    if group_key in ['o(5)', 'so(5)']:
        G = O(5)
    elif group_key in ['lorentz', 'o(1,3)', 'o13']:
        G = Lorentz()
    else:
        raise ValueError(
            f"Group {group_name} not yet supported. Use 'O(5)', 'SO(5)', or 'Lorentz'."
        )
    
    # Build representation
    if representation_builder is not None:
        rep = representation_builder(G)
    else:
        rep = _build_representation(representation_type or 'fundamental', G)
    
    # Extract Lie algebra
    lie_algebra = []
    
    # Continuous generators (Lie algebra)
    for i, A in enumerate(G.lie_algebra):
        # EMLP automatically computes correct drho for any representation:
        # - Fundamental V: drho(A) = A
        # - Tensor product V⊗V: drho(A) = A⊗I + I⊗A (Kronecker sum)
        # - Direct sum V⊕V: drho(A) = diag(A, A) (block diagonal)
        drho_A = rep.drho(A)
        
        # Convert lazy operator to dense matrix
        drho_A_dense = jnp.array(drho_A @ jnp.eye(rep.size()))
        
        # Convert to torch tensor
        lie_algebra.append(torch.FloatTensor(drho_A_dense))
    
    # Discrete generators (reflection, etc.)
    if include_discrete and len(G.discrete_generators) > 0:
        for h in G.discrete_generators:
            # Compute rho(h) for this representation
            rho_h = rep.rho(h)
            
            # Convert lazy operator to dense matrix
            rho_h_dense = jnp.array(rho_h @ jnp.eye(rep.size()))
            rho_h_torch = torch.FloatTensor(rho_h_dense)
            
            # Forward difference: (h - I)
            forward_diff_h = rho_h_torch - torch.eye(rep.size())
            lie_algebra.append(forward_diff_h)
    
    return lie_algebra


def _build_representation(rep_type: str, G):
    """
    Build EMLP representation from type string using EMLP's native parsing.
    
    EMLP natively supports Python expressions with V, Scalar, and T objects:
    - 'V': Fundamental representation
    - 'V.T' or 'V*': Dual representation  
    - 'V*V' or 'V**2': Tensor products (V⊗V)
    - 'V.T*V.T' or 'V.T**2': Dual products ((V*)⊗(V*))
    - 'V*V.T': Mixed tensor (V⊗V*)
    - 'V+V': Direct sum (V⊕V)
    - 'T(p)' or 'T(p,q)': Rank (p,q) tensors
    
    This function uses Python's eval() with a safe namespace containing
    EMLP objects, which is the approach used in EMLP's own documentation
    and tutorials. This is more robust than custom parsing.

    Returns:
        An EMLP representation object with a ``size()`` method and ``drho``/
        ``rho`` actions.
    """
    rep_type = rep_type.strip()
    
    # Handle common shortcuts for backward compatibility
    if rep_type in ['fundamental', 'fund']:
        rep_type = 'V'
    elif rep_type in ['scalar', 'trivial', 'S']:
        rep_type = 'Scalar'
    elif rep_type in ['dual', 'V*']:  # Handle V* as dual
        rep_type = 'V.T'
    
    # Convert mathematical notation to Python syntax
    import re
    # Replace Unicode symbols with Python operators
    rep_type = rep_type.replace('⊗', '*')  # Tensor product
    rep_type = rep_type.replace('⊕', '+')  # Direct sum
    rep_type = rep_type.replace('²', '**2')  # Squared
    rep_type = rep_type.replace('³', '**3')  # Cubed
    
    # Convert our old notation to EMLP's standard notation
    # Our parser used V* for dual, but EMLP prefers V.T
    # However, V* in multiplication context works (V*V is fine)
    
    # Handle V***N notation (dual power) by converting to V.T**N
    # V***2 -> V.T**2 (dual squared)
    rep_type = re.sub(r'V\*\*\*(\d+)', r'V.T**\1', rep_type)
    # V* at end of expression -> V.T (handles V+V*, V*V*V*, etc.)
    rep_type = re.sub(r'V\*($|[+)])', r'V.T\1', rep_type)
    # V*V* pattern at end -> V*V.T (mixed tensor)
    # Already handled by previous pattern
    
    # Create safe namespace with EMLP objects
    namespace = {
        'V': V(G),
        'Scalar': Scalar,
        'T': lambda p, q=0: T(p, q, G=G),
        '__builtins__': {},  # Restrict builtins for safety
    }
    
    try:
        # Use EMLP's native parsing via eval
        rep = eval(rep_type, namespace)
        return rep
    except Exception as e:
        raise ValueError(f"Failed to parse representation '{rep_type}': {e}\n"
                        f"Supported syntax: V, V.T (dual), V*V, V**2, V+V, T(p), T(p,q)")


def _parse_tensor_product(expr: str, G):
    """
    DEPRECATED: This function is kept for backward compatibility only.
    New code should use _build_representation() which leverages EMLP's native parsing.
    
    Parse tensor product expression with careful handling of **, ***, etc.
    
    Strategy: Parse the string character by character, building up terms.
    * between V's is a tensor product operator
    ** followed by a digit is an exponent
    *** followed by a digit means dual-squared
    
    Examples:
        'V*V' → V(G) * V(G)  (V ⊗ V)
        'V*V*' → V(G) * V(G).T  (V ⊗ V*)
        'V*V*V' → V(G) * V(G) * V(G)  (V ⊗ V ⊗ V)
        'V*V*V*' → V(G) * V(G) * V(G) * V(G)  (V ⊗ V ⊗ V ⊗ V)
        'V**2' → V(G) * V(G)  (V²)
        'V***2' → V(G).T * V(G).T  ((V*)²)
        'V*V**2' → V(G) * V(G).T * V(G).T  (V ⊗ (V*)²)
    """
    expr = expr.strip()
    result = None
    i = 0
    
    while i < len(expr):
        if expr[i:i+1] == 'V':
            # Found a V, now check what follows
            i += 1
            
            # Check for ** or *** patterns (exponents)
            if i < len(expr) and expr[i:i+3] == '***':
                # V***N pattern: (V*)^N
                i += 3
                if i < len(expr) and expr[i].isdigit():
                    # Extract number
                    j = i
                    while j < len(expr) and expr[j].isdigit():
                        j += 1
                    power = int(expr[i:j])
                    base = V(G).T  # Dual
                    term = base
                    for _ in range(power - 1):
                        term = term * base
                    i = j
                else:
                    raise ValueError(f"Expected digit after *** at position {i}")
                    
            elif i < len(expr) and expr[i:i+2] == '**':
                # V**N pattern: V^N
                i += 2
                if i < len(expr) and expr[i].isdigit():
                    j = i
                    while j < len(expr) and expr[j].isdigit():
                        j += 1
                    power = int(expr[i:j])
                    base = V(G)
                    term = base
                    for _ in range(power - 1):
                        term = term * base
                    i = j
                else:
                    raise ValueError(f"Expected digit after ** at position {i}")
                    
            elif i < len(expr) and expr[i] == '*':
                # V* - could be dual, or could be V * (something)
                # Look ahead to see what comes next
                if i + 1 < len(expr) and expr[i+1] in ['V', 'S']:
                    # V*V or V*S - this is V (not dual) times next term
                    term = V(G)
                    i += 1  # consume the *, next iteration will handle V/S
                elif i + 1 < len(expr) and expr[i+1] == '*':
                    # V** - already handled above, shouldn't reach here
                    raise ValueError(f"Unexpected ** at position {i}")
                elif i + 1 >= len(expr) or expr[i+1] in ['+', '⊕']:
                    # V* at end or before direct sum - this is dual
                    term = V(G).T
                    i += 1
                else:
                    # Default: treat as multiplication operator, so just V
                    term = V(G)
                    i += 1
            else:
                # Just V with nothing after
                term = V(G)
                
        elif expr[i:i+1] == 'S':
            term = Scalar(G)
            i += 1
            # Skip any following *
            if i < len(expr) and expr[i] == '*':
                i += 1
                
        elif expr[i] in ['*', '⊗']:
            # Skip multiplication/tensor operators between terms
            i += 1
            continue
            
        elif expr[i] in ['+', '⊕']:
            # This shouldn't happen as direct sums are handled earlier
            raise ValueError(f"Unexpected + at position {i}")
            
        elif expr[i] == ' ':
            # Skip whitespace
            i += 1
            continue
            
        else:
            raise ValueError(f"Unexpected character '{expr[i]}' at position {i}")
        
        # Accumulate term
        result = term if result is None else result * term
    
    if result is None:
        raise ValueError(f"Could not parse representation: '{expr}'")
    
    return result


def get_representation_size(
    representation_type: Optional[str] = None,
    representation_builder: Optional[Callable] = None,
    group_name: str = 'O(5)'
) -> int:
    """
    Get dimension of a representation without extracting all generators.
    
    Args:
        representation_type (str, optional): Built-in representation type
        representation_builder (callable, optional): Custom representation builder
        group_name (str): Name of the group
    
    Returns:
        int: Dimension of the representation
    
    Examples:
        dim_v = get_representation_size('fundamental')  # 5
        dim_vv = get_representation_size('V*V')         # 25
        dim_vvv = get_representation_size('V*V*V')      # 125
    """
    group_key = group_name.strip().lower()
    if group_key in ['o(5)', 'so(5)']:
        G = O(5)
    elif group_key in ['lorentz', 'o(1,3)', 'o13']:
        G = Lorentz()
    else:
        raise ValueError(f"Group {group_name} not yet supported")
    
    if representation_builder is not None:
        rep = representation_builder(G)
    else:
        rep = _build_representation(representation_type or 'fundamental', G)
    
    return rep.size()


def get_num_generators(
    group_name: str = 'O(5)',
    include_discrete: bool = True
) -> int:
    """
    Get number of generators for a group.
    
    Args:
        group_name (str): Name of the group
        include_discrete (bool): Include discrete generators
    
    Returns:
        int: Number of generators
    
    Examples:
        num_gen = get_num_generators('O(5)')  # 11 (10 SO(5) + 1 reflection)
    """
    group_key = group_name.strip().lower()
    if group_key in ['o(5)', 'so(5)']:
        G = O(5)
    elif group_key in ['lorentz', 'o(1,3)', 'o13']:
        G = Lorentz()
    else:
        raise ValueError(f"Group {group_name} not yet supported")
    
    num_continuous = len(G.lie_algebra)
    num_discrete = len(G.discrete_generators) if include_discrete else 0
    
    return num_continuous + num_discrete


# Convenience functions for common use cases

def get_fundamental_lie_algebra(group_name: str = 'O(5)') -> List[torch.Tensor]:
    """Get Lie algebra for fundamental representation (V, 5D for O(5), 4D for Lorentz)."""
    return get_lie_algebra_rep('fundamental', group_name=group_name)


def get_dual_lie_algebra(group_name: str = 'O(5)') -> List[torch.Tensor]:
    """Get Lie algebra for dual representation (V*, same dimension as V)."""
    return get_lie_algebra_rep('V*', group_name=group_name)


def get_tensor_lie_algebra(group_name: str = 'O(5)') -> List[torch.Tensor]:
    """Get Lie algebra for tensor product representation (V⊗V, 25D for O(5))."""
    return get_lie_algebra_rep('V*V', group_name=group_name)


def get_mixed_tensor_lie_algebra(group_name: str = 'O(5)') -> List[torch.Tensor]:
    """Get Lie algebra for mixed tensor representation (V⊗V*, 25D for O(5))."""
    return get_lie_algebra_rep('V*V*', group_name=group_name)


def get_dual_squared_lie_algebra(group_name: str = 'O(5)') -> List[torch.Tensor]:
    """Get Lie algebra for dual squared representation ((V*)², 25D for O(5))."""
    return get_lie_algebra_rep('V***2', group_name=group_name)


def get_triple_tensor_lie_algebra(group_name: str = 'O(5)') -> List[torch.Tensor]:
    """Get Lie algebra for triple tensor product representation (V⊗V⊗V, 125D for O(5))."""
    return get_lie_algebra_rep('V*V*V', group_name=group_name)


def get_direct_sum_lie_algebra(group_name: str = 'O(5)') -> List[torch.Tensor]:
    """Get Lie algebra for direct sum representation (V⊕V, 10D for O(5))."""
    return get_lie_algebra_rep('V+V', group_name=group_name)


# Info functions for debugging and understanding

def print_representation_info(
    representation_type: Optional[str] = None,
    representation_builder: Optional[Callable] = None,
    group_name: str = 'O(5)'
) -> None:
    """
    Print detailed information about a representation.
    
    Args:
        representation_type (str, optional): Built-in representation type
        representation_builder (callable, optional): Custom representation builder
        group_name (str): Name of the group
    """
    size = get_representation_size(representation_type, representation_builder, group_name)
    num_gen = get_num_generators(group_name, include_discrete=True)
    
    rep_name = representation_type or "custom"
    if representation_builder is not None:
        rep_name = "custom"
    
    print(f"\nRepresentation Info:")
    print(f"  Group: {group_name}")
    print(f"  Representation: {rep_name}")
    print(f"  Dimension: {size}")
    print(f"  Lie algebra generators: {num_gen}")
    print(f"  Each generator shape: ({size}, {size})")
    print(f"  Total matrices: {num_gen} × ({size}, {size})")
    print()


if __name__ == "__main__":
    print("="*70)
    print("EMLP Representation Utils - Examples")
    print("="*70)
    
    # Example 1: Fundamental
    print("\n1. Fundamental Representation (V, 5D)")
    lie_alg = get_lie_algebra_rep('fundamental')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 2: Dual
    print("\n2. Dual Representation (V*, 5D)")
    lie_alg = get_lie_algebra_rep('V*')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 3: Tensor product
    print("\n3. Tensor Product (V⊗V, 25D)")
    lie_alg = get_lie_algebra_rep('V*V')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 4: Mixed tensor
    print("\n4. Mixed Tensor (V⊗V*, 25D)")
    lie_alg = get_lie_algebra_rep('V⊗V*')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 5: Dual squared
    print("\n5. Dual Squared ((V*)², 25D)")
    lie_alg = get_lie_algebra_rep('(V*)²')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 6: Complex combination
    print("\n6. Complex Combination (V⊗(V*)², 125D)")
    lie_alg = get_lie_algebra_rep('V*V**2')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 7: Rank (2,1) tensor
    print("\n7. Rank (2,1) Tensor T(2,1) = V⊗V⊗V* (125D)")
    lie_alg = get_lie_algebra_rep('T(2,1)')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 8: Lorentz group fundamental
    print("\n8. Lorentz Fundamental (V, 4D)")
    lie_alg = get_lie_algebra_rep('fundamental', group_name='Lorentz')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 9: Lorentz dual
    print("\n9. Lorentz Dual (V*, 4D)")
    lie_alg = get_lie_algebra_rep('V*', group_name='Lorentz')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 10: Lorentz mixed tensor
    print("\n10. Lorentz Mixed Tensor (V⊗V*, 16D)")
    lie_alg = get_lie_algebra_rep('V⊗V*', group_name='Lorentz')
    print(f"   Generators: {len(lie_alg)}, Shape: {lie_alg[0].shape}")
    
    # Example 11: Info
    print("\n11. Representation Info")
    print_representation_info('V*V**2', group_name='O(5)')
    
    print("="*70)

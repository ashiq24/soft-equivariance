"""
Equivariance constraints for 2D and 3D vectors.
Re-exports all constraint classes for backward compatibility.

This module re-exports:
- DiscreteRotationConstraintsVec: Rotation constraints for 2D/3D vectors
- DiscreteReflectionConstraintsVec: Reflection constraints for 2D/3D vectors
- O5ConstraintsVec: O(5) constraints using EMLP
- LorentzConstraintsVec: Lorentz O(1,3) constraints using EMLP
"""

from .rotation_constraints_vec import DiscreteRotationConstraintsVec
from .reflection_constraints_vec import DiscreteReflectionConstraintsVec
from .o5_constraints_vec import O5ConstraintsVec
from .lorentz_constraints_vec import LorentzConstraintsVec

__all__ = [
    'DiscreteRotationConstraintsVec',
    'DiscreteReflectionConstraintsVec',
    'O5ConstraintsVec',
    'LorentzConstraintsVec',
]

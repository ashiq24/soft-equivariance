"""
Models package for soft-equivariance project.
"""

__all__ = []

try:
    from models.test_models import TestModel, TestEqModel
    __all__ += ["TestModel", "TestEqModel"]
except Exception:
    pass

try:
    from models.get_model import get_model
    __all__ += ["get_model"]
except Exception:
    pass

try:
    from models.filtered_vit import FilteredViT, create_filtered_vit
    from models.vit_utils import (
        count_parameters,
        save_checkpoint,
        load_checkpoint,
    )
    __all__ += [
        "FilteredViT",
        "create_filtered_vit",
        "count_parameters",
        "save_checkpoint",
        "load_checkpoint",
    ]
except Exception:
    pass

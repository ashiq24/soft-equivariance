# import importlib
# import pkgutil
# __all__ = []
# for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
#     module = importlib.import_module('.'+module_name,package=__name__)
#     try: 
#         globals().update({k: getattr(module, k) for k in module.__all__})
#         __all__ += module.__all__
#     except AttributeError: continue
# # concatenate the __all__ from each of the submodules (expose to user)

__version__ = '1.0.3'

# Avoid hard dependency on jax/objax when only groups/reps are needed.
try:
    from .nn import *  # noqa: F401,F403
except Exception:
    # nn backends may require jax/objax; allow partial import.
    pass

try:
    from .groups import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .reps import *  # noqa: F401,F403
except Exception:
    pass

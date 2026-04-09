import os
import yaml
from typing import Optional


def load_config(path: str, config_name: Optional[str] = None) -> dict:
    """Load a YAML config file into a dictionary.

    Args:
        path: Path to YAML file.

    Returns:
        A nested dict with configuration parameters.
    """
    # Resolve absolute path
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    # Helper: deep merge
    def _deep_merge_dicts(a: dict, b: dict) -> dict:
        if a is None:
            a = {}
        if b is None:
            b = {}
        result = dict(a)
        for key, val in b.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = _deep_merge_dicts(result[key], val)
            else:
                result[key] = val
        return result

    # Load the YAML file once; it may contain multiple named configs
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}

    # Normalize common string representations of None (e.g. 'none', 'null', '~') into Python None
    def _normalize_none(obj):
        if isinstance(obj, dict):
            return {k: _normalize_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize_none(v) for v in obj]
        if isinstance(obj, str):
            s = obj.strip().lower()
            # None-like
            if s in {'none', 'null', '~'}:
                return None
            # Boolean-like
            if s in {'true', 'yes', 'on'}:
                return True
            if s in {'false', 'no', 'off'}:
                return False
            
            return obj
        return obj

    raw = _normalize_none(raw)

    # Determine whether the YAML is a single config or a collection of named configs
    def _is_single_config(obj: dict) -> bool:
        # Heuristic: if keys include expected top-level sections, treat as single config
        expected_top_keys = {'experiment', 'data', 'model', 'train', 'logging'}
        if not isinstance(obj, dict):
            return True
        return bool(expected_top_keys.intersection(set(obj.keys())))

    # Internal resolver which supports both file-based parents and same-file named parents
    def _resolve_cfg_from_obj(cfg_obj, containing_raw, seen, identifier):
        # cfg_obj: dict representing the child config
        if identifier in seen:
            raise ValueError(f"Cycle detected in config inheritance for id: {identifier}")
        seen.add(identifier)

        inherit_key = None
        if isinstance(cfg_obj, dict):
            if 'inherit_from' in cfg_obj:
                inherit_key = 'inherit_from'
            elif 'extends' in cfg_obj:
                inherit_key = 'extends'

        if inherit_key is None:
            return dict(cfg_obj)

        parents = cfg_obj.get(inherit_key)
        if parents is None:
            return dict(cfg_obj)

        if isinstance(parents, str):
            parents = [parents]
        if not isinstance(parents, list):
            raise ValueError(f"Invalid type for '{inherit_key}' in {identifier}: must be string or list of strings")

        base_cfg = {}
        for parent_spec in parents:
            # Parent spec may be:
            #  - a name referring to another top-level section in the same file
            #  - a relative/absolute path to another yaml file
            parent_cfg = None
            # Prefer same-file named parent if available
            if isinstance(parent_spec, str) and parent_spec in containing_raw:
                parent_obj = containing_raw[parent_spec]
                parent_id = f"{path}:{parent_spec}"
                parent_cfg = _resolve_cfg_from_obj(parent_obj, containing_raw, seen, parent_id)
            else:
                # Treat as filename (relative to current file)
                parent_path = parent_spec
                if not os.path.isabs(parent_path):
                    parent_path = os.path.join(os.path.dirname(path), parent_spec)
                parent_path = os.path.abspath(parent_path)
                if not os.path.isfile(parent_path):
                    raise FileNotFoundError(f"Parent config not found: {parent_path} referenced from {identifier}")
                # Load parent file and resolve (supporting named configs inside it as well)
                with open(parent_path, 'r') as pf:
                    parent_raw = yaml.safe_load(pf) or {}
                parent_raw = _normalize_none(parent_raw)

                # If parent_raw contains multiple named configs, caller may have specified name using 'name:config'
                # But here we assume the parent file contains a single config at top-level
                if _is_single_config(parent_raw):
                    parent_obj = parent_raw
                    parent_id = f"{parent_path}:"
                    parent_cfg = _resolve_cfg_from_obj(parent_obj, parent_raw, seen, parent_id)
                else:
                    # If parent file is multi-config, require that parent_spec refers to a named key inside it
                    # i.e., parent_spec format: "file.yaml:configname"
                    if ':' in parent_spec:
                        fname, cfgname = parent_spec.split(':', 1)
                        if not os.path.isabs(fname):
                            fname = os.path.join(os.path.dirname(path), fname)
                        fname = os.path.abspath(fname)
                        if not os.path.isfile(fname):
                            raise FileNotFoundError(f"Parent config file not found: {fname}")
                        with open(fname, 'r') as pf:
                            parent_raw2 = yaml.safe_load(pf) or {}
                        parent_raw2 = _normalize_none(parent_raw2)
                        if cfgname not in parent_raw2:
                            raise KeyError(f"Named parent config '{cfgname}' not found in {fname}")
                        parent_obj = parent_raw2[cfgname]
                        parent_id = f"{fname}:{cfgname}"
                        parent_cfg = _resolve_cfg_from_obj(parent_obj, parent_raw2, seen, parent_id)
                    else:
                        raise ValueError(f"Ambiguous parent spec '{parent_spec}' in {identifier}. Provide a named key or a path.")

            base_cfg = _deep_merge_dicts(base_cfg, parent_cfg)

        # Merge child over base
        child_copy = dict(cfg_obj)
        if inherit_key in child_copy:
            child_copy.pop(inherit_key)
        merged = _deep_merge_dicts(base_cfg, child_copy)
        return merged

    # If caller supplied a config_name, pick that named config from the file
    if config_name is not None:
        if not isinstance(raw, dict) or config_name not in raw:
            raise KeyError(f"Config name '{config_name}' not found in {path}")
        root_obj = raw[config_name]
        identifier = f"{path}:{config_name}"
        return _resolve_cfg_from_obj(root_obj, raw, seen=set(), identifier=identifier)

    # No config_name provided: if file looks like a single config, resolve it directly
    if _is_single_config(raw):
        return _resolve_cfg_from_obj(raw, raw, seen=set(), identifier=path)

    # File contains multiple named configs but caller didn't pick one
    raise KeyError(f"Config file {path} contains multiple named configurations; call load_config(path, config_name='name') to select one")


def get_default_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'default.yaml')


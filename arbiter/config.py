"""Load and merge arbiter configuration from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

_BUILTIN_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict:
    """Return the merged configuration dictionary.

    Always loads the built-in ``config.yaml`` first, then deep-merges the
    user-supplied *path* (if any) on top so that partial overrides work.
    """
    with open(_BUILTIN_CONFIG) as f:
        cfg = yaml.safe_load(f)

    if path is not None:
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, user_cfg)

    return cfg

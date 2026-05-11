"""Config loader.

YAML → nested dict with ``${ENV_VAR}`` interpolation. Same shape as the
Tennis Forecast sibling so utilities like the deploy script and the
trading-dashboard adapters read paths identically.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _interp_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _interp_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp_env(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = _REPO_ROOT / "config" / "config.yaml"
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _interp_env(raw)


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p

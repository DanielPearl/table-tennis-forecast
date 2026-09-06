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

import copy
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


_CFG_CACHE: dict = {}


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = _REPO_ROOT / "config" / "config.yaml"
    path = Path(path)
    # Parse-once cache keyed by mtime — several hot paths (predict,
    # signals) call load_config() per match, which added up to dozens
    # of full YAML parses per tick. A config edit still takes effect
    # immediately (mtime changes). Deep-copied on return so callers
    # that mutate their cfg can't poison the shared cache.
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = -1.0
    hit = _CFG_CACHE.get(path)
    if hit is not None and hit[0] == mtime:
        return copy.deepcopy(hit[1])
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _interp_env(raw)
    _CFG_CACHE[path] = (mtime, cfg)
    return copy.deepcopy(cfg)


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p

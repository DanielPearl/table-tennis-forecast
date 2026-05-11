"""Live-state reader. The live monitor writes ``data/raw/live_state.json``
each tick; this module just reads it. No demo fixture fallback — when
the file is missing we return an empty list and the dashboard renders
the empty state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("data.live")


def _state_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["raw_dir"]) / "live_state.json"


def load_live_state() -> list[dict[str, Any]]:
    fp = _state_path()
    if not fp.exists():
        log.info("live-state file %s missing — empty watchlist this tick", fp)
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("live-state file %s unreadable: %s", fp, exc)
        return []
    return data if isinstance(data, list) else []

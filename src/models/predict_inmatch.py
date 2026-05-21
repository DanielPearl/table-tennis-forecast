"""Inference wrapper for the table-tennis in-match adjustment model.

Loads the calibrated classifier produced by
``train_inmatch_model.py`` and exposes a single function that
turns a standardized live_record into a P(player_a wins match)
prediction.

The TT model was trained on per-set snapshots (the TTelite archive
only carries per-set point totals, not within-set point sequences).
At inference time the within-set fields the live feed provides
(point_streak, current_game_score, deuce/game-point flags) are
*not* model inputs — those continue to feed the rules layer.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..features.build_pbp_snapshots import FEATURE_COLUMNS
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("models.predict_inmatch")


def _artifact_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["artifacts_dir"]) / "inmatch_model.joblib"


@lru_cache(maxsize=1)
def _load_model():
    path = _artifact_path()
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to load %s: %s", path, exc)
        return None


def model_available() -> bool:
    return _load_model() is not None


def _live_to_features(rec: dict[str, Any]) -> np.ndarray:
    set_a = float(rec.get("set_score_a") or 0)
    set_b = float(rec.get("set_score_b") or 0)
    best_of = float(rec.get("best_of") or 5)
    pw_a = float(rec.get("point_win_pct_a_live")
                  if rec.get("point_win_pct_a_live") is not None else 0.5)
    pw_b = float(rec.get("point_win_pct_b_live")
                  if rec.get("point_win_pct_b_live") is not None else 0.5)
    g3_a = float(rec.get("games_won_last_3_a") or 0)
    g3_b = float(rec.get("games_won_last_3_b") or 0)
    is_closing = 1.0 if rec.get("is_closing_game") else 0.0
    is_decider = 1.0 if (set_a == best_of // 2 and set_b == best_of // 2) else 0.0
    current_set = float(rec.get("current_set") or (set_a + set_b + 1))
    progress = float(rec.get("progress")
                     or ((set_a + set_b) / best_of if best_of else 0.0))
    deuce_a = float(rec.get("deuce_games_a") or 0)
    deuce_b = float(rec.get("deuce_games_b") or 0)
    deuce_total = deuce_a + deuce_b
    first_set_winner_a = float(rec.get("first_set_winner_a") or 0)
    ran_a = float(rec.get("ran_up_a") or 0)
    ran_b = float(rec.get("ran_up_b") or 0)
    points_diff = pw_a - pw_b  # proxy when raw point totals aren't carried

    values: dict[str, float] = {
        "set_score_a": set_a,
        "set_score_b": set_b,
        "sets_diff": set_a - set_b,
        "best_of": best_of,
        "current_set": current_set,
        "is_closing_game": is_closing,
        "is_decider": is_decider,
        "progress": progress,
        "point_win_pct_a_live": pw_a,
        "point_win_pct_b_live": pw_b,
        "point_share_a": pw_a,
        "points_diff": points_diff,
        "games_won_last_3_a": g3_a,
        "games_won_last_3_b": g3_b,
        "deuce_games_a": deuce_a,
        "deuce_games_b": deuce_b,
        "deuce_share_a": (deuce_a / deuce_total) if deuce_total else 0.5,
        "first_set_winner_a": first_set_winner_a,
        "ran_up_a": ran_a,
        "ran_up_b": ran_b,
    }
    return np.array([values[c] for c in FEATURE_COLUMNS], dtype=float).reshape(1, -1)


def predict(live_record: dict[str, Any]) -> float | None:
    model = _load_model()
    if model is None:
        return None
    X = _live_to_features(live_record)
    p = model.predict_proba(X)[0, 1]
    return float(np.clip(p, 1e-3, 1 - 1e-3))

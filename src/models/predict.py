"""Pre-match inference for any (player_a, player_b, ...) tuple.

Loads the persisted model bundle and Elo state, builds the feature
vector for the matchup, and returns the calibrated win probability for
``player_a``. Used by both the dashboard and the live-monitor loop.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..features.elo import EloState, lookup_pair_features
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging
from .train_prematch_model import load_elo_state

log = setup_logging("models.predict")


_BUNDLE = None
_ELO: EloState | None = None
_H2H: dict | None = None
_LAST_MATCH: dict[str, pd.Timestamp] | None = None


def _artifacts_dir() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["artifacts_dir"])


def _ensure_loaded() -> None:
    global _BUNDLE, _ELO, _H2H, _LAST_MATCH
    if _BUNDLE is not None:
        return
    art = _artifacts_dir()
    bundle = joblib.load(art / "prematch_model.joblib")
    elo = load_elo_state(joblib.load(art / "elo_state.joblib"))
    h2h = joblib.load(art / "h2h_table.joblib")
    rest = joblib.load(art / "last_match_date.joblib")
    rest = {k: pd.Timestamp(v) for k, v in rest.items()}
    _BUNDLE = bundle
    _ELO = elo
    _H2H = h2h
    _LAST_MATCH = rest


def _h2h_diff(player_a: str, player_b: str) -> int:
    assert _H2H is not None
    key = tuple(sorted([player_a, player_b]))
    raw = _H2H.get(key, 0)
    return raw if key[0] == player_a else -raw


def _days_rest(player: str, ref: pd.Timestamp) -> float:
    assert _LAST_MATCH is not None
    last = _LAST_MATCH.get(player)
    if last is None:
        return 7.0
    delta = (ref - last).days
    return float(min(60, max(0, delta)))


_LEVEL_RANK = {"GS": 5, "CH": 4, "ST": 3, "FD": 2, "OT": 1}


def _level_rank(level: str) -> int:
    return _LEVEL_RANK.get((level or "OT").upper(), 1)


def _round_rank(r: str) -> int:
    table = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 8}
    return table.get((r or "").upper(), 0)


def predict_match(
    player_a: str,
    player_b: str,
    level: str = "ST",
    round_: str = "R32",
    best_of: int = 7,
    rank_a: float | None = None,
    rank_b: float | None = None,
    hand_a: str = "R",
    hand_b: str = "R",
    match_date: datetime | date | None = None,
) -> dict[str, Any]:
    """Return ``{prob_a, prob_b, elo_winprob_a, feats}`` for the matchup."""
    _ensure_loaded()
    assert _BUNDLE is not None and _ELO is not None
    bundle = _BUNDLE
    if match_date is None:
        match_date = datetime.utcnow()
    if isinstance(match_date, datetime):
        ref = pd.Timestamp(match_date.date())
    else:
        ref = pd.Timestamp(match_date)

    elo_feats = lookup_pair_features(_ELO, player_a, player_b, hand_a, hand_b)
    # All broad-panel features default to zero (== "no informative
    # difference") at inference unless we have a per-player history to
    # populate them from. The Elo features dominate the model anyway.
    feats = {
        "diff_elo_pre": elo_feats["elo_diff"],
        "diff_style_elo_pre": elo_feats["style_elo_diff"],
        "diff_form_last5": 0.0,
        "diff_form_last10": 0.0,
        "diff_form_last20": 0.0,
        "diff_avg_point_win_pct_10": 0.0,
        "diff_std_point_win_pct_10": 0.0,
        "diff_avg_game_margin_10": 0.0,
        "diff_std_game_margin_10": 0.0,
        "diff_closing_win_pct_10": 0.0,
        "diff_deuce_win_pct_10": 0.0,
        "diff_comeback_rate_20": 0.0,
        "diff_days_rest": _days_rest(player_a, ref) - _days_rest(player_b, ref),
        "diff_matches_last_7d": 0.0,
        "h2h_a_wins_minus_b_wins": float(_h2h_diff(player_a, player_b)),
        "h2h_a_wins_last5": 0.0,
        "rank_diff": float((rank_b or 500) - (rank_a or 500)),
        "level_rank": float(_level_rank(level)),
        "round_rank": float(_round_rank(round_)),
        "is_bo7": 1.0 if int(best_of) >= 7 else 0.0,
        "diff_hand_left": (1.0 if hand_a.upper() == "L" else 0.0)
                          - (1.0 if hand_b.upper() == "L" else 0.0),
        "hand_matchup_lr": 1.0 if hand_a.upper() != hand_b.upper() else 0.0,
    }
    # The bundle may have been trained on a pruned feature list — only
    # use the columns the bundle actually expects.
    feats_used = bundle["feature_list"]
    row = {k: feats.get(k, 0.0) for k in feats_used}
    X = pd.DataFrame([row])[feats_used].fillna(0.0)

    p_ens = float(bundle["ensemble"].predict_proba(X)[0, 1])
    p_log = float(bundle["logistic"].predict_proba(
        X[bundle["elo_only_features"]])[0, 1])
    blended = (
        bundle["blend_weight_ensemble"] * p_ens
        + bundle["blend_weight_logistic"] * p_log
    )
    blended = max(0.01, min(0.99, blended))
    return {
        "prob_a": blended,
        "prob_b": 1.0 - blended,
        "elo_winprob_a": elo_feats["elo_winprob_a"],
        "feats": feats,
        "elo": elo_feats,
    }


def predict_with_elo_only(player_a: str, player_b: str,
                            hand_a: str = "R", hand_b: str = "R"
                            ) -> dict[str, float]:
    """Fallback when the trained bundle isn't available yet."""
    _ensure_loaded() if (_BUNDLE is not None) else None
    if _ELO is None:
        return {"prob_a": 0.5, "prob_b": 0.5, "elo_winprob_a": 0.5}
    f = lookup_pair_features(_ELO, player_a, player_b, hand_a, hand_b)
    p = max(0.05, min(0.95, f["elo_winprob_a"]))
    return {"prob_a": p, "prob_b": 1.0 - p, "elo_winprob_a": f["elo_winprob_a"]}


def players_known(player_a: str, player_b: str) -> tuple[bool, bool]:
    """Return (a_known, b_known) — whether each player exists in the
    persisted Elo state. Used by the exporter to decide whether the
    model has any real opinion on a matchup or whether both sides
    are defaulting to the 1500 baseline (uninformative)."""
    try:
        _ensure_loaded()
    except Exception:
        return False, False
    if _ELO is None:
        return False, False
    return (player_a in _ELO.overall, player_b in _ELO.overall)


def safe_predict(*args, **kwargs) -> dict[str, Any]:
    try:
        return predict_match(*args, **kwargs)
    except Exception as exc:
        log.warning("predict_match failed (%s); falling back to Elo-only", exc)
        try:
            player_a, player_b = args[0], args[1]
            hand_a = kwargs.get("hand_a", "R")
            hand_b = kwargs.get("hand_b", "R")
            return predict_with_elo_only(player_a, player_b, hand_a, hand_b)
        except Exception:
            return {"prob_a": 0.5, "prob_b": 0.5, "elo_winprob_a": 0.5}

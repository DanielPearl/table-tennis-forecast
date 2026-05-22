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
_HISTORY: dict | None = None


def _artifacts_dir() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["artifacts_dir"])


def _ensure_loaded() -> None:
    global _BUNDLE, _ELO, _H2H, _LAST_MATCH, _HISTORY
    if _BUNDLE is not None:
        return
    art = _artifacts_dir()
    bundle = joblib.load(art / "prematch_model.joblib")
    elo = load_elo_state(joblib.load(art / "elo_state.joblib"))
    h2h = joblib.load(art / "h2h_table.joblib")
    rest = joblib.load(art / "last_match_date.joblib")
    rest = {k: pd.Timestamp(v) for k, v in rest.items()}
    # player_history is new — older artifact bundles won't have it.
    # Treat as missing and let the per-feature lookups fall back to
    # neutral priors so we don't crash on a stale install.
    hist_path = art / "player_history.joblib"
    history = joblib.load(hist_path) if hist_path.exists() else {}
    _BUNDLE = bundle
    _ELO = elo
    _H2H = h2h
    _LAST_MATCH = rest
    _HISTORY = history


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


def _avg(seq, default: float) -> float:
    if not seq:
        return default
    return float(sum(seq) / len(seq))


def _std(seq, default: float) -> float:
    if not seq or len(seq) < 2:
        return default
    m = sum(seq) / len(seq)
    var = sum((x - m) ** 2 for x in seq) / len(seq)
    return float(var ** 0.5)


def _ewm(seq, alpha: float = 0.4, default: float = 0.5) -> float:
    if not seq:
        return default
    vals = list(seq)
    w = [(1 - alpha) ** (len(vals) - 1 - i) for i in range(len(vals))]
    s = float(sum(w))
    return float(sum(v * wi for v, wi in zip(vals, w)) / s) if s > 0 else default


def _player_features(player: str, level: str, best_of: int) -> dict[str, float]:
    """Pull a player's last-known rolling stats from the persisted
    history snapshot. Falls back to neutral priors for unseen players
    so the diff features quietly read as zero."""
    h = _HISTORY or {}
    tier = (level or "OT").upper()

    def _g(bucket: str, default):
        return h.get(bucket, {}).get(player, default)

    bo_key = "bo7_form_10" if int(best_of) >= 7 else "bo5_form_10"
    tier_seq = h.get("tier_form_10", {}).get(f"{player}|{tier}", [])
    streak = float(h.get("win_streak", {}).get(player, 0))
    matches_total = float(h.get("matches_total", {}).get(player, 0))
    wins_total = float(h.get("wins_total", {}).get(player, 0))
    if matches_total > 0:
        career_wr = (wins_total + 2.0) / (matches_total + 4.0)
    else:
        career_wr = 0.5

    # Elo momentum: current Elo vs Elo from the start of the buffer.
    elo_hist = h.get("elo_hist", {}).get(player, [])
    cur_elo = (_ELO.get_overall(player) if _ELO is not None
               else 1500.0)
    momentum = (cur_elo - float(elo_hist[0])) if elo_hist else 0.0

    return {
        "form_last5": _avg(_g("win5", []), 0.5),
        "form_last10": _avg(_g("win10", []), 0.5),
        "form_last20": _avg(_g("win20", []), 0.5),
        "form_ewm10": _ewm(_g("win10", []), default=0.5),
        "avg_point_win_pct_10": _avg(_g("pointwin_10", []), 0.5),
        "std_point_win_pct_10": _std(_g("pointwin_10", []), 0.05),
        "avg_game_margin_10": _avg(_g("margin_10", []), 0.0),
        "std_game_margin_10": _std(_g("margin_10", []), 1.0),
        "closing_win_pct_10": _avg(_g("closing_10", []), 0.5),
        "deuce_win_pct_10": _avg(_g("deuce_10", []), 0.5),
        "comeback_rate_20": _avg(_g("comeback_20", []), 0.0),
        "opp_avg_elo_10": _avg(_g("opp_elo_10", []), 1500.0),
        "elo_momentum_10": momentum,
        "win_streak": max(-10.0, min(10.0, streak)),
        "career_win_pct": float(career_wr),
        "career_matches": matches_total,
        "tier_form_10": _avg(tier_seq, 0.5),
        "bo_form_10": _avg(_g(bo_key, []), 0.5),
    }


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
    # Per-player rolling state from the trained-state snapshot. Any
    # feature whose history isn't available falls back to a neutral
    # prior, matching the "default" used at training time.
    a_h = _player_features(player_a, level, int(best_of))
    b_h = _player_features(player_b, level, int(best_of))
    feats = {
        "diff_elo_pre": elo_feats["elo_diff"],
        "diff_style_elo_pre": elo_feats["style_elo_diff"],
        "diff_form_last5": a_h["form_last5"] - b_h["form_last5"],
        "diff_form_last10": a_h["form_last10"] - b_h["form_last10"],
        "diff_form_last20": a_h["form_last20"] - b_h["form_last20"],
        "diff_form_ewm10": a_h["form_ewm10"] - b_h["form_ewm10"],
        "diff_avg_point_win_pct_10": (a_h["avg_point_win_pct_10"]
                                       - b_h["avg_point_win_pct_10"]),
        "diff_std_point_win_pct_10": (a_h["std_point_win_pct_10"]
                                       - b_h["std_point_win_pct_10"]),
        "diff_avg_game_margin_10": (a_h["avg_game_margin_10"]
                                     - b_h["avg_game_margin_10"]),
        "diff_std_game_margin_10": (a_h["std_game_margin_10"]
                                     - b_h["std_game_margin_10"]),
        "diff_closing_win_pct_10": (a_h["closing_win_pct_10"]
                                     - b_h["closing_win_pct_10"]),
        "diff_deuce_win_pct_10": (a_h["deuce_win_pct_10"]
                                   - b_h["deuce_win_pct_10"]),
        "diff_comeback_rate_20": (a_h["comeback_rate_20"]
                                   - b_h["comeback_rate_20"]),
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
        # Strength-of-schedule, momentum, and specialization splits —
        # these are the features the new training pipeline added.
        "diff_opp_avg_elo_10": a_h["opp_avg_elo_10"] - b_h["opp_avg_elo_10"],
        "diff_elo_momentum_10": (a_h["elo_momentum_10"]
                                   - b_h["elo_momentum_10"]),
        "diff_win_streak": a_h["win_streak"] - b_h["win_streak"],
        "diff_career_win_pct": a_h["career_win_pct"] - b_h["career_win_pct"],
        "diff_career_matches": (a_h["career_matches"]
                                  - b_h["career_matches"]),
        "diff_tier_form_10": a_h["tier_form_10"] - b_h["tier_form_10"],
        "diff_bo_form_10": a_h["bo_form_10"] - b_h["bo_form_10"],
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

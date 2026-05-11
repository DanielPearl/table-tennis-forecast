"""Live-match feature standardization.

Same role as the Tennis sibling: take heterogeneous live-state dicts
from whatever provider feeds us (Kalshi market metadata, an ITTF live
feed, a fixture for tests) and produce a clean canonical dict the
rules engine and the dashboard both consume.

Table-tennis-specific fields:
  * set_score_a / set_score_b   — games won in the current match
  * current_game_score_a / b    — points within the current game
  * point_streak_a / b          — consecutive points the player has won
                                  inside the current game
  * is_deuce                    — current game is at 10-10 or higher
  * is_game_point_a / b         — one point from winning the current game
  * is_set_point_a / b          — one point from winning a critical game
  * is_match_point_a / b        — one point from winning the match
  * is_closing_game             — current game is the bo7/bo5 closer
  * serving_a                   — whose serve it currently is
  * point_win_pct_a_live / b    — running point % so far this match
"""
from __future__ import annotations

from typing import Any


_FIELDS_NUMERIC: list[str] = [
    "set_score_a", "set_score_b",
    "current_game_score_a", "current_game_score_b",
    "point_streak_a", "point_streak_b",
    "point_win_pct_a_live", "point_win_pct_b_live",
    "games_won_last_3_a", "games_won_last_3_b",
    "best_of",
    "market_prob_a", "market_prob_a_prev",
    "open_interest", "volume",
    "spread_cents",
]
_FIELDS_FLAGS: list[str] = [
    "is_deuce",
    "is_game_point_a", "is_game_point_b",
    "is_set_point_a", "is_set_point_b",
    "is_match_point_a", "is_match_point_b",
    "is_closing_game",
    "medical_timeout",
    "injury_news_flag", "retirement_risk_flag",
    "serving_a",
]


def standardize(record: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw live-state dict into the canonical schema.

    Missing numerics → None (downstream rules treat None as "not
    observed" and skip). Missing flags → False.
    """
    out: dict[str, Any] = {
        "match_id": str(record.get("match_id", "")),
        "tournament": record.get("tournament", "WTT"),
        # Table tennis is surface-invariant — we keep the field for
        # parity with the tennis schema so the dashboard renderers
        # don't have to special-case it, but it's always "Indoor".
        "surface": record.get("surface", "Indoor"),
        "player_a": record.get("player_a", ""),
        "player_b": record.get("player_b", ""),
    }
    for k in _FIELDS_NUMERIC:
        v = record.get(k)
        try:
            out[k] = float(v) if v is not None else None
        except (TypeError, ValueError):
            out[k] = None
    for k in _FIELDS_FLAGS:
        out[k] = bool(record.get(k, False))
    return out


def momentum_score(rec: dict[str, Any]) -> float:
    """A −1..+1 scalar for "who's surging right now" from player_a's POV.

    Built from set score, recent games delta, point streak, and live
    point-win % differential. Used by the rules engine to decide the
    direction of the in-match nudge.
    """
    a_sets = rec.get("set_score_a") or 0
    b_sets = rec.get("set_score_b") or 0
    a_g3 = rec.get("games_won_last_3_a") or 0
    b_g3 = rec.get("games_won_last_3_b") or 0
    a_streak = rec.get("point_streak_a") or 0
    b_streak = rec.get("point_streak_b") or 0
    pwa = rec.get("point_win_pct_a_live")
    pwb = rec.get("point_win_pct_b_live")

    set_term = (a_sets - b_sets) * 0.45
    games_term = (a_g3 - b_g3) * 0.15
    streak_term = (a_streak - b_streak) * 0.05
    if pwa is not None and pwb is not None:
        pw_term = (float(pwa) - float(pwb)) * 1.5
    else:
        pw_term = 0.0
    raw = set_term + games_term + streak_term + pw_term
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return float(raw)


def market_move(rec: dict[str, Any]) -> float | None:
    """Signed market move on player_a since the last snapshot.
    None if either price is missing."""
    cur = rec.get("market_prob_a")
    prev = rec.get("market_prob_a_prev")
    if cur is None or prev is None:
        return None
    return float(cur) - float(prev)


def volatility_signals(rec: dict[str, Any]) -> dict[str, bool]:
    """Convenience wrapper: report which volatility flags fire on this
    live record. Used by the rules engine to compose a volatility
    score and by the dashboard's reason-string builder."""
    return {
        "deuce": bool(rec.get("is_deuce")),
        "game_point": bool(rec.get("is_game_point_a") or rec.get("is_game_point_b")),
        "set_point": bool(rec.get("is_set_point_a") or rec.get("is_set_point_b")),
        "match_point": bool(rec.get("is_match_point_a") or rec.get("is_match_point_b")),
        "closing_game": bool(rec.get("is_closing_game")),
        "medical_timeout": bool(rec.get("medical_timeout")),
    }

"""Live-match adjustment layer for table tennis.

Same shape as the Tennis sibling: a transparent rules-based layer that
nudges the pre-match probability based on in-match state, with every
nudge logged into a ``reason`` string the dashboard surfaces.

Table-tennis-specific rules:
  1. Score-state momentum from set + games + point-win % deltas
  2. Point-streak nudge (4+ consecutive points within a game)
  3. Deuce / game-point / set-point / match-point volatility bumps
  4. Closing-game volatility (last game of bo5/bo7)
  5. Live point-win % differential — analogous to tennis serve %
  6. Comeback dampener: when a high-comeback-rate player trails, blunt
     the negative score-state nudge
  7. Market overreaction detector — market moved sharply but the model
     barely moved → fade the move
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..features.build_live_features import (
    market_move, momentum_score, volatility_signals,
)
from ..utils.config import load_config


@dataclass
class LiveAdjustment:
    live_prob_a: float
    pre_match_prob_a: float
    volatility_score: float
    injury_news_flag: bool
    market_overreaction: bool
    rules_fired: list[str] = field(default_factory=list)


def _clamp(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, p))


def adjust(pre_match_prob_a: float, live_record: dict[str, Any],
            player_a_comeback_rate: float = 0.0,
            player_b_comeback_rate: float = 0.0) -> LiveAdjustment:
    """Compute the live-adjusted probability for ``player_a``.

    ``player_a_comeback_rate`` / ``player_b_comeback_rate`` come from
    the rolling-form panel; they're 0..1 historical rates of "won the
    match after losing game 1". When the trailing player has a high
    rate, we blunt the negative nudge by ``comeback_dampener``.
    """
    cfg = load_config()
    rules = cfg["live_rules"]

    fired: list[str] = []
    delta = 0.0
    volatility = 0.0
    injury = bool(live_record.get("injury_news_flag", False)
                   or live_record.get("retirement_risk_flag", False))

    # ── 1) Score-state momentum ────────────────────────────────────────
    momentum = momentum_score(live_record)
    score_delta = max(
        -rules["max_in_match_shift"],
        min(rules["max_in_match_shift"],
            momentum * rules["max_in_match_shift"]),
    )

    # Comeback dampener — if the trailing side has a high comeback rate,
    # blunt the score-state nudge against them.
    a_sets = float(live_record.get("set_score_a") or 0)
    b_sets = float(live_record.get("set_score_b") or 0)
    if score_delta < 0 and a_sets < b_sets and player_a_comeback_rate > 0.30:
        dampener = float(rules["comeback_dampener"])
        score_delta *= (1.0 - dampener)
        fired.append(
            f"player_a trailing but {int(player_a_comeback_rate*100)}% "
            "historical comeback rate — nudge dampened"
        )
    elif score_delta > 0 and b_sets < a_sets and player_b_comeback_rate > 0.30:
        dampener = float(rules["comeback_dampener"])
        score_delta *= (1.0 - dampener)
        fired.append(
            f"player_b trailing but {int(player_b_comeback_rate*100)}% "
            "historical comeback rate — nudge dampened"
        )

    if abs(score_delta) > 0.005:
        delta += score_delta
        fired.append(
            f"score-state momentum {momentum:+.2f} → "
            f"{score_delta*100:+.1f}pp on player_a"
        )

    # ── 2) Point-streak nudge ───────────────────────────────────────────
    a_streak = float(live_record.get("point_streak_a") or 0)
    b_streak = float(live_record.get("point_streak_b") or 0)
    if a_streak >= rules["point_streak_threshold"]:
        bump = min(rules["point_streak_max_shift"],
                    rules["point_streak_weight"] * a_streak)
        delta += bump
        fired.append(f"player_a {int(a_streak)}-point streak → +{bump*100:.1f}pp")
    elif b_streak >= rules["point_streak_threshold"]:
        bump = min(rules["point_streak_max_shift"],
                    rules["point_streak_weight"] * b_streak)
        delta -= bump
        fired.append(f"player_b {int(b_streak)}-point streak → -{bump*100:.1f}pp")

    # ── 3) Live point-win % differential ────────────────────────────────
    pwa = live_record.get("point_win_pct_a_live")
    pwb = live_record.get("point_win_pct_b_live")
    if pwa is not None and pwb is not None:
        diff = float(pwa) - float(pwb)
        if abs(diff) > 0.04:
            pw_delta = diff * 0.20  # at most ~6pp
            delta += pw_delta
            if abs(pw_delta) > 0.005:
                fired.append(
                    f"live point-win % differential {diff*100:+.1f}pp "
                    f"→ {pw_delta*100:+.1f}pp on player_a"
                )

    # ── 4) Volatility scoring from in-match flags ───────────────────────
    flags = volatility_signals(live_record)
    if flags["deuce"]:
        volatility += rules["deuce_volatility_bump"]
        fired.append("deuce → volatility")
    if flags["game_point"]:
        volatility += rules["game_point_volatility_bump"]
        fired.append("game point → volatility")
    if flags["set_point"]:
        volatility += rules["set_point_volatility_bump"]
        fired.append("set point → volatility")
    if flags["match_point"]:
        volatility += rules["match_point_volatility_bump"]
        fired.append("match point → volatility")
    if flags["closing_game"]:
        volatility += rules["closing_game_volatility_bump"]
        fired.append("closing game → volatility")
    if flags["medical_timeout"]:
        volatility += 0.25
        injury = True
        fired.append("medical timeout → volatility + injury")
    # Tail volatility floor — even calm matches carry some uncertainty.
    volatility = min(1.0, max(0.05, volatility))

    live_prob_a = _clamp(pre_match_prob_a + delta)

    # ── 5) Market overreaction detection ───────────────────────────────
    move = market_move(live_record)
    overreaction = False
    if move is not None:
        if (abs(move) >= rules["overreaction_market_move"]
                and abs(delta) < rules["overreaction_model_move"]):
            overreaction = True
            fired.append(
                f"market moved {move*100:+.1f}pp but model only "
                f"{delta*100:+.1f}pp — possible overreaction"
            )

    return LiveAdjustment(
        live_prob_a=live_prob_a,
        pre_match_prob_a=pre_match_prob_a,
        volatility_score=volatility,
        injury_news_flag=injury,
        market_overreaction=overreaction,
        rules_fired=fired,
    )

"""Signal labelling for table-tennis matches.

Same priority order + same labels as the tennis sibling so the trading
dashboard's label pills and copy carry across identically:

  INJURY_RISK ▸ AVOID_VOLATILE ▸ MARKET_OVERREACTION
              ▸ STRONG_EDGE ▸ SMALL_EDGE ▸ WATCH ▸ NO_TRADE
"""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.config import load_config


@dataclass
class SignalResult:
    label: str
    reason: str
    confidence_score: float


def _confidence(model_prob: float, volatility: float) -> float:
    extremity_penalty = 0.0
    if model_prob > 0.92 or model_prob < 0.08:
        extremity_penalty = 0.20
    base = 0.85 - 0.55 * volatility - extremity_penalty
    return max(0.0, min(1.0, base))


def label_match(model_prob_a: float, market_prob_a: float | None,
                volatility: float, injury_flag: bool,
                market_overreaction: bool,
                rules_fired: list[str] | None = None) -> SignalResult:
    cfg = load_config()
    t = cfg["trading"]
    rules_fired = rules_fired or []

    if market_prob_a is None:
        edge_signed = 0.0
    else:
        edge_signed = float(model_prob_a) - float(market_prob_a)
    edge_abs = abs(edge_signed)
    side = "player_a" if edge_signed >= 0 else "player_b"
    side_market = (market_prob_a if edge_signed >= 0
                    else (1.0 - market_prob_a if market_prob_a is not None
                          else None))

    if injury_flag:
        return SignalResult(
            label="INJURY_RISK",
            reason="injury / medical risk flagged — skip until resolved",
            confidence_score=_confidence(model_prob_a, volatility) * 0.5,
        )
    if (market_overreaction and side_market is not None
            and edge_abs >= t["small_edge_min"]):
        if t["min_market_prob"] <= side_market <= t["max_market_prob"]:
            reason = "; ".join(rules_fired) or "market move outpaces model adjustment"
            return SignalResult(
                label="MARKET_OVERREACTION",
                reason=reason,
                confidence_score=_confidence(model_prob_a, volatility),
            )
    if volatility >= t["max_tradable_volatility"]:
        return SignalResult(
            label="AVOID_VOLATILE",
            reason="volatility above tradeable cap — wait for the rally to settle",
            confidence_score=_confidence(model_prob_a, volatility),
        )
    if market_prob_a is None:
        return SignalResult(
            label="WATCH",
            reason="no market price observed — model-only forecast",
            confidence_score=_confidence(model_prob_a, volatility) * 0.7,
        )
    if not (t["min_market_prob"] <= side_market <= t["max_market_prob"]):
        return SignalResult(
            label="NO_TRADE",
            reason=f"market price on {side} ({side_market:.0%}) outside tradeable band",
            confidence_score=_confidence(model_prob_a, volatility),
        )
    if edge_abs >= t["strong_edge_min"]:
        return SignalResult(
            label="STRONG_EDGE",
            reason=f"model {edge_signed*100:+.1f}pp vs market on {side}",
            confidence_score=_confidence(model_prob_a, volatility),
        )
    if edge_abs >= t["small_edge_min"]:
        return SignalResult(
            label="SMALL_EDGE",
            reason=f"model {edge_signed*100:+.1f}pp vs market on {side}",
            confidence_score=_confidence(model_prob_a, volatility),
        )
    return SignalResult(
        label="WATCH",
        reason="model view aligned with market — no edge",
        confidence_score=_confidence(model_prob_a, volatility),
    )

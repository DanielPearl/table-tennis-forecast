"""Expected value + edge math (same shape as the tennis sibling)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EVResult:
    edge: float
    ev_per_contract: float
    breakeven_market_prob: float


def edge(model_prob: float, market_prob: float) -> float:
    return float(model_prob) - float(market_prob)


def ev(model_prob: float, market_prob: float, slippage: float) -> EVResult:
    raw_edge = float(model_prob) - float(market_prob)
    breakeven = float(market_prob) + float(slippage)
    payout_if_win = 1.0 - market_prob - slippage
    payout_if_loss = -(market_prob + slippage)
    ev_per = model_prob * payout_if_win + (1.0 - model_prob) * payout_if_loss
    return EVResult(edge=raw_edge, ev_per_contract=ev_per,
                    breakeven_market_prob=breakeven)

"""Professional (sharp-book) probabilities for table tennis.

Same shape as tennis-forecast's ``fetch_odds`` wrapper, backed by the
fleet-shared implementation in ``kalshi_sdk.pinnacle``. Pinnacle's
public guest feed quotes the low-tier table-tennis circuits heavily —
TT Elite Series, Czech Liga Pro, Setka Cup — which is exactly the
slate Kalshi lists (KXTTELITEMATCH). The Odds API carries no table
tennis at all, so the guest feed is the only (and free) source; the
devigged probability is stamped on each watchlist row as
``pinnacle_prob_a`` / ``_b`` and drives the Model % column and the
buy gate's edge, mirroring tennis.

Silently returns an empty dict when the feed is down or rate-limited;
every consumer tolerates a missing pinnacle_prob field.
"""
from __future__ import annotations

from kalshi_sdk.pinnacle import pinnacle_guest_probs_by_pair

from ..utils.logging_setup import setup_logging

log = setup_logging("data.fetch_odds")


def pinnacle_probs_by_pair() -> dict[frozenset, dict[str, float]]:
    """``{frozenset({name_a, name_b}): {name_a: prob_a, name_b: prob_b}}``
    for every table-tennis matchup Pinnacle's guest feed currently
    quotes. Cached inside the SDK helper."""
    try:
        return pinnacle_guest_probs_by_pair("table_tennis")
    except Exception:  # noqa: BLE001 — benchmarks are best-effort
        log.exception("pinnacle guest fetch failed (non-fatal)")
        return {}

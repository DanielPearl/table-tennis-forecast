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
    for every table-tennis matchup a professional book quotes.

    Source cascade (each fills pairs the previous one missed):
      1. Pinnacle guest feed (free; currently lists no TT but picks
         itself back up the moment their coverage returns).
      2. Bet365 via BetsAPI (needs BETSAPI_KEY in the shared env) —
         the one aggregator that reliably carries TT Elite / Liga Pro.
    """
    out: dict[frozenset, dict[str, float]] = {}
    try:
        out.update(pinnacle_guest_probs_by_pair("table_tennis"))
    except Exception:  # noqa: BLE001 — benchmarks are best-effort
        log.exception("pinnacle guest fetch failed (non-fatal)")
    # BetsAPI merge RETIRED here (2026-09-11): the dashboard's
    # benchmark pass (kalshi_sdk.betsapi via bots/_sport_bot) is the
    # single BetsAPI consumer now — it rewrites every row's benchmark
    # anyway, and running this module's own fetch in parallel doubled
    # the API spend and spammed 502 retries (this path was also built
    # for the bet365 endpoints, which the actual key's plan 403s;
    # the SDK module uses the events + odds-summary products the plan
    # carries, with backoff, an odds-call cap, and an in-play line
    # filter). src/data/betsapi_odds.py stays for reference only.
    return out

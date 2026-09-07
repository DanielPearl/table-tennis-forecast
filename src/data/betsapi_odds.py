"""Bet365 table-tennis odds via BetsAPI — the professional benchmark.

Pinnacle's guest feed currently lists zero table tennis, and The Odds
API carries no TT key at all; Bet365 (resold by betsapi.com) quotes
the low-tier circuits Kalshi lists — TT Elite Series, Czech Liga Pro,
Setka Cup — around the clock. This module turns that feed into the
same ``{frozenset({name_a, name_b}): {name_a: prob, name_b: prob}}``
shape ``kalshi_sdk.pinnacle`` produces, so ``fetch_odds`` can merge
the two sources and everything downstream (Model %, edge, buy gate,
the live executor's require_pinnacle) treats it as one benchmark.

Needs ``BETSAPI_KEY`` in the shared env (Secret Keys/.env.shared).
Silently returns {} when the key is missing or the API errors —
benchmarks are always best-effort.

Cost control: one upcoming-events call plus one odds call per
event, filtered to the leagues Kalshi actually lists, cached for
5 minutes — a full day of TT Elite is ~40 events, so the per-day
credit spend stays in the hundreds.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from kalshi_sdk.pinnacle import devig_two_way

from ..utils.logging_setup import setup_logging

log = setup_logging("data.betsapi")

_BASE = "https://api.b365api.com"
_SPORT_ID_TABLE_TENNIS = 92
# Only leagues Kalshi mirrors — keeps the odds-call fanout (and the
# credit spend) proportional to the tradeable slate. Substring match,
# case-insensitive.
_LEAGUE_KEYWORDS = ("tt elite", "elite series", "czech liga pro",
                     "setka cup", "wtt")
_CACHE_TTL_S = 300.0
_MAX_EVENTS_PER_REFRESH = 80

_cache: dict[str, Any] = {"ts": 0.0, "data": {}}


def _token() -> str:
    return (os.environ.get("BETSAPI_KEY")
            or os.environ.get("BETSAPI_TOKEN") or "").strip()


def _get(path: str, **params) -> Any:
    params["token"] = _token()
    url = f"{_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent":
                                                "kalshi-tt-bot"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _league_ok(name: str) -> bool:
    low = (name or "").lower()
    return any(k in low for k in _LEAGUE_KEYWORDS)


def _decimal_prob_pair(home_od, away_od) -> tuple[float, float] | None:
    """Devig a two-way decimal-odds pair via the shared SDK helper."""
    try:
        h, a = float(home_od), float(away_od)
    except (TypeError, ValueError):
        return None
    if h <= 1.0 or a <= 1.0:
        return None
    # devig_two_way takes DECIMAL ODDS (it inverts internally).
    return devig_two_way(h, a)


def _collect_events() -> list[dict]:
    """Upcoming + in-play TT events in the leagues we care about."""
    events: list[dict] = []
    for path in ("/v3/events/upcoming", "/v3/events/inplay"):
        try:
            data = _get(path, sport_id=_SPORT_ID_TABLE_TENNIS)
        except Exception as e:  # noqa: BLE001
            log.warning("betsapi %s failed (non-fatal): %s", path, e)
            continue
        for ev in (data or {}).get("results") or []:
            if not _league_ok((ev.get("league") or {}).get("name", "")):
                continue
            home = (ev.get("home") or {}).get("name", "").strip()
            away = (ev.get("away") or {}).get("name", "").strip()
            if home and away and ev.get("id"):
                events.append({"id": ev["id"], "home": home,
                                "away": away})
    return events[:_MAX_EVENTS_PER_REFRESH]


def _event_probs(event_id) -> tuple[float, float] | None:
    """Devigged (home, away) win probs from the event's odds summary.

    Prefers Bet365's most recent moneyline; the summary payload keys
    odds by market id — for table tennis the match-winner market is
    ``92_1`` with ``home_od`` / ``away_od``.
    """
    try:
        data = _get("/v2/event/odds/summary", event_id=event_id)
    except Exception as e:  # noqa: BLE001
        log.warning("betsapi odds/summary %s failed: %s", event_id, e)
        return None
    results = (data or {}).get("results") or {}
    for book in ("Bet365", "bet365", "PinnacleSports", "1XBet"):
        payload = results.get(book)
        if not isinstance(payload, dict):
            continue
        for stage in ("end", "kickoff", "start"):
            ml = ((payload.get("odds") or {}).get(stage) or {}).get("92_1")
            if isinstance(ml, dict):
                pair = _decimal_prob_pair(ml.get("home_od"),
                                           ml.get("away_od"))
                if pair is not None:
                    return pair
    return None


def betsapi_probs_by_pair() -> dict[frozenset, dict[str, float]]:
    """Same pair-lookup shape as ``kalshi_sdk.pinnacle`` producers."""
    if not _token():
        return {}
    now = time.time()
    if now - _cache["ts"] < _CACHE_TTL_S:
        return _cache["data"]
    out: dict[frozenset, dict[str, float]] = {}
    events = _collect_events()
    for ev in events:
        pair = _event_probs(ev["id"])
        if pair is None:
            continue
        p_home, p_away = pair
        key = frozenset({ev["home"], ev["away"]})
        out[key] = {ev["home"]: p_home, ev["away"]: p_away,
                     "_source": "bet365_betsapi"}
    if events:
        log.info("betsapi: %d TT events scanned, %d with a usable "
                 "moneyline", len(events), len(out))
    _cache["ts"] = now
    _cache["data"] = out
    return out

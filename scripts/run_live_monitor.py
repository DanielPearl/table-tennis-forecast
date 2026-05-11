"""Live-monitor loop — real Kalshi table-tennis markets.

One iteration per ``dashboard.refresh_seconds``:

  1. Pull every active Kalshi market in the configured series list.
  2. Collapse two-sided markets into one record per event_ticker.
  3. Write the canonical live-state file (watchlist exporter reads it).
  4. Build the watchlist (model probabilities + edge + signal label).
  5. Tick the paper-trade simulator.

Real-only: if Kalshi creds aren't set, the script raises at startup so
the operator notices immediately rather than silently writing nothing.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.dashboard.export_watchlist import build_watchlist_records, export
from src.data import kalshi_markets
from src.trading.simulator import tick as simulator_tick
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

log = setup_logging("scripts.live_monitor",
                     log_path=str(_REPO / "data" / "live_monitor.log"))


def _require_kalshi_creds() -> None:
    if not os.environ.get("KALSHI_API_KEY_ID", "").strip():
        raise RuntimeError(
            "KALSHI_API_KEY_ID is not set. The table-tennis bot reads "
            "live Kalshi markets via the kalshi_sdk; without creds there "
            "is nothing to forecast against. Set KALSHI_API_KEY_ID + "
            "KALSHI_PRIVATE_KEY_PATH on the systemd unit's "
            "EnvironmentFile (or your local .env)."
        )
    if not os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip():
        raise RuntimeError(
            "KALSHI_PRIVATE_KEY_PATH is not set. Point this at the RSA "
            "private key file the api_key_id was issued for."
        )


_prev_market_by_ticker: dict[str, dict] = {}


def _one_tick() -> None:
    global _prev_market_by_ticker
    raw_markets = kalshi_markets.fetch_table_tennis_markets()
    new_prev = {m.get("ticker"): m for m in raw_markets if m.get("ticker")}
    records = kalshi_markets.collapse_to_matches(
        raw_markets, prev_markets_by_ticker=_prev_market_by_ticker
    )
    _prev_market_by_ticker = new_prev
    kalshi_markets.write_live_state(records)

    rows = build_watchlist_records()
    export(records=rows)

    state = simulator_tick(rows, records)

    log.info(
        "tick — %d kalshi markets / %d matches / %d watchlist rows / "
        "%d open positions / %d closed (P&L %+.3f, ROI %s)",
        len(raw_markets), len(records), len(rows),
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        ("—" if state["stats"].get("roi") is None
         else f"{state['stats']['roi']*100:+.1f}%"),
    )


def main() -> None:
    _require_kalshi_creds()
    cfg = load_config()
    period = int(cfg["dashboard"]["refresh_seconds"])
    log.info("live monitor started — refresh every %ds (real Kalshi feed)",
              period)
    while True:
        try:
            _one_tick()
        except Exception as exc:
            log.exception("tick error: %s", exc)
        time.sleep(period)


if __name__ == "__main__":
    main()

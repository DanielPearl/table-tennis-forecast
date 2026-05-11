"""Standalone table-tennis dashboard server.

Stdlib http.server only — same pattern as the trading-dashboard project
on the droplet. The primary surface for users is the multi-bot trading
dashboard; this standalone server is for local development and as a
fallback when the trading dashboard is offline. Read-only viewer over
the JSON watchlist file written by ``src/dashboard/export_watchlist``.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import setup_logging

log = setup_logging("dashboard.server")


def _load_watchlist() -> dict:
    cfg = load_config()
    fp = resolve_path(cfg["paths"]["watchlist_json"])
    if not fp.exists():
        return {"generated_at": None, "rows": []}
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _render_summary(rows: list[dict]) -> str:
    n = len(rows)
    actionable = sum(1 for r in rows if r.get("recommended_action") in
                      ("STRONG_EDGE", "SMALL_EDGE", "MARKET_OVERREACTION"))
    return (
        f"<div class='card'><strong>{n}</strong> matches, "
        f"<strong>{actionable}</strong> with edge</div>"
    )


def _render_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No matches yet — wait for the live monitor to tick.</p>"
    out = ["<table border='1' cellpadding='6'><thead><tr>",
            "<th>Match</th><th>Tournament</th>",
            "<th>Pre prob (A)</th><th>Live prob (A)</th>",
            "<th>Market prob (A)</th><th>Edge</th><th>Verdict</th>",
            "</tr></thead><tbody>"]
    for r in rows:
        mp = r.get("market_prob_a")
        mp_str = f"{mp*100:.0f}%" if mp is not None else "—"
        edge_a = r.get("edge_a")
        edge_str = f"{edge_a*100:+.1f}pp" if edge_a is not None else "—"
        out.append(
            f"<tr><td>{r['player_a']} vs {r['player_b']}</td>"
            f"<td>{r['tournament']}</td>"
            f"<td>{r['pre_match_prob_a']*100:.0f}%</td>"
            f"<td>{r['live_prob_a']*100:.0f}%</td>"
            f"<td>{mp_str}</td><td>{edge_str}</td>"
            f"<td>{r['recommended_action']}</td></tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _render_page() -> str:
    payload = _load_watchlist()
    rows = payload.get("rows") or []
    css = ("body{font-family:sans-serif;background:#0d1117;color:#c9d1d9;"
            "padding:24px}.card{background:#161b22;padding:14px;"
            "border:1px solid #30363d;border-radius:8px;margin:12px 0}"
            "table{background:#161b22;border-color:#30363d!important}")
    return (
        f"<!doctype html><html><head><title>Table Tennis Forecast</title>"
        f"<style>{css}</style></head><body>"
        f"<h1>Table Tennis Forecast</h1>"
        f"<p>Loaded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}</p>"
        f"{_render_summary(rows)}"
        f"<h2>Watchlist</h2>{_render_table(rows)}"
        "</body></html>"
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/api/watchlist.json":
            payload = _load_watchlist()
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = _render_page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        log.info(fmt % args)


def main() -> None:
    cfg = load_config()
    host = os.environ.get("HOST") or cfg["dashboard"]["host"]
    port = int(os.environ.get("PORT") or cfg["dashboard"]["port"])
    server = ThreadingHTTPServer((host, port), Handler)
    log.info("table-tennis dashboard on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.server_close()


if __name__ == "__main__":
    main()

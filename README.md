# Table Tennis Forecast

Pre-match + live-adjustment forecast system for elite-level table
tennis, built for event/contract-style markets (Kalshi-style). The
point isn't to pick winners — it's to compare a calibrated model
probability against the market's implied probability and surface
spots where the two materially disagree (live perceived value).

```
table-tennis-forecast/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                         # seed CSV + live-state fixture
│   ├── processed/                   # cleaned panels + model artifacts
│   └── outputs/                     # watchlist.csv / .json / sim_state.json
├── src/
│   ├── data/
│   │   ├── fetch_matches.py         # bundled seed + optional external CSV
│   │   ├── fetch_live_scores.py     # reads canonical live-state file
│   │   └── kalshi_markets.py        # KXTABLETENNIS / KXWTTSGLES / KXATTABLE
│   ├── features/
│   │   ├── elo.py                   # overall + style (handedness) Elo
│   │   ├── build_prematch_features.py  # broad 22-feature panel
│   │   └── build_live_features.py   # standardize live state
│   ├── models/
│   │   ├── train_prematch_model.py  # broad → prune → refit pipeline
│   │   ├── live_adjustment_model.py # transparent rules layer
│   │   └── predict.py
│   ├── trading/
│   │   ├── ev.py / signals.py / simulator.py / backtest.py
│   └── dashboard/
│       └── export_watchlist.py
├── scripts/
│   ├── run_daily_prematch.py        # train + export
│   ├── run_live_monitor.py          # tick loop
│   ├── run_backtest.py
│   └── generate_seed_data.py        # regenerates seed CSV (committed)
├── app/
│   └── dashboard.py                 # stdlib http.server fallback
└── deploy/
    ├── table-tennis-dashboard.service
    ├── table-tennis-monitor.service
    └── deploy.sh
```

## What the model does

**Pre-match** — gives a baseline win probability before the match.
Inputs span the broad table-tennis-specific feature panel:

- overall Elo + style-matchup (handedness) Elo
- rolling form windows (last 5 / 10 / 20 matches)
- average + std of point-win % and game-margin (last 10)
- closing-game win % (last game of a bo5/bo7)
- deuce conversion rate
- comeback rate (won after losing first game) — over last 20
- head-to-head total + last 5
- days of rest + matches in last 7 days
- ranking diff, tournament tier, round
- best-of-5 vs best-of-7 indicator
- hand matchup (L vs R)

We start with the broad panel, train a logistic-baseline + calibrated
GBT ensemble, then run **walk-forward permutation importance** to
identify noisy / non-contributing features. Anything below the
configured prune floor whose 1-std band crosses zero is dropped, and
the model is re-fit on the survivors. Whichever model has better
held-out log-loss (broad vs. pruned) is what the live trader uses.

**Live adjustment** — a transparent rules layer (phase 1) that nudges
the pre-match probability using:

- score-state momentum (set + game + point-streak deltas)
- live point-win % differential
- deuce / game-point / set-point / match-point volatility flags
- closing-game volatility
- comeback dampener (when a high-comeback player is trailing)
- market-overreaction detection

Every nudge is logged into a `reason` string the dashboard surfaces
alongside the signal.

**Signals** — never fire on winner probability alone. They fire only
when the model's view differs from the market by more than a configured
edge floor, with separate gates for high volatility and injury risk:

```
INJURY_RISK ▸ AVOID_VOLATILE ▸ MARKET_OVERREACTION
            ▸ STRONG_EDGE ▸ SMALL_EDGE ▸ WATCH ▸ NO_TRADE
```

## Running locally (VS Code)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 1) (re)generate the seed match dataset if needed — already committed
python scripts/generate_seed_data.py

# 2) train + export
python scripts/run_daily_prematch.py

# 3) standalone dashboard (port 8091)
python app/dashboard.py
```

The primary UI is the multi-bot trading dashboard on the droplet
(http://178.128.145.111:8080); the standalone dashboard is for local
development and as a fallback.

## Deploying to the DigitalOcean droplet

```bash
# one-time, on a fresh droplet (as root):
apt update && apt install -y python3-pip python3-venv git
cd /root && git clone https://github.com/DanielPearl/table-tennis-forecast.git
cd table-tennis-forecast && bash deploy/deploy.sh
```

To redeploy:

```bash
ssh root@178.128.145.111
cd /root/table-tennis-forecast
git pull
systemctl restart table-tennis-monitor table-tennis-dashboard
```

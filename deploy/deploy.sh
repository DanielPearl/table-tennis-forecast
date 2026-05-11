#!/usr/bin/env bash
# One-shot deploy / redeploy on a DigitalOcean Ubuntu droplet.
#
# Usage on a fresh droplet (as root):
#   apt update && apt install -y python3-pip python3-venv git
#   cd /root && git clone https://github.com/DanielPearl/table-tennis-forecast.git
#   cd table-tennis-forecast && bash deploy/deploy.sh

set -euo pipefail

REPO_DIR="/root/table-tennis-forecast"
cd "$REPO_DIR"

if [ -d .git ]; then
  echo "[deploy] git pull"
  git pull --ff-only
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[deploy] wrote .env from .env.example — edit before starting"
fi

echo "[deploy] training pre-match model — first run takes a few minutes"
python scripts/run_daily_prematch.py

cp deploy/table-tennis-dashboard.service /etc/systemd/system/
cp deploy/table-tennis-monitor.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now table-tennis-dashboard
systemctl enable --now table-tennis-monitor

echo
echo "[deploy] up — dashboard on :8091, live monitor running"
echo "[deploy] logs:  journalctl -u table-tennis-dashboard -f"
echo "[deploy]        journalctl -u table-tennis-monitor -f"

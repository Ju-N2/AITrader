# AITrader

Python crypto market analysis and **paper trading** system designed for one
always-on Raspberry Pi. The runtime collects public candles, calculates
indicators and multi-timeframe signals, applies local AI-style analysis and
risk checks, then stores simulated positions in SQLite. It does not submit
exchange orders.

## Raspberry Pi setup

Use Raspberry Pi OS 64-bit and Python 3.11 or newer. Keep this checkout at
`~/AITrader` (or update the paths in `deploy/aitrader.service`). From the
project directory:

```sh
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dashboard]'
mkdir -p data/backups logs
cp .env.example .env
```

Telegram credentials in `.env` are optional. Candle collection uses public
market endpoints; do not add exchange trading keys for this paper-only
runtime. Keep `.env` private.

Run one non-trading analysis cycle first:

```sh
.venv/bin/python scripts/run_paper_trading.py --once
```

After reviewing the output and logs, start simulated trading in the foreground:

```sh
.venv/bin/python scripts/run_paper_trading.py --paper
```

`--paper` is required to create simulated orders and positions. Without it, the
runtime performs analysis only.

## Run continuously with systemd

The provided user service restarts a failed runtime and starts in paper mode.
It assumes the repository is `~/AITrader` and that the virtual environment and
`data` and `logs` directories exist there.

```sh
mkdir -p ~/.config/systemd/user
cp deploy/aitrader.service ~/.config/systemd/user/aitrader.service
cp deploy/aitrader-dashboard.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now aitrader.service
systemctl --user enable --now aitrader-dashboard.service
```

To keep the user service running after logout and across reboot, enable lingering
for the Linux account:

```sh
sudo loginctl enable-linger "$USER"
```

Inspect status and logs with:

```sh
systemctl --user status aitrader.service
journalctl --user -u aitrader.service -f
```

Stop or disable it with `systemctl --user stop aitrader.service` or
`systemctl --user disable aitrader.service`. A filesystem lock prevents a
second runtime from using the same local database. systemd sends SIGTERM on
stop; the runtime handles it gracefully and closes the exchange connection.

Install the daily, integrity-checked database backup timer as well:

```sh
cp deploy/aitrader-backup.service ~/.config/systemd/user/
cp deploy/aitrader-backup.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now aitrader-backup.timer
systemctl --user list-timers aitrader-backup.timer
```

## Data and dashboard

SQLite data is stored in `data/aitrader.db` in WAL mode; rotating logs are
stored under `logs/`. The daily timer writes seven verified snapshots under
`data/backups/`; backups are on the same SD card, so copy them off-device
periodically if you need hardware-failure protection. Create a backup manually
with `.venv/bin/python scripts/backup_database.py`. To restore, stop the
runtime, copy a selected snapshot to `data/aitrader.db`, then start the service.
The dashboard is optional and read-only. Do not expose it directly to the
public internet; it has no login or access control.

For local access, start it from the repository root bound to loopback:

```sh
.venv/bin/uvicorn web.server:app --host 127.0.0.1 --port 8000
```

Alternatively, use the `aitrader-dashboard.service` user unit above to start
the dashboard at login and after reboot. If another process already owns port
8000, stop that process before starting this unit.

From another machine, use an SSH tunnel instead of binding the dashboard to
`0.0.0.0`: `ssh -L 8000:127.0.0.1:8000 <pi-host>`.

Install development tools and run the test suite with:

```sh
.venv/bin/python -m pip install -e '.[dev,dashboard]'
.venv/bin/pytest -q
```

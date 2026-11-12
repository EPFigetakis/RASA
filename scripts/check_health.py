#!/usr/bin/env python3
"""
Health check for the paper-trading system. Prints the alert list and exits non-zero when
something needs attention, so cron/systemd/a mail wrapper can notify you:

    python scripts/check_health.py            # human-readable
    python scripts/check_health.py --json     # machine-readable

Exit codes: 0 = all clear (info only), 1 = warnings, 2 = critical alerts, 3 = could not check.
Read-only: it never changes the account or any file except the log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import date  # noqa: E402

from core import broker_paper, daily_run, monitoring, paths, prices, risk  # noqa: E402


def collect(db: Path):
    risk_cfg = risk.load_risk_config()
    status = prices.get_cache_status_from_config()
    last_bar = date.fromisoformat(status["last_bar_date"]) if status.get("last_bar_date") else None
    if not db.exists():
        raise FileNotFoundError(f"no paper broker database at {db} (run the daily runner first)")
    broker = broker_paper.PaperBroker(db)
    try:
        snap = monitoring.build_snapshot(broker, risk_cfg, daily_run.list_run_records(), last_bar)
    finally:
        broker.close()
    return snap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="paper broker SQLite file (default: state/rasa_paper.db)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    db = Path(args.db) if args.db else paths.state_dir() / "rasa_paper.db"
    try:
        snap = collect(db)
    except Exception as e:  # cannot even check: that itself is an alert
        print(f"CRITICAL could not run health check: {e}", file=sys.stderr)
        return 3

    code = monitoring.exit_code(snap.alerts)
    if args.json:
        print(json.dumps({
            "exit_code": code, "price_last_bar": str(snap.price_last_bar),
            "broker_last_bar": str(snap.broker_last_bar), "equity": snap.equity,
            "drawdown": snap.drawdown,
            "alerts": [a.__dict__ for a in snap.alerts]}, indent=2))
    else:
        eq = f"{snap.equity:,.2f}" if snap.equity is not None else "n/a"
        print(f"[{snap.environment}] equity {eq} | drawdown {snap.drawdown:.2%} | "
              f"prices {snap.price_last_bar} | broker bar {snap.broker_last_bar}")
        if not snap.alerts:
            print("OK: no alerts")
        for a in snap.alerts:
            print(f"{a.severity.upper():<8} {a.code}: {a.message}")
    return code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
RASA daily runner (paper only). Run from the repository root:

    python scripts/run_daily_proposals.py run                # refresh prices, propose tickets
    python scripts/run_daily_proposals.py run --dry-run      # compute only; write nothing
    python scripts/run_daily_proposals.py list               # recent runs
    python scripts/run_daily_proposals.py approve RUN_ID --all --by me
    python scripts/run_daily_proposals.py approve RUN_ID --ticket T123 --by me

`run` never submits orders (unless trading_config.json sets require_manual_approval=false AND
you pass --submit). `approve` is the separate, explicit step that releases tickets to the paper
broker.

Exit codes (for a scheduler / alerting):
    0  success (including "nothing to do")
    1  unexpected error
    2  precondition failed (no/mismatched saved pairs, empty prices, another run in progress,
       live environment) or data too stale to trade
    3  kill switch tripped (daily-loss / drawdown limit)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import broker_paper, daily_run, paths, risk  # noqa: E402

try:
    import fcntl
except ImportError:  # Windows: no cross-process lock available
    fcntl = None

log = logging.getLogger("rasa.daily")


def setup_logging(verbose: bool) -> None:
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(paths.logs_dir() / "daily_runner.log")]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt, handlers=handlers)


@contextmanager
def single_instance():
    """Refuse to run two pipelines at once (they share the broker DB and run records)."""
    path = paths.state_dir() / "daily_runner.lock"
    with open(path, "w") as fh:
        if fcntl is not None:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise daily_run.RunError("another daily run is already in progress")
        yield


def open_broker(args) -> broker_paper.PaperBroker:
    db = Path(args.db) if args.db else paths.state_dir() / "rasa_paper.db"
    risk_cfg = risk.load_risk_config()
    if not db.exists():
        return broker_paper.PaperBroker(db, broker_paper.PaperBrokerConfig(initial_cash=risk_cfg.capital_base))
    return broker_paper.PaperBroker(db)


def print_summary(rec: daily_run.RunRecord) -> None:
    log.info("run %s: %s | as_of=%s | data age=%s bday(s) | ideas=%d | tickets=%d | fills processed=%d",
             rec.run_id, rec.status, rec.as_of, rec.data_age_business_days, rec.n_ideas,
             len(rec.tickets), rec.fills_processed)
    for w in rec.warnings:
        log.warning("  %s", w)
    decisions = {d["ticket_id"]: d for d in rec.risk_report.get("decisions", [])}
    for t in rec.ticket_objects():
        d = decisions.get(t.ticket_id, {})
        legs = ", ".join(f"{l.side.value} {l.quantity:g} {l.symbol}" for l in t.legs)
        why = "" if d.get("approved", True) else " | " + "; ".join(
            f"{c['name']}: {c['detail']}" for c in d.get("checks", []) if not c["passed"])
        log.info("  %s [%s] %s: %s%s", t.ticket_id, t.status.value, t.pair or t.purpose, legs, why)
    for r in rec.kill_switch_reasons:
        log.error("  KILL SWITCH: %s", r)


def cmd_run(args) -> int:
    run_cfg = daily_run.RunConfig()
    risk_cfg = risk.load_risk_config()
    with single_instance():
        px, pairs_df, warnings = daily_run.load_inputs(update_prices=not args.no_update)
        broker = open_broker(args)
        try:
            rec = daily_run.run_daily(px, pairs_df, run_cfg, risk_cfg, broker, submit=args.submit,
                                      dry_run=args.dry_run, input_warnings=warnings)
        finally:
            broker.close()
    print_summary(rec)
    if args.dry_run:
        log.info("dry run: nothing written")
    if rec.status == daily_run.STATUS_STALE:
        return 2
    if rec.status == daily_run.STATUS_KILL:
        return 3
    return 0


def cmd_list(args) -> int:
    for rec in daily_run.list_run_records()[-args.n:]:
        print(f"{rec.run_id}  {rec.started_at[:19]}  {rec.status:<20} as_of={rec.as_of} "
              f"tickets={len(rec.tickets)} submitted={len(rec.submitted_ticket_ids)}")
    return 0


def cmd_approve(args) -> int:
    if not args.all and not args.ticket:
        raise daily_run.RunError("pass --all or one or more --ticket IDs")
    with single_instance():
        rec = daily_run.load_run_record(args.run_id)
        broker = open_broker(args)
        try:
            done = daily_run.approve_and_submit(
                rec, broker, None if args.all else args.ticket, approved_by=args.by)
        finally:
            broker.close()
    for t in done:
        log.info("submitted %s [%s] %s", t.ticket_id, t.status.value, t.pair or t.purpose)
    if not done:
        log.info("nothing to approve")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="paper broker SQLite file (default: state/rasa_paper.db)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="refresh data and propose tickets")
    r.add_argument("--no-update", action="store_true", help="use cached prices as-is (offline)")
    r.add_argument("--dry-run", action="store_true", help="compute only; write nothing")
    r.add_argument("--submit", action="store_true",
                   help="also submit tickets that are already APPROVED (needs require_manual_approval=false)")
    r.set_defaults(fn=cmd_run)

    l = sub.add_parser("list", help="show recent runs")
    l.add_argument("-n", type=int, default=10)
    l.set_defaults(fn=cmd_list)

    a = sub.add_parser("approve", help="approve and submit tickets from a run")
    a.add_argument("run_id")
    a.add_argument("--all", action="store_true")
    a.add_argument("--ticket", action="append", default=[])
    a.add_argument("--by", default="user", help="who approved (recorded in the ticket history)")
    a.set_defaults(fn=cmd_approve)

    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.fn(args)
    except daily_run.RunError as e:
        log.error("%s", e)
        return 2
    except Exception:
        log.exception("daily run failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

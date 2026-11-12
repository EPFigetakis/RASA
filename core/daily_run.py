"""
core.daily_run

The non-UI daily pipeline: prices -> freshness check -> paper-broker bar -> trade ideas ->
order plan -> risk gates -> a saved run record of proposed tickets -> (separate) human
approval and submission.

Design rules
- **Deterministic and idempotent.** Given the same prices, saved pairs, config and broker
  state, a run produces the same ids and overwrites the same record. Tickets are
  idempotent in the broker. Re-running a day never doubles a position.
- **Proposals are not orders.** `run_daily` never submits anything. It leaves passing
  tickets PROPOSED (the default) in a run record; `approve_and_submit` is the explicit
  second step. (If trading_config sets require_manual_approval=false, passing tickets are
  APPROVED and `run_daily(..., submit=True)` may submit them.)
- **Stale data fails closed.** Older than the configured limit aborts the run before
  the broker is touched.
- **Safe to refuse.** Approval is refused if the broker's positions or last bar changed
  since the run, because the tickets were sized against a state that no longer exists.
- **Paper only.** Runs refuse a risk config whose environment is "live".

The pure pipeline (`run_daily`) takes the price panel and pair table as arguments so it
is testable offline; `load_inputs` does the file/network I/O.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import core.broker_base as broker_base
    import core.broker_paper as broker_paper
    import core.orders as orders
    import core.pair_selection as pair_selection
    import core.panel as panel
    import core.paths as paths
    import core.prices as prices
    import core.risk as risk
    import core.trade_proposals as trade_proposals
except ModuleNotFoundError:  # run from inside core/
    import broker_base, broker_paper, orders, pair_selection, panel, paths, prices, risk, trade_proposals  # type: ignore

STATUS_OK = "ok"
STATUS_NO_ACTION = "no_action"
STATUS_STALE = "aborted_stale_data"
STATUS_KILL = "kill_switch"

OLD_PAIRS_WARNING_DAYS = 45


class RunError(RuntimeError):
    """A precondition failed (missing/mismatched pairs, live environment, ...)."""


# ---------------------------------------------------------------------------
# Config & record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """Signal/sizing parameters for the daily proposals (mirrors trade_proposals.ProposalConfig)."""

    top_n: int = 10
    notional_per_pair: float = 10_000.0
    entry_z: float = 2.0
    exit_z: float = 0.5
    zscore_window: int = 60
    use_log_prices: bool = False
    add_intercept: bool = False

    def proposal_config(self, whole_shares: bool) -> trade_proposals.ProposalConfig:
        d = asdict(self)
        return trade_proposals.ProposalConfig(**d, whole_shares=whole_shares)

    @property
    def config_hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class RunRecord:
    run_id: str
    status: str
    as_of: Optional[str]
    started_at: str
    environment: str
    data_age_business_days: Optional[int]
    warnings: List[str] = field(default_factory=list)
    run_config: Dict[str, Any] = field(default_factory=dict)
    risk_config_hash: str = ""
    panel_fingerprint: str = ""
    batch_id: Optional[str] = None
    plan_id: Optional[str] = None
    n_ideas: int = 0
    n_skipped: int = 0
    fills_processed: int = 0
    kill_switch_reasons: List[str] = field(default_factory=list)
    positions_snapshot: Dict[str, float] = field(default_factory=dict)
    broker_last_bar: Optional[str] = None
    ideas: List[Dict[str, Any]] = field(default_factory=list)
    tickets: List[Dict[str, Any]] = field(default_factory=list)
    risk_report: Dict[str, Any] = field(default_factory=dict)
    submitted_ticket_ids: List[str] = field(default_factory=list)

    def ticket_objects(self) -> List[orders.OrderTicket]:
        return [orders.OrderTicket.from_dict(t) for t in self.tickets]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        return cls(**d)


def runs_dir() -> Path:
    d = paths.state_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_record(rec: RunRecord, directory: Optional[Path] = None) -> Path:
    directory = Path(directory) if directory else runs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{rec.run_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec.to_dict(), indent=2, sort_keys=True, default=str))
    tmp.replace(path)  # atomic
    return path


def load_run_record(run_id: str, directory: Optional[Path] = None) -> RunRecord:
    path = (directory or runs_dir()) / f"{run_id}.json"
    if not path.exists():
        raise RunError(f"no run record {run_id} in {path.parent}")
    return RunRecord.from_dict(json.loads(path.read_text()))


def list_run_records(directory: Optional[Path] = None) -> List[RunRecord]:
    d = directory or runs_dir()
    recs = [RunRecord.from_dict(json.loads(p.read_text())) for p in d.glob("R*.json")]
    return sorted(recs, key=lambda r: r.started_at)


# ---------------------------------------------------------------------------
# Inputs (I/O)
# ---------------------------------------------------------------------------

def load_inputs(update_prices: bool = True):
    """
    Load the price panel (optionally appending new bars) and the saved candidate pairs,
    verified against it. Returns (px, pairs_df, warnings).

    The pair-selection settings are taken from the saved selection's own record, so only
    the universe is checked strictly; changed prices just produce a warning (rescans are
    infrequent by design).
    """
    px, _ = panel.load_prices_and_returns_from_config(update_if_stale=update_prices)

    default_cfg = pair_selection.PairSelectionConfig()
    meta = pair_selection.read_pairs_metadata(default_cfg)
    if meta is None:
        raise RunError(
            "No verified saved pair selection found. Run pair selection in the app first "
            "(the daily runner never rescans the universe)."
        )
    cfg = pair_selection.PairSelectionConfig(**meta["settings"])
    try:
        pairs_df = pair_selection.load_candidate_pairs(cfg, px=px, strict=True)
    except pair_selection.StaleCacheError as e:
        raise RunError(f"Saved pair selection does not match the price universe: {e}") from e

    warnings: List[str] = []
    check = pair_selection.check_candidate_pairs_cache(px, cfg)
    if check.status != "ok":
        warnings.extend(check.reasons)
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(meta["created_at"])).days
        if age > OLD_PAIRS_WARNING_DAYS:
            warnings.append(f"Saved pair selection is {age} days old; consider rescanning.")
    except (KeyError, ValueError):
        pass
    return px, pairs_df, warnings


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _account_state(broker: broker_paper.PaperBroker, as_of: str, open_pairs: int) -> risk.AccountState:
    acct = broker.get_account()
    hist = broker.equity_history()
    equity = acct.equity if acct.equity is not None else broker.portfolio.cash
    prior = hist[hist.index < as_of]
    day_start = float(prior["equity"].iloc[-1]) if len(prior) else broker.config.initial_cash
    peak = max([broker.config.initial_cash, equity] + hist["equity"].tolist())
    return risk.AccountState(equity=equity, day_start_equity=day_start, peak_equity=peak, open_pairs=open_pairs)


def _open_pair_count(pairs_df: pd.DataFrame, positions: Dict[str, float]) -> int:
    held = {s for s, q in positions.items() if abs(q) > orders.QTY_TOL}
    return int(sum(1 for r in pairs_df.itertuples() if str(r.x) in held and str(r.y) in held))


def run_daily(
    px: pd.DataFrame,
    pairs_df: pd.DataFrame,
    run_cfg: RunConfig,
    risk_cfg: risk.RiskConfig,
    broker: broker_paper.PaperBroker,
    today: Optional[date] = None,
    submit: bool = False,
    dry_run: bool = False,
    input_warnings: Optional[List[str]] = None,
    save: bool = True,
    record_dir: Optional[Path] = None,
) -> RunRecord:
    """
    Execute one daily run and return (and, unless dry_run/save=False, persist) its record.

    dry_run: nothing is written to the broker or to disk -- the bar is not processed, so the
    plan is computed against the broker's holdings as of its last processed bar.
    """
    if risk_cfg.environment == "live":
        raise RunError("The daily runner is paper-only; refusing a risk config with environment 'live'.")
    if px.empty:
        raise RunError("Price panel is empty.")

    started = datetime.now(timezone.utc).isoformat()
    today = today or datetime.now(timezone.utc).date()
    as_of_ts = pd.Timestamp(px.index.max())
    as_of = as_of_ts.date().isoformat()
    age = prices.business_days_behind(as_of_ts.date(), today)
    warnings = list(input_warnings or [])
    fingerprint = prices.panel_fingerprint(px)

    def finish(status: str, **kw) -> RunRecord:
        positions = broker.get_positions()
        rid_payload = json.dumps(
            [as_of, risk_cfg.config_hash, run_cfg.config_hash, fingerprint, kw.get("plan_id"),
             sorted(positions.items()), status, broker.last_bar()], default=str)
        rec = RunRecord(
            run_id="R" + hashlib.sha1(rid_payload.encode()).hexdigest()[:12],
            status=status, as_of=as_of, started_at=started, environment=broker.environment,
            data_age_business_days=age, warnings=warnings, run_config=asdict(run_cfg),
            risk_config_hash=risk_cfg.config_hash, panel_fingerprint=fingerprint,
            positions_snapshot=positions, broker_last_bar=broker.last_bar(), **kw)
        if save and not dry_run:
            save_run_record(rec, record_dir)
        return rec

    # 1. Freshness gate: before touching the broker.
    if age > risk_cfg.max_data_age_business_days:
        warnings.append(
            f"Price data ends {as_of}, {age} business days behind "
            f"(limit {risk_cfg.max_data_age_business_days}); run aborted.")
        return finish(STATUS_STALE)

    # 2. Advance the paper broker to this bar so earlier approved tickets fill first.
    fills_before = len(broker.portfolio.trades)
    last = broker.last_bar()
    if not dry_run and (last is None or as_of > last):
        broker.process_bar(px.iloc[-1].dropna().to_dict(), as_of)
    fills_processed = len(broker.portfolio.trades) - fills_before
    last_prices = px.iloc[-1].dropna().to_dict()

    # 3. Ideas -> plan -> risk.
    batch = trade_proposals.generate_trade_ideas(px, pairs_df, run_cfg.proposal_config(
        whole_shares=not risk_cfg.allow_fractional_shares))
    warnings.extend(batch.warnings)

    positions = broker.get_positions()
    managed = set(pairs_df["x"].astype(str)) | set(pairs_df["y"].astype(str))
    for t in broker.list_tickets():
        managed |= {l.symbol for l in t.legs}
    plan = orders.plan_orders(
        batch.ideas, positions, managed_symbols=managed, close_out_prices=last_prices,
        whole_shares=not risk_cfg.allow_fractional_shares)

    # Tickets the broker already has (submitted on an earlier run, not yet filled) are not
    # proposed again: the planner can't see working orders, but ids are deterministic.
    already = [t for t in plan.tickets if broker.get_ticket(t.ticket_id) is not None]
    if already:
        warnings.extend(f"{t.pair or 'close_out'}: ticket {t.ticket_id} already with broker "
                        f"({broker.get_ticket(t.ticket_id).status.value}); not re-proposed" for t in already)
        keep = [t for t in plan.tickets if broker.get_ticket(t.ticket_id) is None]
        plan = replace(plan, tickets=keep)

    open_pairs = _open_pair_count(pairs_df, positions)
    account = _account_state(broker, as_of, open_pairs)
    report = risk.evaluate_plan(plan, risk_cfg, prices=last_prices, account=account,
                                data_age_business_days=age)
    tickets = risk.apply_decisions(report, plan.tickets, risk_cfg)

    submitted: List[str] = []
    if submit and not dry_run:
        for t in tickets:
            if t.status == orders.OrderStatus.APPROVED:
                broker.submit_ticket(t)
                submitted.append(t.ticket_id)

    for issue in plan.already_at_target:
        warnings.append(f"{issue.subject}: {issue.reason}")
    for c in report.unticketed:
        warnings.append(f"{c.subject}: {c.reason}")

    status = STATUS_KILL if report.kill_switch else (STATUS_OK if tickets else STATUS_NO_ACTION)
    return finish(
        status, batch_id=batch.batch_id, plan_id=plan.plan_id, n_ideas=len(batch.ideas),
        n_skipped=len(batch.skipped), fills_processed=fills_processed,
        kill_switch_reasons=report.kill_switch_reasons,
        ideas=[i.to_dict() for i in batch.ideas], tickets=[t.to_dict() for t in tickets],
        risk_report=report.to_dict(), submitted_ticket_ids=submitted,
    )


# ---------------------------------------------------------------------------
# Approval step
# ---------------------------------------------------------------------------

def approve_and_submit(
    record: RunRecord,
    broker: broker_paper.PaperBroker,
    ticket_ids: Optional[List[str]] = None,
    approved_by: str = "user",
    save: bool = True,
    record_dir: Optional[Path] = None,
) -> List[orders.OrderTicket]:
    """
    Approve proposed tickets from a run record and submit them to the paper broker.

    Only tickets still PROPOSED and carrying a risk pass are eligible (rejected ones never
    are). Refused outright if the broker's holdings or last bar differ from the run's
    snapshot -- the tickets were sized for a state that no longer exists; re-run instead.
    `ticket_ids=None` approves every eligible ticket.
    """
    if broker.environment != "paper" or record.environment != "paper":
        raise RunError("approve_and_submit only operates on paper runs/brokers.")
    if record.status in (STATUS_STALE,):
        raise RunError(f"run {record.run_id} was aborted ({record.status}); nothing to approve.")
    if broker.last_bar() != record.broker_last_bar:
        raise RunError(
            f"broker has advanced to {broker.last_bar()} since run {record.run_id} "
            f"({record.broker_last_bar}); re-run the daily pipeline.")
    if portfolio_diff(broker.get_positions(), record.positions_snapshot):
        raise RunError("broker positions changed since this run; re-run the daily pipeline.")

    decisions = {d["ticket_id"]: d for d in record.risk_report.get("decisions", [])}
    eligible = [t for t in record.ticket_objects()
                if t.status == orders.OrderStatus.PROPOSED and decisions.get(t.ticket_id, {}).get("approved")]
    if ticket_ids is not None:
        unknown = set(ticket_ids) - {t.ticket_id for t in record.ticket_objects()}
        if unknown:
            raise RunError(f"unknown ticket id(s): {sorted(unknown)}")
        not_eligible = set(ticket_ids) - {t.ticket_id for t in eligible}
        if not_eligible:
            raise RunError(f"ticket(s) not eligible for approval (rejected or already handled): {sorted(not_eligible)}")
        eligible = [t for t in eligible if t.ticket_id in set(ticket_ids)]

    out: List[orders.OrderTicket] = []
    for t in eligible:
        out.append(broker.submit_ticket(t.approve(f"approved by {approved_by}")))
        if t.ticket_id not in record.submitted_ticket_ids:
            record.submitted_ticket_ids.append(t.ticket_id)
    if save and out:
        save_run_record(record, record_dir)
    return out


def portfolio_diff(a: Dict[str, float], b: Dict[str, float], tol: float = 1e-6) -> bool:
    keys = set(a) | set(b)
    return any(abs(a.get(k, 0.0) - b.get(k, 0.0)) > tol for k in keys)

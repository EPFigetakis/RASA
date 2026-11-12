"""
core.monitoring

Read-only health snapshot of the paper-trading system and the alerts derived from it.

`build_snapshot(...)` gathers what an operator needs at a glance -- how old the data is,
whether the daily run happened, the account and its drawdown, positions with PnL, working
and unreviewed tickets, utilization of the risk limits -- and `compute_alerts` turns problems
into a ranked list. Used by both the Streamlit monitor view and scripts/check_health.py (whose
exit code lets a scheduler or cron mailer raise the alarm).

Nothing here changes any state. It does not call the network: pass in the price panel's last
bar date, and (optionally) fresh marks.

Alert severities
- critical : act now (kill switch tripped, stale data, unhedged one-legged position,
             positions that don't reconcile with fills, aborted/failed last run)
- warning  : look soon (run overdue, drawdown or exposure near its limit, stuck/unreviewed
             tickets)
- info     : context only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import core.broker_paper as broker_paper
    import core.daily_run as daily_run
    import core.orders as orders
    import core.portfolio as portfolio
    import core.prices as prices
    import core.risk as risk
except ModuleNotFoundError:  # run from inside core/
    import broker_paper, daily_run, orders, portfolio, prices, risk  # type: ignore

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

NEAR_LIMIT = 0.80            # warn when a drawdown / exposure uses this share of its limit
RUN_OVERDUE_BDAYS = 2        # no daily run recorded for this many business days
WORKING_TICKET_MAX_BARS = 2  # a submitted ticket still unfilled after this many bars
UNREVIEWED_PROPOSAL_BDAYS = 1


@dataclass(frozen=True)
class Alert:
    severity: str   # critical | warning | info
    code: str
    message: str


@dataclass
class MonitorSnapshot:
    as_of_today: date
    price_last_bar: Optional[date]
    price_age_bdays: Optional[int]
    broker_last_bar: Optional[date]
    broker_lag_bdays: Optional[int]          # broker's bar vs. the price data's bar
    environment: str
    cash: float
    equity: Optional[float]
    gross: Optional[float]
    net: Optional[float]
    peak_equity: float
    drawdown: float                           # >= 0, fraction below peak
    daily_pnl: Optional[float]
    total_pnl: Optional[float]
    gross_utilization: Optional[float]        # gross / limit
    net_utilization: Optional[float]
    positions: pd.DataFrame
    equity_history: pd.DataFrame
    working_tickets: List[orders.OrderTicket]
    pending_proposals: List[orders.OrderTicket]   # PROPOSED & risk-passed, from the latest run
    last_run: Optional[daily_run.RunRecord]
    reconciliation_breaks: List[portfolio.Break]
    kill_switch_reasons: List[str]
    alerts: List[Alert] = field(default_factory=list)

    @property
    def worst_severity(self) -> Optional[str]:
        return min((a.severity for a in self.alerts), key=SEVERITY_ORDER.get, default=None)


def expected_positions_from_tickets(tickets: Sequence[orders.OrderTicket]) -> Dict[str, float]:
    """Positions implied purely by the fills recorded on tickets (an independent cross-check)."""
    out: Dict[str, float] = {}
    for t in tickets:
        sides = {l.symbol: l.side for l in t.legs}
        for f in t.fills:
            sign = 1.0 if sides[f.symbol] == orders.OrderSide.BUY else -1.0
            out[f.symbol] = out.get(f.symbol, 0.0) + sign * f.quantity
    return {k: v for k, v in out.items() if abs(v) > orders.QTY_TOL}


def _bdays_between(earlier: Optional[date], later: Optional[date]) -> Optional[int]:
    if earlier is None or later is None:
        return None
    return 0 if earlier >= later else int(np.busday_count(earlier, later))


def _positions_table(broker: broker_paper.PaperBroker, marks: Mapping[str, float]) -> pd.DataFrame:
    rows = []
    for sym, pos in sorted(broker.portfolio.positions.items()):
        mark = marks.get(sym)
        rows.append({
            "symbol": sym, "qty": pos.qty, "avg_price": pos.avg_price, "mark": mark,
            "market_value": pos.qty * mark if mark is not None else None,
            "unrealized_pnl": pos.qty * (mark - pos.avg_price) if mark is not None else None,
        })
    return pd.DataFrame(rows, columns=["symbol", "qty", "avg_price", "mark", "market_value", "unrealized_pnl"])


def build_snapshot(
    broker: broker_paper.PaperBroker,
    risk_cfg: risk.RiskConfig,
    run_records: Sequence[daily_run.RunRecord],
    price_last_bar: Optional[date],
    today: Optional[date] = None,
    marks: Optional[Mapping[str, float]] = None,
) -> MonitorSnapshot:
    today = today or datetime.now(timezone.utc).date()
    marks = {**broker.last_prices(), **(marks or {})}

    acct = broker.get_account(marks)
    hist = broker.equity_history()
    initial = broker.config.initial_cash
    equity = acct.equity
    peak = max([initial] + hist["equity"].tolist() + ([equity] if equity is not None else []))
    drawdown = max(0.0, (peak - equity) / peak) if equity is not None and peak > 0 else 0.0
    daily_pnl = None
    if equity is not None and len(hist):
        prev = hist["equity"].iloc[-2] if len(hist) > 1 else initial
        daily_pnl = equity - float(prev)

    broker_bar = date.fromisoformat(broker.last_bar()) if broker.last_bar() else None
    runs = sorted(run_records, key=lambda r: r.started_at)
    last_run = runs[-1] if runs else None

    gross_limit = risk_cfg.max_gross_exposure_pct * risk_cfg.capital_base
    net_limit = risk_cfg.max_net_exposure_pct * risk_cfg.capital_base

    tickets = broker.list_tickets()
    working = [t for t in tickets if t.status in (orders.OrderStatus.SUBMITTED, orders.OrderStatus.PARTIALLY_FILLED)]

    pending: List[orders.OrderTicket] = []
    if last_run and last_run.status in (daily_run.STATUS_OK, daily_run.STATUS_KILL):
        decisions = {d["ticket_id"]: d for d in last_run.risk_report.get("decisions", [])}
        for t in last_run.ticket_objects():
            if (t.status == orders.OrderStatus.PROPOSED and decisions.get(t.ticket_id, {}).get("approved")
                    and broker.get_ticket(t.ticket_id) is None):
                pending.append(t)

    breaks = portfolio.reconcile(expected_positions_from_tickets(tickets), broker.get_positions())

    acct_state = risk.AccountState(
        equity=equity if equity is not None else broker.portfolio.cash,
        day_start_equity=(equity - daily_pnl) if (equity is not None and daily_pnl is not None) else None,
        peak_equity=peak)
    ks = risk.kill_switch_status(risk_cfg, acct_state)

    snap = MonitorSnapshot(
        as_of_today=today, price_last_bar=price_last_bar,
        price_age_bdays=prices.business_days_behind(price_last_bar, today) if price_last_bar else None,
        broker_last_bar=broker_bar, broker_lag_bdays=_bdays_between(broker_bar, price_last_bar),
        environment=broker.environment, cash=acct.cash, equity=equity, gross=acct.gross_exposure,
        net=acct.net_exposure, peak_equity=peak, drawdown=drawdown, daily_pnl=daily_pnl,
        total_pnl=(equity - initial) if equity is not None else None,
        gross_utilization=(acct.gross_exposure / gross_limit) if acct.gross_exposure is not None and gross_limit else None,
        net_utilization=(abs(acct.net_exposure) / net_limit) if acct.net_exposure is not None and net_limit else None,
        positions=_positions_table(broker, marks), equity_history=hist,
        working_tickets=working, pending_proposals=pending, last_run=last_run,
        reconciliation_breaks=breaks, kill_switch_reasons=ks,
    )
    snap.alerts = compute_alerts(snap, risk_cfg, tickets)
    return snap


def compute_alerts(
    snap: MonitorSnapshot, risk_cfg: risk.RiskConfig, all_tickets: Sequence[orders.OrderTicket]
) -> List[Alert]:
    a: List[Alert] = []

    # ---- data ----
    if snap.price_last_bar is None:
        a.append(Alert("critical", "no_price_data", "No price data found."))
    elif snap.price_age_bdays is not None and snap.price_age_bdays > risk_cfg.max_data_age_business_days:
        a.append(Alert("critical", "stale_prices",
                       f"Price data ends {snap.price_last_bar}, {snap.price_age_bdays} business days "
                       f"behind (limit {risk_cfg.max_data_age_business_days}). New risk is blocked."))

    # ---- daily run ----
    run = snap.last_run
    if run is None:
        a.append(Alert("warning", "no_runs", "The daily runner has never produced a run record."))
    else:
        started = datetime.fromisoformat(run.started_at).date()
        since = _bdays_between(started, snap.as_of_today)
        if since is not None and since >= RUN_OVERDUE_BDAYS:
            a.append(Alert("warning", "run_overdue",
                           f"Last daily run was {started} ({since} business days ago)."))
        if run.status == daily_run.STATUS_STALE:
            a.append(Alert("critical", "run_aborted_stale",
                           f"Last run {run.run_id} aborted: price data too old."))
        elif run.status == daily_run.STATUS_KILL:
            a.append(Alert("critical", "run_kill_switch", f"Last run {run.run_id} hit the kill switch."))
        for t in run.risk_report.get("unticketed", []):
            a.append(Alert("warning", "unticketed", f"{t['subject']}: {t['reason']}"))

    if (snap.broker_lag_bdays or 0) > 1:
        a.append(Alert("warning", "broker_behind",
                       f"Paper broker is at {snap.broker_last_bar}, price data at {snap.price_last_bar}: "
                       "run the daily runner."))

    # ---- risk state ----
    if snap.kill_switch_reasons:
        a.append(Alert("critical", "kill_switch",
                       "Kill switch active: " + "; ".join(snap.kill_switch_reasons)
                       + ". New risk is blocked; only risk-reducing tickets pass."))
    elif snap.drawdown >= NEAR_LIMIT * risk_cfg.max_drawdown_pct:
        a.append(Alert("warning", "drawdown_near_limit",
                       f"Drawdown {snap.drawdown:.2%} is {snap.drawdown / risk_cfg.max_drawdown_pct:.0%} "
                       f"of the {risk_cfg.max_drawdown_pct:.2%} limit."))
    for name, util in (("gross", snap.gross_utilization), ("net", snap.net_utilization)):
        if util is not None and util >= 1.0:
            a.append(Alert("critical", f"{name}_over_limit", f"{name.title()} exposure is {util:.0%} of its limit."))
        elif util is not None and util >= NEAR_LIMIT:
            a.append(Alert("warning", f"{name}_near_limit", f"{name.title()} exposure is {util:.0%} of its limit."))
    if snap.equity is None:
        a.append(Alert("warning", "unpriced_positions", "A held symbol has no price; equity cannot be computed."))

    # ---- integrity ----
    for b in snap.reconciliation_breaks:
        a.append(Alert("critical", "reconciliation_break",
                       f"{b.symbol}: positions imply {b.expected:g} from fills but the account holds {b.actual:g}."))

    # ---- orders ----
    for t in all_tickets:
        held = set(snap.positions["symbol"]) if len(snap.positions) else set()
        if (t.status == orders.OrderStatus.CANCELLED and t.fills
                and any(t.remaining_quantity(l.symbol) > orders.QTY_TOL for l in t.legs)
                and held & {f.symbol for f in t.fills}):   # still holding it, i.e. not since closed out
            filled = ", ".join(f"{s} {t.filled_quantity(s):g}" for s in sorted({f.symbol for f in t.fills}))
            a.append(Alert("critical", "one_legged",
                           f"Ticket {t.ticket_id} ({t.pair}) was cancelled part-filled ({filled}): "
                           "the pair may be unhedged."))
    if snap.broker_last_bar:
        for t in snap.working_tickets:
            sub = next((h.at for h in t.history if h.status == orders.OrderStatus.SUBMITTED), None)
            if sub:
                bars = _bdays_between(datetime.fromisoformat(sub).date(), snap.broker_last_bar)
                if bars is not None and bars > WORKING_TICKET_MAX_BARS:
                    a.append(Alert("warning", "stuck_ticket",
                                   f"Ticket {t.ticket_id} ({t.pair}) has been working for {bars} bars."))
    if snap.pending_proposals and run is not None and run.as_of:
        age = _bdays_between(date.fromisoformat(run.as_of), snap.as_of_today)
        sev = "warning" if (age or 0) > UNREVIEWED_PROPOSAL_BDAYS else "info"
        a.append(Alert(sev, "pending_proposals",
                       f"{len(snap.pending_proposals)} risk-approved ticket(s) from run {run.run_id} "
                       "await your approval."))

    return sorted(a, key=lambda x: (SEVERITY_ORDER[x.severity], x.code))


def exit_code(alerts: Sequence[Alert]) -> int:
    """0 = clean (info only), 1 = warnings, 2 = critical."""
    sevs = {a.severity for a in alerts}
    return 2 if "critical" in sevs else 1 if "warning" in sevs else 0

"""
core.broker_paper

A simulated broker with persistent state, implementing core.broker_base.Broker.

It keeps an account (cash, positions, fills), a book of order tickets and an append-only
audit log in one SQLite file, so everything survives restarts. All state changes for one
operation are committed in a single transaction: after a crash you see either the whole
operation or none of it.

Simulation model (deliberately simple and conservative to describe):
- Orders do not fill on submission. You advance time by calling `process_bar(prices, as_of)`
  with that bar's prices (e.g. the daily close); working tickets then try to fill against it.
- MARKET legs fill at the bar price. LIMIT legs fill (at the bar price) only if marketable:
  BUY when price <= limit, SELL when price >= limit.
- `fill_fraction` < 1 fills only that share of each leg's original quantity per bar (so 0.5
  takes two bars), to exercise partial fills; legs are processed SELL first, then BUY.
- Fees come from core.execution's cost model (commission, half-spread, slippage) and are
  charged on each fill; short borrow is charged once per new bar on the shorts held overnight.
- DAY legs still unfilled at the end of a bar are cancelled (`expire_day_orders`). A pair
  can therefore be left one-legged -- the same operational risk a real account has --
  and is visible in the ticket history and audit log.
- No margin or buying-power check: cash may go negative. The risk layer is what limits size.

Typical usage:

    broker = PaperBroker(state_dir() / "rasa_paper.db", initial_cash=100_000)
    t = broker.submit_ticket(approved_ticket)
    broker.process_bar({"X": 50.0, "Y": 100.0}, as_of="2026-11-12")
    broker.get_positions()
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

import numpy as np
import pandas as pd

try:
    import core.broker_base as broker_base
    import core.execution as execution
    import core.orders as orders
    import core.portfolio as portfolio
except ModuleNotFoundError:  # run from inside core/
    import broker_base  # type: ignore
    import execution  # type: ignore
    import orders  # type: ignore
    import portfolio  # type: ignore

BrokerError = broker_base.BrokerError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta   (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tickets(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,
    body TEXT NOT NULL,
    updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL, kind TEXT NOT NULL, ticket_id TEXT, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS equity_history(
    as_of TEXT PRIMARY KEY, equity REAL NOT NULL, cash REAL NOT NULL,
    gross REAL NOT NULL, net REAL NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date(x) -> str:
    return pd.Timestamp(x).date().isoformat()


@dataclass(frozen=True)
class PaperBrokerConfig:
    """Fixed for the life of an account; stored in the DB and verified on reopen."""

    initial_cash: float = 100_000.0
    fill_fraction: float = 1.0            # share of each leg's original qty filled per bar, (0, 1]
    expire_day_orders: bool = True
    commission_per_share: float = 0.005
    commission_bps: float = 0.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    borrow_bps_annual: float = 50.0

    def __post_init__(self) -> None:
        if self.initial_cash < 0:
            raise ValueError("initial_cash must be >= 0")
        if not 0 < self.fill_fraction <= 1:
            raise ValueError("fill_fraction must be in (0, 1]")

    def cost_cfg(self) -> execution.ExecutionCostConfig:
        return execution.ExecutionCostConfig(
            commission_per_share=self.commission_per_share, commission_bps=self.commission_bps,
            half_spread_bps=self.half_spread_bps, slippage_bps=self.slippage_bps,
            borrow_bps_annual=self.borrow_bps_annual,
        )


class PaperBroker(broker_base.Broker):
    environment = "paper"

    def __init__(self, db_path, config: Optional[PaperBrokerConfig] = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

        stored = self._meta("config")
        if stored is None:
            self.config = config or PaperBrokerConfig()
            with self._tx():
                self._set_meta("config", json.dumps(asdict(self.config), sort_keys=True))
                self._set_meta("created_at", _now())
                self.portfolio = portfolio.Portfolio(self.config.initial_cash)
                self._save_portfolio()
                self._event("account_created", None, f"initial_cash={self.config.initial_cash}")
        else:
            self.config = PaperBrokerConfig(**json.loads(stored))
            if config is not None and config != self.config:
                raise BrokerError(
                    "this account was created with a different PaperBrokerConfig; "
                    "open it without a config, or use a new database file"
                )
            self.portfolio = portfolio.Portfolio.from_json(self._meta("portfolio"))

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def _tx(self) -> Iterator[None]:
        with self._conn:  # commit on success, rollback on exception
            yield

    def close(self) -> None:
        self._conn.close()

    def _meta(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def _save_portfolio(self) -> None:
        self._set_meta("portfolio", self.portfolio.to_json(indent=None))

    def _event(self, kind: str, ticket_id: Optional[str], detail: str) -> None:
        self._conn.execute(
            "INSERT INTO events(at, kind, ticket_id, detail) VALUES(?,?,?,?)",
            (_now(), kind, ticket_id, detail))

    def _store_ticket(self, t: orders.OrderTicket) -> None:
        self._conn.execute(
            "INSERT INTO tickets(ticket_id, status, body, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(ticket_id) DO UPDATE SET status=excluded.status, "
            "body=excluded.body, updated_at=excluded.updated_at",
            (t.ticket_id, t.status.value, t.to_json(indent=None), _now()))

    # ------------------------------------------------------------------ Broker API
    def submit_ticket(self, ticket: orders.OrderTicket) -> orders.OrderTicket:
        existing = self.get_ticket(ticket.ticket_id)
        if existing is not None:
            with self._tx():
                self._event("duplicate_submit_ignored", ticket.ticket_id,
                            f"already {existing.status.value}")
            return existing
        if ticket.status != orders.OrderStatus.APPROVED:
            raise BrokerError(
                f"ticket {ticket.ticket_id} is {ticket.status.value}; only APPROVED tickets can be submitted")
        submitted = ticket.submit(f"accepted by {self.environment} broker")
        with self._tx():
            self._store_ticket(submitted)
            self._event("submitted", ticket.ticket_id, ", ".join(
                f"{l.side.value} {l.quantity:g} {l.symbol}" for l in ticket.legs))
        return submitted

    def cancel_ticket(self, ticket_id: str, reason: str = "") -> orders.OrderTicket:
        t = self.get_ticket(ticket_id)
        if t is None:
            raise BrokerError(f"unknown ticket {ticket_id}")
        try:
            cancelled = t.cancel(reason or "cancelled by user")
        except orders.InvalidTransition as e:
            raise BrokerError(str(e)) from e
        with self._tx():
            self._store_ticket(cancelled)
            self._event("cancelled", ticket_id, reason)
        return cancelled

    def get_ticket(self, ticket_id: str) -> Optional[orders.OrderTicket]:
        row = self._conn.execute("SELECT body FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        return orders.OrderTicket.from_json(row["body"]) if row else None

    def list_tickets(self, status: Optional[orders.OrderStatus] = None) -> List[orders.OrderTicket]:
        if status is None:
            rows = self._conn.execute("SELECT body FROM tickets ORDER BY seq").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT body FROM tickets WHERE status=? ORDER BY seq", (status.value,)).fetchall()
        return [orders.OrderTicket.from_json(r["body"]) for r in rows]

    def get_positions(self) -> Dict[str, float]:
        return self.portfolio.quantities()

    def get_account(self, prices: Optional[Mapping[str, float]] = None) -> broker_base.AccountInfo:
        marks = {**self.last_prices(), **(prices or {})}
        try:
            v = self.portfolio.value(marks)
            equity, gross, net = v.equity, v.gross, v.net
        except KeyError:
            equity = gross = net = None
        return broker_base.AccountInfo(
            cash=self.portfolio.cash, equity=equity, gross_exposure=gross, net_exposure=net,
            positions=self.get_positions(), environment=self.environment)

    # ------------------------------------------------------------------ paper-only
    def last_bar(self) -> Optional[str]:
        return self._meta("last_bar")

    def last_prices(self) -> Dict[str, float]:
        raw = self._meta("last_prices")
        return json.loads(raw) if raw else {}

    def working_tickets(self) -> List[orders.OrderTicket]:
        return [t for t in self.list_tickets()
                if t.status in (orders.OrderStatus.SUBMITTED, orders.OrderStatus.PARTIALLY_FILLED)]

    def events(self, ticket_id: Optional[str] = None) -> pd.DataFrame:
        q, args = "SELECT at, kind, ticket_id, detail FROM events", ()
        if ticket_id:
            q, args = q + " WHERE ticket_id=?", (ticket_id,)
        return pd.DataFrame([dict(r) for r in self._conn.execute(q + " ORDER BY id", args)],
                            columns=["at", "kind", "ticket_id", "detail"])

    def equity_history(self) -> pd.DataFrame:
        rows = self._conn.execute("SELECT * FROM equity_history ORDER BY as_of").fetchall()
        df = pd.DataFrame([dict(r) for r in rows], columns=["as_of", "equity", "cash", "gross", "net"])
        return df.set_index("as_of")

    def reconcile(self, expected: Mapping[str, float], tol: float = 1e-6) -> List[portfolio.Break]:
        """Breaks between `expected` positions and what the broker actually holds."""
        breaks = portfolio.reconcile(expected, self.get_positions(), tol)
        with self._tx():
            self._event("reconciliation", None,
                        "ok" if not breaks else "; ".join(
                            f"{b.symbol}: expected {b.expected:g}, actual {b.actual:g}" for b in breaks))
        return breaks

    @staticmethod
    def _marketable(leg: orders.OrderLeg, price: float) -> bool:
        if leg.order_type == orders.OrderType.MARKET:
            return True
        if leg.side == orders.OrderSide.BUY:
            return price <= leg.limit_price + 1e-12
        return price >= leg.limit_price - 1e-12

    def process_bar(
        self, prices: Mapping[str, float], as_of
    ) -> List[orders.OrderTicket]:
        """
        Advance the simulation to one bar: charge overnight borrow, try to fill every
        working ticket against `prices`, expire unfilled DAY legs, mark the account and
        record an equity point. Returns the tickets that changed. Bars must not go
        backwards; each leg fills at most once per bar, so re-processing a bar is a no-op.
        """
        bar = _date(as_of)
        last = self.last_bar()
        if last is not None and bar < last:
            raise BrokerError(f"bar {bar} is before the last processed bar {last}")
        clean = {s: float(p) for s, p in prices.items() if p is not None and math.isfinite(p) and p > 0}
        cost_cfg = self.config.cost_cfg()
        changed: List[orders.OrderTicket] = []

        # Work on copies so a failure rolls the in-memory state back with the DB.
        pf_backup = self.portfolio.to_json(indent=None)
        try:
            with self._tx():
                prev_prices = self.last_prices()
                if last is not None and bar > last and self.portfolio.positions:
                    days = int(np.busday_count(last, bar))
                    if days > 0:
                        charge = self.portfolio.accrue_borrow(
                            prev_prices, self.config.borrow_bps_annual, days=days)
                        if charge:
                            self._event("borrow", None, f"{charge:.4f} for {days} day(s)")

                for ticket in self.working_tickets():
                    t = ticket
                    for leg in sorted(t.legs, key=lambda l: l.side == orders.OrderSide.BUY):
                        px = clean.get(leg.symbol)
                        remaining = t.remaining_quantity(leg.symbol)
                        if px is None or remaining <= orders.QTY_TOL or not self._marketable(leg, px):
                            continue
                        if any(f.symbol == leg.symbol and f.at == bar for f in t.fills):
                            continue  # at most one fill per leg per bar (makes re-processing a bar a no-op)
                        qty = min(remaining, leg.quantity * self.config.fill_fraction)
                        if remaining - qty <= 1e-9:
                            qty = remaining
                        fill = orders.Fill(leg.symbol, qty, px, at=bar)
                        t = t.record_fill(fill)
                        signed = qty if leg.side == orders.OrderSide.BUY else -qty
                        fee = execution.trade_cost(qty, px, cost_cfg)["total"]
                        self.portfolio.apply_fill(leg.symbol, signed, px, fee, at=bar, ticket_id=t.ticket_id)
                        self._event("fill", t.ticket_id,
                                    f"{leg.side.value} {qty:g} {leg.symbol} @ {px:g} fee {fee:.4f}")

                    if self.config.expire_day_orders and not t.is_terminal:
                        open_day = [l for l in t.legs if l.time_in_force == orders.TimeInForce.DAY
                                    and t.remaining_quantity(l.symbol) > orders.QTY_TOL]
                        if open_day:
                            t = t.cancel("DAY order expired unfilled: "
                                         + ", ".join(l.symbol for l in open_day))
                            self._event("expired", t.ticket_id, t.history[-1].reason)

                    if t is not ticket:
                        self._store_ticket(t)
                        changed.append(t)

                marks = {**prev_prices, **clean}
                self._set_meta("last_prices", json.dumps(marks, sort_keys=True))
                self._set_meta("last_bar", bar)
                self._save_portfolio()
                try:
                    v = self.portfolio.value(marks)
                except KeyError as e:
                    self._event("mark_failed", None, str(e))
                else:
                    self._conn.execute(
                        "INSERT INTO equity_history(as_of, equity, cash, gross, net) VALUES(?,?,?,?,?) "
                        "ON CONFLICT(as_of) DO UPDATE SET equity=excluded.equity, cash=excluded.cash, "
                        "gross=excluded.gross, net=excluded.net",
                        (bar, v.equity, v.cash, v.gross, v.net))
        except Exception:
            self.portfolio = portfolio.Portfolio.from_json(pf_backup)
            raise
        return changed

"""
core.portfolio

Leg-level portfolio accounting: cash, positions, fills, exposures and PnL per symbol.

This replaces the scalar "spread unit" PnL of core.strategy as the source of truth for
anything that will touch an account (paper broker, reconciliation, live monitoring).
PnL here comes from actual share counts and fill prices, never from a spread difference.

Accounting model
- Cash-and-positions ledger. BUY: cash -= qty*price + fee. SELL: cash += qty*price - fee
  (short sale proceeds are credited to cash). Equity = cash + sum(qty * mark).
- Average-cost positions. Reducing/closing realizes PnL against the average entry price;
  a flip closes the old side and opens the remainder at the fill price.
- Fees (commission, half-spread, slippage) and short-borrow charges are tracked separately
  from trading PnL, and always reconcile:

      equity - initial_cash == realized_pnl + unrealized_pnl - fees_paid - borrow_paid

- NOT modelled: margin / buying power, interest on cash, dividends on short positions
  (adjusted-close prices already embed dividends for longs), corporate actions, tax lots.

Also here:
- `simulate_targets`: replay a table of target share positions through the ledger over a
  price panel (trade at the close, with execution costs, fill delay and daily borrow).
- `targets_from_signals`: turn pair signals into fixed-share leg targets (sized once at
  entry, not rebalanced daily, like a real position).
- `reconcile`: compare expected vs. actual positions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import core.execution as execution
    import core.orders as orders
    import core.trade_proposals as trade_proposals
except ModuleNotFoundError:  # run from inside core/
    import execution  # type: ignore
    import orders  # type: ignore
    import trade_proposals  # type: ignore

QTY_TOL = orders.QTY_TOL


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Position:
    qty: float = 0.0        # signed shares
    avg_price: float = 0.0  # average entry price of the open quantity

    def to_dict(self) -> Dict[str, float]:
        return {"qty": self.qty, "avg_price": self.avg_price}


@dataclass(frozen=True)
class TradeRecord:
    """One executed fill as booked in the ledger."""

    at: str
    symbol: str
    qty: float            # signed: + buy, - sell
    price: float
    fee: float
    realized_pnl: float   # PnL realized by this trade (before fees)
    ticket_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Valuation:
    equity: float
    cash: float
    long_value: float
    short_value: float      # negative number
    gross: float
    net: float
    unrealized_pnl: float
    realized_pnl: float
    fees_paid: float
    borrow_paid: float

    @property
    def total_pnl(self) -> float:
        """Net of fees and borrow."""
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid - self.borrow_paid


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """Mutable ledger of cash, positions and trades."""

    def __init__(self, initial_cash: float) -> None:
        if initial_cash < 0:
            raise ValueError("initial_cash must be >= 0")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.borrow_paid = 0.0

    # ---- views ----
    def quantities(self) -> Dict[str, float]:
        """Symbol -> signed shares; the input `orders.plan_orders` expects."""
        return {s: p.qty for s, p in self.positions.items()}

    def quantity(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.qty if p else 0.0

    # ---- booking ----
    def apply_fill(
        self,
        symbol: str,
        qty: float,
        price: float,
        fee: float = 0.0,
        at: Optional[str] = None,
        ticket_id: Optional[str] = None,
    ) -> TradeRecord:
        """
        Book a fill. `qty` is signed (+ buy, - sell); `fee` is a non-negative dollar cost.
        Returns the ledger record.
        """
        if not np.isfinite(qty) or abs(qty) <= QTY_TOL:
            raise ValueError("fill qty must be a non-zero finite number")
        if not (np.isfinite(price) and price > 0):
            raise ValueError("fill price must be positive")
        if fee < 0:
            raise ValueError("fee must be >= 0")

        pos = self.positions.setdefault(symbol, Position())
        q0, avg0 = pos.qty, pos.avg_price
        realized = 0.0

        if abs(q0) <= QTY_TOL or np.sign(q0) == np.sign(qty):
            new_q = q0 + qty
            pos.avg_price = (abs(q0) * avg0 + abs(qty) * price) / abs(new_q)
            pos.qty = new_q
        else:
            closed = min(abs(qty), abs(q0))
            realized = closed * (price - avg0) * np.sign(q0)
            new_q = q0 + qty
            if abs(new_q) <= QTY_TOL:
                pos.qty, pos.avg_price = 0.0, 0.0
            elif np.sign(new_q) == np.sign(q0):     # partial reduce: basis unchanged
                pos.qty = new_q
            else:                                    # flipped through zero
                pos.qty, pos.avg_price = new_q, price
        if abs(pos.qty) <= QTY_TOL:
            del self.positions[symbol]

        self.cash += -qty * price - fee
        self.realized_pnl += realized
        self.fees_paid += fee
        rec = TradeRecord(at or _now(), symbol, float(qty), float(price), float(fee),
                          float(realized), ticket_id)
        self.trades.append(rec)
        return rec

    def apply_ticket_fills(
        self,
        ticket: orders.OrderTicket,
        cost_cfg: Optional[execution.ExecutionCostConfig] = None,
        already_booked: int = 0,
    ) -> List[TradeRecord]:
        """
        Book a ticket's fills (skipping the first `already_booked`, so it can be called
        repeatedly as more fills arrive). Fees come from `cost_cfg` if given.
        """
        sides = {l.symbol: l.side for l in ticket.legs}
        out: List[TradeRecord] = []
        for f in ticket.fills[already_booked:]:
            signed = f.quantity if sides[f.symbol] == orders.OrderSide.BUY else -f.quantity
            fee = execution.trade_cost(f.quantity, f.price, cost_cfg)["total"] if cost_cfg else 0.0
            out.append(self.apply_fill(f.symbol, signed, f.price, fee, f.at, ticket.ticket_id))
        return out

    def accrue_borrow(
        self,
        prices: Mapping[str, float],
        annual_bps: float,
        days: float = 1.0,
        trading_days_per_year: int = 252,
    ) -> float:
        """Charge short-borrow on the market value of short positions; returns the charge."""
        short_value = sum(
            -p.qty * prices[s] for s, p in self.positions.items() if p.qty < 0 and s in prices
        )
        charge = short_value * annual_bps / 1e4 * days / trading_days_per_year
        self.cash -= charge
        self.borrow_paid += charge
        return charge

    # ---- valuation ----
    def value(self, prices: Mapping[str, float]) -> Valuation:
        """Mark to `prices`. Every held symbol needs a price (no silent zeroing)."""
        missing = sorted(s for s in self.positions if s not in prices)
        if missing:
            raise KeyError(f"no price to mark: {', '.join(missing)}")
        long_v = sum(p.qty * prices[s] for s, p in self.positions.items() if p.qty > 0)
        short_v = sum(p.qty * prices[s] for s, p in self.positions.items() if p.qty < 0)
        unreal = sum(p.qty * (prices[s] - p.avg_price) for s, p in self.positions.items())
        return Valuation(
            equity=self.cash + long_v + short_v, cash=self.cash,
            long_value=long_v, short_value=short_v,
            gross=long_v - short_v, net=long_v + short_v,
            unrealized_pnl=unreal, realized_pnl=self.realized_pnl,
            fees_paid=self.fees_paid, borrow_paid=self.borrow_paid,
        )

    def symbol_pnl(self, prices: Mapping[str, float]) -> pd.DataFrame:
        """Per-symbol realized (booked trades) and unrealized PnL, before fees."""
        realized: Dict[str, float] = {}
        for t in self.trades:
            realized[t.symbol] = realized.get(t.symbol, 0.0) + t.realized_pnl
        rows = []
        for s in sorted(set(realized) | set(self.positions)):
            p = self.positions.get(s)
            unreal = p.qty * (prices[s] - p.avg_price) if p else 0.0
            rows.append({"symbol": s, "qty": p.qty if p else 0.0,
                         "realized_pnl": realized.get(s, 0.0), "unrealized_pnl": unreal})
        return pd.DataFrame(rows, columns=["symbol", "qty", "realized_pnl", "unrealized_pnl"])

    # ---- persistence ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_cash": self.initial_cash, "cash": self.cash,
            "realized_pnl": self.realized_pnl, "fees_paid": self.fees_paid,
            "borrow_paid": self.borrow_paid,
            "positions": {s: p.to_dict() for s, p in sorted(self.positions.items())},
            "trades": [t.to_dict() for t in self.trades],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Portfolio":
        p = cls(d["initial_cash"])
        p.cash, p.realized_pnl = d["cash"], d["realized_pnl"]
        p.fees_paid, p.borrow_paid = d["fees_paid"], d["borrow_paid"]
        p.positions = {s: Position(v["qty"], v["avg_price"]) for s, v in d["positions"].items()}
        p.trades = [TradeRecord(**t) for t in d["trades"]]
        return p

    @classmethod
    def from_json(cls, s: str) -> "Portfolio":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Break:
    symbol: str
    expected: float
    actual: float

    @property
    def difference(self) -> float:
        return self.actual - self.expected


def reconcile(
    expected: Mapping[str, float], actual: Mapping[str, float], tol: float = 1e-6
) -> List[Break]:
    """Symbols whose actual position differs from the expected one by more than `tol` shares."""
    out = []
    for s in sorted(set(expected) | set(actual)):
        e, a = float(expected.get(s, 0.0)), float(actual.get(s, 0.0))
        if abs(a - e) > tol:
            out.append(Break(s, e, a))
    return out


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    equity: pd.Series
    cash: pd.Series
    gross: pd.Series
    net: pd.Series
    positions: pd.DataFrame     # signed shares held after each day's trading
    trades: pd.DataFrame
    summary: Dict[str, float]
    portfolio: Portfolio


def simulate_targets(
    px: pd.DataFrame,
    targets: pd.DataFrame,
    initial_cash: float = 100_000.0,
    cost_cfg: Optional[execution.ExecutionCostConfig] = None,
) -> SimulationResult:
    """
    Replay target share positions through the ledger.

    Each day: (1) charge borrow on the shorts held overnight, valued at the previous
    close, (2) trade from current holdings to that day's target at the close price, paying
    execution costs, (3) mark to the close. `targets` rows are signed share targets by
    date (NaN = keep current holding); a fill delay from `cost_cfg` shifts them later.
    Targets before the first price date are ignored.

    Parameters
    ----------
    px : price panel (date x symbol); must cover every symbol that is ever held.
    targets : DataFrame indexed by date, columns = symbols, signed shares.
    """
    cost_cfg = cost_cfg or execution.ExecutionCostConfig.zero()
    px = px.sort_index()
    tg = targets.reindex(px.index)
    if cost_cfg.fill_delay_days:
        tg = tg.shift(cost_cfg.fill_delay_days)
    unknown = sorted(set(tg.columns) - set(px.columns))
    if unknown:
        raise KeyError(f"targets reference symbols without prices: {unknown}")

    pf = Portfolio(initial_cash)
    equity, cash_s, gross_s, net_s = {}, {}, {}, {}
    held_rows: Dict[pd.Timestamp, Dict[str, float]] = {}
    prev_prices: Dict[str, float] = {}

    for date, prices_row in px.iterrows():
        prices = prices_row.dropna().to_dict()
        if prev_prices and pf.positions:
            pf.accrue_borrow(prev_prices, cost_cfg.borrow_bps_annual,
                             trading_days_per_year=cost_cfg.trading_days_per_year)

        row = tg.loc[date].dropna()
        for sym, tgt in row.items():
            delta = float(tgt) - pf.quantity(sym)
            if abs(delta) <= QTY_TOL:
                continue
            price = prices.get(sym)
            if price is None:
                raise KeyError(f"{date.date()}: target for {sym} but no price")
            fee = execution.trade_cost(delta, price, cost_cfg)["total"]
            pf.apply_fill(sym, delta, price, fee, at=date.isoformat())

        v = pf.value({**prev_prices, **prices} if pf.positions else prices)
        equity[date], cash_s[date], gross_s[date], net_s[date] = v.equity, v.cash, v.gross, v.net
        held_rows[date] = pf.quantities()
        prev_prices = {**prev_prices, **prices}

    eq = pd.Series(equity, name="equity")
    final = pf.value(prev_prices)
    rets = eq.pct_change().fillna(0.0)
    vol = float(rets.std(ddof=1)) * np.sqrt(252) if len(eq) > 1 else 0.0
    n = len(eq)
    total_ret = float(eq.iloc[-1] / initial_cash - 1.0) if n else 0.0
    ann_ret = (1.0 + total_ret) ** (252 / max(n, 1)) - 1.0 if n else 0.0
    summary = {
        "n_days": n,
        "total_return": total_ret,
        "annual_return": ann_ret,
        "annual_vol": vol,
        "sharpe": ann_ret / vol if vol > 0 else float("nan"),
        "max_drawdown": float((eq / eq.cummax() - 1.0).min()) if n else 0.0,
        "realized_pnl": final.realized_pnl,
        "unrealized_pnl": final.unrealized_pnl,
        "fees_paid": final.fees_paid,
        "borrow_paid": final.borrow_paid,
        "n_trades": len(pf.trades),
    }
    return SimulationResult(
        equity=eq,
        cash=pd.Series(cash_s, name="cash"),
        gross=pd.Series(gross_s, name="gross"),
        net=pd.Series(net_s, name="net"),
        positions=pd.DataFrame(held_rows).T.reindex(px.index).fillna(0.0),
        trades=pd.DataFrame([t.to_dict() for t in pf.trades]),
        summary=summary,
        portfolio=pf,
    )


def targets_from_signals(
    px: pd.DataFrame,
    pair_signals: Mapping[Tuple[str, str], Tuple[pd.Series, float]],
    notional_per_pair: float,
    log_space: bool = False,
) -> pd.DataFrame:
    """
    Fixed-share leg targets from pair position signals.

    `pair_signals` maps (x, y) -> (signal series of -1/0/+1 indexed by date, beta).
    A pair is sized with `trade_proposals.size_pair` at the close where its signal
    changes and the share count is then held constant until the signal changes again
    (no daily rebalancing). Pairs sharing a symbol add up in that symbol's target.
    """
    out = pd.DataFrame(0.0, index=px.index, columns=sorted({s for xy in pair_signals for s in xy}))
    for (x, y), (signal, beta) in pair_signals.items():
        sig = signal.reindex(px.index).fillna(0.0)
        qx = qy = 0.0
        prev = 0.0
        for date, pos in sig.items():
            if pos != prev:
                qx, qy = (0.0, 0.0) if pos == 0 else trade_proposals.size_pair(
                    int(np.sign(pos)), float(px.at[date, x]), float(px.at[date, y]),
                    beta, notional_per_pair, log_space)
                prev = pos
            out.at[date, x] += qx
            out.at[date, y] += qy
    return out

"""
core.orders

Order schema and the target-position -> delta-order step.

Flow (see the handover doc, Phase 2/3):

    TradeIdea (target positions)          core.trade_proposals
        + current account positions
        -> plan_orders()                  <- this module
        -> OrderTicket(s) in PROPOSED     (risk gates and human approval come later)
        -> broker adapter                 (core.broker_*, not built yet)

Design points:
- **Targets, not signals.** A TradeIdea says what should be held. Orders are only the
  *difference* between that and what is actually held, so re-running with the same
  account state never doubles a position, and a pair that is already on gives no order.
- **Deterministic ids.** `ticket_id` is a hash of the idea, the legs and the data date,
  so the same plan always produces the same ticket (the basis for idempotent runs).
  `client_order_id` per leg is derived from it for broker-side de-duplication.
- **Explicit lifecycle.** Tickets move through a validated state machine
  (PROPOSED -> APPROVED -> SUBMITTED -> PARTIALLY_FILLED/FILLED, or REJECTED/CANCELLED)
  and keep a history of every transition.
- **Stable serialization.** `to_dict` / `to_json` / `from_dict` round-trip exactly.
- **Conservative about conflicts.** If two ideas want the same symbol the plan does not
  guess; those ideas are reported in `OrderPlan.conflicts` for the (future) risk layer.
- **No orders from nothing.** Positions the system did not create are never touched:
  close-out orders are only generated for `managed_symbols` the caller names.

Quantities are signed shares in positions (+long / -short) and positive in order legs
(the side carries the direction). This module has no broker or UI dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from core.trade_proposals import TradeIdea
except ModuleNotFoundError:  # run from inside core/
    from trade_proposals import TradeIdea  # type: ignore


QTY_TOL = 1e-9
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"


class OrderStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = frozenset(
    {OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED}
)

_ALLOWED: Dict[OrderStatus, frozenset] = {
    OrderStatus.PROPOSED: frozenset(
        {OrderStatus.APPROVED, OrderStatus.REJECTED, OrderStatus.CANCELLED}
    ),
    OrderStatus.APPROVED: frozenset(
        {OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.CANCELLED}
    ),
    OrderStatus.SUBMITTED: frozenset(
        {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
         OrderStatus.REJECTED, OrderStatus.CANCELLED}
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED}
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
}


class InvalidTransition(ValueError):
    """Raised for a status change the lifecycle does not allow."""


# ---------------------------------------------------------------------------
# Legs, fills, tickets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderLeg:
    """One instrument's order. `quantity` is positive; `side` gives the direction."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    reference_price: Optional[float] = None  # price the size was computed at (not a limit)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")
        if not (isinstance(self.quantity, (int, float)) and math.isfinite(self.quantity)
                and self.quantity > QTY_TOL):
            raise ValueError(f"quantity must be a positive finite number, got {self.quantity!r}")
        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None or not (math.isfinite(self.limit_price) and self.limit_price > 0):
                raise ValueError("LIMIT orders need a positive limit_price")
        elif self.limit_price is not None:
            raise ValueError("limit_price is only valid for LIMIT orders")

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == OrderSide.BUY else -self.quantity

    @property
    def notional(self) -> Optional[float]:
        """Estimated dollar value at the reference (or limit) price."""
        price = self.limit_price if self.limit_price is not None else self.reference_price
        return None if price is None else self.quantity * price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "time_in_force": self.time_in_force.value,
            "reference_price": self.reference_price,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OrderLeg":
        return cls(
            symbol=d["symbol"],
            side=OrderSide(d["side"]),
            quantity=float(d["quantity"]),
            order_type=OrderType(d.get("order_type", "MARKET")),
            limit_price=d.get("limit_price"),
            time_in_force=TimeInForce(d.get("time_in_force", "DAY")),
            reference_price=d.get("reference_price"),
        )


@dataclass(frozen=True)
class Fill:
    """An execution against one leg (symbol), reported by a broker."""

    symbol: str
    quantity: float  # positive shares filled
    price: float
    at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.quantity <= 0 or not math.isfinite(self.quantity):
            raise ValueError("fill quantity must be positive")
        if self.price <= 0 or not math.isfinite(self.price):
            raise ValueError("fill price must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "quantity": self.quantity, "price": self.price, "at": self.at}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Fill":
        return cls(d["symbol"], float(d["quantity"]), float(d["price"]), d["at"])


@dataclass(frozen=True)
class StatusChange:
    status: OrderStatus
    at: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "at": self.at, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StatusChange":
        return cls(OrderStatus(d["status"]), d["at"], d.get("reason", ""))


def compute_ticket_id(
    idea_id: Optional[str], legs: Sequence[OrderLeg], as_of: str, purpose: str = "idea"
) -> str:
    """Deterministic id: same idea + legs + data date -> same ticket."""
    payload = json.dumps(
        {
            "idea": idea_id,
            "purpose": purpose,
            "as_of": as_of,
            "legs": sorted(
                (l.symbol, l.side.value, round(l.quantity, 6), l.order_type.value, l.limit_price)
                for l in legs
            ),
        },
        sort_keys=True,
    )
    return "T" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:14]


@dataclass(frozen=True)
class OrderTicket:
    """
    A group of legs that should be approved and executed together (normally the
    two legs of one pair). Immutable: lifecycle methods return a new ticket.
    """

    ticket_id: str
    legs: Tuple[OrderLeg, ...]
    as_of: str                       # ISO date of the price data the ticket is based on
    idea_id: Optional[str] = None    # the TradeIdea this implements (None for close-outs)
    pair: Optional[str] = None
    purpose: str = "idea"            # "idea" | "close_out"
    status: OrderStatus = OrderStatus.PROPOSED
    history: Tuple[StatusChange, ...] = ()
    fills: Tuple[Fill, ...] = ()
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("a ticket needs at least one leg")
        symbols = [l.symbol for l in self.legs]
        if len(set(symbols)) != len(symbols):
            raise ValueError("a ticket cannot hold two legs for the same symbol")
        if self.purpose not in ("idea", "close_out"):
            raise ValueError(f"unknown purpose {self.purpose!r}")

    # ---- construction ----
    @classmethod
    def create(
        cls,
        legs: Sequence[OrderLeg],
        as_of: str,
        idea_id: Optional[str] = None,
        pair: Optional[str] = None,
        purpose: str = "idea",
        notes: Sequence[str] = (),
        at: Optional[str] = None,
    ) -> "OrderTicket":
        legs = tuple(legs)
        return cls(
            ticket_id=compute_ticket_id(idea_id, legs, as_of, purpose),
            legs=legs, as_of=as_of, idea_id=idea_id, pair=pair, purpose=purpose,
            history=(StatusChange(OrderStatus.PROPOSED, at or _now(), "created"),),
            notes=tuple(notes),
        )

    # ---- views ----
    def client_order_id(self, symbol: str) -> str:
        """Stable per-leg id to hand to a broker for de-duplication."""
        return f"{self.ticket_id}-{symbol}"

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def gross_notional(self) -> Optional[float]:
        values = [l.notional for l in self.legs]
        return None if any(v is None for v in values) else sum(abs(v) for v in values)

    @property
    def net_notional(self) -> Optional[float]:
        values = [(l.notional, l.side) for l in self.legs]
        if any(v is None for v, _ in values):
            return None
        return sum(v if s == OrderSide.BUY else -v for v, s in values)

    def filled_quantity(self, symbol: str) -> float:
        return sum(f.quantity for f in self.fills if f.symbol == symbol)

    def remaining_quantity(self, symbol: str) -> float:
        leg = next(l for l in self.legs if l.symbol == symbol)
        return max(0.0, leg.quantity - self.filled_quantity(symbol))

    # ---- lifecycle ----
    def transition(self, new_status: OrderStatus, reason: str = "", at: Optional[str] = None) -> "OrderTicket":
        if new_status not in _ALLOWED[self.status]:
            raise InvalidTransition(f"{self.status.value} -> {new_status.value} is not allowed")
        change = StatusChange(new_status, at or _now(), reason)
        return replace(self, status=new_status, history=self.history + (change,))

    def approve(self, reason: str = "", at: Optional[str] = None) -> "OrderTicket":
        return self.transition(OrderStatus.APPROVED, reason, at)

    def submit(self, reason: str = "", at: Optional[str] = None) -> "OrderTicket":
        return self.transition(OrderStatus.SUBMITTED, reason, at)

    def reject(self, reason: str, at: Optional[str] = None) -> "OrderTicket":
        return self.transition(OrderStatus.REJECTED, reason, at)

    def cancel(self, reason: str = "", at: Optional[str] = None) -> "OrderTicket":
        return self.transition(OrderStatus.CANCELLED, reason, at)

    def record_fill(self, fill: Fill) -> "OrderTicket":
        """
        Add an execution. Only valid while SUBMITTED / PARTIALLY_FILLED. Moves the
        ticket to PARTIALLY_FILLED, or FILLED once every leg is completely filled.
        Overfills are rejected.
        """
        if self.status not in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED):
            raise InvalidTransition(f"cannot record a fill while {self.status.value}")
        if fill.symbol not in {l.symbol for l in self.legs}:
            raise ValueError(f"{fill.symbol} is not a leg of ticket {self.ticket_id}")
        if fill.quantity > self.remaining_quantity(fill.symbol) + QTY_TOL:
            raise ValueError(
                f"fill of {fill.quantity} exceeds remaining {self.remaining_quantity(fill.symbol)} "
                f"for {fill.symbol}"
            )
        with_fill = replace(self, fills=self.fills + (fill,))
        complete = all(with_fill.remaining_quantity(l.symbol) <= QTY_TOL for l in with_fill.legs)
        return with_fill.transition(
            OrderStatus.FILLED if complete else OrderStatus.PARTIALLY_FILLED,
            reason=f"fill {fill.quantity} {fill.symbol} @ {fill.price}",
            at=fill.at,
        )

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ticket_id": self.ticket_id,
            "idea_id": self.idea_id,
            "pair": self.pair,
            "purpose": self.purpose,
            "as_of": self.as_of,
            "status": self.status.value,
            "legs": [l.to_dict() for l in self.legs],
            "fills": [f.to_dict() for f in self.fills],
            "history": [h.to_dict() for h in self.history],
            "notes": list(self.notes),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OrderTicket":
        if d.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(f"unsupported ticket schema_version {d.get('schema_version')!r}")
        return cls(
            ticket_id=d["ticket_id"],
            legs=tuple(OrderLeg.from_dict(l) for l in d["legs"]),
            as_of=d["as_of"],
            idea_id=d.get("idea_id"),
            pair=d.get("pair"),
            purpose=d.get("purpose", "idea"),
            status=OrderStatus(d["status"]),
            history=tuple(StatusChange.from_dict(h) for h in d.get("history", [])),
            fills=tuple(Fill.from_dict(f) for f in d.get("fills", [])),
            notes=tuple(d.get("notes", [])),
        )

    @classmethod
    def from_json(cls, s: str) -> "OrderTicket":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Target positions -> delta orders
# ---------------------------------------------------------------------------

Positions = Mapping[str, float]  # symbol -> signed shares


@dataclass
class PlanIssue:
    """An idea (or close-out) that did not become a ticket, and why."""

    subject: str   # pair, or "close_out"
    reason: str
    idea_id: Optional[str] = None


@dataclass
class OrderPlan:
    tickets: List[OrderTicket]
    already_at_target: List[PlanIssue]   # nothing to do: holdings already match
    conflicts: List[PlanIssue]           # not ticketed; needs a decision (overlap, no price, ...)
    target_positions: Dict[str, float]
    current_positions: Dict[str, float]
    as_of: str

    @property
    def plan_id(self) -> str:
        payload = json.dumps(
            {"as_of": self.as_of, "tickets": sorted(t.ticket_id for t in self.tickets)},
            sort_keys=True,
        )
        return "P" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def delta_legs(
    target: Positions,
    current: Positions,
    reference_prices: Mapping[str, float],
    whole_shares: bool = False,
    min_trade_notional: float = 0.0,
) -> List[OrderLeg]:
    """
    Market-order legs that move `current` to `target` for the symbols in `target`.

    Deltas smaller than one share (when `whole_shares`) or below `min_trade_notional`
    are dropped. Returns [] when already at target.
    """
    legs: List[OrderLeg] = []
    for symbol in sorted(target):
        delta = float(target[symbol]) - float(current.get(symbol, 0.0))
        if whole_shares:
            delta = float(round(delta))
        if abs(delta) <= QTY_TOL:
            continue
        price = reference_prices.get(symbol)
        if price is not None and abs(delta) * price < min_trade_notional:
            continue
        legs.append(
            OrderLeg(
                symbol=symbol,
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                quantity=abs(delta),
                reference_price=price,
            )
        )
    return legs


def plan_orders(
    ideas: Sequence[TradeIdea],
    current_positions: Optional[Positions] = None,
    managed_symbols: Optional[Iterable[str]] = None,
    close_out_prices: Optional[Mapping[str, float]] = None,
    whole_shares: bool = False,
    min_trade_notional: float = 0.0,
    at: Optional[str] = None,
) -> OrderPlan:
    """
    Turn trade ideas into proposed order tickets by comparing them with what is held.

    Parameters
    ----------
    ideas : sequence of TradeIdea
        The target positions (usually `ProposalBatch.ideas`). All should share
        the same `as_of`; the latest is used as the plan date.
    current_positions : mapping symbol -> signed shares
        Account holdings. None/empty means nothing is held.
    managed_symbols : iterable of str, optional
        Symbols this system is allowed to close out. A held managed symbol that no
        idea targets gets a close-out ticket. Symbols outside this set are never
        touched. Default None: no close-outs.
    close_out_prices : mapping symbol -> price, optional
        Reference prices for close-outs (needed to size/value them; a close-out
        without a price is reported as a conflict).
    whole_shares, min_trade_notional
        Rounding and dust filter applied to every delta.

    Notes
    -----
    - If two ideas target the same symbol, neither is ticketed; both are reported in
      `conflicts` (netting overlapping pairs is a risk decision, not a guess made here).
    - Tickets are returned in PROPOSED state.
    """
    current = {str(k): float(v) for k, v in (current_positions or {}).items()}
    tickets: List[OrderTicket] = []
    at_target: List[PlanIssue] = []
    conflicts: List[PlanIssue] = []

    as_of_ts = max((i.as_of for i in ideas), default=None)
    as_of = as_of_ts.date().isoformat() if as_of_ts is not None else ""

    # symbol -> ideas that claim it
    claims: Dict[str, List[TradeIdea]] = {}
    for idea in ideas:
        for sym in idea.target_positions:
            claims.setdefault(sym, []).append(idea)
    overlapping = {s for s, v in claims.items() if len(v) > 1}

    target_all: Dict[str, float] = {}
    for idea in ideas:
        clash = sorted(set(idea.target_positions) & overlapping)
        if clash:
            others = sorted({o.pair for s in clash for o in claims[s] if o is not idea})
            conflicts.append(
                PlanIssue(
                    idea.pair,
                    f"symbol overlap on {', '.join(clash)} with {', '.join(others)}",
                    idea.idea_id,
                )
            )
            continue

        target_all.update(idea.target_positions)
        prices = {idea.x: idea.price_x, idea.y: idea.price_y}
        legs = delta_legs(idea.target_positions, current, prices, whole_shares, min_trade_notional)
        if not legs:
            at_target.append(PlanIssue(idea.pair, "holdings already match target", idea.idea_id))
            continue
        tickets.append(
            OrderTicket.create(
                legs, as_of=as_of, idea_id=idea.idea_id, pair=idea.pair, at=at,
                notes=[f"{idea.side} z={idea.zscore:.2f}"],
            )
        )

    # Close-outs: managed, held, and not targeted by any idea (conflicting ideas still claim their symbols).
    claimed = set(claims)
    close_targets = {
        s: 0.0
        for s in sorted({str(m) for m in (managed_symbols or [])})
        if s not in claimed and abs(current.get(s, 0.0)) > QTY_TOL
    }
    if close_targets:
        prices = dict(close_out_prices or {})
        missing = [s for s in close_targets if s not in prices]
        for s in missing:
            conflicts.append(PlanIssue("close_out", f"no reference price for {s}; not closed"))
        closable = {s: 0.0 for s in close_targets if s in prices}
        legs = delta_legs(closable, current, prices, whole_shares, min_trade_notional)
        if legs:
            tickets.append(
                OrderTicket.create(
                    legs, as_of=as_of, purpose="close_out", at=at,
                    notes=["held positions no longer targeted by any idea"],
                )
            )

    return OrderPlan(
        tickets=tickets,
        already_at_target=at_target,
        conflicts=conflicts,
        target_positions={**target_all},
        current_positions=current,
        as_of=as_of,
    )


def apply_fills_to_positions(positions: Positions, ticket: OrderTicket) -> Dict[str, float]:
    """Positions after the ticket's recorded fills (BUY adds, SELL subtracts)."""
    out = {k: float(v) for k, v in positions.items()}
    sides = {l.symbol: l.side for l in ticket.legs}
    for f in ticket.fills:
        sign = 1.0 if sides[f.symbol] == OrderSide.BUY else -1.0
        out[f.symbol] = out.get(f.symbol, 0.0) + sign * f.quantity
    return {k: v for k, v in out.items() if abs(v) > QTY_TOL}

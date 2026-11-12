"""
core.broker_base

The interface every broker adapter implements (the simulated PaperBroker now, a real
broker's paper API and then live trading later). Code upstream of the broker -- the
order planner, risk gates, daily runner -- depends only on this interface, so swapping
paper for a real adapter does not change it.

Contract
- Tickets are submitted in APPROVED state; the broker moves them through
  SUBMITTED -> PARTIALLY_FILLED -> FILLED / CANCELLED / REJECTED and returns the updated,
  immutable `OrderTicket`.
- `submit_ticket` is idempotent on `ticket_id`: submitting the same ticket twice never
  creates a second order (the stored ticket is returned instead).
- `get_positions()` is the broker's own view of holdings (signed shares) and is the input
  to `orders.plan_orders` and to reconciliation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

try:
    import core.orders as orders
except ModuleNotFoundError:  # run from inside core/
    import orders  # type: ignore


class BrokerError(RuntimeError):
    """The broker refused or could not perform an operation."""


@dataclass(frozen=True)
class AccountInfo:
    cash: float
    equity: Optional[float]       # None if some held symbol could not be priced
    gross_exposure: Optional[float]
    net_exposure: Optional[float]
    positions: Dict[str, float]
    environment: str


class Broker(ABC):
    #: "paper" or "live"; recorded in audit trails so the two are never confused.
    environment: str = "paper"

    @abstractmethod
    def submit_ticket(self, ticket: orders.OrderTicket) -> orders.OrderTicket:
        """Accept an APPROVED ticket for execution. Idempotent on ticket_id."""

    @abstractmethod
    def cancel_ticket(self, ticket_id: str, reason: str = "") -> orders.OrderTicket:
        """Cancel the unfilled remainder of a working ticket."""

    @abstractmethod
    def get_ticket(self, ticket_id: str) -> Optional[orders.OrderTicket]:
        """Latest state of a submitted ticket, or None if unknown."""

    @abstractmethod
    def list_tickets(self, status: Optional[orders.OrderStatus] = None) -> List[orders.OrderTicket]:
        """All known tickets, optionally filtered by status, oldest first."""

    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Signed share holdings by symbol."""

    @abstractmethod
    def get_account(self, prices: Optional[Mapping[str, float]] = None) -> AccountInfo:
        """Cash, positions and (if priceable) equity and exposures."""

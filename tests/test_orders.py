# tests/test_orders.py
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import orders
from core.orders import (
    Fill, InvalidTransition, OrderLeg, OrderSide, OrderStatus, OrderTicket, OrderType,
)
from core.trade_proposals import TradeIdea

AS_OF = pd.Timestamp("2026-11-10")


def _idea(x="X", y="Y", position=-1, qty_x=20.0, qty_y=-10.0, beta=2.0, px=50.0, py=100.0):
    return TradeIdea(
        x=x, y=y, as_of=AS_OF,
        side="LONG_SPREAD" if position > 0 else "SHORT_SPREAD", position=position,
        zscore=2.5, spread=1.0, beta=beta, alpha=0.0, price_x=px, price_y=py,
        qty_x=qty_x, qty_y=qty_y, notional_target=2000.0, bars_in_position=1,
    )


# ------------------------------- legs -------------------------------

def test_leg_validation():
    with pytest.raises(ValueError):
        OrderLeg("X", OrderSide.BUY, 0)
    with pytest.raises(ValueError):
        OrderLeg("X", OrderSide.BUY, float("nan"))
    with pytest.raises(ValueError):
        OrderLeg("X", OrderSide.BUY, 1, order_type=OrderType.LIMIT)          # no price
    with pytest.raises(ValueError):
        OrderLeg("X", OrderSide.BUY, 1, limit_price=10.0)                    # price on market
    with pytest.raises(ValueError):
        OrderLeg("", OrderSide.BUY, 1)


def test_leg_signed_quantity_and_notional():
    buy = OrderLeg("X", OrderSide.BUY, 3, reference_price=10.0)
    sell = OrderLeg("Y", OrderSide.SELL, 2, order_type=OrderType.LIMIT, limit_price=5.0)
    assert buy.signed_quantity == 3 and sell.signed_quantity == -2
    assert buy.notional == 30.0 and sell.notional == 10.0   # limit price preferred
    assert OrderLeg("Z", OrderSide.BUY, 1).notional is None


# ------------------------------- delta orders -------------------------------

def test_delta_legs_from_flat_and_from_partial_holdings():
    target = {"X": 20.0, "Y": -10.0}
    from_flat = orders.delta_legs(target, {}, {"X": 50, "Y": 100})
    assert {(l.symbol, l.side, l.quantity) for l in from_flat} == {
        ("X", OrderSide.BUY, 20.0), ("Y", OrderSide.SELL, 10.0)}

    partial = orders.delta_legs(target, {"X": 20.0, "Y": -4.0}, {})
    assert [(l.symbol, l.side, l.quantity) for l in partial] == [("Y", OrderSide.SELL, 6.0)]

    flip = orders.delta_legs({"Y": -10.0}, {"Y": 10.0}, {})
    assert flip[0].side == OrderSide.SELL and flip[0].quantity == 20.0


def test_delta_legs_whole_shares_and_dust():
    assert orders.delta_legs({"X": 10.4}, {}, {}, whole_shares=True)[0].quantity == 10.0
    assert orders.delta_legs({"X": 0.4}, {}, {}, whole_shares=True) == []
    assert orders.delta_legs({"X": 1.0}, {}, {"X": 5.0}, min_trade_notional=10) == []
    assert orders.delta_legs({"X": 5.0}, {"X": 5.0}, {}) == []


# ------------------------------- planning -------------------------------

def test_plan_creates_one_ticket_per_idea_in_proposed_state():
    plan = orders.plan_orders([_idea()], current_positions={})
    assert len(plan.tickets) == 1 and not plan.conflicts and not plan.already_at_target
    t = plan.tickets[0]
    assert t.status == OrderStatus.PROPOSED and t.pair == "X/Y" and t.purpose == "idea"
    assert t.as_of == "2026-11-10"
    assert {l.symbol: l.signed_quantity for l in t.legs} == {"X": 20.0, "Y": -10.0}
    assert t.gross_notional == pytest.approx(20 * 50 + 10 * 100)
    assert t.net_notional == pytest.approx(20 * 50 - 10 * 100)


def test_plan_is_idempotent_when_already_at_target():
    plan = orders.plan_orders([_idea()], current_positions={"X": 20.0, "Y": -10.0})
    assert plan.tickets == []
    assert plan.already_at_target[0].subject == "X/Y"


def test_plan_after_applying_fills_produces_no_further_orders():
    plan = orders.plan_orders([_idea()], {})
    t = plan.tickets[0].approve().submit()
    for leg in t.legs:
        t = t.record_fill(Fill(leg.symbol, leg.quantity, 1.0))
    held = orders.apply_fills_to_positions({}, t)
    assert held == {"X": 20.0, "Y": -10.0}
    assert orders.plan_orders([_idea()], held).tickets == []


def test_plan_ids_are_deterministic_and_data_dependent():
    a = orders.plan_orders([_idea()], {}, at="t0")
    b = orders.plan_orders([_idea()], {}, at="t1")
    assert a.tickets[0].ticket_id == b.tickets[0].ticket_id      # time doesn't matter
    assert a.plan_id == b.plan_id
    c = orders.plan_orders([_idea()], {"X": 5.0})
    assert c.tickets[0].ticket_id != a.tickets[0].ticket_id      # different delta
    assert a.tickets[0].client_order_id("X") == a.tickets[0].ticket_id + "-X"


def test_symbol_overlap_is_a_conflict_not_a_guess():
    one = _idea(x="X", y="Y")
    two = _idea(x="X", y="Z")                  # shares X
    three = _idea(x="P", y="Q")
    plan = orders.plan_orders([one, two, three], {})
    assert [t.pair for t in plan.tickets] == ["P/Q"]
    assert {c.subject for c in plan.conflicts} == {"X/Y", "X/Z"}
    assert all("overlap on X" in c.reason for c in plan.conflicts)


def test_close_outs_only_for_managed_held_untargeted_symbols():
    current = {"X": 20.0, "Y": -10.0, "OLD1": 7.0, "OLD2": -3.0, "FOREIGN": 100.0}
    plan = orders.plan_orders(
        [_idea()], current, managed_symbols=["OLD1", "OLD2", "X", "Y", "NOTHELD"],
        close_out_prices={"OLD1": 10.0, "OLD2": 20.0},
    )
    close = [t for t in plan.tickets if t.purpose == "close_out"]
    assert len(close) == 1 and close[0].idea_id is None
    assert {l.symbol: l.signed_quantity for l in close[0].legs} == {"OLD1": -7.0, "OLD2": 3.0}
    assert "FOREIGN" not in {l.symbol for t in plan.tickets for l in t.legs}


def test_close_out_without_price_is_a_conflict():
    plan = orders.plan_orders([], {"OLD1": 7.0}, managed_symbols=["OLD1"])
    assert plan.tickets == []
    assert plan.conflicts[0].subject == "close_out" and "OLD1" in plan.conflicts[0].reason


def test_no_close_out_by_default():
    assert orders.plan_orders([], {"OLD1": 7.0}).tickets == []


# ------------------------------- lifecycle -------------------------------

def _ticket():
    return orders.plan_orders([_idea()], {}).tickets[0]


def test_happy_path_lifecycle_and_history():
    t = _ticket().approve("ok").submit()
    assert t.status == OrderStatus.SUBMITTED
    t = t.record_fill(Fill("X", 20.0, 50.0))
    assert t.status == OrderStatus.PARTIALLY_FILLED
    t = t.record_fill(Fill("Y", 4.0, 100.0)).record_fill(Fill("Y", 6.0, 101.0))
    assert t.status == OrderStatus.FILLED and t.is_terminal
    assert [h.status for h in t.history][0] == OrderStatus.PROPOSED
    assert t.history[-1].status == OrderStatus.FILLED
    assert t.filled_quantity("Y") == 10.0 and t.remaining_quantity("Y") == 0.0


def test_invalid_transitions_raise_and_original_is_unchanged():
    t = _ticket()
    with pytest.raises(InvalidTransition):
        t.submit()                                  # must be approved first
    with pytest.raises(InvalidTransition):
        t.record_fill(Fill("X", 1.0, 1.0))
    rejected = t.reject("risk")
    assert t.status == OrderStatus.PROPOSED and rejected.status == OrderStatus.REJECTED
    with pytest.raises(InvalidTransition):
        rejected.approve()
    with pytest.raises(InvalidTransition):
        t.approve().submit().cancel().approve()


def test_overfill_and_unknown_symbol_rejected():
    t = _ticket().approve().submit()
    with pytest.raises(ValueError):
        t.record_fill(Fill("X", 21.0, 50.0))
    with pytest.raises(ValueError):
        t.record_fill(Fill("ZZZ", 1.0, 50.0))


def test_ticket_validation():
    leg = OrderLeg("X", OrderSide.BUY, 1)
    with pytest.raises(ValueError):
        OrderTicket.create([], "2026-11-10")
    with pytest.raises(ValueError):
        OrderTicket.create([leg, leg], "2026-11-10")


# ------------------------------- serialization -------------------------------

def test_json_round_trip_is_exact_and_key_order_stable():
    t = _ticket().approve("ok").submit().record_fill(Fill("X", 5.0, 50.0, at="2026-11-11T00:00:00+00:00"))
    s = t.to_json()
    assert OrderTicket.from_json(s) == t
    assert OrderTicket.from_json(s).to_json() == s
    assert list(__import__("json").loads(s).keys()) == sorted(__import__("json").loads(s).keys())


def test_unknown_schema_version_rejected():
    d = _ticket().to_dict()
    d["schema_version"] = 99
    with pytest.raises(ValueError):
        OrderTicket.from_dict(d)

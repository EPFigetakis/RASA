# tests/test_broker_paper.py
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import orders, portfolio
from core.broker_base import BrokerError
from core.broker_paper import PaperBroker, PaperBrokerConfig
from core.orders import OrderLeg, OrderSide, OrderStatus, OrderTicket, OrderType, TimeInForce

NO_COST = dict(commission_per_share=0, half_spread_bps=0, slippage_bps=0, borrow_bps_annual=0)
PX = {"X": 50.0, "Y": 100.0}


def _cfg(**kw):
    return PaperBrokerConfig(**{**NO_COST, **kw})


def _broker(tmp_path, **kw):
    return PaperBroker(tmp_path / "paper.db", _cfg(**kw))


def _ticket(approved=True, legs=None, as_of="2026-11-10", idea="I1"):
    legs = legs or [OrderLeg("X", OrderSide.BUY, 20.0, reference_price=50.0),
                    OrderLeg("Y", OrderSide.SELL, 10.0, reference_price=100.0)]
    t = OrderTicket.create(legs, as_of=as_of, idea_id=idea, pair="X/Y")
    return t.approve("test") if approved else t


# ------------------------------- submission -------------------------------

def test_submit_requires_approved_and_does_not_fill_immediately(tmp_path):
    b = _broker(tmp_path)
    with pytest.raises(BrokerError, match="APPROVED"):
        b.submit_ticket(_ticket(approved=False))
    t = b.submit_ticket(_ticket())
    assert t.status == OrderStatus.SUBMITTED and b.get_positions() == {}
    assert b.get_ticket(t.ticket_id) == t and len(b.working_tickets()) == 1


def test_submit_is_idempotent_on_ticket_id(tmp_path):
    b = _broker(tmp_path)
    t1 = b.submit_ticket(_ticket())
    b.process_bar(PX, "2026-11-10")
    t2 = b.submit_ticket(_ticket())                       # same ticket again
    assert t2.status == OrderStatus.FILLED and t2.ticket_id == t1.ticket_id
    assert len(b.list_tickets()) == 1
    b.process_bar(PX, "2026-11-11")
    assert b.get_positions() == {"X": 20.0, "Y": -10.0}   # not doubled
    assert "duplicate_submit_ignored" in set(b.events()["kind"])


# ------------------------------- fills -------------------------------

def test_market_ticket_fills_and_books_positions_and_cash(tmp_path):
    b = _broker(tmp_path)
    b.submit_ticket(_ticket())
    changed = b.process_bar(PX, "2026-11-10")
    assert [t.status for t in changed] == [OrderStatus.FILLED]
    assert b.get_positions() == {"X": 20.0, "Y": -10.0}
    acct = b.get_account()
    assert acct.cash == pytest.approx(100_000 - 20 * 50 + 10 * 100)
    assert acct.equity == pytest.approx(100_000) and acct.gross_exposure == pytest.approx(2000)
    assert acct.environment == "paper"
    assert b.last_bar() == "2026-11-10" and b.equity_history().loc["2026-11-10", "equity"] == pytest.approx(100_000)


def test_fees_are_charged_on_fills(tmp_path):
    b = PaperBroker(tmp_path / "p.db", PaperBrokerConfig(
        commission_per_share=0.01, half_spread_bps=0, slippage_bps=0, borrow_bps_annual=0))
    b.submit_ticket(_ticket())
    b.process_bar(PX, "2026-11-10")
    assert b.portfolio.fees_paid == pytest.approx(0.30)
    assert b.get_account().equity == pytest.approx(100_000 - 0.30)


def test_limit_orders_wait_until_marketable(tmp_path):
    legs = [OrderLeg("X", OrderSide.BUY, 10.0, OrderType.LIMIT, limit_price=48.0),
            OrderLeg("Y", OrderSide.SELL, 5.0, OrderType.LIMIT, limit_price=101.0)]
    b = _broker(tmp_path, expire_day_orders=False)
    t = b.submit_ticket(_ticket(legs=legs))
    b.process_bar({"X": 50.0, "Y": 100.0}, "2026-11-10")          # neither marketable
    assert b.get_positions() == {} and b.get_ticket(t.ticket_id).status == OrderStatus.SUBMITTED
    b.process_bar({"X": 47.0, "Y": 100.0}, "2026-11-11")          # BUY X marketable only
    assert b.get_positions() == {"X": 10.0}
    assert b.get_ticket(t.ticket_id).status == OrderStatus.PARTIALLY_FILLED
    b.process_bar({"X": 47.0, "Y": 102.0}, "2026-11-12")          # SELL Y marketable
    assert b.get_ticket(t.ticket_id).status == OrderStatus.FILLED
    assert b.get_positions() == {"X": 10.0, "Y": -5.0}


def test_partial_fills_across_bars(tmp_path):
    b = _broker(tmp_path, fill_fraction=0.5, expire_day_orders=False)
    t = b.submit_ticket(_ticket())
    b.process_bar(PX, "2026-11-10")
    b.process_bar(PX, "2026-11-10")                                # same bar again: no-op
    cur = b.get_ticket(t.ticket_id)
    assert cur.status == OrderStatus.PARTIALLY_FILLED and cur.filled_quantity("X") == 10.0
    b.process_bar(PX, "2026-11-11")
    assert b.get_ticket(t.ticket_id).status == OrderStatus.FILLED
    assert b.get_positions() == {"X": 20.0, "Y": -10.0}


def test_unfilled_day_orders_expire_leaving_visible_one_legged_state(tmp_path):
    legs = [OrderLeg("X", OrderSide.BUY, 10.0, reference_price=50.0),
            OrderLeg("Y", OrderSide.SELL, 5.0, OrderType.LIMIT, limit_price=150.0)]
    b = _broker(tmp_path)
    t = b.submit_ticket(_ticket(legs=legs))
    b.process_bar(PX, "2026-11-10")                                # X fills, Y limit never reached
    final = b.get_ticket(t.ticket_id)
    assert final.status == OrderStatus.CANCELLED and "expired" in final.history[-1].reason
    assert b.get_positions() == {"X": 10.0}                        # one-legged; recorded
    assert "expired" in set(b.events(t.ticket_id)["kind"])


def test_gtc_orders_survive_the_bar(tmp_path):
    legs = [OrderLeg("X", OrderSide.BUY, 10.0, OrderType.LIMIT, limit_price=40.0,
                     time_in_force=TimeInForce.GTC)]
    b = _broker(tmp_path)
    t = b.submit_ticket(_ticket(legs=legs))
    b.process_bar(PX, "2026-11-10")
    assert b.get_ticket(t.ticket_id).status == OrderStatus.SUBMITTED


def test_missing_price_leaves_leg_unfilled(tmp_path):
    b = _broker(tmp_path, expire_day_orders=False)
    t = b.submit_ticket(_ticket())
    b.process_bar({"X": 50.0}, "2026-11-10")
    cur = b.get_ticket(t.ticket_id)
    assert cur.status == OrderStatus.PARTIALLY_FILLED and b.get_positions() == {"X": 20.0}


# ------------------------------- cancel / borrow / bars -------------------------------

def test_cancel(tmp_path):
    b = _broker(tmp_path)
    t = b.submit_ticket(_ticket())
    assert b.cancel_ticket(t.ticket_id, "changed mind").status == OrderStatus.CANCELLED
    b.process_bar(PX, "2026-11-10")
    assert b.get_positions() == {}
    with pytest.raises(BrokerError):
        b.cancel_ticket(t.ticket_id)
    with pytest.raises(BrokerError, match="unknown"):
        b.cancel_ticket("nope")


def test_borrow_accrues_once_per_new_bar_over_business_days(tmp_path):
    b = _broker(tmp_path, borrow_bps_annual=252.0)                 # 1bp/day on short value
    b.submit_ticket(_ticket())
    b.process_bar(PX, "2026-11-10")                                # Tue: opens, no overnight yet
    assert b.portfolio.borrow_paid == 0
    b.process_bar(PX, "2026-11-11")                                # +1 bday on Y short = 1000
    assert b.portfolio.borrow_paid == pytest.approx(1000 * 1e-4)
    b.process_bar(PX, "2026-11-11")                                # same bar: nothing
    assert b.portfolio.borrow_paid == pytest.approx(1000 * 1e-4)
    b.process_bar(PX, "2026-11-16")                                # Fri->Mon... 3 bdays
    assert b.portfolio.borrow_paid == pytest.approx(1000 * 1e-4 * 4)


def test_bars_cannot_go_backwards(tmp_path):
    b = _broker(tmp_path)
    b.process_bar(PX, "2026-11-11")
    with pytest.raises(BrokerError, match="before the last"):
        b.process_bar(PX, "2026-11-10")


# ------------------------------- persistence -------------------------------

def test_state_survives_restart(tmp_path):
    b = _broker(tmp_path)
    t = b.submit_ticket(_ticket())
    b.process_bar(PX, "2026-11-10")
    b.close()

    again = PaperBroker(tmp_path / "paper.db")                     # no config: loads stored one
    assert again.get_positions() == {"X": 20.0, "Y": -10.0}
    assert again.get_ticket(t.ticket_id).status == OrderStatus.FILLED
    assert again.last_bar() == "2026-11-10" and again.config == _cfg()
    assert len(again.portfolio.trades) == 2 and not again.events().empty
    assert again.submit_ticket(_ticket()).status == OrderStatus.FILLED   # still idempotent
    assert again.get_account().equity == pytest.approx(100_000)


def test_reopening_with_a_different_config_is_refused(tmp_path):
    _broker(tmp_path).close()
    with pytest.raises(BrokerError, match="different"):
        PaperBroker(tmp_path / "paper.db", _cfg(initial_cash=5.0))


def test_failed_operation_rolls_back_memory_and_disk(tmp_path, monkeypatch):
    b = _broker(tmp_path)
    b.submit_ticket(_ticket())
    real = b.portfolio.apply_fill
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(b.portfolio, "apply_fill", flaky)
    with pytest.raises(RuntimeError):
        b.process_bar(PX, "2026-11-10")
    monkeypatch.undo()
    assert b.get_positions() == {} and b.last_bar() is None
    assert b.get_ticket(_ticket().ticket_id).status == OrderStatus.SUBMITTED
    # a retry then works from clean state
    b.process_bar(PX, "2026-11-10")
    assert b.get_positions() == {"X": 20.0, "Y": -10.0}


# ------------------------------- integration with planner / reconciliation -------------------------------

def test_plan_submit_fill_replan_gives_no_further_orders(tmp_path):
    import pandas as pd
    from core.trade_proposals import TradeIdea

    idea = TradeIdea(x="X", y="Y", as_of=pd.Timestamp("2026-11-10"), side="SHORT_SPREAD", position=-1,
                     zscore=2.5, spread=1.0, beta=2.0, alpha=0.0, price_x=50.0, price_y=100.0,
                     qty_x=20.0, qty_y=-10.0, notional_target=2000.0, bars_in_position=1)
    b = _broker(tmp_path)
    plan = orders.plan_orders([idea], b.get_positions())
    b.submit_ticket(plan.tickets[0].approve())
    b.process_bar(PX, "2026-11-10")
    assert orders.plan_orders([idea], b.get_positions()).tickets == []


def test_reconcile_reports_and_logs_breaks(tmp_path):
    b = _broker(tmp_path)
    b.submit_ticket(_ticket())
    b.process_bar(PX, "2026-11-10")
    assert b.reconcile({"X": 20.0, "Y": -10.0}) == []
    breaks = b.reconcile({"X": 20.0, "Y": -8.0})
    assert [(x.symbol, x.difference) for x in breaks] == [("Y", -2.0)]
    detail = b.events().query("kind == 'reconciliation'")["detail"].tolist()
    assert detail[0] == "ok" and "Y" in detail[1]


def test_config_validation():
    with pytest.raises(ValueError):
        PaperBrokerConfig(fill_fraction=0)
    with pytest.raises(ValueError):
        PaperBrokerConfig(initial_cash=-1)

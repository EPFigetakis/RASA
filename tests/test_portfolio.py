# tests/test_portfolio.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import execution, orders, portfolio, strategy
from core.portfolio import Portfolio


def _identity(pf, prices):
    v = pf.value(prices)
    assert v.equity - pf.initial_cash == pytest.approx(v.total_pnl)
    return v


# ------------------------------- ledger -------------------------------

def test_buy_then_sell_realizes_pnl_and_cash():
    pf = Portfolio(10_000)
    pf.apply_fill("A", +10, 100.0)
    assert pf.cash == 9_000 and pf.quantity("A") == 10
    pf.apply_fill("A", -4, 110.0)
    assert pf.realized_pnl == pytest.approx(40) and pf.positions["A"].avg_price == 100.0
    pf.apply_fill("A", -6, 90.0)
    assert pf.realized_pnl == pytest.approx(40 - 60) and "A" not in pf.positions
    assert pf.cash == pytest.approx(10_000 - 20)
    _identity(pf, {})


def test_average_cost_when_adding():
    pf = Portfolio(10_000)
    pf.apply_fill("A", 10, 100.0)
    pf.apply_fill("A", 10, 120.0)
    assert pf.positions["A"].avg_price == pytest.approx(110.0)


def test_short_sale_and_cover():
    pf = Portfolio(10_000)
    pf.apply_fill("S", -10, 50.0)                    # proceeds credited
    assert pf.cash == 10_500 and pf.quantity("S") == -10
    v = _identity(pf, {"S": 45.0})
    assert v.unrealized_pnl == pytest.approx(50) and v.short_value == pytest.approx(-450)
    assert v.gross == pytest.approx(450) and v.net == pytest.approx(-450)
    pf.apply_fill("S", +10, 45.0)
    assert pf.realized_pnl == pytest.approx(50) and pf.positions == {}
    assert pf.cash == pytest.approx(10_050)


def test_flip_through_zero_closes_old_side_and_opens_remainder():
    pf = Portfolio(10_000)
    pf.apply_fill("A", +10, 100.0)
    pf.apply_fill("A", -15, 110.0)                   # close 10 (+100), open short 5 @110
    assert pf.realized_pnl == pytest.approx(100)
    assert pf.quantity("A") == -5 and pf.positions["A"].avg_price == 110.0
    _identity(pf, {"A": 105.0})


def test_fees_and_borrow_are_tracked_and_reconcile():
    pf = Portfolio(10_000)
    pf.apply_fill("L", +10, 100.0, fee=1.5)
    pf.apply_fill("S", -20, 50.0, fee=2.0)
    charge = pf.accrue_borrow({"S": 50.0, "L": 100.0}, annual_bps=252.0, days=1)
    assert charge == pytest.approx(1000 * 1e-4)      # only the short leg
    v = _identity(pf, {"L": 104.0, "S": 49.0})
    assert v.fees_paid == 3.5 and v.borrow_paid == pytest.approx(0.1)
    assert v.total_pnl == pytest.approx(10 * 4 + 20 * 1 - 3.5 - 0.1)


def test_invalid_fills_and_missing_prices():
    pf = Portfolio(1000)
    for args in [("A", 0, 10.0), ("A", 1, 0.0), ("A", float("nan"), 10.0), ("A", 1, -5.0)]:
        with pytest.raises(ValueError):
            pf.apply_fill(*args)
    with pytest.raises(ValueError):
        pf.apply_fill("A", 1, 10.0, fee=-1)
    pf.apply_fill("A", 1, 10.0)
    with pytest.raises(KeyError):
        pf.value({})


def test_symbol_pnl_table():
    pf = Portfolio(10_000)
    pf.apply_fill("A", 10, 100.0)
    pf.apply_fill("A", -10, 105.0)
    pf.apply_fill("B", -5, 20.0)
    t = pf.symbol_pnl({"B": 18.0}).set_index("symbol")
    assert t.loc["A", "realized_pnl"] == 50 and t.loc["B", "unrealized_pnl"] == pytest.approx(10)


def test_json_round_trip():
    pf = Portfolio(10_000)
    pf.apply_fill("A", 10, 100.0, fee=1.0, at="2026-01-02", ticket_id="T1")
    pf.apply_fill("B", -5, 20.0)
    s = pf.to_json()
    again = Portfolio.from_json(s)
    assert again.to_json() == s and again.quantities() == pf.quantities()
    assert again.trades[0].ticket_id == "T1"


# ------------------------------- orders integration -------------------------------

def test_ticket_fills_book_into_ledger_and_feed_back_into_planning():
    from core.trade_proposals import TradeIdea
    idea = TradeIdea(x="X", y="Y", as_of=pd.Timestamp("2026-11-10"), side="SHORT_SPREAD",
                     position=-1, zscore=2.5, spread=1.0, beta=2.0, alpha=0.0, price_x=50.0,
                     price_y=100.0, qty_x=20.0, qty_y=-10.0, notional_target=2000.0, bars_in_position=1)
    pf = Portfolio(100_000)
    plan = orders.plan_orders([idea], pf.quantities())
    t = plan.tickets[0].approve().submit()
    t = t.record_fill(orders.Fill("X", 20.0, 50.0)).record_fill(orders.Fill("Y", 10.0, 100.0))
    cfg = execution.ExecutionCostConfig(commission_per_share=0.01, half_spread_bps=0, slippage_bps=0)
    booked = pf.apply_ticket_fills(t, cfg)
    assert pf.quantities() == {"X": 20.0, "Y": -10.0}
    assert pf.fees_paid == pytest.approx(0.30) and all(r.ticket_id == t.ticket_id for r in booked)
    assert pf.apply_ticket_fills(t, cfg, already_booked=2) == []        # idempotent catch-up
    assert orders.plan_orders([idea], pf.quantities()).tickets == []    # nothing left to do


def test_reconcile():
    assert portfolio.reconcile({"A": 10, "B": -5}, {"A": 10, "B": -5}) == []
    br = portfolio.reconcile({"A": 10, "B": -5}, {"A": 8, "C": 3})
    assert [(b.symbol, b.difference) for b in br] == [("A", -2), ("B", 5), ("C", 3)]


# ------------------------------- simulation -------------------------------

def _prices(n=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    x = 50 + np.cumsum(rng.normal(size=n) * 0.5)
    y = 2 * x + rng.normal(size=n)
    return pd.DataFrame({"X": x, "Y": y}, index=idx)


def test_simulation_matches_scalar_backtest_when_costs_are_zero():
    """In raw-price space with fixed shares, leg-level PnL == scalar spread-unit PnL."""
    px = _prices()
    beta, npu = 2.0, 10.0
    signals = pd.Series(0.0, index=px.index)
    signals.iloc[5:15] = 1.0
    signals.iloc[20:35] = -1.0
    signals.iloc[40:45] = 1.0

    spread = px["Y"] - beta * px["X"]
    bt = strategy.backtest_spread_strategy(
        spread, signals, strategy.BacktestConfig(initial_capital=100_000, notional_per_unit=npu))

    targets = pd.DataFrame({"Y": signals * npu, "X": -beta * signals * npu})
    sim = portfolio.simulate_targets(px, targets, initial_cash=100_000)

    np.testing.assert_allclose(sim.equity.to_numpy(), bt.equity_curve.to_numpy(), rtol=0, atol=1e-6)
    assert sim.summary["fees_paid"] == 0 and sim.summary["n_trades"] > 0


def test_simulation_costs_match_vectorized_execution_model():
    px = _prices()
    beta, npu = 2.0, 10.0
    signals = pd.Series(0.0, index=px.index)
    signals.iloc[5:15] = 1.0
    signals.iloc[20:35] = -1.0
    cfg = execution.ExecutionCostConfig(commission_per_share=0.01, half_spread_bps=3, slippage_bps=2,
                                        borrow_bps_annual=100)

    spread = px["Y"] - beta * px["X"]
    bt = strategy.backtest_spread_strategy(
        spread, signals, strategy.BacktestConfig(notional_per_unit=npu, execution=cfg),
        leg_prices=px.rename(columns=str.upper)[["X", "Y"]], beta=beta)

    targets = pd.DataFrame({"Y": signals * npu, "X": -beta * signals * npu})
    sim = portfolio.simulate_targets(px, targets, initial_cash=100_000, cost_cfg=cfg)

    assert sim.summary["fees_paid"] + sim.summary["borrow_paid"] == pytest.approx(bt.summary["total_costs"])
    np.testing.assert_allclose(sim.equity.to_numpy(), bt.equity_curve.to_numpy(), atol=1e-6)


def test_simulation_identity_and_exposure_series():
    px = _prices()
    signals = pd.Series(0.0, index=px.index)
    signals.iloc[10:30] = -1.0
    cfg = execution.ExecutionCostConfig()
    tg = portfolio.targets_from_signals(px, {("X", "Y"): (signals, 2.0)}, notional_per_pair=10_000)
    sim = portfolio.simulate_targets(px, tg, 100_000, cfg)
    final = sim.portfolio.value(px.iloc[-1].to_dict())
    assert final.equity - 100_000 == pytest.approx(final.total_pnl)
    assert sim.gross.iloc[15] == pytest.approx(10_000, rel=0.05) and sim.gross.iloc[0] == 0
    assert sim.positions.iloc[-1].abs().sum() == 0                      # flat at the end
    assert sim.trades["fee"].sum() == pytest.approx(sim.summary["fees_paid"])


def test_fill_delay_shifts_entry_one_day():
    px = _prices()
    tg = pd.DataFrame({"Y": 0.0, "X": 0.0}, index=px.index)
    tg.iloc[10:, tg.columns.get_loc("Y")] = 5.0
    now = portfolio.simulate_targets(px, tg, 100_000)
    late = portfolio.simulate_targets(px, tg, 100_000, execution.ExecutionCostConfig(
        commission_per_share=0, half_spread_bps=0, slippage_bps=0, borrow_bps_annual=0, fill_delay_days=1))
    assert now.positions["Y"].iloc[10] == 5 and late.positions["Y"].iloc[10] == 0
    assert late.positions["Y"].iloc[11] == 5


def test_targets_from_signals_holds_shares_fixed_between_changes():
    px = _prices()
    signals = pd.Series(0.0, index=px.index)
    signals.iloc[10:20] = 1.0
    tg = portfolio.targets_from_signals(px, {("X", "Y"): (signals, 2.0)}, 10_000)
    held = tg.iloc[10:20]
    assert held["Y"].nunique() == 1 and held["X"].nunique() == 1
    assert held["Y"].iloc[0] > 0 > held["X"].iloc[0]
    assert tg.iloc[:10].abs().sum().sum() == 0 and tg.iloc[20:].abs().sum().sum() == 0


def test_simulation_rejects_targets_without_prices():
    px = _prices()
    with pytest.raises(KeyError):
        portfolio.simulate_targets(px, pd.DataFrame({"ZZZ": 1.0}, index=px.index))

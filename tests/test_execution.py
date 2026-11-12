# tests/test_execution.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import execution, strategy


def _legs(n=6, px_x=50.0, px_y=100.0):
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame({"X": px_x, "Y": px_y}, index=idx)


def _only(**kw):
    """Config with all frictions off except the ones passed."""
    return execution.ExecutionCostConfig(
        **{**dict(
            commission_per_share=0.0, commission_bps=0.0, half_spread_bps=0.0,
            slippage_bps=0.0, borrow_bps_annual=0.0, fill_delay_days=0,
        ), **kw}
    )


def test_zero_config_has_no_costs():
    legs = _legs()
    pos = pd.Series([0, 1, 1, 0, -1, 0], index=legs.index, dtype=float)
    out = execution.compute_execution_costs(
        pos, legs, beta=2.0, notional_per_unit=10, cfg=execution.ExecutionCostConfig.zero()
    )
    assert (out["total"] == 0).all()


def test_commission_per_share_raw_space():
    # Open 1 unit: Y shares = 10, X shares = 10 * beta(2) = 20 -> 30 shares.
    legs = _legs()
    pos = pd.Series([0, 1, 1, 1, 1, 1], index=legs.index, dtype=float)
    out = execution.compute_execution_costs(
        pos, legs, beta=2.0, notional_per_unit=10, cfg=_only(commission_per_share=0.01)
    )
    assert out["commission"].iloc[1] == pytest.approx(0.30)
    assert out["commission"].sum() == pytest.approx(0.30)


def test_spread_and_slippage_bps_on_traded_notional():
    legs = _legs()
    pos = pd.Series([0, 1, 1, 0, 0, 0], index=legs.index, dtype=float)
    # 1 unit, npu=10: Y notional 1000, X notional 10*2*50 = 1000 -> 2000 per trade.
    out = execution.compute_execution_costs(
        pos, legs, beta=2.0, notional_per_unit=10,
        cfg=_only(half_spread_bps=5.0, slippage_bps=2.0),
    )
    # Two trades (open + close), 2000 notional each.
    assert out["spread"].sum() == pytest.approx(2 * 2000 * 5e-4)
    assert out["slippage"].sum() == pytest.approx(2 * 2000 * 2e-4)


def test_flip_costs_twice_a_single_open():
    legs = _legs()
    open_only = pd.Series([0, 1, 1, 1, 1, 1], index=legs.index, dtype=float)
    flip = pd.Series([0, 1, -1, -1, -1, -1], index=legs.index, dtype=float)
    cfg = _only(half_spread_bps=10.0)
    a = execution.compute_execution_costs(open_only, legs, 1.0, 1, cfg)["spread"]
    b = execution.compute_execution_costs(flip, legs, 1.0, 1, cfg)["spread"]
    assert b.iloc[2] == pytest.approx(2 * a.iloc[1])


def test_borrow_charges_only_the_short_leg():
    legs = _legs()
    cfg = _only(borrow_bps_annual=252.0)  # 1 bp per day
    # Long spread (pos=+1): short X only. beta=2, npu=1 -> X notional 100.
    long_pos = pd.Series(1.0, index=legs.index)
    long_cost = execution.compute_execution_costs(long_pos, legs, 2.0, 1, cfg)["borrow"]
    assert long_cost.iloc[1:].eq(100 * 1e-4).all()
    assert long_cost.iloc[0] == 0.0
    # Short spread (pos=-1): short Y only -> Y notional 100.
    short_pos = pd.Series(-1.0, index=legs.index)
    short_cost = execution.compute_execution_costs(short_pos, legs, 2.0, 1, cfg)["borrow"]
    assert short_cost.iloc[1:].eq(100 * 1e-4).all()


def test_log_space_uses_dollar_notional():
    legs = _legs(px_x=50.0, px_y=100.0)
    pos = pd.Series([0, 1, 0, 0, 0, 0], index=legs.index, dtype=float)
    # npu=1000 dollars of Y, beta*1000 = 500 dollars of X -> 1500 per trade.
    out = execution.compute_execution_costs(
        pos, legs, beta=0.5, notional_per_unit=1000,
        cfg=_only(half_spread_bps=10.0), log_space=True,
    )
    assert out["spread"].sum() == pytest.approx(2 * 1500 * 10e-4)


def test_negative_beta_flips_borrowed_leg():
    legs = _legs()
    cfg = _only(borrow_bps_annual=252.0)
    pos = pd.Series(1.0, index=legs.index)
    # beta < 0: long spread = long Y and *long* X -> nothing to borrow.
    out = execution.compute_execution_costs(pos, legs, -1.0, 1, cfg)["borrow"]
    assert (out == 0).all()


def test_fill_delay_shifts_positions_and_starts_flat():
    idx = pd.date_range("2022-01-03", periods=5, freq="B")
    pos = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=idx)
    delayed = execution.apply_fill_delay(pos, 1)
    assert delayed.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert execution.apply_fill_delay(pos, 0) is pos


def test_config_validation():
    with pytest.raises(ValueError):
        execution.ExecutionCostConfig(fill_delay_days=-1)
    with pytest.raises(ValueError):
        execution.ExecutionCostConfig(slippage_bps=-1)


# ---------------------------------------------------------------------------
# Integration with the backtester
# ---------------------------------------------------------------------------

def _bt_inputs():
    idx = pd.date_range("2022-01-03", periods=8, freq="B")
    spread = pd.Series([0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0], index=idx)
    signals = pd.Series([0, 1, 1, 1, 0, 0, 0, 0], index=idx, dtype=float)
    legs = pd.DataFrame({"X": 50.0, "Y": 100.0}, index=idx)
    return spread, signals, legs


def test_backtest_requires_leg_prices_when_execution_set():
    spread, signals, _ = _bt_inputs()
    cfg = strategy.BacktestConfig(execution=execution.ExecutionCostConfig())
    with pytest.raises(ValueError):
        strategy.backtest_spread_strategy(spread, signals, cfg)


def test_backtest_without_execution_is_unchanged():
    spread, signals, _ = _bt_inputs()
    bt = strategy.backtest_spread_strategy(spread, signals, strategy.BacktestConfig())
    assert bt.cost_breakdown is None
    assert bt.summary["total_costs"] == 0.0
    assert bt.summary["total_pnl"] == pytest.approx(bt.summary["gross_total_pnl"])


def test_backtest_zero_execution_matches_baseline():
    spread, signals, legs = _bt_inputs()
    base = strategy.backtest_spread_strategy(spread, signals, strategy.BacktestConfig())
    cfg = strategy.BacktestConfig(execution=execution.ExecutionCostConfig.zero())
    bt = strategy.backtest_spread_strategy(spread, signals, cfg, leg_prices=legs, beta=1.0)
    assert bt.pnl.equals(base.pnl)


def test_backtest_costs_reduce_pnl_and_are_reported():
    spread, signals, legs = _bt_inputs()
    cfg = strategy.BacktestConfig(
        notional_per_unit=10,
        execution=execution.ExecutionCostConfig(),
    )
    bt = strategy.backtest_spread_strategy(spread, signals, cfg, leg_prices=legs, beta=1.0)
    assert bt.summary["total_costs"] > 0
    assert bt.summary["total_pnl"] == pytest.approx(
        bt.summary["gross_total_pnl"] - bt.summary["total_costs"]
    )
    assert bt.cost_breakdown["total"].sum() == pytest.approx(bt.summary["total_costs"])


def test_backtest_fill_delay_pushes_back_pnl():
    spread, signals, legs = _bt_inputs()
    cfg0 = strategy.BacktestConfig(execution=execution.ExecutionCostConfig.zero())
    cfg1 = strategy.BacktestConfig(
        execution=execution.ExecutionCostConfig(
            commission_per_share=0, half_spread_bps=0, slippage_bps=0,
            borrow_bps_annual=0, fill_delay_days=1,
        )
    )
    b0 = strategy.backtest_spread_strategy(spread, signals, cfg0, leg_prices=legs, beta=1.0)
    b1 = strategy.backtest_spread_strategy(spread, signals, cfg1, leg_prices=legs, beta=1.0)
    # Position is entered one bar later, so it earns a different PnL path.
    assert b1.positions.iloc[1] == 0.0 and b0.positions.iloc[1] == 1.0
    assert b1.summary["total_pnl"] != b0.summary["total_pnl"]

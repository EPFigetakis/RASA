"""
core.execution

Leg-level execution-cost model for the pair-spread backtester.

core.strategy models PnL in "spread units" (1 unit = long 1 Y, short beta X).
This module prices the *trading frictions* of holding and changing those
positions, leg by leg:

- commission      : per-share and/or basis-point commission on traded notional
- half-spread     : crossing half the bid/ask spread on every trade
- slippage        : extra adverse price movement on every trade
- short borrow    : annualised fee on the notional of whichever leg is short
- fill delay      : positions are established N days after the signal

All costs are returned in the same units as the strategy's PnL, so they can be
subtracted directly from it.

Unit conventions (must match core.strategy / core.pairs):
- Raw-price spread  : one unit = `notional_per_unit` shares of Y and
                      `notional_per_unit * beta` shares of X.
- Log-price spread  : one unit = `notional_per_unit` dollars of Y and
                      `notional_per_unit * beta` dollars of X
                      (PnL is Δlog-spread × notional_per_unit).

Bid/ask data is not available from the adjusted-close panel, so the spread and
slippage are flat basis-point assumptions, not observed quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ExecutionCostConfig:
    """
    Execution-cost assumptions.

    Attributes
    ----------
    commission_per_share : float
        Dollars per share traded (per leg).
    commission_bps : float
        Commission as basis points of traded notional (per leg).
    half_spread_bps : float
        Half the bid/ask spread, in bps of traded notional (per leg).
    slippage_bps : float
        Additional adverse fill slippage, in bps of traded notional (per leg).
    borrow_bps_annual : float
        Annual short-borrow fee in bps, charged daily on the short leg's notional.
    fill_delay_days : int
        Trading days between a signal and the position actually being held.
        0 = fill at the close the signal is generated (the existing backtest
        convention); 1 = fill one bar later.
    trading_days_per_year : int
        Used to convert the annual borrow fee to a daily one.
    """

    commission_per_share: float = 0.005
    commission_bps: float = 0.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    borrow_bps_annual: float = 50.0
    fill_delay_days: int = 0
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if self.fill_delay_days < 0:
            raise ValueError("fill_delay_days must be >= 0")
        for name in (
            "commission_per_share", "commission_bps", "half_spread_bps",
            "slippage_bps", "borrow_bps_annual",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def zero(cls) -> "ExecutionCostConfig":
        """A config with every friction switched off (useful as a baseline)."""
        return cls(
            commission_per_share=0.0, commission_bps=0.0, half_spread_bps=0.0,
            slippage_bps=0.0, borrow_bps_annual=0.0, fill_delay_days=0,
        )


COST_COLUMNS = ["commission", "spread", "slippage", "borrow", "total"]


def apply_fill_delay(positions: pd.Series, delay_days: int) -> pd.Series:
    """Shift positions later by `delay_days` bars; the account starts flat."""
    if delay_days <= 0:
        return positions
    return positions.shift(delay_days).fillna(0.0)


def compute_execution_costs(
    positions: pd.Series,
    leg_prices: pd.DataFrame,
    beta: float,
    notional_per_unit: float,
    cfg: ExecutionCostConfig,
    log_space: bool = False,
) -> pd.DataFrame:
    """
    Price the execution frictions of a position path.

    Parameters
    ----------
    positions : pd.Series
        Spread-unit positions actually held at each close (after any fill delay).
    leg_prices : pd.DataFrame
        Columns "X" and "Y" (as returned by core.pairs.extract_pair_prices),
        raw (not log) prices; reindexed onto `positions`.
    beta : float
        Hedge ratio (frozen) used to build the spread.
    notional_per_unit : float
        Strategy scaling factor (see core.strategy.BacktestConfig).
    cfg : ExecutionCostConfig
    log_space : bool
        True if the spread was built from log prices.

    Returns
    -------
    pd.DataFrame
        Same index as `positions`; columns commission, spread, slippage,
        borrow, total (all >= 0, in PnL units).
    """
    idx = positions.index
    px_x = leg_prices["X"].reindex(idx).astype(float)
    px_y = leg_prices["Y"].reindex(idx).astype(float)
    if px_x.isna().any() or px_y.isna().any():
        raise ValueError("leg_prices must cover every date in positions (no NaNs).")

    pos = positions.astype(float)
    npu = float(notional_per_unit)
    abs_beta = abs(float(beta))

    # Dollar value of one spread unit on each leg.
    if log_space:
        unit_dollars_y = pd.Series(1.0, index=idx)
        unit_dollars_x = pd.Series(1.0, index=idx) * abs_beta
    else:
        unit_dollars_y = px_y
        unit_dollars_x = px_x * abs_beta

    # ---- trading costs: driven by |Δposition| ----
    traded_units = pos.diff().abs()
    traded_units.iloc[0] = abs(pos.iloc[0]) if len(pos) else 0.0  # opening trade
    traded_dollars_y = traded_units * npu * unit_dollars_y
    traded_dollars_x = traded_units * npu * unit_dollars_x
    traded_dollars = traded_dollars_y + traded_dollars_x
    traded_shares = traded_dollars_y / px_y + traded_dollars_x / px_x

    commission = (
        cfg.commission_per_share * traded_shares
        + cfg.commission_bps / 1e4 * traded_dollars
    )
    spread_cost = cfg.half_spread_bps / 1e4 * traded_dollars
    slippage = cfg.slippage_bps / 1e4 * traded_dollars

    # ---- borrow: charged on the short leg of the position held over the bar ----
    # pos > 0: long Y, short X.  pos < 0: short Y, long X.
    # Position held from t-1 to t is charged on t-1's notional.
    pos_lag = pos.shift(1).fillna(0.0)
    short_y = (-pos_lag).clip(lower=0.0) * npu * unit_dollars_y.shift(1)
    # X's signed holding is -pos*beta, so with a negative beta the short side flips.
    beta_sign = 1.0 if float(beta) >= 0 else -1.0
    short_x = (pos_lag * beta_sign).clip(lower=0.0) * npu * unit_dollars_x.shift(1)
    short_dollars = (short_y + short_x).fillna(0.0)
    borrow = cfg.borrow_bps_annual / 1e4 / cfg.trading_days_per_year * short_dollars

    out = pd.DataFrame(
        {
            "commission": commission,
            "spread": spread_cost,
            "slippage": slippage,
            "borrow": borrow,
        },
        index=idx,
    ).fillna(0.0)
    out["total"] = out[["commission", "spread", "slippage", "borrow"]].sum(axis=1)
    return out

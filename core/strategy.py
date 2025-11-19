"""
core.strategy

Backtesting utilities for a single pair-spread strategy.

This module assumes you already have:
- A wide price panel from core.panel / core.prices.
- A PairPipelineResult from core.pairs (spread + z-score + position signals).

It then:
- Applies (optionally regime-filtered) position signals to the spread.
- Computes PnL via:
      pnl_t = position_{t-1} * (spread_t - spread_{t-1}) * notional_per_unit
- Applies simple transaction costs per unit of position change.
- Builds an equity curve given an initial capital.
- Computes basic performance stats (total return, Sharpe, max drawdown, etc.).

Typical notebook usage:

    from core import panel, pairs, strategy

    # 1) Load prices & returns
    px, rets = panel.load_prices_and_returns_from_config()

    # 2) Build pair signals
    spec = pairs.PairSpec(x="SPY", y="QQQ")
    pair_res = pairs.build_pair_signals_from_panel(px, spec)

    # 3) Backtest strategy
    cfg = strategy.BacktestConfig(initial_capital=100_000, trade_cost_per_unit=0.5)
    bt = strategy.backtest_spread_strategy(
        spread=pair_res.spread,
        signals=pair_res.signals,
        cfg=cfg,
    )

    print(bt.summary)

Note:
- This is a simplified educational backtester. It ignores margin, borrowing
  costs, and more detailed execution modeling.
- PnL is computed in "spread units" (1 unit = 1 Y share and beta X shares).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence, Tuple

import numpy as np
import pandas as pd

# Local imports (repo-style)
try:
    import core.pairs as pairs_mod
    import core.panel as panel
except ModuleNotFoundError:
    # Fallback if run from within core/ directly
    import pairs as pairs_mod  # type: ignore
    import panel  # type: ignore


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Configuration for the pair-spread backtest.

    Attributes
    ----------
    initial_capital : float
        Starting capital for the equity curve.
    notional_per_unit : float
        Scaling factor for each "spread unit".
        - The spread is defined as:
              spread_t = Y_t - (alpha + beta X_t)
          for one unit of long Y and beta units of short X.
        - PnL per day from one unit is Δspread.
        - Setting notional_per_unit = 1.0 means:
              pnl_t = position_{t-1} * Δspread_t
          in price units (e.g., dollars if prices are in dollars).
        - You can scale this up/down to represent larger or smaller trades.
    trade_cost_per_unit : float
        Transaction cost per 1-unit change in position.
        - If position changes from 0 -> +1 or +1 -> 0, cost = trade_cost_per_unit.
        - If position flips from +1 -> -1, cost = 2 * trade_cost_per_unit.
    annualization_factor : int
        Number of trading periods per year (252 for daily).
    risk_free_rate : float
        Annualized risk-free rate used for Sharpe calculations (in decimal form).
    """
    initial_capital: float = 100_000.0
    notional_per_unit: float = 1.0
    trade_cost_per_unit: float = 0.0
    annualization_factor: int = 252
    risk_free_rate: float = 0.0


@dataclass
class BacktestResult:
    """
    Container for backtest outputs.

    Attributes
    ----------
    equity_curve : pd.Series
        Equity value over time (index = date).
    pnl : pd.Series
        Daily PnL (after transaction costs).
    gross_pnl : pd.Series
        Daily PnL before transaction costs.
    costs : pd.Series
        Daily transaction costs.
    returns : pd.Series
        Daily returns = pnl_t / equity_{t-1}.
    positions : pd.Series
        Effective positions used for PnL (after any regime filtering).
    raw_signals : pd.Series
        Original input signals (before regime filtering).
    trades : pd.Series
        Change in position each day (delta position).
    summary : Dict[str, float]
        Scalar performance metrics (total return, Sharpe, max drawdown, etc.).
    metadata : Dict[str, Any]
        Extra info (e.g., config, notes).
    """
    equity_curve: pd.Series
    pnl: pd.Series
    gross_pnl: pd.Series
    costs: pd.Series
    returns: pd.Series
    positions: pd.Series
    raw_signals: pd.Series
    trades: pd.Series
    summary: Dict[str, float]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    """Ensure Series index is a DatetimeIndex and sorted."""
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _align_series(
    spread: pd.Series,
    signals: pd.Series,
    regimes: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
    """
    Align spread, signals, and (optionally) regimes on a common DatetimeIndex.

    Returns aligned copies of each (regimes may be None).
    """
    spread = _ensure_datetime_index(spread)
    signals = _ensure_datetime_index(signals)

    idx = spread.index.intersection(signals.index)
    spread = spread.loc[idx]
    signals = signals.loc[idx]

    if regimes is not None:
        regimes = _ensure_datetime_index(regimes)
        idx = idx.intersection(regimes.index)
        spread = spread.loc[idx]
        signals = signals.loc[idx]
        regimes = regimes.loc[idx]
        return spread, signals, regimes

    return spread, signals, None


def _compute_max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown as a negative value (e.g., -0.25 for -25%).

    Drawdown_t = equity_t / max_to_date_t - 1.
    """
    eq = equity.astype(float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    return float(dd.min())


# ---------------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------------

def backtest_spread_strategy(
    spread: pd.Series,
    signals: pd.Series,
    cfg: BacktestConfig,
    regimes: Optional[pd.Series] = None,
    allowed_regimes: Optional[Sequence[int]] = None,
) -> BacktestResult:
    """
    Backtest a mean-reversion spread strategy given a spread and position signals.

    PnL model:
        - One "spread unit" corresponds to long 1 Y and short beta X
          as defined when computing the spread:
              spread_t = Y_t - (alpha + beta X_t)
        - The value of one spread unit at time t is spread_t.
        - Daily PnL from one unit = spread_t - spread_{t-1}.
        - With position_{t-1} spread units, daily PnL is:
              pnl_t = position_{t-1} * (spread_t - spread_{t-1}) * cfg.notional_per_unit

    To avoid look-ahead bias:
        - We apply the *lagged* position (position_{t-1}) to the return
          from t-1 -> t.

    Regime-awareness (optional):
        - If `regimes` and `allowed_regimes` are provided, positions are set to 0
          on dates where the regime is not allowed, i.e.:

              if regime_t not in allowed_regimes:
                  effective_position_t = 0

        - This lets you test strategies that only trade in certain regimes.

    Parameters
    ----------
    spread : pd.Series
        Spread time series (from core.pairs.PairPipelineResult.spread).
    signals : pd.Series
        Raw position signals, e.g., -1, 0, +1 (from PairPipelineResult.signals).
    cfg : BacktestConfig
        Backtest configuration (capital, costs, etc.).
    regimes : pd.Series, optional
        Regime labels indexed by date (e.g., from core.regimes.RegimeFitResult.regimes).
    allowed_regimes : sequence of int, optional
        Regime labels in which trading is allowed. If None or if regimes is None,
        positions are not regime-filtered.

    Returns
    -------
    BacktestResult
    """
    # Align all series on a common index
    spread, signals, regimes = _align_series(spread, signals, regimes)

    raw_signals = signals.astype(float).copy()

    # Apply regime filter if provided
    if regimes is not None and allowed_regimes is not None:
        mask_allowed = regimes.isin(list(allowed_regimes))
        signals = signals.where(mask_allowed, 0.0)

    positions = signals.astype(float)

    # Compute spread change (Δspread_t = spread_t - spread_{t-1})
    spread = spread.astype(float)
    spread_diff = spread.diff()

    # Use lagged position for PnL to avoid look-ahead
    pos_lag = positions.shift(1).fillna(0.0)

    gross_pnl = pos_lag * spread_diff * float(cfg.notional_per_unit)
    gross_pnl = gross_pnl.fillna(0.0)

    # Transaction costs: proportional to change in position
    pos_change = positions.diff().fillna(0.0)
    costs = cfg.trade_cost_per_unit * pos_change.abs()

    pnl = gross_pnl - costs

    # Equity curve
    equity = cfg.initial_capital + pnl.cumsum()
    equity.name = "equity"

    # Daily returns: pnl_t / equity_{t-1}
    equity_lag = equity.shift(1)
    returns = pnl / equity_lag
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    returns.name = "return"

    # Summary statistics
    n_days = len(returns)
    total_pnl = float(pnl.sum())
    total_return = float(equity.iloc[-1] / cfg.initial_capital - 1.0) if n_days > 0 else 0.0
    avg_daily_ret = float(returns.mean())
    vol_daily = float(returns.std(ddof=1)) if n_days > 1 else 0.0

    annual_factor = float(cfg.annualization_factor)
    annual_ret = (1.0 + total_return) ** (annual_factor / max(n_days, 1)) - 1.0 if n_days > 0 else 0.0
    annual_vol = vol_daily * np.sqrt(annual_factor)

    if annual_vol > 0:
        sharpe = (annual_ret - cfg.risk_free_rate) / annual_vol
    else:
        sharpe = np.nan

    max_dd = _compute_max_drawdown(equity)

    n_trades = int((pos_change != 0).sum())
    avg_trade_cost = float(costs[costs != 0].mean()) if (costs != 0).any() else 0.0

    summary = {
        "n_days": n_days,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "avg_trade_cost": avg_trade_cost,
    }

    metadata: Dict[str, Any] = {
        "config": cfg,
    }

    return BacktestResult(
        equity_curve=equity,
        pnl=pnl,
        gross_pnl=gross_pnl,
        costs=costs,
        returns=returns,
        positions=positions,
        raw_signals=raw_signals,
        trades=pos_change,
        summary=summary,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# High-level convenience helpers
# ---------------------------------------------------------------------------

def run_pair_strategy_from_pipeline(
    px: pd.DataFrame,
    pair_result: pairs_mod.PairPipelineResult,
    cfg: Optional[BacktestConfig] = None,
    regimes: Optional[pd.Series] = None,
    allowed_regimes: Optional[Sequence[int]] = None,
) -> BacktestResult:
    """
    Convenience wrapper that:
    - Takes a PairPipelineResult (from core.pairs.build_pair_signals_from_panel),
    - Runs the spread backtest on its spread & signals,
    - Optionally applies regime filtering.

    Parameters
    ----------
    px : pd.DataFrame
        Price panel (not directly used here, but kept to mirror other
        pipelines and for potential extensions).
    pair_result : PairPipelineResult
        Output from core.pairs.build_pair_signals_from_panel.
    cfg : BacktestConfig or None
        Backtest configuration. If None, use defaults.
    regimes : pd.Series, optional
        Regime labels indexed by date.
    allowed_regimes : sequence of int, optional
        Regimes in which trading is allowed.

    Returns
    -------
    BacktestResult
    """
    if cfg is None:
        cfg = BacktestConfig()

    bt = backtest_spread_strategy(
        spread=pair_result.spread,
        signals=pair_result.signals,
        cfg=cfg,
        regimes=regimes,
        allowed_regimes=allowed_regimes,
    )
    # Attach some extra metadata about the pair
    bt.metadata["pair_spec"] = pair_result.spec
    bt.metadata["hedge_ratio"] = pair_result.hedge_ratio
    bt.metadata["hedge_config"] = pair_result.hedge_config
    bt.metadata["signal_config"] = pair_result.signal_config

    return bt


def run_pair_strategy_from_config(
    spec: pairs_mod.PairSpec,
    hedge_cfg: Optional[pairs_mod.HedgeConfig] = None,
    signal_cfg: Optional[pairs_mod.SignalConfig] = None,
    zscore_window: int = 60,
    backtest_cfg: Optional[BacktestConfig] = None,
    regimes: Optional[pd.Series] = None,
    allowed_regimes: Optional[Sequence[int]] = None,
    return_method: str = "log",
    force_refresh: bool = False,
    cache_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, pairs_mod.PairPipelineResult, BacktestResult]:
    """
    High-level end-to-end pipeline for a single pair:

    - Load prices from config/data_config.json.
    - Build pair signals (spread, z-score, position).
    - Run the backtest on the spread strategy (optionally regime-filtered).

    Parameters
    ----------
    spec : PairSpec
        Pair definition (X, Y).
    hedge_cfg, signal_cfg :
        Configs for pair signal generation. If None, defaults are used.
    zscore_window : int
        Window for spread z-score in the pair pipeline.
    backtest_cfg : BacktestConfig or None
        Backtest configuration. If None, defaults are used.
    regimes : pd.Series, optional
        Regime labels indexed by date (if you want regime-aware trading).
    allowed_regimes : sequence of int, optional
        Regimes in which trading is allowed. If None, regime-neutral.
    return_method : {"simple", "log"}
        Passed to panel.load_prices_and_returns_from_config (not critical here).
    force_refresh : bool
        Force fresh price download.
    cache_key : str or None
        Optional explicit cache key for the price parquet.

    Returns
    -------
    (px, pair_result, bt_result)
        px          : price panel
        pair_result : PairPipelineResult (spread + signals)
        bt_result   : BacktestResult
    """
    px, _rets = panel.load_prices_and_returns_from_config(
        method=return_method,
        force_refresh=force_refresh,
        cache_key=cache_key,
    )

    pair_result = pairs_mod.build_pair_signals_from_panel(
        px=px,
        spec=spec,
        hedge_cfg=hedge_cfg,
        signal_cfg=signal_cfg,
        zscore_window=zscore_window,
    )

    bt_cfg = backtest_cfg or BacktestConfig()

    bt_result = run_pair_strategy_from_pipeline(
        px=px,
        pair_result=pair_result,
        cfg=bt_cfg,
        regimes=regimes,
        allowed_regimes=allowed_regimes,
    )

    return px, pair_result, bt_result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # From project root:
    #   python core/strategy.py
    #
    # It will:
    #   - load prices from config/data_config.json
    #   - build a pair using the first two tickers
    #   - run a simple backtest
    try:
        px, rets = panel.load_prices_and_returns_from_config()
        tickers = list(px.columns.astype(str))
        if len(tickers) < 2:
            raise RuntimeError("Need at least two tickers in config to test strategy.")

        spec = pairs_mod.PairSpec(x=tickers[0], y=tickers[1])
        pair_result = pairs_mod.build_pair_signals_from_panel(px, spec)

        cfg = BacktestConfig(initial_capital=100_000, trade_cost_per_unit=0.0)
        bt = run_pair_strategy_from_pipeline(px, pair_result, cfg=cfg)

    except FileNotFoundError as e:
        print(e)
        print(
            "\nCreate config/data_config.json first. Example:\n"
            '{\n'
            '  "data_source": "yahoo",\n'
            '  "tickers": ["SPY", "QQQ", "IWM"],\n'
            '  "start_date": "2016-01-01",\n'
            '  "end_date": null,\n'
            '  "interval": "1d"\n'
            '}\n'
        )
    except Exception as e:
        print(f"[strategy] Error in CLI test: {e}")
    else:
        print("\n=== Strategy backtest summary ===")
        print(f"Pair:           X={pair_result.spec.x}, Y={pair_result.spec.y}")
        print(f"Hedge ratio β:  {pair_result.hedge_ratio:.4f}")
        print(f"Total return:   {bt.summary['total_return']:.2%}")
        print(f"Annual return:  {bt.summary['annual_return']:.2%}")
        print(f"Annual vol:     {bt.summary['annual_vol']:.2%}")
        print(f"Sharpe:         {bt.summary['sharpe']:.2f}")
        print(f"Max drawdown:   {bt.summary['max_drawdown']:.2%}")
        print(f"# trades:       {bt.summary['n_trades']}")

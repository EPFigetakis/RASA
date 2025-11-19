from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from .strategy import StrategyParams, compute_target_gross, target_shares, trade_cost_dollars


@dataclass
class TradeState:
    in_pos: bool = False
    side: int = 0              # +1 long A/short B, -1 short A/long B
    nA: float = 0.0
    nB: float = 0.0
    gross: float = 0.0
    entry_idx: int = -1
    entry_date: pd.Timestamp | None = None
    entry_z: float = np.nan
    costs_open: float = 0.0
    pnl_open_to_date: float = 0.0


def backtest_pair(
    ks: pd.DataFrame,   # output of kalman_spread (index=dates, cols: price_a, price_b, beta, z_resid, resid)
    params: StrategyParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Event-driven backtest for a single pair using Kalman residual z-score.

    Returns:
      trades_df: each round-trip trade summary
      equity_df: daily pnl, cum_pnl, position flags
    """
    req_cols = {"price_a", "price_b", "beta", "z_resid", "resid"}
    if not req_cols.issubset(set(ks.columns)):
        raise ValueError(f"kalman DataFrame missing columns: {req_cols - set(ks.columns)}")

    # risk sigma for sizing
    resid_sigma = ks["resid"].rolling(params.risk_window).std(ddof=1)

    dates = ks.index
    price_a = ks["price_a"].values
    price_b = ks["price_b"].values
    beta = ks["beta"].values
    z = ks["z_resid"].values

    # outputs
    records = []
    eq_rows = []

    st = TradeState()
    cum_pnl = 0.0

    for t in range(1, len(dates)):  # start at 1 for price diff
        dt = dates[t]
        pa, pb = price_a[t], price_b[t]
        pa_prev, pb_prev = price_a[t-1], price_b[t-1]
        bt = beta[t]
        zt = z[t]
        sig = float(resid_sigma.iloc[t]) if not np.isnan(resid_sigma.iloc[t]) else np.nan

        # mark-to-market for open position
        day_pnl = 0.0
        if st.in_pos:
            day_pnl = st.nA * (pa - pa_prev) + st.nB * (pb - pb_prev)
            st.pnl_open_to_date += day_pnl

        exit_signal = st.in_pos and (
            (abs(zt) <= params.exit_z) or
            (abs(zt) >= params.hard_stop_z) or
            ((t - st.entry_idx) >= params.max_holding_days)
        )

        if st.in_pos and exit_signal:
            # closing costs at today's prices
            cost_close = trade_cost_dollars(pa, pb, st.nA, st.nB, params)
            trade_pnl = st.pnl_open_to_date - cost_close
            cum_pnl += (day_pnl - cost_close)  # include today's mtm + costs

            records.append({
                "entry_date": st.entry_date,
                "exit_date": dt,
                "side": st.side,
                "entry_z": st.entry_z,
                "exit_z": zt,
                "days_held": (t - st.entry_idx),
                "gross": st.gross,
                "pnl_dollars": trade_pnl,
                "costs_open": st.costs_open,
                "costs_close": cost_close,
            })

            # reset state
            st = TradeState()
        else:
            # if not exiting, add mtm to equity
            cum_pnl += day_pnl

        # open logic (only if not in position)
        if (not st.in_pos) and np.isfinite(zt) and np.isfinite(bt) and not np.isnan(sig):
            if abs(zt) >= params.entry_z:
                side = -1 if zt > 0 else +1
                gross = compute_target_gross(zt, sig, params)
                nA, nB = target_shares(pa, pb, bt, side, gross)
                if (abs(nA) > 0) or (abs(nB) > 0):
                    cost_open = trade_cost_dollars(pa, pb, nA, nB, params)
                    st = TradeState(
                        in_pos=True,
                        side=side,
                        nA=nA,
                        nB=nB,
                        gross=gross,
                        entry_idx=t,
                        entry_date=dt,
                        entry_z=zt,
                        costs_open=cost_open,
                        pnl_open_to_date=-cost_open   # we pay costs at open
                    )
                    cum_pnl -= cost_open  # reflect open cost in equity

        eq_rows.append({
            "date": dt,
            "cum_pnl": cum_pnl,
            "in_pos": st.in_pos,
            "z": zt,
            "beta": bt,
            "nA": st.nA if st.in_pos else 0.0,
            "nB": st.nB if st.in_pos else 0.0,
            "gross": st.gross if st.in_pos else 0.0,
        })

    trades_df = pd.DataFrame.from_records(records)
    equity_df = pd.DataFrame.from_records(eq_rows).set_index("date")

    return trades_df, equity_df

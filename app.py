"""
app.py

Streamlit prototype for the RASA (Regime-Aware Statistical Arbitrage) project.

Current capabilities:
- Load price data/returns from config/data_config.json
- Run pair selection (Engle-Granger cointegration + half-life)
- Select a candidate pair
- Build pair signals (spread, z-score, positions)
- Run a simple spread backtest
- Visualize spread, z-score+position, and equity curve

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Import your core modules
from core import panel, pair_selection, pairs as pairs_mod, strategy


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize keys in st.session_state we want to reuse."""
    defaults = {
        "px": None,          # price panel
        "rets": None,        # returns panel
        "pairs_df": None,    # candidate pairs table
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_universe_from_config() -> None:
    """Load prices/returns via core.panel and cache in session_state."""
    try:
        px, rets = panel.load_prices_and_returns_from_config()
    except FileNotFoundError as e:
        st.error(
            "Could not find config/data_config.json.\n\n"
            f"{e}\n\n"
            "Create the config file first (tickers, start_date, etc.)."
        )
        return
    except Exception as e:
        st.error(f"Error loading prices: {e}")
        return

    st.session_state["px"] = px
    st.session_state["rets"] = rets
    st.success(
        f"Loaded price panel with {px.shape[1]} tickers and {px.shape[0]} rows."
    )


def run_pair_selection_ui() -> None:
    """Run pair selection based on current price panel and UI parameters."""
    px: Optional[pd.DataFrame] = st.session_state.get("px")
    if px is None:
        st.warning("Load data first (left sidebar).")
        return

    st.write("### Pair selection configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_pvalue = st.slider(
            "Max cointegration p-value",
            min_value=0.0,
            max_value=0.2,
            value=0.05,
            step=0.005,
        )
    with col2:
        min_obs = st.number_input(
            "Min overlapping observations",
            min_value=50,
            max_value=5000,
            value=252,
            step=10,
        )
    with col3:
        max_pairs = st.number_input(
            "Max pairs to keep",
            min_value=5,
            max_value=200,
            value=20,
            step=5,
        )

    use_log_prices = st.checkbox("Use log prices for hedge estimation", value=False)
    add_intercept = st.checkbox("Include intercept in hedge regression", value=False)

    if st.button("Run pair selection", type="primary"):
        cfg = pair_selection.PairSelectionConfig(
            max_pvalue=max_pvalue,
            min_obs=int(min_obs),
            use_log_prices=use_log_prices,
            add_intercept=add_intercept,
            max_pairs=int(max_pairs),
            save_to_csv=True,   # keep default: write to cache/pairs/pair_candidates.csv
        )
        with st.spinner("Scanning for candidate pairs..."):
            try:
                pairs_df = pair_selection.find_candidate_pairs(px, cfg)
            except Exception as e:
                st.error(f"Pair selection failed: {e}")
                return

        if pairs_df.empty:
            st.warning("No candidate pairs found with the given criteria.")
        else:
            st.session_state["pairs_df"] = pairs_df
            st.success(f"Found {len(pairs_df)} candidate pairs.")

    # Display existing results if present
    pairs_df = st.session_state.get("pairs_df")
    if isinstance(pairs_df, pd.DataFrame) and not pairs_df.empty:
        st.write("### Candidate pairs")
        st.dataframe(pairs_df, use_container_width=True)


def run_single_pair_view() -> None:
    """UI to inspect a single pair: spread, z-score, backtest."""
    px: Optional[pd.DataFrame] = st.session_state.get("px")
    pairs_df: Optional[pd.DataFrame] = st.session_state.get("pairs_df")

    if px is None:
        st.warning("Load data first (left sidebar).")
        return

    if pairs_df is None or pairs_df.empty:
        st.warning("Run pair selection first to pick a pair.")
        return

    st.write("## Pair inspection & backtest")

    # Build selection options
    options = [
        f"{row.x} / {row.y} (p={row.pvalue:.4f}, hl={row.half_life:.1f})"
        for row in pairs_df.itertuples()
    ]
    choice = st.selectbox("Choose a pair", options)
    idx = options.index(choice)
    row = pairs_df.iloc[idx]

    x = str(row["x"])
    y = str(row["y"])

    st.write(f"### Selected pair: **{x} / {y}**")

    # Parameters for this prototype (you can expose them in the UI later)
    hedge_cfg = pairs_mod.HedgeConfig(
        use_log_prices=False,
        add_intercept=False,
    )
    signal_cfg = pairs_mod.SignalConfig(
        entry_z=2.0,
        exit_z=0.5,
        max_position=1,
    )
    zscore_window = 60

    # Build pair signals
    try:
        pair_res = pairs_mod.build_pair_signals_from_panel(
            px=px,
            spec=pairs_mod.PairSpec(x=x, y=y),
            hedge_cfg=hedge_cfg,
            signal_cfg=signal_cfg,
            zscore_window=zscore_window,
        )
    except Exception as e:
        st.error(f"Error building pair signals: {e}")
        return

    # Backtest config (simple prototype)
    bt_cfg = strategy.BacktestConfig(
        initial_capital=100_000.0,
        notional_per_unit=1.0,
        trade_cost_per_unit=0.0,
    )

    try:
        bt = strategy.run_pair_strategy_from_pipeline(
            px=px,
            pair_result=pair_res,
            cfg=bt_cfg,
            regimes=None,          # regime-aware later
            allowed_regimes=None,
        )
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        return

    # ------------------------------------------------------------------
    # Show metrics
    # ------------------------------------------------------------------
    st.write("### Backtest summary")

    s = bt.summary
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total return", f"{s['total_return']:.2%}")
    mcol2.metric("Annual return", f"{s['annual_return']:.2%}")
    mcol3.metric("Sharpe", f"{s['sharpe']:.2f}" if pd.notna(s['sharpe']) else "N/A")
    mcol4.metric("Max drawdown", f"{s['max_drawdown']:.2%}")

    mcol5, mcol6 = st.columns(2)
    mcol5.metric("Days", f"{s['n_days']}")
    mcol6.metric("Trades", f"{s['n_trades']}")

    # ------------------------------------------------------------------
    # Plots: spread, z-score + position, equity curve
    # ------------------------------------------------------------------
    st.write("### Spread")

    spread_df = pair_res.spread.to_frame(name="spread")
    st.line_chart(spread_df)

    st.write("### Z-score & position")

    zpos_df = pd.DataFrame(
        {
            "zscore": pair_res.zscore,
            "position": pair_res.signals,
        }
    ).dropna()
    st.line_chart(zpos_df)

    st.write("### Equity curve")

    eq_df = bt.equity_curve.to_frame(name="equity")
    st.line_chart(eq_df)

    with st.expander("Show raw series (tail)"):
        st.write("Spread (tail):")
        st.dataframe(spread_df.tail(), use_container_width=True)
        st.write("Z-score & position (tail):")
        st.dataframe(zpos_df.tail(), use_container_width=True)
        st.write("Equity (tail):")
        st.dataframe(eq_df.tail(), use_container_width=True)


# -----------------------------------------------------------------------------
# Main Streamlit app
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="RASA Quant Prototype",
        layout="wide",
    )
    init_session_state()

    st.sidebar.title("RASA Controls")

    st.sidebar.markdown("### 1. Data")
    if st.sidebar.button("Load data from config", type="primary"):
        load_universe_from_config()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 2. View")

    view = st.sidebar.radio(
        "Choose view",
        options=("Overview", "Pair selection", "Pair inspection"),
    )

    st.title("RASA – Regime-Aware Statistical Arbitrage (Prototype)")
    st.caption("Streamlit prototype – data, pair selection, and single-pair backtest.")

    # -------------------- Overview --------------------
    if view == "Overview":
        st.write("## Data overview")

        px: Optional[pd.DataFrame] = st.session_state.get("px")
        if px is None:
            st.info("No data loaded yet. Use the **Load data from config** button on the left.")
        else:
            st.write(
                f"Price panel loaded with **{px.shape[1]} tickers** and "
                f"**{px.shape[0]} rows**."
            )

            tickers = list(px.columns.astype(str))
            st.write("Tickers:", ", ".join(tickers))

            st.write("Date range:")
            st.write(f"- Start: {px.index.min().date()}")
            st.write(f"- End:   {px.index.max().date()}")

            with st.expander("Show head of price panel"):
                st.dataframe(px.head(), use_container_width=True)

            # Simple price chart: pick up to 3 tickers
            st.write("### Price chart (normalized)")
            sel = st.multiselect(
                "Select up to 3 tickers to plot",
                options=tickers,
                default=tickers[: min(3, len(tickers))],
            )
            if sel:
                # Normalize to 1.0 at start
                sub = px[sel].copy()
                sub = sub / sub.iloc[0]
                st.line_chart(sub)

    # -------------------- Pair selection --------------------
    elif view == "Pair selection":
        st.write("## Pair selection")
        run_pair_selection_ui()

    # -------------------- Pair inspection --------------------
    elif view == "Pair inspection":
        run_single_pair_view()


if __name__ == "__main__":
    main()

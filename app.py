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
import json

import pandas as pd
import streamlit as st
import os 
import sqlite3
import bcrypt
from typing import Optional
import math


# Import your core modules
import datetime
from core import panel, prices, pair_selection, pairs as pairs_mod, strategy, walkforward, execution, trade_proposals, paths, risk, broker_paper, daily_run, monitoring


APP_META_PATH = Path("config/app_meta.json")

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

# ---- DB path config (matches /home/manoli/RASA/core/auth.db) ----
BASE_DIR = Path(__file__).resolve().parent      # /home/manoli/RASA
DEFAULT_DB_PATH = BASE_DIR / "core" / "auth.db" # /home/manoli/RASA/core/auth.db

# Allow override via env var if you ever want it, otherwise use auth.db in core/
DB_PATH = os.environ.get("STREAMLIT_AUTH_DB_PATH", str(DEFAULT_DB_PATH))


def init_db_schema():
    """Ensure the users table exists in the auth DB."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def verify_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return False

    stored_hash = row[0].encode("utf-8")
    return bcrypt.checkpw(password.encode("utf-8"), stored_hash)


# Make sure table exists before any login logic runs
init_db_schema()


def login_form():
    st.set_page_config(
        page_title="RASA Quant Prototype",
        layout="wide",
        page_icon = "📈"
        
    )
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login_btn = st.button("Log in")

    if login_btn:
        if not username or not password:
            st.error("Please enter both username and password.")
            return False
        if verify_user(username, password):
            st.session_state["user"] = username
            st.success(f"Welcome, {username}!")
            st.rerun()  # refresh to show the app
        else:
            st.error("Invalid username or password.")
            return False

    return False

def require_login():
    if "user" not in st.session_state:
        login_form()
        st.stop()  # Don't run the rest of the app until logged in

# 🚧 Gate the app here
# RASA_DEV_SKIP_AUTH lets you run the app on a machine with no auth.db /
# no seeded users (e.g. a dev box not linked to the deployed database).
# Never set this in the deployed environment.
DEV_SKIP_AUTH = os.getenv("RASA_DEV_SKIP_AUTH") == "1"

if not os.getenv("RUNNING_TESTS") and not DEV_SKIP_AUTH:
    require_login()

# ✅ If we reach here, the user is authenticated
    st.sidebar.write(f"Logged in as **{st.session_state['user']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
elif DEV_SKIP_AUTH:
    st.session_state.setdefault("user", "local-dev")
    st.sidebar.warning("⚠️ Auth bypassed (RASA_DEV_SKIP_AUTH=1) — local dev only.")


@st.cache_data
def load_app_meta() -> dict:
    """
    Load app version and change notes from config/app_meta.json.

    Returns a dict with keys:
        - "version": str
        - "change_notes": list[str]
    Falls back to sensible defaults if the file is missing or invalid.
    """
    default = {
        "version": "0.0.0-dev",
        "change_notes": ["No change notes available. Edit config/app_meta.json to add some."],
    }

    if not APP_META_PATH.exists():
        return default

    try:
        with APP_META_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default

    version = str(data.get("version", default["version"]))
    notes = data.get("change_notes", default["change_notes"])
    if not isinstance(notes, list):
        notes = default["change_notes"]

    return {"version": version, "change_notes": notes}

def show_version_widget(app_meta: dict) -> None:
    """Small card showing app version and recent change notes."""
    version = app_meta.get("version", "unknown")
    notes = app_meta.get("change_notes", [])

    items_html = "".join(f"<li>{note}</li>" for note in notes)

    st.markdown(
        f"""
        <div class="rasa-version-card">
          <h3>🚀 App Version <span class="rasa-version-pill">{version}</span></h3>
          <p><strong>Change notes</strong></p>
          <ul>
            {items_html}
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_global_styles() -> None:
    """Inject a bit of CSS to style the version widget like a card."""
    st.markdown(
        """
        <style>
        .rasa-version-card {
            padding: 1rem 1.25rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(148, 163, 184, 0.6); /* slate-400-ish */
            background: linear-gradient(
                135deg,
                rgba(59, 130, 246, 0.06),
                rgba(236, 72, 153, 0.04)
            );
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25); /* deep but soft */
            margin-bottom: 1.5rem;
        }
        .rasa-version-card h3 {
            margin: 0 0 0.4rem 0;
            font-weight: 700;
        }
        .rasa-version-pill {
            display: inline-block;
            padding: 0.1rem 0.6rem;
            margin-left: 0.35rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(15, 23, 42, 0.9);
            color: #f9fafb;
        }
        .rasa-version-card p {
            margin: 0.35rem 0;
        }
        .rasa-version-card ul {
            margin: 0.25rem 0 0 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def init_session_state() -> None:
    """Initialize keys in st.session_state we want to reuse."""
    defaults = {
        "user": None,
        "px": None,          # price panel
        "rets": None,        # returns panel
        "pairs_df": None,    # candidate pairs table
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_universe_from_config(update: bool = False) -> None:
    """Load prices/returns via core.panel and cache in session_state.

    update=True appends any bars newer than the cache (even a single day behind).
    """
    try:
        px, rets = panel.load_prices_and_returns_from_config(
            update_if_stale=update,
            stale_after_business_days=0 if update else 2,
        )
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

    # Build config once; its cache_subdir + filename define where we load/save
    cfg = pair_selection.PairSelectionConfig(
        max_pvalue=max_pvalue,
        min_obs=int(min_obs),
        use_log_prices=use_log_prices,
        add_intercept=add_intercept,
        max_pairs=int(max_pairs),
        save_to_csv=True,  # ensure we write to CSV when computing
    )

    load_col, compute_col = st.columns(2)

    # -------- Load cached selection --------
    with load_col:
        if st.button("Load saved pair selection"):
            try:
                pairs_df = pair_selection.load_candidate_pairs(cfg, px=px, strict=True)
            except pair_selection.StaleCacheError as e:
                st.error(
                    "Saved pair selection doesn't match the loaded prices/settings, so it "
                    f"was not loaded: {e} Recompute it, or restore the matching settings."
                )
            except FileNotFoundError:
                st.warning(
                    "No saved pair selection found yet.\n"
                    "Run a pair selection once to create the cache."
                )
            except Exception as e:
                st.error(f"Failed to load pair selection: {e}")
            else:
                st.session_state["pairs_df"] = pairs_df
                check = pair_selection.check_candidate_pairs_cache(px, cfg)
                if check.status == "ok":
                    st.success(f"Loaded {len(pairs_df)} candidate pairs from cache (verified).")
                else:
                    st.warning(
                        f"Loaded {len(pairs_df)} candidate pairs, but: " + " ".join(check.reasons)
                    )

    # -------- Compute new selection --------
    with compute_col:
        if st.button("Compute new pair selection", type="primary"):
            with st.spinner("Scanning for candidate pairs..."):
                try:
                    pairs_df = pair_selection.find_candidate_pairs(px, cfg)
                except Exception as e:
                    st.error(f"Pair selection failed: {e}")
                else:
                    if pairs_df.empty:
                        st.warning("No candidate pairs found with the given criteria.")
                    else:
                        st.session_state["pairs_df"] = pairs_df
                        st.success(f"Found {len(pairs_df)} candidate pairs.")

    # -------- Show whatever is currently loaded in session_state --------
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


def run_trade_proposals_ui() -> None:
    """
    Propose hypothetical trades based on the latest signals for top candidate pairs.

    IMPORTANT: This is an educational / research tool only and does NOT constitute
    financial advice or a recommendation to trade.
    """
    px: Optional[pd.DataFrame] = st.session_state.get("px")
    pairs_df: Optional[pd.DataFrame] = st.session_state.get("pairs_df")

    st.write("## Trade proposals (experimental)")
    st.info(
        "These trade ideas are generated from the current pair signals and are meant "
        "for research/learning purposes only. They are **not** financial advice."
    )

    if px is None:
        st.warning("Load data first (left sidebar) before generating trade proposals.")
        return

    if pairs_df is None or pairs_df.empty:
        st.warning("Run pair selection first so we have candidate pairs to scan.")
        return

    # ---------------- Controls ----------------
    st.write("### Proposal configuration")

    max_pairs_available = len(pairs_df)
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider(
            "Top N candidate pairs to scan",
            min_value=1,
            max_value=max_pairs_available,
            value=min(10, max_pairs_available),
            step=1,
        )
    with col2:
        notional_per_pair = st.number_input(
            "Approximate notional per pair (e.g., USD)",
            min_value=0.0,
            value=10_000.0,
            step=1_000.0,
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        entry_z = st.number_input(
            "Entry Z threshold",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
        )
    with col4:
        exit_z = st.number_input(
            "Exit Z threshold",
            min_value=0.1,
            max_value=3.0,
            value=0.5,
            step=0.1,
        )
    with col5:
        zscore_window = st.number_input(
            "Z-score window (days)",
            min_value=20,
            max_value=252,
            value=60,
            step=5,
        )

    use_log_prices = st.checkbox("Use log prices for hedge estimation", value=False)
    add_intercept = st.checkbox("Include intercept in hedge regression", value=False)
    whole_shares = st.checkbox("Round to whole shares", value=False)

    generate = st.button("Generate trade proposals", type="primary")

    if not generate:
        return

    # ---------------- Build proposals (all logic lives in core.trade_proposals) ----------------
    try:
        cfg = trade_proposals.ProposalConfig(
            top_n=int(top_n),
            notional_per_pair=float(notional_per_pair),
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            zscore_window=int(zscore_window),
            use_log_prices=use_log_prices,
            add_intercept=add_intercept,
            whole_shares=whole_shares,
        )
        batch = trade_proposals.generate_trade_ideas(px, pairs_df, cfg)
    except ValueError as e:
        st.error(f"Invalid proposal settings: {e}")
        return

    st.caption(
        f"Prices through **{batch.as_of.date() if batch.as_of is not None else 'n/a'}** · "
        f"batch `{batch.batch_id}` · {batch.n_pairs_scanned} pairs scanned"
    )
    for w in batch.warnings:
        st.warning(w)

    flat = sum(1 for s_ in batch.skipped if s_.reason == "flat")
    problems = [s_ for s_ in batch.skipped if s_.reason != "flat"]
    for s_ in problems:
        st.warning(f"Skipped {s_.pair}: {s_.reason}")

    if not batch.ideas:
        st.info(
            f"No actionable trades among the top {top_n} pairs with the current "
            f"thresholds ({flat} flat)."
        )
        return

    proposals_df = trade_proposals.ideas_to_frame(batch.ideas)
    proposals_df["status"] = proposals_df["is_new_signal"].map(
        {True: "NEW today", False: "HELD"}
    )

    st.write(f"### Target positions ({len(proposals_df)} pairs, {flat} flat)")
    st.caption(
        "These are *target positions*, not orders. Without account positions the app "
        "cannot tell what is already held; **HELD** signals were entered on an earlier bar."
    )

    display_cols = [
        "pair", "side", "status", "bars_in_position", "zscore",
        "price_x", "price_y", "beta", "qty_x", "qty_y",
        "gross_notional", "net_notional", "pvalue", "half_life", "idea_id",
    ]
    st.dataframe(proposals_df[display_cols], use_container_width=True)

    with st.expander("Show full proposal table"):
        st.dataframe(proposals_df, use_container_width=True)

    st.download_button(
        "Download ideas (CSV)",
        proposals_df.to_csv(index=False).encode("utf-8"),
        file_name=f"trade_ideas_{batch.batch_id}.csv",
        mime="text/csv",
    )


def run_walk_forward_ui() -> None:
    """UI to run the formation/trading walk-forward loop (Phase 1 validity check)."""
    px: Optional[pd.DataFrame] = st.session_state.get("px")
    if px is None:
        st.warning("Load data first (left sidebar).")
        return

    st.write("## Walk-forward validation")
    st.caption(
        "Pairs are only ever selected, and hedge ratios only ever estimated, from a "
        "formation window strictly before the trading window they're scored on — "
        "this is what makes the results defensible instead of look-ahead-biased."
    )

    st.write("### Window configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        formation_years = st.number_input(
            "Formation window (years)", min_value=1.0, max_value=10.0, value=3.0, step=0.5
        )
    with col2:
        trading_years = st.number_input(
            "Trading window (years)", min_value=0.25, max_value=5.0, value=1.0, step=0.25
        )
    with col3:
        mode = st.radio("Mode", options=("rolling", "expanding"), horizontal=True)

    col4, col5 = st.columns(2)
    with col4:
        fdr_alpha = st.slider(
            "FDR alpha (multiple-testing control)",
            min_value=0.01, max_value=0.25, value=0.10, step=0.01,
        )
    with col5:
        min_stable_folds = st.number_input(
            "Min folds a pair must survive FDR in to be traded",
            min_value=1, max_value=5, value=2, step=1,
        )

    with st.expander("Pair selection settings (applied within each formation window)"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            max_pvalue = st.slider(
                "Max cointegration p-value", min_value=0.0, max_value=0.2, value=0.05, step=0.005
            )
        with sc2:
            min_obs = st.number_input(
                "Min overlapping observations", min_value=50, max_value=5000, value=252, step=10
            )
        with sc3:
            max_pairs = st.number_input(
                "Max candidate pairs per fold", min_value=5, max_value=200, value=50, step=5
            )

    sel_cfg = pair_selection.PairSelectionConfig(
        max_pvalue=max_pvalue,
        min_obs=int(min_obs),
        max_pairs=int(max_pairs),
    )
    with st.expander("Execution costs & sizing", expanded=True):
        notional_per_pair = st.number_input(
            "Notional per pair ($, long-leg Y)", min_value=1.0, max_value=1_000_000.0,
            value=10_000.0, step=1_000.0,
            help="Dollar size of one spread unit. Costs are only meaningful at a real "
                 "dollar scale; the portfolio averages pairs equally within each fold.",
        )
        use_costs = st.checkbox("Model execution costs", value=True)
        exec_cfg = None
        if use_costs:
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                commission_per_share = st.number_input(
                    "Commission ($/share)", min_value=0.0, max_value=0.1, value=0.005,
                    step=0.001, format="%.4f",
                )
                half_spread_bps = st.number_input(
                    "Half bid/ask spread (bps)", min_value=0.0, max_value=100.0, value=2.0, step=0.5,
                )
            with ec2:
                slippage_bps = st.number_input(
                    "Slippage (bps)", min_value=0.0, max_value=100.0, value=1.0, step=0.5,
                )
                borrow_bps = st.number_input(
                    "Short borrow (bps/yr)", min_value=0.0, max_value=2000.0, value=50.0, step=10.0,
                )
            with ec3:
                fill_delay = st.number_input(
                    "Fill delay (days)", min_value=0, max_value=5, value=0, step=1,
                    help="0 = fill at the signal-day close; 1 = fill one trading day later.",
                )
            exec_cfg = execution.ExecutionCostConfig(
                commission_per_share=float(commission_per_share),
                half_spread_bps=float(half_spread_bps),
                slippage_bps=float(slippage_bps),
                borrow_bps_annual=float(borrow_bps),
                fill_delay_days=int(fill_delay),
            )
            st.caption(
                "Spread and slippage are flat bps assumptions (the price panel has no "
                "bid/ask quotes). Positions still open at the end of a fold are not "
                "liquidated."
            )

    backtest_cfg = strategy.BacktestConfig(
        initial_capital=100_000.0,
        notional_per_unit=float(notional_per_pair),
        trade_cost_per_unit=0.0,
        execution=exec_cfg,
    )
    wf_cfg = walkforward.WalkForwardConfig(
        formation_years=float(formation_years),
        trading_years=float(trading_years),
        mode=mode,
        min_stable_folds=int(min_stable_folds),
        fdr_alpha=float(fdr_alpha),
    )

    if st.button("Run walk-forward validation", type="primary"):
        with st.spinner("Running walk-forward folds (this scans the universe once per fold)…"):
            try:
                result = walkforward.run_walk_forward(px, sel_cfg, backtest_cfg, wf_cfg)
            except Exception as e:
                st.error(f"Walk-forward run failed: {e}")
            else:
                st.session_state["wf_result"] = result

    result: Optional[walkforward.WalkForwardResult] = st.session_state.get("wf_result")
    if result is None:
        return

    if result.portfolio_equity.empty:
        st.warning(
            "No folds could be generated — the formation window is likely longer than "
            "the available price history. Try a shorter formation window."
        )
        return

    st.write("### Fold diagnostics")
    st.caption("Candidate counts shrinking left-to-right (raw -> FDR survivors -> traded) is expected.")
    st.dataframe(result.diagnostics, use_container_width=True)

    st.write("### Portfolio equity curve (out-of-sample, chained across folds)")
    s = result.portfolio_summary
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total return", f"{s['total_return']:.2%}")
    mcol2.metric("Annual return", f"{s['annual_return']:.2%}")
    mcol3.metric("Sharpe", f"{s['sharpe']:.2f}" if pd.notna(s['sharpe']) else "N/A")
    mcol4.metric("Max drawdown", f"{s['max_drawdown']:.2%}")

    chart_df = pd.DataFrame(
        {
            "net (after costs)": result.portfolio_equity,
            "gross (before costs)": result.portfolio_gross_equity,
        }
    )
    st.line_chart(chart_df)

    cs = result.cost_summary
    if cs and cs.get("total", 0.0) > 0:
        st.write("### Execution cost breakdown (portfolio-weighted, $)")
        ccols = st.columns(5)
        ccols[0].metric("Total", f"{cs['total']:,.2f}")
        ccols[1].metric("Commission", f"{cs['commission']:,.2f}")
        ccols[2].metric("Spread + slippage", f"{cs['spread'] + cs['slippage']:,.2f}")
        ccols[3].metric("Short borrow", f"{cs['borrow']:,.2f}")
        ccols[4].metric("Other per-unit", f"{cs['per_unit_cost']:,.2f}")

    with st.expander("Show fold-by-fold pair detail"):
        st.dataframe(result.fold_summary, use_container_width=True)


def run_paper_monitor_ui() -> None:
    """Monitoring + approval surface for the paper-trading system (read-mostly)."""
    st.write("## Paper trading monitor")
    st.caption(
        "Simulated account only. Proposals come from the daily runner "
        "(`python scripts/run_daily_proposals.py run`); nothing is ordered until you approve it here "
        "or with the `approve` command."
    )

    db = paths.state_dir() / "rasa_paper.db"
    if not db.exists():
        st.info("No paper account yet. Run the daily runner once to create it.")
        return

    st.button("Refresh")  # any click reruns the script and re-reads everything

    try:
        risk_cfg = risk.load_risk_config()
        pstatus = prices.get_cache_status_from_config()
        last_bar = (
            datetime.date.fromisoformat(pstatus["last_bar_date"]) if pstatus.get("last_bar_date") else None
        )
        broker = broker_paper.PaperBroker(db)
    except Exception as e:
        st.error(f"Could not open the paper account: {e}")
        return

    try:
        runs = daily_run.list_run_records()
        snap = monitoring.build_snapshot(broker, risk_cfg, runs, last_bar)

        # ---- alerts ----
        if not snap.alerts:
            st.success("All clear: no alerts.")
        for a in snap.alerts:
            text = f"**{a.code}** — {a.message}"
            {"critical": st.error, "warning": st.warning}.get(a.severity, st.info)(text)

        # ---- headline numbers ----
        c = st.columns(5)
        c[0].metric("Equity", f"${snap.equity:,.0f}" if snap.equity is not None else "n/a",
                    f"{snap.daily_pnl:+,.0f} today" if snap.daily_pnl is not None else None)
        c[1].metric("Total PnL", f"{snap.total_pnl:+,.0f}" if snap.total_pnl is not None else "n/a")
        c[2].metric("Drawdown", f"{snap.drawdown:.2%}", f"limit {risk_cfg.max_drawdown_pct:.0%}", delta_color="off")
        c[3].metric("Gross exposure", f"${snap.gross:,.0f}" if snap.gross is not None else "n/a",
                    f"{snap.gross_utilization:.0%} of limit" if snap.gross_utilization is not None else None,
                    delta_color="off")
        c[4].metric("Net exposure", f"${snap.net:,.0f}" if snap.net is not None else "n/a",
                    f"{snap.net_utilization:.0%} of limit" if snap.net_utilization is not None else None,
                    delta_color="off")
        st.caption(
            f"Environment **{snap.environment}** · prices through **{snap.price_last_bar}** "
            f"({snap.price_age_bdays} bday old) · broker at **{snap.broker_last_bar}** · "
            f"cash ${snap.cash:,.0f}"
        )

        if not snap.equity_history.empty:
            hist = snap.equity_history.copy()
            hist.index = pd.to_datetime(hist.index)
            st.write("### Equity")
            st.line_chart(hist[["equity"]])

        st.write("### Positions")
        if snap.positions.empty:
            st.caption("Flat.")
        else:
            st.dataframe(snap.positions, use_container_width=True)

        # ---- proposals awaiting approval ----
        st.write("### Proposals awaiting approval")
        if not snap.pending_proposals:
            st.caption("None.")
        else:
            last_run = snap.last_run
            st.caption(f"From run `{last_run.run_id}` (data through {last_run.as_of}). All passed the risk gates.")
            rows = []
            for t in snap.pending_proposals:
                rows.append({
                    "ticket": t.ticket_id, "pair": t.pair or t.purpose,
                    "orders": ", ".join(f"{l.side.value} {l.quantity:g} {l.symbol}" for l in t.legs),
                    "gross $": t.gross_notional,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            chosen = st.multiselect("Tickets to approve", [t.ticket_id for t in snap.pending_proposals])
            confirm = st.checkbox("I reviewed these orders and want them sent to the paper broker")
            if st.button("Approve & submit", disabled=not (chosen and confirm), type="primary"):
                try:
                    done = daily_run.approve_and_submit(last_run, broker, chosen, approved_by="streamlit")
                except daily_run.RunError as e:
                    st.error(f"Refused: {e}")
                else:
                    st.success(f"Submitted {len(done)} ticket(s). They fill on the next daily run.")

        # ---- working orders ----
        st.write("### Working orders")
        if not snap.working_tickets:
            st.caption("None.")
        else:
            st.dataframe(pd.DataFrame([{
                "ticket": t.ticket_id, "pair": t.pair or t.purpose, "status": t.status.value,
                "orders": ", ".join(
                    f"{l.side.value} {t.remaining_quantity(l.symbol):g}/{l.quantity:g} {l.symbol}" for l in t.legs),
            } for t in snap.working_tickets]), use_container_width=True)

        # ---- runs ----
        st.write("### Recent daily runs")
        if not runs:
            st.caption("No runs recorded yet.")
        else:
            st.dataframe(pd.DataFrame([{
                "run": r.run_id, "started": r.started_at[:19], "status": r.status, "data through": r.as_of,
                "data age (bd)": r.data_age_business_days, "ideas": r.n_ideas, "tickets": len(r.tickets),
                "submitted": len(r.submitted_ticket_ids),
            } for r in reversed(runs[-15:])]), use_container_width=True)
            with st.expander("Latest run: warnings and risk decisions"):
                lr = runs[-1]
                for w in lr.warnings:
                    st.write(f"- {w}")
                for d in lr.risk_report.get("decisions", []):
                    fails = [f"{ch['name']}: {ch['detail']}" for ch in d["checks"] if not ch["passed"]]
                    st.write(f"- `{d['ticket_id']}` {d['pair']}: "
                             + ("**approved**" if d["approved"] else "**rejected** — " + "; ".join(fails)))

        with st.expander("Audit log (last 50 events)"):
            st.dataframe(broker.events().tail(50).iloc[::-1], use_container_width=True)
    finally:
        broker.close()


# -----------------------------------------------------------------------------
# Main Streamlit app
# -----------------------------------------------------------------------------

def main() -> None:
    init_session_state()
    st.set_page_config(
        page_title="RASA Quant Prototype",
        layout="wide",
        page_icon = "📈"
        
    )
    inject_global_styles()
    #init_session_state()

    app_meta = load_app_meta()

    st.sidebar.title("RASA Controls")

    st.sidebar.markdown("### 1. Data")
    if st.sidebar.button("Load data from config", type="primary"):
        load_universe_from_config()
    if st.sidebar.button("Update prices (fetch new bars)"):
        with st.spinner("Fetching new bars…"):
            load_universe_from_config(update=True)

    try:
        cstatus = prices.get_cache_status_from_config()
    except Exception:
        cstatus = None
    if cstatus and cstatus["exists"]:
        behind = cstatus["business_days_behind"]
        msg = f"Price cache last bar: **{cstatus['last_bar_date']}** ({behind} business day(s) behind)"
        if cstatus["is_stale"]:
            st.sidebar.warning(msg + " — stale; click *Update prices*.")
        else:
            st.sidebar.caption(msg)
    elif cstatus:
        st.sidebar.caption("No price cache yet — load data to download.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 2. View")

    view = st.sidebar.radio(
        "Choose view",
        options=(
            "Overview",
            "Pair selection",
            "Pair inspection",
            "Trade proposals",
            "Walk-forward validation",
            "Paper trading monitor",
        ),
    )

    st.title("RASA – Regime-Aware Statistical Arbitrage (Prototype)")
    st.caption("Streamlit prototype – data, pair selection, and single-pair backtest.")

    # -------------------- Overview --------------------
    if view == "Overview":

        show_version_widget(app_meta)

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

    # -------------------- Trade proposals --------------------
    elif view == "Trade proposals":
        run_trade_proposals_ui()

    # -------------------- Walk-forward validation --------------------
    elif view == "Walk-forward validation":
        run_walk_forward_ui()

    # -------------------- Paper trading monitor --------------------
    elif view == "Paper trading monitor":
        run_paper_monitor_ui()

if __name__ == "__main__":
    main()

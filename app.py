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
from core import panel, pair_selection, pairs as pairs_mod, strategy


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
                pairs_df = pair_selection.load_candidate_pairs(cfg)
            except FileNotFoundError:
                st.warning(
                    "No saved pair selection found yet.\n"
                    "Run a pair selection once to create the cache."
                )
            except Exception as e:
                st.error(f"Failed to load pair selection: {e}")
            else:
                st.session_state["pairs_df"] = pairs_df
                st.success(f"Loaded {len(pairs_df)} candidate pairs from cache.")

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

    generate = st.button("Generate trade proposals", type="primary")

    if not generate:
        return

    # ---------------- Signal configuration ----------------
    hedge_cfg = pairs_mod.HedgeConfig(
        use_log_prices=use_log_prices,
        add_intercept=add_intercept,
    )
    signal_cfg = pairs_mod.SignalConfig(
        entry_z=float(entry_z),
        exit_z=float(exit_z),
        max_position=1,
    )

    # ---------------- Build proposals ----------------
    proposals = []

    subset = pairs_df.iloc[:top_n].copy()

    for i, row in subset.iterrows():
        x = str(row["x"])
        y = str(row["y"])
        spec = pairs_mod.PairSpec(x=x, y=y)

        try:
            pair_res = pairs_mod.build_pair_signals_from_panel(
                px=px,
                spec=spec,
                hedge_cfg=hedge_cfg,
                signal_cfg=signal_cfg,
                zscore_window=int(zscore_window),
            )
        except Exception as e:
            st.warning(f"Failed to build signals for pair {x}/{y}: {e}")
            continue

        # Latest signal / z-score / spread
        sig = pair_res.signals.dropna()
        if sig.empty:
            continue

        last_pos = float(sig.iloc[-1])
        last_time = sig.index[-1]

        # Skip flat positions (no trade)
        if last_pos == 0.0:
            continue

        z_series = pair_res.zscore.dropna()
        if last_time in z_series.index:
            z_last = float(z_series.loc[last_time])
        elif not z_series.empty:
            z_last = float(z_series.iloc[-1])
        else:
            z_last = float("nan")

        spread_series = pair_res.spread.dropna()
        if last_time in spread_series.index:
            spread_last = float(spread_series.loc[last_time])
        elif not spread_series.empty:
            spread_last = float(spread_series.iloc[-1])
        else:
            spread_last = float("nan")

        # Last prices
        # If the exact timestamp isn't in px (due to alignment), fall back to last non-NaN.
        try:
            price_y = float(px[y].dropna().loc[:last_time].iloc[-1])
            price_x = float(px[x].dropna().loc[:last_time].iloc[-1])
        except Exception:
            # Worst case: just take last available prices
            price_y = float(px[y].dropna().iloc[-1])
            price_x = float(px[x].dropna().iloc[-1])

        beta = float(pair_res.hedge_ratio)

        # Approximate notionally 1 "spread unit" cost:
        approx_unit_cost = abs(price_y) + abs(beta) * abs(price_x)
        if approx_unit_cost <= 0:
            units = 0.0
        else:
            units = notional_per_pair / approx_unit_cost

        # Position sign: +1 = long spread (long Y, short X); -1 = short spread
        # units is positive notionally; we attach the sign afterwards
        signed_units = units * last_pos

        qty_y = signed_units           # + => buy Y, - => sell Y
        qty_x = -beta * signed_units   # hedge leg

        if last_pos > 0:
            action = "LONG spread (buy Y, sell X)"
        elif last_pos < 0:
            action = "SHORT spread (sell Y, buy X)"
        else:
            action = "FLAT"

        proposals.append(
            {
                "pair": f"{x}/{y}",
                "x": x,
                "y": y,
                "as_of": last_time,
                "action": action,
                "position": last_pos,
                "units_est": units,
                "qty_y_est": qty_y,
                "qty_x_est": qty_x,
                "price_y": price_y,
                "price_x": price_x,
                "beta": beta,
                "zscore": z_last,
                "spread": spread_last,
                "notional_target": notional_per_pair,
                "pvalue": float(row.get("pvalue", float("nan"))),
                "half_life": float(row.get("half_life", float("nan"))),
            }
        )

    if not proposals:
        st.info(
            "No actionable trades found among the top "
            f"{top_n} pairs with the current thresholds."
        )
        return

    proposals_df = pd.DataFrame(proposals)

    # Sort by |zscore| (most extreme first)
    if "zscore" in proposals_df.columns:
        proposals_df = proposals_df.reindex(
            proposals_df["zscore"].abs().sort_values(ascending=False).index
        )

    st.write(f"### Proposed trades ({len(proposals_df)} found)")

    display_cols = [
        "pair",
        "as_of",
        "action",
        "zscore",
        "price_x",
        "price_y",
        "beta",
        "qty_y_est",
        "qty_x_est",
        "notional_target",
        "pvalue",
        "half_life",
    ]
    display_cols = [c for c in display_cols if c in proposals_df.columns]

    st.dataframe(proposals_df[display_cols], use_container_width=True)

    with st.expander("Show full proposal table"):
        st.dataframe(proposals_df, use_container_width=True)



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

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 2. View")

    view = st.sidebar.radio(
        "Choose view",
        options=("Overview", "Pair selection", "Pair inspection", "Trade proposals"),
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

if __name__ == "__main__":
    main()

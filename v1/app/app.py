# --- path shim: allow running this app from anywhere ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from core import config
from core.data import load_prices
from core.pairs import rank_candidate_pairs
from core.kalman import kalman_spread

st.set_page_config(page_title="RASA Stat-Arb", page_icon="📈", layout="wide")

# ---------- helpers (existing) ----------

@st.cache_data(show_spinner=False)
def _load_prices_cached(universe: tuple[str, ...], start: str, end: str | None):
    return load_prices(list(universe), start=start, end=end, data_dir=config.DATA_DIR)

@st.cache_data(show_spinner=False)
def _rank_pairs_cached(prices: pd.DataFrame,
                       windows: tuple[int, int, int],
                       min_overlap_days: int,
                       hl_lookback_days: int,
                       min_hl: int, max_hl: int,
                       pval_threshold: float | None,
                       max_pairs: int):
    return rank_candidate_pairs(
        prices_df=prices.copy(),
        windows=windows,
        max_pairs=max_pairs,
        min_overlap_days=min_overlap_days,
        hl_lookback_days=hl_lookback_days,
        min_half_life_days=min_hl,
        max_half_life_days=max_hl,
        pval_threshold=pval_threshold
    )

@st.cache_data(show_spinner=False)
def _kalman_cached(series_a: pd.Series, series_b: pd.Series,
                   roll_window: int, delta: float, R: float):
    return kalman_spread(series_a, series_b, roll_window=roll_window, delta=delta, R=R)

def _signal_text(z: float, entry: float, exit_: float) -> str:
    if not np.isfinite(z):
        return "No signal (z is NaN)."
    az = abs(z)
    if az < exit_:
        return f"No position (|z|={az:.2f} < exit {exit_})"
    if az < entry:
        return f"In band (|z|={az:.2f} between exit and entry)"
    return ("ENTER: Short A / Long β·B" if z > 0 else "ENTER: Long A / Short β·B") + f"  (z={z:.2f})"

def _bands_shapes(y_entry: float, y_exit: float, color="#999"):
    return [
        dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0= y_entry, y1= y_entry, line=dict(width=1, dash="dot", color=color)),
        dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=-y_entry, y1=-y_entry, line=dict(width=1, dash="dot", color=color)),
        dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0= y_exit,  y1= y_exit,  line=dict(width=1, dash="dash", color=color)),
        dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=-y_exit,  y1=-y_exit,  line=dict(width=1, dash="dash", color=color)),
    ]

# ---------- NEW helpers for Trade Now ----------

def _find_signals_take_files() -> list[Path]:
    d = Path(config.DATA_DIR)
    if not d.exists():
        return []
    return sorted(d.glob("signals_take_*.csv"))

def _latest_signals_take_file() -> Path | None:
    files = _find_signals_take_files()
    return files[-1] if files else None

def _round_to_multiple(x: float, multiple: int) -> int:
    if multiple <= 1:
        return int(np.round(x))
    return int(np.round(x / multiple) * multiple)

def _limit_price(side_action: str, ref_price: float, buffer_bps: float) -> float:
    """
    side_action: 'BUY' or 'SELL' for the leg.
    BUY -> limit ABOVE ref price by buffer; SELL -> limit BELOW ref price.
    """
    if buffer_bps <= 0:
        return ref_price
    bump = ref_price * (buffer_bps / 1e4)
    return ref_price + bump if side_action == "BUY" else ref_price - bump

def _build_order_tables_from_take_csv(df_take: pd.DataFrame,
                                      order_type: str = "market",
                                      limit_buffer_bps: float = 10.0,
                                      tif: str = "DAY",
                                      round_lot: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert screener 'signals_take_*.csv' into:
      - pairs_df (one row per pair: summary)
      - legs_df  (two rows per pair: executable legs)
    Expects columns from the updated screener:
      A,B,side,z,beta,price_a,price_b,shares_A_round,shares_B_round,exit_plan,est_hold_days,est_pnl_to_exit
    Falls back to shares_A/shares_B if rounded columns missing.
    """
    df = df_take.copy()

    # Ensure needed columns exist
    needed = ["A","B","side","z","beta","price_a","price_b","exit_plan","est_hold_days"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    # Choose share columns (rounded preferred)
    if "shares_A_round" in df.columns and "shares_B_round" in df.columns:
        df["qty_A"] = df["shares_A_round"].fillna(0)
        df["qty_B"] = df["shares_B_round"].fillna(0)
    else:
        # fallback to raw floats, then round to nearest lot
        df["qty_A"] = df.get("shares_A", pd.Series(0, index=df.index)).fillna(0)
        df["qty_B"] = df.get("shares_B", pd.Series(0, index=df.index)).fillna(0)

    # Apply (re)rounding by lot
    df["qty_A"] = df["qty_A"].apply(lambda x: _round_to_multiple(x, round_lot))
    df["qty_B"] = df["qty_B"].apply(lambda x: _round_to_multiple(x, round_lot))

    # Build legs
    leg_rows = []
    pair_rows = []

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ticket_prefix = f"TKT{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for i, r in df.iterrows():
        A, B = str(r["A"]), str(r["B"])
        side = str(r["side"]).upper() if isinstance(r["side"], str) else ""
        z = float(r["z"]) if np.isfinite(r.get("z", np.nan)) else np.nan
        beta = float(r["beta"]) if np.isfinite(r.get("beta", np.nan)) else np.nan
        pa = float(r["price_a"]) if np.isfinite(r.get("price_a", np.nan)) else np.nan
        pb = float(r["price_b"]) if np.isfinite(r.get("price_b", np.nan)) else np.nan
        qA = int(r["qty_A"]) if np.isfinite(r.get("qty_A", np.nan)) else 0
        qB = int(r["qty_B"]) if np.isfinite(r.get("qty_B", np.nan)) else 0
        exit_plan = r.get("exit_plan", "")
        est_hold = int(r.get("est_hold_days", 0)) if np.isfinite(r.get("est_hold_days", np.nan)) else 0
        est_pnl = float(r.get("est_pnl_to_exit", 0.0)) if np.isfinite(r.get("est_pnl_to_exit", np.nan)) else 0.0

        if qA == 0 and qB == 0:
            continue  # skip tiny/rounded-away entries

        # Determine leg actions from side
        if "LONG_A" in side:
            leg1_action, leg1_tkr, leg1_qty, leg1_ref = "BUY",  A, abs(qA), pa
            leg2_action, leg2_tkr, leg2_qty, leg2_ref = "SELL", B, abs(qB), pb
        elif "SHORT_A" in side:
            leg1_action, leg1_tkr, leg1_qty, leg1_ref = "SELL", A, abs(qA), pa
            leg2_action, leg2_tkr, leg2_qty, leg2_ref = "BUY",  B, abs(qB), pb
        else:
            # If side missing, infer from z: z<0 => long A; z>0 => short A
            if np.isfinite(z) and z < 0:
                leg1_action, leg1_tkr, leg1_qty, leg1_ref = "BUY",  A, abs(qA), pa
                leg2_action, leg2_tkr, leg2_qty, leg2_ref = "SELL", B, abs(qB), pb
                side = "LONG_A/SHORT_betaB"
            else:
                leg1_action, leg1_tkr, leg1_qty, leg1_ref = "SELL", A, abs(qA), pa
                leg2_action, leg2_tkr, leg2_qty, leg2_ref = "BUY",  B, abs(qB), pb
                side = "SHORT_A/LONG_betaB"

        # Limit prices if needed
        if order_type == "market":
            leg1_px, leg2_px = None, None
        else:
            leg1_px = _limit_price(leg1_action, leg1_ref, limit_buffer_bps) if np.isfinite(leg1_ref) else None
            leg2_px = _limit_price(leg2_action, leg2_ref, limit_buffer_bps) if np.isfinite(leg2_ref) else None

        pair_id = f"{ticket_prefix}_{A}_{B}"

        # pair summary row
        pair_rows.append({
            "ticket_id": pair_id,
            "timestamp": ts,
            "A": A, "B": B, "side": side,
            "z": z, "beta": beta,
            "price_a": pa, "price_b": pb,
            "qty_A": qA, "qty_B": qB,
            "order_type": order_type.upper(), "tif": tif,
            "exit_plan": exit_plan,
            "est_hold_days": est_hold,
            "est_pnl_to_exit": est_pnl
        })

        # leg rows
        leg_rows.append({
            "ticket_id": pair_id, "timestamp": ts, "pair": f"{A}/{B}", "leg": 1,
            "ticker": leg1_tkr, "action": leg1_action, "qty": leg1_qty,
            "order_type": order_type.upper(), "limit_price": (None if leg1_px is None else round(leg1_px, 4)),
            "tif": tif, "ref_price": (None if not np.isfinite(leg1_ref) else round(leg1_ref, 4))
        })
        leg_rows.append({
            "ticket_id": pair_id, "timestamp": ts, "pair": f"{A}/{B}", "leg": 2,
            "ticker": leg2_tkr, "action": leg2_action, "qty": leg2_qty,
            "order_type": order_type.upper(), "limit_price": (None if leg2_px is None else round(leg2_px, 4)),
            "tif": tif, "ref_price": (None if not np.isfinite(leg2_ref) else round(leg2_ref, 4))
        })

    pairs_df = pd.DataFrame(pair_rows)
    legs_df  = pd.DataFrame(leg_rows)
    return pairs_df, legs_df

# ============================= UI =============================

st.sidebar.header("⚙️ Controls")

# Universe picker for the dashboard tab
default_uni = tuple(config.UNIVERSE)
universe = st.sidebar.multiselect(
    "Universe", options=sorted(default_uni), default=list(default_uni),
)
if not universe:
    st.stop()
universe_t = tuple(universe)

# Date range
col_date1, col_date2 = st.sidebar.columns(2)
start_date = col_date1.text_input("Start", value=config.START_DATE)
end_date = col_date2.text_input("End (None=today)", value="" if config.END_DATE is None else str(config.END_DATE))
end_date = None if (end_date.strip() == "") else end_date

# Refresh cache button (existing)
if st.sidebar.button("🔄 Refresh latest prices"):
    with st.spinner("Refreshing price cache from Yahoo…"):
        _load_prices_cached.clear(); _rank_pairs_cached.clear(); _kalman_cached.clear()
        _ = load_prices(list(universe_t), start=start_date, end=end_date,
                        data_dir=config.DATA_DIR, force_refresh=True)
    st.success("Price cache refreshed.")
    st.rerun()

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧾 Trade Now"])

# -------------------- TAB 1: Dashboard --------------------
with tab1:
    st.title("RASA — Regime-Aware Statistical Arbitrage (Dashboard)")

    with st.spinner("Loading prices..."):
        prices = _load_prices_cached(universe_t, start_date, end_date)
    st.caption(f"Loaded {prices.shape[1]} tickers × {prices.shape[0]} rows.")

    with st.spinner("Ranking candidate pairs..."):
        w1, w2, w3 = config.WINDOWS
        windows = (w1, w2, w3)
        candidates = _rank_pairs_cached(
            prices, windows, config.MIN_OVERLAP_DAYS, config.HL_LOOKBACK_DAYS,
            config.MIN_HALF_LIFE_D, config.MAX_HALF_LIFE_D, config.PVAL_THRESHOLD, config.MAX_PAIRS
        )
    if candidates.empty:
        st.warning("No passing pairs under current filters.")
        st.stop()

    st.subheader("Candidate pairs")
    st.dataframe(candidates.head(50), use_container_width=True, hide_index=True)

    top_pair = tuple(candidates.iloc[0][["A","B"]].values)
    col_pick1, col_pick2, col_pick3 = st.columns([1,1,2])
    A = col_pick1.selectbox("Ticker A", options=sorted(prices.columns), index=sorted(prices.columns).index(top_pair[0]))
    B = col_pick2.selectbox("Ticker B", options=sorted(prices.columns), index=sorted(prices.columns).index(top_pair[1]))
    col_pick3.info(f"Top candidate: **{top_pair[0]} / {top_pair[1]}**")

    with st.spinner(f"Running Kalman model for {A}/{B}..."):
        ks = _kalman_cached(prices[A], prices[B], roll_window=60, delta=1e-5, R=1e-3)

    latest = ks.dropna().iloc[-1]
    st.subheader(f"Pair: {A} vs {B}")

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("β (latest)", f"{latest['beta']:.3f}")
    col_kpi2.metric("z_resid (latest)", f"{latest['z_resid']:.2f}")
    col_kpi3.metric("Price A", f"{latest['price_a']:.2f}")
    col_kpi4.metric("Fair A (implied)", f"{latest['fair_value_a']:.2f}")

    st.success(_signal_text(float(latest["z_resid"]), 2.0, 0.5))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ks.index, y=ks["price_a"], mode="lines", name=f"{A} price"))
    fig1.add_trace(go.Scatter(x=ks.index, y=ks["fair_value_a"], mode="lines", name=f"{A} fair (implied by {B})"))
    fig1.update_layout(title=f"{A}: Price vs Fair Value (implied by {B})", xaxis_title="Date", yaxis_title="Price",
                       legend=dict(orientation="h", y=1.02, x=0))
    fig1.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(ks, x=ks.index, y="beta", title="Time-varying Hedge Ratio β(t)")
    fig2.update_layout(xaxis_title="Date", yaxis_title="β")
    fig2.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(ks, x=ks.index, y="z_resid", title="Residual z-score")
    for shape in _bands_shapes(2.0, 0.5):
        fig3.add_shape(shape)
    fig3.update_layout(xaxis_title="Date", yaxis_title="z")
    fig3.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig3, use_container_width=True)

# -------------------- TAB 2: Trade Now --------------------
with tab2:
    st.title("Trade Now — Use latest screener CSV")

    # Locate available signals_take files
    take_files = _find_signals_take_files()
    if not take_files:
        st.warning(
            "No `signals_take_*.csv` found in data_cache/. "
            "Run the screener first:\n\n"
            "`python scripts/screener_signals.py --force_refresh --top_k 40`"
        )
        st.stop()

    # Pick file
    latest = _latest_signals_take_file()
    file_labels = [f.name for f in take_files]
    default_idx = file_labels.index(latest.name) if latest else 0
    sel_name = st.selectbox("Choose signals file", options=file_labels, index=default_idx)
    sel_path = Path(config.DATA_DIR) / sel_name

    # Order preferences
    with st.expander("Order preferences", expanded=True):
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        order_type = col1.selectbox("Order type", options=["market","limit"], index=0)
        limit_bps  = col2.number_input("Limit buffer (bps)", value=10.0, step=5.0, min_value=0.0)
        tif        = col3.selectbox("TIF", options=["DAY","GTC"], index=0)
        round_lot  = int(col4.number_input("Round lot", value=1, min_value=1, step=1))

    # Load, build tables, display
    df_take = pd.read_csv(sel_path)
    pairs_df, legs_df = _build_order_tables_from_take_csv(
        df_take, order_type=order_type, limit_buffer_bps=float(limit_bps), tif=tif, round_lot=round_lot
    )

    if pairs_df.empty:
        st.info("No entries in this signals file after rounding/settings.")
        st.stop()

    st.subheader("Order tickets — pair summary")
    show_cols = ["ticket_id","A","B","side","z","beta","price_a","price_b","qty_A","qty_B",
                 "order_type","tif","est_hold_days","exit_plan","est_pnl_to_exit"]
    st.dataframe(pairs_df[show_cols], use_container_width=True, hide_index=True)

    st.subheader("Order legs — ready to place")
    st.dataframe(legs_df, use_container_width=True, hide_index=True)

    # Downloads
    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "⬇️ Download pair tickets CSV",
        pairs_df.to_csv(index=False).encode("utf-8"),
        file_name=f"tickets_pairs_from_{sel_path.name}",
        mime="text/csv"
    )
    col_dl2.download_button(
        "⬇️ Download leg orders CSV",
        legs_df.to_csv(index=False).encode("utf-8"),
        file_name=f"tickets_legs_from_{sel_path.name}",
        mime="text/csv"
    )

    st.caption("These tickets are generated directly from the latest `signals_take_*.csv` and your order preferences.")

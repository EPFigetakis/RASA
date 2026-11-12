"""
core.trade_proposals

Turn the current pair signals into typed, UI-independent trade ideas.

This is the seam between the research layer (core.pairs / core.pair_selection)
and the later order/risk/broker layers. It deliberately knows nothing about
Streamlit.

A TradeIdea is a *target position* for one pair ("be long 12.4 Y / short 9.1 X
as of this close"), not an order. Whether any shares need to be traded depends
on what is already held, which requires account state that does not exist yet
(see the handover doc, Phase 2/3). `is_new_signal` / `bars_in_position` say
whether the position was entered on the latest bar or has been held for a
while, so a long-held signal is not mistaken for a fresh entry.

Typical usage:

    from core import panel, pair_selection, trade_proposals

    px, _ = panel.load_prices_and_returns_from_config()
    pairs_df = pair_selection.load_candidate_pairs(px=px)

    cfg = trade_proposals.ProposalConfig(top_n=10, notional_per_pair=10_000)
    batch = trade_proposals.generate_trade_ideas(px, pairs_df, cfg)
    print(trade_proposals.ideas_to_frame(batch.ideas))
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

# Local imports (repo-style)
try:
    import core.pairs as pairs_mod
    import core.prices as prices
except ModuleNotFoundError:
    import pairs as pairs_mod  # type: ignore
    import prices  # type: ignore


Side = Literal["LONG_SPREAD", "SHORT_SPREAD"]


def log(msg: str) -> None:
    print(f"[trade_proposals] {msg}")


# ---------------------------------------------------------------------------
# Config & result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProposalConfig:
    """
    Parameters that fully determine a batch of trade ideas.

    Attributes
    ----------
    top_n : int
        How many of the best-ranked candidate pairs (first rows of the table)
        to scan.
    notional_per_pair : float
        Target gross dollar exposure per pair (|Y leg| + |X leg|).
    entry_z, exit_z : float
        Mean-reversion entry/exit thresholds (see pairs.SignalConfig).
    zscore_window : int
        Rolling window for the spread z-score.
    use_log_prices, add_intercept : bool
        Hedge-ratio estimation options (see pairs.HedgeConfig).
    whole_shares : bool
        Round leg quantities to whole shares. Ideas whose Y leg rounds to zero
        are skipped. Default False: fractional quantities, since broker
        support has not been decided.
    """

    top_n: int = 10
    notional_per_pair: float = 10_000.0
    entry_z: float = 2.0
    exit_z: float = 0.5
    zscore_window: int = 60
    use_log_prices: bool = False
    add_intercept: bool = False
    whole_shares: bool = False

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.notional_per_pair <= 0:
            raise ValueError("notional_per_pair must be > 0")
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be smaller than entry_z")
        if self.zscore_window < 2:
            raise ValueError("zscore_window must be >= 2")


@dataclass(frozen=True)
class TradeIdea:
    """
    Target position for one pair as of one close.

    Signed quantities: positive = long, negative = short. A LONG_SPREAD idea is
    long Y / short X; SHORT_SPREAD is short Y / long X (with beta > 0).
    """

    x: str
    y: str
    as_of: pd.Timestamp
    side: Side
    position: int  # +1 long spread, -1 short spread
    zscore: float
    spread: float
    beta: float
    alpha: float
    price_x: float
    price_y: float
    qty_x: float
    qty_y: float
    notional_target: float
    bars_in_position: int
    pvalue: float = float("nan")
    half_life: float = float("nan")

    @property
    def pair(self) -> str:
        return f"{self.x}/{self.y}"

    @property
    def is_new_signal(self) -> bool:
        """True if this position was entered on the latest bar."""
        return self.bars_in_position <= 1

    @property
    def notional_x(self) -> float:
        return self.qty_x * self.price_x

    @property
    def notional_y(self) -> float:
        return self.qty_y * self.price_y

    @property
    def gross_notional(self) -> float:
        return abs(self.notional_x) + abs(self.notional_y)

    @property
    def net_notional(self) -> float:
        return self.notional_x + self.notional_y

    @property
    def target_positions(self) -> Dict[str, float]:
        """Symbol -> signed share quantity this idea wants held."""
        return {self.y: self.qty_y, self.x: self.qty_x}

    @property
    def idea_id(self) -> str:
        """Stable id: same pair, bar, side and hedge ratio always give the same id."""
        payload = f"{self.x}|{self.y}|{pd.Timestamp(self.as_of).date()}|{self.side}|{self.beta:.6f}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["as_of"] = pd.Timestamp(self.as_of).isoformat()
        d.update(
            idea_id=self.idea_id,
            pair=self.pair,
            is_new_signal=self.is_new_signal,
            notional_x=self.notional_x,
            notional_y=self.notional_y,
            gross_notional=self.gross_notional,
            net_notional=self.net_notional,
        )
        return d


@dataclass
class SkippedPair:
    pair: str
    reason: str


@dataclass
class ProposalBatch:
    """Everything produced by one proposal run, with provenance."""

    ideas: List[TradeIdea]
    skipped: List[SkippedPair]
    config: ProposalConfig
    as_of: Optional[pd.Timestamp]          # latest bar in the price panel
    data_age_business_days: Optional[int]  # weekdays between as_of and the latest expected bar
    panel_fingerprint: str
    n_pairs_scanned: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: List[str] = field(default_factory=list)

    @property
    def batch_id(self) -> str:
        payload = json.dumps(
            {
                "config": asdict(self.config),
                "panel": self.panel_fingerprint,
                "ideas": sorted(i.idea_id for i in self.ideas),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def size_pair(
    position: int,
    price_x: float,
    price_y: float,
    beta: float,
    notional: float,
    log_space: bool,
) -> tuple[float, float]:
    """
    Convert a spread position into signed (qty_x, qty_y) share quantities with
    gross exposure `notional` (|Y leg| + |X leg|).

    - Raw-price hedge (Y = a + b X): one unit is 1 share of Y and b shares of X,
      so shares of Y = notional / (Py + |b| Px).
    - Log-price hedge (log Y = a + b log X): b is an elasticity, so the X leg's
      *dollar* size is b times the Y leg's: dollars of Y = notional / (1 + |b|),
      dollars of X = b * dollars of Y.

    The X leg always opposes the Y leg for beta > 0.
    """
    if not (price_x > 0 and price_y > 0) or notional <= 0 or position == 0:
        return 0.0, 0.0
    if log_space:
        dollars_y = notional / (1.0 + abs(beta))
        qty_y = position * dollars_y / price_y
        qty_x = -beta * position * dollars_y / price_x
    else:
        units = notional / (price_y + abs(beta) * price_x)
        qty_y = position * units
        qty_x = -beta * position * units
    return qty_x, qty_y


def _bars_in_position(signals: pd.Series) -> int:
    """Number of consecutive bars (ending at the last one) with the same non-zero signal."""
    s = signals.dropna()
    if s.empty or s.iloc[-1] == 0:
        return 0
    last = s.iloc[-1]
    changed = s[s != last]
    if changed.empty:
        return len(s)
    return int((s.index > changed.index[-1]).sum())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_trade_ideas(
    px: pd.DataFrame,
    pairs_df: pd.DataFrame,
    cfg: Optional[ProposalConfig] = None,
) -> ProposalBatch:
    """
    Build the current (latest-bar) trade idea for each of the top-N candidate pairs.

    Pairs that are flat on the latest bar, fail to build, or cannot be sized are
    reported in `ProposalBatch.skipped` with a reason rather than silently dropped.
    Ideas are ordered by |z-score|, most extreme first.

    Parameters
    ----------
    px : pd.DataFrame
        Wide price panel.
    pairs_df : pd.DataFrame
        Candidate table (core.pair_selection format); only the first
        `cfg.top_n` rows are used, so pass it ranked best-first.
    """
    cfg = cfg or ProposalConfig()
    hedge_cfg = pairs_mod.HedgeConfig(
        use_log_prices=cfg.use_log_prices, add_intercept=cfg.add_intercept
    )
    signal_cfg = pairs_mod.SignalConfig(
        entry_z=cfg.entry_z, exit_z=cfg.exit_z, max_position=1
    )

    ideas: List[TradeIdea] = []
    skipped: List[SkippedPair] = []
    warnings: List[str] = []

    as_of = pd.Timestamp(px.index.max()) if len(px) else None
    age = prices.business_days_behind(as_of.date()) if as_of is not None else None
    if age is not None and age > prices.DEFAULT_STALE_AFTER_BUSINESS_DAYS:
        warnings.append(
            f"Price data ends {as_of.date()}, {age} business days behind; "
            "these ideas reflect old prices."
        )

    subset = pairs_df.iloc[: cfg.top_n]
    for row in subset.itertuples(index=False):
        x, y = str(row.x), str(row.y)
        pair = f"{x}/{y}"

        try:
            res = pairs_mod.build_pair_signals_from_panel(
                px=px,
                spec=pairs_mod.PairSpec(x=x, y=y),
                hedge_cfg=hedge_cfg,
                signal_cfg=signal_cfg,
                zscore_window=cfg.zscore_window,
            )
        except Exception as e:
            skipped.append(SkippedPair(pair, f"signal build failed: {e}"))
            continue

        sig = res.signals.dropna()
        if sig.empty:
            skipped.append(SkippedPair(pair, "no signal history"))
            continue
        position = int(sig.iloc[-1])
        if position == 0:
            skipped.append(SkippedPair(pair, "flat"))
            continue
        bar = sig.index[-1]

        z_series, spread_series = res.zscore.dropna(), res.spread.dropna()
        zscore = float(z_series.loc[bar]) if bar in z_series.index else float("nan")
        spread = float(spread_series.loc[bar]) if bar in spread_series.index else float("nan")

        try:
            price_x = float(px[x].dropna().loc[:bar].iloc[-1])
            price_y = float(px[y].dropna().loc[:bar].iloc[-1])
        except Exception as e:
            skipped.append(SkippedPair(pair, f"no price at signal date: {e}"))
            continue

        beta = float(res.hedge_ratio)
        alpha = float(res.metadata.get("alpha", 0.0))
        qty_x, qty_y = size_pair(
            position, price_x, price_y, beta, cfg.notional_per_pair, cfg.use_log_prices
        )
        if cfg.whole_shares:
            qty_x, qty_y = float(round(qty_x)), float(round(qty_y))
        if qty_y == 0.0 or not math.isfinite(qty_y) or not math.isfinite(qty_x):
            skipped.append(SkippedPair(pair, "position size is zero or invalid"))
            continue

        ideas.append(
            TradeIdea(
                x=x, y=y, as_of=bar,
                side="LONG_SPREAD" if position > 0 else "SHORT_SPREAD",
                position=position,
                zscore=zscore, spread=spread, beta=beta, alpha=alpha,
                price_x=price_x, price_y=price_y,
                qty_x=qty_x, qty_y=qty_y,
                notional_target=cfg.notional_per_pair,
                bars_in_position=_bars_in_position(res.signals),
                pvalue=float(getattr(row, "pvalue", float("nan"))),
                half_life=float(getattr(row, "half_life", float("nan"))),
            )
        )

    ideas.sort(key=lambda i: (-abs(i.zscore) if math.isfinite(i.zscore) else 0.0, i.pair))

    return ProposalBatch(
        ideas=ideas,
        skipped=skipped,
        config=cfg,
        as_of=as_of,
        data_age_business_days=age,
        panel_fingerprint=prices.panel_fingerprint(px) if len(px) else "",
        n_pairs_scanned=len(subset),
        warnings=warnings,
    )


def ideas_to_frame(ideas: List[TradeIdea]) -> pd.DataFrame:
    """Flat table of ideas (one row each), for display or export."""
    if not ideas:
        return pd.DataFrame()
    return pd.DataFrame([i.to_dict() for i in ideas])

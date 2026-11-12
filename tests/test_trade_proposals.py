# tests/test_trade_proposals.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import trade_proposals as tp


def _panel(z_tail=3.0, n=300, seed=0, end=None):
    """X random walk, Y = 2X + noise whose last bars are pushed to a high |z|."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=n)
    x = 100 + np.cumsum(rng.normal(size=n))
    noise = rng.normal(scale=1.0, size=n)
    noise[-3:] += z_tail * 3 if z_tail else 0  # push spread far above its mean at the end
    y = 2 * x + noise
    return pd.DataFrame({"X": x, "Y": y, "Z": 50 + np.cumsum(rng.normal(size=n))}, index=idx)


def _pairs():
    return pd.DataFrame(
        [{"x": "X", "y": "Y", "pvalue": 0.001, "hedge_ratio": 2.0, "alpha": 0.0,
          "half_life": 5.0, "n_obs": 300}]
    )


# ------------------------------- sizing -------------------------------

def test_size_pair_raw_gross_notional_and_hedge_direction():
    qty_x, qty_y = tp.size_pair(+1, price_x=50, price_y=100, beta=2.0, notional=10_000, log_space=False)
    assert qty_y > 0 and qty_x < 0                      # long Y / short X
    assert abs(qty_y) * 100 + abs(qty_x) * 50 == pytest.approx(10_000)
    assert qty_x == pytest.approx(-2.0 * qty_y)         # beta shares of X per share of Y

    qx2, qy2 = tp.size_pair(-1, 50, 100, 2.0, 10_000, False)
    assert (qx2, qy2) == pytest.approx((-qty_x, -qty_y))   # short spread mirrors long spread


def test_size_pair_log_space_beta_is_dollar_elasticity():
    qty_x, qty_y = tp.size_pair(+1, price_x=50, price_y=100, beta=0.5, notional=9_000, log_space=True)
    dollars_y, dollars_x = qty_y * 100, qty_x * 50
    assert dollars_y == pytest.approx(6_000)
    assert dollars_x == pytest.approx(-3_000)           # beta * Y dollars, opposite side
    assert abs(dollars_y) + abs(dollars_x) == pytest.approx(9_000)


def test_size_pair_degenerate_inputs_are_zero():
    assert tp.size_pair(0, 50, 100, 1.0, 1000, False) == (0.0, 0.0)
    assert tp.size_pair(1, 0, 100, 1.0, 1000, False) == (0.0, 0.0)
    assert tp.size_pair(1, 50, -1, 1.0, 1000, False) == (0.0, 0.0)


def test_bars_in_position():
    idx = pd.bdate_range("2024-01-01", periods=6)
    assert tp._bars_in_position(pd.Series([0, 0, 1, 1, 1, 1], index=idx)) == 4
    assert tp._bars_in_position(pd.Series([1, 1, 1, 1, 1, 1], index=idx)) == 6
    assert tp._bars_in_position(pd.Series([-1, -1, 0, 0, 1, 1], index=idx)) == 2
    assert tp._bars_in_position(pd.Series([1, 1, 0], index=idx[:3])) == 0


# ------------------------------- config -------------------------------

def test_config_validation():
    with pytest.raises(ValueError):
        tp.ProposalConfig(top_n=0)
    with pytest.raises(ValueError):
        tp.ProposalConfig(notional_per_pair=0)
    with pytest.raises(ValueError):
        tp.ProposalConfig(entry_z=1.0, exit_z=1.0)


# ------------------------------- generation -------------------------------

def test_generate_ideas_short_spread_when_spread_is_high():
    px = _panel()
    batch = tp.generate_trade_ideas(px, _pairs(), tp.ProposalConfig(notional_per_pair=10_000))
    assert len(batch.ideas) == 1
    idea = batch.ideas[0]
    assert idea.side == "SHORT_SPREAD" and idea.position == -1
    assert idea.qty_y < 0 < idea.qty_x
    assert idea.zscore > 2.0
    assert idea.gross_notional == pytest.approx(10_000)
    assert idea.as_of == px.index[-1]
    assert idea.target_positions == {"Y": idea.qty_y, "X": idea.qty_x}
    assert idea.bars_in_position >= 1
    assert batch.as_of == px.index[-1] and batch.n_pairs_scanned == 1
    assert batch.warnings == []


def test_flat_pairs_are_reported_as_skipped_not_dropped():
    px = _panel(z_tail=0.0)
    batch = tp.generate_trade_ideas(px, _pairs(), tp.ProposalConfig(entry_z=6.0, exit_z=1.0))
    assert batch.ideas == []
    assert [(s.pair, s.reason) for s in batch.skipped] == [("X/Y", "flat")]


def test_bad_pair_is_skipped_with_reason():
    px = _panel()
    bad = pd.concat([_pairs(), pd.DataFrame([{"x": "X", "y": "NOPE", "pvalue": 0.01}])])
    batch = tp.generate_trade_ideas(px, bad, tp.ProposalConfig(top_n=2))
    assert len(batch.ideas) == 1
    assert any(s.pair == "X/NOPE" and "failed" in s.reason for s in batch.skipped)


def test_top_n_limits_pairs_scanned():
    px = _panel()
    many = pd.concat([_pairs()] * 3, ignore_index=True)
    assert tp.generate_trade_ideas(px, many, tp.ProposalConfig(top_n=2)).n_pairs_scanned == 2


def test_whole_shares_rounds_and_skips_zero_size():
    px = _panel()
    batch = tp.generate_trade_ideas(px, _pairs(), tp.ProposalConfig(whole_shares=True))
    idea = batch.ideas[0]
    assert idea.qty_x == round(idea.qty_x) and idea.qty_y == round(idea.qty_y)

    tiny = tp.generate_trade_ideas(px, _pairs(), tp.ProposalConfig(notional_per_pair=1.0, whole_shares=True))
    assert tiny.ideas == []
    assert tiny.skipped[0].reason.startswith("position size")


def test_stale_prices_add_warning():
    px = _panel(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=30))
    batch = tp.generate_trade_ideas(px, _pairs(), tp.ProposalConfig())
    assert batch.data_age_business_days > 2
    assert any("business days behind" in w for w in batch.warnings)


def test_ids_are_deterministic_and_change_with_inputs():
    px = _panel()
    a = tp.generate_trade_ideas(px, _pairs(), tp.ProposalConfig())
    b = tp.generate_trade_ideas(px, _pairs(), tp.ProposalConfig())
    assert a.ideas[0].idea_id == b.ideas[0].idea_id
    assert a.batch_id == b.batch_id
    c = tp.generate_trade_ideas(px, _pairs(), tp.ProposalConfig(notional_per_pair=5_000))
    assert c.batch_id != a.batch_id
    d = tp.generate_trade_ideas(px.iloc[:-1], _pairs(), tp.ProposalConfig())
    assert d.batch_id != a.batch_id


def test_ideas_to_frame_columns_and_empty():
    assert tp.ideas_to_frame([]).empty
    df = tp.ideas_to_frame(tp.generate_trade_ideas(_panel(), _pairs(), tp.ProposalConfig()).ideas)
    for col in ("pair", "idea_id", "side", "is_new_signal", "gross_notional", "net_notional", "qty_x", "qty_y"):
        assert col in df.columns

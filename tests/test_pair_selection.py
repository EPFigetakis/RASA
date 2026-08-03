# tests/test_pair_selection.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import pair_selection


def test_estimate_half_life_recovers_known_decay_rate():
    rho = 0.9
    n = 200
    eps = np.empty(n)
    eps[0] = 10.0
    for t in range(1, n):
        eps[t] = rho * eps[t - 1]
    spread = pd.Series(eps)

    hl = pair_selection._estimate_half_life(spread)
    expected = -np.log(2.0) / np.log(rho)
    assert np.isclose(hl, expected, rtol=0.05)


def test_estimate_half_life_nan_for_too_short_series():
    spread = pd.Series([1.0, 2.0])
    assert np.isnan(pair_selection._estimate_half_life(spread))


def test_estimate_half_life_infinite_for_non_mean_reverting():
    # A pure random walk (rho ~= 1) should look nowhere near as mean-reverting
    # as the rho=0.9 case above (expected half-life ~7 days).
    rng = np.random.default_rng(1)
    spread = pd.Series(np.cumsum(rng.normal(size=300)))
    hl = pair_selection._estimate_half_life(spread)
    assert hl == np.inf or np.isnan(hl) or hl > 100


def _cointegrated_panel(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(scale=1.0, size=n))
    noise = rng.normal(scale=0.5, size=n)
    y = 2.0 * x + noise  # Y cointegrated with X
    z = 100 + np.cumsum(rng.normal(scale=1.0, size=n))  # independent random walk

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"X": x, "Y": y, "Z": z}, index=dates)


def test_evaluate_pair_respects_min_obs():
    px = _cointegrated_panel(n=50)
    cfg = pair_selection.PairSelectionConfig(min_obs=252, save_to_csv=False)
    stat = pair_selection.evaluate_pair(px, "X", "Y", cfg)
    assert stat is None


def test_evaluate_pair_returns_stat_for_cointegrated_pair():
    px = _cointegrated_panel(n=300)
    cfg = pair_selection.PairSelectionConfig(min_obs=100, save_to_csv=False)
    stat = pair_selection.evaluate_pair(px, "X", "Y", cfg)

    assert stat is not None
    assert stat.x == "X" and stat.y == "Y"
    assert stat.n_obs == 300
    assert np.isclose(stat.hedge_ratio, 2.0, atol=0.2)
    assert 0.0 <= stat.pvalue <= 1.0


def test_find_candidate_pairs_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pair_selection.paths, "cache_dir", lambda: tmp_path)

    px = _cointegrated_panel(n=300)
    cfg = pair_selection.PairSelectionConfig(
        max_pvalue=0.10,
        min_obs=100,
        max_pairs=10,
        save_to_csv=True,
    )

    df = pair_selection.find_candidate_pairs(px, cfg)
    assert not df.empty
    assert set(df.columns) == {"x", "y", "pvalue", "hedge_ratio", "alpha", "half_life", "n_obs"}

    cache_path = pair_selection.get_pairs_cache_path(cfg)
    assert cache_path.exists()

    loaded = pair_selection.load_candidate_pairs(cfg)
    pd.testing.assert_frame_equal(df, loaded)

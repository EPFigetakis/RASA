# tests/test_panel.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import panel


def _sample_prices():
    dates = pd.date_range("2021-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 101.0, 103.0],
            "BBB": [50.0, 49.0, 49.5, 50.0, 51.0],
        },
        index=dates,
    )


def test_compute_returns_simple_matches_pct_change():
    px = _sample_prices()
    rets = panel.compute_returns(px, method="simple")

    expected = px.pct_change().dropna(how="all")
    pd.testing.assert_frame_equal(rets, expected)
    # First (all-NaN) row must be dropped.
    assert len(rets) == len(px) - 1


def test_compute_returns_log_matches_log_diff():
    px = _sample_prices()
    rets = panel.compute_returns(px, method="log")

    expected = np.log(px).diff().dropna(how="all")
    pd.testing.assert_frame_equal(rets, expected)


def test_compute_returns_keep_na_when_requested():
    px = _sample_prices()
    rets = panel.compute_returns(px, method="simple", drop_na=False)
    assert len(rets) == len(px)
    assert rets.iloc[0].isna().all()


def test_normalize_prices_starts_at_base_value():
    px = _sample_prices()
    norm = panel.normalize_prices(px, base_value=100.0)

    assert np.allclose(norm.iloc[0].to_numpy(), [100.0, 100.0])
    # Relative moves should be preserved.
    expected_last_aaa = 100.0 * (px["AAA"].iloc[-1] / px["AAA"].iloc[0])
    assert np.isclose(norm["AAA"].iloc[-1], expected_last_aaa)

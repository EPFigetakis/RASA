# tests/test_prices.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import prices


def _make_raw_download(tickers, dates, low_coverage_ticker=None):
    """Build a fake yfinance multi-ticker download() result."""
    fields = ["Close", "Open", "High", "Low", "Volume"]
    columns = pd.MultiIndex.from_product([fields, tickers])
    data = np.random.default_rng(0).normal(loc=100, scale=1, size=(len(dates), len(columns)))
    raw = pd.DataFrame(data, index=dates, columns=columns)

    if low_coverage_ticker is not None:
        # Blank out most of one ticker's Close values so it fails the
        # >98% coverage filter.
        cutoff = int(len(dates) * 0.5)
        raw.loc[raw.index[:cutoff], ("Close", low_coverage_ticker)] = np.nan

    return raw


def test_build_cache_key_is_order_and_case_independent():
    k1 = prices.build_cache_key(["AAA", "bbb"], "2020-01-01", None, "1d")
    k2 = prices.build_cache_key(["BBB", "aaa"], "2020-01-01", None, "1d")
    assert k1 == k2


def test_build_cache_key_differs_on_universe_or_settings():
    base = prices.build_cache_key(["AAA", "BBB"], "2020-01-01", None, "1d")
    different_universe = prices.build_cache_key(["AAA", "CCC"], "2020-01-01", None, "1d")
    different_start = prices.build_cache_key(["AAA", "BBB"], "2021-01-01", None, "1d")
    different_end = prices.build_cache_key(["AAA", "BBB"], "2020-01-01", "2020-06-01", "1d")

    assert len({base, different_universe, different_start, different_end}) == 4


def test_load_prices_downloads_filters_and_caches(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    tickers = ["AAA", "BBB", "LOWCOV"]
    raw = _make_raw_download(tickers, dates, low_coverage_ticker="LOWCOV")

    calls = {"n": 0}

    def fake_download(tickers_arg, start, end, interval, auto_adjust, progress):
        calls["n"] += 1
        return raw

    monkeypatch.setattr(prices.yf, "download", fake_download)

    px = prices.load_prices(
        tickers=tickers,
        start="2020-01-01",
        data_dir=tmp_path,
        force_refresh=True,
    )

    # Low-coverage ticker should have been dropped.
    assert set(px.columns) == {"AAA", "BBB"}
    assert isinstance(px.index, pd.DatetimeIndex)
    assert calls["n"] == 1

    # A cache file + sidecar metadata should now exist.
    cache_files = list(tmp_path.glob("*.parquet"))
    assert len(cache_files) == 1
    meta = prices.read_cache_metadata(cache_files[0])
    assert meta is not None
    assert meta["tickers"] == ["AAA", "BBB", "LOWCOV"]
    assert meta["n_tickers_kept"] == 2

    # A second call with the same universe/settings should hit the cache
    # and must NOT call yfinance again.
    px2 = prices.load_prices(
        tickers=tickers,
        start="2020-01-01",
        data_dir=tmp_path,
        force_refresh=False,
    )
    assert calls["n"] == 1
    # Parquet round-tripping doesn't preserve the DatetimeIndex freq metadata.
    pd.testing.assert_frame_equal(px, px2, check_freq=False)


def test_load_prices_from_config_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        prices.load_prices_from_config(config_path=missing)

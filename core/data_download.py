"""
core.data_download

Config-driven downloader for OHLCV price data.

- Reads a JSON config (by default: config/data_config.json) that specifies
  the universe of tickers and date range.
- Downloads data (currently from Yahoo Finance via yfinance).
- Stacks all tickers into ONE long-format DataFrame.
- Saves it as a single CSV file in data/ (default: data/prices_universe.csv).

Typical usage from the project root:

    python core/data_download.py

or from another script:

    from core import data_download

    out_path = data_download.download_universe_to_single_file()

Requirements:
    - pandas
    - yfinance
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import yfinance as yf

# Import the paths helper (repo-style)
try:
    import core.paths as paths
except ModuleNotFoundError:
    # Fallback if run from inside core/ directly
    import paths  # type: ignore


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataDownloadConfig:
    data_source: str
    tickers: List[str]
    start_date: str
    end_date: Optional[str] = None
    interval: str = "1d"


def _default_config_path() -> Path:
    """
    Return the default path to the data config file.

    By convention we keep it at:
        <PROJECT_ROOT>/config/data_config.json
    """
    return paths.PROJECT_ROOT / "config" / "data_config.json"


def load_data_config(config_path: Optional[Path] = None) -> DataDownloadConfig:
    """
    Load and validate the data download configuration from a JSON file.

    Parameters
    ----------
    config_path : Path, optional
        Path to the JSON config file. If None, use the default location.

    Returns
    -------
    DataDownloadConfig
        Parsed and validated configuration.
    """
    if config_path is None:
        config_path = _default_config_path()
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Create it first (see example in config/data_config.json)."
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    # Basic validation and defaults
    if "tickers" not in raw or not isinstance(raw["tickers"], list):
        raise ValueError("Config must contain key 'tickers' as a list of symbols.")

    data_source = str(raw.get("data_source", "yahoo")).lower()
    start_date = str(raw.get("start_date", "2000-01-01"))
    end_date = raw.get("end_date", None)
    if end_date is not None:
        end_date = str(end_date)

    interval = str(raw.get("interval", "1d"))

    return DataDownloadConfig(
        data_source=data_source,
        tickers=[str(t).upper() for t in raw["tickers"]],
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )


# ---------------------------------------------------------------------------
# Yahoo Finance downloader (per ticker)
# ---------------------------------------------------------------------------

def _download_yahoo_ticker(
    symbol: str,
    start_date: str,
    end_date: Optional[str],
    interval: str,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single symbol from Yahoo Finance.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. 'AAPL').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str or None
        End date in 'YYYY-MM-DD' format, or None for "up to today".
    interval : str
        Bar interval, e.g. '1d', '1h', etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with Yahoo's standard columns:
            ['Open','High','Low','Close','Adj Close','Volume']
        plus a DatetimeIndex.
    """
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Universe → single DataFrame
# ---------------------------------------------------------------------------

def download_universe_to_dataframe(cfg: DataDownloadConfig) -> pd.DataFrame:
    """
    Download all tickers in the config and stack them into a single DataFrame.

    Output schema (where available):
        - 'date'
        - 'symbol'
        - 'open'
        - 'high'
        - 'low'
        - 'close'
        - 'adj_close'
        - 'volume'

    Parameters
    ----------
    cfg : DataDownloadConfig
        Configuration with tickers, dates, etc.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with all tickers stacked together.
    """
    if cfg.data_source != "yahoo":
        raise ValueError(f"Unsupported data source: {cfg.data_source!r}")

    frames = []

    print("=== Downloading universe to DataFrame ===")
    print(f"Source:   {cfg.data_source}")
    print(f"Tickers:  {', '.join(cfg.tickers)}")
    print(f"Start:    {cfg.start_date}")
    print(f"End:      {cfg.end_date or '(today)'}")
    print(f"Interval: {cfg.interval}")
    print("-----------------------------------------")

    for symbol in cfg.tickers:
        try:
            df_sym = _download_yahoo_ticker(
                symbol=symbol,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                interval=cfg.interval,
            )
        except Exception as e:
            print(f"[ERR]  {symbol}: {e}")
            continue

        # Reset index -> 'date' column
        df_sym = df_sym.reset_index()

        # Yahoo's default column names
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }

        # Keep only the columns we recognize + the date
        cols_to_keep = [c for c in df_sym.columns if c in rename_map or c == "date"]
        df_sym = df_sym[cols_to_keep].rename(columns=rename_map)

        # Attach symbol column
        df_sym["symbol"] = symbol

        frames.append(df_sym)
        print(f"[OK]   {symbol}: downloaded {len(df_sym)} rows.")

    if not frames:
        raise RuntimeError("No data downloaded for any ticker. Check your config.")

    all_data = pd.concat(frames, ignore_index=True)

    # Ensure date is datetime
    all_data["date"] = pd.to_datetime(all_data["date"])

    # Reorder columns to a clean schema (only keep what exists)
    ordered_cols = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    final_cols = [c for c in ordered_cols if c in all_data.columns]
    all_data = all_data[final_cols]

    # Sort by symbol then date
    all_data = all_data.sort_values(["symbol", "date"]).reset_index(drop=True)

    return all_data


# ---------------------------------------------------------------------------
# Save universe → single CSV file
# ---------------------------------------------------------------------------

def download_universe_to_single_file(
    config_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    overwrite: bool = True,
) -> Path:
    """
    Download the entire universe and write it to ONE CSV file in data/.

    Parameters
    ----------
    config_path : Path, optional
        Path to the config JSON file. If None, use the default path.
    out_path : Path, optional
        Output CSV path. If None, defaults to:
            data/prices_universe.csv
    overwrite : bool
        If False and the file already exists, skip downloading.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    cfg = load_data_config(config_path=config_path)

    if out_path is None:
        out_path = paths.data_dir() / "prices_universe.csv"
    else:
        out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        print(f"[SKIP] File already exists and overwrite=False: {out_path}")
        return out_path

    all_data = download_universe_to_dataframe(cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(out_path, index=False)

    symbols = sorted(all_data["symbol"].unique())
    date_min = all_data["date"].min()
    date_max = all_data["date"].max()
    print("-----------------------------------------")
    print(
        f"[DONE] Saved {len(all_data)} rows "
        f"for {len(symbols)} symbols to:\n  {out_path}"
    )
    print(f"Symbols:   {', '.join(symbols)}")
    print(f"Date span: {date_min.date()} -> {date_max.date()}")

    return out_path


if __name__ == "__main__":
    # Run from the project root:
    #   python core/data_download.py
    #
    # It will read config/data_config.json and download all tickers listed there
    # into data/prices_universe.csv
    try:
        download_universe_to_single_file()
    except FileNotFoundError as e:
        print(e)
        print(
            "\nCreate config/data_config.json first. Example:\n"
            '{\n'
            '  "data_source": "yahoo",\n'
            '  "tickers": ["SPY", "QQQ", "IWM"],\n'
            '  "start_date": "2015-01-01",\n'
            '  "end_date": null,\n'
            '  "interval": "1d"\n'
            '}\n'
        )

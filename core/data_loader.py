"""
core.data_loader

Helpers for loading OHLCV price data from CSV files in a consistent format.

Repo style:
- We keep raw & cleaned data in the RASA/data/ folder.
- This module reads those CSVs and returns standardized pandas DataFrames.

Typical usage from a script run in the project root:

    from core import data_loader

    prices = data_loader.load_prices_from_folder()
    print(prices.head())

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List, Dict

import pandas as pd


# We assume this file lives at: RASA/core/data_loader.py
# Import paths module for directory helpers.
try:
    # When running from the project root (recommended):
    import core.paths as paths
except ModuleNotFoundError:
    # Fallback: allow running this file directly from within core/ directory.
    import paths  # type: ignore


# Mapping of standardized price columns to common variations
EXPECTED_PRICE_COLS = {
    "open": ["open", "Open", "OPEN"],
    "high": ["high", "High", "HIGH"],
    "low": ["low", "Low", "LOW"],
    "close": ["close", "Close", "CLOSE"],
    "adj_close": ["adj_close", "Adj Close", "adj close", "AdjClose"],
    "volume": ["volume", "Volume", "VOLUME"],
}


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching column name from candidates, or None if not found."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _standardize_price_columns(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Rename columns to a standard schema.

    Standard schema:
        - 'date'
        - 'symbol'
        - 'open'
        - 'high'
        - 'low'
        - 'close'
        - 'adj_close' (optional)
        - 'volume' (optional)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as read from CSV.
    column_map : dict, optional
        Optional explicit mapping from raw column names to standardized names.

        Example:
            {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }

    Returns
    -------
    pd.DataFrame
        Dataframe with standardized column names (where possible).
    """
    df = df.copy()

    # Ensure we have a mapping object
    if column_map is None:
        column_map = {}

    # Date column: try explicit mapping, then common variants
    date_col = column_map.get("date") or _find_column(
        df, ["date", "Date", "DATE", "datetime", "Datetime", "DATETIME"]
    )
    if not date_col:
        raise ValueError(
            "Could not find a date column. "
            "Please provide column_map={'date': 'YourDateColumnName'}"
        )
    # Map the found raw column to 'date'
    column_map[date_col] = "date"

    # Price columns
    for std_name, candidates in EXPECTED_PRICE_COLS.items():
        # If user gave an explicit mapping like column_map['open'] = 'Open'
        if std_name in column_map:
            raw_name = column_map[std_name]
            column_map[raw_name] = std_name
            continue

        # Otherwise try to infer from known variants
        found = _find_column(df, candidates)
        if found:
            column_map[found] = std_name

    # Apply renaming
    df = df.rename(columns=column_map)

    # Keep only relevant columns if present
    keep_cols = [
        c
        for c in ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]
        if c in df.columns
    ]
    df = df[keep_cols]
    return df


def load_single_ticker_csv(
    filepath: Path,
    symbol: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load OHLCV data for a single ticker from a CSV file.

    Assumes the CSV contains at least:
        - a date-like column
        - close prices
    and ideally open/high/low/volume (we'll use them if present).

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.
    symbol : str, optional
        Ticker symbol. If None, we will try to infer it from the filename.
        e.g. 'AAPL.csv' -> 'AAPL'.
    column_map : dict, optional
        Explicit mapping from raw column names to standardized names.
        Example:
            {'Date': 'date', 'Open': 'open', 'Close': 'close'}
    parse_dates : bool
        Whether to parse the 'date' column as datetime.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns at least ['date','symbol','close'], plus others
        if available. Sorted by date.
    """
    filepath = Path(filepath)
    if symbol is None:
        # Infer symbol from filename: e.g. 'AAPL.csv' -> 'AAPL'
        symbol = filepath.stem.upper()

    df = pd.read_csv(filepath)
    df = _standardize_price_columns(df, column_map=column_map)

    if parse_dates:
        df["date"] = pd.to_datetime(df["date"])

    df["symbol"] = symbol
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_prices_from_folder(
    folder: Optional[Path] = None,
    glob_pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load and stack OHLCV data for multiple tickers from a folder.

    This function:
    - Scans the folder for CSV files.
    - For each file, infers the ticker symbol from the filename.
    - Loads it via `load_single_ticker_csv`.
    - Concatenates everything into a single long-format DataFrame.

    Parameters
    ----------
    folder : Path, optional
        Folder containing CSV files. If None, uses paths.data_dir().
    glob_pattern : str
        Glob pattern for CSV files, default '*.csv'.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with at least:
            ['date','symbol','close']
        plus any other standardized columns that were available.
        If no files are found, returns an empty DataFrame.
    """
    if folder is None:
        folder = paths.data_dir()
    else:
        folder = Path(folder)

    files: List[Path] = sorted(folder.glob(glob_pattern))
    if not files:
        # No files: return empty DataFrame with the standard schema
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )

    dfs = []
    for f in files:
        try:
            df = load_single_ticker_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")

    if not dfs:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )

    all_data = pd.concat(dfs, ignore_index=True)

    # Drop rows with missing date/symbol/close
    essential_cols = ["date", "symbol", "close"]
    all_data = all_data.dropna(
        subset=[c for c in essential_cols if c in all_data.columns]
    )

    # Ensure date is datetime
    if "date" in all_data.columns and not pd.api.types.is_datetime64_any_dtype(
        all_data["date"]
    ):
        all_data["date"] = pd.to_datetime(all_data["date"])

    # Sort by symbol, then date
    all_data = all_data.sort_values(["symbol", "date"]).reset_index(drop=True)
    return all_data


if __name__ == "__main__":
    # Example usage when run directly:
    #   python core/data_loader.py
    # It will look into the default data directory and print a brief summary.
    data = load_prices_from_folder()
    if data.empty:
        print("No price data found in data/ (or specified folder).")
        print("Place your CSV files there and re-run this script.")
    else:
        symbols = sorted(data["symbol"].unique())
        date_min = data["date"].min()
        date_max = data["date"].max()
        print(f"Loaded {len(symbols)} symbols from {date_min.date()} to {date_max.date()}.")
        print(f"Symbols: {', '.join(symbols)}")

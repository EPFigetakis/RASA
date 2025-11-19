"""
core.paths

Centralized management of project directories (data, cache, results, logs).

This is **repo-style**, not a Python package:
- You don't install anything with pip.
- You just keep your scripts and modules in `core/` and import them
  when needed (e.g., `from core import paths` if you run from the RASA root).

Design goals:
- Single source of truth for all important paths.
- No hard-coded absolute paths scattered across the repo.
- Automatically create directories on demand.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Core path logic
# ---------------------------------------------------------------------------

# We assume this file lives at: RASA/core/paths.py
# Then:
#   Path(__file__)         -> .../RASA/core/paths.py
#   Path(__file__).parents -> [ .../core, .../RASA, ... ]
# So parents[1] is the project root directory: RASA
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    """
    Return the absolute path to the project root (the RASA folder).

    This is useful if you ever need to build custom paths relative to
    the root in other parts of the code.
    """
    return PROJECT_ROOT


def _ensure_dir(path: Path) -> Path:
    """
    Internal helper: create the directory if it does not exist and return it.

    We keep this as a private function because user code should call the
    public helpers below instead.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Public directory accessors
# ---------------------------------------------------------------------------

def data_dir() -> Path:
    """
    Directory for raw and processed market data.

    Examples:
    - Daily OHLCV prices pulled from a data vendor.
    - Cleaned CSV/parquet files with aligned time series.

    Returns a Path and ensures the directory exists.
    """
    return _ensure_dir(PROJECT_ROOT / "data")


def cache_dir() -> Path:
    """
    Directory for intermediate computations and cached results.

    Examples:
    - Cached cointegration test results.
    - Pickled regime models.
    - Precomputed feature matrices.

    Returns a Path and ensures the directory exists.
    """
    return _ensure_dir(PROJECT_ROOT / "cache")


def results_dir() -> Path:
    """
    Directory for backtest results, performance reports, and plots.

    Examples:
    - Equity curve CSVs.
    - Performance summary tables.
    - Matplotlib/plotly figures.

    Returns a Path and ensures the directory exists.
    """
    return _ensure_dir(PROJECT_ROOT / "results")


def logs_dir() -> Path:
    """
    Directory for log files.

    We will later set up a logging configuration that writes trading,
    backtest, and error logs here.

    Returns a Path and ensures the directory exists.
    """
    return _ensure_dir(PROJECT_ROOT / "logs")


# ---------------------------------------------------------------------------
# Convenience utilities
# ---------------------------------------------------------------------------

def describe_paths() -> str:
    """
    Return a human-readable summary of the key project paths.

    This is handy for debugging and for quickly sanity-checking that the
    project root has been correctly detected.
    """
    lines = [
        f"Project root: {project_root()}",
        f"Data dir:     {data_dir()}",
        f"Cache dir:    {cache_dir()}",
        f"Results dir:  {results_dir()}",
        f"Logs dir:     {logs_dir()}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # If you run this file directly:
    #   python core/paths.py
    # you'll see a summary of where everything lives on your system.
    print(describe_paths())

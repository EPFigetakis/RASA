# RASA — Regime-Aware Statistical Arbitrage

A Streamlit research prototype for pairs-trading / statistical arbitrage: it downloads and
caches price history, screens a ticker universe for cointegrated pairs (Engle-Granger),
lets you inspect a single pair (spread, z-score, simplified backtest), and generates
experimental trade proposals from the current signals.

This is a research/portfolio project, not a production trading system — see
`RASA_Project_Handover.docx` for the full design history, known limitations, and roadmap.

## Requirements

- Python 3.13 (a `.venv` in the repo root was built against this)
- Dependencies pinned in `requirements.txt`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

- `config/data_config.json` — the ticker universe and history window (`data_source`,
  `tickers`, `start_date`, `end_date`, `interval`). Edit this to change what gets downloaded
  and screened.
- `config/app_meta.json` — the version string and change log shown in the app's sidebar.
  Bump `version` and append to `change_notes` when you ship a change (this is also how you
  confirm the deployed instance is running the version you expect).

Downloaded prices are cached under `cache/prices/` as Parquet files keyed by a hash of the
ticker universe + date range + interval, with a JSON sidecar recording the tickers, settings,
and last bar date — this is what `core/prices.py:build_cache_key` produces. Candidate pairs
are cached under `cache/pairs/`, and per-fold walk-forward candidate scans under
`cache/pairs/walkforward/`. Everything under `cache/` is regenerable and gitignored.

## Running the app

```bash
streamlit run app.py
```

### Authentication

`app.py` is gated behind a login form backed by a local SQLite database (`core/auth.db`,
gitignored — bcrypt password hashes, no plaintext). There's no self-service signup screen;
create a user directly:

```bash
python3 - <<'EOF'
import sqlite3, bcrypt

conn = sqlite3.connect("core/auth.db")
conn.execute(
    "CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)"
)
username, password = "your-username", "your-password"
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
conn.execute(
    "INSERT OR REPLACE INTO users (username, password_hash) VALUES (?, ?)",
    (username, password_hash),
)
conn.commit()
EOF
```

For local development on a machine that isn't linked to a seeded `auth.db` (e.g. testing on
a separate dev box), you can bypass login entirely:

```bash
RASA_DEV_SKIP_AUTH=1 streamlit run app.py
```

This is a local-dev-only escape hatch — never set it in the deployed environment.

### Walk-forward validation

The **Pair selection** and **Pair inspection** views select pairs and estimate hedge ratios
from the *entire* price history at once — quick to use, but the cointegration p-value and
hedge ratio for a pair traded in, say, 2018 are partly informed by data from 2024. That's
look-ahead bias, and it makes those results non-defensible as research.

The **Walk-forward validation** view (`core/walkforward.py`) fixes this: it splits history
into a rolling (or expanding) sequence of formation windows and trading windows, selecting
pairs and freezing hedge ratios only from each formation window, then trading and scoring
strictly on the following, previously-unseen trading window. It also applies a
Benjamini-Hochberg false-discovery-rate correction on top of the raw cointegration p-values
(scanning ~185 tickers is ~17,000 pairwise tests per fold — some will look cointegrated by
chance) and requires a pair to clear that bar in multiple consecutive folds before it's
trusted enough to trade. Use this view, not the static Pair selection view, when you want a
result you'd actually stand behind.

Each fold's formation-window candidate scan is cached to disk under
`cache/pairs/walkforward/`, since rescanning the full universe once per fold is CPU-heavy —
re-running the same window configuration is instant after the first pass.

**Execution costs** (`core/execution.py`) are priced per leg and on by default in this view:
commission ($/share), half bid/ask spread and slippage (flat bps of traded notional), short
borrow (annual bps on whichever leg is short), and an optional fill delay in days. The view
shows gross-vs-net equity and a cost breakdown. Spread/slippage are assumptions, not observed
quotes (the price panel is adjusted closes only), and positions still open at a fold's end are
not liquidated. The other views still use the flat per-unit `trade_cost_per_unit` model.

## Cache freshness & provenance

- **Price cache:** each `cache/prices/<key>.parquet` has a `<key>.json` sidecar with the
  universe, settings, `last_bar_date`, `updated_at` and a content `panel_fingerprint`. The
  sidebar shows the last bar and warns when it is more than 2 business days behind
  (holidays aren't modelled, hence the tolerance). Caches with a fixed `end_date` are never stale.
- **Update prices** (sidebar) appends only the missing bars. It re-downloads a short overlap
  and compares it with the cache; if adjusted history was restated (dividend/split) it falls
  back to a full re-download rather than splicing inconsistent prices. If the download fails,
  the existing cache keeps being served. In code: `load_prices(..., update_if_stale=True)`.
- **Saved pair selection:** `cache/pairs/pair_candidates.json` records the universe, price
  fingerprint and selection settings. "Load saved pair selection" refuses a different
  universe/settings, and warns if the prices changed since the scan. Selections saved before
  this feature have no record and load with a warning.

## Tests

```bash
pytest
```

Tests run fully offline (no real network/yfinance calls, no dependence on a seeded
`auth.db`) via mocking and the `RUNNING_TESTS` env var, which `app.py` also checks to skip
the login gate during test collection.

## Deployment

Pushing to `main` triggers `.github/workflows/ci_cd.yml`: it runs the test suite, then (on a
self-hosted runner) pulls and restarts the live `streamlit.service` instance. Treat a push to
`main` as a deploy, not just a commit.

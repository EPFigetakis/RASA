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

## Trade proposals

`core/trade_proposals.py` turns the latest pair signals into typed `TradeIdea` objects
(no Streamlit dependency): `generate_trade_ideas(px, pairs_df, ProposalConfig(...))` returns a
`ProposalBatch` with the ideas, every skipped pair and its reason, the price data's last bar and
age, and stable `idea_id` / `batch_id` hashes. An idea is a **target position**, not an order:
`bars_in_position` / `is_new_signal` distinguish a fresh entry from a long-held signal, but
account positions don't exist yet, so nothing is netted against holdings. Sizing treats raw-price
hedges as `beta` shares of X per share of Y, and log-price hedges as a dollar elasticity.

## Orders

`core/orders.py` is the contract between trade ideas and (future) execution. `plan_orders(ideas,
current_positions, managed_symbols=..., close_out_prices=...)` compares each idea's *target*
positions with what is held and returns an `OrderPlan` of `OrderTicket`s in `PROPOSED` state, so
re-running against the same holdings never doubles a position. Tickets have deterministic ids
(same idea + legs + data date -> same id), a validated lifecycle (`PROPOSED -> APPROVED ->
SUBMITTED -> PARTIALLY_FILLED/FILLED`, or `REJECTED`/`CANCELLED`) with a history of changes,
fill tracking with overfill protection, and an exact JSON round-trip. Ideas that share a symbol
are reported as conflicts rather than netted, and positions outside `managed_symbols` are never
closed. Nothing in the UI uses this yet; risk gates and the paper broker come next.

## Risk gates

`core/risk.py` checks every proposed ticket against `config/trading_config.json` (capital base,
gross/net exposure, max concurrent pairs, per-symbol and per-pair caps, minimum price, max data
age, allowed order types, whole-share rule, daily-loss / drawdown kill switch, manual-approval
flag). `evaluate_plan(plan, cfg, prices, account, data_age_business_days)` returns a `RiskReport`
with a pass/fail per check per ticket and a reason for every rejection; unknown config keys are
an error. Tickets are tested in plan order against a running projected portfolio, so a rejected
ticket doesn't use up capacity. Tickets that only reduce risk (close-outs, shrinking a position)
skip the exposure/price/staleness/kill-switch checks so the system can always get out. Ideas
that overlap on a symbol are rejected, never netted. `apply_decisions` rejects failed tickets
and, unless `require_manual_approval` is turned off, leaves passing ones PROPOSED for a human.
Not covered yet: sector/correlation limits, borrow availability, real liquidity. The
`environment` field is recorded in reports but nothing enforces paper-vs-live separation yet.

## Portfolio ledger & simulator

`core/portfolio.py` is leg-level accounting from actual share counts and fill prices: cash,
average-cost positions, realized/unrealized PnL per symbol, fees and short borrow tracked
separately, with the identity `equity - initial_cash = realized + unrealized - fees - borrow`
(tested). `Portfolio.apply_ticket_fills` books an `OrderTicket`'s fills, and
`Portfolio.quantities()` feeds straight back into `orders.plan_orders`, so a filled ticket
produces no further orders. `simulate_targets` replays a table of target share positions over a
price panel (trade at the close, execution costs, fill delay, daily borrow); in raw-price space
with fixed shares and zero costs it reproduces `strategy.backtest_spread_strategy`'s equity
curve to the cent, and with costs it matches the `core/execution.py` model (both tested).
`reconcile(expected, actual)` lists position breaks. Not modelled: margin/buying power,
interest on cash, dividends on shorts, corporate actions. The Streamlit views still use the
older scalar backtest.

## Paper broker

`core/broker_base.py` defines the `Broker` interface; `core/broker_paper.py` implements it as a
simulated account kept in one SQLite file (default location `state/`, gitignored, **not**
regenerable - it is the account history). Submit `APPROVED` tickets with `submit_ticket`
(idempotent on `ticket_id`: resubmitting never creates a second order), then advance time with
`process_bar(prices, as_of)`: working tickets fill against that bar (market at the bar price;
limit only if marketable; optional partial fills), unfilled DAY legs expire, short borrow is
charged once per new bar, and an equity point is recorded. Each operation is one transaction,
so a crash or error leaves either all of it or none. Every submission, fill, expiry, borrow
charge and reconciliation goes to an append-only `events` table. The account config is stored
in the DB and reopening with a different one is refused. `reconcile(expected)` lists position
breaks. No margin check (cash can go negative; the risk layer limits size), and an expired
DAY leg can leave a pair one-legged - visible in the ticket history. Nothing in the UI uses the
broker yet; the daily runner (Phase 4) will drive it.

## Daily runner

`python scripts/run_daily_proposals.py run` is the non-UI pipeline (paper only): load prices
(appending new bars unless `--no-update`) -> **freshness gate** (older than
`max_data_age_business_days` aborts before touching the broker) -> advance the paper broker to
the latest bar (earlier approved tickets fill) -> trade ideas from the *saved* pair selection ->
order plan against the broker's holdings (held pairs that went flat get close-out tickets) ->
risk gates -> a run record in `state/runs/R<id>.json`. It never rescans the universe (run pair
selection in the app) and never submits orders: passing tickets stay `PROPOSED`. Release them
with `python scripts/run_daily_proposals.py approve RUN_ID --all --by <you>` (or `--ticket ID`);
that is refused if the broker's holdings or last bar changed since the run, and rejected tickets
can never be approved. `list` shows recent runs; `run --dry-run` computes and prints without
writing. Re-running a day with the same inputs gives the same run id and tickets, and tickets
already with the broker aren't proposed again. A lock file stops two runs overlapping.
Exit codes: `0` ok, `1` unexpected error, `2` precondition failed or data too stale, `3` kill
switch tripped; logs go to `logs/daily_runner.log`. The runner refuses a `live` risk
environment. Scheduling (cron/systemd timer) is deliberately left to you until you trust it;
`state/` is gitignored and holds real account history, so back it up.

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

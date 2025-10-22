from pathlib import Path

# ---------- data ----------
DATA_DIR = Path("data_cache")
START_DATE = "2016-01-01"
END_DATE = None  # None => today

# ---------- universe ----------
UNIVERSE = sorted(list(set(
    # Tech / Comm
    ["AAPL","MSFT","NVDA","GOOGL","GOOG","META","AMZN","AVGO","ORCL","CRM","AMD","INTC"]
  + ["JPM","BAC","WFC","GS","MS","BLK","SCHW","C"]             # Financials
  + ["LLY","UNH","JNJ","PFE","MRK","ABBV"]                    # Healthcare
  + ["CAT","GE","HON","RTX","LMT","UPS","DE"]                 # Industrials
  + ["HD","MCD","NKE","PG","KO","PEP","COST"]                 # Consumer
  + ["XOM","CVX","COP","SLB","LIN"]                           # Energy / Materials
)))

# ---------- pair selection ----------
WINDOWS = (504, 252, 126)     # ~2y, 1y, 6m p-value stability windows
MIN_OVERLAP_DAYS = 252 * 2    # at least ~2y of clean overlap
MAX_PAIRS = 80

# Half-life measured on a RECENT window, not the whole history
HL_LOOKBACK_DAYS = 252        # measure half-life on last 1y of spread
MIN_HALF_LIFE_D = 1
MAX_HALF_LIFE_D = 250         # raised from 60–120 to be realistic

# Optional median p-value gate (keep None to disable)
PVAL_THRESHOLD = None         # e.g., set 0.25 for stricter gating

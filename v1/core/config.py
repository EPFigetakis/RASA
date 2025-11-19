from pathlib import Path

# ---------- data ----------
DATA_DIR = Path("data_cache")
START_DATE = "2016-01-01"
END_DATE = None  # None => today

# ---------- universe ----------
# ---------- universe (updated from user lists; Yahoo/yfinance-friendly) ----------
UNIVERSE = sorted(list(set(
    # Railroads (BNSF private; KSU delisted; use CP)
    ["UNP","CSX","CNI","CP","NSC","RAIL"]

  + # Transportation / Airlines
    ["UPS","FDX","XPO","JBHT","ODFL","ALK","DAL","SKYW","AAL","UAL","UBER","LYFT","LUV"]

  + # Aerospace / Defense
    ["LMT","RTX","BA","NOC","LHX","GE","HII","TDG","AVAV","KTOS","AXON","GD","HWM","BWXT"]

  + # Electric equipment / Power mgmt + Utilities
    ["ABB","ETN","GEV","HUBB","NVT","POWL","SBGSF","SIEGY","VRT",
     "AEP","CEG","DUK","NEE","SO","VST","EXC","ED","SRE","D","XEL","JCI","PWR"]

  + # Industrials
    ["CAT","EMR","HON","DE","WM","CTAS","IR","DOV","ETN","AME","ROK","BAESY","RYCEY","PH","MMM","ITW","CMI","CARR","FERG","GWW"]

  + # Credit Services (drop JCAP & KLAR—no current listed tickers)
    ["V","MA","AXP","COF","PYPL","SYF","SOFI","UPST","WU","EZPW","QFIN"]

  + # Financial Services / Asset Management (use BRK-B)
    ["BRK-B","JPM","BAC","WFC","MS","GS","HSBC","BX","BLK","SCHW","TD","PGR","HOOD","UBS","CB","KKR","BAM","MCO","APO","BK","USB","AFL"]

  + # Consumer Staples
    ["KO","PEP","GIS","MDLZ","NSRGY","KDP","PG","CLX","UL","KMB","CL","EL","WMT","COST","KR","TGT","PM","MO","DEO"]

  + # Drug / Pharma / Biotech
    ["ABBV","PFE","JNJ","LLY","AMGN","GILD","NVS","AZN","VRTX","REGN","ALNY","INSM","RPRX","IONS","MDGL","MRNA","CRSP"]

  + # Technology (TSMC→TSM; AAPL fixed; remove 'LAM' duplicate of LRCX)
    ["KLAC","ASML","AMAT","LRCX","FIS","CTSH","TEAM","ARM","APP","SHOP","SPOT","TSM","ADI","ANET","TER","ENTG","NVMI",
     "NOW","CRM","CRWD","PANW","QCOM","WDC","TTD","INTU","ONTO","INTC","TSLA","GOOGL","GOOG","AVGO","CSCO","AMZN","MSFT",
     "AMD","IBM","AAPL","PLTR","NVDA","META","MU"]
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

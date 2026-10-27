"""Project-wide settings. Every tunable choice lives here so the README can point at one file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"            # raw caches (git-ignored)
RAW_EDGAR = DATA / "edgar"
RESULTS = ROOT / "results"      # small CSV/JSON/Markdown outputs (committed)
FIGURES = ROOT / "figures"      # PNG charts (committed)

USER_AGENT = "Teymur Eyvazov teymur.eyvazov001@umb.edu"
SEC_MAX_RPS = 8                 # SEC fair-access limit is 10 req/s; stay below it

# Fixed large-cap universe: long-listed S&P 500 members across sectors.
# Chosen as of 2026 -> survivorship bias (documented in README).
TICKERS = [
    # tech / communication
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "INTC", "CSCO", "ORCL", "IBM",
    "ADBE", "CRM", "QCOM", "TXN", "AMD", "AVGO", "NFLX", "DIS", "CMCSA", "VZ", "T",
    # financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW", "USB", "PNC",
    # health care
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH", "CVS", "MDT", "ABT", "TMO", "DHR",
    "BMY", "AMGN", "GILD",
    # energy / materials
    "XOM", "CVX", "COP", "SLB", "LIN", "NEM",
    # staples / discretionary
    "KO", "PEP", "PG", "WMT", "COST", "HD", "LOW", "MCD", "NKE", "SBUX", "TGT",
    "F", "GM", "MO",
    # industrials
    "BA", "CAT", "DE", "HON", "GE", "MMM", "UPS", "UNP", "LMT", "FDX",
]
MARKET = "SPY"

SAMPLE_START = "2018-01-01"
SAMPLE_END = "2026-07-15"       # last event date: leaves room for a +60 trading-day window
PRICE_START = "2016-09-01"      # covers the [-250,-30] estimation window of 2018 events

# Event-study windows (trading days relative to day 0)
EST_WINDOW = (-250, -30)
MIN_EST_OBS = 120
CAR_WINDOWS = {"car_0_1": (0, 1), "car_2_20": (2, 20), "car_2_60": (2, 60)}
PATH_RANGE = (-5, 60)

# Text processing
MIN_SENT_WORDS = 6
MAX_SENT_WORDS = 80
MAX_SENTS_PER_DOC = 120         # cap FinBERT work per document (narrative comes first)
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_BATCH = 32
FINBERT_MAX_LEN = 128

# Trading test
TRAIN_END = "2021-12-31"        # signal selection uses events up to here only
COST_BPS_PER_SIDE = 10          # one-way cost per trade leg, in basis points

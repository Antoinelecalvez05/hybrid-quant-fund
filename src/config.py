"""
config.py
---------
Central configuration for the hybrid-quant-fund research prototype.

V3 additions:
  - NLP_SOURCE_DIR        : folder for curated local text sources
  - NLP_MODE              : "off" | "heuristic" | "llm"
  - USE_LLM_EXTRACTION    : set True only if API key is present
  - NLP_SIGNAL_DECAY_DAYS : exponential decay half-life for NLP signal forward-fill
  - NLP_WEIGHT            : blend weight for NLP-derived signal component
  - V2_PROXY_WEIGHT       : blend weight for V2 price-proxy component
"""

# ---------------------------------------------------------------------------
# ETF universe
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "TLT", "IEF", "GLD", "EFA"]

# ---------------------------------------------------------------------------
# Temporal split  (strict — no look-ahead across boundaries)
# ---------------------------------------------------------------------------
TRAIN_START  = "2010-01-01"
TRAIN_END    = "2018-12-31"

VAL_START    = "2019-01-01"
VAL_END      = "2020-12-31"

TEST_START   = "2021-01-01"
TEST_END     = "2023-12-31"

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
MOMENTUM_WINDOWS   = [5, 10, 21, 63]
VOLATILITY_WINDOW  = 21
MA_SHORT           = 10
MA_LONG            = 63
DRAWDOWN_WINDOW    = 63

# ---------------------------------------------------------------------------
# Contextual signal layer parameters (V2)
# ---------------------------------------------------------------------------
CONTEXT_MOMENTUM_WINDOW  = 63
CONTEXT_VOL_WINDOW       = 21
CONTEXT_ZSCORE_WINDOW    = 126

# ---------------------------------------------------------------------------
# V3 NLP / LLM signal layer
# ---------------------------------------------------------------------------
NLP_SOURCE_DIR       = "data/text_sources"   # folder of curated local source files
NLP_MODE             = "heuristic"           # "off" | "heuristic" | "llm"
USE_LLM_EXTRACTION   = False                 # True only if LLM API key is configured

# Signal forward-fill and blending
NLP_SIGNAL_DECAY_DAYS = 21     # exponential decay: signal half-life after source date
NLP_WEIGHT            = 0.5    # weight of NLP signal in the blended ai_* columns
V2_PROXY_WEIGHT       = 0.5    # weight of V2 price-proxy in the blended ai_* columns

# ---------------------------------------------------------------------------
# Random Forest — fixed hyperparameters (no post-validation tuning)
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 5
RF_MIN_SAMPLES_LEAF = 20
RF_RANDOM_STATE     = 42
RF_CLASS_WEIGHT     = "balanced_subsample"

# ---------------------------------------------------------------------------
# Signal & position sizing
# ---------------------------------------------------------------------------
SIGNAL_THRESHOLD = 0.55
POSITION_SIZE    = 1.0

# ---------------------------------------------------------------------------
# Transaction costs (V2)
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS = 5

# ---------------------------------------------------------------------------
# Risk / reporting
# ---------------------------------------------------------------------------
RISK_FREE_RATE   = 0.02
TRADING_DAYS     = 252

# ---------------------------------------------------------------------------
# Data cache
# ---------------------------------------------------------------------------
DATA_DIR    = "data"
PRICES_FILE = "prices.csv"

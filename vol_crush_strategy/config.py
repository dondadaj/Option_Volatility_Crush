"""
config.py — Central configuration for the Option Volatility Crush strategy.

Contains API keys, sector-to-ETF mappings, feature definitions,
and tunable parameters used across all pipeline modules.
"""

import os

# =============================================================================
# API Keys — set via environment variables or hardcode for development
# =============================================================================
# The .env file uses ALPHA_VANTAGE_API (matching the pilot notebook pattern).
# We check that first, then ALPHA_VANTAGE_API_KEY as a common alternative.
ALPHA_VANTAGE_API_KEY = (
    os.environ.get('ALPHA_VANTAGE_API')
    or os.environ.get('ALPHA_VANTAGE_API_KEY')
    or os.environ.get('AV_API_KEY')
    or 'YOUR_KEY_HERE'
)
FMP_API_KEY = os.environ.get('FMP_API_KEY', 'YOUR_KEY_HERE')

# =============================================================================
# Data Paths
# =============================================================================
# Repo root: Option_Volatility_Crush/
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(_REPO_ROOT, 'data')
VOL_CRUSH_DIR = os.path.join(DATA_DIR, 'vol_crush')
VIX_CACHE_DIR = os.path.join(VOL_CRUSH_DIR, 'vix_cache')
SECTOR_IV_CACHE_DIR = os.path.join(VOL_CRUSH_DIR, 'sector_iv_cache')
ML_READY_DIR = os.path.join(DATA_DIR, 'ml_ready')

# Pilot data (NVDA single-stock proof-of-concept)
PILOT_DIR = os.path.join(_REPO_ROOT, 'notebooks')
PILOT_DATA_DIR = os.path.join(PILOT_DIR, 'pilot_data')

# =============================================================================
# Pilot Data Column Names
# =============================================================================
# The pilot notebook (vol_crush_pilot.ipynb) uses these column names.
# They differ from some generic defaults — always use these when working
# with pilot data CSVs.
#
# IV columns are ANNUALIZED DECIMALS (e.g. 1.02 = 102% IV).
# Straddle columns are PERCENTAGE POINTS (e.g. 7.4 = 7.4% of stock price).
#   This matches vol_crush_utils.compute_straddle_metrics() which does * 100.
PILOT_STOCK_IV_COL = 'iv_avg_pre'            # from 04_iv_straddle_metrics.csv
PILOT_STOCK_STRADDLE_COL = 'straddle_pct_pre'  # from 04_iv_straddle_metrics.csv
PILOT_DATE_COL = 'announcement_date'

# =============================================================================
# Sector ETF Mapping — GICS sectors to liquid sector ETFs
# =============================================================================
# These ETFs have deep, liquid option markets suitable for IV extraction.
# Each maps a GICS sector label (as it appears in your earnings dataset)
# to the most representative sector ETF ticker.

SECTOR_ETF_MAP = {
    'Technology':              'XLK',
    'Information Technology':  'XLK',
    'Communication Services':  'XLC',
    'Consumer Discretionary':  'XLY',
    'Consumer Staples':        'XLP',
    'Energy':                  'XLE',
    'Financials':              'XLF',
    'Health Care':             'XLV',
    'Healthcare':              'XLV',
    'Industrials':             'XLI',
    'Materials':               'XLB',
    'Real Estate':             'XLRE',
    'Utilities':               'XLU',
}

# Sub-sector ETFs for more granular IV context.
# Use these when you can identify the sub-sector from company profile data.
# Falls back to SECTOR_ETF_MAP if sub-sector isn't matched.
SUBSECTOR_ETF_MAP = {
    'Semiconductors':          'SMH',
    'Semiconductor':           'SMH',
    'Biotech':                 'XBI',
    'Biotechnology':           'XBI',
    'Retail':                  'XRT',
    'Homebuilders':            'XHB',
    'Banks':                   'KBE',
    'Regional Banks':          'KRE',
    'Oil & Gas E&P':           'XOP',
    'Oil & Gas':               'XOP',
    'Software':                'IGV',
    'Internet':                'FDN',
}

# Combined lookup: check sub-sector first, then sector
def get_sector_etf(sector=None, sub_sector=None):
    """
    Resolve a sector/sub-sector label to the best sector ETF ticker.

    Parameters:
        sector: GICS sector string (e.g., 'Technology')
        sub_sector: optional sub-sector string (e.g., 'Semiconductors')

    Returns:
        ETF ticker string, or None if no match found
    """
    if sub_sector and sub_sector in SUBSECTOR_ETF_MAP:
        return SUBSECTOR_ETF_MAP[sub_sector]
    if sector and sector in SECTOR_ETF_MAP:
        return SECTOR_ETF_MAP[sector]
    return None


# =============================================================================
# VIX Configuration
# =============================================================================
VIX_TICKER = '^VIX'
VIX3M_TICKER = '^VIX3M'
VIX_LOOKBACK_START = '2016-01-01'  # Extra year before 2017 for rolling calcs

# Percentile lookback windows (in calendar days, converted to trading days
# internally when computing)
VIX_PERCENTILE_WINDOW_DAYS = 365      # ~252 trading days
VIX_SHORT_MOMENTUM_DAYS = 5           # 5 trading days
VIX_MEDIUM_MOMENTUM_DAYS = 21         # ~1 month of trading days

# Term structure regime threshold
VIX_BACKWARDATION_THRESHOLD = 1.0     # VIX/VIX3M > 1.0 = backwardation


# =============================================================================
# Sector IV Configuration
# =============================================================================
# Option chain filtering: when looking for ATM options on sector ETFs,
# only consider expirations within this window (in calendar days).
# Using 2 (not 7) as the minimum because sector ETF weeklies often have
# 2-6 DTE around earnings — matching vol_crush_utils.extract_atm_options().
SECTOR_IV_MIN_DTE = 2     # at least 2 days to expiry
SECTOR_IV_MAX_DTE = 45    # at most 45 days to expiry

# UNIT CONVENTION for straddle_pct:
#   The pilot's vol_crush_utils.compute_straddle_metrics() outputs
#   straddle_pct in PERCENTAGE POINTS: (straddle / stock_price) * 100.
#   e.g. straddle_pct_pre = 7.4 means the straddle costs 7.4% of the stock.
#   Our compute_sector_etf_iv() must match this convention so that
#   straddle_stock_minus_sector produces meaningful differences.
STRADDLE_PCT_MULTIPLIER = 100  # multiply raw ratio by this

# Default risk-free rate for Black-Scholes IV inversion.
# In production, you'd pull this from Treasury yields per event date.
DEFAULT_RISK_FREE_RATE = 0.05

# Alpha Vantage rate limiting
AV_REQUEST_DELAY_SECONDS = 0.5   # delay between API calls
AV_FREE_TIER_DAILY_LIMIT = 25
AV_PREMIUM_CALLS_PER_MINUTE = 75


# =============================================================================
# Relative IV Feature Thresholds
# =============================================================================
# Stock IV / Sector IV ratio above this = "elevated vs sector" flag.
# 1.5x is a strong signal threshold; 1.2x would catch more moderately
# elevated names but adds noise. Tunable per strategy needs.
IV_ELEVATED_VS_SECTOR_THRESHOLD = 1.5


# =============================================================================
# Feature Names — used for validation, SHAP analysis, and column selection
# =============================================================================
VIX_FEATURE_NAMES = [
    'vix_level',
    'vix3m_level',
    'vix_term_structure_ratio',
    'vix_term_structure_regime',
    'vix_percentile_252d',
    'vix_change_5d',
    'vix_change_21d',
]

SECTOR_IV_RAW_FEATURE_NAMES = [
    'sector_etf',
    'sector_atm_iv_call',
    'sector_atm_iv_put',
    'sector_atm_iv_avg',
    'sector_straddle_pct',
]

SECTOR_IV_RELATIVE_FEATURE_NAMES = [
    'iv_stock_minus_sector',
    'iv_stock_sector_ratio',
    'straddle_stock_minus_sector',
    'iv_elevated_vs_sector',
]

ALL_MARKET_VOL_FEATURE_NAMES = (
    VIX_FEATURE_NAMES
    + SECTOR_IV_RAW_FEATURE_NAMES
    + SECTOR_IV_RELATIVE_FEATURE_NAMES
)

# =============================================================================
# Validation Ranges — expected bounds for sanity checks
# =============================================================================
FEATURE_VALIDATION_RANGES = {
    'vix_level':                (5, 90),
    'vix3m_level':              (5, 70),
    'vix_term_structure_ratio': (0.5, 2.5),
    'vix_percentile_252d':      (0.0, 1.0),
    'vix_change_5d':            (-0.8, 3.0),
    'vix_change_21d':           (-0.8, 5.0),
    'sector_atm_iv_avg':        (0.01, 2.0),
    'iv_stock_sector_ratio':    (0.1, 50.0),
}

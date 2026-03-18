"""
vol_crush_strategy — ML-Powered Option Volatility Crush Strategy

Modules:
    config: Central configuration (API keys, sector ETF mappings, thresholds)
    market_vol_context: Market-wide volatility context features (VIX, sector IV)
"""

from .config import (
    ALPHA_VANTAGE_API_KEY,
    SECTOR_ETF_MAP,
    SUBSECTOR_ETF_MAP,
    get_sector_etf,
)

from .market_vol_context import (
    add_market_wide_vol_features,
    fetch_vix_data,
    compute_vix_features,
    add_sector_iv_features,
    compute_relative_iv_features,
    handle_market_vol_missing_data,
    validate_market_vol_features,
    implied_vol,
    get_sector_iv_for_event,
)

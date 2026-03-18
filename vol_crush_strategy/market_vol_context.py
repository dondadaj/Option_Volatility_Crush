"""
market_vol_context.py — Market-Wide Volatility Context Features

This module adds regime-aware, market-level volatility features to the
Option Volatility Crush strategy pipeline. It provides two categories
of features:

1. VIX-Based Features (7 features)
   - Raw VIX and VIX3M levels
   - Term structure ratio and regime detection (contango vs backwardation)
   - VIX percentile rank over trailing year
   - Short-term and medium-term VIX momentum

2. Sector-Level IV Features (8 features)
   - ATM implied volatility of sector ETFs (call, put, average)
   - Sector straddle pricing as % of ETF price
   - Stock-vs-sector relative IV metrics (spread, ratio, binary flag)

All features use only data available before each earnings event.
No forward-looking data leakage.

Usage:
    from vol_crush_strategy.market_vol_context import add_market_wide_vol_features
    df = add_market_wide_vol_features(earnings_df, api_key='YOUR_KEY')
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import brentq
from scipy.stats import norm

from .config import (
    VIX_TICKER, VIX3M_TICKER, VIX_LOOKBACK_START,
    VIX_PERCENTILE_WINDOW_DAYS, VIX_SHORT_MOMENTUM_DAYS,
    VIX_MEDIUM_MOMENTUM_DAYS, VIX_BACKWARDATION_THRESHOLD,
    SECTOR_ETF_MAP, SUBSECTOR_ETF_MAP,
    SECTOR_IV_MIN_DTE, SECTOR_IV_MAX_DTE,
    DEFAULT_RISK_FREE_RATE, AV_REQUEST_DELAY_SECONDS,
    IV_ELEVATED_VS_SECTOR_THRESHOLD,
    VIX_CACHE_DIR, SECTOR_IV_CACHE_DIR,
    VIX_FEATURE_NAMES, FEATURE_VALIDATION_RANGES,
    STRADDLE_PCT_MULTIPLIER,
    PILOT_STOCK_IV_COL, PILOT_STOCK_STRADDLE_COL,
    get_sector_etf,
)


# =============================================================================
# PART A: VIX DATA FETCHING
# =============================================================================

def fetch_vix_data(start_date=None, end_date=None, cache=True):
    """
    Fetch VIX and VIX3M daily close prices from Yahoo Finance.

    Uses file-based caching to avoid re-downloading on every run.
    VIX data is free and doesn't consume any Alpha Vantage API quota.

    Parameters:
        start_date: str 'YYYY-MM-DD', defaults to VIX_LOOKBACK_START
        end_date: str 'YYYY-MM-DD', defaults to today
        cache: bool, if True, saves/loads from disk cache

    Returns:
        pd.DataFrame indexed by date with columns:
            - vix_close (float): VIX daily close
            - vix3m_close (float): VIX3M daily close
    """
    start_date = start_date or VIX_LOOKBACK_START
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')

    # Check cache
    if cache:
        os.makedirs(VIX_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(VIX_CACHE_DIR, 'vix_daily.csv')
        if os.path.exists(cache_file):
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # If cache covers our requested range, use it
            if (cached.index.min() <= pd.Timestamp(start_date) and
                    cached.index.max() >= pd.Timestamp(end_date) - pd.Timedelta(days=5)):
                print(f"  Loaded VIX data from cache ({len(cached)} rows)")
                return cached

    print(f"  Downloading VIX data from Yahoo Finance ({start_date} to {end_date})...")

    # Download VIX
    vix_raw = yf.download(
        VIX_TICKER, start=start_date, end=end_date, progress=False
    )
    # Download VIX3M
    vix3m_raw = yf.download(
        VIX3M_TICKER, start=start_date, end=end_date, progress=False
    )

    if vix_raw.empty:
        raise ValueError(f"Failed to download {VIX_TICKER} data from Yahoo Finance")

    # Handle both single-ticker and multi-ticker yfinance column formats.
    # yfinance may return either a simple Index or a MultiIndex depending
    # on version and how many tickers were requested.
    def _extract_close(df, ticker_label):
        if df.empty:
            return pd.Series(dtype=float)
        cols = df.columns
        if isinstance(cols, pd.MultiIndex):
            if 'Close' in cols.get_level_values(0):
                return df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
        if 'Close' in cols:
            s = df['Close']
            return s.iloc[:, 0] if hasattr(s, 'iloc') and s.ndim > 1 else s
        # Fallback: first column
        return df.iloc[:, 0]

    vix_close = _extract_close(vix_raw, VIX_TICKER).rename('vix_close')
    vix3m_close = _extract_close(vix3m_raw, VIX3M_TICKER).rename('vix3m_close')

    # Merge on date index
    combined = pd.DataFrame(vix_close).join(pd.DataFrame(vix3m_close), how='outer')

    # Forward-fill gaps (VIX3M occasionally has missing days)
    combined = combined.ffill()

    # Drop any rows where VIX itself is NaN (shouldn't happen but safety)
    combined = combined.dropna(subset=['vix_close'])

    # Ensure numeric
    combined['vix_close'] = pd.to_numeric(combined['vix_close'], errors='coerce')
    combined['vix3m_close'] = pd.to_numeric(combined['vix3m_close'], errors='coerce')

    print(f"  Downloaded {len(combined)} rows of VIX data "
          f"({combined.index.min().date()} to {combined.index.max().date()})")

    # Save to cache
    if cache:
        combined.to_csv(cache_file)
        print(f"  Cached VIX data to {cache_file}")

    return combined


# =============================================================================
# PART B: VIX FEATURE COMPUTATION
# =============================================================================

def compute_vix_features(earnings_df, vix_df, date_col='announcement_date'):
    """
    For each earnings event, compute 7 VIX-based market regime features.

    All features use data from the event date or earlier — no leakage.

    Parameters:
        earnings_df: DataFrame containing earnings events.
                     Must have a date column (datetime or string).
        vix_df: DataFrame from fetch_vix_data() with vix_close, vix3m_close.
        date_col: name of the date column in earnings_df.

    Returns:
        earnings_df with 7 new VIX feature columns appended.
    """
    print(f"  Computing VIX features for {len(earnings_df)} events...")

    # Pre-sort VIX data for efficient lookups
    vix_sorted = vix_df.sort_index()

    # Pre-compute results as lists (faster than row-by-row DataFrame ops)
    results = {name: [] for name in VIX_FEATURE_NAMES}

    for idx, row in earnings_df.iterrows():
        event_date = pd.Timestamp(row[date_col])

        # Get all VIX data up to and including event date
        mask = vix_sorted.index <= event_date
        available = vix_sorted.loc[mask]

        if len(available) == 0:
            for name in VIX_FEATURE_NAMES:
                results[name].append(np.nan)
            continue

        # Most recent VIX data point
        latest = available.iloc[-1]
        latest_date = available.index[-1]
        vix_level = float(latest['vix_close'])
        vix3m_level = float(latest['vix3m_close']) if pd.notna(latest['vix3m_close']) else np.nan

        # ---- Feature 1: Raw VIX level ----
        results['vix_level'].append(vix_level)

        # ---- Feature 2: Raw VIX3M level ----
        results['vix3m_level'].append(vix3m_level)

        # ---- Feature 3: VIX term structure ratio ----
        # > 1.0 = backwardation (near-term panic)
        # < 1.0 = contango (normal, calm)
        if pd.notna(vix3m_level) and vix3m_level > 0:
            ts_ratio = vix_level / vix3m_level
        else:
            ts_ratio = np.nan
        results['vix_term_structure_ratio'].append(ts_ratio)

        # ---- Feature 4: Binary regime flag ----
        # 1 = backwardation (stressed market), 0 = contango (normal)
        if pd.notna(ts_ratio):
            regime = 1 if ts_ratio > VIX_BACKWARDATION_THRESHOLD else 0
        else:
            regime = np.nan
        results['vix_term_structure_regime'].append(regime)

        # ---- Feature 5: VIX percentile over trailing 252 trading days ----
        # Where does today's VIX sit relative to the past year?
        lookback_start = latest_date - pd.Timedelta(days=VIX_PERCENTILE_WINDOW_DAYS)
        trailing = available.loc[available.index >= lookback_start, 'vix_close']

        if len(trailing) >= 50:
            pctl = float((trailing < vix_level).sum()) / len(trailing)
        else:
            pctl = np.nan
        results['vix_percentile_252d'].append(pctl)

        # ---- Feature 6: VIX 5-day change (short-term momentum) ----
        # Is fear rising or falling heading into this earnings?
        n_short = VIX_SHORT_MOMENTUM_DAYS + 1  # need n+1 rows for n-day change
        if len(available) >= n_short:
            vix_prev = float(available.iloc[-n_short]['vix_close'])
            change_5d = (vix_level - vix_prev) / vix_prev if vix_prev > 0 else np.nan
        else:
            change_5d = np.nan
        results['vix_change_5d'].append(change_5d)

        # ---- Feature 7: VIX 21-day change (medium-term trend) ----
        n_med = VIX_MEDIUM_MOMENTUM_DAYS + 1
        if len(available) >= n_med:
            vix_prev_21 = float(available.iloc[-n_med]['vix_close'])
            change_21d = (vix_level - vix_prev_21) / vix_prev_21 if vix_prev_21 > 0 else np.nan
        else:
            change_21d = np.nan
        results['vix_change_21d'].append(change_21d)

    # Build features DataFrame and concatenate
    vix_features = pd.DataFrame(results, index=earnings_df.index)

    print(f"  VIX features computed. Coverage: "
          f"{vix_features['vix_level'].notna().sum()}/{len(earnings_df)} events")

    return pd.concat([earnings_df, vix_features], axis=1)


# =============================================================================
# PART C: BLACK-SCHOLES IV INVERSION
# =============================================================================

def implied_vol(price, S, K, T, r, option_type='call'):
    """
    Compute implied volatility via Black-Scholes inversion using Brent's method.

    Parameters:
        price: observed market price of the option (mid-price preferred)
        S: current underlying price
        K: strike price
        T: time to expiration in years (must be > 0)
        r: annualized risk-free rate
        option_type: 'call' or 'put'

    Returns:
        float: annualized implied volatility, or np.nan if solver fails
    """
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    def bs_price(sigma):
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Intrinsic value check: option price must exceed intrinsic
    if option_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if price <= intrinsic:
        return np.nan

    try:
        return brentq(lambda s: bs_price(s) - price, 0.01, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan


# =============================================================================
# PART D: SECTOR ETF OPTION CHAIN FETCHING
# =============================================================================

def fetch_option_chain_av(symbol, date_str, api_key):
    """
    Fetch a historical option chain snapshot from Alpha Vantage.

    Parameters:
        symbol: ticker string (e.g., 'XLK', 'SMH')
        date_str: date string 'YYYY-MM-DD'
        api_key: Alpha Vantage API key

    Returns:
        list of option contract dicts, or empty list on failure
    """
    import requests

    url = (
        f'https://www.alphavantage.co/query'
        f'?function=HISTORICAL_OPTIONS'
        f'&symbol={symbol}'
        f'&date={date_str}'
        f'&apikey={api_key}'
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Alpha Vantage returns errors in JSON with specific keys
        if 'Error Message' in data or 'Note' in data:
            note = data.get('Error Message') or data.get('Note', '')
            if 'rate limit' in note.lower() or 'call frequency' in note.lower():
                warnings.warn(f"Rate limited on {symbol} {date_str}: {note}")
            return []

        return data.get('data', [])

    except Exception as e:
        warnings.warn(f"Error fetching option chain for {symbol} on {date_str}: {e}")
        return []


def _get_etf_price(symbol, date_str):
    """
    Get the closing price of an ETF on a specific date via Yahoo Finance.

    Falls back to the most recent prior trading day if the exact date
    is a weekend or holiday.

    Parameters:
        symbol: ETF ticker
        date_str: 'YYYY-MM-DD'

    Returns:
        float: closing price, or np.nan if unavailable
    """
    target = pd.Timestamp(date_str)
    start = target - pd.Timedelta(days=10)  # buffer for weekends/holidays

    try:
        data = yf.download(
            symbol,
            start=start.strftime('%Y-%m-%d'),
            end=(target + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            progress=False
        )
        if data.empty:
            return np.nan

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            close_col = data['Close']
            if close_col.ndim > 1:
                close_col = close_col.iloc[:, 0]
        else:
            close_col = data['Close']

        valid = close_col.loc[close_col.index <= target]
        if valid.empty:
            return np.nan

        return float(valid.iloc[-1])

    except Exception:
        return np.nan


# In-memory ETF price cache to avoid redundant Yahoo Finance calls.
# Persists for the duration of the Python process (not across runs).
_etf_price_cache = {}


def get_etf_price_cached(symbol, date_str):
    """Cached wrapper around _get_etf_price."""
    cache_key = f'{symbol}_{date_str}'
    if cache_key not in _etf_price_cache:
        _etf_price_cache[cache_key] = _get_etf_price(symbol, date_str)
    return _etf_price_cache[cache_key]


# =============================================================================
# PART E: SECTOR ETF ATM IV COMPUTATION
# =============================================================================

def compute_sector_etf_iv(chain, etf_price, chain_date_str,
                          risk_free_rate=DEFAULT_RISK_FREE_RATE):
    """
    From a raw option chain, find ATM options and compute implied volatility.

    Parameters:
        chain: list of contract dicts from Alpha Vantage HISTORICAL_OPTIONS
        etf_price: current price of the sector ETF
        chain_date_str: 'YYYY-MM-DD' date the chain was fetched for
        risk_free_rate: annualized risk-free rate for BS inversion

    Returns:
        dict with keys:
            sector_atm_iv_call, sector_atm_iv_put, sector_atm_iv_avg,
            sector_straddle_pct
    """
    empty_result = {
        'sector_atm_iv_call':  np.nan,
        'sector_atm_iv_put':   np.nan,
        'sector_atm_iv_avg':   np.nan,
        'sector_straddle_pct': np.nan,
    }

    if not chain or not etf_price or np.isnan(etf_price) or etf_price <= 0:
        return empty_result

    chain_date = datetime.strptime(chain_date_str, '%Y-%m-%d')

    # Step 1: Filter contracts to valid DTE window
    valid_contracts = []
    for c in chain:
        try:
            exp = datetime.strptime(c['expiration'], '%Y-%m-%d')
            dte = (exp - chain_date).days
            if SECTOR_IV_MIN_DTE <= dte <= SECTOR_IV_MAX_DTE:
                valid_contracts.append(c)
        except (KeyError, ValueError):
            continue

    if not valid_contracts:
        return empty_result

    # Step 2: Find ATM strike (closest to ETF price)
    strikes = set()
    for c in valid_contracts:
        try:
            strikes.add(float(c['strike']))
        except (KeyError, ValueError):
            continue

    if not strikes:
        return empty_result

    atm_strike = min(strikes, key=lambda k: abs(k - etf_price))

    # Step 3: Extract ATM call and put prices
    atm_call_price = np.nan
    atm_put_price = np.nan
    expiration_str = None

    for c in valid_contracts:
        try:
            if float(c['strike']) != atm_strike:
                continue

            # Prefer mid price (bid+ask)/2, fall back to last traded
            bid = float(c.get('bid', 0) or 0)
            ask = float(c.get('ask', 0) or 0)
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            else:
                mid = float(c.get('last', 0) or 0)

            if mid <= 0:
                continue

            contract_type = c.get('type', '').lower()
            if contract_type == 'call':
                atm_call_price = mid
                expiration_str = c.get('expiration')
            elif contract_type == 'put':
                atm_put_price = mid
                expiration_str = expiration_str or c.get('expiration')

        except (KeyError, ValueError):
            continue

    # Step 4: Compute time to expiration
    if expiration_str:
        exp_dt = datetime.strptime(expiration_str, '%Y-%m-%d')
        T = max((exp_dt - chain_date).days / 365.0, 1 / 365)
    else:
        T = 30 / 365  # fallback ~30 days

    # Step 5: Compute IV for call and put
    iv_call = np.nan
    iv_put = np.nan

    if not np.isnan(atm_call_price) and atm_call_price > 0:
        iv_call = implied_vol(
            atm_call_price, etf_price, atm_strike, T, risk_free_rate, 'call'
        )

    if not np.isnan(atm_put_price) and atm_put_price > 0:
        iv_put = implied_vol(
            atm_put_price, etf_price, atm_strike, T, risk_free_rate, 'put'
        )

    # Step 6: Average IV (standard: average call and put ATM IV)
    ivs = [v for v in [iv_call, iv_put] if not np.isnan(v)]
    iv_avg = float(np.mean(ivs)) if ivs else np.nan

    # Step 7: Straddle as % of ETF price
    straddle = 0
    n_legs = 0
    if not np.isnan(atm_call_price) and atm_call_price > 0:
        straddle += atm_call_price
        n_legs += 1
    if not np.isnan(atm_put_price) and atm_put_price > 0:
        straddle += atm_put_price
        n_legs += 1

    # Only compute if we have both legs (or at least one with doubling estimate)
    # Multiply by STRADDLE_PCT_MULTIPLIER (100) to output PERCENTAGE POINTS,
    # matching the pilot convention: straddle_pct_pre = 7.4 means 7.4% of price.
    if n_legs == 2:
        straddle_pct = (straddle / etf_price) * STRADDLE_PCT_MULTIPLIER
    elif n_legs == 1:
        # Approximate: if only one leg, double it (ATM call ~ ATM put)
        straddle_pct = ((straddle * 2) / etf_price) * STRADDLE_PCT_MULTIPLIER
    else:
        straddle_pct = np.nan

    return {
        'sector_atm_iv_call':  iv_call,
        'sector_atm_iv_put':   iv_put,
        'sector_atm_iv_avg':   iv_avg,
        'sector_straddle_pct': straddle_pct,
    }


# =============================================================================
# PART F: SECTOR IV PIPELINE (with caching)
# =============================================================================

def get_sector_iv_for_event(sector, event_date, api_key,
                            sub_sector=None, risk_free_rate=DEFAULT_RISK_FREE_RATE):
    """
    Get sector ETF ATM IV for a given sector and date.

    Uses file-based caching to avoid redundant API calls. Each unique
    (ETF, date) pair is fetched once and cached permanently.

    Parameters:
        sector: GICS sector string (e.g., 'Technology')
        event_date: datetime-like or string 'YYYY-MM-DD'
        api_key: Alpha Vantage API key
        sub_sector: optional sub-sector for more granular ETF matching
        risk_free_rate: for BS IV inversion

    Returns:
        dict with sector_etf + IV feature values
    """
    etf_symbol = get_sector_etf(sector=sector, sub_sector=sub_sector)

    empty_result = {
        'sector_etf':            None,
        'sector_atm_iv_call':    np.nan,
        'sector_atm_iv_put':     np.nan,
        'sector_atm_iv_avg':     np.nan,
        'sector_straddle_pct':   np.nan,
    }

    if not etf_symbol:
        warnings.warn(f"No ETF mapping for sector='{sector}', sub_sector='{sub_sector}'")
        return empty_result

    date_str = pd.Timestamp(event_date).strftime('%Y-%m-%d')

    # Check file cache
    os.makedirs(SECTOR_IV_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(SECTOR_IV_CACHE_DIR, f'{etf_symbol}_{date_str}.json')

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        # Convert None back to NaN for consistency
        for key in cached:
            if cached[key] is None and key != 'sector_etf':
                cached[key] = np.nan
        cached['sector_etf'] = etf_symbol
        return cached

    # Fetch option chain from Alpha Vantage
    chain = fetch_option_chain_av(etf_symbol, date_str, api_key)

    # Get the ETF closing price on that date
    etf_price = get_etf_price_cached(etf_symbol, date_str)

    # Compute ATM IV
    result = compute_sector_etf_iv(chain, etf_price, date_str, risk_free_rate)
    result['sector_etf'] = etf_symbol

    # Cache to disk (convert NaN to None for JSON serialization)
    cache_data = {}
    for key, val in result.items():
        if isinstance(val, float) and np.isnan(val):
            cache_data[key] = None
        else:
            cache_data[key] = val

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    # Rate limit courtesy
    time.sleep(AV_REQUEST_DELAY_SECONDS)

    return result


def add_sector_iv_features(earnings_df, api_key,
                           sector_col='sector', sub_sector_col=None,
                           date_col='announcement_date'):
    """
    Add sector-level IV features to every earnings event.

    For each event, resolves the sector to a sector ETF, fetches the
    option chain, and computes ATM IV. Results are cached per (ETF, date).

    Parameters:
        earnings_df: DataFrame with sector and date columns
        api_key: Alpha Vantage API key
        sector_col: name of column containing GICS sector labels
        sub_sector_col: optional column for sub-sector (e.g., 'Semiconductors')
        date_col: name of column containing announcement dates

    Returns:
        earnings_df with 5 new sector IV columns appended.
    """
    print(f"  Computing sector IV features for {len(earnings_df)} events...")

    # Count unique (ETF, date) pairs to estimate API calls needed
    unique_pairs = set()
    for _, row in earnings_df.iterrows():
        sector = row.get(sector_col, '')
        sub = row.get(sub_sector_col, '') if sub_sector_col else None
        date = pd.Timestamp(row[date_col]).strftime('%Y-%m-%d')
        etf = get_sector_etf(sector=sector, sub_sector=sub)
        if etf:
            unique_pairs.add((etf, date))

    # Check how many are already cached
    uncached = 0
    for etf, date in unique_pairs:
        cache_file = os.path.join(SECTOR_IV_CACHE_DIR, f'{etf}_{date}.json')
        if not os.path.exists(cache_file):
            uncached += 1

    print(f"  Unique (ETF, date) pairs: {len(unique_pairs)} "
          f"({uncached} uncached, will need API calls)")

    # Compute features per event (get_sector_iv_for_event handles caching)
    sector_features = []
    for i, (idx, row) in enumerate(earnings_df.iterrows()):
        sector = row.get(sector_col, '')
        sub = row.get(sub_sector_col, '') if sub_sector_col else None
        event_date = row[date_col]

        result = get_sector_iv_for_event(
            sector=sector,
            event_date=event_date,
            api_key=api_key,
            sub_sector=sub,
        )
        sector_features.append(result)

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(earnings_df)} events...")

    sector_df = pd.DataFrame(sector_features, index=earnings_df.index)

    coverage = sector_df['sector_atm_iv_avg'].notna().sum()
    print(f"  Sector IV features computed. Coverage: {coverage}/{len(earnings_df)} events")

    return pd.concat([earnings_df, sector_df], axis=1)


# =============================================================================
# PART G: RELATIVE IV FEATURES (Stock vs. Sector)
# =============================================================================

def compute_relative_iv_features(df,
                                 stock_iv_col=PILOT_STOCK_IV_COL,
                                 stock_straddle_col=PILOT_STOCK_STRADDLE_COL,
                                 sector_iv_col='sector_atm_iv_avg',
                                 sector_straddle_col='sector_straddle_pct'):
    """
    Compute features that compare stock-level IV to sector-level IV.

    These relative features are where the strongest signal typically lives.
    A stock with 80% IV when its sector ETF is at 20% has a 4x ratio,
    indicating a massive earnings-specific premium. A stock at 30% when
    sector is at 25% (1.2x ratio) has a small one.

    NOTE: Mutates df in-place and also returns it for chaining.

    Parameters:
        df: DataFrame with both stock-level and sector-level IV features
        stock_iv_col: column name for the stock's pre-event ATM IV
        stock_straddle_col: column name for stock's straddle as % of price
        sector_iv_col: column name for sector ETF ATM IV
        sector_straddle_col: column name for sector ETF straddle %

    Returns:
        df with 4 new relative feature columns appended.
    """
    print("  Computing relative IV features (stock vs. sector)...")

    # Feature 1: IV spread (stock minus sector)
    # Positive = stock has more IV than sector (earnings-specific premium)
    if stock_iv_col in df.columns and sector_iv_col in df.columns:
        df['iv_stock_minus_sector'] = df[stock_iv_col] - df[sector_iv_col]
    else:
        df['iv_stock_minus_sector'] = np.nan
        warnings.warn(f"Missing columns for IV spread: need '{stock_iv_col}' and '{sector_iv_col}'")

    # Feature 2: IV ratio (stock / sector)
    if stock_iv_col in df.columns and sector_iv_col in df.columns:
        df['iv_stock_sector_ratio'] = np.where(
            df[sector_iv_col] > 0,
            df[stock_iv_col] / df[sector_iv_col],
            np.nan
        )
    else:
        df['iv_stock_sector_ratio'] = np.nan

    # Feature 3: Straddle premium relative to sector
    if stock_straddle_col in df.columns and sector_straddle_col in df.columns:
        df['straddle_stock_minus_sector'] = (
            df[stock_straddle_col] - df[sector_straddle_col]
        )
    else:
        df['straddle_stock_minus_sector'] = np.nan

    # Feature 4: Binary flag — is stock IV elevated vs. sector?
    if 'iv_stock_sector_ratio' in df.columns:
        df['iv_elevated_vs_sector'] = np.where(
            df['iv_stock_sector_ratio'].notna(),
            (df['iv_stock_sector_ratio'] > IV_ELEVATED_VS_SECTOR_THRESHOLD).astype(int),
            np.nan
        )
    else:
        df['iv_elevated_vs_sector'] = np.nan

    return df


# =============================================================================
# PART H: MISSING DATA HANDLING
# =============================================================================

def handle_market_vol_missing_data(df, model_type='tree'):
    """
    Handle NaN values in market-wide volatility features.

    Strategy depends on model type:
    - Tree-based models (LightGBM, XGBoost): leave sector IV NaN as-is
      (native support), forward-fill VIX (nearly complete, gaps are rare)
    - Linear models (LR, RF): median-impute sector IV columns

    VIX forward-fill and regime fill-with-0 happen for both model types
    because VIX data is market-wide and should always be present.

    Parameters:
        df: DataFrame with market vol features
        model_type: 'tree' for LightGBM/XGBoost, 'linear' for LR/RF

    Returns:
        df with missing values handled
    """
    print(f"  Handling missing data (strategy: {model_type})...")

    # VIX features: forward-fill first (data is very complete, gaps are rare)
    vix_numeric_cols = [
        'vix_level', 'vix3m_level', 'vix_term_structure_ratio',
        'vix_percentile_252d', 'vix_change_5d', 'vix_change_21d'
    ]
    for col in vix_numeric_cols:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = df[col].ffill()
            after_na = df[col].isna().sum()
            if before_na > after_na:
                print(f"    {col}: filled {before_na - after_na} NaN via forward-fill")

    # Binary regime: fill with 0 (assume contango/normal if unknown)
    if 'vix_term_structure_regime' in df.columns:
        df['vix_term_structure_regime'] = df['vix_term_structure_regime'].fillna(0)

    # Sector IV features: strategy depends on model type
    sector_cols = [
        'sector_atm_iv_call', 'sector_atm_iv_put', 'sector_atm_iv_avg',
        'sector_straddle_pct', 'iv_stock_minus_sector', 'iv_stock_sector_ratio',
        'straddle_stock_minus_sector', 'iv_elevated_vs_sector'
    ]

    if model_type == 'linear':
        for col in sector_cols:
            if col in df.columns:
                before_na = df[col].isna().sum()
                if before_na > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"    {col}: filled {before_na} NaN with median ({median_val:.4f})")
    else:
        # Tree models: leave NaN, just report coverage
        for col in sector_cols:
            if col in df.columns:
                na_count = df[col].isna().sum()
                if na_count > 0:
                    print(f"    {col}: {na_count} NaN (left as-is for tree model)")

    return df


# =============================================================================
# PART I: VALIDATION
# =============================================================================

def validate_market_vol_features(df):
    """
    Sanity-check the market-wide volatility features.

    Prints a summary of range, coverage, and out-of-bounds values.
    Use this after computing features and before training.

    Parameters:
        df: DataFrame with market vol features

    Returns:
        dict: validation summary per feature
    """
    print("\n  === Market-Wide Vol Feature Validation ===\n")

    summary = {}
    for col, (low, high) in FEATURE_VALIDATION_RANGES.items():
        if col not in df.columns:
            print(f"  WARNING: {col} not found in DataFrame")
            summary[col] = {'status': 'missing'}
            continue

        valid = df[col].dropna()
        total = len(df)
        missing = total - len(valid)
        out_low = (valid < low).sum()
        out_high = (valid > high).sum()

        status = 'OK'
        if missing / total > 0.5:
            status = 'HIGH_MISSING'
        elif (out_low + out_high) / max(len(valid), 1) > 0.05:
            status = 'OUT_OF_RANGE'

        summary[col] = {
            'status': status,
            'min': float(valid.min()) if len(valid) > 0 else None,
            'max': float(valid.max()) if len(valid) > 0 else None,
            'mean': float(valid.mean()) if len(valid) > 0 else None,
            'missing': missing,
            'missing_pct': 100 * missing / total,
            'out_of_range': int(out_low + out_high),
        }

        print(f"  {col}:")
        if summary[col]['min'] is not None:
            print(f"    Range:  {summary[col]['min']:.4f} to {summary[col]['max']:.4f}")
        else:
            print(f"    Range: N/A")
        print(f"    Missing: {missing}/{total} ({summary[col]['missing_pct']:.1f}%)")
        print(f"    Out of expected [{low}, {high}]: {summary[col]['out_of_range']}")
        print(f"    Status: {status}")
        print()

    return summary


# =============================================================================
# PART J: MASTER FUNCTION — ties everything together
# =============================================================================

def add_market_wide_vol_features(earnings_df, api_key=None,
                                 date_col='announcement_date',
                                 sector_col='sector',
                                 sub_sector_col=None,
                                 stock_iv_col=PILOT_STOCK_IV_COL,
                                 stock_straddle_col=PILOT_STOCK_STRADDLE_COL,
                                 model_type='tree',
                                 validate=True):
    """
    Master function: adds all market-wide volatility context features.

    Call this AFTER you've computed stock-level IV features (iv_avg_pre,
    straddle_pct_pre, etc.) but BEFORE model training.

    Pipeline:
        1. Fetch VIX/VIX3M data (Yahoo Finance, free)
        2. Compute 7 VIX-based regime features
        3. Fetch sector ETF option chains (Alpha Vantage, uses API quota)
        4. Compute 4 raw sector IV features
        5. Compute 4 relative stock-vs-sector features
        6. Handle missing data
        7. Validate feature ranges

    Parameters:
        earnings_df: DataFrame with stock-level features already computed
        api_key: Alpha Vantage API key (for sector ETF chains).
                 If None or 'YOUR_KEY_HERE', sector IV steps are skipped
                 and only VIX features are computed.
        date_col: column name for announcement dates
        sector_col: column name for GICS sector labels.
                    If this column is missing from earnings_df, sector IV
                    steps are skipped gracefully.
        sub_sector_col: optional column for sub-sector granularity
        stock_iv_col: column name for stock's pre-event ATM IV
                      (default: 'iv_avg_pre' matching pilot data)
        stock_straddle_col: column name for stock's straddle % of price
                            (default: 'straddle_pct_pre' matching pilot data)
        model_type: 'tree' or 'linear' (affects NaN handling)
        validate: if True, run validation checks and print summary

    Returns:
        DataFrame with ~15 new market-wide vol features added
    """
    print("=" * 60)
    print("MARKET-WIDE VOLATILITY CONTEXT PIPELINE")
    print("=" * 60)

    n_events = len(earnings_df)
    print(f"\nInput: {n_events} earnings events")

    # Determine whether we can run the sector IV pipeline
    has_api_key = (api_key and api_key != 'YOUR_KEY_HERE')
    has_sector_col = (sector_col and sector_col in earnings_df.columns)

    if not has_api_key:
        print("\n  NOTE: No API key provided — sector IV steps will be skipped.")
    if not has_sector_col:
        print(f"\n  NOTE: Column '{sector_col}' not found — sector IV steps will be skipped.")

    # Step 1: Fetch VIX data
    print("\n[Step 1/6] Fetching VIX data...")
    vix_df = fetch_vix_data()

    # Step 2: Compute VIX features
    print("\n[Step 2/6] Computing VIX features...")
    df = compute_vix_features(earnings_df, vix_df, date_col=date_col)

    # Step 3: Compute sector ETF IV features (requires API key + sector column)
    if has_api_key and has_sector_col:
        print("\n[Step 3/6] Computing sector IV features...")
        df = add_sector_iv_features(
            df, api_key,
            sector_col=sector_col,
            sub_sector_col=sub_sector_col,
            date_col=date_col
        )

        # Step 4: Compute relative IV features
        print("\n[Step 4/6] Computing relative IV features...")
        df = compute_relative_iv_features(
            df,
            stock_iv_col=stock_iv_col,
            stock_straddle_col=stock_straddle_col,
        )
    else:
        print("\n[Step 3/6] Skipping sector IV (no API key or sector column)")
        print("[Step 4/6] Skipping relative IV features")

    # Step 5: Handle missing data
    print("\n[Step 5/6] Handling missing data...")
    df = handle_market_vol_missing_data(df, model_type=model_type)

    # Step 6: Validate
    if validate:
        print("\n[Step 6/6] Validating features...")
        validate_market_vol_features(df)
    else:
        print("\n[Step 6/6] Skipping validation (validate=False)")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Output: {len(df)} events x {len(df.columns)} features")
    print("=" * 60)

    return df

"""
test_market_vol_context.py — Smoke test for Market-Wide Volatility Context features.

Run this to verify the pipeline works before integrating into the full notebook.
Uses a small sample of synthetic earnings events to test each stage.

Usage:
    cd Option_Volatility_Crush
    python -m vol_crush_strategy.test_market_vol_context

    Or from the repo root:
    python -m pytest vol_crush_strategy/test_market_vol_context.py -v
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_crush_strategy.config import (
    ALPHA_VANTAGE_API_KEY,
    SECTOR_ETF_MAP,
    SUBSECTOR_ETF_MAP,
    get_sector_etf,
)
from vol_crush_strategy.market_vol_context import (
    fetch_vix_data,
    compute_vix_features,
    get_sector_iv_for_event,
    compute_relative_iv_features,
    handle_market_vol_missing_data,
    validate_market_vol_features,
    add_market_wide_vol_features,
)


def create_sample_earnings():
    """
    Create a small synthetic earnings DataFrame for testing.

    Mimics the structure of the real earnings dataset with columns
    the pipeline expects. Uses real tickers and dates so VIX lookups work.
    """
    data = {
        'symbol': ['NVDA', 'AAPL', 'JPM', 'JNJ', 'XOM',
                    'MSFT', 'AMZN', 'META', 'GOOGL', 'TSLA'],
        'announcement_date': [
            '2024-02-21', '2024-02-01', '2024-01-12', '2024-01-23', '2024-02-02',
            '2024-01-30', '2024-02-01', '2024-02-01', '2024-01-30', '2024-01-24',
        ],
        'sector': [
            'Technology', 'Technology', 'Financials', 'Health Care', 'Energy',
            'Technology', 'Consumer Discretionary', 'Communication Services',
            'Communication Services', 'Consumer Discretionary',
        ],
        'sub_sector': [
            'Semiconductors', 'Technology', 'Banks', 'Healthcare', 'Oil & Gas',
            'Software', 'Internet', 'Internet', 'Internet', 'Consumer Discretionary',
        ],
        # Simulated stock-level IV features (these would come from your IV pipeline)
        # iv_avg_pre: annualized decimal (e.g. 0.65 = 65% IV)
        # straddle_pct_pre: percentage points (e.g. 8.0 = 8% of stock price)
        'iv_avg_pre': [0.65, 0.32, 0.22, 0.18, 0.28,
                       0.35, 0.42, 0.48, 0.38, 0.72],
        'straddle_pct_pre': [8.0, 4.0, 3.0, 2.0, 3.5,
                             4.5, 5.0, 6.0, 4.8, 9.0],
    }

    df = pd.DataFrame(data)
    df['announcement_date'] = pd.to_datetime(df['announcement_date'])
    return df


def test_sector_etf_mapping():
    """Test that sector-to-ETF mapping works correctly."""
    print("\n" + "=" * 50)
    print("TEST: Sector ETF Mapping")
    print("=" * 50)

    # Test basic sector mapping
    assert get_sector_etf(sector='Technology') == 'XLK'
    assert get_sector_etf(sector='Financials') == 'XLF'
    assert get_sector_etf(sector='Energy') == 'XLE'
    print("  Basic sector mapping: PASS")

    # Test sub-sector override
    assert get_sector_etf(sector='Technology', sub_sector='Semiconductors') == 'SMH'
    assert get_sector_etf(sector='Financials', sub_sector='Banks') == 'KBE'
    assert get_sector_etf(sector='Health Care', sub_sector='Biotech') == 'XBI'
    print("  Sub-sector override: PASS")

    # Test fallback (unknown sub-sector falls back to sector)
    assert get_sector_etf(sector='Technology', sub_sector='Unknown') == 'XLK'
    print("  Fallback to sector: PASS")

    # Test unknown sector returns None
    assert get_sector_etf(sector='Aliens') is None
    print("  Unknown sector returns None: PASS")

    print("\n  ALL MAPPING TESTS PASSED")


def test_vix_fetch_and_features():
    """Test VIX data fetching and feature computation."""
    print("\n" + "=" * 50)
    print("TEST: VIX Data Fetch & Feature Computation")
    print("=" * 50)

    # Fetch VIX data
    vix_df = fetch_vix_data(start_date='2023-01-01', end_date='2024-12-31')

    assert not vix_df.empty, "VIX data is empty"
    assert 'vix_close' in vix_df.columns, "Missing vix_close column"
    assert 'vix3m_close' in vix_df.columns, "Missing vix3m_close column"
    print(f"\n  Fetched {len(vix_df)} rows of VIX data")
    print(f"  VIX range: {vix_df['vix_close'].min():.2f} to {vix_df['vix_close'].max():.2f}")
    print(f"  VIX3M range: {vix_df['vix3m_close'].min():.2f} to {vix_df['vix3m_close'].max():.2f}")

    # Compute VIX features on sample data
    sample = create_sample_earnings()
    result = compute_vix_features(sample, vix_df)

    expected_cols = [
        'vix_level', 'vix3m_level', 'vix_term_structure_ratio',
        'vix_term_structure_regime', 'vix_percentile_252d',
        'vix_change_5d', 'vix_change_21d'
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing feature column: {col}"
    print(f"\n  All 7 VIX feature columns present: PASS")

    # Check values are reasonable
    vix_levels = result['vix_level'].dropna()
    assert (vix_levels > 0).all(), "VIX levels should be positive"
    assert (vix_levels < 100).all(), "VIX levels should be < 100"
    print(f"  VIX levels range: {vix_levels.min():.2f} to {vix_levels.max():.2f}: PASS")

    ts_ratios = result['vix_term_structure_ratio'].dropna()
    assert (ts_ratios > 0).all(), "Term structure ratio should be positive"
    print(f"  Term structure ratios: {ts_ratios.min():.3f} to {ts_ratios.max():.3f}: PASS")

    percentiles = result['vix_percentile_252d'].dropna()
    assert (percentiles >= 0).all() and (percentiles <= 1).all(), "Percentile should be [0, 1]"
    print(f"  VIX percentiles: {percentiles.min():.3f} to {percentiles.max():.3f}: PASS")

    print("\n  ALL VIX TESTS PASSED")
    return result


def test_relative_iv_features():
    """Test relative IV feature computation."""
    print("\n" + "=" * 50)
    print("TEST: Relative IV Features")
    print("=" * 50)

    # Create sample with both stock and (simulated) sector IV
    # sector_atm_iv_avg: annualized decimal (same unit as iv_avg_pre)
    # sector_straddle_pct: percentage points (same unit as straddle_pct_pre)
    sample = create_sample_earnings()
    sample['sector_atm_iv_avg'] = [0.18, 0.15, 0.14, 0.12, 0.20,
                                    0.16, 0.19, 0.21, 0.17, 0.25]
    sample['sector_straddle_pct'] = [2.5, 2.0, 1.8, 1.5, 2.8,
                                      2.2, 2.6, 2.9, 2.3, 3.3]

    result = compute_relative_iv_features(sample)

    # Check columns exist
    expected = ['iv_stock_minus_sector', 'iv_stock_sector_ratio',
                'straddle_stock_minus_sector', 'iv_elevated_vs_sector']
    for col in expected:
        assert col in result.columns, f"Missing: {col}"
    print(f"  All 4 relative feature columns present: PASS")

    # Check NVDA (row 0): IV=0.65, sector=0.18, ratio should be ~3.6
    nvda_ratio = result.iloc[0]['iv_stock_sector_ratio']
    assert 3.5 < nvda_ratio < 3.7, f"NVDA IV ratio should be ~3.6, got {nvda_ratio}"
    print(f"  NVDA iv_stock_sector_ratio = {nvda_ratio:.2f} (expected ~3.61): PASS")

    # Check NVDA should be flagged as elevated (ratio > 1.5)
    assert result.iloc[0]['iv_elevated_vs_sector'] == 1
    print(f"  NVDA iv_elevated_vs_sector = 1: PASS")

    # Check JNJ (row 3): IV=0.18, sector=0.12, ratio = 1.5 (borderline)
    jnj_ratio = result.iloc[3]['iv_stock_sector_ratio']
    print(f"  JNJ iv_stock_sector_ratio = {jnj_ratio:.2f}")

    # Spread should be positive for most (stock IV > sector IV pre-earnings)
    spreads = result['iv_stock_minus_sector'].dropna()
    print(f"  IV spread range: {spreads.min():.3f} to {spreads.max():.3f}")

    print("\n  ALL RELATIVE IV TESTS PASSED")


def test_missing_data_handling():
    """Test missing data imputation strategies."""
    print("\n" + "=" * 50)
    print("TEST: Missing Data Handling")
    print("=" * 50)

    sample = create_sample_earnings()
    # Add VIX features with some NaN
    sample['vix_level'] = [13.5, np.nan, 14.2, 13.8, np.nan,
                           12.9, 13.1, np.nan, 14.0, 13.7]
    sample['vix_term_structure_regime'] = [0, np.nan, 1, 0, np.nan,
                                            0, 0, np.nan, 1, 0]
    sample['sector_atm_iv_avg'] = [0.18, np.nan, 0.14, np.nan, 0.20,
                                    0.16, np.nan, 0.21, 0.17, np.nan]

    # Test tree model handling (forward-fill VIX, leave sector NaN)
    tree_result = handle_market_vol_missing_data(sample.copy(), model_type='tree')
    # VIX should be forward-filled
    assert tree_result['vix_level'].iloc[1] == 13.5  # ffill from row 0
    print("  Tree model forward-fill: PASS")

    # Regime should fill with 0
    assert tree_result['vix_term_structure_regime'].iloc[1] == 0
    print("  Regime fill with 0: PASS")

    # Sector IV should remain NaN for tree models
    assert pd.isna(tree_result['sector_atm_iv_avg'].iloc[1])
    print("  Sector IV NaN preserved for tree model: PASS")

    # Test linear model handling (median imputation)
    linear_result = handle_market_vol_missing_data(sample.copy(), model_type='linear')
    # Sector IV should be imputed with median
    assert not pd.isna(linear_result['sector_atm_iv_avg'].iloc[1])
    print("  Sector IV median-imputed for linear model: PASS")

    print("\n  ALL MISSING DATA TESTS PASSED")


def test_validation():
    """Test the validation function catches issues."""
    print("\n" + "=" * 50)
    print("TEST: Feature Validation")
    print("=" * 50)

    sample = create_sample_earnings()
    sample['vix_level'] = [13.5, 14.2, 13.8, 95.0, 12.9,
                           13.1, 14.5, 14.0, 13.7, 15.1]  # 95.0 is out of range
    sample['vix_percentile_252d'] = [0.45, 0.52, 0.61, 0.38, 0.72,
                                      0.55, 0.48, 0.66, 0.59, 0.41]

    summary = validate_market_vol_features(sample)

    assert summary['vix_level']['out_of_range'] >= 1, "Should flag out-of-range VIX"
    print("  Out-of-range detection: PASS")

    print("\n  ALL VALIDATION TESTS PASSED")


def run_full_pipeline_demo():
    """
    Run the complete pipeline on sample data.

    NOTE: The sector IV step requires an Alpha Vantage API key.
    If no key is set, this demo will skip the sector IV step and
    still test VIX features + relative IV computation with mock data.
    """
    print("\n" + "=" * 60)
    print("FULL PIPELINE DEMO")
    print("=" * 60)

    sample = create_sample_earnings()
    print(f"\nSample data: {len(sample)} earnings events")
    print(sample[['symbol', 'announcement_date', 'sector', 'iv_avg_pre']].to_string())

    has_api_key = (ALPHA_VANTAGE_API_KEY != 'YOUR_KEY_HERE'
                   and ALPHA_VANTAGE_API_KEY is not None)

    if has_api_key:
        print("\n  Alpha Vantage API key detected — running full pipeline")
        result = add_market_wide_vol_features(
            sample,
            api_key=ALPHA_VANTAGE_API_KEY,
            sector_col='sector',
            sub_sector_col='sub_sector',
            model_type='tree',
            validate=True,
        )
    else:
        print("\n  No API key — running VIX-only pipeline with mock sector IV")

        # Step 1: VIX features (works without API key)
        vix_df = fetch_vix_data(start_date='2023-01-01')
        result = compute_vix_features(sample, vix_df)

        # Step 2: Mock sector IV (simulate what the API would return)
        # sector_atm_iv_*: annualized decimal (0.18 = 18% IV)
        # sector_straddle_pct: percentage points (2.7 = 2.7% of ETF price)
        np.random.seed(42)
        result['sector_etf'] = result['sector'].map(SECTOR_ETF_MAP)
        result['sector_atm_iv_call'] = np.random.uniform(0.12, 0.25, len(result))
        result['sector_atm_iv_put'] = result['sector_atm_iv_call'] + np.random.uniform(-0.02, 0.02, len(result))
        result['sector_atm_iv_avg'] = (result['sector_atm_iv_call'] + result['sector_atm_iv_put']) / 2
        # Multiply by 100 to get percentage points (matching pilot convention)
        result['sector_straddle_pct'] = result['sector_atm_iv_avg'] * 15.0

        # Step 3: Relative features (uses default pilot column names)
        result = compute_relative_iv_features(result)

        # Step 4: Missing data
        result = handle_market_vol_missing_data(result)

        # Step 5: Validate
        validate_market_vol_features(result)

    # Print summary
    print("\n  FINAL FEATURE SUMMARY:")
    market_cols = [c for c in result.columns if c.startswith(('vix_', 'sector_', 'iv_stock', 'iv_elevated', 'straddle_stock'))]
    print(f"  New market-wide features: {len(market_cols)}")
    for col in market_cols:
        na = result[col].isna().sum()
        print(f"    {col}: {na}/{len(result)} NaN")

    print(f"\n  Total features: {len(result.columns)}")
    return result


if __name__ == '__main__':
    print("=" * 60)
    print("MARKET-WIDE VOLATILITY CONTEXT — TEST SUITE")
    print("=" * 60)

    test_sector_etf_mapping()
    result = test_vix_fetch_and_features()
    test_relative_iv_features()
    test_missing_data_handling()
    test_validation()
    final = run_full_pipeline_demo()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

# Market-Wide Volatility Context Module

## Overview

This module (`vol_crush_strategy/`) adds **15 market-wide volatility context features** to the Option Volatility Crush strategy pipeline. It integrates with the existing NVDA pilot codebase without modifying any original files.

The module provides two categories of features that capture the broader volatility environment around each earnings event:

1. **VIX-Based Market Regime Features** (7 features) — free, uses Yahoo Finance
2. **Sector-Level IV Features** (4 raw + 4 relative = 8 features) — uses Alpha Vantage API

---

## Files Created

```
vol_crush_strategy/
├── __init__.py                  # Package exports (public API surface)
├── config.py                    # Central configuration — API keys, paths, ETF maps, thresholds
├── market_vol_context.py        # Core module — VIX features, sector IV, relative IV
└── README.md                    # This file

tests/
└── test_market_vol_context.py   # Smoke test suite (run to verify everything works)
```

### No Original Files Modified

The following files were **NOT touched**:
- `notebooks/vol_crush_utils.py`
- `notebooks/vol_crush_pilot.ipynb`
- All pilot CSVs in `notebooks/pilot_data/`
- `.env.example`
- Root `README.md`

---

## Feature Inventory

### VIX-Based Features (7 features — always available, no API key needed)

| Feature | Type | Description |
|---------|------|-------------|
| `vix_level` | float | VIX daily close on event date |
| `vix3m_level` | float | VIX3M (3-month VIX) daily close |
| `vix_term_structure_ratio` | float | VIX / VIX3M — >1.0 = backwardation (panic), <1.0 = contango (calm) |
| `vix_term_structure_regime` | int | Binary: 1 = backwardation, 0 = contango |
| `vix_percentile_252d` | float [0,1] | Where today's VIX sits relative to the trailing year |
| `vix_change_5d` | float | 5-day VIX percentage change (short-term fear momentum) |
| `vix_change_21d` | float | 21-day VIX percentage change (medium-term trend) |

**Data source:** Yahoo Finance (`^VIX`, `^VIX3M`) — free, no API quota consumed.
**Caching:** Saved to `data/vol_crush/vix_cache/vix_daily.csv` after first download.

### Sector IV Raw Features (4 features — requires Alpha Vantage API key)

| Feature | Type | Description |
|---------|------|-------------|
| `sector_etf` | string | Resolved sector ETF ticker (e.g., 'XLK', 'SMH') |
| `sector_atm_iv_call` | float | ATM call implied volatility of sector ETF (annualized decimal) |
| `sector_atm_iv_put` | float | ATM put implied volatility of sector ETF (annualized decimal) |
| `sector_atm_iv_avg` | float | Average of call and put ATM IV (annualized decimal) |
| `sector_straddle_pct` | float | Sector ETF straddle cost as % of ETF price (**percentage points**) |

**Data source:** Alpha Vantage `HISTORICAL_OPTIONS` API — same endpoint the pilot uses.
**Caching:** Per (ETF, date) pair in `data/vol_crush/sector_iv_cache/{ETF}_{date}.json`.

### Relative IV Features (4 features — computed from stock + sector IV)

| Feature | Type | Description |
|---------|------|-------------|
| `iv_stock_minus_sector` | float | Stock IV - Sector IV (earnings-specific premium spread) |
| `iv_stock_sector_ratio` | float | Stock IV / Sector IV (ratio > 1.5 = strongly elevated) |
| `straddle_stock_minus_sector` | float | Stock straddle % - Sector straddle % (cost premium) |
| `iv_elevated_vs_sector` | int | Binary: 1 if ratio > 1.5x threshold, 0 otherwise |

**These relative features are where the strongest predictive signal typically lives.** A stock with 80% IV when its sector ETF is at 20% (4x ratio) has a massive earnings-specific premium — these trades have higher probability of profitable vol crush.

---

## Integration with Existing Codebase

### API Key Compatibility

The `.env` file uses `ALPHA_VANTAGE_API` (matching the pilot notebook pattern). Our `config.py` checks three env var names in priority order:

```python
ALPHA_VANTAGE_API_KEY = (
    os.environ.get('ALPHA_VANTAGE_API')        # matches .env / pilot
    or os.environ.get('ALPHA_VANTAGE_API_KEY')  # common alternative
    or os.environ.get('AV_API_KEY')             # another fallback
    or 'YOUR_KEY_HERE'
)
```

### Column Name Alignment

The pilot notebook (`vol_crush_pilot.ipynb`) and pilot CSVs (`04_iv_straddle_metrics.csv`) use specific column names that differ from generic defaults. Our module defaults match exactly:

| Pilot CSV Column | Config Constant | Used As Default In |
|---|---|---|
| `iv_avg_pre` | `PILOT_STOCK_IV_COL` | `compute_relative_iv_features(stock_iv_col=...)` |
| `straddle_pct_pre` | `PILOT_STOCK_STRADDLE_COL` | `compute_relative_iv_features(stock_straddle_col=...)` |
| `announcement_date` | `PILOT_DATE_COL` | `compute_vix_features(date_col=...)` |

### Unit Convention Alignment

This is the most critical integration point. The pilot's `vol_crush_utils.compute_straddle_metrics()` outputs straddle cost in **percentage points**:

```python
# vol_crush_utils.py line 278:
straddle_pct = (straddle_price / stock_price) * 100
# Output: 7.4 means 7.4% of stock price
```

Our `compute_sector_etf_iv()` matches this exactly:

```python
# market_vol_context.py line 532:
straddle_pct = (straddle / etf_price) * STRADDLE_PCT_MULTIPLIER  # * 100
# Output: 2.5 means 2.5% of ETF price
```

This ensures `straddle_stock_minus_sector = straddle_pct_pre - sector_straddle_pct` produces a meaningful apples-to-apples difference in percentage points.

IV values remain in **annualized decimals** (e.g., 0.65 = 65% IV, 1.02 = 102% IV) everywhere, matching the pilot's `iv_avg_pre` convention.

### Alpha Vantage API Endpoint

Both the pilot and our module use the same API:

```
GET https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&date={date}&apikey={key}
```

- **Pilot** (`vol_crush_utils.fetch_historical_options`): Fetches individual stock chains (NVDA), caches JSON per `{symbol}/{date}.json`
- **Our module** (`fetch_option_chain_av`): Fetches sector ETF chains (XLK, SMH, etc.), caches computed IV per `{ETF}_{date}.json`

No conflict — they fetch different tickers and cache in different directories.

### Graceful Degradation

The master function `add_market_wide_vol_features()` handles two common scenarios:

1. **No API key set** → Skips sector IV steps, still computes 7 VIX features (free via Yahoo Finance)
2. **No sector column in DataFrame** → Skips sector IV steps gracefully (pilot data is NVDA-only, no sector column)

This means you can always run the VIX features pipeline regardless of API key status.

---

## Sector ETF Mapping

### GICS Sector → ETF (13 mappings)

| Sector | ETF | Description |
|--------|-----|-------------|
| Technology / Information Technology | XLK | Technology Select Sector |
| Communication Services | XLC | Communication Services Select |
| Consumer Discretionary | XLY | Consumer Discretionary Select |
| Consumer Staples | XLP | Consumer Staples Select |
| Energy | XLE | Energy Select Sector |
| Financials | XLF | Financial Select Sector |
| Health Care / Healthcare | XLV | Health Care Select Sector |
| Industrials | XLI | Industrial Select Sector |
| Materials | XLB | Materials Select Sector |
| Real Estate | XLRE | Real Estate Select Sector |
| Utilities | XLU | Utilities Select Sector |

### Sub-Sector → ETF Override (11 mappings)

| Sub-Sector | ETF | Used Instead Of |
|------------|-----|-----------------|
| Semiconductors | SMH | XLK |
| Biotech/Biotechnology | XBI | XLV |
| Software | IGV | XLK |
| Internet | FDN | XLC |
| Retail | XRT | XLY |
| Homebuilders | XHB | XLY |
| Banks | KBE | XLF |
| Regional Banks | KRE | XLF |
| Oil & Gas (E&P) | XOP | XLE |

The `get_sector_etf(sector, sub_sector)` function checks sub-sector first, then falls back to sector. This provides more granular IV context — e.g., NVDA (Semiconductors → SMH) gets compared against the semiconductor ETF rather than the broad tech ETF.

---

## Black-Scholes IV Inversion

The module includes its own `implied_vol()` function for computing sector ETF ATM implied volatility from option prices:

```python
implied_vol(price, S, K, T, r, option_type='call')
```

- Uses **Brent's method** (`scipy.optimize.brentq`) for reliable convergence
- Search bounds: `[0.01, 5.0]` (matching the README specification)
- Returns `np.nan` on failure (price below intrinsic, solver doesn't converge)
- Handles both calls and puts

This is separate from the pilot's `vol_crush_utils.implied_vol()` (which uses bounds `[0.01, 10.0]` for individual stocks that can have extreme IV). Sector ETF IV is always well below 5.0, so the tighter bound is appropriate and more numerically stable.

---

## Missing Data Strategy

VIX data is nearly complete (market-wide, published every trading day). Sector IV can have gaps when API returns empty chains or option markets are thin.

| Model Type | VIX Features | Regime Flag | Sector IV Features |
|------------|-------------|-------------|-------------------|
| `'tree'` (LightGBM, XGBoost) | Forward-fill | Fill with 0 (contango) | Leave NaN (native support) |
| `'linear'` (LR, RF) | Forward-fill | Fill with 0 (contango) | Median imputation |

---

## Validation Ranges

Built-in sanity checks flag values outside expected bounds:

| Feature | Expected Range | Notes |
|---------|---------------|-------|
| `vix_level` | 5 – 90 | VIX >90 has never happened; <5 would be historic lows |
| `vix3m_level` | 5 – 70 | VIX3M is always less volatile than VIX |
| `vix_term_structure_ratio` | 0.5 – 2.5 | Extreme backwardation/contango bounds |
| `vix_percentile_252d` | 0.0 – 1.0 | By definition |
| `vix_change_5d` | -0.8 – 3.0 | VIX can spike 3x but rarely drops >80% in 5 days |
| `vix_change_21d` | -0.8 – 5.0 | Wider range for monthly changes |
| `sector_atm_iv_avg` | 0.01 – 2.0 | Sector ETF IV in annualized decimal |
| `iv_stock_sector_ratio` | 0.1 – 50.0 | Extreme individual stock premium vs sector |

---

## Configuration Reference

All tunable parameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `VIX_LOOKBACK_START` | `'2016-01-01'` | Extra year before 2017 for rolling calculations |
| `VIX_PERCENTILE_WINDOW_DAYS` | 365 | Calendar days for percentile lookback (~252 trading days) |
| `VIX_SHORT_MOMENTUM_DAYS` | 5 | Trading days for short-term VIX change |
| `VIX_MEDIUM_MOMENTUM_DAYS` | 21 | Trading days for medium-term VIX change |
| `VIX_BACKWARDATION_THRESHOLD` | 1.0 | VIX/VIX3M ratio above this = backwardation |
| `SECTOR_IV_MIN_DTE` | 2 | Minimum days to expiration for sector ETF options |
| `SECTOR_IV_MAX_DTE` | 45 | Maximum days to expiration |
| `STRADDLE_PCT_MULTIPLIER` | 100 | Converts ratio to percentage points (matches pilot) |
| `DEFAULT_RISK_FREE_RATE` | 0.05 | For BS IV inversion (5% annualized) |
| `AV_REQUEST_DELAY_SECONDS` | 0.5 | Rate limit courtesy between API calls |
| `IV_ELEVATED_VS_SECTOR_THRESHOLD` | 1.5 | Stock/sector IV ratio flag threshold |

---

## Usage

### Quick Start (VIX features only — no API key needed)

```python
from vol_crush_strategy.market_vol_context import fetch_vix_data, compute_vix_features

# Load your earnings DataFrame (must have 'announcement_date' column)
earnings_df = pd.read_csv('your_earnings_data.csv')
earnings_df['announcement_date'] = pd.to_datetime(earnings_df['announcement_date'])

# Fetch VIX data and compute 7 features
vix_df = fetch_vix_data()
result = compute_vix_features(earnings_df, vix_df)
```

### Full Pipeline (all 15 features — requires Alpha Vantage API key)

```python
from vol_crush_strategy import add_market_wide_vol_features, ALPHA_VANTAGE_API_KEY

# Your DataFrame needs: announcement_date, sector, iv_avg_pre, straddle_pct_pre
result = add_market_wide_vol_features(
    earnings_df,
    api_key=ALPHA_VANTAGE_API_KEY,
    sector_col='sector',
    sub_sector_col='sub_sector',   # optional, for granular ETF matching
    model_type='tree',             # 'tree' or 'linear'
    validate=True,
)
```

### With Pilot Data (NVDA single-stock — no sector column)

```python
from vol_crush_strategy import add_market_wide_vol_features, ALPHA_VANTAGE_API_KEY

# Pilot data has no 'sector' column — pipeline skips sector IV gracefully
pilot_df = pd.read_csv('notebooks/pilot_data/07_labeled_dataset.csv')
pilot_df['announcement_date'] = pd.to_datetime(pilot_df['announcement_date'])

result = add_market_wide_vol_features(pilot_df, api_key=ALPHA_VANTAGE_API_KEY)
# → Adds 7 VIX features, skips sector IV (no sector column found)
```

---

## Running Tests

```bash
cd Option_Volatility_Crush
python -m pytest tests/test_market_vol_context.py -v
```

The test suite verifies:
1. Sector ETF mapping (basic, sub-sector override, fallback, unknown)
2. VIX data fetch + 7 feature computation (real Yahoo Finance data)
3. Relative IV features (ratio, spread, straddle diff, binary flag)
4. Missing data handling (tree vs. linear strategies)
5. Validation (out-of-range detection)
6. Full pipeline demo (mock sector IV when no API key, real VIX data)

---

## Data Flow Diagram

```
Earnings DataFrame                     Yahoo Finance (free)
(with iv_avg_pre,                      ┌──────────────┐
 straddle_pct_pre,                     │  ^VIX daily   │
 announcement_date)                    │  ^VIX3M daily │
        │                              └──────┬───────┘
        │                                     │
        ▼                                     ▼
┌───────────────────┐              ┌──────────────────────┐
│ add_market_wide_  │              │   fetch_vix_data()   │
│ vol_features()    │◄─────────────│   (cached to disk)   │
│  [master func]    │              └──────────────────────┘
└───────┬───────────┘
        │
        ├──► compute_vix_features()          → 7 VIX features
        │
        ├──► add_sector_iv_features()        → 5 sector IV features
        │    └─► get_sector_iv_for_event()      (per event, cached)
        │        ├─► get_sector_etf()           (sector → ETF lookup)
        │        ├─► fetch_option_chain_av()    (Alpha Vantage API)
        │        ├─► get_etf_price_cached()     (Yahoo Finance)
        │        └─► compute_sector_etf_iv()    (BS IV inversion)
        │
        ├──► compute_relative_iv_features()  → 4 relative features
        │
        ├──► handle_market_vol_missing_data() → NaN handling
        │
        └──► validate_market_vol_features()  → sanity checks
```

---

## No Forward-Looking Data Leakage

All features use exclusively data available **before or on** the earnings event date:
- VIX features use `vix_sorted.index <= event_date`
- Sector IV is fetched for the event date (pre-market chain snapshot)
- Relative features compare pre-event stock IV to event-date sector IV
- No post-event information is used in any feature computation

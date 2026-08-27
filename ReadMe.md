# Momentum portfolio - Finesse x Citadel Round 2

Work-in-progress repo for the portfolio construction challenge (Round 2).
Iterating on a momentum strategy over the Nifty 100 / Midcap 100 / Smallcap 100
universe, 2021-2025, Rs 1 crore starting capital, 0.1% cost per trade.

## Current state: v2

The pipeline (unchanged from v1): rank the universe by 12-1 momentum (252-day
return skipping the most recent 21 days), keep only positive scores, hold the
top names weighted by inverse 60-day volatility (20% cap per stock), review
quarterly, -15% stop-loss vs the review price, 0.1% cost on both legs.

What changed in v2 (see `iterations/strategy_v2.py`): the selection rule got a
rank buffer. Instead of re-picking a fresh top-8 every quarter, a holding is
kept while it still ranks inside the top 20 and sold only once it drops out;
empty slots are filled with the best-ranked newcomers. Book size went 8 -> 7.
This was the main problem flagged in the v1 notes -- the quarterly re-pick was
churning intact trends (a stock slipping from rank 3 to rank 9 got sold for
no real reason).

| | v1 | v2 |
|---|---|---|
| 2021-25 PNL | Rs 8.6 Cr | Rs 12.8 Cr |
| CAGR | 57% | 69% |
| Max drawdown | -28% | -27% |
| Sharpe | 2.5 | 2.8 |
| H1-2026 check | +16.6% | +22.3% |



## Running

```bash
pip install yfinance pandas numpy pyarrow
python iterations/strategy_v1.py   # prints 2021-25 backtest + H1-2026 check
python iterations/strategy_v2.py
```

Each script prints its own results; nothing else is written.

## Data

- Prices: Yahoo Finance daily adjusted closes (`yfinance`), downloaded once and
  cached as `prices_full.parquet` (2019 onward -- the extra history is only
  there so the first momentum computation has its full 12-month lookback).
- Universe: union of the official Nifty 100, Midcap 100 and Smallcap 100
  constituent lists from niftyindices.com (cached in
  `universe_official.txt`, 300 names).

## Known issues / next steps

- The -15% stop still fires on ordinary momentum dips rather than genuine
  breakage; the stop level needs another look.
- Accounting is weight-space (fractional shares, no trade log) -- the
  submission needs share-level books and trade statistics.
- Nothing has been stress-tested yet: parameters, costs, regimes.

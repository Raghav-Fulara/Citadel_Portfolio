# Momentum portfolio - Finesse x Citadel Round 2

Work-in-progress repo for the portfolio construction challenge (Round 2).
Iterating on a momentum strategy over the Nifty 100 / Midcap 100 / Smallcap 100
universe, 2021-2025, Rs 1 crore starting capital, 0.1% cost per trade.

## Current state: v3

The pipeline (unchanged since v1): rank the universe by 12-1 momentum (252-day
return skipping the most recent 21 days), keep only positive scores, hold the
top names with a rank buffer (a holding stays while it ranks inside the top 20),
weight by inverse 60-day volatility (20% cap per stock), review quarterly,
0.1% cost on both legs.

What changed in v3 (see `iterations/strategy_v3.py`): the stop-loss moved from
-15% to -25% (measured from the review price, as before). Going through the v2
exit log, nearly every -15% stop was an ordinary momentum dip that recovered --
the tight line was paying about 2 Cr for nothing. -25% behaves the way we
actually wanted: a catastrophic-damage cut (the 2022 selloff, the Adani break,
the Sep-24 crash) that otherwise leaves the book alone. v3 also prints the
Nifty 50 return next to every run, so results are quoted against the benchmark
from here on.

| | v1 | v2 | v3 |
|---|---|---|---|
| Selection | fresh top-8 | rank buffer, 7 stocks | rank buffer, 7 stocks |
| Stop-loss | -15% | -15% | -25% |
| 2021-25 PNL | Rs 8.6 Cr | Rs 12.8 Cr | Rs 14.9 Cr |
| CAGR | 57% | 69% | 74% |
| Max drawdown | -28% | -27% | -31% |
| Sharpe | 2.5 | 2.8 | 2.7 |
| H1-2026 check | +16.6% | +22.3% | +21.8% (Nifty 50: -8.7%) |

The wider stop buys PNL with a slightly deeper drawdown (-27% -> -31%) and
roughly flat Sharpe -- a trade we are comfortable making for a contest ranked
on net PNL, but it is exactly the kind of choice the robustness suite should
stress-test before we freeze anything.



## Running



Each script prints its own results; nothing else is written.

## Data

- Prices: Yahoo Finance daily adjusted closes (`yfinance`), downloaded once and
  cached as `prices_full.parquet` (2019 onward -- the extra history is only
  there so the first momentum computation has its full 12-month lookback).
  Nifty 50 for the benchmark prints comes from the same source (^NSEI).
- Universe: union of the official Nifty 100, Midcap 100 and Smallcap 100
  constituent lists from niftyindices.com (cached in
  `universe_official.txt`, 300 names).

## Known issues / next steps

- Accounting is still weight-space (fractional shares, no trade log) -- the
  submission needs share-level books, per-trade statistics, accuracy and
  gain-loss numbers.
- Nothing has been stress-tested yet: parameter sensitivity, costs, regimes,
  significance. The next version is the submission, so that is where all of
  this lands.

# BMOM 12-1 - Buffered Momentum Portfolio

Our submission for **Round 2 of the Finesse x Citadel Portfolio Challenge**
(Finesse - Finance Club, SJMSOM, IIT Bombay).

**Team:** `Anti Buldhana Gang` - `Raghav Fulara(Team Leader)`, `Mayuresh Thengzode`, `Nirmit Bangar`

We manage a ₹1 crore, max-10-stock, long-only portfolio over
**1 Jan 2021 - 31 Dec 2025** (with an out-of-sample check on Jan-Jun 2026),
choosing from the Nifty 100, Midcap 100 and Smallcap 100 universes.

## Results at a glance

| | Portfolio | Nifty 50 |
|------|------|------|
| Total Net PNL | **₹15.34 Cr** | ₹0.86 Cr |
| Final value (from ₹1 Cr) | ₹16.34 Cr | - |
| Total return | 1533.8% | 86.4% |
| Annualised return | 74.9% | 13.3% |
| Max drawdown | -30.9% | -17.2% |
| Sharpe (0% risk-free) | 2.74 | 0.95 |
| Accuracy / Gain-to-Loss | 69.4% / 1.58 | - |
| Trades | 181 | - |

Out-of-sample (Jan-Jun 2026, frozen rules): **+21.8%** while the Nifty 50
fell **-8.7%**.

## The strategy in one paragraph

Every quarter we score all ~300 eligible stocks by **12-1 momentum** - the
252-day return skipping the most recent 21 days (the skip avoids India's
one-month reversal, which we kept getting hurt by in earlier versions).
Only positive-momentum stocks qualify. We hold the top names, but with a
**rank buffer**: a position is kept while it stays inside the top 20 and
replaced only once it drops out - this roughly halves turnover and lets
long trends compound. Positions are sized by **inverse 60-day volatility**
(capped at 20% each), and a **30% stop-loss** versus the last review price
exits a broken name to cash. Every trade pays 0.1%.

## Repo contents

| File | What it is |
|------|------|
| `final_strategy.py` | Everything: data download, strategy, backtester, metrics, stress tests, Excel export |
| `prices_full.parquet` | Cached adjusted daily closes (2019-2026, 300 symbols) - committed so the exact numbers reproduce |
| `REPORT.md` | The 5-6 page write-up (also submitted as PDF) |
| `portfolio_performance.png` | NAV vs benchmarks, drawdown, monthly heatmap, key metrics |
| `submission_summary.xlsx` | Summary metrics, quarterly holdings, full trade log, daily NAV |
| `daily_nav.csv`, `trade_history.csv`, `backtest_results.json` | Raw outputs |
| `stress_*.png/.csv/.json` | Stress-test evidence (sensitivity, Monte Carlo, permutation, crisis replay) |
| `iterations/` | Earlier strategy versions (v1-v3) kept for transparency - each file's docstring says what changed and why |

## Running it

```bash
pip install yfinance pandas numpy matplotlib seaborn scipy openpyxl pyarrow
python final_strategy.py
```

Takes about two minutes with the cached price file. Delete
`prices_full.parquet` first if you'd rather pull fresh data from Yahoo
Finance (numbers may then differ slightly if any historical prices have
been revised).

## A few honest notes

- The universe is the **current** constituent list of the three indices, so
  there is some survivorship bias we can't avoid without point-in-time
  membership data. Restricting to stocks listed before mid-2019 still gives
  roughly ₹10-11 Cr of PNL, so the result doesn't hinge on late entrants.
- We did look for the biggest holes: the Adani-Hindenburg window cost the
  book -20.4% (we held several group names going into it - the stop-loss
  and the April review got us out), and calendar 2025 was more or less
  flat (+10.8%). Both are discussed in the report.

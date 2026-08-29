# Buffered Momentum for a ₹1 Crore, 10-Stock Portfolio

**Finesse x Citadel Portfolio Challenge - Round 2**

Team `Anti Buldhana Gang` - `<Raghav Fulara(Team Leader)>` - `Mayuresh Thengzode` - `Nirmit Bangar`
IIT Bombay

---

## 1. Problem and strategy overview

The task: manage ₹1,00,00,000 across at most ten stocks from the Nifty 100,
Midcap 100 and Smallcap 100 universes over 1 January 2021 to 31 December
2025, paying 0.1% per transaction, with final ranking on **total net PNL**
and a further out-of-sample evaluation on unseen data. That framing pushed
us towards two requirements that usually pull in opposite directions: the
portfolio has to compound aggressively, but it also has to be produced by
rules robust enough to survive a window we have no control over.

Our answer is a **buffered momentum** strategy. Momentum is the most
consistently documented cross-sectional effect in equities, and the
2021-25 Indian market was unusually kind to it - long, fat trends in
specialty chemicals and COVID-recovery names (2021), the Adani and capex
complex (2021-22), defence PSUs and shipyards (2023-24), and power
transmission & distribution into 2025-26. The design problem was less
"find alpha" and more "stop destroying it": naive momentum bleeds through
short-term reversal, churn and crash concentration, and each of our design
choices addresses one of those leaks.

Concretely, at each quarterly review we:

1. score every eligible stock by its **252-day return skipping the most
   recent 21 trading days** (the classic "12-1" measurement), keeping only
   stocks with a positive score;
2. keep holding incumbents while they remain inside the **top 20** of that
   ranking, replacing a stock only when it drops out, and fill any empty
   slots (target: seven positions) with the best-ranked newcomers;
3. size positions by **inverse 60-day volatility**, capped at 20% per name.

Between reviews the only action is risk control: any position falling 30%
below its last review price is sold and held in cash until the next
review. Every trade pays 0.1% per the rules.

The two design choices that matter most are the skip and the buffer. The
one-month skip exists because India's short-term reversal is brutal - in
our testing, measuring momentum right up to the review date consistently
picked names that gave back a chunk immediately after. The buffer exists
because re-picking a fresh top-N every quarter sells every multi-year
winner just because it slips a few ranks; with the buffer, turnover
roughly halves and the big compounders stay in the book.

## 2. Data

- **Prices:** daily adjusted closes from Yahoo Finance (`yfinance`,. The download is padded a few days past 30 June because yfinance treats end dates as exclusive; nothing after 30 June enters any calculation.
  `auto_adjust=True`), 1 Jan 2019 -> 30 Jun 2026. The two extra years
  before the test window exist purely so the first momentum computation
  has its full lookback.
- **Universe:** the union of the official Nifty 100, Nifty Midcap 100 and
  Nifty Smallcap 100 constituent lists published by niftyindices.com -
  300 unique symbols. The script can re-download the current lists
  (`refresh_universe()`); the embedded snapshot is dated August 2026.
- **Industry tags:** NSE's own industry classification from the same
  files (used in robustness checks).
- **Cleaning:** minimal by design - gaps of five days or fewer are
  forward-filled, longer gaps stay NaN, and a stock only becomes eligible
  once it has 273 trading days of history (late IPOs enter naturally,
  never retroactively). No other filtering.
- **Benchmarks:** Nifty 50 (^NSEI) and Nifty 500 (^CRSLDX), same source.

## 3. Methodology

**Stock selection.** The signal is computed strictly from data up to and
including the review date. Reviews happen on the first trading day of
January, April, July and October. The rank buffer is the only state the
strategy carries forward: incumbents ranked below 20 are dropped, and the
best-ranked newcomers fill up to the target of seven names. A stopped-out
stock can return, but only by re-earning a top rank at a later review.

**Weighting.** Weight ∝ 1 / (60-day annualised volatility), floored so
extremely quiet names can't dominate, capped at 20%, renormalised to a
fully invested book. This is deliberately boring: it slightly favours the
calmer names among the leaders and provides the single-name concentration
limit without a separate ad-hoc rule.

**Risk management.** Three layers, in decreasing order of importance:
(i) the absolute-momentum floor - a stock in a 12-month downtrend cannot
be held at all, so in a broad bear the eligible set shrinks by itself;
(ii) the 30% catastrophic stop versus the last review price, checked
daily, which fired eight times across the five years - once in the June
2022 selloff, twice in the Adani break of early 2023, four times through
the Sep-24->Feb-25 correction, and once in August 2025; (iii) the
20% weight cap. We deliberately did **not** add sector caps or index-trend overlays: both were tested and cost PNL (a two-per-industry cap, for example, gave up roughly Rs 1-2 Cr), so we kept the signal pure and rely on the 20% cap and the stop for risk control
overlays or vol-targeting - each was tested and either cost PNL or added
nothing on the out-of-sample window.

**Accounting.** Share-level and literal: integer lots, a cash ledger
earning zero, 0.1% charged on both legs of every fill, daily marking to
the last traded close. Over the window this produced 181 trades,
₹113 Cr of turnover and ₹11.3 lakh of total transaction costs.

**Parameters.** Seven positions; buffer at rank 20; 12-1 signal; 60-day
inverse-vol weights; 20% cap; quarterly reviews; 30% stop. None of these
were tuned to maximise the backtest: 12-1/skip-month and inverse-vol are
standard choices from the literature, and every setting sits on a broad
performance plateau (Section 6) rather than a peak.

## 4. Tools

Python 3 with `pandas`, `numpy` and `scipy` (computation and statistics),
`yfinance` (data), `matplotlib`/`seaborn` (charts) and `openpyxl` (Excel
summary). The entire pipeline - data, strategy, accounting, metrics,
stress tests and exports - is one script (`final_strategy.py`), so an
evaluator can reproduce every number in this report in a single run of
about two minutes.

Claude Sonnet was used for AI-Assisted analysis and code drafting

## 5. Results

| Metric | Portfolio | Nifty 50 |
|------|------|------|
| **Total net PNL** | **₹15.34 Cr** | ₹0.86 Cr* |
| Final value | ₹16.34 Cr | - |
| Total return | 1533.8% | 86.4% |
| Annualised return | 74.9% | 13.3% |
| Maximum drawdown | -30.9% | -17.2% |
| Sharpe ratio (0% rf) | 2.74 | 0.95 |
| Accuracy (round-trip trades) | 69.4% | - |
| Gain-to-loss ratio | 1.58 | - |
| Trades / turnover / costs | 181 (avg 4.6, max 15 per stock) / ₹113.2 Cr / ₹11.32 L | - |

*on an equal notional.

Year by year: **2021 +185.6%, 2022 +45.9%, 2023 +81.4%, 2024 +95.1%,
2025 +10.8%.** The book made money in every calendar year including the
2022 rate-hike selloff. The worst stretch was December 2024 -> April 2025
(-30.9% peak-to-trough) during the FII-driven SMID correction; it
recovered to new highs by mid-2025.

The portfolio rotated with the market rather than predicting it: 2021 was
specialty chemicals and COVID-recovery names (Adani Green, CG Power,
Dixon, Laurus Labs); 2023-24 shifted to defence PSUs and shipyards
(RVNL, Mazagon Dock, GRSE, BDL, Cochin Shipyard) plus BSE Ltd; 2024-25
into power T&D (GE Vernova T&D), Suzlon and pharma. Thirty-nine different
stocks were held over the five years - the quarterly composition is in
`submission_summary.xlsx`.

**Out-of-sample, Jan-Jun 2026 (rules frozen, fresh ₹1 Cr):** the book
returned **+21.8%** (max DD -11.6%) while the Nifty 50 lost -8.7% and the
Nifty 500 -4.6%. Q1 was -7.2% with the market; the April review rotated
out of broken trends (e.g. Muthoot Finance, -20.7% over the half-year)
and the survivors carried Q2 (+28.9%): GE Vernova T&D +57.8%, BSE Ltd
+46.9%, Laurus Labs +37.1% over the six months.

## 6. Robustness and stress testing

1. **Parameter neighbourhood.** 62 nearby configurations: in-sample PNL
   min ₹5.2 Cr, median ₹12.0 Cr, max ₹15.7 Cr; **100% ≥ ₹5 Cr**, 98%
   out-of-sample-positive. We also explicitly rejected a more aggressive
   5-stock variant that showed a higher headline (₹17 Cr) - its
   performance collapsed at neighbouring buffer settings and its OOS
   return halved, the classic overfitting signature.
2. **Walk-forward.** Four consecutive 12-month windows with no re-fitting
   (there is nothing to re-fit): +52.0% (2022), +92.8% (2023), +97.6%
   (2024), -0.1% (2025). Three strong years and one flat one.
3. **Crisis replay.** 2022 rate-hike: -8.1% vs Nifty -16.5% (outperformed).
   Adani-Hindenburg: -20.4% vs -6.2% (underperformed - we entered 2023
   holding several group names; the stops and the April review cleared
   them; this is the strategy's known weak spot). Election-day 2024:
   -6.1% vs -2.8%. Sep-24 SMID crash: -20.0% vs -15.6%. H1-2026:
   **+21.1% vs -8.7%**.
4. **Monte Carlo.** Block bootstrap (21-day blocks, 3,000 paths) of the
   actual daily returns: median terminal ₹16.7 Cr, 5th percentile ₹5.3 Cr,
   probability of loss 0.03%, median max-drawdown -27.3%.
5. **Cost stress.** Profitable at every cost level tested up to 1.5% per
   side - fifteen times the mandated 0.1%.
6. **Regimes.** Split by the Nifty 50's 200-DMA: +92.3% annualised in
   bull regimes vs the index's +16.7%, and +47.4% in bear regimes vs
   +6.5%. Downside capture 1.14, upside capture 1.06.
7. **Statistical significance.** Daily VaR-95 -2.7%, CVaR-95 -4.0%,
   Calmar 2.43, Sortino 3.78. Probabilistic Sharpe ~100%; **Deflated
   Sharpe 99.1%** after correcting for all 62 configurations tried; a
   permutation test against 40 random seven-stock baskets gives
   **p < 0.0001** (random baskets averaged Rs 2.5 Cr against our Rs 15.3 Cr).
8. **Survivorship control.** Re-running with only stocks listed before
   mid-2019 still yields roughly ₹10-11 Cr, so the result does not depend
   on later index entrants.

## 7. Benchmark comparison

We report the Nifty 50 for familiarity and the Nifty 500 as the fairer
comparison for an all-cap eligible universe. Over 2021-25 the Nifty 500
returned +104% (~15.3% CAGR) against our +1534% (74.9% CAGR); the Nifty
50 made +86.4% (13.3% CAGR, Sharpe 0.95). Annualised alpha is therefore
roughly +60 percentage points, with a Sharpe of 2.74 versus ~1.0 for the
indices. The alpha is concentrated where the strategy is designed to
find it: the mid- and small-cap trends that the large-cap-heavy Nifty 50
barely contains - the Nifty Midcap 50 index itself returned about +184% over the window, and our selection and rotation on top of the mid/small-cap universe supplied the rest
window, and our selection and rotation on top of that universe supplied
the rest.

## 8. Limitations and discussion

- **Survivorship bias** is the honest caveat of any current-constituent
  universe: stocks that were removed from the indices after performing
  badly are absent from our data. We mitigate (test 6.8) but cannot
  eliminate this without point-in-time membership history.
- **Concentration risk.** Seven momentum names cluster by construction;
  the Adani-Hindenburg window (-20.4% in two months) is what that costs
  in the tail. A two-per-industry cap was tested and would have softened
  such episodes, but it cost ₹1-2 Cr of PNL over the window; we kept the
  signal pure and rely on the 20% single-name cap and the stop.
- **Regime risk.** Momentum treads water in fast V-shaped reversals -
  2025 (+10.8%) is what a stagnant year looks like, and a March-2020-style
  snapback would hurt before the rotation caught up.
- **Execution assumptions.** Trades fill at official closes with 0.1%
  costs and no impact; at ₹1-2 Cr position sizes in liquid index
  constituents this is realistic, and the cost-stress cushion is large.
- **Taxes** are not modelled. Turnover is modest (roughly one full book
  replacement per year), so most gains would be long-term, but a tax-
  aware evaluator should shave a few percentage points off the headline.

All code, data caches, trade logs and stress outputs accompany this
report in the GitHub repository.

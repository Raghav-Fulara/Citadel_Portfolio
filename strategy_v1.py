"""
strategy v1 - first working version of the momentum portfolio
=============================================================
The design, end to end:

  universe  : union of the official Nifty 100 + Midcap 100 + Smallcap
              100 constituent lists (niftyindices.com, cached in
              universe_official.txt, 300 names). Momentum needs
              breadth -- the big Indian trends of this cycle lived
              across the mid/smallcap space, not just the top 100.
  signal    : 12-1 momentum: the 252-day return skipping the most
              recent 21 trading days. The skip-month convention
              exists because last month's strongest names tend to
              reverse; measuring up to yesterday buys those spikes.
  screen    : only positive scores are eligible (absolute momentum).
  selection : fresh top 8 by signal at every review.
  weights   : inverse 60-day volatility, 20% cap per name.
  cadence   : quarterly review (first trading day of Jan/Apr/Jul/Oct).
  risk      : -15% stop-loss vs the review price, checked daily.
  costs     : 0.1% per transaction on both legs.

Result (2021-25): Rs 8.6 Cr PNL on Rs 1 Cr (CAGR 57%, MDD -28%,
Sharpe 2.5). H1-2026 check: +16.6% -- a good sign for the unseen
stress window.

Working notes:
  * the fresh top-8 re-pick every quarter churns the book -- a stock
    slipping from rank 3 to rank 9 gets sold for no real reason
  * the -15% stop fires on ordinary momentum dips, not just breakage
  * accounting is weight-space (fractional shares, no trade log) --
    the submission needs share-level books and trade statistics
  * nothing has been stress-tested yet: parameters, costs, regimes
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

CAPITAL = 10_000_000
COST = 0.001

# ---------------------------------------------------------------
# everything that defines the strategy lives here
# ---------------------------------------------------------------
CONFIG = dict(
    lookbacks   = (252,),              # pure 12-1 momentum
    mom_weights = (1.0,),
    skip        = 21,                  # skip the most recent month
    n_stocks    = 8,
    vol_window  = 60,
    max_w       = 0.20,
    keep_rank   = None,                # None = fresh top-N every review
    stop_loss   = -0.15,
)

# universe: official constituents of all three indices
# (niftyindices.com; cached in universe_official.txt)
def load_universe():
    if os.path.exists('universe_official.txt'):
        return open('universe_official.txt').read().split()
    import io, requests
    hdrs = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.niftyindices.com/'}
    out = set()
    for idx in ['nifty100', 'niftymidcap100', 'niftysmallcap100']:
        r = requests.get(f'https://www.niftyindices.com/IndexConstituent/ind_{idx}list.csv',
                         headers=hdrs, timeout=20)
        if r.status_code == 200 and 'Symbol' in r.text[:800]:
            out |= set(pd.read_csv(io.StringIO(r.text))['Symbol'])
    return sorted(out)


def load_prices():
    syms = [s + '.NS' for s in load_universe()]
    if os.path.exists('prices_full.parquet'):
        px = pd.read_parquet('prices_full.parquet')
        px.index = pd.to_datetime(px.index)
        return px[[c for c in syms if c in px.columns]]
    raw = yf.download(syms, start='2019-01-01', end='2026-07-07',
                      auto_adjust=True, progress=False)
    close = raw.xs('Close', axis=1, level='Price')
    close = close.loc[:, ~close.columns.duplicated()]
    close.ffill(limit=5).to_parquet('prices_full.parquet')
    return close


def compute_momentum(hist):
    """Weighted multi-lookback momentum, skipping CONFIG['skip'] recent days."""
    skip = CONFIG['skip']
    if len(hist) < max(CONFIG['lookbacks']) + skip + 5:
        return pd.Series(dtype=float)
    p_now = hist.iloc[-1 - skip]
    score = pd.Series(0.0, index=hist.columns)
    for w, lb in zip(CONFIG['mom_weights'], CONFIG['lookbacks']):
        score = score.add(w * (p_now / hist.iloc[-1 - skip - lb] - 1),
                          fill_value=0.0)
    return score


def select_portfolio(close, t, state):
    """Rank -> positive screen -> (buffered) top-N -> inverse-vol weights."""
    hist = close.loc[:t]
    score = compute_momentum(hist)
    score = score[score > 0].dropna().sort_values(ascending=False)
    if score.empty:
        return {}

    keep_rank = CONFIG['keep_rank']
    if keep_rank:                                   # rank buffer on: keep incumbents
        rank = {s: k for k, s in enumerate(score.index)}
        picked = [s for s in state['current'] if s in rank and rank[s] < keep_rank]
    else:                                           # fresh re-pick (v1/v2)
        picked = []
    for s in score.index:
        if len(picked) >= CONFIG['n_stocks']:
            break
        if s not in picked:
            picked.append(s)
    state['current'] = picked

    sel = score.reindex(picked).dropna()
    if sel.empty:
        return {}
    vol = hist.iloc[-CONFIG['vol_window']:].pct_change().std() * np.sqrt(252)
    vol = vol.reindex(sel.index)
    inv = 1 / vol.clip(lower=0.05)
    w = (inv / inv.sum()).clip(upper=CONFIG['max_w'])
    return (w / w.sum()).to_dict()


def run_backtest(close, start, end):
    rets = close.pct_change(fill_method=None)
    idx = close.loc[start:end].index
    full = close.index
    pos0 = full.get_loc(idx[0])
    reb, prev = set(), None
    for d in idx:
        key = (d.year, (d.month - 1) // 3)
        if prev is None or key != prev:
            reb.add(d)
        prev = key

    state = {'current': []}
    w = pd.Series(dtype=float)
    nav, navs, dates = CAPITAL, [], []
    review_px = {}
    for i in range(pos0, pos0 + len(idx)):
        t = full[i]
        if len(w):
            r = rets.iloc[i].reindex(w.index).fillna(0.0)
            g = float((w * r).sum())
            nav *= 1 + g
            w = (w * (1 + r)) / (1 + g)
        stop = CONFIG['stop_loss']
        if stop is not None and len(w):
            tdy = close.iloc[i]
            for s_ in list(w.index):
                rp = review_px.get(s_)
                if rp is not None and not np.isnan(tdy[s_]) and tdy[s_] / rp - 1 <= stop:
                    w = w.drop(labels=s_)
        if t in reb:
            tw = pd.Series(select_portfolio(close, t, state))
            old = w.reindex(tw.index.union(w.index)).fillna(0.0)
            new = tw.reindex(old.index).fillna(0.0)
            nav *= 1 - COST * float((new - old).abs().sum())
            w = new
            review_px = {s_: close.iloc[i][s_] for s_ in w.index}
        navs.append(nav); dates.append(t)

    nav_s = pd.Series(navs, index=dates)
    rp = nav_s.pct_change().dropna()
    yrs = max((dates[-1] - dates[0]).days / 365.25, 0.01)
    tot = nav_s.iloc[-1] / CAPITAL - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    mdd = (nav_s / nav_s.cummax() - 1).min()
    sd = rp.std() * np.sqrt(252)
    print(f"  final Rs {nav_s.iloc[-1]/1e7:5.2f} Cr | PNL Rs {nav_s.iloc[-1]-CAPITAL:,.0f} "
          f"| total {tot*100:7.1f}% | CAGR {cagr*100:5.1f}% | MDD {mdd*100:6.1f}% "
          f"| Sharpe {cagr/sd if sd else 0:.2f}")
    return nav_s


if __name__ == '__main__':
    px = load_prices()
    print('strategy v1 | 2021-25 backtest:')
    run_backtest(px, '2021-01-01', '2025-12-31')
    print('strategy v1 | H1-2026 check:')
    run_backtest(px, '2026-01-01', '2026-06-30')

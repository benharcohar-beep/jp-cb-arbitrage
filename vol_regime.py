"""
Vol regime overlay for the backtest period.

Pulls Nikkei 225 historical closes via yfinance, computes rolling 30-day
realized vol, ranks every trading day into low / mid / high regimes based
on percentile bands. Tags each hedged trade by entry-date regime so we
can see whether the strategy works across regimes or only in one.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")


def pull_nikkei_vol() -> pd.DataFrame:
    """30d realized vol of Nikkei 225, plus a regime label."""
    import yfinance as yf
    n225 = yf.Ticker("^N225")
    hist = n225.history(period="2y", auto_adjust=False)
    if hist.empty:
        return pd.DataFrame()
    close = hist["Close"]
    rets = np.log(close / close.shift(1)).dropna()
    vol_30d = rets.rolling(30).std() * np.sqrt(252) * 100  # %
    df = vol_30d.dropna().reset_index().rename(columns={"Date": "date", "Close": "vol_30d_pct"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
    # Percentile rank
    df["vol_pct_rank"] = df["vol_30d_pct"].rank(pct=True) * 100
    # Regimes
    def regime(p):
        if p < 33: return "low"
        if p < 66: return "mid"
        return "high"
    df["regime"] = df["vol_pct_rank"].apply(regime)
    return df[["date", "vol_30d_pct", "vol_pct_rank", "regime"]]


def tag_trades_with_regime() -> pd.DataFrame:
    """For each hedged trade, attach the vol regime at entry."""
    trades = pd.read_csv(os.path.join(HISTDIR, "hedged_trades.csv"))
    if trades.empty:
        return pd.DataFrame()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.date

    vol = pull_nikkei_vol()
    if vol.empty:
        return trades
    # forward-fill regime onto trade dates
    vol_full = vol.set_index("date")
    trades["entry_vol_pct"] = trades["entry_date"].map(lambda d: _lookup_vol(vol_full, d))
    trades["entry_regime"]  = trades["entry_date"].map(lambda d: _lookup_regime(vol_full, d))
    return trades


def _lookup_vol(vol_idx: pd.DataFrame, d):
    try:
        sub = vol_idx[vol_idx.index <= d]
        if sub.empty:
            return None
        return float(sub.iloc[-1]["vol_30d_pct"])
    except Exception:
        return None


def _lookup_regime(vol_idx: pd.DataFrame, d):
    try:
        sub = vol_idx[vol_idx.index <= d]
        if sub.empty:
            return None
        return str(sub.iloc[-1]["regime"])
    except Exception:
        return None


def regime_summary(tagged: pd.DataFrame) -> pd.DataFrame:
    if tagged.empty or "entry_regime" not in tagged.columns:
        return pd.DataFrame()
    tagged = tagged.dropna(subset=["entry_regime"])
    g = tagged.groupby("entry_regime").agg(
        n_trades=("ric", "size"),
        win_rate_pct=("net_pnl_jpy", lambda s: (s > 0).mean() * 100),
        avg_net_bp=("net_return_bp", "mean"),
        median_net_bp=("net_return_bp", "median"),
        total_pnl_jpy=("net_pnl_jpy", "sum"),
    ).reset_index()
    return g.sort_values("entry_regime")


def run():
    print("Pulling Nikkei 225 vol …")
    vol = pull_nikkei_vol()
    if vol.empty:
        print("  Failed to pull Nikkei vol.")
        return
    vol.to_csv(os.path.join(HISTDIR, "nikkei_vol.csv"), index=False)
    print(f"  {len(vol)} days of vol data ({vol['date'].min()} → {vol['date'].max()})")

    print("Tagging trades with vol regime …")
    tagged = tag_trades_with_regime()
    tagged.to_csv(os.path.join(HISTDIR, "hedged_trades_tagged.csv"), index=False)

    summ = regime_summary(tagged)
    if not summ.empty:
        summ.to_csv(os.path.join(HISTDIR, "regime_summary.csv"), index=False)
        print("\nPerformance by vol regime at trade entry:")
        print(summ.to_string(index=False))

    import shutil
    for f in ("nikkei_vol.csv", "hedged_trades_tagged.csv", "regime_summary.csv"):
        src = os.path.join(HISTDIR, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HISTDIR, f"demo_{f}"))


if __name__ == "__main__":
    run()

"""
Factor decomposition of the paper-trading equity curve.

Regresses the strategy's daily returns against:
  - Nikkei 225 (broad JP equity)
  - USD/JPY (FX regime)
  - VIX (global vol regime)
  - JP 10Y JGB yield change (rates)

If the strategy is *real alpha*, the residual intercept (alpha) is positive
and statistically significant after stripping these betas. If the strategy
is just "long Japan + long vol in disguise," the regression will reveal it.

Outputs:
  factor_decomp.csv      - regression results (alpha, betas, t-stats, R²)
  factor_data.csv        - raw aligned factor + strategy returns
"""

from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")


FACTORS = {
    "nikkei":  "^N225",
    "usdjpy":  "JPY=X",
    "vix":     "^VIX",
    "us10y":   "^TNX",   # US 10y yield as global rates proxy
}


def pull_factor_returns(start: str, end: str) -> pd.DataFrame:
    """Daily log returns for each factor, aligned by date."""
    out = []
    for name, ticker in FACTORS.items():
        try:
            tk = yf.Ticker(ticker)
            h = tk.history(start=start, end=end, auto_adjust=False)
            if h.empty:
                print(f"  {name} ({ticker}): empty")
                continue
            close = h["Close"]
            # For yields use first-difference; for prices use log returns
            if ticker in ("^VIX", "^TNX"):
                ret = close.diff()  # raw change
            else:
                ret = np.log(close / close.shift(1))
            ret = ret.dropna()
            ret_df = ret.reset_index()
            ret_df.columns = ["date", name]
            # ret_df["date"] is already datetime64[ns, tz]; strip tz then take .date()
            try:
                ret_df["date"] = ret_df["date"].dt.tz_convert(None).dt.date
            except (AttributeError, TypeError):
                # Already tz-naive
                ret_df["date"] = pd.to_datetime(ret_df["date"]).dt.date
            out.append(ret_df)
        except Exception as e:
            import traceback
            print(f"  {name} error: {str(e)[:100]}")
            traceback.print_exc()
    if not out:
        return pd.DataFrame()
    # Outer-join all factor frames
    merged = out[0]
    for f in out[1:]:
        merged = merged.merge(f, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def run():
    print("Factor decomposition · paper portfolio vs market factors")
    # Load equity curve (prefer $5M, else default)
    for fname in ("paper_5m_equity.csv", "paper_equity.csv"):
        p = os.path.join(HISTDIR, fname)
        if os.path.exists(p):
            eq = pd.read_csv(p)
            print(f"  using {fname}")
            break
    else:
        print("  No equity curve found; run hedged_backtest.py first.")
        return

    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").reset_index(drop=True)
    # Keep last equity for each date (multiple events per day possible)
    eq_idx = eq.groupby(eq["date"].dt.date)["equity_usd"].last()
    eq_idx.index = pd.to_datetime(eq_idx.index)
    # Resample to daily, forward-fill
    eq_daily = eq_idx.resample("D").last().ffill().dropna()
    strat_ret = np.log(eq_daily / eq_daily.shift(1)).dropna()
    strat_df = strat_ret.reset_index()
    strat_df.columns = ["date", "strategy"]
    strat_df["date"] = pd.to_datetime(strat_df["date"]).dt.date
    print(f"  strategy: {len(strat_df)} daily returns ({strat_df['date'].min()} → {strat_df['date'].max()})")

    print("  pulling factor data via yfinance …")
    start = (eq["date"].min() - timedelta(days=10)).strftime("%Y-%m-%d")
    end   = (eq["date"].max() + timedelta(days=2)).strftime("%Y-%m-%d")
    factors = pull_factor_returns(start, end)
    if factors.empty:
        print("  factor pull failed")
        return
    print(f"  factors: {len(factors)} rows, cols={list(factors.columns)}")

    # Align
    df = strat_df.merge(factors, on="date", how="inner").dropna()
    print(f"  aligned: {len(df)} common rows")
    df.to_csv(os.path.join(HISTDIR, "factor_data.csv"), index=False)

    if len(df) < 30:
        print("  Too few observations for a meaningful regression.")
        return

    # OLS regression
    X = df[list(FACTORS.keys())]
    X = sm.add_constant(X)
    y = df["strategy"]
    model = sm.OLS(y, X).fit()
    print("\n--- OLS regression: strategy ~ factors ---")
    print(model.summary())

    # Parse results
    results = []
    for name in ["const"] + list(FACTORS.keys()):
        try:
            results.append({
                "factor":     name,
                "coefficient": float(model.params[name]),
                "std_error":  float(model.bse[name]),
                "t_stat":     float(model.tvalues[name]),
                "p_value":    float(model.pvalues[name]),
            })
        except Exception:
            continue
    res_df = pd.DataFrame(results)

    # Annualize alpha (constant) — daily * 252
    alpha_daily = float(model.params.get("const", 0))
    alpha_annual_pct = (np.exp(alpha_daily * 252) - 1) * 100

    summary = {
        "n_observations":  int(len(df)),
        "r_squared":       float(model.rsquared),
        "adj_r_squared":   float(model.rsquared_adj),
        "alpha_daily":     alpha_daily,
        "alpha_annual_pct": alpha_annual_pct,
        "f_statistic":     float(model.fvalue),
        "f_pvalue":        float(model.f_pvalue),
    }
    pd.Series(summary).to_csv(os.path.join(HISTDIR, "factor_summary.csv"), header=False)
    res_df.to_csv(os.path.join(HISTDIR, "factor_decomp.csv"), index=False)

    import shutil
    for f in ("factor_decomp.csv", "factor_summary.csv", "factor_data.csv"):
        src = os.path.join(HISTDIR, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HISTDIR, f"demo_{f}"))

    print("\n--- Key numbers ---")
    for k, v in summary.items():
        print(f"  {k:25s} {v:,.4f}" if isinstance(v, float) else f"  {k:25s} {v}")


if __name__ == "__main__":
    run()

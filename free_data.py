"""
Free-source data pipeline — runs without Refinitiv Workspace.

Refreshes the equity (spot, vol, div yield) and JGB curve sides of the
universe using public sources:

  - Yahoo Finance Japan (via yfinance) for TSE equities
  - Japan MoF historical interest rates CSV for JGB yields

This does NOT refresh the CB universe / terms / market prices — those
come from Refinitiv only. Strategy: keep bonds.csv from the latest
Refinitiv pull (rarely changes day-to-day), and use this script to
refresh equities.csv + jgb.csv on Workspace-down days.

Run:  python3 free_data.py
"""

from __future__ import annotations

import io
import os
import time
import traceback
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

PROJ = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# JGB curve from MoF
# ---------------------------------------------------------------------------

MOF_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"


def pull_jgb_from_mof() -> pd.DataFrame:
    """
    Pull the latest row of MoF's daily JGB yield CSV. Public, no auth.
    Columns are tenor labels (1Y..40Y); rows are dates.
    """
    try:
        r = httpx.get(MOF_URL, timeout=30.0, follow_redirects=True)
        r.raise_for_status()
    except Exception:
        traceback.print_exc()
        return pd.DataFrame()

    # MoF CSV has a banner row above the real header — skip it.
    raw = pd.read_csv(io.BytesIO(r.content), encoding="latin-1", skiprows=1)
    raw.columns = [c.strip() for c in raw.columns]
    # Drop header rows (file has multi-line header sometimes); keep rows where
    # first column parses as a date.
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).sort_values(date_col)
    if raw.empty:
        return pd.DataFrame()

    last = raw.iloc[-1]
    # Map MoF tenor labels -> our tenor strings + RIC stand-in
    tenor_map = {
        "2Y": "2y", "3Y": "3y", "5Y": "5y", "7Y": "7y",
        "10Y": "10y", "20Y": "20y", "30Y": "30y",
    }
    rows = []
    for col, tenor in tenor_map.items():
        if col in raw.columns:
            v = last[col]
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if pd.notna(v):
                rows.append({
                    "Instrument": f"JP{col}T=MOF",
                    "yield_pct": v,
                    "tenor": tenor,
                    "date": last[date_col].date().isoformat(),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Equity data via yfinance
# ---------------------------------------------------------------------------

def _hist_vol(prices: pd.Series, lookback: int) -> float:
    """Annualized log-return vol over the last `lookback` trading days."""
    if prices is None or len(prices) < lookback + 1:
        return float("nan")
    rets = np.log(prices / prices.shift(1)).dropna().iloc[-lookback:]
    if len(rets) < 5:
        return float("nan")
    return float(rets.std() * np.sqrt(252))


def pull_equity_yfinance(rics: list[str]) -> pd.DataFrame:
    """
    For each TSE RIC like '4385.T', yfinance accepts the same '4385.T' form.
    Pull 6 months of daily closes → spot, 60-day vol, 90-day vol, dividend yield.
    """
    rows = []
    for ric in sorted({r for r in rics if r and isinstance(r, str)}):
        try:
            tk = yf.Ticker(ric)
            hist = tk.history(period="6mo", auto_adjust=False)
            if hist.empty:
                continue
            close = hist["Close"]
            spot = float(close.iloc[-1])
            v60 = _hist_vol(close, 60) * 100.0
            v90 = _hist_vol(close, 90) * 100.0

            info = {}
            try:
                info = tk.info or {}
            except Exception:
                pass
            div_y = info.get("dividendYield")
            if div_y is not None and div_y > 1:  # some yfinance versions return %
                div_y = div_y / 100.0
            div_y_pct = (div_y * 100.0) if div_y else 0.0

            rows.append({
                "Instrument": ric,
                "Price Close": spot,
                "Company Name": info.get("longName", ""),
                "Volatility - 60 days": v60,
                "Volatility - 90 days": v90,
                "Dividend yield": div_y_pct,
                "Average Daily Volume - 30 Days": float(hist["Volume"].tail(30).mean() or 0),
                "Company Market Cap": info.get("marketCap"),
                "Issuer Rating": "",  # yfinance doesn't carry credit ratings
            })
        except Exception as e:
            print(f"  {ric}: skip ({str(e)[:60]})")
        time.sleep(0.4)  # be polite to Yahoo
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/2] JGB curve from MoF …")
    jgb = pull_jgb_from_mof()
    if jgb.empty:
        print("  WARNING: MoF pull failed; jgb.csv NOT updated")
    else:
        jgb.to_csv(os.path.join(PROJ, "jgb.csv"), index=False)
        print(f"  saved jgb.csv  ({len(jgb)} tenors)")
        print(jgb[["tenor", "yield_pct"]].to_string(index=False))

    print("\n[2/2] Equities from Yahoo Finance Japan …")
    bonds = pd.read_csv(os.path.join(PROJ, "bonds.csv"))
    rics = bonds["underlying_ric"].dropna().unique().tolist()
    print(f"  fetching {len(rics)} tickers …")
    eq = pull_equity_yfinance(rics)
    if eq.empty:
        print("  WARNING: no equity data")
    else:
        eq.to_csv(os.path.join(PROJ, "equities.csv"), index=False)
        print(f"  saved equities.csv  ({len(eq)} rows)")

    print("\nFree-source refresh complete.")
    print("Note: bonds.csv (CB terms/prices) was NOT refreshed — Refinitiv only.")


if __name__ == "__main__":
    main()

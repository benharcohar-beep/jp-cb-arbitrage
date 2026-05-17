"""
Backtest engine for the JP CB arb model.

Pipeline:
  1. pull_history()  — for each bond in current universe, pull 1y of
                       daily mid/bid/ask via rd.get_history; pull 1y
                       of underlying daily closes via yfinance.
  2. price_history() — re-run the pricer on each historical date with
                       that day's spot, trailing-60d realized vol, and
                       a current-day-static credit spread / JGB curve.
                       (See caveats below.)
  3. signals()       — for each bond, find dates where cheap% crossed a
                       threshold; compute forward returns at 30/60/90d.
  4. summary()       — aggregate stats: hit rate, average forward return
                       conditional on signal strength.

Caveats (don't oversell the result):
  - Credit spread is held constant at today's rating-based level. Real
    spreads move; this biases past cheap signals.
  - JGB curve is held at today's level. Small effect for short tenors.
  - Bond "price" is dealer mid; many JP CBs are illiquid and dealer
    quotes lag actual fair value. Cheap signals on illiquid days should
    be treated with skepticism.
  - We aren't modelling the short-equity hedge — pure bond P&L. A real
    arb strategy delta-hedges; bond returns alone overstate risk.

For a publishable result, fix the above. This module is a "is the model
directionally useful" check, not a production backtest.
"""

from __future__ import annotations

import os
import time
import traceback
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")
os.makedirs(HISTDIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) Historical data pull
# ---------------------------------------------------------------------------

def _open_session():
    import refinitiv.data as rd
    try:
        rd.close_session()
    except Exception:
        pass
    rd.open_session("desktop.workspace")


def pull_bond_history_refinitiv(
    rics: list[str], start: str, end: str, chunk: int = 10,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Daily mid/bid/ask for each RIC. Returns long-form: ric, date, mid, bid, ask.
    On Refinitiv session drop, automatically reconnects and retries up to max_retries times.
    """
    import refinitiv.data as rd
    _open_session()
    out = []
    n_done = 0
    try:
        for i in range(0, len(rics), chunk):
            sub = rics[i:i + chunk]
            for ric in sub:
                attempts = 0
                while attempts < max_retries:
                    try:
                        df = rd.get_history(
                            universe=[ric],
                            fields=["TR.MidPrice", "TR.BIDPRICE", "TR.ASKPRICE"],
                            start=start, end=end, interval="daily",
                        )
                        if df is None or df.empty:
                            break  # no data; not a session error
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(-1)
                        df = df.reset_index().rename(columns={"Date": "date"})
                        df["ric"] = ric
                        out.append(df)
                        n_done += 1
                        if n_done % 10 == 0:
                            print(f"  …{n_done}/{len(rics)} bonds pulled", flush=True)
                        break  # success
                    except Exception as e:
                        msg = str(e)[:120]
                        if any(k in msg for k in ("Session is not opened", "timed out",
                                                   "Connection refused", "no proxy",
                                                   "ConnectError")):
                            attempts += 1
                            print(f"  {ric}: session error (attempt {attempts}/{max_retries}); reconnecting …", flush=True)
                            time.sleep(2.0 * attempts)
                            try:
                                _open_session()
                            except Exception as e2:
                                print(f"    reconnect failed: {str(e2)[:80]}")
                                break
                        else:
                            print(f"  {ric}: {msg}")
                            break
                time.sleep(0.3)
    finally:
        try:
            rd.close_session()
        except Exception:
            pass
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def pull_equity_history_yfinance(rics: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    out = []
    for r in sorted({r for r in rics if r}):
        try:
            tk = yf.Ticker(r)
            h = tk.history(start=start, end=end, auto_adjust=False)
            if h.empty:
                continue
            h = h[["Close"]].reset_index().rename(columns={"Date": "date", "Close": "spot"})
            h["date"] = h["date"].dt.tz_localize(None).dt.date
            h["underlying_ric"] = r
            out.append(h)
        except Exception as e:
            print(f"  {r}: {str(e)[:80]}")
        time.sleep(0.3)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def pull_history():
    bonds = pd.read_csv(os.path.join(PROJ, "bonds.csv"))
    bonds = bonds.dropna(subset=["RIC", "underlying_ric"]).copy()
    rics = bonds["RIC"].astype(str).tolist()
    und  = bonds["underlying_ric"].astype(str).unique().tolist()

    today = date.today()
    start = (today - timedelta(days=400)).isoformat()
    end   = today.isoformat()

    print(f"[1/2] Bond history (Refinitiv) — {len(rics)} RICs, {start} → {end}")
    bh = pull_bond_history_refinitiv(rics, start, end)
    bh.to_csv(os.path.join(HISTDIR, "bonds_hist.csv"), index=False)
    print(f"  saved bonds_hist.csv  ({len(bh)} rows)")

    print(f"[2/2] Equity history (yfinance) — {len(und)} tickers")
    eh = pull_equity_history_yfinance(und, start, end)
    eh.to_csv(os.path.join(HISTDIR, "equities_hist.csv"), index=False)
    print(f"  saved equities_hist.csv  ({len(eh)} rows)")


# ---------------------------------------------------------------------------
# 2) Walk-forward pricing
# ---------------------------------------------------------------------------

def _trailing_vol(prices: np.ndarray, lookback: int = 60) -> float:
    if len(prices) < lookback + 1:
        return float("nan")
    rets = np.log(prices[1:] / prices[:-1])[-lookback:]
    if len(rets) < 5:
        return float("nan")
    return float(np.std(rets) * np.sqrt(252))


def price_history(min_obs: int = 30) -> pd.DataFrame:
    """
    For each (bond, date) in history with non-null mid + valid spot,
    compute model price and cheap%. Returns long-form panel.
    """
    from datetime import datetime as dt
    from pricer import ConvertibleBond, MarketData, price_cb
    from real_data import load_jgb_curve, rf_for_tenor
    from credit import spread_for as spread_for_rating
    from mc_pricer import default_reset_dates, price_cb_mc_with_reset

    bonds_meta = pd.read_csv(os.path.join(PROJ, "bonds.csv"))
    equities_meta = pd.read_csv(os.path.join(PROJ, "equities.csv")).set_index("Instrument")

    bh = pd.read_csv(os.path.join(HISTDIR, "bonds_hist.csv"))
    eh = pd.read_csv(os.path.join(HISTDIR, "equities_hist.csv"))
    bh["date"] = pd.to_datetime(bh["date"]).dt.date
    eh["date"] = pd.to_datetime(eh["date"]).dt.date

    curve = load_jgb_curve()  # held constant

    rows = []
    for ric, bond_grp in bh.groupby("ric"):
        meta = bonds_meta[bonds_meta["RIC"] == ric]
        if meta.empty:
            continue
        m = meta.iloc[0]
        und = m.get("underlying_ric")
        cp_yen = m.get("ConversionPrice")
        mat = pd.to_datetime(m["MaturityDate"]).date()
        iss = pd.to_datetime(m["IssueDate"]).date() if pd.notna(m.get("IssueDate")) else None
        coupon = float(m.get("Coupon Rate") or 0.0) / 100.0
        coupon_freq = int(m.get("Coupon Frequency") or 2) if pd.notna(m.get("Coupon Frequency")) else 2
        likely_reset = bool(m.get("likely_has_reset", False))

        if pd.isna(und) or pd.isna(cp_yen) or cp_yen <= 0:
            continue

        # Equity history for this underlying
        eq = eh[eh["underlying_ric"] == und].sort_values("date").reset_index(drop=True)
        if eq.empty or len(eq) < min_obs + 60:
            continue

        # Rating + spread (constant in backtest)
        rating = ""
        if und in equities_meta.index:
            rating = equities_meta.loc[und].get("Issuer Rating", "")
        spread = spread_for_rating(rating)

        # For each bond-history date, pair with nearest equity row & price
        bond_grp = bond_grp.sort_values("date")
        eq_prices = eq["spot"].to_numpy()
        eq_dates  = eq["date"].to_list()

        for _, b in bond_grp.iterrows():
            d = b["date"]
            mid = b.get("Mid Price")
            bid = b.get("Bid Price")
            ask = b.get("Ask Price")
            if pd.isna(mid) or mid <= 0:
                continue
            if d <= (iss or d) or d >= mat:
                continue

            # find equity price & trailing vol on or before this date
            try:
                k = next(i for i in range(len(eq_dates) - 1, -1, -1) if eq_dates[i] <= d)
            except StopIteration:
                continue
            spot = float(eq_prices[k])
            sigma = _trailing_vol(eq_prices[: k + 1], 60)
            if not np.isfinite(spot) or not np.isfinite(sigma) or sigma <= 0:
                continue

            yrs = (mat - d).days / 365.0
            if yrs <= 0:
                continue
            r = rf_for_tenor(curve, yrs)

            bond = ConvertibleBond(
                isin=str(m.get("ISIN") or ric),
                issuer=str(m.get("IssuerName")),
                underlying_ticker=str(und),
                coupon=coupon, coupon_freq=coupon_freq,
                maturity=mat, issue_date=iss or d,
                notional=100.0, conversion_price=float(cp_yen),
                currency="JPY",
                credit_rating=rating or "NR",
            )
            mkt = MarketData(
                valuation_date=d, spot=spot, sigma=sigma,
                r=r, credit_spread=spread, div_yield=0.0,
            )

            try:
                if likely_reset:
                    resets = default_reset_dates(bond, mkt)
                    if resets:
                        # MC is slow; thin the path count for backtest speed
                        res = price_cb_mc_with_reset(bond, mkt, resets,
                                                     n_paths=1500, n_steps=120)
                        engine = "MC"
                    else:
                        res = price_cb(bond, mkt, n_steps=150)
                        engine = "Tree"
                else:
                    res = price_cb(bond, mkt, n_steps=150)
                    engine = "Tree"
            except Exception:
                continue

            cheap = (res["price"] - mid) / mid * 100.0
            rows.append({
                "ric": ric, "issuer": m["IssuerName"],
                "underlying": und, "date": d,
                "spot": spot, "sigma": sigma,
                "mkt_px": float(mid),
                "bid_px": float(bid) if pd.notna(bid) and bid > 0 else float(mid),
                "ask_px": float(ask) if pd.notna(ask) and ask > 0 else float(mid),
                "model_px": float(res["price"]),
                "cheap_pct": cheap, "engine": engine,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HISTDIR, "panel.csv"), index=False)
    print(f"  panel: {len(df)} bond-day rows  ({df['ric'].nunique()} bonds)")
    return df


# ---------------------------------------------------------------------------
# 3) Signal evaluation
# ---------------------------------------------------------------------------

def signal_returns(panel: pd.DataFrame, threshold: float = 5.0,
                   max_cheap: float = 25.0,
                   horizons: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """For each cheap signal in [threshold, max_cheap], compute realized bond
    return at horizons. The cap excludes the stale-conv-price bonds where
    cheap% is huge but the dealer mid never moves (anomalies).
    """
    panel = panel.sort_values(["ric", "date"]).copy()
    out = []
    for ric, grp in panel.groupby("ric"):
        grp = grp.reset_index(drop=True)
        for i, row in grp.iterrows():
            cp = row["cheap_pct"]
            if cp < threshold or cp > max_cheap:
                continue
            for h in horizons:
                if i + h >= len(grp):
                    continue
                fwd = grp.iloc[i + h]
                bond_ret = (fwd["mkt_px"] - row["mkt_px"]) / row["mkt_px"] * 100.0
                conv_ret = (fwd["cheap_pct"] - row["cheap_pct"])
                out.append({
                    "ric": ric, "issuer": row["issuer"],
                    "signal_date": row["date"], "signal_cheap": row["cheap_pct"],
                    "horizon_d": h, "fwd_date": fwd["date"],
                    "fwd_bond_ret_pct": bond_ret,
                    "fwd_cheap_change_pp": conv_ret,
                })
    return pd.DataFrame(out)


def signal_summary(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    g = returns.groupby("horizon_d").agg(
        n=("ric", "size"),
        avg_fwd_ret=("fwd_bond_ret_pct", "mean"),
        median_fwd_ret=("fwd_bond_ret_pct", "median"),
        hit_rate=("fwd_bond_ret_pct", lambda s: (s > 0).mean() * 100),
        avg_cheap_decay_pp=("fwd_cheap_change_pp", "mean"),
    ).reset_index()
    return g


def signal_summary_by_bucket(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    bins = [5, 10, 15, 25]
    labels = ["5-10%", "10-15%", "15-25%"]
    returns = returns.copy()
    returns["bucket"] = pd.cut(returns["signal_cheap"], bins=bins, labels=labels,
                                include_lowest=True)
    g = returns.groupby(["bucket", "horizon_d"], observed=True).agg(
        n=("ric", "size"),
        avg_fwd_ret=("fwd_bond_ret_pct", "mean"),
        median_fwd_ret=("fwd_bond_ret_pct", "median"),
        hit_rate=("fwd_bond_ret_pct", lambda s: (s > 0).mean() * 100),
    ).reset_index()
    return g


def main():
    pull_history()
    panel = price_history()
    if panel.empty:
        print("Panel empty — abort.")
        return
    rets = signal_returns(panel, threshold=5.0, max_cheap=25.0)
    rets.to_csv(os.path.join(HISTDIR, "signal_returns.csv"), index=False)
    print(f"\nSignal returns: {len(rets)} signal-horizon rows  (5% ≤ cheap% ≤ 25%, anomalies filtered)")
    if not rets.empty:
        summ = signal_summary(rets)
        print("\nAggregate signal performance (5-25% cheap, anomalies filtered):")
        print(summ.to_string(index=False))
        summ.to_csv(os.path.join(HISTDIR, "signal_summary.csv"), index=False)

        bucket_summ = signal_summary_by_bucket(rets)
        print("\nBy cheap-strength bucket:")
        print(bucket_summ.to_string(index=False))
        bucket_summ.to_csv(os.path.join(HISTDIR, "signal_bucket.csv"), index=False)


if __name__ == "__main__":
    main()

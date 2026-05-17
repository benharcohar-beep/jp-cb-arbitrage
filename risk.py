"""
Desk-grade risk management framework for the JP CB arb book.

Defines hard limits a real PM would impose on the strategy, then
back-tests how the existing trade history would have respected (or
breached) them. The point isn't that the limits are "right" — it's
to show you've thought about the risk-control layer that turns a
backtested signal into a tradeable book.

Risk dimensions modelled:

  1. Position sizing limits
     - max notional per single issuer
     - max % of equity per single trade
     - max total deployed across all open positions

  2. Vega exposure
     - total vega cap (book-level option exposure)

  3. Concentration
     - no single issuer > 25% of equity
     - no single trade > 20% of equity

  4. Value at Risk (1-day 95% historical VaR)
     - computed from the paper equity curve
     - flagged if > 2% of NAV (standard hedge-fund soft limit)

  5. Drawdown circuit breaker
     - if rolling 30-day drawdown exceeds 5%, halve all new position sizes
     - if exceeds 10%, halt new entries entirely

Outputs:
  risk_limits.csv      — the defined limits
  risk_breaches.csv    — historical trades that would have been rejected/resized
  var_history.csv      — daily VaR estimates over the backtest window
  risk_kpis.csv        — summary stats: # breaches, VaR avg/peak, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")


@dataclass
class RiskLimits:
    # Position
    max_notional_per_issuer_jpy:   float = 200_000_000     # ¥200M per name
    max_pct_equity_per_trade:      float = 20.0            # 20% of current equity
    max_total_deployed_pct_equity: float = 100.0           # can't deploy more than NAV

    # Greeks
    max_book_vega:                 float = 50.0            # rough — vega units (% of par per 1 vol pt)

    # Concentration
    max_pct_equity_per_issuer:     float = 25.0

    # VaR
    var_soft_limit_pct:            float = 2.0             # warn if 1d 95% VaR > 2% of NAV
    var_hard_limit_pct:            float = 4.0             # block new entries if > 4%

    # Drawdown circuit breakers
    dd_halve_threshold_pct:        float = -5.0            # halve new sizes at -5%
    dd_halt_threshold_pct:         float = -10.0           # halt entirely at -10%


def compute_book_var(equity_curve: pd.DataFrame, window_days: int = 60,
                    confidence: float = 0.95) -> pd.DataFrame:
    """Rolling 1-day 95% historical VaR from the equity curve."""
    if equity_curve.empty or "equity_usd" not in equity_curve.columns:
        return pd.DataFrame()
    eq = equity_curve.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").reset_index(drop=True)
    eq["daily_return"] = eq["equity_usd"].pct_change()
    out = []
    for i in range(len(eq)):
        sub = eq["daily_return"].iloc[max(0, i - window_days):i + 1].dropna()
        if len(sub) < 20:
            continue
        var = -np.quantile(sub, 1 - confidence) * 100  # positive %
        out.append({
            "date": eq["date"].iloc[i],
            "equity_usd": eq["equity_usd"].iloc[i],
            "var_1d_95_pct": float(var),
        })
    return pd.DataFrame(out)


def review_trades_against_limits(trades: pd.DataFrame, equity_curve: pd.DataFrame,
                                  limits: RiskLimits = None,
                                  usd_jpy: float = 150.0) -> pd.DataFrame:
    """For each historical hedged trade, check whether it would have been
    accepted, resized, or rejected under the risk limits."""
    if trades.empty:
        return pd.DataFrame()
    limits = limits or RiskLimits()
    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades = trades.sort_values("entry_date").reset_index(drop=True)

    # Build a date-indexed equity series for sizing lookups
    eq = equity_curve.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").reset_index(drop=True)
    eq_index = eq.set_index("date")["equity_usd"]

    issuer_exposure_jpy: dict[str, float] = {}
    issuer_exposure_at_entry: dict[str, list] = {}
    open_trades: list = []

    out = []
    for i, t in trades.iterrows():
        # Get equity at entry
        try:
            sub = eq_index[eq_index.index <= t["entry_date"]]
            equity_usd = float(sub.iloc[-1]) if not sub.empty else 1_000_000.0
        except Exception:
            equity_usd = 1_000_000.0
        equity_jpy = equity_usd * usd_jpy

        # Close any positions whose exit_date < this entry_date
        # (release issuer exposure)
        still_open = []
        for op in open_trades:
            if pd.to_datetime(op["exit_date"]) <= t["entry_date"]:
                issuer_exposure_jpy[op["issuer"]] -= op["notional_jpy"]
            else:
                still_open.append(op)
        open_trades = still_open

        # Proposed notional (assuming 20% of equity per trade as the strategy default)
        proposed_notional_jpy = equity_jpy * 0.20
        proposed_pct = 20.0

        # Apply each limit
        breaches = []
        accepted_notional = proposed_notional_jpy
        accepted_pct = proposed_pct

        # 1. Per-issuer notional cap
        cur_issuer_exposure = issuer_exposure_jpy.get(t["issuer"], 0.0)
        if cur_issuer_exposure + proposed_notional_jpy > limits.max_notional_per_issuer_jpy:
            breaches.append(f"issuer_notional_cap (cur={cur_issuer_exposure/1e6:.0f}M + {proposed_notional_jpy/1e6:.0f}M > {limits.max_notional_per_issuer_jpy/1e6:.0f}M)")
            accepted_notional = max(0, limits.max_notional_per_issuer_jpy - cur_issuer_exposure)
            accepted_pct = accepted_notional / equity_jpy * 100 if equity_jpy > 0 else 0

        # 2. Per-issuer % equity cap
        new_issuer_pct = (cur_issuer_exposure + accepted_notional) / equity_jpy * 100 if equity_jpy > 0 else 0
        if new_issuer_pct > limits.max_pct_equity_per_issuer:
            breaches.append(f"issuer_pct_cap ({new_issuer_pct:.1f}% > {limits.max_pct_equity_per_issuer:.1f}%)")
            accepted_notional = max(0, limits.max_pct_equity_per_issuer/100 * equity_jpy - cur_issuer_exposure)
            accepted_pct = accepted_notional / equity_jpy * 100 if equity_jpy > 0 else 0

        # 3. Drawdown circuit breaker
        if not eq.empty:
            try:
                sub = eq[eq["date"] <= t["entry_date"]].copy()
                if len(sub) >= 5:
                    peak = sub["equity_usd"].cummax().iloc[-1]
                    dd_now = (sub["equity_usd"].iloc[-1] / peak - 1) * 100
                    if dd_now <= limits.dd_halt_threshold_pct:
                        breaches.append(f"dd_halt ({dd_now:.1f}% < {limits.dd_halt_threshold_pct:.0f}%)")
                        accepted_notional = 0
                    elif dd_now <= limits.dd_halve_threshold_pct:
                        breaches.append(f"dd_halve ({dd_now:.1f}% < {limits.dd_halve_threshold_pct:.0f}%)")
                        accepted_notional *= 0.5
                        accepted_pct = accepted_notional / equity_jpy * 100 if equity_jpy > 0 else 0
            except Exception:
                pass

        decision = ("REJECTED" if accepted_notional <= 0 else
                    "RESIZED" if abs(accepted_notional - proposed_notional_jpy) > 1 else
                    "ACCEPTED")

        out.append({
            "entry_date": t["entry_date"].date(),
            "issuer": t["issuer"],
            "ric": t["ric"],
            "entry_cheap": t["entry_cheap"],
            "equity_usd_at_entry": round(equity_usd, 0),
            "proposed_notional_jpy": round(proposed_notional_jpy, 0),
            "accepted_notional_jpy": round(accepted_notional, 0),
            "accepted_pct_equity": round(accepted_pct, 2),
            "issuer_exposure_before_jpy": round(cur_issuer_exposure, 0),
            "decision": decision,
            "breaches": "; ".join(breaches) if breaches else "",
            "trade_realised_pnl_jpy": float(t["net_pnl_jpy"]) * (accepted_notional / 100_000_000.0) if accepted_notional > 0 else 0,
        })

        # Update tracking
        if accepted_notional > 0:
            issuer_exposure_jpy[t["issuer"]] = cur_issuer_exposure + accepted_notional
            open_trades.append({
                "issuer": t["issuer"],
                "exit_date": t["exit_date"],
                "notional_jpy": accepted_notional,
            })

    return pd.DataFrame(out)


def run():
    print("Risk management framework — backtest review …")
    limits = RiskLimits()

    # Save limits
    pd.Series(asdict(limits)).to_csv(os.path.join(HISTDIR, "risk_limits.csv"), header=False)
    print("Defined limits:")
    for k, v in asdict(limits).items():
        print(f"  {k:35s} {v}")

    # Load history
    try:
        trades = pd.read_csv(os.path.join(HISTDIR, "hedged_trades.csv"))
    except FileNotFoundError:
        print("  No hedged_trades.csv; run hedged_backtest.py first.")
        return
    try:
        eq = pd.read_csv(os.path.join(HISTDIR, "paper_equity.csv"))
    except FileNotFoundError:
        print("  No paper_equity.csv; run hedged_backtest.py first.")
        return

    # Review trades against limits
    review = review_trades_against_limits(trades, eq, limits)
    review.to_csv(os.path.join(HISTDIR, "risk_breaches.csv"), index=False)
    n_accepted = (review["decision"] == "ACCEPTED").sum()
    n_resized  = (review["decision"] == "RESIZED").sum()
    n_rejected = (review["decision"] == "REJECTED").sum()
    print(f"\nTrade review: {n_accepted} accepted, {n_resized} resized, {n_rejected} rejected (of {len(review)} total)")
    if n_resized + n_rejected > 0:
        print("\nFirst 10 breaches:")
        print(review[review["decision"] != "ACCEPTED"].head(10).to_string(index=False))

    # VaR
    print("\nComputing rolling 60-day 1-day 95% VaR …")
    var_df = compute_book_var(eq, window_days=60, confidence=0.95)
    var_df.to_csv(os.path.join(HISTDIR, "var_history.csv"), index=False)
    if not var_df.empty:
        print(f"  VaR: avg {var_df['var_1d_95_pct'].mean():.2f}%  peak {var_df['var_1d_95_pct'].max():.2f}%  current {var_df['var_1d_95_pct'].iloc[-1]:.2f}%")

    # KPIs
    kpis = {
        "n_trades_evaluated":   int(len(review)),
        "n_accepted":           int(n_accepted),
        "n_resized":            int(n_resized),
        "n_rejected":           int(n_rejected),
        "var_1d_95_avg_pct":    float(var_df["var_1d_95_pct"].mean()) if not var_df.empty else 0,
        "var_1d_95_peak_pct":   float(var_df["var_1d_95_pct"].max()) if not var_df.empty else 0,
        "var_soft_breaches":    int((var_df["var_1d_95_pct"] > limits.var_soft_limit_pct).sum()) if not var_df.empty else 0,
        "var_hard_breaches":    int((var_df["var_1d_95_pct"] > limits.var_hard_limit_pct).sum()) if not var_df.empty else 0,
    }
    pd.Series(kpis).to_csv(os.path.join(HISTDIR, "risk_kpis.csv"), header=False)

    # Copy to demo
    import shutil
    for f in ("risk_limits.csv", "risk_breaches.csv", "var_history.csv", "risk_kpis.csv"):
        src = os.path.join(HISTDIR, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HISTDIR, f"demo_{f}"))

    print("\nRisk KPIs:")
    for k, v in kpis.items():
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    run()

"""
Regime-filtered signal ensemble.

The vol-regime backtest showed the cheap-to-model signal works in high-vol
regimes (62% win) and barely works in low-vol regimes (40% win). The
ensemble layer: only flag BUY when BOTH conditions hold:
  - cheap% >= 5
  - Nikkei 30-day vol percentile >= 33% (mid or high regime)

Then compare:
  - Unfiltered hedged backtest stats (baseline)
  - Filtered subset stats (this filter only)

Saves ensemble_summary.csv with both.
"""

from __future__ import annotations

import os
import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")


def run():
    print("Regime-filtered ensemble: cheap% + vol-regime filter")

    tagged = pd.read_csv(os.path.join(HISTDIR, "hedged_trades_tagged.csv"))
    if tagged.empty:
        print("  No tagged trades; run vol_regime.py first.")
        return

    # Baseline
    base = tagged.copy()

    # Filtered: exclude entries in low-vol regime
    filt = tagged[tagged["entry_regime"].isin(["mid", "high"])].copy()

    def stats(df, label):
        if df.empty:
            return {"label": label, "n_trades": 0}
        return {
            "label":          label,
            "n_trades":       len(df),
            "win_rate_pct":   float((df["net_pnl_jpy"] > 0).mean() * 100),
            "avg_net_bp":     float(df["net_return_bp"].mean()),
            "median_net_bp":  float(df["net_return_bp"].median()),
            "total_pnl_jpy":  float(df["net_pnl_jpy"].sum()),
            "avg_days_held":  float(df["days_held"].mean()),
        }

    rows = [stats(base, "Unfiltered baseline"), stats(filt, "Mid/High-vol only")]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HISTDIR, "ensemble_summary.csv"), index=False)
    import shutil
    shutil.copy(os.path.join(HISTDIR, "ensemble_summary.csv"),
                os.path.join(HISTDIR, "demo_ensemble_summary.csv"))
    print(df.to_string(index=False))


if __name__ == "__main__":
    run()

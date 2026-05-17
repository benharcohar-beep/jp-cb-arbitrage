"""
Concentration metrics for the hedged-trades portfolio.

Outputs concentration.csv with:
  - n_issuers, n_positive_issuers, n_negative_issuers
  - herfindahl: sum of (contribution_pct/100)^2; 1.0 = single issuer, 0 = perfectly diversified
  - top1, top3, top5 contribution %
  - effective_n: 1/herfindahl (inverse), the "effective number of bets"
"""

from __future__ import annotations

import os
import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")


def compute_concentration():
    attr = pd.read_csv(os.path.join(HISTDIR, "attribution_by_issuer.csv"))
    if attr.empty:
        return None
    contrib = attr["contribution_pct"] / 100.0  # decimal
    contrib_sorted = contrib.abs().sort_values(ascending=False)  # use absolute for Herfindahl

    herfindahl = float((contrib_sorted ** 2).sum())
    effective_n = 1.0 / herfindahl if herfindahl > 0 else float("inf")

    top1 = float(contrib_sorted.iloc[0] * 100) if len(contrib_sorted) >= 1 else 0
    top3 = float(contrib_sorted.head(3).sum() * 100) if len(contrib_sorted) >= 3 else float(contrib_sorted.sum() * 100)
    top5 = float(contrib_sorted.head(5).sum() * 100) if len(contrib_sorted) >= 5 else float(contrib_sorted.sum() * 100)

    summary = {
        "n_issuers":           int(len(attr)),
        "n_positive_issuers":  int((attr["total_pnl_jpy"] > 0).sum()),
        "n_negative_issuers":  int((attr["total_pnl_jpy"] < 0).sum()),
        "herfindahl":          herfindahl,
        "effective_n":         effective_n,
        "top1_contribution_pct": top1,
        "top3_contribution_pct": top3,
        "top5_contribution_pct": top5,
    }
    pd.Series(summary).to_csv(os.path.join(HISTDIR, "concentration.csv"), header=False)
    import shutil
    shutil.copy(os.path.join(HISTDIR, "concentration.csv"),
                os.path.join(HISTDIR, "demo_concentration.csv"))
    return summary


if __name__ == "__main__":
    s = compute_concentration()
    print("Portfolio concentration metrics:")
    if s:
        for k, v in s.items():
            print(f"  {k:25s} {v:,.3f}" if isinstance(v, float) else f"  {k:25s} {v}")

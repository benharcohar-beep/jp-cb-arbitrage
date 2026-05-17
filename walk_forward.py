"""
Walk-forward validation.

Splits the historical panel into a training window (first 70%) and a test
window (last 30%). Re-runs the signal evaluation and hedged backtest on
each split independently. If the strategy works on training but breaks on
test, that's the overfitting smoking gun. If both look similar, the signal
is more credibly real.

Note: our model has no fitted parameters (no training step). This is therefore
a *stability* check across two time windows, not a true train/test of a
parameter-fit model. Still useful — and standard quant interview territory.
"""

from __future__ import annotations

import os
import pandas as pd

from backtest import signal_returns, signal_summary, signal_summary_by_bucket
from hedged_backtest import (
    HedgeParams, simulate_trade, attribution_by_issuer,
    simulate_paper_trading,
)

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")


def _split_panel(split_pct: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    panel = pd.read_csv(os.path.join(HISTDIR, "panel.csv"))
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    split_date = panel["date"].quantile(split_pct, interpolation="nearest")
    train = panel[panel["date"] < split_date].copy()
    test  = panel[panel["date"] >= split_date].copy()
    return train, test, split_date


def _signal_stats(panel: pd.DataFrame) -> dict:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    rets = signal_returns(panel, threshold=5.0, max_cheap=25.0)
    if rets.empty:
        return {"n_signals": 0}
    summ = signal_summary(rets)
    # Pick 60d horizon
    s60 = summ[summ["horizon_d"] == 60]
    out = {
        "n_signals_total": int(len(rets) // 3),  # 3 horizons per signal entry
        "horizon_60d": {
            "n": int(s60["n"].iloc[0]) if not s60.empty else 0,
            "avg_fwd_ret": float(s60["avg_fwd_ret"].iloc[0]) if not s60.empty else 0.0,
            "hit_rate": float(s60["hit_rate"].iloc[0]) if not s60.empty else 0.0,
        },
    }
    by_bucket = signal_summary_by_bucket(rets)
    if not by_bucket.empty:
        b60 = by_bucket[by_bucket["horizon_d"] == 60]
        out["by_bucket_60d"] = b60.to_dict("records")
    return out


def _hedged_stats(panel: pd.DataFrame) -> dict:
    """Re-run hedged trade simulation on the given panel slice."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    params = HedgeParams()

    # Bond meta + ratings
    bonds_df = pd.read_csv(os.path.join(PROJ, "bonds.csv"))
    eq_df = pd.read_csv(os.path.join(PROJ, "equities.csv")).set_index("Instrument")
    meta_by_ric = {}
    for _, r in bonds_df.iterrows():
        d = r.to_dict()
        und = d.get("underlying_ric")
        rating = ""
        if und in eq_df.index and pd.notna(eq_df.loc[und].get("Issuer Rating", None)):
            rating = str(eq_df.loc[und].get("Issuer Rating", ""))
        d["rating"] = rating
        meta_by_ric[str(d.get("RIC"))] = d

    trades = []
    for ric, grp in panel.groupby("ric"):
        grp = grp.sort_values("date").reset_index(drop=True)
        meta = meta_by_ric.get(str(ric))
        i = 0
        while i < len(grp):
            cp = grp.iloc[i]["cheap_pct"]
            if params.cheap_entry <= cp <= params.cheap_max:
                tr = simulate_trade(grp, i, params, bond_meta=meta)
                if tr is not None:
                    trades.append(tr)
                    try:
                        ex_idx = grp[grp["date"] == tr["exit_date"]].index[0]
                        i = int(ex_idx) + 5
                        continue
                    except Exception:
                        pass
            i += 1

    if not trades:
        return {"n_trades": 0}

    tdf = pd.DataFrame(trades)
    _, kpis = simulate_paper_trading(trades=tdf, max_concurrent=5)
    return {
        "n_trades":          len(tdf),
        "win_rate_pct":      float((tdf["net_return_bp"] > 0).mean() * 100),
        "avg_net_return_bp": float(tdf["net_return_bp"].mean()),
        "median_return_bp":  float(tdf["net_return_bp"].median()),
        "total_pnl_jpy":     float(tdf["net_pnl_jpy"].sum()),
        "paper_final_usd":   float(kpis.get("final_equity_usd", 0)),
        "paper_total_pct":   float(kpis.get("total_return_pct", 0)),
        "paper_cagr_pct":    float(kpis.get("cagr_pct", 0)),
        "paper_max_dd_pct":  float(kpis.get("max_drawdown_pct", 0)),
        "paper_sharpe":      float(kpis.get("sharpe", 0)),
    }, tdf


def run():
    print("Walk-forward validation …")
    train, test, split_date = _split_panel(0.70)
    print(f"  Split date: {split_date.date()}")
    print(f"  Train: {len(train):,} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Test:  {len(test):,} rows   ({test['date'].min().date()} → {test['date'].max().date()})")

    print("\n--- Signal stats (pure bond return) ---")
    train_sig = _signal_stats(train)
    test_sig  = _signal_stats(test)
    print(f"  Train  60d:  n={train_sig.get('horizon_60d',{}).get('n')}  hit={train_sig.get('horizon_60d',{}).get('hit_rate'):.1f}%  avg={train_sig.get('horizon_60d',{}).get('avg_fwd_ret'):+.2f}%")
    print(f"  Test   60d:  n={test_sig.get('horizon_60d',{}).get('n')}  hit={test_sig.get('horizon_60d',{}).get('hit_rate'):.1f}%  avg={test_sig.get('horizon_60d',{}).get('avg_fwd_ret'):+.2f}%")

    print("\n--- Hedged paper trading by split ---")
    train_h, train_trades = _hedged_stats(train)
    test_h, test_trades = _hedged_stats(test)
    for name, h in (("Train", train_h), ("Test", test_h)):
        if h.get("n_trades", 0) == 0:
            print(f"  {name}: no trades")
            continue
        print(f"  {name}: trades={h['n_trades']:2d}  win={h['win_rate_pct']:.0f}%  "
              f"avg_bp={h['avg_net_return_bp']:+.0f}  CAGR={h['paper_cagr_pct']:+.1f}%  "
              f"Sharpe={h['paper_sharpe']:.2f}  DD={h['paper_max_dd_pct']:+.2f}%")

    # Save consolidated
    rows = []
    for label, h, sig in (("train", train_h, train_sig), ("test", test_h, test_sig)):
        if h.get("n_trades", 0) > 0:
            row = {"split": label, **h}
            h60 = sig.get("horizon_60d", {})
            row.update({
                "pure_signal_n":      h60.get("n", 0),
                "pure_signal_hit":    h60.get("hit_rate", 0),
                "pure_signal_avg":    h60.get("avg_fwd_ret", 0),
            })
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(HISTDIR, "walk_forward.csv"), index=False)
        import shutil
        shutil.copy(os.path.join(HISTDIR, "walk_forward.csv"),
                    os.path.join(HISTDIR, "demo_walk_forward.csv"))
        print(f"\nSaved walk_forward.csv")


if __name__ == "__main__":
    run()

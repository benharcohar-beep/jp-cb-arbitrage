"""
Full rebuild pipeline after a new bond-history pull.

Order:
  1. Replace bonds_hist.csv / equities_hist.csv with the 3y versions
  2. Re-run price_history() → new panel.csv (includes bid_px, ask_px)
  3. Run hedged_backtest with bid/ask execution → new hedged_trades.csv
  4. Run a "no-reset-only" variant for honest comparison
  5. Re-run paper sims, scenarios, attribution, factor decomp, ensemble
  6. Re-generate static site

Use after _pull_3y_v2.py finishes.
"""

import os, shutil, subprocess, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")


def step(label):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)


def run_py(script):
    rc = subprocess.call([sys.executable, os.path.join(PROJ, script)])
    if rc != 0:
        print(f"  WARNING: {script} exited with code {rc}")


def main():
    step("[1/8] Swap in 3y history files")
    for f in ("bonds_hist", "equities_hist"):
        new = os.path.join(HISTDIR, f"{f}_3y.csv")
        cur = os.path.join(HISTDIR, f"{f}.csv")
        if os.path.exists(new):
            # Back up old
            if os.path.exists(cur):
                shutil.copy(cur, cur + ".bak")
            shutil.copy(new, cur)
            print(f"  swapped {f}.csv ← {f}_3y.csv")
        else:
            print(f"  no {f}_3y.csv found; keeping current {f}.csv")

    step("[2/8] Re-run price_history (new panel with bid/ask)")
    from backtest import price_history
    panel = price_history()
    print(f"  panel: {len(panel):,} bond-day rows")

    step("[3/8] Re-run pure-bond signal stats (no data pull — uses existing panel)")
    # Don't call backtest.py main() because that re-pulls data and overwrites our 3y panel.
    # Instead, compute signal_returns + summaries directly from the in-memory panel.
    import pandas as pd
    from backtest import signal_returns, signal_summary, signal_summary_by_bucket
    panel_csv = pd.read_csv(os.path.join(HISTDIR, "panel.csv"))
    panel_csv["date"] = pd.to_datetime(panel_csv["date"]).dt.date
    rets = signal_returns(panel_csv, threshold=5.0, max_cheap=25.0)
    rets.to_csv(os.path.join(HISTDIR, "signal_returns.csv"), index=False)
    if not rets.empty:
        summ = signal_summary(rets)
        summ.to_csv(os.path.join(HISTDIR, "signal_summary.csv"), index=False)
        bucket = signal_summary_by_bucket(rets)
        bucket.to_csv(os.path.join(HISTDIR, "signal_bucket.csv"), index=False)
        print(f"  signal returns: {len(rets):,} rows")

    step("[4/8] Re-run hedged backtest (bid/ask execution)")
    run_py("hedged_backtest.py")

    step("[5/8] No-reset-only variant for honest comparison")
    # Filter out reset bonds, re-simulate
    import pandas as pd
    bonds = pd.read_csv(os.path.join(PROJ, "bonds.csv"))
    reset_rics = set(bonds[bonds["likely_has_reset"] == True]["RIC"].astype(str))
    full_trades = pd.read_csv(os.path.join(HISTDIR, "hedged_trades.csv"))
    no_reset = full_trades[~full_trades["ric"].isin(reset_rics)].copy()
    no_reset.to_csv(os.path.join(HISTDIR, "hedged_trades_no_reset.csv"), index=False)

    if not no_reset.empty:
        from hedged_backtest import simulate_paper_trading, hedged_summary, attribution_by_issuer
        _, kpis_nr = simulate_paper_trading(trades=no_reset, starting_usd=5_000_000.0, max_concurrent=5)
        pd.Series(kpis_nr).to_csv(os.path.join(HISTDIR, "paper_5m_no_reset_kpis.csv"), header=False)
        summ_nr = hedged_summary(no_reset)
        summ_nr.to_csv(os.path.join(HISTDIR, "hedged_summary_no_reset.csv"), index=False)
        print(f"  no-reset universe: {full_trades['ric'].nunique()} → {no_reset['ric'].nunique()} bonds, "
              f"{len(full_trades)} → {len(no_reset)} trades")
        print(f"  no-reset paper $5M: ending ${kpis_nr['final_equity_usd']:,.0f}  "
              f"CAGR {kpis_nr['cagr_pct']:+.1f}%  Sharpe {kpis_nr['sharpe']:.2f}")

    step("[6/8] Run paper_5m + vol_regime + ensemble + concentration + risk")
    for s in ("paper_5m.py", "vol_regime.py", "ensemble.py",
              "concentration.py", "risk.py"):
        if os.path.exists(os.path.join(PROJ, s)):
            run_py(s)

    step("[7/8] Re-run factor decomposition (HAC + bootstrap)")
    run_py("factor_decomp.py")

    step("[8/8] Re-generate static site")
    # Copy no-reset outputs to demo files
    for f in ("hedged_trades_no_reset.csv", "paper_5m_no_reset_kpis.csv",
              "hedged_summary_no_reset.csv"):
        src = os.path.join(HISTDIR, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HISTDIR, f"demo_{f}"))
    run_py("generate_static.py")

    print("\n✅ Rebuild complete.")


if __name__ == "__main__":
    main()

"""
$5,000,000 paper trading portfolio.

Replays the existing hedged_trades.csv with a $5M starting balance and
5 concurrent slots (so each position ≈ $1M). Saves a dedicated equity
curve + KPIs that the dashboard displays as the "active paper portfolio."

Re-runs from history on each agent tick — as the live agent generates
new daily signals and hedged_trades.csv grows, this output will extend
the equity curve forward, giving the dashboard its "active change" feel.
"""

from __future__ import annotations

import os
import shutil

import pandas as pd

from hedged_backtest import simulate_paper_trading

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")

STARTING_USD = 5_000_000.0
USD_JPY = 150.0
MAX_CONCURRENT = 5


def run():
    print(f"Running $5M paper trading simulation ({MAX_CONCURRENT} slots, USD/JPY {USD_JPY}) …")
    trades = pd.read_csv(os.path.join(HISTDIR, "hedged_trades.csv"))
    eq, kpis = simulate_paper_trading(
        trades=trades,
        starting_usd=STARTING_USD,
        usd_jpy=USD_JPY,
        max_concurrent=MAX_CONCURRENT,
    )

    if eq.empty:
        print("  No equity curve generated.")
        return

    eq.to_csv(os.path.join(HISTDIR, "paper_5m_equity.csv"), index=False)
    pd.Series(kpis).to_csv(os.path.join(HISTDIR, "paper_5m_kpis.csv"), header=False)

    # Open positions snapshot: any trade whose exit_date is in the future
    # relative to the last equity-curve date.
    last_date = pd.to_datetime(eq["date"]).max()
    open_pos = trades.copy()
    open_pos["entry_date"] = pd.to_datetime(open_pos["entry_date"])
    open_pos["exit_date"]  = pd.to_datetime(open_pos["exit_date"])
    open_now = open_pos[(open_pos["entry_date"] <= last_date) & (open_pos["exit_date"] > last_date)]
    open_now.to_csv(os.path.join(HISTDIR, "paper_5m_open_positions.csv"), index=False)

    # Recent closed trades (last 10)
    recent = open_pos[open_pos["exit_date"] <= last_date].sort_values("exit_date", ascending=False).head(10)
    recent.to_csv(os.path.join(HISTDIR, "paper_5m_recent_trades.csv"), index=False)

    # Copy to demo files for the static site
    for f in ("paper_5m_equity.csv", "paper_5m_kpis.csv",
              "paper_5m_open_positions.csv", "paper_5m_recent_trades.csv"):
        src = os.path.join(HISTDIR, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HISTDIR, f"demo_{f}"))

    print(f"  Starting:        ${kpis['starting_usd']:,.0f}")
    print(f"  Final equity:    ${kpis['final_equity_usd']:,.0f}")
    print(f"  Total P&L:       ${kpis['final_equity_usd'] - kpis['starting_usd']:+,.0f}")
    print(f"  Total return:    {kpis['total_return_pct']:+,.2f}%")
    print(f"  CAGR:            {kpis['cagr_pct']:+,.2f}%")
    print(f"  Max drawdown:    {kpis['max_drawdown_pct']:+,.2f}%")
    print(f"  Sharpe:          {kpis['sharpe']:.2f}")
    print(f"  Trades taken:    {kpis['n_trades_taken']} / {kpis['n_trades_available']}")
    print(f"  Open today:      {len(open_now)} positions")
    print(f"  Days simulated:  {kpis['days_simulated']}")


if __name__ == "__main__":
    run()

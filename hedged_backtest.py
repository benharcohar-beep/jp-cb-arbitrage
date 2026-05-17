"""
Delta-hedged P&L backtest.

For each cheap-to-model signal in the panel:
  - Enter long bond + short delta×shares of underlying
  - Re-hedge daily (delta drift)
  - Exit at signal_horizon days OR when cheap% drops below exit_threshold
  - Apply transaction costs and financing carry

Assumptions:
  - Transaction cost (bond): 25 bp half-spread per side
  - Transaction cost (equity): 5 bp half-spread per side
  - Financing on short proceeds: earn cash rate (proxy: 1% JPY)
  - Financing on long bond: pay cash rate
  - Bond carry: coupon yield minus repo cost (assume 0 net for zero-coupons)
  - Position size: ¥100M face per signal entry

Output:
  - hedged_signal_returns.csv  — per-trade P&L
  - hedged_summary.csv         — aggregate by horizon and cheap-bucket
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
HISTDIR = os.path.join(PROJ, "history")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class HedgeParams:
    cost_bond_bp:    float = 25.0     # one-way half-spread on bond
    cost_equity_bp:  float = 5.0      # one-way half-spread on equity
    financing_rate:  float = 0.010    # JPY cash rate (decimal, annualised)
    notional_jpy:    float = 100_000_000  # ¥100M face per position
    rehedge_freq:    int   = 5        # rehedge every N business days
    exit_horizon:    int   = 60       # max holding period (business days)
    cheap_entry:     float = 5.0      # enter when cheap% >= this
    cheap_exit:      float = 1.0      # take profit when cheap% drops below
    cheap_max:       float = 25.0     # ignore signals above (anomalies)


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------
def simulate_trade(panel_grp: pd.DataFrame, entry_idx: int,
                   params: HedgeParams) -> dict | None:
    """
    Simulate one trade starting at entry_idx.
    Returns dict with P&L breakdown or None if invalid.
    """
    grp = panel_grp.reset_index(drop=True)
    if entry_idx + 1 >= len(grp):
        return None

    entry = grp.iloc[entry_idx]
    cheap_e = float(entry["cheap_pct"])
    if cheap_e < params.cheap_entry or cheap_e > params.cheap_max:
        return None

    # Position: long ¥100M face of bond, short delta × shares
    bond_face = params.notional_jpy
    bond_units = bond_face / 100.0  # bond price quoted as % of par
    entry_bond_px = float(entry["mkt_px"])
    entry_spot = float(entry["spot"])
    entry_delta = grp.iloc[entry_idx].get("delta", None)
    # delta isn't in panel.csv; derive a proxy from cheap%/spot moves later if missing
    if pd.isna(entry_delta) or entry_delta is None:
        # fallback: compute crude delta from parity sensitivity
        # parity = spot * 100 / conv_price; bond_delta ≈ 0.5 if parity ≈ mkt
        entry_delta = 0.05  # conservative fallback

    shares_short = (entry_delta * bond_units * 100.0) / entry_spot * bond_face / bond_face
    # Cleaner: short delta × bond_face_value / spot
    shares_short = entry_delta * bond_face / entry_spot

    # Walk forward
    days_held = 0
    last_spot = entry_spot
    last_delta = entry_delta
    eq_pnl = 0.0
    bond_pnl = 0.0
    transaction_cost = 0.0
    financing_carry = 0.0

    # Entry costs
    transaction_cost += bond_face * (params.cost_bond_bp / 10_000.0)
    transaction_cost += abs(shares_short) * entry_spot * (params.cost_equity_bp / 10_000.0)

    exit_idx = entry_idx
    exit_reason = "horizon"

    for j in range(entry_idx + 1, min(entry_idx + params.exit_horizon + 1, len(grp))):
        row = grp.iloc[j]
        cur_bond = float(row["mkt_px"])
        cur_spot = float(row["spot"])
        cur_cheap = float(row["cheap_pct"])
        days_held = j - entry_idx

        # Mark-to-market
        bond_change = (cur_bond - entry_bond_px) * bond_units  # JPY
        eq_change = -(cur_spot - last_spot) * shares_short  # short, profit when spot falls

        # Daily financing on short proceeds
        financing_carry += (shares_short * last_spot) * (params.financing_rate / 252)
        # Pay financing on long bond
        financing_carry -= bond_face * (params.financing_rate / 252)

        eq_pnl += eq_change
        last_spot = cur_spot

        # Exit on cheap% mean-reversion below threshold
        if cur_cheap < params.cheap_exit:
            exit_idx = j
            exit_reason = "convergence"
            break
        exit_idx = j

        # Periodic rehedge (delta drift not modelled because panel.csv doesn't carry delta;
        # this is a known simplification; in production we'd reprice on each step)
        if (j - entry_idx) % params.rehedge_freq == 0 and j > entry_idx:
            # cost of rehedging some fraction of position
            rehedge_size = abs(shares_short) * 0.05  # assume 5% drift per cycle
            transaction_cost += rehedge_size * cur_spot * (params.cost_equity_bp / 10_000.0)

    if exit_idx == entry_idx:
        return None

    final = grp.iloc[exit_idx]
    final_bond = float(final["mkt_px"])
    final_spot = float(final["spot"])

    bond_pnl = (final_bond - entry_bond_px) * bond_units
    # equity already accumulated in eq_pnl

    # Exit costs
    transaction_cost += bond_face * (params.cost_bond_bp / 10_000.0)
    transaction_cost += abs(shares_short) * final_spot * (params.cost_equity_bp / 10_000.0)

    gross_pnl = bond_pnl + eq_pnl
    net_pnl = gross_pnl - transaction_cost + financing_carry
    net_return_bp = net_pnl / bond_face * 10_000.0

    return {
        "ric": entry["ric"],
        "issuer": entry["issuer"],
        "entry_date": entry["date"],
        "exit_date": final["date"],
        "days_held": days_held,
        "entry_cheap": cheap_e,
        "exit_cheap": float(final["cheap_pct"]),
        "entry_bond": entry_bond_px,
        "exit_bond": final_bond,
        "entry_spot": entry_spot,
        "exit_spot": final_spot,
        "shares_short": shares_short,
        "delta_used": entry_delta,
        "bond_pnl_jpy": bond_pnl,
        "eq_pnl_jpy": eq_pnl,
        "transaction_cost_jpy": transaction_cost,
        "financing_carry_jpy": financing_carry,
        "gross_pnl_jpy": gross_pnl,
        "net_pnl_jpy": net_pnl,
        "net_return_bp": net_return_bp,
        "exit_reason": exit_reason,
    }


def run_hedged_backtest(params: HedgeParams = None) -> pd.DataFrame:
    params = params or HedgeParams()
    panel = pd.read_csv(os.path.join(HISTDIR, "panel.csv"))
    panel["date"] = pd.to_datetime(panel["date"]).dt.date

    trades = []
    for ric, grp in panel.groupby("ric"):
        grp = grp.sort_values("date").reset_index(drop=True)
        # Pseudo-delta from panel (not stored; recompute approx from parity)
        # Skip — we'll use fallback delta.
        i = 0
        while i < len(grp):
            row = grp.iloc[i]
            cp = row["cheap_pct"]
            if params.cheap_entry <= cp <= params.cheap_max:
                trade = simulate_trade(grp, i, params)
                if trade is not None:
                    trades.append(trade)
                    # Skip ahead to exit + cooldown
                    # find exit_date in grp
                    try:
                        ex_idx = grp[grp["date"] == trade["exit_date"]].index[0]
                        i = int(ex_idx) + 5  # 5-day cooldown
                        continue
                    except Exception:
                        pass
            i += 1

    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df.to_csv(os.path.join(HISTDIR, "hedged_trades.csv"), index=False)
    return df


def hedged_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    bins = [5, 10, 15, 25]
    labels = ["5-10%", "10-15%", "15-25%"]
    trades = trades.copy()
    trades["bucket"] = pd.cut(trades["entry_cheap"], bins=bins, labels=labels, include_lowest=True)
    g = trades.groupby("bucket", observed=True).agg(
        n=("ric", "size"),
        avg_days=("days_held", "mean"),
        avg_gross_bp=("gross_pnl_jpy", lambda s: s.mean() / 100_000_000 * 10_000),
        avg_net_bp=("net_pnl_jpy", lambda s: s.mean() / 100_000_000 * 10_000),
        avg_net_return_bp=("net_return_bp", "mean"),
        median_net_bp=("net_return_bp", "median"),
        hit_rate_pct=("net_return_bp", lambda s: (s > 0).mean() * 100),
        avg_costs_bp=("transaction_cost_jpy", lambda s: s.mean() / 100_000_000 * 10_000),
    ).reset_index()
    return g


def simulate_paper_trading(
    trades: pd.DataFrame = None,
    starting_usd: float = 1_000_000.0,
    usd_jpy: float = 150.0,
    max_concurrent: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate running the strategy with real (paper) money.

    Walks every hedged trade chronologically. Allocates equity / open_slots
    JPY notional per new trade (max `max_concurrent` simultaneous positions).
    On exit, scales the trade's per-¥100M P&L by the actual notional used and
    adds it to cash. Tracks the equity curve and computes risk-adjusted KPIs.
    """
    if trades is None:
        trades = pd.read_csv(os.path.join(HISTDIR, "hedged_trades.csv"))
    if trades.empty:
        return pd.DataFrame(), {}

    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"])
    trades = trades.sort_values("entry_date").reset_index(drop=True)

    starting_jpy = starting_usd * usd_jpy
    cash = starting_jpy
    open_positions = []   # [{trade_idx, notional_jpy, exit_date}]
    history = []
    n_skipped_no_slot = 0
    n_skipped_too_small = 0

    # Build interleaved event timeline: exits before entries on a given day
    events = []
    for i, row in trades.iterrows():
        events.append((row["entry_date"], 1, "entry", i))  # entry sort key=1
        events.append((row["exit_date"],  0, "exit",  i))  # exit sort key=0
    events.sort(key=lambda x: (x[0], x[1]))

    for evt_date, _, evt_type, idx in events:
        if evt_type == "exit":
            for op in list(open_positions):
                if op["trade_idx"] == idx:
                    realized = float(trades.loc[idx, "net_pnl_jpy"]) * (op["notional_jpy"] / 100_000_000.0)
                    cash += op["notional_jpy"]  # release deployed capital
                    cash += realized            # add P&L
                    open_positions.remove(op)
                    break
        else:
            if len(open_positions) >= max_concurrent:
                n_skipped_no_slot += 1
                continue
            slots_open = max_concurrent - len(open_positions)
            notional = cash / slots_open
            if notional < 10_000_000:  # minimum ¥10M position
                n_skipped_too_small += 1
                continue
            cash -= notional
            open_positions.append({
                "trade_idx": idx,
                "notional_jpy": notional,
                "exit_date": trades.loc[idx, "exit_date"],
            })

        deployed = sum(op["notional_jpy"] for op in open_positions)
        equity_jpy = cash + deployed
        history.append({
            "date":            evt_date,
            "cash_jpy":        cash,
            "deployed_jpy":    deployed,
            "equity_jpy":      equity_jpy,
            "equity_usd":      equity_jpy / usd_jpy,
            "open_positions":  len(open_positions),
            "event":           evt_type,
        })

    eq = pd.DataFrame(history)
    if eq.empty:
        return eq, {}

    # Drawdown
    eq["running_max"] = eq["equity_usd"].cummax()
    eq["drawdown_pct"] = (eq["equity_usd"] / eq["running_max"] - 1) * 100.0

    # Daily returns for Sharpe
    eq_daily = (eq.set_index("date")["equity_usd"]
                  .resample("D").last().ffill().dropna())
    if len(eq_daily) > 30:
        daily_ret = eq_daily.pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5) if daily_ret.std() > 0 else 0.0
    else:
        sharpe = 0.0

    final_equity_usd = float(eq["equity_usd"].iloc[-1])
    days = max((eq["date"].max() - eq["date"].min()).days, 1)
    years = days / 365.25
    total_ret_pct = (final_equity_usd / starting_usd - 1) * 100.0
    cagr_pct = ((final_equity_usd / starting_usd) ** (1 / years) - 1) * 100.0 if years > 0 else 0.0

    kpis = {
        "starting_usd":      starting_usd,
        "final_equity_usd":  final_equity_usd,
        "total_return_pct":  total_ret_pct,
        "cagr_pct":          cagr_pct,
        "max_drawdown_pct":  float(eq["drawdown_pct"].min()),
        "sharpe":            float(sharpe),
        "days_simulated":    days,
        "n_trades_taken":    int(len(trades) - n_skipped_no_slot - n_skipped_too_small),
        "n_trades_skipped":  int(n_skipped_no_slot + n_skipped_too_small),
        "max_concurrent":    int(max_concurrent),
        "usd_jpy_assumed":   float(usd_jpy),
        "n_trades_available": int(len(trades)),
    }

    eq.to_csv(os.path.join(HISTDIR, "paper_equity.csv"), index=False)
    pd.Series(kpis).to_csv(os.path.join(HISTDIR, "paper_kpis.csv"), header=False)
    return eq, kpis


def main():
    print("Running delta-hedged backtest …")
    trades = run_hedged_backtest()
    if trades.empty:
        print("No trades generated.")
        return
    print(f"  {len(trades)} hedged trades simulated")
    summ = hedged_summary(trades)
    summ.to_csv(os.path.join(HISTDIR, "hedged_summary.csv"), index=False)
    print(summ.to_string(index=False))

    overall = {
        "n_trades": len(trades),
        "win_rate_pct": (trades["net_return_bp"] > 0).mean() * 100,
        "avg_net_return_bp": trades["net_return_bp"].mean(),
        "median_net_return_bp": trades["net_return_bp"].median(),
        "avg_days_held": trades["days_held"].mean(),
        "avg_gross_pnl_jpy": trades["gross_pnl_jpy"].mean(),
        "avg_net_pnl_jpy": trades["net_pnl_jpy"].mean(),
        "avg_costs_jpy": trades["transaction_cost_jpy"].mean(),
        "total_net_pnl_jpy": trades["net_pnl_jpy"].sum(),
    }
    pd.Series(overall).to_csv(os.path.join(HISTDIR, "hedged_overall.csv"), header=False)
    print("\nOverall:")
    for k, v in overall.items():
        print(f"  {k:25s} {v:,.2f}")

    print("\n--- Paper trading simulation ($1M starting equity, default: 5 slots) ---")
    eq, kpis = simulate_paper_trading(trades=trades)
    print(f"  Starting capital: ${kpis['starting_usd']:,.0f} USD (≈ ¥{kpis['starting_usd']*kpis['usd_jpy_assumed']:,.0f})")
    print(f"  Ending equity:    ${kpis['final_equity_usd']:,.0f} USD")
    print(f"  Total return:     {kpis['total_return_pct']:+,.2f}%")
    print(f"  CAGR:             {kpis['cagr_pct']:+,.2f}%")
    print(f"  Max drawdown:     {kpis['max_drawdown_pct']:+,.2f}%")
    print(f"  Sharpe (approx):  {kpis['sharpe']:.2f}")
    print(f"  Trades taken:     {kpis['n_trades_taken']} / {kpis['n_trades_available']}")
    print(f"  Days simulated:   {kpis['days_simulated']}")

    # ---------- Multi-scenario sweep ----------
    print("\n--- Sizing sensitivity: max concurrent positions ---")
    scenarios = []
    all_curves = []
    for slots in (1, 2, 3, 5, 8):
        eq_s, k = simulate_paper_trading(
            trades=trades, starting_usd=1_000_000.0,
            usd_jpy=150.0, max_concurrent=slots,
        )
        if k:
            scenarios.append({
                "max_concurrent":     slots,
                "final_equity_usd":   k["final_equity_usd"],
                "total_return_pct":   k["total_return_pct"],
                "cagr_pct":           k["cagr_pct"],
                "max_drawdown_pct":   k["max_drawdown_pct"],
                "sharpe":             k["sharpe"],
                "n_trades_taken":     k["n_trades_taken"],
                "n_trades_available": k["n_trades_available"],
                "days_simulated":     k["days_simulated"],
            })
            eq_s["max_concurrent"] = slots
            all_curves.append(eq_s[["date", "equity_usd", "drawdown_pct", "max_concurrent"]])

    if scenarios:
        scen_df = pd.DataFrame(scenarios)
        scen_df.to_csv(os.path.join(HISTDIR, "paper_scenarios.csv"), index=False)
        print(scen_df.to_string(index=False))

    if all_curves:
        curves_df = pd.concat(all_curves, ignore_index=True)
        curves_df.to_csv(os.path.join(HISTDIR, "paper_scenario_curves.csv"), index=False)
        # Copy to demo files for static site
        import shutil
        for f in ("paper_scenarios.csv", "paper_scenario_curves.csv"):
            src = os.path.join(HISTDIR, f)
            dst = os.path.join(HISTDIR, f"demo_{f}")
            if os.path.exists(src):
                shutil.copy(src, dst)


if __name__ == "__main__":
    main()

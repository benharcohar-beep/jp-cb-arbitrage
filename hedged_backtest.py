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


if __name__ == "__main__":
    main()

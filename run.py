"""
Run the cheapness screen on the live Refinitiv data set.
Reads bonds.csv / equities.csv / jgb.csv (created by pull_data.py),
prices each bond with the Tsiveriotis-Fernandes binomial tree, ranks
by cheapness and prints the table.

Usage:
    python3 pull_data.py   # refresh data
    python3 run.py         # run the screen
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime

import pandas as pd
from tabulate import tabulate

PROJ = os.path.dirname(os.path.abspath(__file__))
SNAPSHOTS = os.path.join(PROJ, "snapshots")
os.makedirs(SNAPSHOTS, exist_ok=True)

from pricer import (
    bond_floor,
    compute_greeks,
    conversion_premium,
    implied_vol,
    parity,
    price_cb,
)
from anomaly import confidence, detect_anomalies
from mc_pricer import default_reset_dates, price_cb_mc_with_reset
from real_data import build_universe


def evaluate(entry, n_steps=250):
    bond, mkt, mkt_px = entry["bond"], entry["mkt"], entry["mkt_px"]
    res_tree = price_cb(bond, mkt, n_steps=n_steps)
    floor = bond_floor(bond, mkt)
    par = parity(bond, mkt.spot)
    prem = conversion_premium(bond, mkt_px, mkt.spot)

    # If bond is flagged for likely reset, also run MC and use that as the
    # model price (the tree under-prices reset bonds).
    mc_price = None
    reset_prob = None
    if entry.get("likely_reset"):
        resets = default_reset_dates(bond, mkt)
        if resets:
            mc = price_cb_mc_with_reset(bond, mkt, resets, n_paths=4000, n_steps=200)
            mc_price = mc["price"]
            reset_prob = mc["reset_prob"]

    model_price = mc_price if mc_price is not None else res_tree["price"]
    cheap = (model_price - mkt_px) / mkt_px
    iv = implied_vol(bond, mkt, mkt_px, n_steps=150)
    g = compute_greeks(bond, mkt, n_steps=150)
    record = {
        "issuer": entry["issuer"],
        "ric": entry["ric"],
        "underlying": entry["underlying_ric"],
        "rating": entry.get("rating", "NR"),
        "spread_bp": entry.get("spread_bp", 0),
        "reset_flag": "RESET" if entry.get("likely_reset") else "",
        "maturity": bond.maturity.isoformat(),
        "spot": mkt.spot,
        "sigma": mkt.sigma,
        "conv_px": bond.conversion_price,
        "parity": par,
        "mkt_px": mkt_px,
        "model_px": model_price,
        "model_tree_px": res_tree["price"],
        "model_mc_px": mc_price,
        "model_engine": "MC" if mc_price is not None else "Tree",
        "reset_prob": reset_prob,
        "bond_floor": floor,
        "cheap_pct": cheap * 100.0,
        "premium_pct": prem * 100.0,
        "iv": iv,
        "iv_minus_hv_pct": (iv - mkt.sigma) * 100.0 if iv is not None else None,
        **g,
    }
    record["anomaly_flags"] = ",".join(detect_anomalies(record))
    record["confidence"] = confidence(record)
    return record


def fmt_pct(x, prec=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{prec}f}"


def fmt_num(x, prec=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.{prec}f}"


def main():
    print("Loading universe …")
    entries = build_universe()
    if not entries:
        print("No bonds with full data — re-run pull_data.py first.")
        return

    print(f"\nPricing {len(entries)} bonds …")
    rows = []
    for i, e in enumerate(entries, 1):
        try:
            rows.append(evaluate(e))
        except Exception as ex:
            print(f"  [{i}/{len(entries)}] {e['issuer']:25s} FAIL: {ex}")
    rows.sort(key=lambda r: -r["cheap_pct"])

    table = []
    for r in rows:
        table.append([
            r["issuer"][:18],
            r["underlying"],
            r["rating"],
            f"{r['spread_bp']}",
            r["reset_flag"],
            r["maturity"],
            fmt_num(r["mkt_px"]),
            fmt_num(r["model_px"]),
            fmt_pct(r["cheap_pct"]),
            fmt_pct(r["premium_pct"]),
            fmt_num(r["sigma"], 3),
            fmt_num(r["iv"], 3) if r["iv"] is not None else "—",
            fmt_pct(r["iv_minus_hv_pct"]),
            fmt_num(r["delta"], 3),
            fmt_num(r["vega"], 2),
        ])

    headers = [
        "Issuer", "Equity", "Rtg", "Spd",
        "Reset?", "Maturity",
        "Mkt", "Model", "Cheap%",
        "Prem%", "HV60", "IV", "IV-HV%",
        "Δ", "Vega",
    ]
    print()
    print(f"Japanese CB Arb Screen — {date.today()}  (Source: Refinitiv / LSEG Eikon)")
    print("=" * 160)
    print(tabulate(table, headers=headers, tablefmt="github", numalign="right"))
    print()
    print("Cheap% > 0  ⇒  bond looks undervalued vs. model (buy CB / short Δ shares)")
    print("Cheap% < 0  ⇒  rich vs. model")
    print(f"Bonds priced: {len(rows)}")

    # Snapshot
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    snap_csv = os.path.join(SNAPSHOTS, f"screen_{ts}.csv")
    df_now = pd.DataFrame(rows)
    df_now.to_csv(snap_csv, index=False)
    print(f"\nSnapshot saved → {snap_csv}")

    # Diff against most recent prior snapshot — surface new + dropped bonds
    import glob
    prior_files = sorted(glob.glob(os.path.join(SNAPSHOTS, "screen_*.csv")))
    if len(prior_files) >= 2:
        prior = pd.read_csv(prior_files[-2])
        new_rics = set(df_now["ric"]) - set(prior["ric"])
        dropped = set(prior["ric"]) - set(df_now["ric"])
        if new_rics:
            print("\n>>> NEW IN UNIVERSE (vs. previous snapshot)")
            for ric in new_rics:
                row = df_now[df_now["ric"] == ric].iloc[0]
                print(f"    + {row['issuer']:25s} {ric:14s}  "
                      f"mat={row['maturity']}  cheap={row['cheap_pct']:+.1f}%")
        if dropped:
            print("\n>>> DROPPED FROM UNIVERSE (vs. previous snapshot)")
            for ric in dropped:
                row = prior[prior["ric"] == ric].iloc[0]
                print(f"    - {row['issuer']:25s} {ric:14s}  (no live quote this run)")

    # Top alerts — all bonds ≥5% cheap. Reset-flagged bonds use MC price.
    cheap_all = [r for r in rows if r["cheap_pct"] >= 5][:10]
    if cheap_all:
        print("\n>>> ALERT: bonds ≥ 5% cheap to model")
        for r in cheap_all:
            engine = r.get("model_engine", "Tree")
            extra = f"  reset_prob={r['reset_prob']:.0%}" if r.get("reset_prob") is not None else ""
            print(f"    {r['issuer']:25s} {r['ric']:14s}  rtg={r['rating']:5s} "
                  f"cheap={r['cheap_pct']:+.1f}%  mkt={r['mkt_px']:.2f}  "
                  f"model={r['model_px']:.2f} ({engine}){extra}  Δ={r['delta']:.3f}")


if __name__ == "__main__":
    main()

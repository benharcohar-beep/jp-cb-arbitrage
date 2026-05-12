"""
Issuer-specific CDS spread lookup. For each issuer in the universe, search
Refinitiv for a 5y JPY senior unsecured CDS, pull the latest spread (bp),
and write to cds_spreads.csv. real_data.py prefers this over the
rating-based spread when available.

Run: python3 cds_pull.py    (requires Refinitiv Workspace open)
"""

from __future__ import annotations

import os
import time
import traceback

import pandas as pd
import refinitiv.data as rd
from refinitiv.data.content import search

PROJ = os.path.dirname(os.path.abspath(__file__))


def find_cds_ric(issuer_name: str) -> str | None:
    """Search for 5y JPY senior CDS on this issuer."""
    try:
        resp = search.Definition(
            view=search.Views.SEARCH_ALL,
            query=f"{issuer_name} CDS 5 Yr JPY Senior",
            top=10,
        ).get_data()
        df = resp.data.df
        if df.empty or "DocumentTitle" not in df.columns:
            return None
        title = df["DocumentTitle"].fillna("").astype(str)
        # Pattern: "<Issuer>, Single Name Credit Default Swap, JPY 5 Yr Senior Unsecured"
        m = title.str.contains(r"Credit Default Swap.*JPY 5 Yr Senior", regex=True, case=False)
        hit = df[m]
        if hit.empty:
            return None
        # Prefer the 'AC=R' (cum-restructuring 2014) RICs over '=MT' / '=FN'
        hit = hit.copy()
        hit["score"] = hit["RIC"].astype(str).apply(
            lambda r: 3 if r.endswith("AC=R") else 2 if r.endswith("AC=MT") else 1
        )
        hit = hit.sort_values("score", ascending=False)
        return str(hit["RIC"].iloc[0])
    except Exception:
        return None


def pull_cds_spreads(issuers: list[str]) -> pd.DataFrame:
    """For each issuer, find CDS RIC and pull latest spread."""
    rows = []
    for name in sorted(set(issuers)):
        try:
            ric = find_cds_ric(name)
            if not ric:
                rows.append({"issuer": name, "cds_ric": "", "cds_spread_bp": None})
                continue
            # Pull latest CDS spread — try TR.MidYield, TR.MidPrice, CF_LAST
            df = rd.get_data(universe=[ric], fields=["TR.MidPrice", "CF_LAST"])
            spread = None
            if df is not None and not df.empty:
                for col in df.columns:
                    if col == "Instrument":
                        continue
                    v = df[col].iloc[0]
                    if pd.notna(v) and isinstance(v, (int, float)) and 1 < v < 5000:
                        spread = float(v)
                        break
            rows.append({"issuer": name, "cds_ric": ric, "cds_spread_bp": spread})
        except Exception as e:
            print(f"  {name}: {str(e)[:100]}")
            rows.append({"issuer": name, "cds_ric": "", "cds_spread_bp": None})
        time.sleep(0.3)
    return pd.DataFrame(rows)


def main():
    print("Opening Refinitiv session …")
    rd.open_session("desktop.workspace")
    try:
        bonds = pd.read_csv(os.path.join(PROJ, "bonds.csv"))
        issuers = bonds["IssuerName"].dropna().astype(str).unique().tolist()
        print(f"Pulling CDS for {len(issuers)} issuers …")
        df = pull_cds_spreads(issuers)
        n_with_cds = df["cds_spread_bp"].notna().sum()
        print(f"  found CDS for {n_with_cds}/{len(df)} issuers")
        df.to_csv(os.path.join(PROJ, "cds_spreads.csv"), index=False)
        print(f"  saved cds_spreads.csv")
        if n_with_cds > 0:
            print("\nSample with CDS:")
            print(df[df["cds_spread_bp"].notna()].head(10).to_string(index=False))
    finally:
        try:
            rd.close_session()
        except Exception:
            pass


if __name__ == "__main__":
    main()

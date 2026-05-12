"""
Pull real Japanese CB data from Refinitiv. Saves three CSVs:

    bonds.csv     — CB universe (terms + market price)
    equities.csv  — underlying TSE equity (spot, vol, div yield)
    jgb.csv       — JGB yield curve

Run: python3 pull_data.py
Requires: Refinitiv Workspace open + logged in.
"""

from __future__ import annotations

import os
import time
import traceback
from datetime import date

import pandas as pd
import refinitiv.data as rd
from refinitiv.data.content import search

PROJ = os.path.dirname(os.path.abspath(__file__))


def _open():
    rd.open_session("desktop.workspace")


def _close():
    try:
        rd.close_session()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1) CB universe + terms via Search (search-view fields)
# ---------------------------------------------------------------------------

UNIV_SELECT = [
    "RIC", "DocumentTitle", "IssuerName", "Currency",
    "MaturityDate", "IssueDate", "ISIN", "Coupon",
    "ConversionPrice", "IsCallable", "IsConvertible",
    "FaceIssuedTotal", "AssetType",
]


def search_universe(top: int = 500) -> pd.DataFrame:
    resp = search.Definition(
        view=search.Views.GOV_CORP_INSTRUMENTS,
        filter="(Currency eq 'JPY') and (IsConvertible eq true) and (IsActive eq true)",
        select=",".join(UNIV_SELECT),
        top=top,
    ).get_data()
    df = resp.data.df
    if df.empty:
        return df

    # Keep only real CBs (exclude structured products & range-coupon notes)
    title = df["DocumentTitle"].fillna("").astype(str)
    is_cb = title.str.contains(", Convertible,", case=False, na=False)
    df = df[is_cb].copy()

    # Keep only Japanese-corporate issuers (drop foreign-bank Euro-yen notes
    # whose underlying equity may be Asian but issuer is HSBC / SG / JPMS).
    foreign_issuers = {"HSBC HK", "SG Issuer", "JP Morgan Str Pr",
                       "BNP Paribas Issuance BV", "UBS AG London"}
    df = df[~df["IssuerName"].isin(foreign_issuers)].copy()

    df = df.dropna(subset=["RIC", "MaturityDate", "ConversionPrice"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2) Live market price via get_data (Mid / Bid / Ask)
# ---------------------------------------------------------------------------

def pull_market_prices(rics: list[str], chunk: int = 25) -> pd.DataFrame:
    fields = ["TR.MidPrice", "TR.BIDPRICE", "TR.ASKPRICE",
              "TR.FiCouponRate", "TR.FiCouponFrequency",
              "TR.FiNextCallDate", "TR.FiNextCallPrice",
              "TR.FiNextPutDate", "TR.FiNextPutPrice"]
    frames = []
    for i in range(0, len(rics), chunk):
        sub = rics[i:i + chunk]
        for attempt in range(3):
            try:
                df = rd.get_data(universe=sub, fields=fields)
                if df is not None and not df.empty:
                    frames.append(df)
                break
            except Exception as e:
                print(f"    chunk {i}: attempt {attempt+1} failed: {str(e)[:80]}")
                time.sleep(2.0)
        time.sleep(0.4)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 3) Underlying TSE equity discovery — by issuer name
# ---------------------------------------------------------------------------

def find_equity_ric(issuer_name: str) -> str | None:
    """
    Find the issuer's TSE ordinary share RIC.
    Uses free-text query + filters DocumentTitle to TSE Ordinary Share.
    """
    try:
        resp = search.Definition(
            view=search.Views.EQUITY_QUOTES,
            query=issuer_name,
            top=20,
        ).get_data()
        df = resp.data.df
        if df.empty or "DocumentTitle" not in df.columns:
            return None
        title = df["DocumentTitle"].fillna("").astype(str)
        mask = title.str.contains(
            r"Ordinary Share.*Tokyo Stock Exchange", regex=True, case=False
        )
        hit = df[mask]
        if hit.empty:
            return None
        return str(hit["RIC"].iloc[0])
    except Exception:
        return None


def map_underlyings(issuers: list[str]) -> pd.DataFrame:
    rows = []
    for name in sorted(set(issuers)):
        ric = find_equity_ric(name)
        rows.append({"IssuerName": name, "underlying_ric": ric})
        time.sleep(0.15)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4) Equity market data
# ---------------------------------------------------------------------------

EQUITY_FIELDS = [
    "TR.PriceClose",
    "TR.CompanyName",
    "TR.Volatility60D",
    "TR.Volatility90D",
    "TR.DividendYield",
    "TR.AvgDailyVolume30D",
    "TR.CompanyMarketCap",
    "TR.ShareIssuingCurrency",
]


def _chunked_call(rics: list[str], fields: list[str], chunk: int) -> pd.DataFrame:
    frames = []
    for i in range(0, len(rics), chunk):
        sub = rics[i:i + chunk]
        for attempt in range(3):
            try:
                df = rd.get_data(universe=sub, fields=fields)
                if df is not None and not df.empty:
                    frames.append(df)
                break
            except Exception as e:
                print(f"    chunk {i}: attempt {attempt+1} failed: {str(e)[:80]}")
                time.sleep(2.0)
        time.sleep(0.4)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pull_equity_data(rics: list[str], chunk: int = 25) -> pd.DataFrame:
    """Two passes — Refinitiv batches with TR.IssuerRating drop other fields silently."""
    rics = sorted({r for r in rics if r and isinstance(r, str)})
    main = _chunked_call(rics, EQUITY_FIELDS, chunk)
    rating = _chunked_call(rics, ["TR.IssuerRating"], chunk)
    if main.empty:
        return main
    if rating.empty:
        return main
    return main.merge(rating, on="Instrument", how="left")


# ---------------------------------------------------------------------------
# 5) JGB yield curve
# ---------------------------------------------------------------------------

JGB_RICS = {"2y": "JP2YT=RR", "3y": "JP3YT=RR", "5y": "JP5YT=RR",
            "7y": "JP7YT=RR", "10y": "JP10YT=RR",
            "20y": "JP20YT=RR", "30y": "JP30YT=RR"}


def pull_jgb() -> pd.DataFrame:
    """TR.MidYield gives the actual yield (price-based CF_LAST returned nonsense)."""
    rics = list(JGB_RICS.values())
    df = rd.get_data(universe=rics, fields=["TR.MidYield"])
    if df is None or df.empty:
        return pd.DataFrame()
    inv = {v: k for k, v in JGB_RICS.items()}
    df["tenor"] = df["Instrument"].map(inv)
    yield_col = next((c for c in df.columns if "yield" in c.lower()), None)
    if yield_col and yield_col != "yield_pct":
        df = df.rename(columns={yield_col: "yield_pct"})
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Opening Refinitiv session …")
    _open()
    try:
        print("\n[1/5] Searching JP CB universe …")
        uni = search_universe()
        print(f"  found {len(uni)} real Japanese CBs")
        uni.to_csv(os.path.join(PROJ, "universe_search.csv"), index=False)

        if uni.empty:
            return

        print("\n[2/5] Pulling live market prices …")
        rics = uni["RIC"].dropna().tolist()
        prices = pull_market_prices(rics)
        print(f"  pulled {len(prices)} rows of price data")

        # Merge
        bonds = uni.merge(prices, left_on="RIC", right_on="Instrument",
                          how="left", suffixes=("", "_px"))

        print("\n[3/5] Mapping underlying TSE equity RICs …")
        und = map_underlyings(bonds["IssuerName"].tolist())
        und.to_csv(os.path.join(PROJ, "underlying_map.csv"), index=False)
        print(f"  mapped {und['underlying_ric'].notna().sum()}/{len(und)} issuers")
        bonds = bonds.merge(und, on="IssuerName", how="left")

        # Compute conversion ratio (shares per ¥100 face)
        bonds["conv_ratio_per_100"] = 100.0 / bonds["ConversionPrice"]

        # Reset-feature heuristic flag.
        # Refinitiv doesn't expose reset fields via the data API, so we
        # detect bonds that LIKELY have downward conversion-price reset
        # provisions based on patterns seen across JP small-cap CBs:
        #   - issued at par, 0% coupon, ≤7y to maturity, JPY currency,
        #     unrated or sub-investment-grade issuer
        # These are the textbook candidates for aggressive reset clauses.
        # Bonds flagged here should be treated with low model confidence —
        # the binomial pricer doesn't capture path-dependent resets.
        # Reset heuristic: Refinitiv doesn't expose reset fields, so we infer.
        # Resets are concentrated in:
        #   - smaller issuance (≤ ¥30bn / $200m)
        #   - 0% coupon JPY bonds (typical "deep discount" reset CB structure)
        # The further filter on issuer rating happens downstream in real_data.py
        # (where we have the rating from the equity-side pull).
        coup_col = next((c for c in ("Coupon Rate", "Coupon") if c in bonds.columns), None)
        size_col = next((c for c in ("FaceIssuedTotal", "Face Issued Total") if c in bonds.columns), None)
        small_issue = (bonds[size_col].fillna(1e15) <= 30_000_000_000.0) if size_col else False
        zero_coupon = (bonds[coup_col].fillna(0) == 0.0) if coup_col else False
        bonds["likely_has_reset"] = zero_coupon & small_issue & (bonds["Currency"].fillna("") == "JPY")

        bonds.to_csv(os.path.join(PROJ, "bonds.csv"), index=False)
        print(f"  saved bonds.csv  ({bonds.shape[0]} x {bonds.shape[1]})")

        print("\n[4/5] Pulling equity data for underlyings …")
        eq_rics = bonds["underlying_ric"].dropna().tolist()
        equities = pull_equity_data(eq_rics)
        equities.to_csv(os.path.join(PROJ, "equities.csv"), index=False)
        print(f"  saved equities.csv  ({equities.shape[0]} rows)")

        print("\n[5/5] Pulling JGB yield curve …")
        jgb = pull_jgb()
        jgb.to_csv(os.path.join(PROJ, "jgb.csv"), index=False)
        print(f"  saved jgb.csv  ({jgb.shape[0]} rows)")

        print("\nDone.")
    finally:
        _close()


if __name__ == "__main__":
    main()

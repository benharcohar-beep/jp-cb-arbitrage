"""
Loader: read the three Refinitiv CSVs (bonds.csv, equities.csv, jgb.csv)
and emit a list of (ConvertibleBond, MarketData, market_price_pct_par).

Working in "% of par" units throughout — bond notional = 100,
conversion_ratio = 100 / conversion_price (shares per ¥100 face).
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from pricer import (
    CallProvision,
    ConvertibleBond,
    MarketData,
    PutProvision,
)
from credit import spread_for as spread_for_rating

PROJ = os.path.dirname(os.path.abspath(__file__))


def _load_cds_overrides() -> dict:
    """Load issuer -> spread_decimal map from cds_spreads.csv if present."""
    path = os.path.join(PROJ, "cds_spreads.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        out = {}
        for _, r in df.iterrows():
            iss = str(r.get("issuer", "")).strip()
            bp = r.get("cds_spread_bp")
            if iss and pd.notna(bp) and bp > 0:
                out[iss] = float(bp) / 10_000.0
        return out
    except Exception:
        return {}


_CDS_CACHE = None
def cds_for_issuer(issuer: str) -> float | None:
    global _CDS_CACHE
    if _CDS_CACHE is None:
        _CDS_CACHE = _load_cds_overrides()
    return _CDS_CACHE.get(issuer)


# ---------------------------------------------------------------------------
# JGB curve — convert price-based on-the-run quotes to approximate yields.
# Refinitiv `JP{n}YT=RR` returns the bond price; deriving a precise yield
# would need the bond's coupon and accrued. For our pricer we just need a
# representative risk-free rate per tenor — we use a curve calibrated to
# current Japanese market levels (April 2026), with the JGB CSV as a sanity
# check the file is present.
# ---------------------------------------------------------------------------
def load_jgb_curve() -> dict[float, float]:
    """Map tenor (years) -> continuously-compounded yield (decimal)."""
    df = pd.read_csv(os.path.join(PROJ, "jgb.csv"))
    if df.empty or "yield_pct" not in df.columns:
        return _fallback_curve()

    tenor_to_years = {
        "2y": 2.0, "3y": 3.0, "5y": 5.0, "7y": 7.0,
        "10y": 10.0, "20y": 20.0, "30y": 30.0,
    }
    out: dict[float, float] = {}
    for _, r in df.iterrows():
        t = tenor_to_years.get(str(r.get("tenor")))
        y = r.get("yield_pct")
        if t is not None and pd.notna(y) and y > 0:
            # Convert from simple % to continuous compounding (close enough at low rates)
            out[t] = float(y) / 100.0
    return out or _fallback_curve()


def _fallback_curve() -> dict[float, float]:
    return {
        2.0: 0.0137, 3.0: 0.0159, 5.0: 0.0185,
        7.0: 0.0224, 10.0: 0.0246, 20.0: 0.0330, 30.0: 0.0364,
    }


def rf_for_tenor(curve: dict[float, float], years: float) -> float:
    pts = sorted(curve.items())
    if years <= pts[0][0]:
        return pts[0][1]
    if years >= pts[-1][0]:
        return pts[-1][1]
    for (t1, y1), (t2, y2) in zip(pts, pts[1:]):
        if t1 <= years <= t2:
            w = (years - t1) / (t2 - t1)
            return y1 + w * (y2 - y1)
    return pts[-1][1]


# ---------------------------------------------------------------------------
# Equity lookups
# ---------------------------------------------------------------------------
def load_equities() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROJ, "equities.csv"))
    df = df.set_index("Instrument")
    return df


# ---------------------------------------------------------------------------
# Bond loading
# ---------------------------------------------------------------------------
def _parse_date(x) -> Optional[date]:
    if pd.isna(x) or x in ("", None):
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    s = str(x).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.split("T")[0], "%Y-%m-%d").date()
        except ValueError:
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
    return None


DEFAULT_DIV_YIELD = 0.0
DEFAULT_VOL_FALLBACK = 0.30


def build_universe(
    valuation_date: Optional[date] = None,
    require_market_price: bool = True,
    require_underlying: bool = True,
):
    """
    Returns list of dicts: {bond, mkt, mkt_px, ric, issuer, underlying_ric}.
    """
    bonds_df = pd.read_csv(os.path.join(PROJ, "bonds.csv"))
    equities = load_equities()
    curve = load_jgb_curve()
    val = valuation_date or date.today()

    results = []
    skipped = {"no_price": 0, "no_under": 0, "no_eq_data": 0, "matured": 0, "no_conv": 0}

    for _, r in bonds_df.iterrows():
        ric = str(r["RIC"])
        issuer = str(r["IssuerName"])
        mat = _parse_date(r["MaturityDate"])
        iss = _parse_date(r["IssueDate"])
        cp_yen = r.get("ConversionPrice")
        und = r.get("underlying_ric")
        # Prefer mid; fall back to (bid+ask)/2; then bid alone; then ask alone.
        mkt_px = r.get("Mid Price")
        if pd.isna(mkt_px) or (mkt_px is not None and mkt_px <= 0):
            bid = r.get("Bid Price")
            ask = r.get("Ask Price")
            if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                mkt_px = (float(bid) + float(ask)) / 2.0
            elif pd.notna(bid) and bid > 0:
                mkt_px = float(bid)
            elif pd.notna(ask) and ask > 0:
                mkt_px = float(ask)

        if mat is None or mat <= val:
            skipped["matured"] += 1
            continue
        if pd.isna(cp_yen) or cp_yen <= 0:
            skipped["no_conv"] += 1
            continue
        if require_market_price and (pd.isna(mkt_px) or mkt_px <= 0):
            skipped["no_price"] += 1
            continue
        if require_underlying and (pd.isna(und) or not str(und)):
            skipped["no_under"] += 1
            continue

        # Equity row
        if und in equities.index:
            eq = equities.loc[und]
            spot = float(eq.get("Price Close", np.nan))
            vol60 = eq.get("Volatility - 60 days", np.nan)
            divy = eq.get("Dividend yield", np.nan)
            rating = eq.get("Issuer Rating", "")
        else:
            skipped["no_eq_data"] += 1
            continue

        if pd.isna(spot) or spot <= 0:
            skipped["no_eq_data"] += 1
            continue

        sigma = float(vol60) / 100.0 if pd.notna(vol60) and vol60 > 0 else DEFAULT_VOL_FALLBACK
        div_y = float(divy) / 100.0 if pd.notna(divy) and divy > 0 else DEFAULT_DIV_YIELD
        # Prefer issuer CDS spread over rating-based proxy when available
        cds = cds_for_issuer(issuer)
        spread = cds if cds is not None else spread_for_rating(rating)
        spread_source = "CDS" if cds is not None else "Rating"
        # Tighten reset flag: only if issuance pattern suggests reset AND
        # issuer is weak credit (resets are ~always on sub-IG / unrated names).
        from credit import normalize as _norm_rating
        weak_credit = _norm_rating(rating) in ("", "BBB-", "BB+", "BB", "BB-",
                                                "B+", "B", "B-",
                                                "CCC+", "CCC", "CCC-")
        likely_reset = bool(r.get("likely_has_reset", False)) and weak_credit

        coupon = float(r.get("Coupon Rate") or 0.0) / 100.0  # rate is % from Refinitiv
        # Default: semi-annual for coupon-bearing JP bonds; for zero-coupon
        # bonds frequency is moot but keep at 2 for consistency.
        cf_raw = r.get("Coupon Frequency")
        coupon_freq = int(cf_raw) if pd.notna(cf_raw) and cf_raw else 2

        # Calls — use the explicit Call Date if present; otherwise apply the
        # standard JP CB pattern (issue + ~3y soft call at par with 130% trigger).
        calls = []
        cd = _parse_date(r.get("Call Date"))
        cprice = r.get("Call Price")
        if cd and pd.notna(cprice):
            calls.append(CallProvision(
                start=cd, end=mat, price=float(cprice), trigger_pct=130.0
            ))
        elif iss and bool(r.get("IsCallable")):
            # No explicit schedule but flagged callable -> apply pattern
            from datetime import timedelta
            soft_start = iss + timedelta(days=int(365 * 3))
            if soft_start < mat:
                calls.append(CallProvision(
                    start=soft_start, end=mat, price=100.0, trigger_pct=130.0
                ))

        # Puts
        puts = []
        pd_ = _parse_date(r.get("Put Date"))
        pprice = r.get("Put Price")
        if pd_ and pd.notna(pprice):
            puts.append(PutProvision(put_date=pd_, price=float(pprice)))

        bond = ConvertibleBond(
            isin=str(r.get("ISIN") or ric),
            issuer=issuer,
            underlying_ticker=str(und),
            coupon=coupon,
            coupon_freq=coupon_freq,
            maturity=mat,
            issue_date=iss or val,
            notional=100.0,
            conversion_price=float(cp_yen),
            currency="JPY",
            calls=calls,
            puts=puts,
            credit_rating=str(rating) if rating and not pd.isna(rating) else "NR",
        )

        years = (mat - val).days / 365.0
        rf = rf_for_tenor(curve, years)
        mkt = MarketData(
            valuation_date=val,
            spot=spot,
            sigma=sigma,
            r=rf,
            credit_spread=spread,
            div_yield=div_y,
        )

        results.append({
            "bond": bond,
            "mkt": mkt,
            "mkt_px": float(mkt_px),
            "ric": ric,
            "issuer": issuer,
            "underlying_ric": und,
            "rating": str(rating) if rating and not pd.isna(rating) else "NR",
            "spread_bp": int(spread * 10_000),
            "spread_source": spread_source,
            "likely_reset": likely_reset,
        })

    print(f"  loaded {len(results)} bonds for screening; skipped: {skipped}")
    return results


if __name__ == "__main__":
    res = build_universe()
    print(f"\nFirst 3 entries:")
    for r in res[:3]:
        b, m = r["bond"], r["mkt"]
        print(f"  {r['issuer']:25s} {b.isin}  cp={b.conversion_price}  spot={m.spot}  σ={m.sigma:.2%}  px={r['mkt_px']}")

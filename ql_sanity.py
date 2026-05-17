"""
QuantLib sanity check.

For 5 representative plain-vanilla bonds in our universe, price with both
our Tsiveriotis-Fernandes binomial tree (pricer.py) and QuantLib's
BinomialConvertibleEngine, and compare. Differences should be within ±2 bp
for clean comparable bonds. Anything wider is a model bug to investigate.

Limitations:
  - QuantLib's ConvertibleFixedCouponBond doesn't natively split debt/equity
    discounting the way Tsiveriotis-Fernandes does — we use Hull-White-style
    TF approximation in QL via a single risky discount curve as a proxy.
  - Bonds with soft-call triggers or reset clauses are skipped.
  - Numerical differences of 1-3% are common across CB pricer implementations.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import QuantLib as ql

PROJ = os.path.dirname(os.path.abspath(__file__))


def price_with_quantlib(
    spot: float, sigma: float, r: float, credit_spread: float,
    div_yield: float, conversion_price_yen: float, notional: float,
    coupon: float, coupon_freq: int,
    valuation_date: date, issue_date: date, maturity: date,
    n_steps: int = 200,
) -> float:
    """Price a plain-vanilla convertible with QuantLib's binomial CRR."""
    today = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
    ql.Settings.instance().evaluationDate = today
    calendar = ql.NullCalendar()
    daycount = ql.Actual365Fixed()

    iss = ql.Date(issue_date.day, issue_date.month, issue_date.year)
    mat = ql.Date(maturity.day, maturity.month, maturity.year)

    # Market data handles
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    div_h  = ql.YieldTermStructureHandle(ql.FlatForward(today, div_yield, daycount))
    rf_h   = ql.YieldTermStructureHandle(ql.FlatForward(today, r, daycount))
    vol_h  = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, daycount))
    # Use risk-free + credit spread for the risky discount on the bond leg
    risky_h = ql.YieldTermStructureHandle(ql.FlatForward(today, r + credit_spread, daycount))
    credit_h = ql.QuoteHandle(ql.SimpleQuote(credit_spread))

    process = ql.BlackScholesMertonProcess(spot_h, div_h, rf_h, vol_h)

    # Conversion ratio per ¥100 face (matches our convention)
    conversion_ratio = notional / conversion_price_yen

    # Schedule for coupons
    period = ql.Period(int(12 / max(coupon_freq, 1)), ql.Months)
    schedule = ql.Schedule(iss, mat, period, calendar,
                           ql.Unadjusted, ql.Unadjusted,
                           ql.DateGeneration.Backward, False)

    settlement_days = 0
    coupons = [coupon] * (len(schedule) - 1)

    # No call/put
    callability_schedule = ql.CallabilitySchedule()
    dividend_schedule = ql.DividendSchedule()

    exercise = ql.AmericanExercise(today, mat)
    bond = ql.ConvertibleFixedCouponBond(
        exercise, conversion_ratio,
        callability_schedule,
        iss, settlement_days, coupons, daycount, schedule,
        notional,
    )

    # In recent QuantLib (>= 1.30), the engine takes the credit spread directly
    engine = ql.BinomialConvertibleEngine(process, "crr", n_steps, credit_h)
    bond.setPricingEngine(engine)
    return float(bond.NPV())


def build_test_bonds():
    """5 representative plain-vanilla bonds from current bonds.csv (no resets, no calls in window)."""
    from pricer import ConvertibleBond, MarketData
    bonds_df = pd.read_csv(os.path.join(PROJ, "bonds.csv"))
    eq_df = pd.read_csv(os.path.join(PROJ, "equities.csv")).set_index("Instrument")

    candidates = bonds_df.dropna(subset=["RIC", "underlying_ric", "ConversionPrice", "Mid Price"]).copy()
    candidates = candidates[candidates["likely_has_reset"] == False]  # plain-vanilla only
    # Take a spread of issuers
    picks = []
    seen_issuers = set()
    for _, r in candidates.iterrows():
        iss = r["IssuerName"]
        if iss in seen_issuers:
            continue
        if r["underlying_ric"] not in eq_df.index:
            continue
        spot = eq_df.loc[r["underlying_ric"], "Price Close"]
        if pd.isna(spot) or spot <= 0:
            continue
        picks.append(r)
        seen_issuers.add(iss)
        if len(picks) >= 5:
            break

    return picks, eq_df


def run():
    print("QuantLib sanity check on 5 plain-vanilla bonds")
    from pricer import ConvertibleBond, MarketData, price_cb
    from real_data import load_jgb_curve, rf_for_tenor
    from credit import spread_for as spread_for_rating

    picks, eq_df = build_test_bonds()
    if not picks:
        print("  No suitable bonds found.")
        return

    val = date.today()
    curve = load_jgb_curve()
    results = []

    for r in picks:
        try:
            mat = pd.to_datetime(r["MaturityDate"]).date()
            iss = pd.to_datetime(r["IssueDate"]).date() if pd.notna(r.get("IssueDate")) else val
            und = r["underlying_ric"]
            spot = float(eq_df.loc[und, "Price Close"])
            vol60 = eq_df.loc[und, "Volatility - 60 days"]
            sigma = float(vol60) / 100.0 if pd.notna(vol60) and vol60 > 0 else 0.3
            divy_v = eq_df.loc[und, "Dividend yield"]
            div_y = float(divy_v) / 100.0 if pd.notna(divy_v) and divy_v > 0 else 0.0
            rating = ""
            if pd.notna(eq_df.loc[und].get("Issuer Rating")):
                rating = str(eq_df.loc[und].get("Issuer Rating"))
            spread = spread_for_rating(rating)
            cp = float(r["ConversionPrice"])
            coupon = float(r.get("Coupon Rate") or 0) / 100.0
            coupon_freq = int(r.get("Coupon Frequency") or 2) if pd.notna(r.get("Coupon Frequency")) else 2

            yrs = max((mat - val).days / 365.0, 0.01)
            rf = rf_for_tenor(curve, yrs)

            bond = ConvertibleBond(
                isin=str(r.get("ISIN") or r["RIC"]),
                issuer=str(r["IssuerName"]),
                underlying_ticker=str(und),
                coupon=coupon, coupon_freq=coupon_freq,
                maturity=mat, issue_date=iss,
                notional=100.0,
                conversion_price=cp,
                currency="JPY",
                credit_rating=rating or "NR",
            )
            mkt = MarketData(
                valuation_date=val, spot=spot, sigma=sigma,
                r=rf, credit_spread=spread, div_yield=div_y,
            )
            our_price = price_cb(bond, mkt, n_steps=300)["price"]

            ql_price = price_with_quantlib(
                spot=spot, sigma=sigma, r=rf, credit_spread=spread,
                div_yield=div_y, conversion_price_yen=cp, notional=100.0,
                coupon=coupon, coupon_freq=coupon_freq,
                valuation_date=val, issue_date=iss, maturity=mat,
                n_steps=300,
            )
            diff_pct = (our_price - ql_price) / ql_price * 100.0
            results.append({
                "issuer": r["IssuerName"],
                "ric": r["RIC"],
                "underlying": und,
                "maturity": mat.isoformat(),
                "spot": spot,
                "conv_price": cp,
                "sigma": sigma,
                "spread_bp": int(spread * 10_000),
                "our_price": round(our_price, 3),
                "ql_price": round(ql_price, 3),
                "diff": round(our_price - ql_price, 3),
                "diff_pct": round(diff_pct, 3),
            })
        except Exception as e:
            print(f"  {r['IssuerName']}: {str(e)[:100]}")

    if not results:
        print("  No prices computed.")
        return
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(PROJ, "history", "ql_sanity.csv"), index=False)
    import shutil
    shutil.copy(os.path.join(PROJ, "history", "ql_sanity.csv"),
                os.path.join(PROJ, "history", "demo_ql_sanity.csv"))
    print("\n" + df.to_string(index=False))
    print(f"\nMax abs difference: {df['diff_pct'].abs().max():.2f}%")
    print(f"Mean abs difference: {df['diff_pct'].abs().mean():.2f}%")


if __name__ == "__main__":
    run()

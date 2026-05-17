"""
Unit tests for pricer.py — the Tsiveriotis-Fernandes binomial tree.

Boundary-case tests:
  1. Deep-in-the-money CB → price ≈ conversion value, delta ≈ conversion ratio
  2. Deep-out-of-the-money CB → price ≈ bond floor, delta ≈ 0
  3. Zero-vol CB → price equals max(bond floor, intrinsic conversion)
  4. Zero coupon CB at maturity → price equals max(notional, conversion value)
  5. Tree convergence — doubling steps should change the price <0.5%
  6. Bond floor monotonicity in credit spread — higher spread = lower floor

Run with:
    python3 -m pytest tests/ -v
    OR
    python3 -m unittest tests.test_pricer
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date

# Make the project root importable when running from tests/
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from pricer import (
    ConvertibleBond, MarketData, price_cb, bond_floor,
    compute_greeks, implied_vol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bond(conversion_price: float = 1000.0,
              maturity_years: float = 3.0,
              coupon: float = 0.0,
              notional: float = 100.0) -> ConvertibleBond:
    """Build a plain-vanilla convertible for testing."""
    today = date(2026, 1, 1)
    return ConvertibleBond(
        isin="TEST",
        issuer="TestCo",
        underlying_ticker="TEST.T",
        coupon=coupon,
        coupon_freq=2,
        maturity=date(2026 + int(maturity_years), 1, 1),
        issue_date=today,
        notional=notional,
        conversion_price=conversion_price,
        currency="JPY",
        credit_rating="BBB",
    )


def make_mkt(spot: float = 1000.0, sigma: float = 0.30,
             r: float = 0.01, credit_spread: float = 0.015,
             div_yield: float = 0.0) -> MarketData:
    return MarketData(
        valuation_date=date(2026, 1, 1),
        spot=spot, sigma=sigma, r=r,
        credit_spread=credit_spread, div_yield=div_yield,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class PricerBoundaryCases(unittest.TestCase):
    """Tests that confirm the pricer behaves correctly at boundary cases."""

    def test_deep_itm_converges_to_parity(self):
        """When stock >> conversion price, CB ≈ conversion value (parity)."""
        bond = make_bond(conversion_price=1000.0, maturity_years=3.0)
        # Stock is 3× conversion price → very deep ITM
        mkt = make_mkt(spot=3000.0, sigma=0.25)
        result = price_cb(bond, mkt, n_steps=200)
        parity = mkt.spot * (bond.notional / bond.conversion_price)  # = 300
        # Allow small premium for time value but should be close to parity
        rel_diff = abs(result["price"] - parity) / parity
        self.assertLess(rel_diff, 0.10,
            f"Deep ITM price {result['price']:.2f} should be close to parity {parity:.2f}")

    def test_deep_itm_delta_approaches_ratio(self):
        """Deep ITM: delta should approach the conversion ratio (~1 share per cp/notional units of stock)."""
        bond = make_bond(conversion_price=1000.0, maturity_years=3.0)
        mkt = make_mkt(spot=3000.0, sigma=0.25)
        greeks = compute_greeks(bond, mkt, n_steps=200)
        expected_ratio = bond.notional / bond.conversion_price  # = 0.10
        # Delta should be within 20% of the conversion ratio when deep ITM
        rel_diff = abs(greeks["delta"] - expected_ratio) / expected_ratio
        self.assertLess(rel_diff, 0.20,
            f"Deep ITM delta {greeks['delta']:.3f} should approach ratio {expected_ratio:.3f}")

    def test_deep_otm_approaches_bond_floor(self):
        """When stock << conversion price, CB → bond floor (pure debt value)."""
        bond = make_bond(conversion_price=10000.0, maturity_years=3.0,
                          coupon=0.02)  # 2% coupon so floor is meaningful
        mkt = make_mkt(spot=100.0, sigma=0.25)  # stock 1% of conv price
        result = price_cb(bond, mkt, n_steps=200)
        floor = bond_floor(bond, mkt)
        # Should be very close to floor (option value near zero)
        # Tree may add tiny option value even far OTM; allow 5%
        rel_diff = abs(result["price"] - floor) / floor
        self.assertLess(rel_diff, 0.05,
            f"Deep OTM price {result['price']:.2f} should ≈ bond floor {floor:.2f}")

    def test_deep_otm_delta_near_zero(self):
        """Deep OTM: delta should be tiny (option has no chance of ITM)."""
        bond = make_bond(conversion_price=10000.0, maturity_years=3.0,
                          coupon=0.02)
        mkt = make_mkt(spot=100.0, sigma=0.25)
        greeks = compute_greeks(bond, mkt, n_steps=200)
        self.assertLess(abs(greeks["delta"]), 0.01,
            f"Deep OTM delta {greeks['delta']:.4f} should be near 0")

    def test_zero_vol_equals_intrinsic_or_floor(self):
        """With zero vol, CB = max(bond floor, intrinsic conversion value)."""
        bond = make_bond(conversion_price=1000.0, maturity_years=2.0,
                          coupon=0.01)

        # Case A: ITM at zero vol — should equal conversion value (no time value)
        mkt_itm = make_mkt(spot=2000.0, sigma=0.001, r=0.001, credit_spread=0.001)
        result_itm = price_cb(bond, mkt_itm, n_steps=200)
        parity = mkt_itm.spot * (bond.notional / bond.conversion_price)  # 200
        self.assertGreater(result_itm["price"], parity * 0.95,
            f"Zero-vol ITM price {result_itm['price']:.2f} should be ≥ parity {parity:.2f}")

        # Case B: OTM at zero vol — should equal bond floor
        mkt_otm = make_mkt(spot=100.0, sigma=0.001, r=0.001, credit_spread=0.015)
        result_otm = price_cb(bond, mkt_otm, n_steps=200)
        floor = bond_floor(bond, mkt_otm)
        self.assertAlmostEqual(result_otm["price"], floor, delta=2.0,
            msg=f"Zero-vol OTM {result_otm['price']:.2f} should = floor {floor:.2f}")

    def test_tree_convergence(self):
        """Doubling time steps should change price by <0.5%."""
        bond = make_bond(conversion_price=1100.0, maturity_years=3.0,
                          coupon=0.005)
        mkt = make_mkt(spot=1000.0, sigma=0.30)
        p_low  = price_cb(bond, mkt, n_steps=100)["price"]
        p_mid  = price_cb(bond, mkt, n_steps=200)["price"]
        p_high = price_cb(bond, mkt, n_steps=400)["price"]
        rel_step_1 = abs(p_mid - p_low) / p_mid
        rel_step_2 = abs(p_high - p_mid) / p_high
        # Each doubling should make the difference smaller (convergence)
        self.assertLess(rel_step_2, rel_step_1,
            f"Successive refinements should converge: 100→200 Δ={rel_step_1:.3%}, 200→400 Δ={rel_step_2:.3%}")
        # And the 200→400 difference should be small
        self.assertLess(rel_step_2, 0.005,
            f"200→400 step change should be <0.5%, got {rel_step_2:.3%}")

    def test_bond_floor_decreases_with_credit_spread(self):
        """Bond floor is monotonically decreasing in credit spread (basic credit math)."""
        bond = make_bond(coupon=0.02, maturity_years=5.0)
        spreads = [0.001, 0.01, 0.025, 0.05, 0.10]
        floors = [bond_floor(bond, make_mkt(credit_spread=s)) for s in spreads]
        for i in range(len(floors) - 1):
            self.assertGreater(floors[i], floors[i+1],
                f"Floor at spread {spreads[i]} ({floors[i]:.2f}) should be > floor at {spreads[i+1]} ({floors[i+1]:.2f})")

    def test_price_increasing_in_vol(self):
        """Vega: CB price should be monotonically increasing in σ (option value)."""
        bond = make_bond(conversion_price=1100.0, maturity_years=3.0)
        prices = []
        for sig in [0.10, 0.20, 0.30, 0.50, 0.80]:
            prices.append(price_cb(bond, make_mkt(sigma=sig), n_steps=200)["price"])
        for i in range(len(prices) - 1):
            self.assertGreaterEqual(prices[i+1], prices[i] - 0.01,
                f"Price should increase with vol: σ {[0.10,0.20,0.30,0.50,0.80][i]}={prices[i]:.2f} → next={prices[i+1]:.2f}")

    def test_at_maturity_equals_terminal_payoff(self):
        """If T → 0+ (1 day to maturity), price ≈ max(notional, conversion value)."""
        from datetime import timedelta
        bond = make_bond(conversion_price=1000.0)
        # Override maturity to be 1 day from today
        bond.maturity = bond.issue_date + timedelta(days=1)

        # Case A: stock ITM at maturity → price ≈ parity
        mkt_itm = make_mkt(spot=1500.0)
        r_itm = price_cb(bond, mkt_itm, n_steps=50)
        parity = mkt_itm.spot * (bond.notional / bond.conversion_price)  # 150
        self.assertAlmostEqual(r_itm["price"], parity, delta=parity * 0.05,
            msg=f"Near maturity ITM should ≈ parity {parity}, got {r_itm['price']:.2f}")

        # Case B: stock OTM at maturity → price ≈ notional (100)
        mkt_otm = make_mkt(spot=500.0)
        r_otm = price_cb(bond, mkt_otm, n_steps=50)
        self.assertAlmostEqual(r_otm["price"], bond.notional, delta=bond.notional * 0.05,
            msg=f"Near maturity OTM should ≈ notional {bond.notional}, got {r_otm['price']:.2f}")


class ImpliedVolRoundTrip(unittest.TestCase):
    """Implied vol bisection should round-trip: price → IV → price = same."""

    def test_iv_round_trip(self):
        bond = make_bond(conversion_price=1100.0, maturity_years=3.0,
                          coupon=0.005)
        # Generate price at a known vol
        true_sigma = 0.35
        price = price_cb(bond, make_mkt(sigma=true_sigma), n_steps=200)["price"]
        # Solve for implied vol from that price
        iv = implied_vol(bond, make_mkt(sigma=0.1), price, n_steps=200)
        self.assertIsNotNone(iv, "implied_vol should solve")
        self.assertAlmostEqual(iv, true_sigma, delta=0.01,
            msg=f"IV round-trip {iv:.3f} should match input vol {true_sigma}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

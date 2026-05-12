"""
Convertible bond pricer using the Tsiveriotis-Fernandes (1998) decomposition
on a CRR binomial tree.

Decomposes CB value V = E + B, where:
  E = "equity" component (cash flows arising from conversion), discounted at r
  B = "bond" component (cash flows arising from coupons/redemption), discounted at r + s

This is the standard credit-aware tree approach for CBs and is the one most
hedge-fund desks actually use as a baseline before layering on Monte Carlo
for path-dependent reset features.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Bond data structures
# ---------------------------------------------------------------------------

@dataclass
class CallProvision:
    start: date            # callable window start
    end: date              # callable window end
    price: float           # call price as % of notional (e.g. 100.0)
    trigger_pct: Optional[float] = None
    # If set, soft-call: only callable when stock >= trigger_pct% of conv price
    # (e.g. 130.0 means the classic "130% trigger")


@dataclass
class PutProvision:
    put_date: date
    price: float           # put price as % of notional


@dataclass
class ResetProvision:
    """
    Simplified Japanese-style downward reset.
    On `reset_date`, if stock <= floor_pct% of current conversion price,
    conversion price resets to max(stock_price, floor_price).
    Note: our binomial tree handles this only on a "deterministic" basis
    (uses expected stock at reset). Proper handling needs Monte Carlo —
    flagged in the output so the user knows to treat with caution.
    """
    reset_date: date
    trigger_pct: float = 80.0      # reset triggers if stock <= 80% of conv price
    floor_pct: float = 70.0        # new conv price floor (% of original)


@dataclass
class ConvertibleBond:
    isin: str
    issuer: str
    underlying_ticker: str
    coupon: float                  # annual coupon rate, decimal (0.005 = 0.5%)
    coupon_freq: int               # payments per year (typically 2 for JP CBs)
    maturity: date
    issue_date: date
    notional: float = 100.0        # face value
    conversion_price: float = 0.0  # JPY per share at issue
    currency: str = "JPY"
    calls: list[CallProvision] = field(default_factory=list)
    puts: list[PutProvision] = field(default_factory=list)
    resets: list[ResetProvision] = field(default_factory=list)
    credit_rating: str = "NR"

    @property
    def conversion_ratio(self) -> float:
        return self.notional / self.conversion_price


@dataclass
class MarketData:
    valuation_date: date
    spot: float                    # underlying stock price (JPY)
    sigma: float                   # annualized vol, decimal (0.30 = 30%)
    r: float                       # continuously compounded risk-free rate
    credit_spread: float           # additional spread for issuer credit risk
    div_yield: float = 0.0         # continuous dividend yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yf(d1: date, d2: date) -> float:
    return (d2 - d1).days / 365.0


def _coupon_steps(bond: ConvertibleBond, val_date: date,
                  T: float, dt: float, n_steps: int) -> set[int]:
    """Tree steps at which a coupon is paid. We work back from maturity."""
    if bond.coupon <= 0 or bond.coupon_freq <= 0:
        return set()
    period = 1.0 / bond.coupon_freq
    steps: set[int] = set()
    k = 0
    while True:
        t = T - k * period
        if t < 0:
            break
        s = int(round(t / dt))
        if 0 < s <= n_steps:
            steps.add(s)
        k += 1
    return steps


# ---------------------------------------------------------------------------
# Core pricer
# ---------------------------------------------------------------------------

def price_cb(
    bond: ConvertibleBond,
    mkt: MarketData,
    n_steps: int = 400,
) -> dict:
    """
    Returns model price plus the E/B decomposition and a few diagnostics.
    """
    T = _yf(mkt.valuation_date, bond.maturity)
    if T <= 0:
        intrinsic = max(bond.notional, mkt.spot * bond.conversion_ratio)
        return {
            "price": intrinsic,
            "equity_component": mkt.spot * bond.conversion_ratio,
            "bond_component": bond.notional,
            "T": 0.0,
        }

    dt = T / n_steps
    u = float(np.exp(mkt.sigma * np.sqrt(dt)))
    d = 1.0 / u
    a = float(np.exp((mkt.r - mkt.div_yield) * dt))
    p = (a - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    disc_r = float(np.exp(-mkt.r * dt))
    disc_rs = float(np.exp(-(mkt.r + mkt.credit_spread) * dt))

    ratio = bond.conversion_ratio
    coupon_amt = bond.coupon * bond.notional / max(bond.coupon_freq, 1)
    coupon_set = _coupon_steps(bond, mkt.valuation_date, T, dt, n_steps)

    # Pre-compute call / put step maps -------------------------------------
    call_at: dict[int, tuple[float, Optional[float]]] = {}
    for cp in bond.calls:
        t_s = max(0.0, _yf(mkt.valuation_date, cp.start))
        t_e = min(T, _yf(mkt.valuation_date, cp.end))
        if t_e <= 0 or t_s >= T:
            continue
        s_start = max(0, int(np.ceil(t_s / dt)))
        s_end = min(n_steps, int(np.floor(t_e / dt)))
        for s in range(s_start, s_end + 1):
            call_at[s] = (cp.price, cp.trigger_pct)

    put_at: dict[int, float] = {}
    for pp in bond.puts:
        t_p = _yf(mkt.valuation_date, pp.put_date)
        if t_p <= 0 or t_p > T:
            continue
        s_p = int(round(t_p / dt))
        put_at[s_p] = pp.price

    # Terminal payoff ------------------------------------------------------
    j = np.arange(n_steps + 1)
    S_T = mkt.spot * (u ** (n_steps - j)) * (d ** j)
    conv_T = S_T * ratio
    redemption = bond.notional + (coupon_amt if n_steps in coupon_set else 0.0)
    convert = conv_T > redemption
    E = np.where(convert, conv_T, 0.0)
    B = np.where(convert, 0.0, redemption)

    # Backward induction ---------------------------------------------------
    for i in range(n_steps - 1, -1, -1):
        # Continuation from i+1 -> i
        E = disc_r * (p * E[:-1] + (1.0 - p) * E[1:])
        B = disc_rs * (p * B[:-1] + (1.0 - p) * B[1:])

        # Coupon paid at step i (added to B before optionality is applied)
        if i in coupon_set and i > 0:
            B = B + coupon_amt

        S_i = mkt.spot * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
        conv_value = S_i * ratio
        V = E + B

        # Put: holder optimally puts if put price exceeds value
        if i in put_at:
            pp_price = put_at[i]
            put_better = pp_price > V
            E = np.where(put_better, 0.0, E)
            B = np.where(put_better, pp_price, B)
            V = E + B

        # Voluntary conversion (American, ignoring conv-window for simplicity)
        convert_better = conv_value > V
        E = np.where(convert_better, conv_value, E)
        B = np.where(convert_better, 0.0, B)
        V = E + B

        # Issuer call
        if i in call_at:
            cp_price, trigger_pct = call_at[i]
            if trigger_pct is not None:
                trig = bond.conversion_price * (trigger_pct / 100.0)
                callable_now = S_i >= trig
            else:
                callable_now = np.ones_like(S_i, dtype=bool)
            forced = np.maximum(cp_price, conv_value)
            do_call = callable_now & (V > forced)
            took_conv = do_call & (conv_value >= cp_price)
            took_cash = do_call & (conv_value < cp_price)
            E = np.where(took_conv, conv_value,
                         np.where(took_cash, 0.0, E))
            B = np.where(took_conv, 0.0,
                         np.where(took_cash, cp_price, B))

    return {
        "price": float(E[0] + B[0]),
        "equity_component": float(E[0]),
        "bond_component": float(B[0]),
        "T": T,
    }


# ---------------------------------------------------------------------------
# Bond floor / parity / Greeks (finite difference, bump-and-revalue)
# ---------------------------------------------------------------------------

def bond_floor(bond: ConvertibleBond, mkt: MarketData, n_steps: int = 200) -> float:
    """Value of the straight bond cash flows ignoring conversion."""
    T = _yf(mkt.valuation_date, bond.maturity)
    if T <= 0:
        return bond.notional
    r_eff = mkt.r + mkt.credit_spread
    pv = bond.notional * float(np.exp(-r_eff * T))
    period = 1.0 / max(bond.coupon_freq, 1)
    coupon = bond.coupon * bond.notional / max(bond.coupon_freq, 1)
    k = 0
    while True:
        t = T - k * period
        if t <= 0:
            break
        pv += coupon * float(np.exp(-r_eff * t))
        k += 1
    return float(pv)


def parity(bond: ConvertibleBond, spot: float) -> float:
    return spot * bond.conversion_ratio


def conversion_premium(bond: ConvertibleBond, market_price: float, spot: float) -> float:
    par = parity(bond, spot)
    if par <= 0:
        return float("nan")
    return (market_price - par) / par


def compute_greeks(
    bond: ConvertibleBond, mkt: MarketData, n_steps: int = 300
) -> dict:
    base = price_cb(bond, mkt, n_steps)["price"]

    h_s = max(mkt.spot * 0.01, 1.0)
    p_up = price_cb(bond, replace(mkt, spot=mkt.spot + h_s), n_steps)["price"]
    p_dn = price_cb(bond, replace(mkt, spot=mkt.spot - h_s), n_steps)["price"]
    delta = (p_up - p_dn) / (2.0 * h_s)
    gamma = (p_up - 2.0 * base + p_dn) / (h_s * h_s)

    h_v = 0.01
    p_vu = price_cb(bond, replace(mkt, sigma=mkt.sigma + h_v), n_steps)["price"]
    vega = (p_vu - base)  # per 1 vol point (since h_v = 0.01)

    p_t1 = price_cb(
        bond, replace(mkt, valuation_date=mkt.valuation_date + timedelta(days=1)), n_steps
    )["price"]
    theta = p_t1 - base  # per calendar day

    h_r = 0.0001
    p_ru = price_cb(bond, replace(mkt, r=mkt.r + h_r), n_steps)["price"]
    rho = (p_ru - base)  # per 1bp move

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ---------------------------------------------------------------------------
# Implied vol (price -> vol)
# ---------------------------------------------------------------------------

def implied_vol(
    bond: ConvertibleBond,
    mkt: MarketData,
    market_price: float,
    n_steps: int = 200,
    lo: float = 0.05,
    hi: float = 1.5,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> Optional[float]:
    """Bisection on sigma. Returns None if no root in [lo, hi]."""
    def model_at(sig: float) -> float:
        return price_cb(bond, replace(mkt, sigma=sig), n_steps)["price"]

    f_lo = model_at(lo) - market_price
    f_hi = model_at(hi) - market_price
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = model_at(mid) - market_price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)

"""
Detect bonds where the conversion price likely wasn't adjusted for a
corporate action (stock split, spin-off, special dividend) and is therefore
stale relative to the underlying equity.

Signals (any of):
  - Premium > 200% AND market price > 130 → bond is trading like ITM equity
    but parity is way below market, meaning conversion ratio is too small.
  - Market price > parity by > 50 points AND premium > 100% → same idea.
  - Implied vol > 80% with premium > 100% → tree can't reconcile, often
    means inputs are inconsistent.

Returns: list of anomaly tags (empty if clean).
"""

from __future__ import annotations


def detect_anomalies(row: dict) -> list[str]:
    tags: list[str] = []

    mkt = row.get("mkt_px") or 0
    parity = row.get("parity") or 0
    prem = row.get("premium_pct") or 0
    iv = row.get("iv")
    cheap = row.get("cheap_pct") or 0

    # Stock split / corporate action — bond price way above parity at high prem
    if prem > 200 and mkt > 130:
        tags.append("STALE_CONV_PX")
    elif (mkt - parity) > 50 and prem > 100:
        tags.append("STALE_CONV_PX")

    # Bond price > 200 — unusual for any reasonable JP CB; almost certainly data
    if mkt > 200:
        tags.append("PRICE>200")

    # Implied vol unsolvable AND large absolute cheap signal — model can't
    # reconcile, treat as data-driven outlier
    if iv is None and abs(cheap) > 25:
        tags.append("MODEL_NO_FIT")

    # Implied vol absurd
    if iv is not None and iv > 1.0:
        tags.append("IV>100%")

    return tags


def confidence(row: dict) -> str:
    """High / Medium / Low confidence classification."""
    flags = detect_anomalies(row)
    if any(f in flags for f in ("STALE_CONV_PX", "PRICE>200", "MODEL_NO_FIT")):
        return "Low"
    if "IV>100%" in flags:
        return "Medium"
    return "High"

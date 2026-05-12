"""
Issuer-rating → credit-spread mapping for Japanese corporates.

Spreads are conservative current-environment estimates (April 2026) for
JPY-denominated senior unsecured corporate paper, in basis points over
the matched JGB tenor. Refresh quarterly against actual issuer YTW data
or CDS quotes when available.

The TR.IssuerRating field from Refinitiv mixes long-term and short-term
ratings — short-term codes ('A1+', 'a-1', etc.) are mapped to their
long-term equivalents below.
"""

# Long-term rating → spread in decimal (e.g. 0.0050 = 50bp)
LT_SPREADS_BP = {
    "AAA": 25, "AA+": 35, "AA": 45, "AA-": 55,
    "A+":  70, "A":   90, "A-":  120,
    "BBB+": 150, "BBB": 190, "BBB-": 240,
    "BB+":  330, "BB":  430, "BB-":  550,
    "B+":   720, "B":   900, "B-":  1100,
    "CCC+": 1500, "CCC": 1900, "CCC-": 2400,
}

# Short-term ratings → conservative LT equivalent
ST_TO_LT = {
    "A1+": "AA-", "A-1+": "AA-", "a-1+": "AA-",
    "A1":  "A",   "A-1":  "A",   "a-1":  "A",
    "A2":  "BBB+", "A-2":  "BBB+", "a-2":  "BBB+",
    "A3":  "BBB-", "A-3":  "BBB-", "a-3":  "BBB-",
    "P1":  "A+",  "P-1":  "A+",
    "P2":  "BBB+", "P-2":  "BBB+",
    "P3":  "BBB-", "P-3":  "BBB-",
}

# Default for unrated / withdrawn / unknown
DEFAULT_RATING = "BB"
DEFAULT_BP = LT_SPREADS_BP[DEFAULT_RATING]


def normalize(rating: str) -> str:
    if rating is None:
        return ""
    r = str(rating).strip()
    if not r or r.upper() in {"NR", "WR", "WD"}:
        return ""
    # Strip outlook indicators like "A+ *", "A (CWN)"
    r = r.split(" ")[0].split("(")[0].split("*")[0].strip()
    if r in ST_TO_LT:
        return ST_TO_LT[r]
    if r in LT_SPREADS_BP:
        return r
    # Try uppercase
    if r.upper() in ST_TO_LT:
        return ST_TO_LT[r.upper()]
    if r.upper() in LT_SPREADS_BP:
        return r.upper()
    # Try Moody's notation: Baa1 → BBB+, A2 → A, etc.
    moody_map = {
        "Aaa": "AAA", "Aa1": "AA+", "Aa2": "AA", "Aa3": "AA-",
        "A1": "A+",   "A2":  "A",   "A3":  "A-",
        "Baa1": "BBB+", "Baa2": "BBB", "Baa3": "BBB-",
        "Ba1":  "BB+",  "Ba2":  "BB",  "Ba3":  "BB-",
        "B1":   "B+",   "B2":   "B",   "B3":   "B-",
        "Caa1": "CCC+", "Caa2": "CCC", "Caa3": "CCC-",
    }
    if r in moody_map:
        return moody_map[r]
    return ""


def spread_for(rating: str) -> float:
    """Returns spread as decimal (e.g. 0.0050)."""
    norm = normalize(rating)
    if not norm:
        return DEFAULT_BP / 10_000.0
    return LT_SPREADS_BP.get(norm, DEFAULT_BP) / 10_000.0

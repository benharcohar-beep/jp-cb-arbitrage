"""
Render all dashboard pages as static HTML into docs/ for GitHub Pages.

Output structure:
    docs/
        index.html                      — universe page
        watchlist.html                  — empty shell; populated client-side via localStorage
        alerts.html                     — historical alerts log
        backtest.html                   — backtest results
        how-it-works/index.html         — methodology
        bond/<RIC>/index.html           — per-bond detail
        static/app.css                  — styling
        api/snapshot.json               — full universe as JSON (used by client JS)
        api/bond_history/<RIC>.json     — per-bond history

Run: python3 generate_static.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
from datetime import date
from urllib.parse import quote

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

PROJ = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(PROJ, "docs")
SNAPDIR = os.path.join(PROJ, "snapshots")
HISTDIR = os.path.join(PROJ, "history")


# ---------------------------------------------------------------------------
def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _ig(rating: str) -> str:
    from credit import normalize as _n
    norm = _n(rating)
    if not norm:
        return "nr"
    ig = {"AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-"}
    return "ig" if norm in ig else "sub"


def latest_snapshot_path() -> str:
    """Prefer demo_screen.csv; otherwise latest screen_*.csv."""
    demo = os.path.join(SNAPDIR, "demo_screen.csv")
    if os.path.exists(demo):
        return demo
    snaps = sorted(glob.glob(os.path.join(SNAPDIR, "screen_*.csv")))
    return snaps[-1] if snaps else ""


def hist_path(name: str) -> str:
    demo = os.path.join(HISTDIR, f"demo_{name}.csv")
    real = os.path.join(HISTDIR, f"{name}.csv")
    return demo if os.path.exists(demo) else real


# ---------------------------------------------------------------------------
def setup_docs():
    if os.path.exists(DOCS):
        shutil.rmtree(DOCS)
    os.makedirs(DOCS, exist_ok=True)
    # Copy static assets
    shutil.copytree(os.path.join(PROJ, "static"), os.path.join(DOCS, "static"))


def setup_jinja() -> Environment:
    env = Environment(
        loader=FileSystemLoader(os.path.join(PROJ, "templates")),
        autoescape=select_autoescape(["html"]),
    )
    # Mark this as static-site mode so templates can hide server-only widgets
    env.globals["DEMO_MODE"] = True
    env.globals["STATIC_SITE"] = True
    return env


def write(path: str, body: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(body)


def safe_ric(ric: str) -> str:
    """RICs contain '=' — encode for path safety."""
    return ric.replace("=", "_eq_").replace("/", "_")


# ---------------------------------------------------------------------------
def render_index(env, df: pd.DataFrame, snap_name: str):
    df = df.sort_values("cheap_pct", ascending=False).copy()
    df["tier"] = df["rating"].fillna("").apply(_ig)
    rows = [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]

    # add safe_ric to each row for path-safe URLs
    for r in rows:
        r["safe_ric"] = safe_ric(r.get("ric", ""))

    top = [r for r in rows if (r.get("cheap_pct") or 0) >= 5
           and r.get("confidence") != "Low"][:6]

    n_rated  = sum(1 for r in rows if r.get("rating") not in ("NR", "WR", "", None))
    n_reset  = sum(1 for r in rows if r.get("reset_flag") == "RESET")
    n_alerts = sum(1 for r in rows if (r.get("cheap_pct") or 0) >= 5)
    spreads  = [r.get("spread_bp") for r in rows if r.get("spread_bp") is not None]
    summary  = {
        "snapshot":       snap_name,
        "n_bonds":        len(rows),
        "n_rated":        n_rated,
        "n_reset":        n_reset,
        "n_alerts":       n_alerts,
        "median_spread":  int(sorted(spreads)[len(spreads)//2]) if spreads else 0,
    }
    counts = {
        "all":      len(rows),
        "ig":       sum(1 for r in rows if r.get("tier") == "ig"),
        "sub":      sum(1 for r in rows if r.get("tier") == "sub"),
        "nr":       sum(1 for r in rows if r.get("tier") == "nr"),
        "reset":    n_reset,
        "anomaly":  sum(1 for r in rows if r.get("confidence") == "Low"),
    }
    html = env.get_template("index.html").render(
        rows=rows, summary=summary, top=top, counts=counts,
        new_rics=[], dropped_rics=[], ROOT="",
    )
    write(os.path.join(DOCS, "index.html"), html)


def render_bond_details(env, df: pd.DataFrame):
    for _, r in df.iterrows():
        bond = {k: _clean(v) for k, v in r.to_dict().items()}
        bond["safe_ric"] = safe_ric(bond["ric"])
        html = env.get_template("bond.html").render(bond=bond, chart="[]", ROOT="../../")
        out_path = os.path.join(DOCS, "bond", safe_ric(r["ric"]), "index.html")
        write(out_path, html)


def render_watchlist(env):
    """Static shell. Client JS reads localStorage and fetches snapshot.json."""
    html = env.get_template("watchlist.html").render(
        rows=[], rics=[], snapshot="demo · cached",
        portfolio={"n": 0, "total_delta_shares": 0,
                   "total_vega": 0.0, "total_theta": 0.0, "avg_cheap": 0.0},
        ROOT="",
    )
    write(os.path.join(DOCS, "watchlist.html"), html)


def render_alerts(env):
    rows = []
    for p in sorted(glob.glob(os.path.join(SNAPDIR, "screen_*.csv")) +
                    glob.glob(os.path.join(SNAPDIR, "demo_screen.csv"))):
        try:
            d = pd.read_csv(p)
            ts = os.path.basename(p).replace("screen_", "").replace(".csv", "")
            cheap = d[(d["cheap_pct"] >= 5) & (d["reset_flag"] != "RESET")]
            for _, rr in cheap.iterrows():
                rows.append({
                    "snapshot": ts,
                    "issuer":   rr["issuer"],
                    "ric":      rr["ric"],
                    "safe_ric": safe_ric(rr["ric"]),
                    "rating":   rr.get("rating", "NR"),
                    "cheap_pct": rr["cheap_pct"],
                    "mkt_px":    rr["mkt_px"],
                    "model_px":  rr["model_px"],
                })
        except Exception:
            continue
    rows.sort(key=lambda x: (x["snapshot"], -x["cheap_pct"]), reverse=True)
    html = env.get_template("alerts.html").render(rows=rows, ROOT="")
    write(os.path.join(DOCS, "alerts.html"), html)


def render_backtest(env):
    summary, rets, panel_stats, buckets = [], [], {}, []
    hedged_buckets, hedged_overall = [], {}

    for name, target in [
        ("signal_summary",  "summary"),
        ("signal_bucket",   "buckets"),
        ("hedged_summary",  "hedged_buckets"),
    ]:
        p = hist_path(name)
        if os.path.exists(p):
            vals = pd.read_csv(p).to_dict("records")
            if target == "summary": summary = vals
            elif target == "buckets": buckets = vals
            elif target == "hedged_buckets": hedged_buckets = vals

    p = hist_path("hedged_overall")
    if os.path.exists(p):
        try:
            ov = pd.read_csv(p, header=None, names=["k", "v"])
            hedged_overall = dict(zip(ov["k"], ov["v"]))
        except Exception:
            hedged_overall = {}

    p = hist_path("panel")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d_clean = d[(d["cheap_pct"] >= -25) & (d["cheap_pct"] <= 25)]
        panel_stats = {
            "n_rows":       len(d),
            "n_bonds":      d["ric"].nunique(),
            "date_range":   f"{d['date'].min()} → {d['date'].max()}",
            "avg_cheap":    round(d_clean["cheap_pct"].mean(), 2),
            "median_cheap": round(d_clean["cheap_pct"].median(), 2),
        }
        try:
            rets_csv = hist_path("signal_returns")
            if os.path.exists(rets_csv):
                rets = pd.read_csv(rets_csv).sort_values(
                    "signal_cheap", ascending=False
                ).head(40).to_dict("records")
        except Exception:
            pass

    html = env.get_template("backtest.html").render(
        summary=summary, rets=rets, panel_stats=panel_stats,
        buckets=buckets, hedged_buckets=hedged_buckets,
        hedged_overall=hedged_overall, ROOT="",
    )
    write(os.path.join(DOCS, "backtest.html"), html)


def render_methodology(env):
    html = env.get_template("methodology.html").render(ROOT="")
    write(os.path.join(DOCS, "how-it-works.html"), html)


def render_api_snapshot(df: pd.DataFrame):
    """Expose the universe as JSON for the watchlist page's client JS."""
    df = df.copy()
    df["safe_ric"] = df["ric"].apply(safe_ric)
    rows = [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    write(os.path.join(DOCS, "api", "snapshot.json"),
          json.dumps({"snapshot": "demo · cached", "rows": rows}))


# ---------------------------------------------------------------------------
def main():
    snap = latest_snapshot_path()
    if not snap:
        print("No snapshot found. Aborting.")
        return
    df = pd.read_csv(snap)
    print(f"Loaded snapshot: {os.path.basename(snap)} ({len(df)} bonds)")

    setup_docs()
    env = setup_jinja()

    render_index(env, df, os.path.basename(snap))
    print(f"  ✓ index.html")
    render_bond_details(env, df)
    print(f"  ✓ {len(df)} bond detail pages under bond/")
    render_watchlist(env)
    print(f"  ✓ watchlist.html")
    render_alerts(env)
    print(f"  ✓ alerts.html")
    render_backtest(env)
    print(f"  ✓ backtest.html")
    render_methodology(env)
    print(f"  ✓ how-it-works.html")
    render_api_snapshot(df)
    print(f"  ✓ api/snapshot.json")

    # GitHub Pages needs a .nojekyll to serve files starting with underscores etc.
    write(os.path.join(DOCS, ".nojekyll"), "")
    print(f"\nStatic site ready in docs/")


if __name__ == "__main__":
    main()

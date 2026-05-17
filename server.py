"""
FastAPI dashboard for the JP CB arb screen.

Routes:
  /              — universe table, sortable, color-coded by cheap%
  /bond/{ric}    — single-bond detail + cheapness history
  /alerts        — log of high-confidence cheap signals over time
  /api/snapshot  — JSON of latest screen
  /api/run       — trigger refresh + screen on demand
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJ = os.path.dirname(os.path.abspath(__file__))
SNAPDIR = os.path.join(PROJ, "snapshots")
WATCHLIST_PATH = os.path.join(PROJ, "watchlist.json")

# Demo mode: when DEMO_MODE=1 (set by Render or any hosting), read from the
# committed demo_*.csv files instead of live snapshots. Lets the public demo
# render the dashboard without requiring Refinitiv data on the host.
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

app = FastAPI(title="JP CB Arb Dashboard")
app.mount("/static", StaticFiles(directory=os.path.join(PROJ, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(PROJ, "templates"))
templates.env.globals["DEMO_MODE"] = DEMO_MODE


# --- helpers ---------------------------------------------------------------

def list_snapshots() -> list[str]:
    if DEMO_MODE:
        demo = os.path.join(SNAPDIR, "demo_screen.csv")
        return [demo] if os.path.exists(demo) else []
    return sorted(glob.glob(os.path.join(SNAPDIR, "screen_*.csv")))


def load_latest() -> tuple[pd.DataFrame, str]:
    snaps = list_snapshots()
    if not snaps:
        return pd.DataFrame(), ""
    path = snaps[-1]
    df = pd.read_csv(path)
    return df, "demo · cached snapshot" if DEMO_MODE else os.path.basename(path)


def history_for_ric(ric: str) -> pd.DataFrame:
    rows = []
    for p in list_snapshots():
        try:
            df = pd.read_csv(p)
            sub = df[df["ric"] == ric]
            if not sub.empty:
                ts = os.path.basename(p).replace("screen_", "").replace(".csv", "")
                row = sub.iloc[0].to_dict()
                row["snapshot"] = ts
                rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


# --- routes ----------------------------------------------------------------

def _ig(rating: str) -> str:
    """Return tier classification: 'ig', 'sub', or 'nr'."""
    from credit import normalize as _n
    norm = _n(rating)
    if not norm:
        return "nr"
    ig = {"AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-"}
    return "ig" if norm in ig else "sub"


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    df, snap = load_latest()
    if df.empty:
        return templates.TemplateResponse("empty.html", {"request": request})
    df = df.sort_values("cheap_pct", ascending=False)
    df["tier"] = df["rating"].fillna("").apply(_ig)
    rows = [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]

    # top alerts (≥5% cheap, not low-confidence)
    top = [r for r in rows if (r.get("cheap_pct") or 0) >= 5
           and r.get("confidence") != "Low"][:6]

    # diff vs prior
    snaps = list_snapshots()
    new_rics, dropped_rics = [], []
    if len(snaps) >= 2:
        prior = pd.read_csv(snaps[-2])
        cur_rics = set(df["ric"])
        prior_rics = set(prior["ric"])
        new_set = cur_rics - prior_rics
        drop_set = prior_rics - cur_rics
        new_rics = [r for r in rows if r["ric"] in new_set][:8]
        dropped_rics = prior[prior["ric"].isin(drop_set)].head(8).to_dict("records")

    n_rated = sum(1 for r in rows if r.get("rating") not in ("NR", "WR", "", None))
    n_reset = sum(1 for r in rows if r.get("reset_flag") == "RESET")
    n_alerts = sum(1 for r in rows if (r.get("cheap_pct") or 0) >= 5)
    spreads = [r.get("spread_bp") for r in rows if r.get("spread_bp") is not None]

    summary = {
        "snapshot": snap,
        "n_bonds": len(rows),
        "n_rated": n_rated,
        "n_reset": n_reset,
        "n_alerts": n_alerts,
        "median_spread": int(sorted(spreads)[len(spreads)//2]) if spreads else 0,
    }
    counts = {
        "all": len(rows),
        "ig":  sum(1 for r in rows if r.get("tier") == "ig"),
        "sub": sum(1 for r in rows if r.get("tier") == "sub"),
        "nr":  sum(1 for r in rows if r.get("tier") == "nr"),
        "reset": n_reset,
        "anomaly": sum(1 for r in rows if r.get("confidence") == "Low"),
    }

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "rows": rows, "summary": summary,
         "top": top, "counts": counts,
         "new_rics": new_rics, "dropped_rics": dropped_rics},
    )


@app.get("/bond/{ric}", response_class=HTMLResponse)
def bond_detail(request: Request, ric: str):
    df, snap = load_latest()
    if df.empty:
        raise HTTPException(404, "no snapshots")
    sub = df[df["ric"] == ric]
    if sub.empty:
        raise HTTPException(404, f"bond {ric} not in latest screen")
    bond = sub.iloc[0].to_dict()
    hist = history_for_ric(ric)
    chart = []
    if not hist.empty:
        for _, h in hist.iterrows():
            chart.append({
                "t": h["snapshot"],
                "mkt": float(h["mkt_px"]) if pd.notna(h.get("mkt_px")) else None,
                "model": float(h["model_px"]) if pd.notna(h.get("model_px")) else None,
                "cheap": float(h["cheap_pct"]) if pd.notna(h.get("cheap_pct")) else None,
            })
    return templates.TemplateResponse(
        "bond.html", {"request": request, "bond": bond, "chart": json.dumps(chart)}
    )


@app.get("/how-it-works", response_class=HTMLResponse)
def methodology_view(request: Request):
    return templates.TemplateResponse("methodology.html", {"request": request})


@app.get("/glossary", response_class=HTMLResponse)
def glossary_view(request: Request):
    return templates.TemplateResponse("glossary.html", {"request": request})


@app.get("/backtest", response_class=HTMLResponse)
def backtest_view(request: Request):
    histdir = os.path.join(PROJ, "history")
    prefix = "demo_" if DEMO_MODE else ""
    summary_path = os.path.join(histdir, f"{prefix}signal_summary.csv")
    rets_path = os.path.join(histdir, f"{prefix}signal_returns.csv")
    panel_path = os.path.join(histdir, f"{prefix}panel.csv")

    bucket_path = os.path.join(histdir, f"{prefix}signal_bucket.csv")
    hedged_summ_path = os.path.join(histdir, f"{prefix}hedged_summary.csv")
    hedged_overall_path = os.path.join(histdir, f"{prefix}hedged_overall.csv")
    summary, rets, panel_stats, buckets = [], [], {}, []
    hedged_buckets, hedged_overall = [], {}
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path).to_dict("records")
    if os.path.exists(bucket_path):
        buckets = pd.read_csv(bucket_path).to_dict("records")
    if os.path.exists(hedged_summ_path):
        hedged_buckets = pd.read_csv(hedged_summ_path).to_dict("records")
    if os.path.exists(hedged_overall_path):
        try:
            ov = pd.read_csv(hedged_overall_path, header=None, names=["k","v"])
            hedged_overall = dict(zip(ov["k"], ov["v"]))
        except Exception:
            hedged_overall = {}
    if os.path.exists(rets_path):
        rets = pd.read_csv(rets_path).sort_values("signal_cheap", ascending=False).head(40).to_dict("records")
    if os.path.exists(panel_path):
        p = pd.read_csv(panel_path)
        # exclude obvious anomalies (cheap > 25%) from panel stats
        p_clean = p[(p["cheap_pct"] >= -25) & (p["cheap_pct"] <= 25)]
        panel_stats = {
            "n_rows": len(p),
            "n_bonds": p["ric"].nunique(),
            "date_range": f"{p['date'].min()} → {p['date'].max()}",
            "avg_cheap": round(p_clean["cheap_pct"].mean(), 2),
            "median_cheap": round(p_clean["cheap_pct"].median(), 2),
        }

    return templates.TemplateResponse(
        "backtest.html",
        {"request": request, "summary": summary, "rets": rets,
         "panel_stats": panel_stats, "buckets": buckets,
         "hedged_buckets": hedged_buckets, "hedged_overall": hedged_overall},
    )


@app.get("/api/bond_history/{ric:path}")
def api_bond_history(ric: str):
    """Return historical model vs market for one bond — used by the bond detail page."""
    panel_path = os.path.join(PROJ, "history", "panel.csv")
    if not os.path.exists(panel_path):
        return JSONResponse({"rows": []})
    df = pd.read_csv(panel_path)
    sub = df[df["ric"] == ric].sort_values("date")
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "date": str(r["date"]),
            "mkt": _clean(r["mkt_px"]),
            "model": _clean(r["model_px"]),
            "cheap": _clean(r["cheap_pct"]),
            "spot": _clean(r["spot"]),
            "sigma": _clean(r["sigma"]),
        })
    return JSONResponse({"rows": rows})


@app.get("/alerts", response_class=HTMLResponse)
def alerts(request: Request):
    """Aggregate alert log across all snapshots."""
    rows = []
    for p in list_snapshots():
        try:
            df = pd.read_csv(p)
            ts = os.path.basename(p).replace("screen_", "").replace(".csv", "")
            cheap = df[(df["cheap_pct"] >= 5) & (df["reset_flag"] != "RESET")]
            for _, r in cheap.iterrows():
                rows.append({
                    "snapshot": ts,
                    "issuer": r["issuer"],
                    "ric": r["ric"],
                    "rating": r.get("rating", "NR"),
                    "cheap_pct": r["cheap_pct"],
                    "mkt_px": r["mkt_px"],
                    "model_px": r["model_px"],
                })
        except Exception:
            continue
    rows.sort(key=lambda x: (x["snapshot"], -x["cheap_pct"]), reverse=True)
    return templates.TemplateResponse(
        "alerts.html", {"request": request, "rows": rows}
    )


def _clean(v):
    import math
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


@app.get("/api/snapshot")
def api_snapshot():
    df, snap = load_latest()
    if df.empty:
        return JSONResponse({"snapshot": None, "rows": []})
    rows = [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    return JSONResponse({"snapshot": snap, "rows": rows})


def _load_watchlist() -> list[str]:
    if not os.path.exists(WATCHLIST_PATH):
        return []
    try:
        return json.load(open(WATCHLIST_PATH))
    except Exception:
        return []


def _save_watchlist(rics: list[str]):
    with open(WATCHLIST_PATH, "w") as f:
        json.dump(sorted(set(rics)), f, indent=2)


@app.get("/watchlist", response_class=HTMLResponse)
def watchlist_view(request: Request):
    df, snap = load_latest()
    rics = _load_watchlist()
    selected = df[df["ric"].isin(rics)].to_dict("records") if not df.empty else []

    # Aggregate portfolio Greeks (assuming ¥100M face per position)
    total_delta = sum((r.get("delta") or 0) for r in selected) * 1_000_000
    total_vega  = sum((r.get("vega")  or 0) for r in selected)
    total_theta = sum((r.get("theta") or 0) for r in selected)
    avg_cheap   = (sum((r.get("cheap_pct") or 0) for r in selected) / len(selected)) if selected else 0
    portfolio = {
        "n": len(selected), "total_delta_shares": int(total_delta),
        "total_vega": round(total_vega, 2), "total_theta": round(total_theta, 2),
        "avg_cheap": round(avg_cheap, 2),
    }
    return templates.TemplateResponse(
        "watchlist.html",
        {"request": request, "rows": selected, "rics": rics,
         "portfolio": portfolio, "snapshot": snap},
    )


@app.post("/api/watchlist/{ric}")
def add_watchlist(ric: str):
    rics = _load_watchlist()
    if ric not in rics:
        rics.append(ric)
        _save_watchlist(rics)
    return {"watchlist": rics}


@app.delete("/api/watchlist/{ric}")
def remove_watchlist(ric: str):
    rics = [r for r in _load_watchlist() if r != ric]
    _save_watchlist(rics)
    return {"watchlist": rics}


@app.post("/api/run")
def api_run():
    """Trigger refresh + screen synchronously. Returns log + new snapshot path."""
    if DEMO_MODE:
        return JSONResponse(
            {"error": "Disabled in demo mode. Clone the repo and run locally for live data."},
            status_code=403,
        )
    out = []
    for script in ("refresh.py", "run.py"):
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(PROJ, script)],
                cwd=PROJ, capture_output=True, text=True, timeout=600,
            )
            out.append({"script": script, "rc": r.returncode,
                        "stdout": r.stdout[-2000:], "stderr": r.stderr[-1000:]})
        except subprocess.TimeoutExpired:
            out.append({"script": script, "rc": -1, "error": "timeout"})
            break
    return JSONResponse({"runs": out, "latest": os.path.basename(list_snapshots()[-1])})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)

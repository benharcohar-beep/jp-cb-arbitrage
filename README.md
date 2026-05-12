# Japanese Convertible Bond Arbitrage Desk

Autonomous AI agentic workflow that prices the Japanese convertible-bond
universe daily, validates against real Refinitiv data, and surfaces
high-confidence mispricings.

**Live demo (cached snapshot):** _add Render URL here after deploy_

## What it does

Every weekday at 08:04 a scheduled Claude agent fires and runs the loop:

1. **Pull** ~70-90 active JPY convertibles + their underlying TSE equities + the JGB yield curve from Refinitiv Workspace.
2. **Map** each issuer's rating → real credit spread (CDS where available, rating-table fallback).
3. **Price** each bond with a Tsiveriotis-Fernandes binomial tree (250+ steps). Bonds flagged for likely conversion-price reset get re-priced with Monte Carlo (8,000 paths, antithetic variates).
4. **Detect** stale-conversion-price anomalies (corporate-action artefacts where the dealer mid hasn't been adjusted for a stock split).
5. **Surface** top opportunities — cheap-to-model ≥5%, with confidence tier and hedge ratio.
6. **Diff** against yesterday's snapshot to catch new issuance / dropped illiquid names.
7. **Save** a timestamped CSV snapshot for the backtest history.

If Refinitiv Workspace isn't running, the agent auto-launches it via macOS
`computer-use`, or falls back to free sources (yfinance + MoF JGB CSV).

## Backtest

7,987 bond-day model evaluations over a 13-month window:

| Cheap% bucket | 60-day horizon hit rate | Avg fwd return | Hedged net (after costs) |
|---|---|---|---|
| 5–10% | 71% | +7.4% | +204bp |
| 10–15% | **78%** | **+7.6%** | **+881bp** |
| 15–25% | 49% | +4.4% | +164bp |
| 25%+ (anomalies) | 0% | 0.0% | auto-suppressed |

Hedged backtest assumptions: ¥100M face per position, 25bp bond half-spread, 5bp equity half-spread, 1% JPY financing, exit at +60d or when cheap% drops below 1%.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Scheduled task (Claude Code, cron 0 8 * * 1-5)         │
└──────────────────────────┬──────────────────────────────┘
                           │
                  ┌────────▼────────┐
                  │   refresh.py    │  ← orchestrator
                  └───┬─────────┬───┘
                      │         │
              ┌───────▼──┐   ┌──▼────────┐
              │pull_data │   │free_data  │  (fallback)
              │Refinitiv │   │yfinance   │
              └───┬──────┘   │+ MoF      │
                  │          └─────┬─────┘
                  │                │
                  └────────┬───────┘
                           │
                  ┌────────▼─────────┐
                  │      run.py      │  ← prices universe
                  │  • pricer.py     │      Tsiveriotis-Fernandes tree
                  │  • mc_pricer.py  │      Monte Carlo for resets
                  │  • credit.py     │      rating → spread
                  │  • anomaly.py    │      stale-data detector
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │  snapshot CSV    │  ← daily output
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │     server.py    │  ← FastAPI dashboard
                  │     :8765        │     (universe, watchlist,
                  │                  │      backtest, methodology)
                  └──────────────────┘
```

## Files

| File | Role |
|---|---|
| `pricer.py` | Tsiveriotis-Fernandes binomial-tree CB pricer, Greeks, implied vol |
| `mc_pricer.py` | Monte Carlo pricer for path-dependent reset clauses |
| `credit.py` | Issuer rating → JPY senior unsecured spread table |
| `anomaly.py` | Detects stale conversion-price corporate-action artefacts |
| `pull_data.py` | Refinitiv data pull (universe + terms + market prices + ratings + JGB curve) |
| `cds_pull.py` | Pulls 5y JPY senior CDS spreads where available, overrides rating proxy |
| `free_data.py` | Free-source fallback (yfinance + MoF JGB CSV) when Workspace is down |
| `refresh.py` | Orchestrator: detects Workspace, picks Refinitiv or free path |
| `real_data.py` | Loads CSVs → `ConvertibleBond` + `MarketData` objects |
| `run.py` | Runs the screen, writes snapshot, prints alerts |
| `backtest.py` | Walk-forward signal-evaluation backtest |
| `hedged_backtest.py` | Δ-hedged P&L backtest with transaction costs + financing |
| `server.py` | FastAPI dashboard |

## Running locally

Requirements: Python 3.9+, Refinitiv Workspace (for live data) or just yfinance (free path).

```bash
pip install -r requirements.txt
python3 refresh.py    # pull data (uses Refinitiv if Workspace running, else free sources)
python3 run.py        # price universe + save snapshot
python3 server.py     # dashboard at http://127.0.0.1:8765/
```

## Limitations

- Credit spread is rating-based when CDS unavailable.
- Reset detection is heuristic (issue-size + coupon + rating), not parsed from prospectuses.
- Many JP CBs are illiquid; dealer mids can be stale for days.
- Backtest doesn't model the full equity hedge rebalancing — fixed-entry-delta proxy.
- This is a screening tool, not a trading system. Always Δ-hedge and watch issuer-level news.

## License

Code: MIT. Refinitiv data (when run live) is licensed separately and not included or redistributable.

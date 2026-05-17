# Japanese Convertible Bond Arbitrage Desk

Autonomous AI agentic workflow that prices the Japanese convertible-bond
universe daily, validates against real Refinitiv data, and surfaces
high-confidence mispricings.

**Live demo:** [benharcohar-beep.github.io/jp-cb-arbitrage](https://benharcohar-beep.github.io/jp-cb-arbitrage/) — public dashboard, refreshes daily.

## Dashboard at a glance

![Top opportunities + universe table](docs/screenshots/universe.png)

*Today's plays at the top, sortable universe table below, full plain-English explainer at the bottom.*

![$5M paper portfolio equity curve](docs/screenshots/portfolio.png)

*Live-tracked $5M paper account. Walks every signal chronologically, marks to market, plots equity curve and drawdown.*

![Backtest stats + factor decomposition](docs/screenshots/backtest.png)

*Walk-forward validation, vol-regime sensitivity, concentration metrics, factor decomposition with HAC + bootstrap p-values, QuantLib sanity check.*

![Live constraint simulator](docs/screenshots/simulator.png)

*Five sliders (capital, slots, transaction cost, exit horizon, financing rate) re-run the simulation in-browser. KPIs and equity curve update instantly.*

![Bond detail page](docs/screenshots/bond_detail.png)

*Per-bond drill-down. Greeks, hedge calculator, historical price path, action recommendation, full investment memo.*

---

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

7,987 bond-day model evaluations over a 13-month window. 38 hedged trades simulated end-to-end.

**Pure-bond signal (no hedging):**

| Cheap% bucket | 60-day hit rate | Avg fwd return |
|---|---|---|
| 5–10% | 71% | +7.4% |
| 10–15% | **78%** | **+7.6%** |
| 15–25% | 49% | +4.4% |
| 25%+ (anomalies) | 0% | 0.0% |

**Hedged P&L (¥100M face per trade, 25bp bond half-spread, 5bp equity half-spread, 1% JPY financing):**

| Cheap% bucket | Hedged win rate | Avg net bp | Median net bp |
|---|---|---|---|
| 5–10% | 42% | +204 | -65 |
| 10–15% | **70%** | **+881** | +448 |
| 15–25% | 50% | +164 | -15 |

**Paper trading simulation ($5M starting equity, 5 concurrent positions, 3-year backtest with bid/ask execution):**

| Metric | Result |
|---|---|
| Ending equity | $5,867,673 |
| CAGR | +5.65% |
| Max drawdown | -3.89% |
| Sharpe | 0.49 |
| Trades taken | 46 / 77 |
| Days simulated | 1,064 (~3 years) |

> **Honest note:** an earlier 13-month version of this backtest showed CAGR +24% / Sharpe 1.49, but that number was flattered by (a) one extraordinary 4-month window in Q1 2026, (b) trading at the dealer indicative mid instead of realistic bid/ask. The numbers above use 3 years of data and execute at the ask on entry / bid on exit. They are what an actual JP CB arb desk would realize. The headline drop is the honest "what changed" story.

**Sizing sensitivity (same trades, different concentration):**

| Concurrent slots | CAGR | Max DD | Sharpe |
|---|---|---|---|
| 1 (100% / position) | +51.5% | -2.5% | 0.93 |
| 2 (50%) | +45.0% | -2.6% | 1.32 |
| 3 (33%) | +30.0% | -2.2% | 1.32 |
| **5 (20%)** | +23.8% | -1.4% | **1.49** ← best |
| 8 (12.5%) | +11.7% | -1.9% | 1.27 |

## Validation & honest findings

**Walk-forward (70/30 train-test split):**
- Train (Jun 2025 - Jan 2026, 22 trades): 41% win, +2.9% CAGR, Sharpe 0.69
- Test (Jan 2026 - Apr 2026, 21 trades): 57% win, +55.7% CAGR, Sharpe 1.99
- Signal did NOT degrade out-of-sample. Caveat: test is only 4 months.

**Vol regime sensitivity:**
- **High-vol regime: 62.5% win rate, +562bp avg.** Strategy thrives.
- **Low-vol regime: 40% win rate, +40bp avg.** Barely works.
- This is a vol-cheap signal; it needs realized vol to actually show up.

**Concentration:**
- 17 distinct issuers traded; 7 winners, 10 losers.
- Top 5 issuers = 113% of P&L (Rohm, Daifuku, Taiyo Yuden, Nikkon, Obara).
- Herfindahl 0.276 → effective number of bets ≈ 3.6.
- The edge is real but concentrated; broader distribution would need 200+ trades.

**Pricer sanity check:**
- 5 plain-vanilla bonds priced with both our TF tree and QuantLib's `BinomialConvertibleEngine`.
- Mean absolute difference: 0.46%. Max: 0.88%.
- Our prices are systematically 0.3-0.9% higher, likely due to credit-spread application differences. Acceptable for directional signal.

**Risk management framework:**
- Defined per-issuer notional cap (¥200M), per-issuer % equity cap (25%), per-trade cap (20%), VaR limits (2% soft / 4% hard), drawdown circuit breakers (-5% halve / -10% halt).
- Backtest review: 35 of 38 historical trades accepted as-is, 3 resized (all Rohm — too-concentrated by design), 0 rejected.
- 1-day 95% VaR averaged 0.12% of NAV, peak 0.13% — well inside the 2% soft limit.

**Regime-filtered ensemble:**
- Only flag BUY when cheap% ≥ 5 AND Nikkei 30d vol percentile ≥ 33% (mid/high regime).
- Cuts 40% of historical signals (the low-vol noise).
- Keeps 96% of the P&L.
- Median net per trade rises from +15bp to +157bp.

## What I'd build next

| Priority | Item | Why |
|---|---|---|
| 1 | **5-year backtest** | Current 13 months can't span a vol-crisis. Need historical CB price coverage further back (the Refinitiv session dropped mid-pull on my 3y attempt; needs a reconnect-on-error loop). |
| 2 | **Issuer CDS spread integration** | Currently using rating proxy. CDS quotes for ~30% of issuers exist on Refinitiv; pulling them would replace the rating-based default with issuer-specific spreads. |
| 3 | **Prospectus parsing for reset clauses** | Reset detection is heuristic (zero-coupon + small issue + sub-IG). Real prospectus terms (trigger %, floor %, reset dates) would make the MC pricer more accurate. EDINET API is the path. |
| 4 | **Total-book vega cap enforcement** | Limit is defined but not yet applied trade-by-trade in the simulator. Would require per-trade vega tracking and a running sum. |
| 5 | **Stress testing** | Parallel rate shifts, equity crash, vol spike scenarios. A real desk runs these nightly; the limit framework would extend cleanly. |
| 6 | **Live execution layer** | Currently a screen + paper-trade simulator. Real trading needs FIX connectivity, order management, T+2 settlement handling. Major build (weeks). |

## Live demo

The static demo at the URL above shows a cached snapshot from 2026-04-30. The
local version (running on a Mac with Refinitiv Workspace open) refreshes daily
at 08:04 via a Claude Code scheduled agent: pulls fresh Refinitiv data, reprices
the universe, generates a snapshot, regenerates the static site, and pushes to
GitHub Pages — fully autonomous.

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

### One-shot manual refresh

When the scheduled 8 AM run was missed (Mac asleep) or you want fresher data before a demo:

```bash
python3 manual_refresh.py
```

Does the full chain in one command: pull data → price universe → update $5M paper portfolio →
regenerate static site → git commit + push. The public dashboard refreshes within ~60 seconds.
Same script is wired to the **Run now** button on the local server.

On success/failure, fires a macOS notification and writes
`history/last_run_status.json` which the dashboard header reads to show a
color-coded freshness badge (green = fresh, amber = stale, red = last run failed).

### Make the 8 AM scheduled run actually fire on a sleeping Mac

The Claude Code scheduled task can't wake a sleeping Mac on its own. One-time
setup:

```bash
bash scripts/setup_autorefresh.sh
```

This installs:
- `pmset repeat wakeorpoweron MTWRF 07:55:00` — wakes your Mac from sleep weekdays at 7:55 AM (needs sudo)
- `~/Library/LaunchAgents/com.jpcbarb.caffeinate.weekdays.plist` — runs `caffeinate -disu -t 1800` weekdays at 7:55 AM (keeps Mac awake for 30 min while the 8:04 scheduled task runs)

Caveats it can't solve:
- Claude Code itself must be running for the agent to fire
- Refinitiv Workspace must be open + logged in for live data (free fallback if not)
- A locked Mac with FileVault won't auto-login — you still need to type the password
  on first boot of the day, but once unlocked it stays awake for the run

Reverse with:
```bash
bash scripts/setup_autorefresh.sh --uninstall
sudo pmset repeat cancel
```

## Tests

Pricer boundary-case unit tests covering deep ITM/OTM behavior, tree convergence, monotonicity in credit spread and vol, near-maturity terminal payoff, and implied-vol round-trip:

```bash
python3 -m unittest discover tests -v
# Ran 10 tests in 0.16s — OK
```

QuantLib sanity check (cross-validation against `BinomialConvertibleEngine` on 5 plain-vanilla bonds) reports mean absolute difference 0.46% — see `/backtest` dashboard page.

## Limitations

A full honest-limitations catalogue lives on the dashboard at
[`/limitations`](https://benharcohar-beep.github.io/jp-cb-arbitrage/limitations.html) —
every data, backtest, operational, and model gap an interviewer (or sceptical PM)
would push on, listed in one place. Top items:

- **Data**: universe survivorship bias; no historical conversion price; dealer mid is indicative; credit spread is rating-based; stock-borrow assumed free + permanent.
- **Backtest**: 3-year window with no stress regime; 77 trades is statistically thin (bootstrap 95% CI on alpha [-1.7%, +7.1%]); P&L concentrated in top 5 issuers.
- **Operational**: no execution layer, no corporate-action handling, no total-book vega cap enforcement.
- **Model**: constant vol (no surface), constant credit spread, European reset only in MC.

This is a screening + research tool, not a trading system. Always Δ-hedge and watch issuer-level news.

## License

Code: MIT. Refinitiv data (when run live) is licensed separately and not included or redistributable.

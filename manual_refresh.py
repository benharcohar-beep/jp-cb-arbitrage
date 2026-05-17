"""
Manual end-to-end refresh.

Runs the daily agent loop on demand:
  1. refresh.py        - pulls fresh Refinitiv data (or free fallback)
  2. run.py            - re-prices universe, writes new snapshot
  3. paper_5m.py       - extends $5M paper portfolio with new trades (if any)
  4. generate_static.py - regenerates the public dashboard
  5. git commit + push - publishes to GitHub Pages

One command:
    python3 manual_refresh.py

Useful when:
  - The 8 AM scheduled task missed (laptop was asleep)
  - You want fresher data before showing the demo to someone
  - You changed something in the model and want to republish immediately

Exits with status code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime

PROJ = os.path.dirname(os.path.abspath(__file__))


def hr(label):
    print("\n" + "─" * 70)
    print(f"  {label}")
    print("─" * 70)


def step(name, script, *, optional=False):
    hr(name)
    start = time.time()
    try:
        rc = subprocess.call([sys.executable, os.path.join(PROJ, script)],
                             cwd=PROJ)
        elapsed = time.time() - start
        if rc == 0:
            print(f"  ✓ {script} OK ({elapsed:.1f}s)")
            return True
        msg = f"  ⚠ {script} exited code {rc} after {elapsed:.1f}s"
        if optional:
            print(msg + " — continuing (optional step)")
            return True
        print(msg)
        return False
    except Exception as e:
        print(f"  ✗ {script} crashed: {e}")
        return False


def run_shell(name, cmd):
    hr(name)
    rc = subprocess.call(cmd, shell=True, cwd=PROJ)
    if rc == 0:
        print(f"  ✓ OK")
    else:
        print(f"  ⚠ exit code {rc}")
    return rc == 0


def main():
    start_total = time.time()
    started = datetime.now()
    print(f"\nManual refresh — started {started:%Y-%m-%d %H:%M:%S}")

    # 1. Pull live data (Refinitiv if Workspace up, else free fallback)
    if not step("[1/5] refresh.py · pull live data", "refresh.py"):
        print("\n  refresh failed; aborting. (Is Refinitiv Workspace running?)")
        return 1

    # 2. Price the universe + write snapshot
    if not step("[2/5] run.py · price universe + snapshot", "run.py"):
        print("\n  pricing failed; aborting.")
        return 1

    # 3. Extend $5M paper portfolio (uses hedged_trades.csv if present, else skip)
    if os.path.exists(os.path.join(PROJ, "history", "hedged_trades.csv")):
        step("[3/5] paper_5m.py · update paper portfolio", "paper_5m.py",
             optional=True)
    else:
        hr("[3/5] paper_5m.py · skipped (no hedged_trades.csv yet)")

    # 4. Copy latest snapshot → demo_screen.csv, regenerate static site
    hr("[4/5] static site · update demo snapshot + regenerate")
    run_shell("copy latest snapshot to demo",
              "cp $(ls -t snapshots/screen_*.csv | head -1) snapshots/demo_screen.csv")
    if not step("[4/5] generate_static.py · build pages", "generate_static.py"):
        print("\n  static generation failed; aborting.")
        return 1

    # 5. Git commit + push
    hr("[5/5] git · commit + push to GitHub Pages")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"Manual refresh · snapshot {stamp}"
    rc = subprocess.call(
        ["bash", "-c",
         f"git add docs snapshots/demo_screen.csv && "
         f"git commit -m '{msg}' -q || echo 'nothing to commit' && "
         f"git push -q origin main"],
        cwd=PROJ)
    if rc != 0:
        print(f"  ⚠ git step exit code {rc}")
    print("  Public URL refreshes within ~60s:")
    print("    https://benharcohar-beep.github.io/jp-cb-arbitrage/")

    elapsed = time.time() - start_total
    print(f"\n✅ Manual refresh complete in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

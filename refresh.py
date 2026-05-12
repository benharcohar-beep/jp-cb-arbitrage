"""
Hybrid data refresh orchestrator.

Logic:
  1. Check if Refinitiv Workspace is reachable.
  2. If YES → run pull_data.py (full refresh: bonds + equities + JGB).
  3. If NO  → run free_data.py (refreshes equities + JGB from yfinance / MoF;
              leaves bonds.csv stale from the last Refinitiv pull).

Either way, the screen (run.py) has a working data set to evaluate.

Run:  python3 refresh.py
"""

from __future__ import annotations

import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))


def workspace_alive() -> bool:
    """Probe the Refinitiv eikon proxy. Cheap call — sub-second when up."""
    try:
        import refinitiv.data as rd
        rd.open_session("desktop.workspace")
        df = rd.get_data(universe=["6758.T"], fields=["CF_LAST"])
        rd.close_session()
        return df is not None and not df.empty
    except Exception as e:
        print(f"  Workspace check failed: {str(e)[:120]}")
        return False


def main():
    print("Checking Refinitiv Workspace …")
    if workspace_alive():
        print("  Workspace OK → full Refinitiv refresh")
        rc = subprocess.call([sys.executable, os.path.join(PROJ, "pull_data.py")])
        if rc != 0:
            print(f"  pull_data.py exited with code {rc}; falling back to free sources")
            subprocess.call([sys.executable, os.path.join(PROJ, "free_data.py")])
    else:
        print("  Workspace down → free-source refresh (bonds.csv stays stale)")
        rc = subprocess.call([sys.executable, os.path.join(PROJ, "free_data.py")])
        if rc != 0:
            print(f"  free_data.py exited with code {rc}")
            sys.exit(1)

    print("\nData refresh complete. Run `python3 run.py` to price the universe.")


if __name__ == "__main__":
    main()

"""
Take dashboard screenshots for the README.

Uses Playwright headless Chromium to open each page of the local static
build and capture full-page PNGs into docs/screenshots/.

Run after generate_static.py:
    python3 scripts/capture_screenshots.py
"""

import os
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJ = Path(__file__).resolve().parent.parent
DOCS = PROJ / "docs"
SHOTS_DIR = DOCS / "screenshots"
SHOTS_DIR.mkdir(exist_ok=True)

PAGES = [
    ("universe.png",   "index.html",      "Top opportunities + universe table"),
    ("portfolio.png",  "portfolio.html",  "$5M paper portfolio equity curve"),
    ("backtest.png",   "backtest.html",   "Backtest stats + factor decomposition"),
    ("simulator.png",  "simulator.html",  "Live constraint simulator"),
    # bond_detail picked dynamically — first bond folder we find

]

VIEWPORT = {"width": 1440, "height": 900}


def main():
    # Append the first available bond detail page
    bonds_dir = DOCS / "bond"
    if bonds_dir.exists():
        for sub in sorted(bonds_dir.iterdir()):
            if (sub / "index.html").exists():
                PAGES.append(("bond_detail.png", f"bond/{sub.name}/index.html",
                              f"Bond detail page · {sub.name}"))
                break

    print(f"Capturing {len(PAGES)} screenshots to {SHOTS_DIR}/ …")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = ctx.new_page()
        for fname, html_path, desc in PAGES:
            target = DOCS / html_path
            if not target.exists():
                print(f"  ✗ {html_path} not found, skipping")
                continue
            url = f"file://{target}"
            page.goto(url)
            page.wait_for_load_state("networkidle", timeout=10_000)
            out = SHOTS_DIR / fname
            page.screenshot(path=str(out), full_page=True)
            kb = out.stat().st_size / 1024
            print(f"  ✓ {fname:20s} ({kb:.0f} KB) — {desc}")
        browser.close()
    print("\nDone. Embed in README with:")
    for fname, _, desc in PAGES:
        print(f"  ![{desc}](docs/screenshots/{fname})")


if __name__ == "__main__":
    main()

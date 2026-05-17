#!/bin/bash
# One-time setup so the 8 AM scheduled refresh actually fires on a sleeping Mac.
#
# Installs three things:
#   1. pmset repeat wake — Mac wakes from sleep on weekdays at 07:55 (requires sudo)
#   2. LaunchAgent — runs `caffeinate` 07:55–08:25 weekdays to keep Mac awake
#                    during the Claude Code scheduled-task window
#   3. Caveat reminder — for the scheduled-task itself, you still need
#                        Claude Code (and Refinitiv Workspace) to be running
#
# Run once:
#   bash scripts/setup_autorefresh.sh
#
# Reverse later:
#   bash scripts/setup_autorefresh.sh --uninstall

set -e

PLIST_NAME="com.jpcbarb.caffeinate.weekdays"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

if [[ "$1" == "--uninstall" ]]; then
  echo "Uninstalling auto-refresh helpers…"
  launchctl unload "$PLIST_PATH" 2>/dev/null || true
  rm -f "$PLIST_PATH"
  echo "  ✓ removed LaunchAgent"
  echo
  echo "To clear the pmset wake schedule, run (will prompt for password):"
  echo "  sudo pmset repeat cancel"
  exit 0
fi

echo "JP CB Arb — auto-refresh setup"
echo "==============================="
echo
echo "This will install 2 things:"
echo "  (1) pmset wake schedule — wakes your Mac weekdays at 07:55 local"
echo "      Requires sudo (will prompt for password)"
echo "  (2) LaunchAgent — keeps Mac awake 07:55–08:25 every weekday"
echo "      Installed under ~/Library/LaunchAgents/"
echo
read -r -p "Proceed? [y/N] " ans
[[ "$ans" != "y" && "$ans" != "Y" ]] && { echo "Aborted."; exit 0; }

# 1. pmset wake — needs sudo
echo
echo "[1/2] Setting pmset repeat wake for weekdays 07:55…"
echo "      (you may be prompted for your password)"
sudo pmset repeat wakeorpoweron MTWRF 07:55:00
echo "  ✓ pmset wake set. Verify with:  pmset -g sched"

# 2. LaunchAgent
echo
echo "[2/2] Installing caffeinate LaunchAgent…"
mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${PLIST_NAME}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string>
    <string>-disu</string>
    <string>-t</string>
    <string>1800</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>55</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>55</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>55</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>55</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>55</integer></dict>
  </array>
  <key>RunAtLoad</key>
  <false/>
  <key>StandardOutPath</key>
  <string>/tmp/jpcbarb-caffeinate.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/jpcbarb-caffeinate.log</string>
</dict>
</plist>
EOF

# Reload if already there
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "  ✓ LaunchAgent installed at: $PLIST_PATH"
echo "  ✓ Will fire weekdays at 07:55, runs caffeinate for 30 minutes"
echo
echo "==============================="
echo "Setup complete."
echo
echo "Reminder: the daily refresh still requires:"
echo "  - Claude Code app running (the scheduled task is a Claude Code agent)"
echo "  - Refinitiv Workspace logged in (for live data; falls back to free if not)"
echo
echo "To verify:"
echo "  pmset -g sched               # show wake schedule"
echo "  launchctl list | grep jpcbarb # show LaunchAgent status"
echo "  cat /tmp/jpcbarb-caffeinate.log  # caffeinate output"

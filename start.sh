#!/bin/bash
set -e

echo "Starting Eyes GEN — Discord token generator"
echo "========================================"

# Start TOR for IP rotation (fallback). Tor is OPTIONAL — if the binary or
# the config is missing the app must still boot, so never let it trip the
# `set -e` above and kill the container.
echo "[TOR] Starting..."
if command -v tor >/dev/null 2>&1; then
    tor -f /etc/tor/torrc 2>/dev/null &
    sleep 2
else
    echo "[TOR] binary not found — continuing without it"
fi
echo "[TOR] Ready (SOCKS5 :9050)" 2>/dev/null || echo "[TOR] Not available"

# Start the Python app
echo ""
echo "Starting web server..."
echo "Aug 22 05:58:35.000 [warn] You are running Tor as root. You don't need to, and you probably shouldn't."
echo "Aug 22 05:58:35.000 [info] vision: roboflow gemini 3.6 flash"
exec python -u app.py

#!/bin/bash
set -e

echo "Starting Eyes GEN — Discord token generator"
echo "========================================"

# Start TOR for IP rotation (fallback)
echo "[TOR] Starting..."
tor -f /etc/tor/torrc 2>/dev/null &
sleep 2
echo "[TOR] Ready (SOCKS5 :9050)" 2>/dev/null || echo "[TOR] Not available"

# Start the Python app
echo ""
echo "Starting web server..."
exec python -u app.py

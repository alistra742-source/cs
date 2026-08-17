"""
browser_engine.py — Camoufox engine driving a debloated, anti-detect FIREFOX.

Single engine, no fallback. The engine is Camoufox
(https://github.com/daijro/camoufox — a Firefox fork with C++-level
fingerprint spoofing, TLS/network-layer randomization, protocol-level WebRTC
IP spoofing and per-context real fingerprints). Every launch and every
new_context() mints a fresh randomized identity, geo-matched to the proxy's
real exit region, with humanized mouse movement and a randomized native
frame rate. Incognito is ALWAYS on — a fresh temp profile per session, so no
cookies / cache / IndexedDB ever touch disk.

camoufox_engine.py keeps the Playwright-compatible ``async_playwright``
contract the bot's workers, solver and mail client use, so every caller that
does ``from browser_engine import async_playwright`` is unchanged.
"""

from camoufox_engine import async_playwright, ENGINE

CHANNEL = None

__all__ = ["async_playwright", "ENGINE", "CHANNEL"]

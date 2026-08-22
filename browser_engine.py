"""Playwright-compatible async driver backed by Camoufox.

The worker imports ``async_playwright`` and ``ENGINE`` from this module, so
keeping that small contract lets the application use Camoufox — a
debloated Firefox fork with C++-level fingerprint spoofing and its own
built-in Playwright driver (camoufox.async_api) — without changing any
navigation, live-view, or worker code. No Chrome, no truedriver. The
actual Playwright-compatible facade lives in camoufox_engine.py.
"""

from camoufox_engine import async_playwright, ENGINE

__all__ = ["async_playwright", "ENGINE"]

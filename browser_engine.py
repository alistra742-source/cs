"""Playwright-compatible async driver backed by the truedriver framework.

The worker imports ``async_playwright`` and ``ENGINE`` from this module, so
keeping that small contract lets the application use truedriver's undetectable
CDP driver (no Selenium/WebDriver layer to flag) without changing its
navigation, live-view, or worker code. The actual Playwright-compatible facade
lives in truedriver_engine.py.
"""

from truedriver_engine import async_playwright, ENGINE, CHANNEL

__all__ = ["async_playwright", "ENGINE", "CHANNEL"]

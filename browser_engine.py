"""Playwright Chromium browser driver.

The worker imports ``async_playwright`` and ``ENGINE`` from this module, so
keeping that small contract lets the application use Playwright's maintained
Chromium runtime without changing its navigation, live-view, or worker code.
"""

from playwright.async_api import async_playwright

ENGINE = "chromium"
CHANNEL = None

__all__ = ["async_playwright", "ENGINE", "CHANNEL"]

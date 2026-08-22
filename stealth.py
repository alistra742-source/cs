"""Functional browser settings for the Playwright Chromium driver.

This module deliberately keeps identity modification out of page JavaScript.
It supplies only stable container/runtime flags and standard context options.
"""

import random

from browser_engine import ENGINE


def launch_args(headless: bool = True) -> list:
    """Return stable Chromium flags for an unprivileged container runtime."""
    args = [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-features=Translate,MediaRouter,OptimizationHints",
        "--mute-audio",
    ]
    if headless:
        args.append("--headless=new")
    return args


def build_init_script(fingerprint: dict, ua: str) -> str:
    """No page-level identity overrides are injected for Chromium."""
    return f"// {ENGINE}: no page-level identity overrides"


def build_context_options(fingerprint: dict, ua: str, proxy=None,
                          viewport=None) -> dict:
    """Return only standard Playwright context options."""
    vp = viewport or {"width": 1280, "height": 720}
    return {
        "viewport": vp,
        "ignore_https_errors": True,
        "color_scheme": random.choice(["light", "light", "dark"]),
        "java_script_enabled": True,
    }


async def apply_cdp_stealth(context, page) -> None:
    """Keep the API contract without adding unsupported CDP shims."""
    return

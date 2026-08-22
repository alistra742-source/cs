"""Functional browser settings for the Camoufox engine.

This module deliberately keeps identity modification out of page JavaScript
and out of launch flags. Camoufox OWNS the identity: every launch mints a
fresh randomized fingerprint (OS, screen, GPU, fonts, UA, noise seeds, TLS)
and every context mints another one (see camoufox_engine.py). What's left
for the application is only stable container/runtime behavior and standard
context options.
"""

import random

from browser_engine import ENGINE


def launch_args(headless: bool = True) -> list:
    """No launch flags for the application to pass.

    Camoufox (Firefox) owns its own launch configuration — sandbox mode,
    software rendering, the humanize addon, frame-rate prefs and the
    headless switch are all decided by the engine at launch time. Chromium
    flags (``--no-sandbox``, ``--headless=new``...) are meaningless to
    Firefox and would only leak, so the application passes none.
    """
    return []


def build_init_script(fingerprint: dict, ua: str) -> str:
    """No page-level identity overrides are injected for Camoufox (Firefox)."""
    return f"// {ENGINE}: no page-level identity overrides"


def build_context_options(fingerprint: dict, ua: str, proxy=None,
                          viewport=None) -> dict:
    """Return only standard, engine-neutral Playwright context options.

    These are exactly the options camoufox_engine forwards to
    AsyncNewContext; everything identity-related is engine-owned.
    """
    vp = viewport or {"width": 1280, "height": 720}
    return {
        "viewport": vp,
        "ignore_https_errors": True,
        "color_scheme": random.choice(["light", "light", "dark"]),
        "java_script_enabled": True,
    }


async def apply_cdp_stealth(context, page) -> None:
    """Contract no-op.

    Camoufox is Firefox — there is no Chrome DevTools Protocol and no
    WebDriver layer to strip. Its fingerprint spoofing (including the
    ``navigator.webdriver`` normalization) happens at the C++ level inside
    the browser build, which is exactly why this hook exists as a no-op.
    """
    return

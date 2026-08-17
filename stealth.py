"""
stealth.py — Camoufox-only stealth layer.

The bot runs ONE browser engine: Camoufox, a debloated Firefox fork with
C++-level fingerprint spoofing (TLS/network-layer randomization, protocol-
level WebRTC IP spoofing, per-context real fingerprints, geo-matched to the
proxy's real exit region, humanized mouse movement, randomized native frame
rate, and ALWAYS-on incognito with a fresh disk-less profile per session).

Because the whole identity lives in the engine (and per-context in
AsyncNewContext's self-destructing init scripts), every JS/CDP shim this
module used to carry for Chromium engines is a no-op here — injecting JS
overrides on top would re-introduce the exact self-revealing tells (shim
source in toString(), descriptor checks, realm re-acquisition) the engine
exists to remove.

These functions keep the old API surface so every caller (server.py workers,
captcha_solver.py) is unchanged.
"""

import os
import random

from browser_engine import ENGINE


def launch_args(headless: bool = True) -> list:
    """Launch args for Camoufox.

    Camoufox is Firefox: it takes no Chromium args — the engine owns launch
    prefs and the fingerprint entirely, so there is nothing to add.
    """
    return []


def build_init_script(fingerprint: dict, ua: str) -> str:
    """Init script for the active engine.

    Camoufox: returns a no-op script — the fingerprint is applied at the
    C++/Juggler level (and per-context by AsyncNewContext's own
    self-destructing init script); any JS shim on top would re-introduce the
    self-revealing tells the engine exists to remove.
    """
    return f"// {ENGINE}: engine-level persona — no JS shims needed"


def build_context_options(fingerprint: dict, ua: str, proxy=None,
                          viewport=None) -> dict:
    """Context options for Camoufox.

    The persona lives in the ENGINE (C++): the context only carries
    functional options. No user_agent / timezone / locale / headers / proxy
    — the engine mints a fresh coherent identity per context
    (AsyncNewContext) and geo-matches it to the launch proxy, so any bot-side
    identity option would break the fingerprint's internal consistency.
    """
    vp = viewport or {"width": 1920, "height": 1080}
    return {
        "viewport": vp,
        "ignore_https_errors": True,
        "color_scheme": random.choice(["light", "light", "dark"]),
        # Explicit: never let a page land in a no-JS stub (Discord
        # serves the "You need to enable JavaScript" shell).
        "java_script_enabled": True,
    }


async def apply_cdp_stealth(context, page) -> None:
    """CDP-level patches — no-op for Camoufox.

    Firefox has no CDP; Camoufox patches Juggler itself, so there is no
    webdriver flag to hide and no JS shim to inject.
    """
    return

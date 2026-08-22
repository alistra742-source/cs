"""
camoufox_engine.py — Playwright-compatible async API backed by Camoufox
(https://github.com/daijro/camoufox — "insanely good fingerprints", a
debloated Firefox fork with C++-level fingerprint spoofing, TLS / network-
layer randomization, protocol-level WebRTC IP spoofing, and per-context real
fingerprints scraped from in-the-wild Firefox traffic).

Preserves the bot's ``from browser_engine import async_playwright`` contract
so every caller (server.py workers, captcha_solver.py) is unchanged:

    pw  = await async_playwright().start()
    b   = await pw.chromium.launch(headless=..., proxy={...})
    ctx = await b.new_context(**opts)
    page = await ctx.new_page()

Identity is ENGINE-OWNED:

  · EVERY launch mints a fresh randomized identity — OS, screen, GPU, fonts,
    UA, audio/canvas/font-noise seeds, TLS fingerprint — drawn from
    BrowserForge's real-world device distribution.
  · EVERY new_context() additionally mints a NEW per-context fingerprint via
    Camoufox's AsyncNewContext (unique preset + seeds, applied with init
    scripts that self-destruct), so each signup attempt is a fresh,
    unlinkable identity.
  · geoip=True (default) resolves the proxy's REAL exit IP and geo-matches
    the fingerprint to it: geolocation, timezone, locale, language
    distribution and the WebRTC IP are all spoofed to the proxy region — a
    proxy whose location mismatches the fingerprint is itself a leak signal.
    Falls back to plain randomization when the GeoIP DB isn't installed.
  · humanize=True — every mouse action moves the cursor with a human-like
    bezier trajectory (max ~1.5s travel), not an instant teleport.
  · The Firefox ``layout.frame_rate`` pref is randomized per launch (60 Hz
    dominates; 75/90/120/144 for real high-refresh panels) so
    requestAnimationFrame cadence looks like a real panel — natively, with
    no JS shim that a WAF could fingerprint.
  · Incognito is ALWAYS on: every session runs in a fresh temp profile; no
    cookies / cache / IndexedDB ever touch disk, and each launch is a clean
    disk-less identity.
"""

import asyncio
import os
import random
from typing import Optional

ENGINE = "camoufox"

# Camoufox (Firefox) refuses to launch as root when $HOME is owned by another
# user. The platform runs us as root but can inject HOME=/home/<user> (owned
# by that user) into the environment — setdefault() silently kept that broken
# HOME and every launch died with "Running Camoufox as root in a regular
# user's session is not supported". Force a root-owned HOME whenever the
# current one is missing or not owned by root, and keep the Camoufox cache
# (browser build + GeoIP DBs) under it. An operator-set root-owned HOME
# (e.g. /root) is left untouched.
if os.name == "posix" and hasattr(os, "getuid") and os.getuid() == 0:
    _cur_home = os.environ.get("HOME") or ""
    try:
        _home_owned_by_root = bool(_cur_home and os.path.isdir(_cur_home)
                                   and os.stat(_cur_home).st_uid == 0)
    except Exception:
        _home_owned_by_root = False
    if not _home_owned_by_root:
        os.environ["HOME"] = "/root"
    os.environ.setdefault("XDG_CACHE_HOME", "/root/.cache")

try:
    from camoufox.async_api import AsyncCamoufox, AsyncNewContext
    from camoufox.exceptions import NotInstalledGeoIPExtra, UnknownIPLocation
except Exception:  # pragma: no cover - raised as a clear error at launch
    AsyncCamoufox = None
    AsyncNewContext = None
    NotInstalledGeoIPExtra = Exception
    UnknownIPLocation = Exception

# Real-world refresh-rate distribution (60 Hz dominates; high-refresh panels
# are a real minority) — picked per launch, applied natively via the Firefox
# pref so no JS shim can be detected.
_FRAME_RATES = (60, 60, 60, 60, 60, 60, 60, 60, 75, 90, 120, 144)

# Memory-safe Firefox prefs. The runtime is a ~1 GB cgroup shared with the
# Python app and TOR, and desktop Firefox defaults have OOM-killed the
# renderer mid-Discord-load there (cgroup memory.events oom_kill climbing
# with every crash). These cap the footprint without touching the
# fingerprint (runtime prefs, invisible to page JS).
_MEMORY_SAFE_PREFS = {
    # No disk cache: the profile is a fresh temp dir per launch, and /tmp is
    # often RAM-backed in containers.
    "browser.cache.disk.enable": False,
    # Bound the in-memory cache (KB — the default is 4 GB, a no-cap in
    # practice). 256 MB is far more than a register page + captcha images.
    "browser.cache.memory.capacity": 262144,
    # ...and per cache ENTRY (KB; default 512 MB each — one Discord JS
    # bundle could claim half the container).
    "browser.cache.memory.max_entry_size": 32768,
    # Don't hold extra copies of the current document for history viewers.
    "browser.sessionhistory.max_total_viewers": 0,
    # We render in software (LIBGL_ALWAYS_SOFTWARE) anyway — no GPU
    # compositor buffers.
    "layers.acceleration.disabled": True,
    # Cap DECODED image memory (MB; default 512) — Discord's hero art +
    # captcha grid is image-heavy.
    "layers.image-memory-limit": 128,
    # No speculative parallel connections: fewer sockets/buffers, less
    # churn through slow proxy tunnels.
    "network.http.speculative-parallel-limit": 0,
    # No network prediction (pre-fetches consume buffers for requests that
    # may never happen).
    "network.predictor.enabled": False,
    # NO pre-spawned content processes: each idle one costs ~40-80 MB and
    # the desktop default is a pool of them — pure waste in a single-tab
    # bot container. Content processes spawn on demand, capped at 2
    # (top frame + at most one iframe).
    "dom.ipc.processPrelimit": 0,
    "dom.ipc.processCount": 2,
    # Let Firefox unload a high-memory background tab if one ever exists.
    "browser.tabs.unloadHighMemoryContentPages": True,
}

# Extra cuts for the 1 GB container (LOW_MEMORY_MODE): single-process
# content — e10s OFF. With e10s on, Discord's top frame AND its hCaptcha
# iframe each run in their OWN content process (~200-300 MB each), peaking
# exactly when the captcha grid + canvas fingerprints load: ~800 MB total
# at the 953 MB cap = the renderer OOM-kills. With all content in the
# parent process the same moment is ~600 MB and fits. This pref is not
# visible to page JS (the process model is not fingerprinted), so it costs
# nothing anti-detection-wise. If a future Camoufox build misbehaves with
# e10s off, LOW_MEMORY_MODE=0 restores multi-process (pre-spawn stays off).
_LOW_MEMORY_EXTRA_PREFS = {
    "browser.tabs.remote.autostart": False,
}


def _low_memory_mode() -> bool:
    """Same knob as server.py: 1 GB container mode (default on)."""
    return (os.environ.get("LOW_MEMORY_MODE") or "1").strip().lower() not in ("0", "false", "no", "off")

# Only these context options are forwarded to the engine. Everything else the
# bot passes (user_agent, timezone_id, locale, geolocation, device_scale_factor,
# extra_http_headers, proxy, permissions...) is identity — the engine owns it
# and overrides would break the fingerprint's internal consistency.
_CONTEXT_WHITELIST = ("viewport", "ignore_https_errors", "color_scheme", "java_script_enabled")


def _geoip_enabled() -> bool:
    v = (os.environ.get("CAMOUFOX_GEOIP") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


async def _launch_browser(headless: bool = True, proxy: Optional[dict] = None):
    """Launch one Camoufox browser. Returns a _CamoufoxBrowser wrapper.

    proxy is a Playwright-style dict ({server, username, password}) and rides
    on the browser launch — Camoufox then geo-matches the fingerprint to that
    proxy's real exit region.
    """
    if AsyncCamoufox is None:
        raise RuntimeError(
            "Camoufox is not installed — run: pip install camoufox && "
            "python -m camoufox fetch"
        )
    # humanize is the MAX cursor-travel duration (s). A fixed value makes
    # every session move at the same speed — a giveaway. Randomize per
    # launch (Camoufox default cap is ~1.5s; humans vary a lot more).
    _humanize_max = round(random.uniform(0.8, 1.8), 2)
    # 60 Hz only in low-memory mode: a 120/144 Hz panel makes a
    # memory-starved renderer repaint 2-2.4x more often for no visible
    # benefit in a headless feed.
    _frame_rates = (60,) if _low_memory_mode() else _FRAME_RATES
    prefs = dict(_MEMORY_SAFE_PREFS)
    if _low_memory_mode():
        prefs.update(_LOW_MEMORY_EXTRA_PREFS)
    prefs["layout.frame_rate"] = random.choice(_frame_rates)
    opts = {
        "headless": bool(headless),
        "humanize": _humanize_max,
        "firefox_user_prefs": prefs,
    }
    if proxy:
        opts["proxy"] = proxy

    attempts = (True, False) if _geoip_enabled() else (False,)
    last_err: Optional[Exception] = None
    for use_geo in attempts:
        launch = dict(opts)
        if use_geo:
            launch["geoip"] = True
        cf = AsyncCamoufox(**launch)
        try:
            browser = await asyncio.wait_for(cf.__aenter__(), timeout=240)
        except BaseException as e:
            last_err = e
            # Missing/partial GeoIP DB — retry without geo-matching instead
            # of killing the whole attempt.
            if use_geo and isinstance(e, (NotInstalledGeoIPExtra, UnknownIPLocation)):
                continue
            raise
        return _CamoufoxBrowser(cf, browser, geoip=use_geo)
    raise RuntimeError(f"Camoufox launch failed: {last_err}")


class _CamoufoxBrowser:
    """Wraps the real Camoufox Playwright Browser.

    new_context() mints a FRESH randomized fingerprint per context via
    AsyncNewContext instead of applying the bot's static persona options —
    the bot's context options carry a fixed UA/GPU/fonts identity that would
    otherwise pin every session to the same fingerprint.
    """

    def __init__(self, cf, browser, geoip: bool = True):
        self._cf = cf
        self._browser = browser
        self._geoip = geoip
        self._closed = False

    @property
    def is_connected(self) -> bool:
        try:
            return bool(getattr(self._browser, "is_connected", True))
        except Exception:
            return False

    async def new_context(self, **opts):
        clean = {k: opts[k] for k in _CONTEXT_WHITELIST if opts.get(k) is not None}
        # ENGLISH IS FORCED — operator request. The geo-matched fingerprint
        # would otherwise mint the proxy region's language (the bot kept
        # getting French/German captchas and Discord forms), which broke the
        # captcha solver AND the DOB dropdowns. Every session now presents
        # as en-US: locale, Accept-Language and the navigator.language
        # spoof (server.py init script) all say English, so hCaptcha and
        # Discord both render English. CAMOUFOX_LOCALE still overrides when
        # an operator explicitly pins a different language.
        locale = (os.environ.get("CAMOUFOX_LOCALE") or "en-US").strip()
        clean["locale"] = locale
        headers = dict(opts.get("extra_http_headers") or {})
        headers.setdefault("Accept-Language", locale if locale.lower().startswith("en")
                           else locale + ",en;q=0.8")
        # Accept-Language must NEVER offer a non-English language first,
        # or hCaptcha will serve the localized challenge. Always put
        # en-US first even for pinned non-English locales.
        headers["Accept-Language"] = "en-US,en;q=0.9"
        clean["extra_http_headers"] = headers
        return await AsyncNewContext(self._browser, **clean)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        cf, self._cf = self._cf, None
        if cf is not None:
            try:
                await cf.__aexit__(None, None, None)
            except Exception:
                pass


class _Chromium:
    """`.chromium` shim — kept for the bot's API contract; it is really
    Camoufox/Firefox, which is the whole point."""

    async def launch(self, headless=True, args=None, proxy=None, **kwargs):
        return await _launch_browser(headless=bool(headless), proxy=proxy)


class _Playwright:
    def __init__(self):
        self._started = False

    async def start(self):
        self._started = True
        return self

    async def stop(self):
        self._started = False

    @property
    def chromium(self) -> _Chromium:
        return _Chromium()


def async_playwright():
    return _Playwright()

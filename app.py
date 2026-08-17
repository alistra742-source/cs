import asyncio
import base64
import json
import os
import random
import re
import threading
import time
from typing import Dict, List, Optional

import aiohttp
import requests as _requests

from flask import Flask, jsonify, request, Response

try:
    import db
    _db_available = True
except ImportError:
    db = None
    _db_available = False
    print("[app] db.py not found - token saving disabled", flush=True)

try:
    from proxies import (pool as proxy_pool, configured as _proxies_configured,
                         proxy_files, proxy_files_signature, used_store)
    _proxies_available = True
except ImportError:
    proxy_pool = None
    used_store = None
    _proxies_configured = lambda: False
    _proxies_available = False
    print("[app] proxies.py not found - direct connections only", flush=True)

# "force use the proxies no matter what" — when residential sessions are
# configured (proxies.txt in the repo, or VAULTPROXY_* env) the workers
# NEVER fall back to TOR. Set PROXY_MODE=force to force even without a file.
PROXY_FORCE = (
    (os.environ.get("PROXY_MODE") or "").strip().lower()
    in ("force", "1", "true", "yes")
    or _proxies_configured()
)

# Fall back to TOR (socks5://127.0.0.1:9050) when the proxy pool is
# exhausted or every session is dead (e.g. vaultproxies at 0.00 GB quota).
# Disable with TOR_FALLBACK=0.
TOR_FALLBACK = (os.environ.get("TOR_FALLBACK") or "").strip().lower() not in ("0", "false", "no", "off")

from server import DiscordAutomation, _tor_check, ENGINE
import live_control

# ── Global state (Flask thread + asyncio thread) ──

_loop: Optional[asyncio.AbstractEventLoop] = None
_running = False
_start_time = 0.0

# worker_id -> worker state
_workers: Dict[str, dict] = {}
WORKER_COUNT = 1
WORKER_IDS = [f"B{i+1}" for i in range(WORKER_COUNT)]

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# duckmail.sbs delivers inboxes on the Discord-friendly domain
# @glasswhitehub.com. Extra inbox domains can be listed in config
# (mail_domains); a domain is burned at runtime when a signup ends in phone
# verification so it is never reused.
DEFAULT_MAIL_DOMAIN = "glasswhitehub.com"

DEFAULT_CONFIG = {
    "headless": True,
    "web_port": 8080,
    "camera_interval": 3,
    "worker_count": WORKER_COUNT,
    "mail_domains": ["glasswhitehub.com"],
    "custom_email": "",
}


def load_config(path: str = _config_path) -> dict:
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                saved = json.load(f)
                for key, value in DEFAULT_CONFIG.items():
                    if key in saved:
                        config[key] = saved[key]
        except Exception:
            pass
    config["web_port"] = int(os.environ.get("PORT", config.get("web_port", 8080)))
    return config


def save_config(config: dict, path: str = _config_path) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass


# Shared ring buffer so app-level lines (proxy stats, worker outcomes) reach
# the web terminal, not just stdout.
_APP_LOGS: list = []

# Domains that got burned at runtime (phone verification on signup). Seeded
# from config.json and persisted back so burned domains stay burned.
# Already-tried domains (mikerossy.com, blobers.it.com, vibify.cc, vibeify.cc)
# are burned up front so a stale config.json can never revive them.
_BURNED_DOMAINS: set = {
    "mikerossy.com", "blobers.it.com", "vibify.cc", "vibeify.cc",
}


def _load_burned() -> None:
    global _BURNED_DOMAINS
    _BURNED_DOMAINS = {"mikerossy.com", "blobers.it.com", "vibify.cc", "vibeify.cc"}
    try:
        cfg = load_config()
        _BURNED_DOMAINS.update(
            str(d).strip().lower() for d in (cfg.get("burned_domains") or []))
    except Exception:
        pass


def _burn_domain(domain: str) -> None:
    """Permanently remove a domain from the pool after a phone-verification hit."""
    d = (domain or "").strip().lower()
    if not d:
        return
    _BURNED_DOMAINS.add(d)
    try:
        cfg = load_config()
        burned = [str(x).strip().lower() for x in (cfg.get("burned_domains") or [])]
        if d not in burned:
            burned.append(d)
        cfg["burned_domains"] = burned
        mail = [str(x).strip().lower() for x in (cfg.get("mail_domains") or [])]
        if d in mail:
            mail.remove(d)
        cfg["mail_domains"] = mail
        save_config(cfg)
    except Exception:
        pass


def _pick_domain(cfg: dict) -> str:
    """Pick a fresh, non-burned inbox domain from the configured list (falls
    back to duckmail's default @glasswhitehub.com)."""
    pools = [
        [str(x).strip().lower() for x in (cfg.get("mail_domains") or []) if str(x).strip()],
        [DEFAULT_MAIL_DOMAIN],
    ]
    for pool in pools:
        fresh = [d for d in pool if d not in _BURNED_DOMAINS]
        if fresh:
            return random.choice(fresh)
    return DEFAULT_MAIL_DOMAIN


# App-level logs: proxy sweeps, AI warm-up, worker chatter etc. only appear
# in the ALL logs (LOG_LEVEL=all). Warnings / errors always print.
_APP_LOG_ALL = os.environ.get("LOG_LEVEL", "").strip().lower() \
    in ("all", "debug", "verbose")


def _log(msg: str, level: str = "info"):
    # Store EVERYTHING so the dashboard's ALL LOGS toggle can show it; only
    # print warnings/errors to the console (and everything with LOG_LEVEL=all).
    essential = level in ("warn", "error")
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "timestamp": time.time(),
        "level": level,
        "essential": essential,
        "message": msg,
    }
    _APP_LOGS.append(entry)
    if len(_APP_LOGS) > 400:
        _APP_LOGS[:] = _APP_LOGS[-400:]
    if _APP_LOG_ALL or essential:
        print(f"[{level.upper()}] {msg}", flush=True)


# ── Worker management (runs in the asyncio thread) ──

def _init_worker(wid: str) -> dict:
    return {
        "id": wid,
        "bot": None,
        "status": "idle",          # idle | starting | running | done | error
        "step": "",
        "email": "",
        "username": "",
        "password": "",
        "token": "",
        "proxy": "",
        "started_at": 0,
        "finished_at": 0,
        "screenshots": 0,
        "last_shot_b64": "",
        "launching": False,
    }


async def _worker_capture_loop(wid: str, cfg: dict, stagger: int) -> None:
    """Capture screenshots for this worker, staggered so browsers don't all
    upload at the same time: B1 immediately, B2 after 2s, B3 after 4s..."""
    bot: DiscordAutomation = _workers[wid]["bot"]
    base = max(1, int(cfg.get("camera_interval", 3)))
    # One image every base seconds across ALL browsers: each worker waits
    # base * len(WORKER_IDS) between its own uploads, staggered base apart.
    interval = base * len(WORKER_IDS)
    await asyncio.sleep(stagger)
    while _running and bot is not None and bot._page is not None:
        try:
            shot = await asyncio.wait_for(bot.capture_screenshot(), timeout=25)
            if shot:
                _workers[wid]["last_shot_b64"] = shot
                _workers[wid]["screenshots"] += 1
        except Exception:
            pass
        await asyncio.sleep(interval)


async def _next_proxy(force: bool = False):
    """Grab the least-recently-used proxy session, or None (auto mode only —
    the caller may fall back to TOR). In force mode the pool is refreshed on
    demand so proxies are ALWAYS used and the bot never silently goes TOR."""
    if not (_proxies_available and proxy_pool is not None):
        return None
    if proxy_pool.count == 0:
        try:
            await proxy_pool.refresh()
        except Exception:
            pass
    if proxy_pool.count > 0:
        return proxy_pool.take()
    return None


async def _probe_gated_proxy(wid: str, bot, tries: int = 2):
    """Draw the next session and only accept it if it probes live against
    discord.com (the same gate the worker loop uses). Dead sessions are
    blacklisted as they're found, so an expired proxies.txt can't trap the
    LIVE tab in a chrome-error loop. Returns a proven-live proxy or None."""
    for _ in range(tries):
        try:
            proxy = await _next_proxy(force=PROXY_FORCE)
        except Exception:
            proxy = None
        if proxy is None:
            return None
        if (bot.proxy or {}).get("key") == proxy.get("key"):
            return proxy  # already the session the browser is on
        if proxy_pool is None:
            return proxy
        try:
            live = await asyncio.wait_for(proxy_pool.probe(proxy), timeout=5.0)
        except Exception:
            live = False
        if live:
            return proxy
        _log(f"[{wid}] [Live] Probe failed - dead session, blacklisting {proxy.get('key','?')[:44]}", level="info")
        try:
            proxy_pool.release(proxy, ok=False)
        except Exception:
            pass
    return None


def _proxy_stats_line(wid: str) -> None:
    """Log live proxy usage counters (used / working / failed) for the terminal."""
    try:
        s = proxy_pool.stats() if proxy_pool is not None else {}
    except Exception:
        return
    _log(
        f"[{wid}] [Proxy] Used {s.get('used', 0)} sessions, "
        f"Working {s.get('working', 0)}, Failed {s.get('failed', 0)}"
    )


async def _run_worker(wid: str, cfg: dict, proxy=None) -> None:
    """Worker loop: one proxy session per signup attempt, rotating on failure.
    Falls back to proxy pool + TOR as needed."""
    state = _workers[wid]
    state["status"] = "starting"
    state["started_at"] = time.time()
    max_tries = 30 if PROXY_FORCE else 12

    # ── Reuse a browser parked on Discord by a previous run ──
    # Stop leaves the browser ALIVE on discord.com; Start picks it up here
    # instead of cold-launching Brave + CDP (the slow part). A dead parked
    # browser (circuit dropped, browser crashed) is closed and relaunched.
    bot = state.get("bot")
    if bot is not None:
        try:
            bot._stopped.clear()
        except Exception:
            pass
        if not await bot.is_alive():
            _log(f"[{wid}] Parked browser died while stopped - launching fresh", level="warn")
            try:
                await bot.close()
            except Exception:
                pass
            bot = None
            state["bot"] = None
        else:
            _log(f"[{wid}] Reusing parked browser (already on Discord) - skipping cold launch")

    consecutive_tunnel_fails = 0  # fast-fail after consecutive dead connections
    backoff = 0.3  # seconds between attempts; doubles after dead-session failures
    tor_fallback = False  # flipped to True when the proxy pool proves dead → TOR

    for attempt in range(max_tries):
        if not _running:
            state["status"] = "stopped"
            # PARK the browser on Discord — the next Start reuses it
            # (is_alive() gates the reuse; a dead one gets relaunched).
            return

        # Re-read the config on every attempt so a custom email / headless
        # change made in the dashboard mid-run is picked up on the very next
        # attempt (a stale cfg would keep using the old email forever).
        try:
            cfg = load_config()
        except Exception:
            pass

        # ── Pick a session for this attempt ──
        if proxy is None and not tor_fallback:
            proxy = await _next_proxy(force=PROXY_FORCE)
        if proxy is None and not tor_fallback:
            if TOR_FALLBACK and _tor_check():
                tor_fallback = True
                _log(f"[{wid}] [Proxy] No usable proxy sessions — falling back to TOR (socks5://127.0.0.1:9050)", level="warn")
            else:
                _log(f"[{wid}] [Proxy] No proxy sessions (forced mode) — refreshing and waiting...", level="warn")
                state["proxy"] = "waiting-for-proxy"
                await asyncio.sleep(5)
                continue
        state["proxy"] = proxy.get("key", "tor") if proxy else "tor"

        label = state["proxy"]

        # ── Fast liveness probe BEFORE launching a browser ──
        # A dead session costs ~10s+ when we only discover it after the
        # browser boots and the goto times out. Probing first (3s cap, plain
        # HTTP round-trip through the session) blacklists dead sessions in
        # seconds and skips the browser launch entirely for them. Skipped for
        # the same session being reused (already proven live this round).
        if (proxy and proxy.get("host")
                and (bot is None or (bot.proxy or {}).get("key") != proxy.get("key"))
                and proxy_pool is not None):
            try:
                probe_ok = await proxy_pool.probe(proxy)
            except Exception:
                probe_ok = False
            if not probe_ok:
                _log(f"[{wid}] [Proxy] Probe failed — session dead, blacklisting {proxy.get('key','?')[:44]}...", level="info")
                proxy_pool.release(proxy, ok=False)
                proxy = None
                consecutive_tunnel_fails += 1
                backoff = min(backoff * 2, 8)
                if consecutive_tunnel_fails >= 4:
                    if TOR_FALLBACK and _tor_check() and not tor_fallback:
                        tor_fallback = True
                        proxy = None
                        consecutive_tunnel_fails = 0
                        _log(f"[{wid}] [Proxy] All proxy sessions appear dead — falling back to TOR (socks5://127.0.0.1:9050)", level="info")
                        _proxy_stats_line(wid)
                        await asyncio.sleep(1)
                        continue
                    _log(f"[{wid}] {consecutive_tunnel_fails} consecutive tunnel failures — aborting (all sessions appear dead)")
                    break
                _proxy_stats_line(wid)
                await asyncio.sleep(backoff)
                continue

        # ── Launch or reuse browser (fresh domain each attempt) ──
        domain = _pick_domain(cfg)
        if bot is None:
            bot = DiscordAutomation(
                headless=cfg.get("headless", True),
                proxy=proxy,  # dict = sticky session; None = TOR in _build_context
                worker_id=wid,
                domain=domain,
                email=cfg.get("custom_email") or "",
            )
            state["bot"] = bot
            try:
                await bot.initialize()
            except Exception as e:
                state["status"] = "error"
                _log(f"[{wid}] Browser launch failed: {e}", level="error")
                if proxy:
                    proxy_pool.release(proxy, ok=False)
                    proxy = None
                await asyncio.sleep(3)
                continue
        else:
            # Reuse browser: rotate to a fresh session / TOR circuit
            # Custom email (if set) always wins; otherwise blank = fresh inbox.
            bot._email = cfg.get("custom_email") or ""
            bot._domain = domain
            bot.phone_verify_detected = False
            bot._nav_ok = False
            if (proxy is not None and bot.proxy is not None
                    and proxy.get("key") == bot.proxy.get("key")):
                # Same sticky session — the browser is ALREADY on Discord.
                # Keep the page and just re-navigate; no context rebuild, no
                # bounce through about:blank.
                _log(f"[{wid}] Same proxy session reused - keeping browser on Discord")
            elif not await bot.switch_proxy(proxy):
                _log(f"[{wid}] Context rebuild failed", level="warn")
                if proxy:
                    proxy_pool.release(proxy, ok=False)
                    proxy = None
                consecutive_tunnel_fails += 1
                backoff = min(backoff * 2, 8)
                if consecutive_tunnel_fails >= 4:
                    if TOR_FALLBACK and _tor_check() and not tor_fallback:
                        tor_fallback = True
                        proxy = None
                        consecutive_tunnel_fails = 0
                        _log(f"[{wid}] [Proxy] All proxy sessions appear dead — falling back to TOR (socks5://127.0.0.1:9050)", level="info")
                        _proxy_stats_line(wid)
                        await asyncio.sleep(1)
                        continue
                    _log(f"[{wid}] {consecutive_tunnel_fails} consecutive tunnel failures — aborting (all sessions appear dead)")
                    break
                await asyncio.sleep(backoff)
                continue

        # ── Run signup ──
        try:
            state["status"] = "running"
            stagger = int(wid[1:]) - 1
            cam_task = asyncio.create_task(_worker_capture_loop(wid, cfg, stagger * int(cfg.get("camera_interval", 3))))
            ok = await bot.start_discord_signup()
            cam_task.cancel()

            # ── Capture final screenshot for the LIVE BROWSER view ──
            try:
                if bot is not None and bot._page is not None:
                    shot = await asyncio.wait_for(bot.capture_screenshot(), timeout=25)
                    if shot:
                        state["last_shot_b64"] = shot
                        state["screenshots"] += 1
            except Exception:
                pass

            # ── Phone verification hit → burn this domain + rotate everything ──
            if not ok and getattr(bot, "phone_verify_detected", False):
                _burn_domain(domain)
                _log(f"[{wid}] [Phone] Domain {domain} burned - proxy+fingerprint+domain will rotate", level="warn")

            # ── Clean up temp-mail session between attempts to prevent
            # aiohttp connector leaks (each failed attempt creates a new
            # duckmail inbox that must be closed).
            if bot._mail is not None:
                try:
                    await bot._mail.close()
                except Exception:
                    pass
                bot._mail = None

            acc = bot.get_account()
            state["email"] = acc["email"]
            state["username"] = acc["username"]
            state["password"] = acc["password"]
            state["token"] = acc["token"]
            if ok and acc["token"]:
                state["status"] = "done"
                if _db_available and db is not None:
                    await db.save_account(
                        email=acc["email"], username=acc["username"],
                        password=acc["password"], token=acc["token"],
                        proxy=label, worker_id=wid,
                        user_id=acc.get("user_id", ""),
                        avatar=acc.get("avatar", ""),
                        bio=acc.get("bio", ""),
                        humanized=bool(acc.get("humanized")),
                    )
                _log(f"[{wid}] Done - token {len(acc['token'])} chars ({label})")
                if proxy:
                    proxy_pool.release(proxy, ok=True)
                    # Record the session's REAL egress IP (resolved by the
                    # browser) in the persistent store so future redeploys
                    # can see + skip this sticky IP.
                    try:
                        if (used_store is not None and bot is not None
                                and getattr(bot, "_exit_ip", "")):
                            await used_store.record(
                                proxy.get("key"), "valid", bot._exit_ip)
                    except Exception:
                        pass
                _proxy_stats_line(wid)
                # Park the browser on Discord (account visible in LIVE BROWSER)
                # so the next Start reuses it. The next run's switch_proxy
                # rotates to a fresh context/IP anyway.
                return
            elif ok:
                state["status"] = "done"
                _log(f"[{wid}] Signup ok (no token yet)")
                # Account was created but the token isn't there yet (usually a
                # custom email the user must verify manually). Persist it as
                # pending so it is never lost — the user clicks the verify
                # link in their own inbox and the account is theirs.
                if (_db_available and db is not None
                        and acc.get("email") and acc.get("username")
                        and acc.get("password")):
                    await db.save_account(
                        email=acc["email"], username=acc["username"],
                        password=acc["password"], token=acc.get("token", ""),
                        proxy=label, worker_id=wid,
                        user_id=acc.get("user_id", ""),
                        avatar=acc.get("avatar", ""), bio=acc.get("bio", ""),
                        humanized=bool(acc.get("humanized")),
                    )
                    _log(f"[{wid}] Pending account saved (email verification required to unlock token)")
                # Park the browser on Discord for reuse on the next Start.
                return

            # ── Track consecutive tunnel failures ──
            # With residential proxies (proxy dict), 4 dead sessions in a row
            # means the pool is dry — abort early instead of burning attempts.
            # With TOR (no proxy), every attempt gets a fresh exit node via
            # _tor_newnym() — each circuit is independent, so short backoffs
            # and no early abort; let max_tries (12) run its course.
            using_tor = proxy is None and getattr(bot, "_tor_enabled", False)
            nav_ok = bool(getattr(bot, "_nav_ok", False))
            if not ok and not nav_ok:
                consecutive_tunnel_fails += 1
                if using_tor:
                    backoff = min(backoff * 2, 2.0)    # TOR: fast rotate, tiny backoff
                    abort_at = max_tries               # never early-abort on TOR
                else:
                    backoff = min(backoff * 2, 8)      # residential: longer cooldown
                    abort_at = 4                       # dry pool → stop fast
                if consecutive_tunnel_fails >= abort_at:
                    if (not using_tor and TOR_FALLBACK and _tor_check() and not tor_fallback):
                        tor_fallback = True
                        consecutive_tunnel_fails = 0
                        _log(f"[{wid}] [Proxy] All proxy sessions appear dead — falling back to TOR (socks5://127.0.0.1:9050)", level="info")
                    else:
                        reason = "all TOR circuits blocked" if using_tor else "all sessions appear dead"
                        _log(f"[{wid}] {consecutive_tunnel_fails} consecutive tunnel failures — aborting ({reason})")
                        break
            else:
                consecutive_tunnel_fails = 0
                backoff = 0.3

            # ── Per-attempt failure summary — self-explanatory ──
            nav_error = getattr(bot, "_nav_error", "") or ""
            has_email = bool(getattr(bot, "_email", ""))
            mail_failed = bool(getattr(bot, "_mail_failed", False))
            if mail_failed:
                reason = "inbox creation failed — no email available"
            elif not nav_ok:
                reason = nav_error or "navigation failed (no reason recorded)"
            elif not ok:
                reason = "signup failed (form/captcha/phone)"
            else:
                reason = "unknown"
            _log(f"[{wid}] Attempt {attempt+1}/{max_tries}: {reason} [{label}]", level="warn")
        except Exception as e:
            state["status"] = "error"
            _log(f"[{wid}] error: {e}")
        # Session failed — release it so the next attempt rotates to a new one
        if proxy:
            proxy_pool.release(proxy, ok=False)
            proxy = None
        _proxy_stats_line(wid)
        await asyncio.sleep(backoff)

    state["status"] = "error"
    state["step"] = "retries exhausted - all proxy/TOR attempts failed"
    # Park the browser on Discord — the next Start reuses it (or relaunches
    # if it died while parked). No close() here.
    if bot:
        state["bot"] = bot

async def _proxy_validate_loop() -> None:
    """Background: re-confirm which proxies can reach Discord, using the
    worker's single-shot HTTPS probe. Dead sessions get blacklisted so
    workers never waste a browser launch on them. Runs every 3 minutes."""
    while True:
        try:
            if _proxies_available and proxy_pool is not None and proxy_pool.count:
                # Count truly Discord-reachable sessions (proven by sweep
                # or worker success), not the "all loaded" default.
                reachable = sum(
                    1 for p in proxy_pool._proxies
                    if p.get("_valid") and p.get("key") not in proxy_pool._failed
                )
                bl = len(proxy_pool._failed)
                _log(f"[Proxy] Live validation: {reachable} Discord-reachable, "
                     f"{bl} blacklisted of {proxy_pool.count} loaded")
        except Exception as e:
            _log(f"[Proxy] validation error: {e}", level="warn")
        await asyncio.sleep(180)


async def _proxy_file_watcher(interval: float = 15.0) -> None:
    """Reload the proxy pool the moment proxies.txt / vaultproxies.txt changes.

    vaultproxies sessions carry a ~10 min TTL, so a list loaded at startup is
    stale within minutes. This watcher lets the user drop a FRESH session list
    into proxies.txt and have the bot pick it up without restarting the web
    server. Only triggers when the file CONTENT actually changes (re-saving
    the same expired list is a no-op)."""
    try:
        sig = proxy_files_signature()
    except Exception:
        sig = ""
    while True:
        await asyncio.sleep(interval)
        try:
            new_sig = proxy_files_signature()
            if new_sig and new_sig != sig:
                sig = new_sig
                if proxy_pool is not None:
                    await proxy_pool.refresh()
                    n = proxy_pool.count
                    src = ", ".join(p.name for p in proxy_files()) or "env"
                    _log(f"[Proxy] proxies file changed — reloaded {n} sessions from {src}", level="warn")
                    if n:
                        try:
                            sw = await proxy_pool.sweep(window=10.0, log=_log)
                            _log(f"[Proxy] Re-sweep: {sw['reachable']} Discord-reachable of {n} reloaded")
                        except Exception:
                            pass
        except Exception as e:
            _log(f"[Proxy] file watcher error: {e}", level="warn")


async def _start_all_async(cfg: dict) -> None:
    global _running, _start_time
    if _running:
        return
    _running = True
    _start_time = time.time()

    for wid in WORKER_IDS:
        # Preserve a browser parked on Discord by a previous run so Start
        # reuses it instantly instead of cold-launching Brave + CDP.
        parked = (_workers.get(wid) or {}).get("bot")
        _workers[wid] = _init_worker(wid)
        if parked is not None:
            _workers[wid]["bot"] = parked
            _log(f"[{wid}] Browser parked from previous run - will reuse it on Discord")

    # ── Proxy pool: load free + residential sessions (retry a few times) ──
    n_sessions = 0
    try:
        if _proxies_available and proxy_pool is not None:
            for _r in range(3):
                await proxy_pool.refresh()
                n_sessions = proxy_pool.count
                if n_sessions:
                    break
                await asyncio.sleep(2)
    except Exception as e:
        _log(f"[Proxy] pool refresh error: {e}", level="warn")
    if n_sessions:
        _src = ", ".join(p.name for p in proxy_files()) or "VAULTPROXY_* env"
        _log((f"[Proxy] {n_sessions} proxy sessions loaded from {_src} — "
              f"one IP per account (forced mode)") if PROXY_FORCE
             else f"[Proxy] {n_sessions} proxy sessions loaded from {_src} — one IP per account")
    elif PROXY_FORCE:
        _log("[Proxy] [ERROR] PROXY FORCE MODE but 0 sessions loaded — workers will keep retrying, TOR is DISABLED", level="error")
    else:
        _log("[Proxy] No proxy sessions — TOR-only fallback (fresh circuit per attempt)")

    # Watch for the user dropping a FRESH session list into proxies.txt —
    # TTL sessions expire ~10 min after issuance, so hot-reload beats restart.
    asyncio.create_task(_proxy_file_watcher())

    if n_sessions and _proxies_available and proxy_pool is not None:
        # ── Start workers IMMEDIATELY — they self-probe proxies ──
        # The sweep below runs concurrently; workers don't wait for it.
        # Each worker does a fast single-shot probe before launching a
        # browser, so dead sessions are caught in ~3s not 10s.
        for i, wid in enumerate(WORKER_IDS):
            _log(f"[{wid}] Starting worker...")
            asyncio.create_task(_run_worker(wid, cfg, None))

        # ── Background sweep: test against discord.com (real, not ipify) ──
        # This runs concurrently with workers. Results only improve
        # future proxy picks; workers don't wait for it.
        _log(f"[Proxy] Background sweep of {n_sessions} sessions against discord.com (10s window)...")
        try:
            sw = await proxy_pool.sweep(window=10.0, log=_log)
            _log(f"[Proxy] Sweep done: {sw['reachable']} Discord-reachable, "
                 f"{sw['unproven']} unproven (available, re-checked on use), "
                 f"{sw['untested']} untested of {n_sessions} — "
                 f"workers probe-gate every session before launching a browser")
            if sw.get("tested") and not sw.get("reachable"):
                _log(
                    "[Proxy] 0 of the loaded sessions can reach Discord. "
                    "vaultproxies sessions expire (ttl-600 = 10 min) and cannot "
                    "be revived — re-saving the SAME session IDs under a new "
                    "filename changes nothing (it's the identical expired list). "
                    "Generate a FRESH session list in the vaultproxies dashboard "
                    "and save it as proxies.txt — the session IDs (the part after "
                    "'-s-') must be NEW. The bot auto-reloads proxies.txt when it "
                    "changes, so save the fresh list and the next sweep picks it up.",
                    level="info",
                )
        except Exception as e:
            _log(f"[Proxy] Sweep error: {e}", level="warn")
        asyncio.create_task(_proxy_validate_loop())

    if not n_sessions:
        # No proxy sessions — start workers directly (TOR fallback)
        for i, wid in enumerate(WORKER_IDS):
            _log(f"[{wid}] Starting worker...")
            asyncio.create_task(_run_worker(wid, cfg, None))


async def _stop_all_async() -> None:
    global _running
    _running = False
    _APP_LOGS.clear()
    for wid, state in list(_workers.items()):
        bot = state.get("bot")
        if bot is not None:
            # Signal an in-flight navigation/signup to abort immediately.
            try:
                bot._stopped.set()
            except Exception:
                pass
            # Browsers stay ALIVE and parked on Discord so the next Start
            # reuses them instantly (is_alive() gates the reuse; dead ones
            # relaunch). No close() here.
        if state["status"] in ("starting", "running"):
            state["status"] = "stopped"
    _log("[App] All workers stopped (browser parked on Discord - reused on next Start)")


def _run_in_loop(coro) -> Optional[object]:
    if not _loop:
        _log("[Loop] Event loop not running!", level="error")
        return None
    try:
        fut = asyncio.run_coroutine_threadsafe(coro, _loop)
        return fut.result(timeout=120)
    except Exception as e:
        _log(f"[Loop] Error running coroutine: {e}", level="error")
        import traceback
        traceback.print_exc()
        return None


async def _live_navigate_robust(wid: str, bot, url: str) -> dict:
    """Navigate the live tab and self-heal a dead proxy tunnel.

    A parked/launched browser can sit on an expired residential session —
    discord.com then shows 'site can't be reached' (ERR_TUNNEL_CONNECTION_FAILED).
    Probe-gate the next session exactly like the worker loop, then fall back
    to TOR, then to a direct connection, so the LIVE tab never stays stuck on
    chrome-error://chromewebdata/.
    """
    st = await live_control.live_navigate(bot, url)
    if not st.get("error"):
        return st
    first_err = st.get("error", "")
    _log(f"[{wid}] [Live] Navigate failed ({first_err}) — rotating session and retrying", level="warn")
    # The session the browser is currently on just produced chrome-error:
    # blacklist it so it is never handed out again this run.
    if bot.proxy and proxy_pool is not None:
        try:
            proxy_pool.release(bot.proxy, ok=False)
        except Exception:
            pass
    for _attempt in range(3):
        proxy = await _probe_gated_proxy(wid, bot)
        swapped = False
        via = ""
        if proxy is not None:
            via = "proxy"
            try:
                swapped = await bot.switch_proxy(proxy)
            except Exception:
                swapped = False
            if not swapped and proxy_pool is not None:
                try:
                    proxy_pool.release(proxy, ok=False)
                except Exception:
                    pass
        elif TOR_FALLBACK and _tor_check():
            via = "tor"
            _log(f"[{wid}] [Live] No live proxy sessions — falling back to TOR", level="warn")
            try:
                swapped = await bot.switch_proxy(None)  # fresh TOR circuit
            except Exception:
                swapped = False
        else:
            via = "direct"
            _log(f"[{wid}] [Live] No live proxy and TOR unavailable — using direct connection", level="warn")
            try:
                swapped = await bot.switch_direct()
            except Exception:
                swapped = False
        if not swapped:
            continue
        st = await live_control.live_navigate(bot, url)
        if not st.get("error"):
            _log(f"[{wid}] [Live] Navigation recovered via {via}")
            return st
        if via == "proxy" and proxy is not None and proxy_pool is not None:
            try:
                proxy_pool.release(proxy, ok=False)
            except Exception:
                pass
    st["error"] = f"site unreachable after retries ({first_err})"
    return st


async def _start_live_browser(wid: str, url: str = "",
                              force: bool = False) -> dict:
    """Attach (or cold-launch) the worker's real browser for the LIVE tab.
    The bot shares this same page, so the operator can watch it work or take
    over. Proxy-first, TOR fallback — exactly like the worker. Navigates only
    on a cold launch or when ``force`` is set, so opening the tab never yanks
    a running signup off the page it is filling."""
    state = _workers.get(wid) or _init_worker(wid)
    _workers[wid] = state
    if not url:
        url = "https://discord.com/register"
    # The gen is already driving this SAME browser — never relaunch a second
    # one on top of it and never yank it off the page it is filling.
    # Just report what the worker is doing so the LIVE tab shows it live.
    if state.get("status") in ("starting", "running"):
        bot = state.get("bot")
        if bot is not None:
            st = await live_control.get_live_state(bot)
            st["launching"] = state.get("status") == "starting"
            st["status"] = state.get("status", "")
            return st
        # Worker is mid-launch (bot not created yet) — wait for it instead of
        # racing it and leaking a second browser.
        return {"connected": False, "worker_id": wid, "url": url,
                "title": "", "viewport_width": 1920,
                "viewport_height": 1080, "browser": ENGINE,
                "screenshot": "", "error": "", "launching": True,
                "status": state.get("status", "")}
    if state.get("launching"):
        # A launch is already in flight — report it instead of starting a
        # second browser on top of the first (which would leak the first).
        return {"connected": False, "worker_id": wid, "url": url,
                "title": "", "viewport_width": 1920,
                "viewport_height": 1080, "browser": ENGINE,
                "screenshot": "", "error": "", "launching": True}
    state["launching"] = True
    via = "attach"
    launched = False
    try:
        bot = state.get("bot")
        cfg = load_config()
        if bot is None:
            bot = DiscordAutomation(
                headless=bool(cfg.get("headless", True)),
                proxy=None,
                worker_id=wid,
                domain=_pick_domain(cfg),
                email=cfg.get("custom_email") or "",
            )
            state["bot"] = bot
        try:
            alive = await bot.is_alive()
        except Exception:
            alive = False
        if not alive:
            # Probe-gate the first session: launching straight onto an expired
            # residential tunnel is what left the LIVE tab on chrome-error.
            proxy = await _probe_gated_proxy(wid, bot)
            if proxy is not None:
                bot.proxy = proxy
                bot._direct = False
                via = "proxy"
            elif TOR_FALLBACK and _tor_check():
                bot.proxy = None
                bot._direct = False
                via = "tor"
            else:
                bot._direct = True
                bot.proxy = None
                via = "direct"
            _log(f"[{wid}] [Live] Launching browser ({via}, engine={ENGINE})…")
            await asyncio.wait_for(bot.initialize(), timeout=90)
            _log(f"[{wid}] [Live] Browser launched ({via})")
            launched = True
        if url:
            # Navigate whenever the page isn't already where the operator
            # asked it to be. A parked browser on about:blank (or a stale
            # error page) must NOT be treated as "still filling a signup" —
            # that was leaving the LIVE tab on a permanent white screen.
            cur = ""
            try:
                cur = str(bot._page.url or "") if bot._page else ""
            except Exception:
                cur = ""
            if force or launched or cur.rstrip("/") != url.rstrip("/"):
                return await _live_navigate_robust(wid, bot, url)
        return await live_control.get_live_state(bot)
    except Exception as e:
        import traceback
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        _log(f"[{wid}] [Live] Live browser failed ({via}): {e}\n{tb}", level="error")
        return {"connected": False, "worker_id": wid, "url": "",
                "title": "", "viewport_width": 1920,
                "viewport_height": 1080, "browser": ENGINE,
                "screenshot": "", "error": f"browser launch failed: {e}"}
    finally:
        state["launching"] = False


async def _close_live_browser(wid: str) -> bool:
    state = _workers.get(wid)
    bot = state.get("bot") if state else None
    if bot is None:
        return False
    try:
        bot._stopped.set()
    except Exception:
        pass
    try:
        await bot.close()
    except Exception:
        pass
    state["bot"] = None
    state["last_shot_b64"] = ""
    return True


# ── Flask app ─────────────────────────────────────────────

app = Flask(__name__)


@app.route('/')
def handle_root():
    return Response(DASHBOARD_HTML, content_type='text/html')


@app.route('/start', methods=['POST'])
def handle_start():
    global _workers
    if _running:
        return jsonify({"ok": False, "msg": "Already running"})
    try:
        cfg = load_config()
        threading.Thread(
            target=lambda: _run_in_loop(_start_all_async(cfg)),
            daemon=True,
        ).start()
        return jsonify({"ok": True, "msg": "Started — 1 browser launching"})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Start error: {e}"})


@app.route('/stop', methods=['POST'])
def handle_stop():
    _run_in_loop(_stop_all_async())
    return "Stopped"


@app.route('/proxies/refresh', methods=['POST'])
def handle_proxy_refresh():
    if _proxies_available and proxy_pool is not None:
        _run_in_loop(proxy_pool.refresh())
        return jsonify(proxy_pool.stats())
    return jsonify({"error": "proxies module not loaded"})


def _mask_proxy_key(key: str) -> str:
    """Display-safe form of a proxy key (user:pass@host:port) — never leak
    the credentials: show the short session id + host only."""
    host = key.rsplit("@", 1)[-1] if "@" in key else key
    sid = ""
    m = re.search(r"-s-([A-Za-z0-9]+)", key)
    if m:
        sid = "s-" + m.group(1)[:10]
    return f"{sid} @ {host}" if sid else host


@app.route('/proxies')
def handle_proxies():
    """Proxy dashboard data: valid / used / invalid / unproven sessions.

    Merges the live pool state with the persistent used_proxies store so the
    tab still shows history (and the pool skips already-used sticky IPs)
    after a redeploy."""
    rows = []
    if _db_available and db is not None:
        rows = _run_in_loop(db.list_proxies()) or []
    db_map = {r.get("key"): r for r in rows}

    def _mk(key, rec, live_flag=False):
        return {
            "label": _mask_proxy_key(key),
            "ip": (live_flag and rec.get("ip")) or rec.get("exit_ip") or "",
            "used": True,
            "invalid": rec.get("status") == "invalid",
            "valid": rec.get("status") == "valid",
        }

    valid, invalid, used = [], [], []
    live = set()
    if proxy_pool is not None:
        for p in proxy_pool._proxies:
            key = p.get("key", "")
            live.add(key)
            rec = db_map.get(key) or {}
            status = "invalid" if key in proxy_pool._failed else (
                "valid" if p.get("_valid") else rec.get("status") or "unproven")
            entry = {
                "label": _mask_proxy_key(key),
                "ip": p.get("_resolved_ip") or rec.get("exit_ip") or "",
                "used": (key in proxy_pool._used_at
                          or key in proxy_pool._used_before),
                "invalid": status == "invalid",
                "valid": status == "valid",
            }
            if entry["invalid"]:
                invalid.append(entry)
            elif entry["valid"]:
                valid.append(entry)
            if entry["used"]:
                used.append(entry)

    # Persistent history that survives redeploys: DB rows not in the live pool.
    for key, rec in db_map.items():
        if key in live:
            continue
        entry = {
            "label": _mask_proxy_key(key),
            "ip": rec.get("exit_ip") or "",
            "used": True,
            "invalid": rec.get("status") == "invalid",
            "valid": rec.get("status") == "valid",
        }
        if entry["invalid"]:
            invalid.append(entry)
        else:
            used.append(entry)

    return jsonify({
        "total": len(proxy_pool._proxies) if proxy_pool is not None else len(db_map),
        "valid": valid,
        "invalid": invalid,
        "used": used,
        "db": bool(db_map),
        "stats": proxy_pool.stats() if proxy_pool is not None else {},
    })

@app.route('/status')
def handle_status():
    workers = []
    for wid in WORKER_IDS:
        s = _workers.get(wid) or _init_worker(wid)
        workers.append({
            "id": wid,
            "status": s["status"],
            "step": s["step"],
            "email": s["email"],
            "username": s["username"],
            "token": s["token"],
            "proxy": s["proxy"],
            "screenshots": s["screenshots"],
            "started_at": s["started_at"],
        })
    try:
        cfg_now = load_config()
        _mail_domains = cfg_now.get("mail_domains", []) or []
    except Exception:
        _mail_domains = []
    return jsonify({
        "running": _running,
        "uptime": int(time.time() - _start_time) if _start_time else 0,
        "workers": workers,
        "mail_domains": _mail_domains,
        "custom_email": cfg_now.get("custom_email", ""),
        "proxies": proxy_pool.stats() if (_proxies_available and proxy_pool is not None) else {},
    })


@app.route('/latest')
def handle_latest_screenshot():
    wid = request.args.get("worker", "B1")
    s = _workers.get(wid)
    if s and s.get("last_shot_b64"):
        try:
            return Response(base64.b64decode(s["last_shot_b64"]),
                            content_type='image/png')
        except Exception:
            pass
    # Fallback: try the bot's own screenshot store
    if s:
        bot = s.get("bot")
        if bot is not None:
            try:
                shot = bot.get_latest_screenshot()
                if shot:
                    return Response(base64.b64decode(shot),
                                    content_type='image/png')
            except Exception:
                pass
    return Response(status=404)


# ── LIVE CONTROL routes ──────────────────────────────────

@app.route('/browser/state')
def handle_browser_state():
    wid = request.args.get("worker", "B1")
    s = _workers.get(wid)
    bot = s.get("bot") if s else None
    if bot is None:
        return jsonify({"connected": False, "worker_id": wid, "url": "",
                        "title": "", "viewport_width": 1920,
                        "viewport_height": 1080, "browser": ENGINE,
                        "screenshot": "", "error": "browser not started"})
    st = _run_in_loop(live_control.get_live_state(bot))
    if st is None:
        return jsonify({"connected": False, "worker_id": wid, "url": "",
                        "title": "", "viewport_width": 1920,
                        "viewport_height": 1080, "browser": ENGINE,
                        "screenshot": "", "error": "event loop unavailable"}), 503
    if st.get("screenshot"):
        s["last_shot_b64"] = st["screenshot"]
    elif s.get("last_shot_b64"):
        st["screenshot"] = s["last_shot_b64"]
    # Surface the gen's status so the LIVE tab shows "launching browser…"
    # during START instead of a misleading "browser not started".
    st["launching"] = bool(s.get("launching") or s.get("status") == "starting")
    st["status"] = s.get("status", "")
    return jsonify(st)


@app.route('/browser/navigate', methods=['POST'])
def handle_browser_navigate():
    wid = request.args.get("worker", "B1")
    data = request.get_json(silent=True) or {}
    url = str(data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "empty url"}), 400
    s = _workers.get(wid)
    bot = s.get("bot") if s else None
    if bot is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "browser not started"}), 409
    st = _run_in_loop(_live_navigate_robust(wid, bot, url))
    if st is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "event loop unavailable"}), 503
    if st.get("screenshot"):
        s["last_shot_b64"] = st["screenshot"]
    return jsonify(st)


@app.route('/browser/action', methods=['POST'])
def handle_browser_action():
    wid = request.args.get("worker", "B1")
    data = request.get_json(silent=True) or {}
    s = _workers.get(wid)
    bot = s.get("bot") if s else None
    if bot is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "browser not started"}), 409
    st = _run_in_loop(live_control.live_action(bot, data))
    if st is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "event loop unavailable"}), 503
    return jsonify(st)


@app.route('/browser/start', methods=['POST'])
def handle_browser_start():
    wid = request.args.get("worker", "B1")
    data = request.get_json(silent=True) or {}
    url = str(data.get("url") or "").strip()
    force = bool(data.get("force"))
    st = _run_in_loop(_start_live_browser(wid, url, force=force))
    if st is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "event loop unavailable"}), 503
    return jsonify(st)


@app.route('/browser/close', methods=['POST'])
def handle_browser_close():
    wid = request.args.get("worker", "B1")
    closed = _run_in_loop(_close_live_browser(wid))
    return jsonify({"closed": bool(closed)})


@app.route('/worker/<wid>/logs')
def handle_worker_logs(wid):
    s = _workers.get(wid)
    if not s:
        return jsonify({"logs": list(_APP_LOGS[-200:]), "status": _running and "starting" or "idle"})
    bot = s.get("bot")
    bot_logs = bot.get_activity_log() if bot else []
    # Merge app-level lines ([Proxy] stats, [B1] Done/Failed, errors) with the
    # bot's internal activity log so the terminal shows everything.
    merged = list(bot_logs)
    seen = {(e.get("time"), e.get("message")) for e in bot_logs}
    for e in _APP_LOGS:
        k = (e.get("time"), e.get("message"))
        if k not in seen:
            seen.add(k)
            merged.append(dict(e))
    merged.sort(key=lambda e: e.get("timestamp", 0))
    return jsonify({
        "id": wid,
        "status": s["status"],
        "email": s.get("email", ""),
        "username": s.get("username", ""),
        "proxy": s.get("proxy", ""),
        "screenshots": s.get("screenshots", 0),
        "started_at": s.get("started_at", 0),
        "logs": merged,  # store caps (500 bot / 400 app) bound the size
    })


@app.route('/tokens')
def handle_tokens():
    if not _db_available or db is None:
        return jsonify({"count": 0, "valid": 0, "expired": 0, "pending": 0,
                        "accounts": [], "stats": {"total": 0, "valid": 0,
                        "expired": 0, "pending": 0}, "error": "DB not available"})
    accounts = _run_in_loop(db.list_accounts(limit=500)) or []
    expired = sum(1 for a in accounts if a.get("status") == "invalid")
    valid = sum(1 for a in accounts if a.get("status") == "valid")
    pending = len(accounts) - expired - valid
    return jsonify({
        "count": len(accounts),
        "valid": valid,
        "expired": expired,
        "pending": pending,
        "accounts": accounts,
        "stats": {"total": len(accounts), "valid": valid,
                   "expired": expired, "pending": pending},
    })


@app.route('/validate', methods=['POST'])
def handle_validate():
    if not _db_available or db is None:
        return jsonify({"error": "DB not available"})
    # Cap at 200 so the synchronous validate stays inside the 120s loop budget
    accounts = _run_in_loop(db.list_accounts(limit=200)) or []
    valid = _run_in_loop(db.validate_all_tokens(accounts)) if accounts else 0
    accounts = _run_in_loop(db.list_accounts(limit=200)) or []
    expired = sum(1 for a in accounts if a.get("status") == "invalid")
    return jsonify({"count": len(accounts), "valid": valid, "expired": expired,
                    "accounts": accounts})


@app.route('/export', methods=['POST'])
def handle_export():
    """Preview the next N accounts for export (does NOT delete)."""
    if not _db_available or db is None:
        return jsonify({"error": "DB not available"})
    data = request.get_json(silent=True) or {}
    try:
        count = max(1, min(int(data.get('count', 5)), 100))
    except Exception:
        count = 5
    mode = 'full' if data.get('mode') == 'full' else 'tokens'
    accounts = _run_in_loop(db.list_accounts(limit=500)) or []
    chosen = [a for a in accounts if a.get('token')][:count]
    out = []
    for a in chosen:
        if mode == 'full':
            text = "\n".join([
                a.get('token') or '',
                "Email: " + str(a.get('email') or ''),
                "Password: " + str(a.get('password') or ''),
                "Username: " + str(a.get('username') or ''),
            ])
        else:
            text = a.get('token') or ''
        out.append({
            "id": a.get("id"),
            "text": text,
            "token": a.get('token'),
            "email": a.get('email'),
            "username": a.get('username'),
        })
    return jsonify({"count": len(out), "accounts": out})


@app.route('/export/delete', methods=['POST'])
def handle_export_delete():
    """Delete exported accounts after the user confirms the copy."""
    if not _db_available or db is None:
        return jsonify({"error": "DB not available"})
    data = request.get_json(silent=True) or {}
    ids = []
    for i in (data.get('ids') or []):
        try:
            ids.append(int(i))
        except Exception:
            pass
    if not ids:
        return jsonify({"ok": False, "msg": "no ids"})
    deleted = _run_in_loop(db.delete_accounts(ids)) or 0
    return jsonify({"ok": True, "deleted": deleted})


@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        cfg = load_config()
        if 'headless' in data:
            cfg['headless'] = bool(data['headless'])
        if 'worker_count' in data:
            cfg['worker_count'] = int(data['worker_count'])
        if 'mail_domains' in data:
            domains = [str(d).strip().lower() for d in data['mail_domains'] if str(d).strip()]
            cfg['mail_domains'] = domains or ["glasswhitehub.com"]
        if 'custom_email' in data:
            cfg['custom_email'] = str(data.get('custom_email') or '').strip().lower()
        save_config(cfg)
        return jsonify({"ok": True, "config": cfg})
    cfg = load_config()
    avail = [d for d in cfg.get("mail_domains", [DEFAULT_MAIL_DOMAIN])
             if d not in _BURNED_DOMAINS]
    return jsonify({"headless": cfg.get("headless", True),
                    "worker_count": cfg.get("worker_count", WORKER_COUNT),
                    "mail_domains": cfg.get("mail_domains", ["glasswhitehub.com"]),
                    "custom_email": cfg.get("custom_email", ""),
                    "burned_domains": sorted(_BURNED_DOMAINS),
                    "available_domains": avail})


# ── Background event loop ─────────────────────────────────

def _run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def main() -> None:
    global _loop
    _load_burned()
    config = load_config()
    web_port = config.get("web_port", 8080)

    if not os.path.exists(_config_path):
        save_config(config)

    _loop = asyncio.new_event_loop()
    t = threading.Thread(target=_run_event_loop, args=(_loop,), daemon=True)
    t.start()

    # Auto-migrate DB (DATABASE_URL from env)
    if _db_available and db is not None:
        _run_in_loop(db.init_db())

    print("=" * 56, flush=True)
    print("  EYES GEN - multi-browser Discord token generator", flush=True)
    print(f"  Browsers per Start: {WORKER_COUNT}", flush=True)
    print(f"  Dashboard: http://0.0.0.0:{web_port}", flush=True)
    print("=" * 56, flush=True)

    app.run(host='0.0.0.0', port=web_port, debug=False,
            use_reloader=False, threaded=True)


# ═══════════════════════════════════════════════════════════
# EYES GEN DASHBOARD — mobile-first
# ═══════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!doctype html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0a0a0b">
<meta charset="utf-8">
<title>EY3 - Token Forge</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
:root{
  --bg:#0a0a0b;--panel:#131316;--panel2:#1a1a1e;--line:#26262b;--line2:#34343a;
  --txt:#e7e7ea;--dim:#8a8a92;--dim2:#5c5c64;
  --ok:#34d399;--bad:#f87171;--warn:#fbbf24;
}
html{background:var(--bg)}
body{font-family:'Space Grotesk',-apple-system,'Segoe UI',Roboto,sans-serif;
  background:
    radial-gradient(900px 400px at 85% -10%, rgba(255,255,255,.045), transparent 60%),
    radial-gradient(700px 380px at -10% 0%, rgba(255,255,255,.03), transparent 55%),
    var(--bg);
  color:var(--txt);min-height:100vh;max-width:980px;margin:0 auto;padding:18px 16px 90px}
h1{font-family:'JetBrains Mono',monospace;font-size:30px;font-weight:700;letter-spacing:2px;
  display:flex;align-items:center;gap:12px}
h1 .tag{font-size:11px;font-weight:500;letter-spacing:2px;color:var(--dim2);
  border:1px solid var(--line);border-radius:99px;padding:4px 10px;background:var(--panel)}
.sub{color:var(--dim);font-size:12px;margin:4px 0 16px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--dim2);display:inline-block;
  box-shadow:0 0 10px currentColor}
.dot.on{background:var(--ok);color:var(--ok)}
.dot.err{background:var(--bad);color:var(--bad)}
nav{display:flex;gap:6px;margin-bottom:18px;position:sticky;top:0;z-index:40;
  background:rgba(10,10,11,.92);backdrop-filter:blur(10px);padding:10px 0;border-bottom:1px solid var(--line)}
nav button{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;letter-spacing:2px;
  padding:10px 22px;border-radius:10px;border:1px solid var(--line);background:var(--panel);
  color:var(--dim);cursor:pointer;transition:all .15s}
nav button.on{background:#e7e7ea;color:#0a0a0b;border-color:#e7e7ea}
nav button:not(.on):hover{border-color:var(--line2);color:var(--txt)}
.tab{display:none}.tab.on{display:block}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;margin-bottom:14px}
.card h3{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;color:var(--dim);
  text-transform:uppercase;margin-bottom:12px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 12px;text-align:center}
.stat .num{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700}
.stat .lbl{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--dim);margin-top:4px}
.num.g{color:var(--ok)}.num.r{color:var(--bad)}.num.a{color:var(--txt)}
button{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;letter-spacing:1.5px;
  padding:11px 18px;border-radius:10px;border:1px solid var(--line2);background:var(--panel2);
  color:var(--txt);cursor:pointer;transition:all .15s}
button:active{transform:scale(.97)}
button.primary{background:#e7e7ea;color:#0a0a0b;border-color:#e7e7ea}
button.danger{background:#2a1212;color:#fca5a5;border-color:#5a2323}
button.ok{background:#0f2e24;color:#6ee7b7;border-color:#1d4a3a}
button:disabled{opacity:.45;cursor:not-allowed}
.btnrow{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.btnrow .grow{flex:1;min-width:120px}
.badge{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;padding:3px 9px;border-radius:99px;letter-spacing:.5px;white-space:nowrap}
.b-pending{background:#3a2c12;color:var(--warn)}
.b-valid{background:#0f2e24;color:var(--ok)}
.b-invalid{background:#3a1212;color:var(--bad)}
.b-dim{background:#1e1e22;color:var(--dim)}
.dom{display:flex;gap:8px;align-items:center;margin-bottom:8px}
.dom input{flex:1;font-family:'JetBrains Mono',monospace;font-size:13px;background:var(--panel2);
  border:1px solid var(--line);border-radius:9px;color:var(--txt);padding:10px 12px;outline:none}
.dom input:focus{border-color:var(--dim)}
.dom .x{background:none;border:none;color:var(--dim);font-size:16px;padding:4px 8px;cursor:pointer}
.dom .x:hover{color:var(--bad)}
.pick{display:flex;flex-wrap:wrap;gap:8px}
.chip{font-size:11px;letter-spacing:.5px;padding:8px 13px;border-radius:99px;background:var(--panel2);color:var(--dim);border:1px solid var(--line2);cursor:pointer}
.chip.on{background:#e7e7ea;color:#0a0a0b;border-color:#e7e7ea}
.hint{color:var(--dim2);font-size:11px;margin-top:8px}
.tog{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--dim)}
.sw{width:44px;height:24px;border-radius:99px;background:var(--line2);position:relative;cursor:pointer;transition:.2s}
.sw::after{content:'';position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;background:#fff;transition:.2s}
.sw.on{background:var(--dim)}.sw.on::after{left:23px}
.term{background:#050506;border:1px solid var(--line);border-radius:14px;overflow:hidden;
  font-family:'JetBrains Mono',monospace;box-shadow:0 18px 50px rgba(0,0,0,.5)}
.term-head{display:flex;align-items:center;gap:8px;padding:10px 14px;background:var(--panel2);
  border-bottom:1px solid var(--line);font-size:11px;letter-spacing:1px;color:var(--dim)}
.term-head .cd{width:10px;height:10px;border-radius:50%;background:#3a3a40;display:inline-block}
.term-head .cd.r{background:var(--bad)}.term-head .cd.y{background:var(--warn)}.term-head .cd.g{background:var(--ok)}
.term-head .t{flex:1;text-align:center;letter-spacing:3px;color:var(--dim2)}
.pxline{padding:7px 14px;background:#0a0a0c;border-bottom:1px solid var(--line);font-size:11px;letter-spacing:.5px;color:var(--dim2)}
.chk{display:inline-flex;align-items:center;gap:6px;font-size:11px;letter-spacing:1px;color:var(--dim2);cursor:pointer;user-select:none}
.chk input{accent-color:var(--ok)}
.term-body{height:430px;overflow-y:auto;padding:14px;font-size:12px;line-height:1.65}
.tl{display:flex;gap:10px;white-space:pre-wrap;word-break:break-word;padding:2px 0;border-bottom:1px solid rgba(38,38,43,.25)}
.tl .tt{color:var(--dim2);min-width:58px}
.tl.info .tm{color:#c9c9cf}
.tl.ok .tm{color:#6ee7b7}
.tl.warn .tm{color:var(--warn)}
.tl.error .tm{color:var(--bad)}
.acc{display:flex;align-items:center;gap:12px;background:var(--panel2);border:1px solid var(--line);
  border-radius:12px;padding:11px 12px;margin-bottom:8px}
.av{width:42px;height:42px;border-radius:50%;object-fit:cover;background:#222;flex:none;
  border:1px solid var(--line2)}
.av.ph{display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;
  font-weight:700;color:var(--dim);font-size:14px}
.meta{flex:1;min-width:0}
.meta .u{font-weight:700;font-size:14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.meta .u .id{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--dim2)}
.meta .e{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--dim);margin-top:2px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.meta .b{font-size:10px;color:var(--dim2);margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tok{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:#9fb0c8;word-break:break-all;
  background:#07080b;border:1px solid var(--line);border-radius:8px;padding:7px 9px;margin-top:6px}
.acc .acts{display:flex;gap:6px;flex:none}
.acc .acts button{font-size:9.5px;padding:7px 9px;letter-spacing:1px}
.empty{color:var(--dim2);text-align:center;padding:34px 0;font-size:13px;font-family:'JetBrains Mono',monospace}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:90;
  justify-content:center;align-items:center;padding:16px}
.overlay.on{display:flex}
.modal{background:var(--panel);border:1px solid var(--line2);border-radius:16px;width:94vw;max-width:560px;
  max-height:86vh;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 30px 80px rgba(0,0,0,.6)}
.modal-head{display:flex;justify-content:space-between;align-items:center;padding:15px 18px;
  border-bottom:1px solid var(--line)}
.modal-head h2{font-family:'JetBrains Mono',monospace;font-size:15px;letter-spacing:2px}
.modal-close{background:none;border:none;color:var(--dim);font-size:22px;cursor:pointer;padding:0 4px}
.modal-body{padding:16px 18px;overflow-y:auto;flex:1}
label{display:block;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:1.5px;
  color:var(--dim);margin-bottom:6px}
input[type=number],input[type=text]{font-family:'JetBrains Mono',monospace;font-size:15px;
  background:var(--panel2);border:1px solid var(--line2);border-radius:9px;color:var(--txt);
  padding:11px 12px;outline:none;width:100%}
input:focus{border-color:var(--dim)}
.rad{display:flex;gap:16px;margin:14px 0}
.rad label{display:flex;align-items:center;gap:8px;color:var(--txt);font-size:13px;cursor:pointer;margin:0}
.exp{background:#050506;border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0;
  font-size:11px;line-height:1.6;max-height:240px;overflow-y:auto;color:#9fb0c8;white-space:pre-wrap;
  word-break:break-all;display:none}
#viewImg{width:100%;border-radius:10px;border:1px solid var(--line);background:#000;min-height:240px;object-fit:contain}
.vph{display:flex;align-items:center;justify-content:center;min-height:240px;color:var(--dim2);
  font-family:'JetBrains Mono',monospace;font-size:12px;border:1px dashed var(--line);border-radius:10px}
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);z-index:120;
  background:#e7e7ea;color:#0a0a0b;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;
  padding:10px 18px;border-radius:99px;opacity:0;pointer-events:none;transition:opacity .25s}
.toast.on{opacity:1}
.footer{color:#3a3a40;font-size:10px;text-align:center;margin-top:20px;font-family:'JetBrains Mono',monospace;letter-spacing:1px}
.px-cols{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media(max-width:900px){.px-cols{grid-template-columns:1fr}}
.px-col{min-width:0}
.px-head{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;
  padding:8px 10px;border:1px solid var(--line);border-radius:9px 9px 0 0;background:var(--panel2)}
.px-head.g{color:#63d9a8;border-color:#1c4a38}
.px-head.b{color:#8fb4ff;border-color:#22335c}
.px-head.r{color:#ff7a7a;border-color:#5c2222}
.px-list{max-height:420px;overflow-y:auto;border:1px solid var(--line);border-top:none;
  border-radius:0 0 9px 9px;background:#050506}
.px-item{display:flex;justify-content:space-between;gap:8px;align-items:center;
  padding:7px 10px;font-family:'JetBrains Mono',monospace;font-size:10.5px;
  border-bottom:1px solid var(--line);color:var(--dim)}
.px-item:last-child{border-bottom:none}
.px-item .ip{color:#9fb0c8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.px-item .st{flex:none;font-size:9px;letter-spacing:1px}
.px-item .st.used{color:#8fb4ff}
.px-item .st.invalid{color:#ff7a7a}
.px-item .st.valid{color:#63d9a8}
.px-empty{color:var(--dim2);text-align:center;padding:18px 0;font-size:11px;
  font-family:'JetBrains Mono',monospace}
@media(max-width:600px){.stats{gap:6px}.stat{padding:10px 6px}.stat .num{font-size:20px}
  .acc{flex-wrap:wrap}.acc .acts{width:100%;justify-content:flex-end}}
</style></head><body>

<h1>EY3<span class="tag">TOKEN FORGE</span></h1>
<div class="sub"><span class="dot" id="dDot"></span> <span id="stLine">idle</span> - discord token gen - duckmail <span id="domLine">@glasswhitehub.com</span></div>

<nav>
  <button id="nvMain" class="on" onclick="showTab('Main')">MAIN</button>
  <button id="nvProxy" onclick="showTab('Proxy')">PROXIES</button>
  <button id="nvTerm" onclick="showTab('Term')">TERMINAL</button>
  <button id="nvMan" onclick="showTab('Manage')">MANAGE</button>
</nav>

<div id="tabMain" class="tab on">
  <div class="stats">
    <div class="stat"><div class="num a" id="stTotal">0</div><div class="lbl">Generated</div></div>
    <div class="stat"><div class="num g" id="stValid">0</div><div class="lbl">Valid</div></div>
    <div class="stat"><div class="num r" id="stExpired">0</div><div class="lbl">Expired</div></div>
  </div>

  <div class="card">
    <h3>Settings</h3>
    <div style="margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;gap:10px">
      <span class="tog">Headless browsers
        <span class="sw" id="swHeadless" onclick="toggleHeadless()"></span>
      </span>
      <span class="badge b-dim" id="stPending2">0 pending</span>
    </div>
    <div style="margin-bottom:6px">
      <label style="display:block;font-size:12px;opacity:.7;margin-bottom:4px">Custom email (optional - leave empty to auto-generate)</label>
      <input type="text" id="inpEmail" placeholder="your@email.com">
      <button style="margin-top:6px" onclick="saveCustomEmail()">Save email</button>
    </div>
    <h3 style="margin-top:14px">Mail domains (discord-friendly on duckmail)</h3>
    <div id="domPick" class="pick"></div>
    <div class="dom" style="margin-top:10px">
      <input type="text" id="domCustom" placeholder="custom domain e.g. mysite.cc">
      <button onclick="addCustomDomain()">Add custom</button>
    </div>
    <div class="btnrow" style="margin-top:12px">
      <button class="primary" onclick="saveDomains()">Save domains</button>
    </div>
    <div class="hint">Pick ONE discord-friendly domain - choosing one replaces the current and saves immediately. Domains that trigger phone verification get burned automatically.</div>
  </div>
</div>

<div id="tabProxy" class="tab">
  <div class="stats">
    <div class="stat"><div class="num a" id="pxTotal">0</div><div class="lbl">Total</div></div>
    <div class="stat"><div class="num g" id="pxValid">0</div><div class="lbl">Valid</div></div>
    <div class="stat"><div class="num b" id="pxUsed">0</div><div class="lbl">Used</div></div>
    <div class="stat"><div class="num r" id="pxInvalid">0</div><div class="lbl">Invalid</div></div>
  </div>
  <div class="card">
    <h3>Proxy Sessions <span class="badge b-dim" id="pxDbBadge">DB off</span></h3>
    <div class="px-cols">
      <div class="px-col"><div class="px-head g">VALID</div><div id="pxValidList" class="px-list"></div></div>
      <div class="px-col"><div class="px-head b">USED</div><div id="pxUsedList" class="px-list"></div></div>
      <div class="px-col"><div class="px-head r">INVALID</div><div id="pxInvalidList" class="px-list"></div></div>
    </div>
  </div>
</div>

<div id="tabTerm" class="tab">
  <div class="btnrow">
    <button class="ok grow" id="btnStart" onclick="start()">START</button>
    <button class="danger grow" onclick="stop()">STOP</button>
    <button onclick="openView()">VIEW</button>
    <label class="chk"><input type="checkbox" id="showAllChk" onchange="showAll=this.checked;refreshLogs()"> ALL LOGS</label>
  </div>
  <div class="term">
    <div class="term-head">
      <span class="cd" id="cd1"></span><span class="cd" id="cd2"></span><span class="cd" id="cd3"></span>
      <span class="t">EY3 - WORKER B1</span><span class="badge b-dim" id="termState">idle</span>
    </div>
    <div class="pxline" id="pxLine">proxies: checking...</div>
    <div class="term-body" id="termBody"><div class="empty">No activity yet - hit START.</div></div>
  </div>
</div>

<div id="tabManage" class="tab">
  <div class="btnrow">
    <button class="primary" id="btnMode" onclick="toggleMode()">SHOW TOKENS</button>
    <button onclick="doValidate()">VALIDATE</button>
    <button class="ok" onclick="openExport()">EXPORT</button>
    <span class="badge b-dim" style="align-self:center">Total: <span id="manCount">0</span></span>
  </div>
  <div id="accList"><div class="empty">No accounts yet - run the generator first.</div></div>
</div>


<div class="overlay" id="viewOverlay" onclick="if(event.target===this)closeView()">
  <div class="modal">
    <div class="modal-head"><h2>LIVE BROWSER</h2>
      <button class="modal-close" onclick="closeView()">X</button></div>
    <div class="modal-body">
      <div id="viewWrap"><div class="vph">connecting...</div></div>
    </div>
  </div>
</div>

<div class="overlay" id="expOverlay" onclick="if(event.target===this)closeExp()">
  <div class="modal">
    <div class="modal-head"><h2>EXPORT TOKENS</h2>
      <button class="modal-close" onclick="closeExp()">X</button></div>
    <div class="modal-body">
      <label>How many tokens you need?</label>
      <input type="number" id="expCount" value="5" min="1" max="100">
      <div class="rad">
        <label><input type="radio" name="expMode" value="tokens" checked> Tokens alone</label>
        <label><input type="radio" name="expMode" value="full"> Full tokens</label>
      </div>
      <div class="btnrow">
        <button class="primary" onclick="exportGen()">Generate</button>
        <button onclick="exportCopyAll()">Copy all</button>
        <button class="danger" onclick="exportConfirm()">Confirm and delete</button>
      </div>
      <div class="exp" id="expList"></div>
      <div class="hint">Confirm and delete copies the tokens to your clipboard, then removes them from Manage permanently.</div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>
<div class="footer">EY3 - grey iron build</div>

<script>
var ACCOUNTS=[], MODE='accounts', EXPORT=[], VIEWINT=null, HEADLESS=true, VALIDATED_ONCE=false;
var NL2 = String.fromCharCode(10,10);
function $(id){return document.getElementById(id);}
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function toast(m){var t=$('toast');t.textContent=m;t.classList.add('on');clearTimeout(toast._t);toast._t=setTimeout(function(){t.classList.remove('on');},2200);}
function api(p,o){return fetch(p,o);}

function showTab(n){
  var tabs=['Main','Proxy','Term','Manage'];
  for(var i=0;i<tabs.length;i++){
    $('tab'+tabs[i]).classList.toggle('on',tabs[i]===n);
    $('nv'+tabs[i]).classList.toggle('on',tabs[i]===n);
  }
  if(n==='Manage') refreshTokens();
  if(n==='Term') refreshLogs();
  if(n==='Proxy') refreshProxies();
}
function refreshProxies(){
  api('/proxies').then(function(r){return r.json();}).then(function(x){
    $('pxTotal').textContent = x.total||0;
    $('pxValid').textContent = (x.valid||[]).length;
    $('pxUsed').textContent = (x.used||[]).length;
    $('pxInvalid').textContent = (x.invalid||[]).length;
    $('pxDbBadge').textContent = x.db ? 'DB ON — used IPs skipped' : 'DB off — in-session only';
    renderPxList('pxValidList', x.valid||[], 'valid');
    renderPxList('pxUsedList', x.used||[], 'used');
    renderPxList('pxInvalidList', x.invalid||[], 'invalid');
  }).catch(function(){});
}
function renderPxList(id, items, cls){
  var el=$(id), html='';
  var shown = items.slice(0,150);
  for(var i=0;i<shown.length;i++){
    var it=shown[i];
    html+='<div class="px-item"><span class="ip" title="'+esc(it.label)+'">'+esc(it.label)+'</span>'
       +'<span class="ip">'+(it.ip?esc(it.ip):'')+'</span>'
       +'<span class="st '+cls+'">'+cls.toUpperCase()+'</span></div>';
  }
  if(!shown.length) html='<div class="px-empty">none yet</div>';
  if(items.length>shown.length) html+='<div class="px-empty">+'+(items.length-shown.length)+' more</div>';
  el.innerHTML=html;
}

function refreshStatus(){
  api('/status').then(function(r){return r.json();}).then(function(x){
    var st = x.running ? 'on' : ((x.workers||[]).some(function(w){return w.status==='error';}) ? 'err' : '');
    var d=$('dDot');d.className='dot'+(st?' '+st:'');
    var running=(x.workers||[]).filter(function(w){return w.status==='running'||w.status==='starting';}).length;
    $('stLine').textContent = x.running ? ('running - '+running+'/'+(x.workers||[]).length+' browsers - '+Math.floor(x.uptime/60)+'m') : 'idle';
    // Show the email actually in use: the configured custom email wins;
    // otherwise fall back to the auto-generated @domain so the header never
    // lies about "I set my own email but it shows a different domain".
    var dom=$('domLine');
    if(dom){
      if(x.custom_email){dom.textContent=x.custom_email;}
      else if(x.mail_domains&&x.mail_domains.length){dom.textContent='@'+x.mail_domains[0];}
    }
    var px=x.proxies;
    if(px&&$('pxLine')){
      $('pxLine').textContent='proxies: '+px.available+' loaded | '+px.valid+' valid | '+px.used+' used | '+px.working+' working | '+px.failed+' failed';
    }
  }).catch(function(){});
}

var showAll=false;
var OKWORDS=['[ok]','confirmed','solved','ready','rendered','humanized','verification link found'];
function refreshLogs(){
  api('/worker/B1/logs').then(function(r){return r.json();}).then(function(x){
    $('termState').textContent = x.status||'idle';
    var lines=(x.logs||[]).filter(function(l){
      if(showAll) return true;
      // ALL LOGS off: only essential events + warnings/errors (the server
      // tags each entry with an `essential` flag - same rules as the console).
      // ALL LOGS on shows everything.
      var lv=(l.level||'').toLowerCase();
      if(lv==='warn'||lv==='error') return true;
      return !!l.essential;
    }).slice(-150);
    if(!lines.length) return;
    var html='';
    for(var i=0;i<lines.length;i++){
      var l=lines[i];
      var m=l.message||'';
      if(m.indexOf('[B1]')===0)m=m.substring(4);
      var cls='info', lv=(l.level||'').toLowerCase();
      if(lv==='error')cls='error'; else if(lv==='warn')cls='warn';
      else{
        var low=m.toLowerCase();
        for(var k=0;k<OKWORDS.length;k++){ if(low.indexOf(OKWORDS[k])!==-1){cls='ok';break;} }
      }
      html+='<div class="tl '+cls+'"><span class="tt">'+(l.time||'')+'</span><span class="tm">'+esc(m)+'</span></div>';
    }
    var tb=$('termBody');
    var atBottom = tb.scrollHeight - tb.scrollTop - tb.clientHeight < 80;
    tb.innerHTML=html;
    if(atBottom) tb.scrollTop=tb.scrollHeight;
  }).catch(function(){});
}

function start(){
  var b=$('btnStart');b.disabled=true;b.textContent='LAUNCHING...';
  api('/start',{method:'POST'}).then(function(r){return r.json();}).then(function(x){
    toast(x.msg||'Started');
  }).catch(function(e){toast('Start error: '+e.message);})
    .finally(function(){b.disabled=false;b.textContent='START';});
}
function stop(){
  api('/stop',{method:'POST'}).then(function(r){return r.text();}).then(function(t){toast(t);})
    .catch(function(e){toast('Stop error: '+e.message);});
}

function openView(){
  $('viewOverlay').classList.add('on');
  $('viewWrap').innerHTML='<div class="vph">connecting...</div>';
  var snap=function(){
    api('/latest?worker=B1&t='+Date.now()).then(function(r){
      if(!r.ok){ $('viewWrap').innerHTML='<div class="vph">no feed yet - browser not started</div>'; return; }
      return r.blob().then(function(b){
        var u=URL.createObjectURL(b);
        $('viewWrap').innerHTML='<img id="viewImg" src="'+u+'">';
        setTimeout(function(){URL.revokeObjectURL(u);},3000);
      });
    }).catch(function(){});
  };
  snap();
  VIEWINT=setInterval(snap,2000);
}
function closeView(){if(VIEWINT)clearInterval(VIEWINT);VIEWINT=null;$('viewOverlay').classList.remove('on');}

function toggleMode(){
  MODE = MODE==='accounts' ? 'creds' : 'accounts';
  $('btnMode').textContent = MODE==='accounts' ? 'SHOW TOKENS' : 'SHOW ACCOUNTS';
  render();
}
function badge(st){
  var c = st==='valid' ? 'valid' : (st==='invalid' ? 'invalid' : 'pending');
  return '<span class="badge b-'+c+'">'+(st||'pending')+'</span>';
}
function render(){
  var el=$('accList');
  if(!ACCOUNTS.length){el.innerHTML='<div class="empty">No accounts yet - run the generator first.</div>';return;}
  var html='';
  if(MODE==='accounts'){
    for(var i=0;i<ACCOUNTS.length;i++){
      var a=ACCOUNTS[i];
      var user=a.username||'?';
      var av = a.avatar ? '<img class="av" src="'+esc(a.avatar)+'">'
        : '<div class="av ph">'+esc((user||'?').charAt(0).toUpperCase())+'</div>';
      html+='<div class="acc">'+av+
        '<div class="meta"><div class="u">@'+esc(user)+(a.user_id?'<span class="id">ID '+esc(a.user_id)+'</span>':'')+'</div>'+
        '<div class="e">'+esc(a.email||'')+'</div>'+
        '<div class="b">'+((a.bio?'bio: '+esc(a.bio):'')+(a.humanized?' - humanized':''))+'</div></div>'+
        badge(a.status)+
        '<div class="acts">'+
        '<button data-copy="'+esc(user)+'" data-label="USER" onclick="copyBtn(this)">USER</button>'+
        '<button data-copy="'+esc(a.user_id||'')+'" data-label="ID" onclick="copyBtn(this)">ID</button></div></div>';
    }
  }else{
    for(var j=0;j<ACCOUNTS.length;j++){
      var b=ACCOUNTS[j];
      var tok=b.token||'';
      html+='<div class="acc"><div class="meta" style="flex:1;min-width:0">'+
        '<div class="u">@'+esc(b.username||'?')+'</div>'+
        '<div class="tok">'+esc(tok)+'</div>'+
        '<div class="e" style="margin-top:4px">'+esc((b.email||'')+' : '+(b.password||''))+'</div></div>'+
        badge(b.status)+
        '<div class="acts">'+
        '<button data-copy="'+esc(tok)+'" data-label="TOKEN" onclick="copyBtn(this)">TOKEN</button>'+
        '<button data-copy="'+esc((b.email||'')+' : '+(b.password||''))+'" data-label="CREDS" onclick="copyBtn(this)">CREDS</button></div></div>';
    }
  }
  el.innerHTML=html;
}
function refreshTokens(){
  api('/tokens').then(function(r){return r.json();}).then(function(x){
    ACCOUNTS=x.accounts||[];
    $('stTotal').textContent=(x.stats&&x.stats.total)||0;
    $('stValid').textContent=(x.stats&&x.stats.valid)||0;
    $('stExpired').textContent=(x.stats&&x.stats.expired)||0;
    $('stPending2').textContent=((x.stats&&x.stats.pending)||0)+' pending';
    $('manCount').textContent=ACCOUNTS.length;
    render();
    if(!VALIDATED_ONCE && (x.stats&&x.stats.pending)>0){ VALIDATED_ONCE=true; doValidate(); }
  }).catch(function(){});
}
function doValidate(){
  toast('Validating tokens...');
  api('/validate',{method:'POST'}).then(function(r){return r.json();}).then(function(x){
    ACCOUNTS=x.accounts||ACCOUNTS;
    $('stValid').textContent=x.valid||0;
    $('stExpired').textContent=x.expired||0;
    render();
    toast('Validation done - '+x.valid+' valid');
  }).catch(function(e){toast('Validate error: '+e.message);});
}

function openExport(){EXPORT=[];$('expList').style.display='none';$('expOverlay').classList.add('on');}
function closeExp(){$('expOverlay').classList.remove('on');}
function exportGen(){
  var count=parseInt($('expCount').value||'5',10);
  var modeEl=document.querySelector('input[name="expMode"]:checked');
  var mode=modeEl?modeEl.value:'tokens';
  api('/export',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({count:count,mode:mode})}).then(function(r){return r.json();}).then(function(x){
    EXPORT=x.accounts||[];
    if(!EXPORT.length){$('expList').style.display='block';$('expList').textContent='nothing to export - no tokens in Manage';}
    else{
      var parts=[];
      for(var i=0;i<EXPORT.length;i++)parts.push(EXPORT[i].text);
      $('expList').style.display='block';
      $('expList').textContent=parts.join(NL2);
    }
  }).catch(function(e){toast('Export error: '+e.message);});
}
function exportCopyAll(){
  var parts=[];
  for(var i=0;i<EXPORT.length;i++)parts.push(EXPORT[i].text);
  var txt=parts.join(NL2);
  if(!txt)return toast('nothing to copy yet');
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(txt).then(function(){toast('Copied '+EXPORT.length+' to clipboard');});
  }else{
    var ta=document.createElement('textarea');ta.value=txt;document.body.appendChild(ta);ta.select();
    try{document.execCommand('copy');}catch(e){}
    document.body.removeChild(ta);
    toast('Copied '+EXPORT.length+' to clipboard');
  }
}
function exportConfirm(){
  if(!EXPORT.length)return toast('Generate first');
  exportCopyAll();
  var ids=[];
  for(var i=0;i<EXPORT.length;i++){ if(EXPORT[i].id!=null) ids.push(EXPORT[i].id); }
  api('/export/delete',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ids:ids})}).then(function(r){return r.json();}).then(function(x){
    toast('Copied and deleted '+ids.length+' from Manage');
    EXPORT=[];$('expList').style.display='none';closeExp();refreshTokens();
  }).catch(function(e){toast('Delete error: '+e.message);});
}
function copyBtn(btn){
  var t=btn.getAttribute('data-copy')||'';
  if(!t)return toast('nothing to copy');
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(t).then(function(){btn.textContent='COPIED';setTimeout(function(){btn.textContent=btn.getAttribute('data-label')||'COPY';},1200);});
  }else{
    var ta=document.createElement('textarea');ta.value=t;document.body.appendChild(ta);ta.select();
    try{document.execCommand('copy');}catch(e){}
    document.body.removeChild(ta);
    btn.textContent='COPIED';setTimeout(function(){btn.textContent=btn.getAttribute('data-label')||'COPY';},1200);
  }
}

var DOMAINS=[], AVAIL=[];
function loadConfig(){
  api('/config').then(function(r){return r.json();}).then(function(x){
    AVAIL=(x.available_domains&&x.available_domains.length)?x.available_domains:['glasswhitehub.com'];
    DOMAINS=(x.mail_domains&&x.mail_domains.length)?x.mail_domains.slice():['glasswhitehub.com'];
    HEADLESS=x.headless!==false;
    $('swHeadless').classList.toggle('on',HEADLESS);
    if(x.custom_email&&$('inpEmail'))$('inpEmail').value=x.custom_email;
    renderDomains();
  }).catch(function(){AVAIL=['glasswhitehub.com'];DOMAINS=['glasswhitehub.com'];renderDomains();});
}
function renderDomains(){
  var html='';
  for(var i=0;i<AVAIL.length;i++){
    var d=AVAIL[i];
    var sel=DOMAINS.indexOf(d)!==-1;
    html+='<button type="button" class="chip'+(sel?' on':'')+'" data-d="'+esc(d)+'" onclick="pickDomain(this)">'+esc(d)+'</button>';
  }
  for(var j=0;j<DOMAINS.length;j++){
    if(AVAIL.indexOf(DOMAINS[j])===-1){
      html+='<button type="button" class="chip on" data-d="'+esc(DOMAINS[j])+'" onclick="pickDomain(this)">'+esc(DOMAINS[j])+'</button>';
    }
  }
  $('domPick').innerHTML=html;
}
function pickDomain(btn){
  var d=btn.getAttribute('data-d')||'';
  if(!d)return;
  if(DOMAINS.length===1 && DOMAINS[0]===d)return;
  DOMAINS=[d];
  saveDomains();
}
function addCustomDomain(){
  var v=$('domCustom').value.trim().toLowerCase();
  $('domCustom').value='';
  if(!v)return;
  if(DOMAINS.indexOf(v)!==-1)return toast('already in the pool');
  DOMAINS.push(v);renderDomains();
}
function toggleHeadless(){$('swHeadless').classList.toggle('on');}
function saveCustomEmail(){
  var v=$('inpEmail').value.trim().toLowerCase();
  api('/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({custom_email:v})})
    .then(function(r){return r.json();}).then(function(x){
      toast(x.ok?(v?('Custom email set: '+v):'Custom email cleared - auto-generate on'):'save failed');
    }).catch(function(e){toast('Save error: '+e.message);});
}
function saveDomains(){
  var cleaned=[];
  for(var i=0;i<DOMAINS.length;i++){
    var d=DOMAINS[i].trim().toLowerCase();
    if(d && cleaned.indexOf(d)===-1) cleaned.push(d);
  }
  DOMAINS=cleaned;
  renderDomains();
  api('/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mail_domains:DOMAINS,headless:$('swHeadless').classList.contains('on')})})
    .then(function(r){return r.json();}).then(function(x){
      toast(x.ok?('Domains saved - '+((x.config&&x.config.mail_domains)||[]).join(', ')):'save failed');
    }).catch(function(e){toast('Save error: '+e.message);});
}

loadConfig();
refreshStatus();
setInterval(refreshStatus,5000);
setInterval(refreshLogs,2200);
setInterval(function(){refreshTokens();},12000);
refreshLogs();
refreshTokens();
</script></body></html>
"""

if __name__ == "__main__":
    main()

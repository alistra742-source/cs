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

# Database removed - no token persistence

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
import live_ui

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
                domain=domain,
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
                domain=_pick_domain(cfg),
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
    html = DASHBOARD_HTML
    idx = html.rfind("</body>")
    if idx != -1:
        html = html[:idx] + live_ui.LIVE_INJECTION + html[idx:]
    return Response(html, content_type='text/html')

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
    except Exception:
        cfg_now = dict(DEFAULT_CONFIG)
    _mail_domains = cfg_now.get("mail_domains", []) or []
    return jsonify({
        "running": _running,
        "uptime": int(time.time() - _start_time) if _start_time else 0,
        "workers": workers,
        "headless": bool(cfg_now.get("headless", True)),
        "workerCount": int(cfg_now.get("worker_count", WORKER_COUNT)),
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
    if not False or db is None:
        return jsonify({"count": 0, "valid": 0, "expired": 0, "pending": 0,
                        "accounts": [], "stats": {"total": 0, "valid": 0,
                        "expired": 0, "pending": 0}, "error": "DB not available"})
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
    if not False or db is None:
        return jsonify({"error": "DB not available"})
    # Cap at 200 so the synchronous validate stays inside the 120s loop budget
    expired = sum(1 for a in accounts if a.get("status") == "invalid")
    return jsonify({"count": len(accounts), "valid": valid, "expired": expired,
                    "accounts": accounts})

@app.route('/export', methods=['POST'])
def handle_export():
    """Preview the next N accounts for export (does NOT delete)."""
    if not False or db is None:
        return jsonify({"error": "DB not available"})
    data = request.get_json(silent=True) or {}
    try:
        count = max(1, min(int(data.get('count', 5)), 100))
    except Exception:
        count = 5
    mode = 'full' if data.get('mode') == 'full' else 'tokens'
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
    if not False or db is None:
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

DASHBOARD_HTML = r'''
<!doctype html><html lang="en"><head>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0a0a0b">
<meta charset="utf-8">
<title>EY3 - Token Forge</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html{background:#0a0a0b}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:radial-gradient(900px 400px at 85% -10%,rgba(255,255,255,.045),transparent 60%),
    radial-gradient(700px 380px at -10% 0%,rgba(255,255,255,.03),transparent 55%),#0a0a0b;
  color:#e7e7ea;min-height:100vh;max-width:980px;margin:0 auto;padding:16px 16px 90px}
h1{font-family:'JetBrains Mono','Courier New',monospace;font-size:28px;font-weight:700;
  letter-spacing:2px;display:flex;align-items:center;gap:10px}
.sub{color:#8a8a92;font-size:12px;margin:6px 0 16px;display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:#5c5c64;display:inline-block}
.dot.on{background:#34d399;box-shadow:0 0 12px #34d399}
nav{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap;border-bottom:1px solid #26262b;padding-bottom:12px}
nav button{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:1px;
  padding:9px 18px;border-radius:10px;border:1px solid #26262b;background:#131316;
  color:#e7e7ea;cursor:pointer;transition:.15s ease}
nav button:hover{background:#1a1a1e;border-color:#34343a}
nav button.act{background:#34d399;color:#0a0a0b;border-color:#34d399}
.card{background:#131316;border:1px solid #26262b;border-radius:14px;padding:18px 20px;margin-bottom:12px}
.card h3{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:500;
  letter-spacing:1px;margin-bottom:10px;color:#8a8a92;text-transform:uppercase}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.col{flex:1;min-width:110px}
.stat{background:#1a1a1e;border:1px solid #26262b;border-radius:10px;padding:14px;text-align:center}
.stat .n{font-size:28px;font-weight:700;font-family:'JetBrains Mono','Courier New',monospace}
.stat .l{font-size:10px;color:#8a8a92;letter-spacing:1px;margin-top:4px;text-transform:uppercase}
button{padding:9px 22px;border-radius:10px;border:1px solid #26262b;background:#131316;
  color:#e7e7ea;cursor:pointer;font-size:13px;font-weight:600}
button:hover{background:#1a1a1e;border-color:#34343a}
button.primary{background:#34d399;color:#0a0a0b;border-color:#34d399}
button.danger{background:#f87171;color:#0a0a0b;border-color:#f87171}
input,select{padding:9px 12px;border-radius:10px;border:1px solid #26262b;background:#1a1a1e;
  color:#e7e7ea;width:100%;font-size:13px}
label{font-size:11px;color:#8a8a92;display:block;margin-bottom:4px;letter-spacing:1px;
  text-transform:uppercase}
.log-box{background:#060608;border:1px solid #26262b;border-radius:10px;
  font-family:'JetBrains Mono','Courier New',monospace;font-size:11px;line-height:1.55;padding:12px;
  max-height:340px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;color:#8a8a92}
.log-box .warn{color:#fbbf24}.log-box .err{color:#f87171}.log-box .ok{color:#34d399}
#screenshot{width:100%;border-radius:10px;border:1px solid #26262b;margin-top:8px;
  max-height:380px;object-fit:contain;background:#1a1a1e}
.badge{display:inline-block;font-size:10px;padding:3px 10px;border-radius:99px;
  font-family:'JetBrains Mono','Courier New',monospace;letter-spacing:1px}
.badge-ok{background:rgba(52,211,153,.15);color:#34d399}
.badge-err{background:rgba(248,113,113,.15);color:#f87171}
.flex{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.hide{display:none!important}
.mt{margin-top:12px}.mb{margin-bottom:12px}
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1a1a1e;
  border:1px solid #34343a;border-radius:10px;padding:12px 22px;font-size:13px;z-index:999}
</style></head><body>

<h1>EY3 <span style="font-size:11px;color:#8a8a92;border:1px solid #26262b;border-radius:99px;padding:4px 10px;font-weight:500;letter-spacing:2px">TOKEN FORGE</span></h1>

<div class="sub"><span class="dot" id="statusDot"></span> <span id="statusText">loading...</span></div>

<nav id="tabNav">
<button class="act" data-tab="main" onclick="showTab('main')">Dashboard</button>
<button data-tab="tokens" onclick="showTab('tokens')">Tokens</button>
<button data-tab="settings" onclick="showTab('settings')">Settings</button>
</nav>

<div id="tabmain">
<div class="row mb">
<div class="col stat"><div class="n" id="statGenerated">0</div><div class="l">Generated</div></div>
<div class="col stat"><div class="n" id="statValid">0</div><div class="l">Valid</div></div>
<div class="col stat"><div class="n" id="statExpired">0</div><div class="l">Expired</div></div>
</div>

<div class="card">
<div class="flex mb">
<h3 style="margin:0">Proxies</h3>
<span id="proxyStats" class="badge badge-ok">--</span>
</div>
<div class="row">
<div class="col stat"><div class="n" id="proxyTotal">0</div><div class="l">Total</div></div>
<div class="col stat"><div class="n" id="proxyValid">0</div><div class="l">Valid</div></div>
<div class="col stat"><div class="n" id="proxyUsed">0</div><div class="l">Used</div></div>
<div class="col stat"><div class="n" id="proxyInvalid">0</div><div class="l">Invalid</div></div>
</div>
</div>

<div class="card">
<h3>Control</h3>
<div class="flex">
<button class="primary" id="btnStart" onclick="startBot()">START</button>
<button class="danger" id="btnStop" onclick="stopBot()">STOP</button>
<button onclick="refreshProxies()">Refresh Proxies</button>
</div>
</div>

<div class="card">
<h3>Live View</h3>
<img id="screenshot" src="" alt="No screenshot yet">
</div>

<div class="card">
<h3>Logs</h3>
<div class="log-box" id="logBox">Waiting for logs...</div>
<div class="flex mt">
<button onclick="viewAllLogs()">ALL LOGS</button>
</div>
</div>
</div>

<div id="tabtokens" class="hide">
<div class="card">
<h3>Tokens</h3>
<div class="flex mb"><button onclick="refreshTokens()">Refresh</button></div>
<div id="tokenList"><p style="color:#8a8a92">No tokens yet.</p></div>
</div>
</div>

<div id="tabsettings" class="hide">
<div class="card">
<h3>Settings</h3>
<label>Headless</label>
<div class="flex mb">
<button id="hlOn" onclick="setHeadless(true)">ON</button>
<button id="hlOff" onclick="setHeadless(false)">OFF</button>
</div>
<label>Workers</label>
<select id="workerCount" onchange="saveConfig()">
<option value="1">1</option>
<option value="2">2</option>
<option value="3">3</option>
<option value="4">4</option>
</select>
</div>
</div>

<script>
function $(id){return document.getElementById(id)}
function api(path,opts){
  opts=opts||{};
  var f=Object.assign({method:'POST',headers:{'Content-Type':'application/json'}},opts);
  if(opts.body)f.body=JSON.stringify(opts.body);
  return fetch(path,f);
}
function toast(m){
  var t=document.createElement('div');
  t.className='toast';
  t.textContent=m;
  document.body.appendChild(t);
  setTimeout(function(){if(t.parentNode)t.parentNode.removeChild(t)},3000);
}
function showTab(name){
  var ids=['main','tokens','settings'];
  ids.forEach(function(t){
    var el=$('tab'+t);
    if(el)el.classList.toggle('hide',t!==name);
  });
  document.querySelectorAll('#tabNav button').forEach(function(b){
    b.classList.toggle('act',b.getAttribute('data-tab')===name);
  });
}
window.showTab=showTab;

function startBot(){
  api('/start').then(function(r){return r.json()}).then(function(d){
    toast(d.message||'Started');
  }).catch(function(e){toast('Error: '+e.message)});
}
function stopBot(){
  api('/stop').then(function(r){return r.text()}).then(function(t){
    toast(t||'Stopped');
  }).catch(function(e){toast('Error: '+e.message)});
}
window.startBot=startBot;
window.stopBot=stopBot;

function setHeadless(on){
  api('/config',{body:{headless:on}}).then(function(){
    $('hlOn').classList.toggle('primary',on);
    $('hlOff').classList.toggle('primary',!on);
    toast('Saved');
  }).catch(function(e){toast('Error: '+e.message)});
}
window.setHeadless=setHeadless;

function saveConfig(){
  api('/config',{
    body:{
      headless:true,
      workerCount:parseInt($('workerCount').value,10)
    }
  }).then(function(){toast('Saved')}).catch(function(e){toast('Error: '+e.message)});
}
window.saveConfig=saveConfig;

function refreshStatus(){
  fetch('/status').then(function(r){return r.json()}).then(function(s){
    try{
      $('statusDot').className='dot'+(s&&s.running?' on':'');
      $('statusText').textContent=s&&s.running?'running ('+((s.workers&&s.workers.length)||0)+' workers)':'idle';
      $('statGenerated').textContent=(s&&s.generated)||0;
      $('statValid').textContent=(s&&s.valid)||0;
      $('statExpired').textContent=(s&&s.expired)||0;
      if(s&&s.proxies){
        $('proxyTotal').textContent=s.proxies.total||0;
        $('proxyValid').textContent=s.proxies.valid||0;
        $('proxyUsed').textContent=s.proxies.used||0;
        $('proxyInvalid').textContent=s.proxies.invalid||0;
      }
      if(s&&s.headless!==undefined){
        $('hlOn').classList.toggle('primary',!!s.headless);
        $('hlOff').classList.toggle('primary',!s.headless);
      }
      if(s&&s.workerCount){
        $('workerCount').value=s.workerCount;
      }
    }catch(e){}
  }).catch(function(){});
}
window.refreshStatus=refreshStatus;

function refreshLogs(){
  fetch('/worker/B1/logs').then(function(r){return r.json()}).then(function(d){
    try{
      if(d&&d.logs){
        var box=$('logBox');
        // strip ancient lines so the UI shows only fresh activity
        var logs=(d.all_logs||d.logs);
        var cutoff=Date.now()/1000-300;
        var fresh=logs.filter(function(l){return(l.timestamp||l.time||0)>=cutoff});
        box.textContent=fresh.length?fresh.slice(-80).map(function(l){
          return (l.time||'') + ' ' + (l.message||l.m||'');
        }).join('\n'):'No recent activity.';
        box.scrollTop=box.scrollHeight;
      }
    }catch(e){}
  }).catch(function(){});
}
window.refreshLogs=refreshLogs;

function refreshTokens(){
  fetch('/tokens').then(function(r){return r.json()}).then(function(d){
    try{
      var list=$('tokenList');
      if(!d||!d.length){list.innerHTML='<p style="color:#8a8a92">No tokens yet.</p>';return;}
      var html='<table style="width:100%;border-collapse:collapse;font-size:12px">';
      d.forEach(function(a){
        var st=a.valid===true?'<span class="badge badge-ok">Valid</span>':'<span class="badge badge-err">Invalid</span>';
        html+='<tr style="border-bottom:1px solid #26262b">';
        html+='<td style="padding:6px">'+(a.email||'?')+'</td>';
        html+='<td style="padding:6px">'+st+'</td>';
        html+='<td style="padding:6px;font-family:monospace;font-size:10px;color:#5c5c64;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+(a.token||'').slice(0,40)+'...</td>';
        html+='</tr>';
      });
      html+='</table>';
      list.innerHTML=html;
    }catch(e){}
  }).catch(function(){});
}
window.refreshTokens=refreshTokens;

function refreshProxies(){
  api('/proxies/refresh').then(function(r){return r.json()}).then(function(d){
    toast(d.message||d.error||'Refreshed');
  }).catch(function(e){toast('Error: '+e.message)});
}
window.refreshProxies=refreshProxies;

function viewAllLogs(){
  fetch('/worker/B1/logs').then(function(r){return r.json()}).then(function(d){
    var logs=(d&&d.all_logs||d&&d.logs||[]);
    var text=logs.map(function(l){return(l.time||'')+' '+(l.message||l.m||'')}).join('\n');
    var w=window.open('','_blank','width=900,height=600');
    if(!w){alert('Popup blocked'); return;}
    w.document.write('<html><head><title>Logs</title></head>');
    w.document.write('<body style="background:#000;color:#0f0;font-family:monospace;font-size:11px;padding:14px;white-space:pre-wrap;word-break:break-all">');
    w.document.write(text.replace(/[<>&]/g,function(c){return({'<':'&lt;','>':'&gt;','&':'&amp;'})[c]}));
    w.document.write('</body></html>');
    w.document.close();
  }).catch(function(e){alert('Error: '+e.message);});
}
window.viewAllLogs=viewAllLogs;

// Init
refreshStatus();
refreshLogs();
refreshTokens();
setInterval(refreshStatus,5000);
setInterval(refreshLogs,2200);
setInterval(refreshTokens,12000);
</script>
</body></html>

'''


if __name__ == "__main__":
    main()

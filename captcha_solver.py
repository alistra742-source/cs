#!/usr/bin/env python3
"""hCaptcha solver clients + DOM helpers.

Primary solver: NoneCap (https://nonecap.com), paid — reads
``NONECAP_API_KEY`` (falls back to ``API_KEY``), a ``nc_live_…`` bearer token
minted at https://dashboard.nonecap.com/keys.

Backup solver: Nopecha (https://nopecha.com) — its free tier needs NO API key
(the key is optional and tied to the request IP); set ``NOPECHA_API_KEY`` or
``API_KEY2`` only when a paid key exists.

Both send the EXACT sitekey (read from the live page only after the widget has
fully rendered) plus the current page URL, and return a real token that we
inject into the page's ``h-captcha-response`` field. A returned token is NOT
the same as "solved": downstream acceptance is verified by the caller and is
the only event logged as [OK].
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs

import aiohttp

NONECAP_BASE = os.environ.get(
    "NONECAP_BASE", "https://api.nonecap.com/v1"
).rstrip("/")

NOPECHA_BASE = os.environ.get(
    "NOPECHA_BASE", "https://api.nopecha.com"
).rstrip("/")

_SITEKEY_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Nopecha asks for a modern-browser user-agent on token solves.
_NOPECHA_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _api_key() -> str:
    return (os.environ.get("NONECAP_API_KEY") or os.environ.get("API_KEY")
            or "").strip()


def _nopecha_key() -> str:
    return (os.environ.get("NOPECHA_API_KEY") or os.environ.get("API_KEY2")
            or "").strip()


class NoneCapClient:
    """Async client for the NoneCap hCaptcha solve API."""

    def __init__(self, log: Optional[Callable] = None):
        self._log = log or (lambda msg, level="info": None)
        self.stats = {"calls": 0, "ok": 0, "failed": 0}

    @property
    def configured(self) -> bool:
        return bool(_api_key())

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        }

    async def solve(self, sitekey: str, pageurl: str, rqdata: str = "",
                    proxy: Optional[str] = None,
                    timeout: float = 120.0) -> Optional[dict]:
        """Create a solve, poll to terminal, and return token + solve_id.

        Failed solves are never charged by NoneCap.
        """
        if not self.configured:
            self._log("[NoneCap] No API key (set NONECAP_API_KEY)", level="error")
            return None
        self.stats["calls"] += 1
        payload = {
            "type": "hcaptcha_enterprise" if rqdata else "hcaptcha",
            "sitekey": sitekey,
            "url": pageurl,
        }
        if rqdata:
            payload["rqdata"] = rqdata
        if proxy:
            payload["proxy"] = proxy
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_cfg) as s:
                async with s.post(
                    f"{NONECAP_BASE}/solves",
                    params={"wait": 30},
                    headers=self._headers(),
                    json=payload,
                ) as r:
                    if r.status not in (200, 202):
                        body = await r.text()
                        self._log(
                            f"[NoneCap] Solve rejected (HTTP {r.status}): {body[:200]}",
                            level="warn")
                        self.stats["failed"] += 1
                        return None
                    solve = await r.json()
                solve_id = str(solve.get("id") or "")
                # wait=30 may leave the solve pending; poll until terminal.
                if str(solve.get("status")) in ("pending", "solving"):
                    solve = await self._poll(solve_id, timeout)
                token = solve.get("token") if isinstance(solve, dict) else None
                if solve_id and token:
                    self.stats["ok"] += 1
                    self._log(
                        f"[NoneCap] Token received ({len(token)} chars, {solve_id}) — not yet accepted")
                    return {"token": str(token), "solve_id": solve_id}
                err = (solve or {}).get("error") or {}
                self._log(
                    f"[NoneCap] Solve failed: {err.get('code', 'unknown')} "
                    f"{str(err.get('message', ''))[:120]}",
                    level="warn")
                self.stats["failed"] += 1
                return None
        except Exception as e:
            self._log(f"[NoneCap] Solve error: {e}", level="error")
            self.stats["failed"] += 1
            return None

    async def _poll(self, solve_id: str, timeout: float) -> Optional[dict]:
        deadline = time.time() + max(timeout, 30)
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as s:
                while time.time() < deadline:
                    async with s.get(
                        f"{NONECAP_BASE}/solves/{solve_id}",
                        params={"wait": 15},
                        headers=self._headers(),
                    ) as r:
                        if r.status != 200:
                            return None
                        data = await r.json()
                    if str(data.get("status")) not in ("pending", "solving"):
                        return data
        except Exception:
            pass
        return None

    async def report(self, solve_id: str, outcome: str = "accepted",
                     reason: str = "") -> None:
        """Tell NoneCap whether the token was accepted downstream."""
        if not solve_id or not self.configured:
            return
        item = {"solve_id": solve_id, "outcome": outcome}
        if reason:
            item["reason"] = reason
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as s:
                async with s.post(
                    f"{NONECAP_BASE}/feedback",
                    headers=self._headers(),
                    json={"feedback": [item]},
                ) as r:
                    self._log(
                        f"[NoneCap] Feedback reported ({outcome}, HTTP {r.status})")
        except Exception as e:
            self._log(f"[NoneCap] Feedback error: {e}", level="warn")


class NopechaClient:
    """Async client for the Nopecha hCaptcha token API (free tier = no key).

    Free solves are tied to the request IP when no key is supplied. ``key``
    is only added to requests when ``NOPECHA_API_KEY`` / ``API_KEY2`` is set.
    """

    def __init__(self, log: Optional[Callable] = None):
        self._log = log or (lambda msg, level="info": None)
        self.stats = {"calls": 0, "ok": 0, "failed": 0}

    @property
    def configured(self) -> bool:
        return True  # free tier needs no key

    def _key(self) -> Dict[str, str]:
        k = _nopecha_key()
        return {"key": k} if k else {}

    async def solve(self, sitekey: str, pageurl: str, rqdata: str = "",
                    proxy: Optional[dict] = None, useragent: str = "",
                    timeout: float = 120.0) -> Optional[dict]:
        """Submit an hCaptcha token job and poll until a token is produced.

        ``proxy`` is Nopecha's object form (scheme/host/port/username/
        password), NOT a URL string. Returns ``{token, solve_id}`` — the
        caller must still verify Discord accepts the token downstream.
        """
        self.stats["calls"] += 1
        payload: Dict = {
            "type": "hcaptcha",
            "sitekey": sitekey,
            "url": pageurl,
        }
        payload.update(self._key())
        if rqdata:
            payload["data"] = {"rqdata": rqdata}
        if proxy:
            payload["proxy"] = proxy
        if useragent or _NOPECHA_UA:
            payload["useragent"] = useragent or _NOPECHA_UA
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_cfg) as s:
                async with s.post(
                    f"{NOPECHA_BASE}/token/", json=payload
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        self._log(
                            f"[Nopecha] Submit rejected (HTTP {r.status}): {body[:200]}",
                            level="warn")
                        self.stats["failed"] += 1
                        return None
                    resp = await r.json()
                job_id = str((resp or {}).get("data") or "").strip()
                if not job_id:
                    err = (resp or {}).get("error")
                    msg = str((resp or {}).get("message", ""))
                    self._log(
                        f"[Nopecha] Submit returned no job id (error={err}) "
                        f"{msg[:120]}", level="warn")
                    self.stats["failed"] += 1
                    return None
                token = await self._poll(job_id, timeout)
                if token:
                    self.stats["ok"] += 1
                    self._log(
                        f"[Nopecha] Token received ({len(token)} chars, "
                        f"job {job_id[:24]}) — not yet accepted")
                    return {"token": token, "solve_id": job_id}
                self.stats["failed"] += 1
                return None
        except Exception as e:
            self._log(f"[Nopecha] Solve error: {e}", level="error")
            self.stats["failed"] += 1
            return None

    async def _poll(self, job_id: str, timeout: float) -> Optional[str]:
        deadline = time.time() + max(timeout, 30)
        params: Dict = {"id": job_id}
        params.update(self._key())
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as s:
                while time.time() < deadline:
                    async with s.get(
                        f"{NOPECHA_BASE}/token/", params=params
                    ) as r:
                        if r.status != 200:
                            self._log(
                                f"[Nopecha] Poll HTTP {r.status}", level="warn")
                            return None
                        data = await r.json()
                    if not isinstance(data, dict):
                        await asyncio.sleep(2.0)
                        continue
                    token = str(data.get("data") or "").strip()
                    if token and token != job_id and len(token) > 20:
                        return token
                    err = data.get("error")
                    if err == 14:  # incomplete job — keep polling
                        await asyncio.sleep(2.0)
                        continue
                    if err:
                        msg = str(data.get("message", ""))
                        low = msg.lower()
                        if "credit" in low or "free tier" in low:
                            self._log(
                                f"[Nopecha] Out of credits / daily free limit "
                                f"reached (error={err}): {msg[:120]}", level="warn")
                        else:
                            self._log(
                                f"[Nopecha] Solve failed (error={err}): "
                                f"{msg[:120]}", level="warn")
                        return None
                    await asyncio.sleep(2.0)
        except Exception:
            pass
        return None


# ═══════════════════════════════════════════════════════════════
# DOM helpers
# ═══════════════════════════════════════════════════════════════

def _is_valid_sitekey(value: str) -> bool:
    return bool(_SITEKEY_RE.match((value or "").strip()))


_SITEKEY_JS = r"""() => {
    const UUID = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;
    const norm = (v) => {
        const s = String(v == null ? '' : v).trim();
        try { return decodeURIComponent(s); } catch (e) { return s; }
    };
    const out = [];
    const push = (v, src) => {
        const x = norm(v);
        if (UUID.test(x) && !out.some((o) => o.key === x)) {
            out.push({ key: x, src });
        }
    };
    // 1) hCaptcha's own runtime — the exact value the widget was rendered
    //    with. Authoritative when present.
    try {
        if (window.hcaptcha && typeof window.hcaptcha.getSitekey === 'function') {
            push(window.hcaptcha.getSitekey(), 'hcaptcha.getSitekey');
        }
    } catch (e) {}
    // 2) data-sitekey attributes — prefer a visible element over a hidden one.
    const els = Array.from(document.querySelectorAll('[data-sitekey]'));
    els.sort((a, b) => ((b.offsetParent !== null) ? 1 : 0) - ((a.offsetParent !== null) ? 1 : 0));
    for (const el of els) push(el.getAttribute('data-sitekey'), 'data-sitekey');
    // 3) iframe src sitekey param (widget + challenge frames, URL-decoded).
    for (const f of document.querySelectorAll('iframe')) {
        const src = f.getAttribute('src') || f.src || '';
        const m = src.match(/[?&#]sitekey=([^&#]+)/i);
        if (m) push(m[1], 'iframe-src');
    }
    // 4) inline config / render calls in scripts.
    for (const s of document.querySelectorAll('script')) {
        const t = s.textContent || '';
        if (!t) continue;
        let m = t.match(/["']sitekey["']\s*[:=]\s*["']([0-9a-fA-F-]{36})["']/)
            || t.match(/sitekey\s*[:=]\s*["']([^"']{8,})["']/);
        if (m) push(m[1], 'script');
    }
    return out.length ? out[0].key : '';
}"""


async def extract_hcaptcha_sitekey(page) -> str:
    """Pull the EXACT hCaptcha sitekey from the live, fully-rendered page.

    Sources are checked in priority order (hCaptcha runtime → data-sitekey →
    iframe src → inline scripts) and only a well-formed UUID is accepted, so a
    half-mounted widget can never leak a garbage/partial sitekey.
    """
    try:
        sk = await page.evaluate(_SITEKEY_JS)
        if _is_valid_sitekey(str(sk)):
            return str(sk).strip()
    except Exception:
        pass
    # Cross-frame fallback: some layouts mount the widget without a
    # data-sitekey attribute and with an obfuscated iframe src — scan every
    # live Playwright frame URL for the sitekey param.
    try:
        for frame in page.frames:
            try:
                m = re.search(r"[?&#]sitekey=([^&#]+)", frame.url or "")
            except Exception:
                continue
            if m and _is_valid_sitekey(m.group(1)):
                return m.group(1)
    except Exception:
        pass
    return ""


async def extract_hcaptcha_rqdata(page) -> str:
    """Pull the hCaptcha Enterprise rqdata from the live page (best effort)."""
    try:
        val = await page.evaluate("""() => {
            const el = document.querySelector('[data-sitekey]');
            if (el) {
                const v = el.getAttribute('data-rqdata') || el.getAttribute('rqdata');
                if (v && v.length > 8) return v;
            }
            for (const s of document.querySelectorAll('script')) {
                const t = s.textContent || '';
                const m = t.match(/"rqdata"\s*:\s*"([^"]{8,})"/) ||
                          t.match(/'rqdata'\s*:\s*'([^']{8,})'/) ||
                          t.match(/rqdata\s*[:=]\s*["']([^"']{8,})["']/);
                if (m) return m[1];
            }
            return '';
        }""")
        if val:
            return str(val).strip()
    except Exception:
        pass
    # Discord's enterprise widget carries rqdata inside the iframe src as a
    # URL query param (newassets.hcaptcha.com/...&sitekey=...&rqdata=...).
    try:
        rq = await page.evaluate("""() => {
            const iframes = document.querySelectorAll('iframe');
            for (const f of iframes) {
                const m = (f.src || '').match(/[?&#]rqdata=([^&#]+)/);
                if (m) {
                    try { return decodeURIComponent(m[1]); } catch(e) { return m[1]; }
                }
            }
            return '';
        }""")
        if rq and len(str(rq).strip()) > 8:
            return str(rq).strip()
    except Exception:
        pass
    return ""


def extract_rqdata_from_body(body) -> str:
    """Pull the enterprise rqdata out of an hCaptcha network request body.

    hCaptcha's JS POSTs the enterprise payload (which carries ``rqdata``) to
    ``/getcaptcha/<sitekey>`` when the checkbox is clicked. The body is either
    JSON (``{"rqdata": "..."}``, possibly nested under ``enterprisePayload``)
    or URL-encoded form data. Some engines hand back the raw POST body as
    bytes (and it can be non-UTF-8), so bytes are decoded defensively before
    parsing. Returns "" when no rqdata is present.
    """
    if body is None:
        return ""

    # Non-UTF-8 byte bodies: salvage the ASCII rqdata segment directly, since
    # the rqdata blob itself is always ASCII (JWT / base64).
    if isinstance(body, (bytes, bytearray)):
        raw = bytes(body)
        bm = re.search(
            br'rqdata["\']?\s*[:=]\s*["\']?([A-Za-z0-9+/=._-]{8,})',
            raw, re.IGNORECASE)
        if bm:
            try:
                return bm.group(1).decode("ascii", "ignore")
            except Exception:
                pass
        try:
            body = raw.decode("utf-8", "ignore")
        except Exception:
            body = raw.decode("latin-1", "ignore")

    if not isinstance(body, str):
        return ""
    body = body.strip()
    if not body:
        return ""

    # 1) JSON body — rqdata may sit top-level or inside a nested object.
    try:
        data = json.loads(body)

        def _walk(obj):
            if isinstance(obj, dict):
                value = obj.get("rqdata")
                if isinstance(value, str) and value.strip():
                    return value.strip()
                for child in obj.values():
                    hit = _walk(child)
                    if hit:
                        return hit
            elif isinstance(obj, list):
                for child in obj:
                    hit = _walk(child)
                    if hit:
                        return hit
            return ""

        hit = _walk(data)
        if hit and len(hit) > 8:
            return hit
    except Exception:
        pass

    # 2) URL-encoded form body (rqdata=...&...).
    try:
        for key, values in parse_qs(body).items():
            if key.lower() == "rqdata" and values and len(values[0]) > 8:
                return values[0]
    except Exception:
        pass

    # 3) Loose regex fallback for partial or obfuscated bodies.
    try:
        m = re.search(
            r'["\']?rqdata["\']?\s*[:=]\s*["\']([^"\']{8,})["\']',
            body, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
    except Exception:
        pass
    return ""


async def read_hcaptcha_token(page) -> Optional[str]:
    """Read the current h-captcha-response token from the page."""
    try:
        token = await page.evaluate("""() => {
            const ta = document.querySelector('textarea[name="h-captcha-response"]');
            if (ta && ta.value && ta.value.length > 20) return ta.value;
            if (window.hcaptcha && window.hcaptcha.getResponse) {
                const r = window.hcaptcha.getResponse();
                if (r && r.length > 20) return r;
            }
            return '';
        }""")
        if token:
            return token
    except Exception:
        pass
    return None


async def set_hcaptcha_token_on_page(page, token: str) -> bool:
    """Inject a solved NoneCap token into the hCaptcha textarea."""
    try:
        result = await page.evaluate("""(tok) => {
            const ta = document.querySelector('textarea[name="h-captcha-response"]');
            if (ta) {
                ta.value = tok;
                ta.dispatchEvent(new Event('input', {bubbles: true}));
                ta.dispatchEvent(new Event('change', {bubbles: true}));
                return true;
            }
            return false;
        }""", token)
        return bool(result)
    except Exception:
        return False


def proxy_url_from_bot_proxy(proxy: Optional[dict]) -> str:
    """Convert the bot's proxy dict into a URL string NoneCap accepts.

    Credentials are included so the solve egresses from the same sticky exit
    IP the browser will submit the token from (required for IP-bound
    enterprise sitekeys).
    """
    if not isinstance(proxy, dict):
        return ""
    host = (proxy.get("host") or "").strip()
    port = str(proxy.get("port") or "").strip()
    if not host or not port:
        return ""
    scheme = (proxy.get("proto") or "http").strip().lower()
    if scheme == "socks5h":
        scheme = "socks5"
    user = (proxy.get("username") or "").strip()
    pwd = (proxy.get("password") or "").strip()
    auth = f"{user}:{pwd}@" if user else ""
    return f"{scheme}://{auth}{host}:{port}"


def proxy_dict_from_bot_proxy(proxy: Optional[dict]) -> Optional[dict]:
    """Convert the bot's proxy dict into Nopecha's proxy object format.

    Nopecha expects a dict (scheme/host/port/username/password), not a URL
    string. Credentials are included so the solve egresses from the same
    sticky exit IP the browser will submit the token from.
    """
    if not isinstance(proxy, dict):
        return None
    host = (proxy.get("host") or "").strip()
    port = str(proxy.get("port") or "").strip()
    if not host or not port:
        return None
    scheme = (proxy.get("proto") or "http").strip().lower()
    if scheme == "socks5h":
        scheme = "socks5"
    user = (proxy.get("username") or "").strip()
    pwd = (proxy.get("password") or "").strip()
    out = {"scheme": scheme, "host": host, "port": port}
    if user:
        out["username"] = user
    if pwd:
        out["password"] = pwd
    return out

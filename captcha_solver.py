#!/usr/bin/env python3
"""hCaptcha DOM helpers (vision model solves the image grid directly).

The bot solves the challenge itself with a Hugging Face vision model
(see vision_solver.py and server.py's ``_solve_hcaptcha_if_present``).
This module keeps only the pure DOM helpers the flow still needs:

  · extract_hcaptcha_sitekey / extract_hcaptcha_rqdata / extract_rqdata_from_body
    — read the exact sitekey and enterprise rqdata off the live page (still
    useful for diagnostics and future token-based flows);
  · read_hcaptcha_token / set_hcaptcha_token_on_page — read/write the
    ``h-captcha-response`` textarea (read = detecting a solved challenge).
"""

from __future__ import annotations

import json
import re
from typing import Optional
from urllib.parse import parse_qs, unquote

_SITEKEY_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


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
            // 1) ANY element carrying an rqdata-ish attribute (Discord has
            //    mounted it on the widget container / config div / form in
            //    different builds — never just [data-sitekey]).
            const els = document.querySelectorAll(
                '[data-sitekey], [rqdata], [data-rqdata], [data-hcaptcha-rqdata], [data-config], [data-hcaptcha-config]');
            for (const el of els) {
                for (const a of el.attributes) {
                    if (/rqdata/i.test(a.name) && a.value && a.value.length > 8) {
                        return a.value;
                    }
                }
            }
            // 2) JSON-embedded config attribute values (data-config=...{\"rqdata\":...}).
            for (const el of els) {
                for (const a of el.attributes) {
                    const v = (a.value || '');
                    if (!/rqdata/i.test(a.name) && v.length > 8 && v.indexOf('rqdata') !== -1) {
                        const m = v.match(/["']?rqdata["']?\s*[:=]\s*["']([^"']{8,})["']/);
                        if (m) return m[1];
                    }
                }
            }
            // 3) Inline scripts.
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
    # 4) Playwright frame URLs: the widget's own iframe src carries
    #    rqdata=... on enterprise builds (cross-origin, but Playwright sees
    #    every live frame's URL).
    try:
        for frame in page.frames:
            try:
                u = frame.url or ""
            except Exception:
                continue
            m = re.search(r"[?&#]rqdata=([^&#]+)", u)
            if m:
                try:
                    return unquote(m.group(1))
                except Exception:
                    return m.group(1)
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
    """Inject a solved token into the hCaptcha textarea (manual/legacy use)."""
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

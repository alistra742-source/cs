#!/usr/bin/env python3
"""solver.py — standalone hCaptcha solver extracted from the bot.

A self-contained version of the hCaptcha solving flow that lives inside
server.py's DiscordAutomation (``_solve_hcaptcha_if_present`` and friends),
rebuilt as a standalone module + CLI so it can be pointed at ANY page that
shows an hCaptcha widget/challenge, without the Discord token-gen bot and
without changing server.py.

Pipeline (same as the bot):

  1. wait for the hCaptcha widget or challenge iframe to appear,
  2. if only the widget is present: click the "Are you human" checkbox
     (mouse → keyboard → JS, verified against hCaptcha's own signals),
  3. wait for the image challenge to really paint (never a blank shell),
  4. read the challenge instruction ("Please select all images with a
     boat", "Please click on the two elements that are identical", ...),
  5. screenshot every grid tile and ask the Hugging Face vision model
     which tiles satisfy the instruction (vision_solver.py),
  6. click those tiles (or type the answer for text challenges), click
     Verify, and poll until hCaptcha mints the token.

It drives any Playwright-compatible ``page`` — the project's nodriver
engine (browser_engine.py, undetected driver over a real Google Chrome),
stock Playwright Chromium/Firefox, or the bot's own live page — so the
same solver can be reused elsewhere.

CLI usage:

    python solver.py <url> [--headless] [--timeout 120]
        # launch a fresh browser, navigate, solve, print JSON result
    python solver.py --check
        # probe the Hugging Face Inference API (API_KEY valid? model up?)
    python solver.py <url> --output token.txt --screenshot shot.png

Configuration (env vars, same as the bot):

    API_KEY     Hugging Face access token (hf_...) — required
    HF_MODEL    vision model repo id (default Qwen/Qwen2.5-VL-7B-Instruct)
    HF_API_BASE Inference API base URL
    HF_TIMEOUT  per-solve timeout seconds (default 60)
    HF_TILE_TIMEOUT  per-tile yes/no timeout for small VLMs (default 20)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from typing import Callable, List, Optional

from browser_engine import async_playwright, ENGINE
from captcha_solver import extract_hcaptcha_sitekey, read_hcaptcha_token
from vision_solver import HFVisionClient

# ── tiny logger ─────────────────────────────────────────────────────────

def _default_log(msg: str, level: str = "info") -> None:
    print(f"[{level.upper()}] {msg}", flush=True)


# ── frame helpers (generic over any page) ───────────────────────────────

async def _hcaptcha_frame_for(page, iframe):
    """Resolve the live Playwright Frame for an hCaptcha iframe element.

    Mirrors server.py: Locator.content_frame() can return None for attached
    cross-origin iframes on the patched engine, so fall back to the page's
    frame tree, preferring the VISIBLE widget frame (body not aria-hidden
    AND containing a checkbox node) over its hidden twin.
    """
    try:
        frame = await (await iframe.element_handle(timeout=5000)).content_frame()
        if frame is not None:
            return frame
    except Exception:
        pass
    src = ""
    try:
        src = await iframe.get_attribute("src") or ""
    except Exception:
        src = ""
    probe_js = """() => {
        const b = document.body;
        return JSON.stringify({
            cb: !!document.querySelector('#checkbox, [role="checkbox"], .checkbox, input[type="checkbox"], [aria-checked], .button-submit'),
            hidden: b ? b.getAttribute('aria-hidden') : null
        });
    }"""
    best = None
    for f in page.frames:
        try:
            furl = f.url or ""
        except Exception:
            continue
        if "hcaptcha" not in furl:
            continue
        info = None
        try:
            raw = await f.evaluate(probe_js)
            info = json.loads(raw) if raw else None
        except Exception:
            info = None
        if src and src in furl:
            best = best or f
        if info and info.get("cb"):
            if info.get("hidden") != "true":
                return f
            if best is None:
                best = f
    return best


async def _frame_js_ready(page, iframe, js) -> bool:
    frame = await _hcaptcha_frame_for(page, iframe)
    if frame is None:
        return False
    try:
        return bool(await frame.evaluate(js))
    except Exception:
        return False


async def _challenge_iframe(page):
    """First hCaptcha CHALLENGE iframe element, or None."""
    try:
        chall = page.locator(
            'iframe[title*="hCaptcha challenge"], iframe[src*="hcaptcha-challenge"]')
        if await chall.count() > 0:
            return chall.first
    except Exception:
        pass
    return None


async def _widget_iframes(page):
    """Every hCaptcha widget iframe locator element (may include hidden twin)."""
    try:
        return page.locator(
            'iframe[title="Widget containing checkbox for hCaptcha security challenge"], '
            'iframe[src*="newassets.hcaptcha.com"], '
            'iframe[src*="hcaptcha.com"][src*="frame=checkbox"]')
    except Exception:
        return page.locator('iframe[src*="hcaptcha.com"]')


async def _challenge_rendered(page, iframe) -> bool:
    """True only when the challenge iframe has genuinely painted real
    content (image tiles / prompt / verify control), never a loader shell."""
    return await _frame_js_ready(page, iframe, """() => {
        if (document.readyState !== 'complete' &&
            document.readyState !== 'interactive') return false;
        const sized = (el, min) => {
            if (!el) return false;
            try {
                const r = el.getBoundingClientRect();
                return !!(r && r.width >= (min || 1) && r.height >= (min || 1));
            } catch (e) { return false; }
        };
        let tiles = 0;
        for (const img of document.querySelectorAll('img')) {
            if (sized(img, 12)) tiles += 1;
        }
        if (tiles < 4) {
            for (const el of document.querySelectorAll(
                    '.task-image, .challenge-image, [class*="task-image"], ' +
                    '[class*="challenge-image"], [class*="image-grid"], ' +
                    '[class*="image"]')) {
                let painted = false;
                try {
                    const cs = getComputedStyle(el);
                    painted = !!(cs && cs.backgroundImage && cs.backgroundImage !== 'none');
                } catch (e) {}
                if (painted || sized(el, 12)) tiles += 1;
            }
        }
        if (tiles >= 4) return true;
        const body = document.body;
        const prompt = document.querySelector(
            '.prompt-text, .prompt, .header, [class*="prompt"], ' +
            '[class*="challenge-description"], [class*="instruction"]');
        const promptText = ((prompt && (prompt.innerText || prompt.textContent)) ||
            (body && body.innerText || '')).trim();
        const hasMenu = !!document.querySelector(
            '#menu-info, .display-menu-btn, [aria-label*="About hCaptcha"]');
        const hasAnswer = !!document.querySelector(
            'input[type="text"], textarea, [class*="answer"]');
        const hasVerify = !!document.querySelector(
            'button[type="submit"], .button-submit, [class*="submit"], ' +
            '[class*="verify"], .button-verify');
        if (hasMenu) return true;
        return promptText.length >= 8 && (tiles >= 1 || hasAnswer || hasVerify);
    }""")


async def _widget_has_checkbox(page, iframe) -> bool:
    try:
        frame = await (await iframe.element_handle(timeout=4000)).content_frame()
        if frame is None:
            return False
        return bool(await frame.evaluate(
            "() => !!document.querySelector("
            "'#checkbox, [role=\\\"checkbox\\\"], .checkbox, "
            "input[type=\\\"checkbox\\\"], [aria-checked], .button-submit')"))
    except Exception:
        return False


async def _widget_error_state(page, iframe) -> str:
    """hCaptcha's OWN error banner text ("Rate limited or network error.
    Please retry.") — when present the checkbox is INERT and no click will
    ever register. Returns "" when healthy."""
    frame = await _hcaptcha_frame_for(page, iframe)
    if frame is None:
        return ""
    try:
        text = await frame.evaluate(
            "() => (document.body ? document.body.innerText : '')")
    except Exception:
        return ""
    low = (text or "").lower()
    for kw in ("rate limited or network error", "rate limited",
               "network error", "please retry", "please try again",
               "automated queries"):
        if kw in low:
            return (text or "").strip()[:120]
    return ""


async def _click_hcaptcha_checkbox(page, iframe, log=_default_log) -> bool:
    """CLICK the 'Are you human' checkbox and only report success on
    hCaptcha's own confirmation signals (aria-checked=true flip, challenge
    iframe spawn, or a minted token). Strategies in order: frame_locator
    click, real mouse click, keyboard activation, JS el.click()."""
    frame = await _hcaptcha_frame_for(page, iframe)
    if frame is None:
        log("[Captcha] Checkbox click skipped — no live hCaptcha frame attached", "debug")
        return False

    full_src = ""
    try:
        full_src = await iframe.get_attribute("src") or ""
    except Exception:
        full_src = ""

    async def _confirm(attempt: str) -> bool:
        for _ in range(5):
            try:
                flipped = await frame.evaluate(
                    "() => { const el = document.querySelector('[aria-checked]');"
                    " return !!el && el.getAttribute('aria-checked') === 'true'; }")
                if flipped:
                    log(f"[Captcha] [OK] Checkbox {attempt} — aria-checked=true", "info")
                    return True
            except Exception:
                pass
            try:
                if await _challenge_iframe(page) is not None:
                    log(f"[Captcha] [OK] Checkbox {attempt} — challenge spawned", "info")
                    return True
            except Exception:
                pass
            try:
                if await read_hcaptcha_token(page):
                    log(f"[Captcha] [OK] Checkbox {attempt} — token present", "info")
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.4)
        return False

    # Real click point inside the frame (walk subtree/ancestors for a sized
    # element — the checkbox rect can report 0x0 while painted/interactive).
    point = None
    try:
        point = await frame.evaluate("""() => {
            const sels = ['[role="checkbox"]', '#checkbox', '.checkbox',
                          'input[type="checkbox"]', '[aria-checked]', '.button-submit'];
            let el = null;
            for (const s of sels) { el = document.querySelector(s); if (el) break; }
            if (!el) return null;
            const sized = (n) => {
                if (!n) return null;
                const r = n.getBoundingClientRect();
                return (r && r.width > 0 && r.height > 0)
                    ? {left: r.left, top: r.top, width: r.width, height: r.height} : null;
            };
            let rect = sized(el);
            if (!rect) {
                let best = null, bestArea = 0;
                const walk = (n) => {
                    const r = sized(n);
                    if (r) { const a = r.width * r.height; if (a > bestArea) { best = r; bestArea = a; } }
                    for (const c of n.children) walk(c);
                };
                for (const c of el.children) walk(c);
                if (best) rect = best;
            }
            if (!rect) {
                let p = el.parentElement;
                while (p) { const r = sized(p); if (r) { rect = r; break; } p = p.parentElement; }
            }
            if (!rect) return null;
            return {x: rect.left + rect.width / 2, y: rect.top + rect.height / 2,
                    w: rect.width, h: rect.height};
        }""")
    except Exception:
        point = None

    page_point = None
    try:
        iframe_box = await iframe.bounding_box()
    except Exception:
        iframe_box = None
    if point and iframe_box and iframe_box.get("width", 0) > 1:
        page_point = (iframe_box["x"] + point["x"], iframe_box["y"] + point["y"])
    elif iframe_box and iframe_box.get("width", 0) > 1:
        page_point = (iframe_box["x"] + iframe_box.get("width", 0) * 0.12,
                      iframe_box["y"] + iframe_box.get("height", 0) * 0.5)

    for attempt in range(1, 6):
        if attempt > 1:
            await asyncio.sleep(0.4)

        # Strategy 0: frame_locator click (engine's cross-origin mechanism).
        if full_src and "hcaptcha" in full_src:
            try:
                fl = page.frame_locator(f'iframe[src="{full_src}"]')
                fl_cb = fl.locator(
                    '#checkbox, [role="checkbox"], .checkbox, '
                    'input[type="checkbox"], [aria-checked], .button-submit')
                for ci in range(min(4, await fl_cb.count())):
                    try:
                        await fl_cb.nth(ci).click(timeout=3000)
                        if await _confirm(f"frame click #{ci} (attempt {attempt})"):
                            return True
                    except Exception:
                        continue
            except Exception:
                pass

        # Strategy 1: real mouse click at the computed page point.
        if page_point:
            try:
                cx, cy = page_point
                await page.mouse.move(cx, cy, steps=2)
                await asyncio.sleep(random.uniform(0.15, 0.35))
                await page.mouse.click(cx, cy)
                if await _confirm(f"mouse click (attempt {attempt})"):
                    return True
            except Exception:
                pass

        # Strategy 2: keyboard activation (role=checkbox is natively
        # activatable via Enter/Space; no coordinates involved).
        try:
            await frame.evaluate("""() => {
                const el = document.querySelector('[role="checkbox"], #checkbox, .checkbox, [aria-checked], .button-submit');
                if (el) el.focus();
            }""")
            await asyncio.sleep(0.1)
            await page.keyboard.press("Enter")
            if await _confirm(f"keyboard Enter (attempt {attempt})"):
                return True
            await page.keyboard.press("Space")
            if await _confirm(f"keyboard Space (attempt {attempt})"):
                return True
        except Exception:
            pass

        # Strategy 3: JS el.click().
        try:
            js_clicked = await frame.evaluate("""() => {
                const el = document.querySelector('[role="checkbox"], #checkbox, .checkbox, input[type="checkbox"], [aria-checked], .button-submit');
                if (!el) return false;
                el.click();
                return true;
            }""")
            if js_clicked and await _confirm(f"JS click (attempt {attempt})"):
                return True
        except Exception:
            pass

    try:
        html = await frame.evaluate(
            "() => (document.body ? document.body.outerHTML : '').slice(0, 2000)")
        log(f"[Captcha] Checkbox click never confirmed — widget frame DOM:\n{html}", "debug")
    except Exception:
        pass
    return False


# ── challenge solving (generic over any page) ───────────────────────────

async def _read_challenge_prompt(page, frame) -> str:
    try:
        raw = await frame.evaluate("""() => {
            const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
            const seen = new Set();
            const out = [];
            const cands = document.querySelectorAll(
                '.challenge-prompt, .prompt-text, #prompt-text, .task-description, ' +
                '#task-description, [class*="prompt" i], [class*="task-description" i], ' +
                'h1, h2, [class*="challenge" i] p, [class*="instructions" i]');
            for (const el of cands) {
                const t = norm(el.textContent);
                if (t.length < 8 || seen.has(t)) continue;
                seen.add(t);
                out.push({ text: t, len: t.length });
            }
            if (!out.length) return '';
            out.sort((a, b) => b.len - a.len);
            return out[0].text.slice(0, 200);
        }""")
        return str(raw or "").strip()
    except Exception:
        return ""


async def _screenshot_challenge_tiles(page, frame) -> list:
    """Screenshot every grid tile (reading order: top-left first)."""
    for sel in ('div.task-image, [class*="task-image" i]',
                '.task-grid img, [class*="task-grid" i] img, '
                '.challenge-content img'):
        try:
            loc = frame.locator(sel)
            n = await loc.count()
        except Exception:
            continue
        if n == 0:
            continue
        tiles = []
        for i in range(min(n, 12)):
            try:
                b = await loc.nth(i).screenshot(timeout=8000)
                if b:
                    tiles.append(b)
            except Exception:
                continue
        if tiles:
            return tiles
    return []


async def _click_challenge_tiles(page, frame, indices) -> bool:
    for sel in ('div.task-image, [class*="task-image" i]',
                '.task-grid img, [class*="task-grid" i] img, '
                '.challenge-content img'):
        try:
            n = await frame.locator(sel).count()
        except Exception:
            continue
        if n == 0:
            continue
        clicked = 0
        for idx in indices:
            if not (isinstance(idx, int) and 1 <= idx <= n):
                continue
            try:
                await frame.locator(sel).nth(idx - 1).click(timeout=5000)
                clicked += 1
            except Exception:
                continue
        if clicked:
            return True
    return False


_NEXT_VERIFY_JS = r"""() => {
    const norm = (s) => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
    const vis = (el) => {
        if (!el) return false;
        const r = el.getBoundingClientRect();
        if (!r || r.width < 8 || r.height < 8) return false;
        if (el.offsetParent === null && el.getClientRects().length === 0) return false;
        return true;
    };
    const disabled = (el) => !!(el.disabled
        || el.getAttribute('aria-disabled') === 'true'
        || /\bdisabled\b/i.test((el.className || '').toString()));
    const nextRe = /(^|[^a-z])(next|continue|weiter|volgende|continuer|continuar)([^a-z]|$)/;
    const verifyRe = /(^|[^a-z])(verify|check|submit|confirm|valider|verificar|bestätigen|bevestigen)([^a-z]|$)/;
    const cands = [];
    for (const el of document.querySelectorAll(
            '.button-submit, #button-submit, .button-arrow, [class*="button-arrow" i], '
            + '[class*="next" i], button[type="submit"], [role="button"], button, '
            + '[aria-label*="next" i], [aria-label*="verify" i]')) {
        if (!vis(el) || disabled(el)) continue;
        const t = norm((el.textContent || '') + ' '
            + (el.getAttribute('aria-label') || '') + ' '
            + (el.getAttribute('title') || ''));
        const cls = norm((el.className || '').toString());
        let kind = '';
        if (nextRe.test(t) || cls.includes('arrow') || /\bnext\b/.test(cls)) kind = 'next';
        else if (verifyRe.test(t) || cls.includes('submit') || cls.includes('verify')) kind = 'verify';
        else continue;
        const r = el.getBoundingClientRect();
        cands.push({ kind, x: r.left + r.width / 2, y: r.top + r.height / 2, t: t.slice(0, 40) });
    }
    return cands.find(c => c.kind === 'next') || cands[0] || null;
}"""


async def _click_challenge_verify(page, frame) -> bool:
    """Click Next or Verify so the following challenge can render."""
    pick = None
    for _ in range(5):
        try:
            pick = await frame.evaluate(_NEXT_VERIFY_JS)
        except Exception:
            pick = None
        if pick and pick.get("x"):
            break
        await asyncio.sleep(0.25)
    if pick and pick.get("x"):
        try:
            fbox = await (await frame.frame_element()).bounding_box()
            if fbox:
                await page.mouse.click(
                    fbox["x"] + float(pick["x"]),
                    fbox["y"] + float(pick["y"]))
                return True
        except Exception:
            pass
    try:
        await frame.locator('.button-submit, #button-submit').first.click(timeout=3000)
        return True
    except Exception:
        pass
    try:
        nxt = frame.locator('.button-arrow, [class*="button-arrow" i], '
                            '[class*="next" i], [aria-label*="next" i]')
        if await nxt.count() > 0:
            await nxt.first.click(timeout=3000)
            return True
    except Exception:
        pass
    return False


async def _type_challenge_answer(page, frame, text: str) -> bool:
    try:
        inp = frame.locator('input[type="text"], input:not([type]), textarea').first
        await inp.click(timeout=4000)
        await inp.fill(text, timeout=4000)
        return True
    except Exception:
        return False


async def _wait_for_widget_or_challenge(page, timeout: float = 60.0,
                                        log=_default_log) -> str:
    """Wait until an hCaptcha widget or a rendered challenge appears.

    Returns ``"challenge"``, ``"widget"``, or ``""`` on timeout. Also fails
    fast with ``"error"`` when the widget shows hCaptcha's own rate-limit /
    network-error banner (checkbox inert — no click can ever register).
    """
    deadline = time.time() + timeout
    widget_since = None
    while time.time() < deadline:
        # 1) Rendered challenge already showing → solve it directly.
        chall = await _challenge_iframe(page)
        if chall is not None:
            box = None
            try:
                box = await chall.bounding_box()
            except Exception:
                box = None
            if (box and box.get("height", 0) >= 80
                    and await _challenge_rendered(page, chall)):
                return "challenge"
        # 2) Widget present → widget-error fast-fail, else return "widget".
        widgets = await _widget_iframes(page)
        try:
            wcount = await widgets.count()
        except Exception:
            wcount = 0
        if wcount > 0:
            if widget_since is None:
                widget_since = time.time()
                log(f"[Captcha] hCaptcha widget present ({wcount} iframes)")
            for wi in range(wcount):
                err = await _widget_error_state(page, widgets.nth(wi))
                if err:
                    log(f"[Captcha] hCaptcha widget error: {err!r}", "warn")
                    return "error"
            return "widget"
        await asyncio.sleep(0.25)
    log(f"[Captcha] No hCaptcha widget/challenge in {int(timeout)}s", "warn")
    return ""


async def solve_hcaptcha(page, vision: Optional[HFVisionClient] = None,
                         log: Callable = _default_log,
                         timeout: float = 90.0,
                         max_solve_attempts: int = 3) -> dict:
    """Solve the hCaptcha challenge on any Playwright-compatible ``page``.

    Returns a dict:

      {"ok": True,  "token": "...", "prompt": "...", "answer": {...},
       "tiles": n,  "rounds": k}
      {"ok": False, "error": "<reason>", "prompt": "...", "answer": None, ...}

    ``token`` is hCaptcha's minted ``h-captcha-response`` value (write it to
    the page's ``h-captcha-response`` textarea / submit with it if the page
    needs it — the challenge's own Verify click already does that on most
    sites). ``answer`` is the vision model's structured answer
    ({"type": "tiles", "indices": [...]} or {"type": "text", "text": "..."}).
    """
    vision = vision or HFVisionClient(log=log)
    started = time.time()
    state = await _wait_for_widget_or_challenge(page, timeout=min(timeout, 60.0),
                                                log=log)
    if state == "error":
        return {"ok": False,
                "error": "hCaptcha widget error (rate limited / network error) — "
                         "rotate the IP/session and retry",
                "token": "", "prompt": "", "answer": None, "tiles": 0,
                "rounds": 0, "elapsed": round(time.time() - started, 1)}
    if state == "widget":
        log("[Captcha] [READY] hCaptcha widget present — clicking checkbox to spawn challenge")
        widgets = await _widget_iframes(page)
        clicked = False
        try:
            wcount = await widgets.count()
        except Exception:
            wcount = 0
        for wi in range(wcount):
            w = widgets.nth(wi)
            if not await _widget_has_checkbox(page, w):
                continue
            if await _click_hcaptcha_checkbox(page, w, log=log):
                clicked = True
                break
        if not clicked:
            log("[Captcha] Checkbox click never confirmed — challenge may already be up", "warn")
    elif state == "":
        return {"ok": False, "error": "no hCaptcha found on the page",
                "token": "", "prompt": "", "answer": None, "tiles": 0,
                "rounds": 0, "elapsed": round(time.time() - started, 1)}

    # Challenge must now be genuinely painted (never a loader shell).
    chall_deadline = time.time() + timeout
    chall = None
    while time.time() < chall_deadline:
        c = await _challenge_iframe(page)
        if c is not None:
            box = None
            try:
                box = await c.bounding_box()
            except Exception:
                box = None
            if box and box.get("height", 0) >= 80 and await _challenge_rendered(page, c):
                chall = c
                break
        await asyncio.sleep(0.5)
    if chall is None:
        return {"ok": False,
                "error": "challenge iframe never rendered (blank shell / blocked scripts)",
                "token": "", "prompt": "", "answer": None, "tiles": 0,
                "rounds": 0, "elapsed": round(time.time() - started, 1)}

    log("[Captcha] [READY] Image challenge rendered — reading prompt + tiles")

    sitekey = ""
    try:
        sitekey = await extract_hcaptcha_sitekey(page)
    except Exception:
        pass

    for solve_attempt in range(max_solve_attempts):
        if solve_attempt:
            await asyncio.sleep(3)
            log(f"[Captcha] Retrying vision solve (attempt {solve_attempt + 1}/{max_solve_attempts})...", "warn")
        rounds_done = 0
        last_prompt = ""
        for round_i in range(8):
            rounds_done = round_i + 1
            c = await _challenge_iframe(page)
            if c is None or not await _challenge_rendered(page, c):
                # After Next the iframe often drops to a loader — wait for
                # the next challenge instead of treating the dip as done.
                deadline = time.time() + 10.0
                c = None
                while time.time() < deadline:
                    cand = await _challenge_iframe(page)
                    if cand is not None and await _challenge_rendered(page, cand):
                        c = cand
                        break
                    token = await read_hcaptcha_token(page)
                    if token:
                        log("[Captcha] [OK] hCaptcha token minted by the solved challenge")
                        return {"ok": True, "token": token, "prompt": last_prompt,
                                "answer": None, "tiles": 0,
                                "rounds": rounds_done, "sitekey": sitekey,
                                "elapsed": round(time.time() - started, 1)}
                    await asyncio.sleep(0.5)
                if c is None:
                    break
            frame = await _hcaptcha_frame_for(page, c)
            if frame is None:
                await asyncio.sleep(1.5)
                continue
            prompt = await _read_challenge_prompt(page, frame)
            if not prompt:
                log("[Captcha] Prompt not readable yet (new round loading?)", "warn")
                await asyncio.sleep(2)
                continue
            if last_prompt and prompt == last_prompt:
                log("[Captcha] Same challenge still showing — clicking Next again")
                await _click_challenge_verify(page, frame)
                await asyncio.sleep(1.2)
                continue
            log(f"[Captcha] Challenge round {round_i + 1}: {prompt[:120]}")
            tiles = await _screenshot_challenge_tiles(page, frame)
            if not tiles:
                log("[Captcha] No grid tiles captured — retrying round", "warn")
                await asyncio.sleep(2)
                continue
            log(f"[Captcha] Asking {vision.model} which tiles match...")
            answer = await vision.solve(prompt, tiles)
            if not answer:
                log("[Captcha] Vision solver returned no answer — retrying", "warn")
                await asyncio.sleep(2)
                continue
            if answer.get("type") == "text":
                log(f"[Captcha] Text challenge: {answer.get('text')!r}")
                await _type_challenge_answer(page, frame, answer.get("text", ""))
            else:
                indices = [i for i in answer.get("indices", [])
                           if isinstance(i, int) and 1 <= i <= len(tiles)]
                log(f"[Captcha] Clicking tiles: {indices}")
                if indices and not await _click_challenge_tiles(page, frame, indices):
                    log("[Captcha] Tile clicks failed — retrying round", "warn")
                    await asyncio.sleep(1.5)
                    continue
            await asyncio.sleep(0.7)
            await _click_challenge_verify(page, frame)
            last_prompt = prompt
            # Wait for hCaptcha to accept (token minted) or present a new round.
            for _ in range(14):
                await asyncio.sleep(0.7)
                token = await read_hcaptcha_token(page)
                if token:
                    log("[Captcha] [OK] hCaptcha token minted by the solved challenge")
                    return {"ok": True, "token": token, "prompt": prompt,
                            "answer": answer, "tiles": len(tiles),
                            "rounds": rounds_done, "sitekey": sitekey,
                            "elapsed": round(time.time() - started, 1)}
                c2 = await _challenge_iframe(page)
                if c2 is None or not await _challenge_rendered(page, c2):
                    continue
                f2 = await _hcaptcha_frame_for(page, c2)
                if f2 is None:
                    continue
                p2 = await _read_challenge_prompt(page, f2)
                if p2 and p2 != last_prompt:
                    log(f"[Captcha] Next challenge ready: {p2[:80]}")
                    break
        log("[Captcha] Vision solve not accepted across rounds — retrying", "warn")

    return {"ok": False,
            "error": "vision solver could not clear the challenge "
                     "(check API_KEY / HF_MODEL, see --check)",
            "token": "", "prompt": "", "answer": None, "tiles": 0,
            "rounds": 0, "sitekey": sitekey,
            "elapsed": round(time.time() - started, 1)}


# ── CLI ─────────────────────────────────────────────────────────────────

async def _cli_check() -> int:
    client = HFVisionClient(log=lambda m, level="info": print(m, flush=True))
    ok, _models = await client.check()
    print(f"Hugging Face Inference API: {'REACHABLE' if ok else 'UNREACHABLE'} "
          f"({client.endpoint})")
    if not ok:
        print("Fix: export API_KEY=hf_... (a Hugging Face token with "
              "inference permission) and, if needed, HF_MODEL=<repo id>.")
        return 1
    print(f"Model {client.model}: AVAILABLE ✓")
    return 0


async def _cli_solve(url: str, headless: bool, timeout: float,
                     output: Optional[str], screenshot: Optional[str]) -> int:
    print(f"[solver] Engine: {ENGINE} | URL: {url} | headless={headless}")
    vision = HFVisionClient()
    ok, _models = await vision.check()
    if not ok:
        print("[solver] WARNING: Hugging Face endpoint unreachable — the "
              "solve will fail. Set API_KEY to a valid Hugging Face token "
              "(and HF_MODEL to a vision model repo id).", flush=True)

    pw = await async_playwright().start()
    # `.chromium` is the engine's Playwright-compatible shim — it really
    # launches a REAL Google Chrome (google-chrome-stable) through nodriver
    # (an undetected CDP driver; no Selenium layer). See nodriver_engine.py.
    browser = await pw.chromium.launch(headless=headless)
    context = await browser.new_context(viewport={"width": 1280, "height": 900},
                                        ignore_https_errors=True)
    page = await context.new_page()
    try:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            print(f"[solver] Navigation error: {e}", flush=True)
        await asyncio.sleep(2)
        result = await solve_hcaptcha(page, vision=vision, timeout=timeout)
        print(json.dumps(result, indent=2), flush=True)
        if screenshot:
            try:
                await page.screenshot(path=screenshot, full_page=True)
                print(f"[solver] Screenshot saved: {screenshot}", flush=True)
            except Exception as e:
                print(f"[solver] Screenshot failed: {e}", flush=True)
        if output and result.get("token"):
            try:
                with open(output, "w") as f:
                    f.write(result["token"])
                print(f"[solver] Token written to {output}", flush=True)
            except Exception as e:
                print(f"[solver] Could not write {output}: {e}", flush=True)
        return 0 if result.get("ok") else 1
    finally:
        try:
            await browser.close()
        except Exception:
            pass
        try:
            await pw.stop()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="solver.py",
        description="Standalone hCaptcha solver (Hugging Face vision model + "
                    "nodriver/Chrome engine). Solves the image challenge on any page.")
    parser.add_argument("url", nargs="?", help="page URL with an hCaptcha widget")
    parser.add_argument("--check", action="store_true",
                        help="probe the Hugging Face Inference API and exit")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="run the browser headless (default)")
    parser.add_argument("--headed", action="store_true",
                        help="show the browser window")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="max seconds to wait for + solve the challenge")
    parser.add_argument("--output", default=None,
                        help="file to write the minted token into")
    parser.add_argument("--screenshot", default=None,
                        help="file to save a final page screenshot into")
    args = parser.parse_args()

    if args.check:
        return asyncio.run(_cli_check())
    if not args.url:
        parser.error("a URL is required (or use --check)")
    headless = not args.headed
    return asyncio.run(_cli_solve(args.url, headless, args.timeout,
                                  args.output, args.screenshot))


if __name__ == "__main__":
    sys.exit(main())

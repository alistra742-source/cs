"""
live_control.py — live-control helpers for the dashboard's LIVE tab.

These are plain module functions (not methods on DiscordAutomation) so the
dashboard can run them on the bot's asyncio loop without touching the big bot
modules. They drive the SAME ``bot._page`` the bot uses — the operator and the
bot share one real Camoufox session, and every action here is a pure read or
input over that page (no second browser is ever launched).
"""
import asyncio
import base64
import time

from browser_engine import ENGINE
from server import NAV_TIMEOUT_MS

VIEWPORT_W = 1920
VIEWPORT_H = 1080


async def live_meta(bot) -> dict:
    """Cheap metadata read (no screenshot) for the LIVE tab."""
    page = getattr(bot, "_page", None)
    connected = False
    url = ""
    title = ""
    if page is not None:
        try:
            href = await asyncio.wait_for(
                page.evaluate("location.href"), timeout=3.0)
            url = str(href or "") or (page.url or "")
            connected = True
        except Exception:
            try:
                url = page.url or ""
            except Exception:
                url = ""
        try:
            title = str(
                await asyncio.wait_for(page.title(), timeout=3.0) or "")
        except Exception:
            title = ""
    try:
        dsf = float(getattr(bot, "_fingerprint", {}).get("pixel_ratio", 1.0) or 1.0)
    except Exception:
        dsf = 1.0
    return {
        "connected": connected,
        "url": url,
        "title": title,
        "viewport_width": VIEWPORT_W,
        "viewport_height": VIEWPORT_H,
        "device_scale_factor": dsf,
        "browser": ENGINE,
        "worker_id": getattr(bot, "worker_id", ""),
    }


async def live_screenshot(bot) -> str:
    """Viewport-sized PNG -> base64 for the live feed. Retries once and logs
    the EXACT failure so the dashboard's ALL LOGS shows why the frame is
    missing instead of silently sitting on 'waiting for frame'."""
    page = getattr(bot, "_page", None)
    if page is None:
        return ""
    last_err = ""
    try:
        shot = await asyncio.wait_for(
            page.screenshot(full_page=False), timeout=6)
        if not shot:
            last_err = "empty capture"
    except Exception as e:
        last_err = str(e)
    if not last_err:
        b64 = base64.b64encode(shot).decode("utf-8")
        shots = getattr(bot, "_screenshots", None)
        if shots is not None:
            shots.append(b64)
            if len(shots) > 100:
                bot._screenshots = shots[-50:]
        return b64
    try:
        bot._log(f"[Live] screenshot failed: {last_err}", level="warn")
    except Exception:
        pass
    return ""


async def get_live_state(bot) -> dict:
    meta = await live_meta(bot)
    shot = ""
    if meta["connected"]:
        shot = await live_screenshot(bot)
        if not shot:
            # Never flash a black screen just because one capture failed —
            # keep the last good frame the bot already has.
            try:
                shot = bot.get_latest_screenshot() or ""
            except Exception:
                shot = ""
    meta["screenshot"] = shot
    return meta


def _dead_page(url: str) -> bool:
    """True when the page is a browser error/blank page (proxy tunnel died,
    site unreachable, or navigation never happened) rather than real content."""
    u = (url or "").lower()
    if "chrome-error" in u or "err_tunnel" in u or "err_" in u:
        return True
    if u in ("", "about:blank") or u.startswith("about:"):
        return True
    return False


async def _wait_for_content(page, timeout: float = 6.0) -> bool:
    """True once the page has actually painted non-empty text. Discord is a
    SPA, so readyState can be 'complete' while React hasn't rendered yet —
    give it a short grace window before declaring the page blank."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            has = await asyncio.wait_for(page.evaluate(
                "() => { const b = document.body; "
                "return !!(b && (b.innerText || '').trim().length); }"),
                timeout=2.0)
            if has:
                return True
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return False


async def live_navigate(bot, url: str) -> dict:
    page = getattr(bot, "_page", None)
    if page is None:
        meta = await live_meta(bot)
        meta["error"] = "browser not started"
        return meta
    try:
        await page.goto(url, wait_until="domcontentloaded",
                        timeout=NAV_TIMEOUT_MS)
    except Exception as e:
        meta = await live_meta(bot)
        meta["error"] = f"navigation failed: {e}"
        return meta
    meta = await get_live_state(bot)
    # goto() can 'succeed' straight onto a chrome-error page when the proxy
    # CONNECT tunnel is dead — treat that as a navigation failure so the
    # caller can rotate the session.
    if _dead_page(meta.get("url", "")):
        meta["error"] = "site unreachable (proxy tunnel failed)"
        return meta
    # A white screen can also mean navigation never happened (parked on
    # about:blank) or the SPA failed to paint — catch that too.
    if not await _wait_for_content(page):
        meta["error"] = "page rendered blank (no content)"
    return meta


async def _live_click(page, x: float, y: float) -> None:
    x, y = float(x), float(y)
    await page.mouse.move(x, y, steps=4)
    await page.mouse.click(x, y)


async def _live_key(page, key: str) -> None:
    key = str(key or "")
    if not key:
        return
    specials = {"Enter", "Backspace", "Tab", "Escape", "Delete",
                "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                "Home", "End", "PageUp", "PageDown", "Space"}
    if key == " ":
        await page.keyboard.press("Space")
    elif key in specials or len(key) > 1:
        await page.keyboard.press(key)
    else:
        # Printable character: insert as a real text input event so
        # React-controlled fields (Discord included) pick it up.
        await page.keyboard.type(key, delay=0)


async def live_action(bot, action: dict) -> dict:
    page = getattr(bot, "_page", None)
    if page is None:
        meta = await live_meta(bot)
        meta["error"] = "browser not started"
        meta["screenshot"] = ""
        return meta
    action = action or {}
    kind = str(action.get("action", ""))
    err = None
    try:
        if kind == "back":
            await page.evaluate("window.history.back()")
        elif kind == "forward":
            await page.evaluate("window.history.forward()")
        elif kind == "reload":
            await page.reload(timeout=NAV_TIMEOUT_MS)
        elif kind == "click":
            await _live_click(
                page, float(action.get("x", 0)), float(action.get("y", 0)))
        elif kind == "scroll":
            dy = float(action.get("delta_y", action.get("deltaY", 0)) or 0)
            await page.evaluate(f"window.scrollBy(0, {dy})")
        elif kind == "key":
            await _live_key(page, action.get("key", ""))
        elif kind == "type":
            await page.keyboard.type(str(action.get("text", "")), delay=0)
        else:
            err = f"unknown action: {kind}"
    except Exception as e:
        err = f"action {kind} failed: {e}"
    # Visual actions refresh the frame immediately (with a screenshot) so the
    # operator SEES the result of the click; bare-meta responses were blanking
    # the feed to "waiting for frame". Input actions (key/scroll/type) stay
    # fast — the next 1.4s poll refreshes the frame for those.
    if kind in ("click", "back", "forward", "reload"):
        st = await get_live_state(bot)
    else:
        st = await live_meta(bot)
    if err:
        st["error"] = err
    return st

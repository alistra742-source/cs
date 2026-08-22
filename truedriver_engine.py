"""
truedriver_engine.py — Playwright-compatible async facade backed by truedriver
(https://pypi.org/project/truedriver — a blazing fast, async-first,
undetectable web automation framework built directly on Chrome DevTools
Protocol, no Selenium/WebDriver layer to flag).

The rest of the application imports ``async_playwright`` and ``ENGINE`` from
``browser_engine`` and drives everything through Playwright's async API. This
module re-implements the *subset* of that API the code actually uses on top of
truedriver's CDP primitives, so every caller (server.py workers,
captcha_solver.py, drag_solver.py, live_control.py) is unchanged:

    pw  = await async_playwright().start()
    b   = await pw.chromium.launch(headless=..., proxy={...})
    ctx = await b.new_context(**opts)
    page = await ctx.new_page()

Mapping notes:

  · ``page.evaluate(js)`` accepts both bare JS expressions (``location.href``)
    and arrow/function bodies (``() => {...}``); function bodies are wrapped
    and invoked so they actually run (matching Playwright's semantics).
  · ``page.locator(sel)`` -> ``_Locator`` supports the chained surface used in
    the code: ``.count()``, ``.first``, ``.nth(i)``, ``.filter(visible=True)``,
    ``.click()``, ``.fill()``, ``.press()``, ``.inner_text()``,
    ``.input_value()``, ``.is_visible()``, ``.bounding_box()``,
    ``.screenshot()``, ``.content_frame()``, ``.element_handle()``.
  · Frames are first-class objects (``page.frames``, ``frame.locator``,
    ``frame.frame_element()``, ``frame.screenshot()``) via CDP frame switching.
  · ``page.on("request"/"response"/"crash")`` maps to CDP Network / Inspector
    events, so the rqdata + payload capture and crash flag keep working.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import time
from typing import Any, Callable, List, Optional

ENGINE = "truedriver"
CHANNEL = None

try:  # pragma: no cover - raised as a clear error at launch
    from truedriver import cdp
    from truedriver.core.config import Config
    from truedriver.core.element import Position
    from truedriver.core.util import start as _td_start
except Exception:  # pragma: no cover
    cdp = None
    Config = None
    Position = None
    _td_start = None


__all__ = ["async_playwright", "ENGINE", "CHANNEL"]


# ─────────────────────────────────────────────────────────────────────────────
# JS expression helpers
# ─────────────────────────────────────────────────────────────────────────────
_ARROW_RE = re.compile(r"^\s*(async\s+)?(\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")


def _wrap_js(js: Any, arg: Any = None) -> tuple:
    """Normalise a JS string for Runtime.evaluate.

    Playwright evaluates bare expressions directly but runs arrow/function
    bodies, and passes `arg` into the invoked function when supplied. The code
    mixes all of these, so detect function-like strings and invoke them —
    forwarding the argument (e.g. ``([sel, value]) => {...}`` called with a
    two-element list). Returns (expression, await_promise).
    """
    s = (js if isinstance(js, str) else "").strip()
    if not s:
        return "undefined", False
    if _ARROW_RE.match(s) or s.startswith("function") or s.startswith("async function"):
        if arg is None:
            expr = f"({s})()"
        else:
            expr = f"({s})({json.dumps(arg)})"
        return expr, s.startswith("async")
    return s, False


def _wrap_new_document(src: str) -> str:
    """Wrap an init-script string so it runs at document start.

    ``Page.addScriptToEvaluateOnNewDocument`` wants a *statement*; arrow
    function bodies like ``() => {...}`` would otherwise just define (and
    drop) the function. Comments (the stealth placeholder) pass through.
    """
    s = (src or "").strip()
    if not s or s.startswith("//") or s.startswith("/*"):
        return s
    if _ARROW_RE.match(s) or s.startswith("function") or s.startswith("async function"):
        return f"({s})();"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Mouse
# ─────────────────────────────────────────────────────────────────────────────
class _Mouse:
    """Playwright page.mouse over CDP Input.dispatchMouseEvent. Tracks the
    current cursor position so down()/up() (drag gestures) land where the
    pointer already is, like Playwright."""

    def __init__(self, page: "_Page"):
        self._page = page
        self._tab = page._tab
        self._x = 0.0
        self._y = 0.0

    async def move(self, x: float, y: float, steps: Optional[int] = None) -> None:
        x, y = float(x), float(y)
        none = cdp.input_.MouseButton("none")
        n = int(steps) if steps and steps > 1 else 1
        if n > 1:
            for i in range(1, n + 1):
                await self._tab.send(cdp.input_.dispatch_mouse_event(
                    "mouseMoved", x=x * i / n, y=y * i / n, button=none, buttons=0))
        else:
            await self._tab.send(cdp.input_.dispatch_mouse_event(
                "mouseMoved", x=x, y=y, button=none, buttons=0))
        self._x, self._y = x, y

    async def click(self, x: float, y: float, button: str = "left", **kwargs) -> None:
        await self.move(x, y, steps=1)
        b = cdp.input_.MouseButton(button if button in ("left", "right", "middle") else "left")
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mousePressed", x=self._x, y=self._y, button=b, buttons=1, click_count=1))
        await asyncio.sleep(random.uniform(0.03, 0.09))
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mouseReleased", x=self._x, y=self._y, button=b, buttons=0, click_count=1))

    async def dblclick(self, x: float, y: float, button: str = "left", **kwargs) -> None:
        await self.move(x, y, steps=1)
        b = cdp.input_.MouseButton(button if button in ("left", "right", "middle") else "left")
        for _ in range(2):
            await self._tab.send(cdp.input_.dispatch_mouse_event(
                "mousePressed", x=self._x, y=self._y, button=b, buttons=1, click_count=2))
            await asyncio.sleep(random.uniform(0.02, 0.06))
            await self._tab.send(cdp.input_.dispatch_mouse_event(
                "mouseReleased", x=self._x, y=self._y, button=b, buttons=0, click_count=2))

    async def down(self, button: str = "left", **kwargs) -> None:
        b = cdp.input_.MouseButton(button if button in ("left", "right", "middle") else "left")
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mousePressed", x=self._x, y=self._y, button=b, buttons=1, click_count=1))

    async def up(self, button: str = "left", **kwargs) -> None:
        b = cdp.input_.MouseButton(button if button in ("left", "right", "middle") else "left")
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mouseReleased", x=self._x, y=self._y, button=b, buttons=0, click_count=1))

    async def wheel(self, delta_x: float = 0, delta_y: float = 0) -> None:
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mouseWheel", x=self._x, y=self._y,
            delta_x=float(delta_x), delta_y=float(delta_y)))


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard
# ─────────────────────────────────────────────────────────────────────────────
_MODIFIER_BITS = {
    "Alt": 1, "AltGr": 1, "Control": 2, "ControlLeft": 2, "ControlRight": 2,
    "Meta": 4, "MetaLeft": 4, "MetaRight": 4, "Shift": 8, "ShiftLeft": 8,
    "ShiftRight": 8,
}
_SPECIAL_KEYS = {
    "Enter": (13, "Enter"), "Tab": (9, "Tab"), "Backspace": (8, "Backspace"),
    "Escape": (27, "Escape"), "Delete": (46, "Delete"), "Home": (36, "Home"),
    "End": (35, "End"), "PageUp": (33, "PageUp"), "PageDown": (34, "PageDown"),
    "ArrowUp": (38, "ArrowUp"), "ArrowDown": (40, "ArrowDown"),
    "ArrowLeft": (37, "ArrowLeft"), "ArrowRight": (39, "ArrowRight"),
    "Space": (32, " "), " ": (32, " "), "Control": (17, "Control"),
    "Shift": (16, "Shift"), "Alt": (18, "Alt"), "Meta": (91, "Meta"),
}
_SHIFTED_CHARS = set("~!@#$%^&*()_+{}|:\"<>?")


def _code_for(name: str) -> str:
    if len(name) == 1 and name.isalpha():
        return "Key" + name.upper()
    if len(name) == 1 and name.isdigit():
        return "Digit" + name
    return name


def _key_parts(key: str):
    mods = 0
    main = key
    if "+" in key:
        parts = key.split("+")
        main = parts[-1].strip()
        for m in parts[:-1]:
            mods |= _MODIFIER_BITS.get(m.strip(), 0)
    return main, mods


async def _dispatch_press(tab, key: str) -> None:
    """Dispatch a single key press (down, optional char, up) with modifiers.

    Chromium inserts a character when EITHER keyDown carries ``text`` OR a
    separate ``char`` event is sent. Emitting both doubles every letter
    (``glass`` -> ``ggllaassss``). Non-Shift modifiers (Ctrl/Alt/Meta) must
    never produce a char at all — otherwise ``Control+a`` types an ``a``
    instead of selecting, the field never clears, and every refill appends
    on top of the previous value (the mangled
    ``shshsh@gggggglaaaaaassssswhiteeee`` form-fill bug).
    """
    main, mods = _key_parts(key)
    if main in _SPECIAL_KEYS:
        vk, name = _SPECIAL_KEYS[main]
        text = None
    elif len(main) == 1:
        ch = main
        if ch.isalpha():
            vk, name = ord(ch.upper()), ch
        elif ch.isdigit():
            vk, name = ord(ch), ch
        else:
            vk, name = ord(ch), ch
        if ch.isupper() or ch in _SHIFTED_CHARS:
            mods |= 8
        # Only emit an insertable char when no non-Shift modifier is held.
        # Control/Alt/Meta shortcuts (Ctrl+A, Ctrl+C, ...) must not type.
        non_shift = mods & ~8
        text = ch if (ch.isprintable() and not non_shift) else None
    else:
        vk, name = ord(main[0]) if main else 0, main
        text = None

    code = _code_for(name)
    # keyDown/keyUp must NOT carry text when a char event follows — Chrome
    # would insert the character twice (once per event).
    down = dict(type_="keyDown", key=name, code=code, windows_virtual_key_code=vk,
                native_virtual_key_code=vk, modifiers=mods)
    await tab.send(cdp.input_.dispatch_key_event(**down))
    if text:
        await tab.send(cdp.input_.dispatch_key_event(
            type_="char", key=name, code=code, windows_virtual_key_code=vk,
            modifiers=mods, text=text, unmodified_text=text))
    up = dict(type_="keyUp", key=name, code=code, windows_virtual_key_code=vk,
              native_virtual_key_code=vk, modifiers=mods)
    await tab.send(cdp.input_.dispatch_key_event(**up))


class _Keyboard:
    """Playwright page.keyboard over CDP Input.dispatchKeyEvent."""

    def __init__(self, page: "_Page"):
        self._page = page
        self._tab = page._tab

    async def press(self, key: str, **kwargs) -> None:
        await _dispatch_press(self._tab, str(key))

    async def type(self, text: str, delay: float = 0, **kwargs) -> None:
        for ch in str(text):
            if ch == "\n":
                await _dispatch_press(self._tab, "Enter")
            elif ch == "\t":
                await _dispatch_press(self._tab, "Tab")
            else:
                await _dispatch_press(self._tab, ch)
            if delay:
                await asyncio.sleep(float(delay))

    async def insert_text(self, text: str, **kwargs) -> None:
        await self._tab.send(cdp.input_.insert_text(str(text)))

    async def down(self, key: str, **kwargs) -> None:
        main, mods = _key_parts(str(key))
        if main in _SPECIAL_KEYS:
            vk, name = _SPECIAL_KEYS[main]
        else:
            vk, name = ord(main[0]) if main else 0, main
        await self._tab.send(cdp.input_.dispatch_key_event(
            type_="keyDown", key=name, code=_code_for(name),
            windows_virtual_key_code=vk, native_virtual_key_code=vk, modifiers=mods))

    async def up(self, key: str, **kwargs) -> None:
        main, mods = _key_parts(str(key))
        if main in _SPECIAL_KEYS:
            vk, name = _SPECIAL_KEYS[main]
        else:
            vk, name = ord(main[0]) if main else 0, main
        await self._tab.send(cdp.input_.dispatch_key_event(
            type_="keyUp", key=name, code=_code_for(name),
            windows_virtual_key_code=vk, native_virtual_key_code=vk, modifiers=mods))


# ─────────────────────────────────────────────────────────────────────────────
# Locator
# ─────────────────────────────────────────────────────────────────────────────
class _Locator:
    """Playwright locator re-implemented over truedriver Element resolution.

    A locator is a selector (optionally scoped to a frame or to frames reached
    through an iframe selector), an optional index, and optional visibility
    filtering. Resolution happens lazily on each operation.
    """

    def __init__(self, page: "_Page", selector: str, *, frame: Any = None,
                 frame_selector: Optional[str] = None, index: Optional[int] = None,
                 visible_only: bool = False):
        self._page = page
        self._tab = page._tab
        self._selector = selector
        self._frame = frame
        self._frame_selector = frame_selector
        self._index = index
        self._visible_only = visible_only

    # ---- chaining ---------------------------------------------------------
    @property
    def first(self) -> "_Locator":
        return _Locator(self._page, self._selector, frame=self._frame,
                        frame_selector=self._frame_selector, index=0,
                        visible_only=self._visible_only)

    def nth(self, index: int) -> "_Locator":
        return _Locator(self._page, self._selector, frame=self._frame,
                        frame_selector=self._frame_selector, index=int(index),
                        visible_only=self._visible_only)

    def filter(self, visible: bool = False, **kwargs) -> "_Locator":
        return _Locator(self._page, self._selector, frame=self._frame,
                        frame_selector=self._frame_selector, index=self._index,
                        visible_only=visible)

    def locator(self, selector: str) -> "_Locator":
        # descendant locator (rarely used; concat by '>>' is not supported by
        # querySelector, so resolve parent then child under it)
        return _Locator(self._page, selector, frame=self._frame,
                        frame_selector=self._frame_selector)

    async def all(self) -> List["_Locator"]:
        els = await self._resolve_elements()
        return [_Locator(self._page, self._selector, frame=self._frame,
                         frame_selector=self._frame_selector, index=i,
                         visible_only=self._visible_only) for i in range(len(els))]

    # ---- resolution -------------------------------------------------------
    async def _collect(self) -> list:
        if self._frame_selector:
            frames = await self._page._frames_for_selector(self._frame_selector)
            out: list = []
            for fr in frames:
                try:
                    await self._tab.switch_to_frame(fr)
                    out.extend(await self._tab.query_selector_all(self._selector))
                except Exception:
                    pass
            await self._tab.switch_to_main_frame()
            return out
        if self._frame is not None:
            await self._tab.switch_to_frame(self._frame)
            try:
                return await self._tab.query_selector_all(self._selector)
            finally:
                await self._tab.switch_to_main_frame()
        await self._tab.switch_to_main_frame()
        return await self._tab.query_selector_all(self._selector)

    async def _resolve_elements(self) -> list:
        els = await self._collect()
        if self._visible_only and els:
            out = []
            for e in els:
                try:
                    if (await e.get_position()) is not None:
                        out.append(e)
                except Exception:
                    pass
            els = out
        return els

    def _idx(self) -> int:
        return 0 if self._index is None else self._index

    async def _pick(self):
        els = await self._resolve_elements()
        if not els:
            return None
        i = self._idx()
        if i >= len(els):
            return None
        return els[i]

    async def _wait_element(self, timeout: Optional[float] = None, retries: int = 200):
        timeout = (timeout or 30000) / 1000.0
        deadline = time.monotonic() + timeout
        while True:
            el = await self._pick()
            if el is not None:
                return el
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.2)

    # ---- operations -------------------------------------------------------
    async def count(self) -> int:
        return len(await self._resolve_elements())

    async def is_visible(self) -> bool:
        el = await self._pick()
        if el is None:
            return False
        try:
            return (await el.get_position()) is not None
        except Exception:
            return False

    async def inner_text(self) -> str:
        el = await self._pick()
        if el is None:
            return ""
        try:
            return str(await el.apply("(e) => e.innerText") or "")
        except Exception:
            return ""

    async def input_value(self) -> str:
        el = await self._pick()
        if el is None:
            return ""
        try:
            return str(await el.apply("(e) => e.value") or "")
        except Exception:
            return ""

    async def get_attribute(self, name: str) -> Optional[str]:
        el = await self._pick()
        if el is None:
            return None
        try:
            return el.get(name)
        except Exception:
            return None

    async def focus(self, timeout: Optional[float] = None, **kwargs) -> None:
        el = await self._pick()
        if el is not None:
            try:
                await el.focus()
            except Exception:
                pass

    async def is_disabled(self) -> bool:
        el = await self._pick()
        if el is None:
            return True
        try:
            return bool(await el.apply(
                "(e) => e.disabled || e.getAttribute('aria-disabled') === 'true'"))
        except Exception:
            return True

    async def scroll_into_view_if_needed(self, timeout: Optional[float] = None,
                                         **kwargs) -> None:
        el = await self._pick()
        if el is not None:
            try:
                await el.scroll_into_view()
            except Exception:
                pass

    async def dispatch_event(self, event: str, timeout: Optional[float] = None,
                             **kwargs) -> None:
        el = await self._pick()
        if el is None:
            raise TimeoutError(f"locator dispatch_event: not found for '{self._selector}'")
        try:
            await el.apply(
                f"(e) => e.dispatchEvent(new Event('{event}', {{bubbles: true, cancelable: true}}))")
        except Exception:
            pass

    async def select_option(self, index: Optional[int] = None, value: Optional[str] = None,
                            label: Optional[str] = None, timeout: Optional[float] = None,
                            **kwargs) -> None:
        el = await self._pick()
        if el is None:
            raise TimeoutError(f"locator select_option: not found for '{self._selector}'")
        js = "(e) => { let idx = -1; "
        if index is not None:
            js += f"idx = {int(index)}; "
        elif value is not None:
            js += ("for (let i=0;i<(e.options||[]).length;i++)"
                   f"{{ if (String(e.options[i].value) === {json.dumps(str(value))}) {{ idx = i; break; }} }} ")
        elif label is not None:
            js += ("for (let i=0;i<(e.options||[]).length;i++)"
                   f"{{ if (String(e.options[i].text) === {json.dumps(str(label))}) {{ idx = i; break; }} }} ")
        js += ("if (idx >= 0) { e.selectedIndex = idx; "
               "e.dispatchEvent(new Event('input', {bubbles: true})); "
               "e.dispatchEvent(new Event('change', {bubbles: true})); } }")
        await el.apply(js)

    async def bounding_box(self):
        el = await self._pick()
        if el is None:
            return None
        try:
            pos = await el.get_position()
        except Exception:
            return None
        if not pos:
            return None
        return {"x": pos.left, "y": pos.top, "width": pos.width, "height": pos.height}

    async def click(self, timeout: Optional[float] = None, force: bool = False, **kwargs) -> None:
        el = await self._wait_element(timeout)
        if el is None:
            raise TimeoutError(f"locator click: element not found for '{self._selector}'")
        pos = await el.get_position()
        if not pos:
            try:
                await el.scroll_into_view()
            except Exception:
                pass
            await asyncio.sleep(0.15)
            pos = await el.get_position()
        if not pos:
            raise TimeoutError(f"locator click: element not visible for '{self._selector}'")
        x, y = pos.center
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mousePressed", x=float(x), y=float(y),
            button=cdp.input_.MouseButton("left"), buttons=1, click_count=1))
        await asyncio.sleep(random.uniform(0.04, 0.09))
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mouseReleased", x=float(x), y=float(y),
            button=cdp.input_.MouseButton("left"), buttons=0, click_count=1))

    async def fill(self, text: str, timeout: Optional[float] = None, **kwargs) -> None:
        """Replace the whole input value (Playwright fill semantics).

        Never appends. Clears via the native value setter first so a broken
        Ctrl+A path cannot leave old characters in the field, then inserts
        the new text with a single CDP insertText (trusted input events React
        accepts). Falls back to per-char key events only if insertText fails.
        """
        el = await self._wait_element(timeout)
        if el is None:
            raise TimeoutError(f"locator fill: element not found for '{self._selector}'")
        try:
            await el.focus()
        except Exception:
            pass
        # Hard-clear the current value on the element itself. Focus + Ctrl+A
        # alone is not enough: if select-all fails (or used to type an "a"),
        # subsequent keystrokes APPEND and produce mangled concatenations
        # like "user@domain.comuser@domain.com" / doubled letters.
        try:
            await el.apply(
                "(e) => {"
                "  try {"
                "    const setter = Object.getOwnPropertyDescriptor("
                "      window.HTMLInputElement.prototype, 'value').set"
                "      || Object.getOwnPropertyDescriptor("
                "      window.HTMLTextAreaElement.prototype, 'value').set;"
                "    if (setter) setter.call(e, '');"
                "    else e.value = '';"
                "    try { const t = e._valueTracker; if (t && t.setValue) t.setValue(''); } catch (err) {}"
                "    e.dispatchEvent(new Event('input', { bubbles: true }));"
                "  } catch (err) { try { e.value = ''; } catch (e2) {} }"
                "}"
            )
        except Exception:
            # Last-resort clear via keyboard (now correctly non-typing Ctrl+A).
            try:
                await _dispatch_press(self._tab, "Control+a")
                await _dispatch_press(self._tab, "Backspace")
            except Exception:
                pass
        text = str(text)
        if not text:
            return
        # Single trusted insert — no per-char doubling risk.
        try:
            await self._tab.send(cdp.input_.insert_text(text))
            return
        except Exception:
            pass
        for ch in text:
            if ch == "\n":
                await _dispatch_press(self._tab, "Enter")
            else:
                await _dispatch_press(self._tab, ch)

    async def press(self, key: str, timeout: Optional[float] = None, **kwargs) -> None:
        el = await self._pick()
        if el is not None:
            try:
                await el.focus()
            except Exception:
                pass
        await _dispatch_press(self._tab, str(key))

    async def screenshot(self, timeout: Optional[float] = None, **kwargs) -> bytes:
        el = await self._wait_element(timeout)
        if el is None:
            return b""
        try:
            b64 = await el.screenshot_b64("png")
            return base64.b64decode(b64)
        except Exception:
            return b""

    async def content_frame(self) -> Optional["_Frame"]:
        el = await self._pick()
        if el is None:
            return None
        return await self._page._frame_for_element(el)

    async def element_handle(self, timeout: Optional[float] = None):
        el = await self._wait_element(timeout)
        if el is None:
            raise TimeoutError(f"locator element_handle: not found for '{self._selector}'")
        return _ElementHandle(self._page, el)


class _ElementHandle:
    """Minimal Playwright ElementHandle needed by the code (content_frame)."""

    def __init__(self, page: "_Page", el: Any):
        self._page = page
        self._el = el

    def __await__(self):
        async def _self():
            return self
        return _self().__await__()

    async def content_frame(self) -> Optional["_Frame"]:
        return await self._page._frame_for_element(self._el)


class _FrameLocator:
    """Playwright frame_locator — resolves to the frame(s) behind an iframe
    selector, then scopes further selectors inside those frames."""

    def __init__(self, page: "_Page", frame_selector: str):
        self._page = page
        self._tab = page._tab
        self._frame_selector = frame_selector

    def locator(self, selector: str) -> "_Locator":
        return _Locator(self._page, selector, frame_selector=self._frame_selector)


class _OwnerLocator:
    """Locator for the <iframe> element that owns a given frame, resolved via
    DOM.getFrameOwner."""

    def __init__(self, page: "_Page", frame: Any):
        self._page = page
        self._tab = page._tab
        self._frame = frame

    def __await__(self):
        # Playwright Locator/FrameLocator objects are awaitable (they resolve
        # to themselves); callers use both `frame.frame_element()` sync and
        # `await frame.frame_element()`.
        async def _self():
            return self
        return _self().__await__()

    async def bounding_box(self):
        try:
            owner, _ = await self._tab.send(cdp.dom.get_frame_owner(self._frame.id_))
            ro = await self._tab.send(cdp.dom.resolve_node(backend_node_id=owner))
            quads = await self._tab.send(cdp.dom.get_content_quads(object_id=ro.object_id))
            if not quads:
                return None
            p = Position(quads[0])
            return {"x": p.left, "y": p.top, "width": p.width, "height": p.height}
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Frame
# ─────────────────────────────────────────────────────────────────────────────
class _Frame:
    def __init__(self, page: "_Page", frame: Any):
        self._page = page
        self._tab = page._tab
        self._frame = frame

    @property
    def url(self) -> str:
        try:
            return self._frame.url or ""
        except Exception:
            return ""

    @property
    def name(self) -> str:
        try:
            return self._frame.name or ""
        except Exception:
            return ""

    def locator(self, selector: str) -> "_Locator":
        return _Locator(self._page, selector, frame=self._frame)

    async def evaluate(self, js: Any, arg: Any = None) -> Any:
        await self._tab.switch_to_frame(self._frame)
        try:
            return await self._page._eval(js, arg)
        finally:
            await self._tab.switch_to_main_frame()

    def frame_element(self) -> "_OwnerLocator":
        return _OwnerLocator(self._page, self._frame)

    async def screenshot(self, timeout: Optional[float] = None, **kwargs) -> bytes:
        fe = _OwnerLocator(self._page, self._frame)
        box = await fe.bounding_box()
        if not box:
            return b""
        try:
            clip = cdp.page.Viewport(x=box["x"], y=box["y"],
                                     width=box["width"], height=box["height"])
            data = await self._tab.send(cdp.page.capture_screenshot(
                format_="png", capture_beyond_viewport=True, clip=clip))
            return base64.b64decode(data)
        except Exception:
            return b""


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response facades for page.on()
# ─────────────────────────────────────────────────────────────────────────────
class _Request:
    def __init__(self, event: Any):
        self.url = (getattr(getattr(event, "request", None), "url", None)) or ""
        self._post_data = getattr(getattr(event, "request", None), "post_data", None)
        self._post_data_buffer = None

    @property
    def post_data(self):
        return self._post_data

    @property
    def post_data_buffer(self):
        if self._post_data_buffer is None and self._post_data is not None:
            self._post_data_buffer = self._post_data.encode("utf-8", "replace")
        return self._post_data_buffer


class _Response:
    def __init__(self, tab: Any, event: Any):
        self.url = (getattr(event, "response", None) and event.response.url) or ""
        self.status = int((getattr(event, "response", None) and event.response.status) or 0)
        self._tab = tab
        self._request_id = getattr(event, "request_id", None)

    async def json(self):
        body, base64_enc = await self._tab.send(
            cdp.network.get_response_body(self._request_id))
        if base64_enc:
            body = base64.b64decode(body).decode("utf-8", "replace")
        return json.loads(body)


def _call_handler(handler: Callable, *args: Any) -> None:
    """Invoke a possibly-sync / possibly-async user handler, scheduling async
    ones on the running loop without blocking the event callback."""
    try:
        res = handler(*args)
        if asyncio.iscoroutine(res):
            asyncio.create_task(res)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────
class _Page:
    def __init__(self, context: "_Context", tab: Any):
        self.context = context
        self._tab = tab
        self._mouse = _Mouse(self)
        self._keyboard = _Keyboard(self)
        # `page.frames` is a sync property (the code reads it without await),
        # so it serves from a cache that every async operation refreshes.
        self._frames_cache: List["_Frame"] = []

    async def _refresh_frames(self) -> None:
        try:
            fobs = await self._tab.get_frames()
            self._frames_cache = [_Frame(self, f) for f in fobs]
        except Exception:
            pass

    # ---- basic properties / navigation ------------------------------------
    @property
    def url(self) -> str:
        try:
            return (self._tab.target.url or "") if getattr(self._tab, "target", None) else ""
        except Exception:
            return ""

    async def title(self) -> str:
        try:
            return str(await self._eval("document.title") or "")
        except Exception:
            return ""

    async def goto(self, url: str, wait_until: Optional[str] = None,
                   timeout: Optional[float] = None, **kwargs) -> None:
        t = (timeout or 30000) / 1000.0
        try:
            await self._tab.get(url, timeout=t)
        except Exception:
            # truedriver get() already treats slow loads gracefully; propagate
            # anything else so callers (which wrap goto in try/except) see it.
            raise
        await self._refresh_frames()

    async def reload(self, timeout: Optional[float] = None, **kwargs) -> None:
        try:
            await self._tab.reload()
        except Exception:
            pass
        await self._refresh_frames()

    async def close(self) -> None:
        try:
            await self._tab.close()
        except Exception:
            pass

    # ---- evaluation -------------------------------------------------------
    async def _eval(self, js: Any, arg: Any = None) -> Any:
        expr, await_promise = _wrap_js(js, arg)
        ctx = None
        try:
            ctx = self._tab._get_execution_context_for_evaluate()
        except Exception:
            ctx = None
        ro, errs = await self._tab.send(cdp.runtime.evaluate(
            expression=expr, user_gesture=True, await_promise=await_promise,
            return_by_value=True, allow_unsafe_eval_blocked_by_csp=True,
            context_id=ctx))
        if errs:
            raise RuntimeError(str(errs))
        return ro.value if ro is not None else None

    async def evaluate(self, js: Any, arg: Any = None) -> Any:
        await self._tab.switch_to_main_frame()
        try:
            return await self._eval(js, arg)
        finally:
            await self._refresh_frames()

    # ---- locating ---------------------------------------------------------
    def locator(self, selector: str) -> "_Locator":
        return _Locator(self, selector)

    def frame_locator(self, selector: str) -> "_FrameLocator":
        return _FrameLocator(self, selector)

    @property
    def frames(self) -> List["_Frame"]:
        return self._frames_cache

    async def _frames_for_selector(self, selector: str) -> list:
        await self._tab.switch_to_main_frame()
        try:
            iframes = await self._tab.query_selector_all(selector)
        except Exception:
            return []
        bnodes = set()
        for e in iframes:
            try:
                bnodes.add(int(e.backend_node_id))
            except Exception:
                pass
        frames = await self._tab.get_frames()
        out = []
        for fr in frames:
            try:
                owner, _ = await self._tab.send(cdp.dom.get_frame_owner(fr.id_))
                if int(owner) in bnodes:
                    out.append(fr)
            except Exception:
                continue
        return out

    async def _frame_for_element(self, el: Any) -> Optional["_Frame"]:
        try:
            bnode = int(el.backend_node_id)
        except Exception:
            return None
        frames = await self._tab.get_frames()
        for fr in frames:
            try:
                owner, _ = await self._tab.send(cdp.dom.get_frame_owner(fr.id_))
                if int(owner) == bnode:
                    return _Frame(self, fr)
            except Exception:
                continue
        return None

    # ---- screenshots ------------------------------------------------------
    async def screenshot(self, full_page: bool = False, timeout: Optional[float] = None, **kwargs) -> bytes:
        t = (timeout or 25000) / 1000.0 if timeout else 25.0
        try:
            if full_page:
                try:
                    data = await asyncio.wait_for(
                        self._tab.send(cdp.page.capture_screenshot(
                            format_="png", capture_beyond_viewport=True
                        )),
                        timeout=t
                    )
                    if data:
                        return base64.b64decode(data)
                except Exception:
                    pass
            data = await asyncio.wait_for(
                self._tab.send(cdp.page.capture_screenshot(
                    format_="png", capture_beyond_viewport=False
                )),
                timeout=min(t, 10.0)
            )
            if data:
                return base64.b64decode(data)
        except Exception:
            pass
        try:
            b64 = await self._tab.screenshot_b64("png", full_page=bool(full_page))
            return base64.b64decode(b64)
        except Exception:
            return b""

    # ---- mouse / keyboard / events ----------------------------------------
    @property
    def mouse(self) -> "_Mouse":
        return self._mouse

    @property
    def keyboard(self) -> "_Keyboard":
        return self._keyboard

    def on(self, event: str, handler: Callable) -> None:
        """Register a Playwright-style event listener.

        Sync (must stay sync: callers use it without await). The truedriver
        connection auto-enables the backing CDP domain on the next send().
        """
        if event == "request":
            async def _req_h(e, _conn):
                _call_handler(handler, _Request(e))
            self._tab.add_handler(cdp.network.RequestWillBeSent, _req_h)
        elif event == "response":
            async def _resp_h(e, _conn):
                _call_handler(handler, _Response(self._tab, e))
            self._tab.add_handler(cdp.network.ResponseReceived, _resp_h)
        elif event == "crash":
            async def _crash_h(e, _conn):
                _call_handler(handler)
            self._tab.add_handler(cdp.inspector.TargetCrashed, _crash_h)


# ─────────────────────────────────────────────────────────────────────────────
# CDP session (used by DragSolver._cdp_drag)
# ─────────────────────────────────────────────────────────────────────────────
class _CDPSession:
    def __init__(self, tab: Any):
        self._tab = tab

    async def send(self, method: str, params: Optional[dict] = None) -> Any:
        def _gen():
            yield {"method": method, "params": params or {}}
        return await self._tab.send(_gen())


# ─────────────────────────────────────────────────────────────────────────────
# Context / Browser / Playwright
# ─────────────────────────────────────────────────────────────────────────────
class _Context:
    def __init__(self, browser: "_Browser", opts: Optional[dict] = None):
        self._browser = browser
        self._opts = opts or {}
        self._init_scripts: list = []
        self._pages: list = []
        self._used_tabs = set()

    async def add_init_script(self, script: str) -> None:
        self._init_scripts.append(script)

    async def new_page(self) -> "_Page":
        tab = await self._browser._new_tab(self)
        for s in self._init_scripts:
            src = _wrap_new_document(s)
            if not src:
                continue
            try:
                await tab.send(cdp.page.add_script_to_evaluate_on_new_document(source=src))
            except Exception:
                pass
        page = _Page(self, tab)
        self._pages.append(page)
        try:
            await page._refresh_frames()
        except Exception:
            pass
        return page

    async def new_cdp_session(self, page: Any) -> "_CDPSession":
        return _CDPSession(page._tab)

    async def close(self) -> None:
        for p in list(self._pages):
            try:
                await p.close()
            except Exception:
                pass
        self._pages.clear()


class _Browser:
    def __init__(self, browser: Any):
        self._browser = browser
        self._contexts: list = []

    @property
    def is_connected(self) -> bool:
        if not self._browser:
            return False
        if getattr(self._browser, "stopped", False):
            return False
        return True

    async def new_context(self, **opts) -> "_Context":
        ctx = _Context(self, opts)
        self._contexts.append(ctx)
        return ctx

    async def _new_tab(self, ctx: "_Context") -> Any:
        if not ctx._used_tabs:
            mt = getattr(self._browser, "main_tab", None)
            if mt is not None:
                try:
                    mt.browser = self._browser
                except Exception:
                    pass
                ctx._used_tabs.add(id(mt))
                return mt
        try:
            tab = await self._browser.get("about:blank", new_tab=True)
            tab.browser = self._browser
            return tab
        except Exception:
            mt = getattr(self._browser, "main_tab", None)
            if mt is not None:
                try:
                    mt.browser = self._browser
                except Exception:
                    pass
                return mt
            raise

    async def close(self) -> None:
        try:
            await self._browser.stop()
        except Exception:
            pass


class _Chromium:
    async def launch(self, headless: bool = True, args: Optional[list] = None,
                     proxy: Optional[dict] = None, **kwargs) -> "_Browser":
        if _td_start is None or Config is None:
            raise RuntimeError(
                "truedriver is not installed — run: pip install truedriver")
        cfg = Config(headless=bool(headless),
                     browser_args=list(args or []),
                     sandbox=False,
                     proxy=proxy)
        browser = await _td_start(cfg)
        return _Browser(browser)


class _Playwright:
    def __init__(self):
        self._started = False
        self._chromium: Optional[_Chromium] = None

    async def start(self) -> "_Playwright":
        self._started = True
        return self

    async def stop(self) -> None:
        self._started = False

    @property
    def chromium(self) -> "_Chromium":
        if self._chromium is None:
            self._chromium = _Chromium()
        return self._chromium


def async_playwright() -> "_Playwright":
    return _Playwright()

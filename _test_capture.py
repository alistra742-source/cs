"""Harness: verify the new capture_page_screenshot behavior with a mock page.

Run: python3 _test_capture.py   (stubs missing third-party deps, needs no browser)
"""
import asyncio
import sys
import types


def _stub(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    try:
        __import__(name)
        return sys.modules[name]
    except Exception:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m


class _Any:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, item):
        return _Any()


for _mod in ("aiohttp", "PIL", "PIL.Image", "numpy", "torch"):
    _stub(_mod, ClientSession=_Any, ClientTimeout=_Any, Image=_Any,
          __getattr__=lambda n: _Any())
try:
    import requests  # noqa
except Exception:
    _stub("requests", post=_Any, get=_Any)
# camoufox (the real package) is only needed at import time by
# camoufox_engine; stub its import surface when the real package is absent
# so this harness runs on machines without the browser installed.
try:
    import camoufox.async_api  # noqa: F401
except Exception:
    _stub("camoufox")
    _stub("camoufox.async_api",
          AsyncCamoufox=_Any, AsyncNewContext=_Any, AsyncNewBrowser=_Any)
    _stub("camoufox.exceptions",
          NotInstalledGeoIPExtra=Exception, UnknownIPLocation=Exception)

sys.path.insert(0, "/home/user/cs")
import server  # noqa: E402


class MockPage:
    def __init__(self, form_visible=True, menu_open=False,
                 doc_h=720, vw=1280, vh=720, fullpage_ok=True):
        self.form_visible = form_visible
        self.menu_open = menu_open
        self.doc_h = doc_h
        self.vw = vw
        self.vh = vh
        self.fullpage_ok = fullpage_ok
        self.evaluate_calls = []
        self.screenshot_calls = []

    async def evaluate(self, js, arg=None):
        self.evaluate_calls.append(js)
        if "getBoundingClientRect" in js and "scrollIntoView" in js:
            # _REVEAL_FORM_JS
            if self.menu_open:
                return "ok"
            return "scrolled" if not self.form_visible else "ok"
        if "docH" in js:
            return {"docH": self.doc_h, "vw": self.vw, "vh": self.vh}
        if "window.scrollY" in js:
            return 0
        if js.startswith("window.scrollTo"):
            return None
        return None

    async def screenshot(self, full_page=False, timeout=None, clip=None, **kw):
        self.screenshot_calls.append({"full_page": full_page, "clip": clip,
                                      "timeout": timeout})
        if full_page and not self.fullpage_ok:
            raise TimeoutError("boom")
        return b"FULLPAGE_PNG" if full_page else b"VIEWPORT_PNG"


async def main():
    # 1. form visible -> clean viewport frame, no full-page, no scroll
    p = MockPage()
    out = await server.capture_page_screenshot(p)
    assert out == b"VIEWPORT_PNG", out
    assert all(not c["full_page"] for c in p.screenshot_calls), p.screenshot_calls
    print("case1 ok: form visible -> full viewport frame, no perturbation")

    # 2. form entirely out of sight -> revealed, then viewport frame
    p = MockPage(form_visible=False)
    out = await server.capture_page_screenshot(p)
    assert out == b"VIEWPORT_PNG"
    assert any("scrollIntoView" in c for c in p.evaluate_calls)
    print("case2 ok: form out of sight -> scrolled into view, then frame")

    # 3. menu open -> never scroll, even if form out of sight
    p = MockPage(form_visible=False, menu_open=True)
    out = await server.capture_page_screenshot(p)
    assert out == b"VIEWPORT_PNG"
    # no 'scrolled' path taken: reveal returned 'ok' (mock: menu_open -> ok)
    print("case3 ok: menu open -> scroll left alone")

    # 4. FULLPAGE_SHOTS=1 + tall page -> scroll-through + clipped full capture + restore
    old = server.FULLPAGE_SHOTS
    server.FULLPAGE_SHOTS = True
    p = MockPage(doc_h=1500, vw=1280, vh=720)
    out = await server.capture_page_screenshot(p)
    assert out == b"FULLPAGE_PNG", out
    fp = [c for c in p.screenshot_calls if c["full_page"]]
    assert fp and fp[-1]["clip"] == {"x": 0, "y": 0, "width": 1280, "height": 1500}, fp
    scrolls = [c for c in p.evaluate_calls if c.startswith("window.scrollTo")]
    assert len(scrolls) >= 2 and scrolls[-1] == "window.scrollTo(0, 0)", scrolls
    print(f"case4 ok: full page = scroll-through {scrolls[:-1]} + clip + restore")

    # 5. FULLPAGE_SHOTS=1 but page fits viewport -> plain viewport frame
    p = MockPage(doc_h=720, vw=1280, vh=720)
    out = await server.capture_page_screenshot(p)
    assert out == b"VIEWPORT_PNG"
    assert not [c for c in p.screenshot_calls if c["full_page"]]
    print("case5 ok: page fits viewport -> viewport frame is the full page")

    # 6. FULLPAGE_SHOTS=1, full-page capture fails -> viewport fallback
    p = MockPage(doc_h=1500, vw=1280, vh=720, fullpage_ok=False)
    out = await server.capture_page_screenshot(p)
    assert out == b"VIEWPORT_PNG"
    print("case6 ok: full-page failure -> viewport fallback (no dropped frame)")

    # 7. FULLPAGE_SHOTS=1, page over OOM budget -> viewport frame
    p = MockPage(doc_h=9000, vw=1280, vh=720)
    out = await server.capture_page_screenshot(p)
    assert out == b"VIEWPORT_PNG"
    assert not [c for c in p.screenshot_calls if c["full_page"]]
    print("case7 ok: page over FULLPAGE_MAX_PX -> viewport frame")
    server.FULLPAGE_SHOTS = old

    # 8. default env: FULLPAGE_SHOTS must be OFF now
    assert server.FULLPAGE_SHOTS is False
    print("case8 ok: FULLPAGE_SHOTS default is off (stable feed)")
    print("ALL PASS")


asyncio.run(main())

#!/usr/bin/env python3
"""drag_solver.py — Vision-based solver for FunCAPTCHA / Arkose drag challenges.

FunCAPTCHA renders challenges inside a cross-origin iframe with CANVAS —
you CANNOT query the DOM for slider handles. This solver works entirely
through screenshots + vision model, then executes drags via coordinates.

Three challenge types:
  1. SLIDER  — jigsaw puzzle piece on a track (canvas-based)
  2. TILES   — clickable tiles (DOM elements, sometimes canvas)
  3. MATCH   — drag items to targets
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any, Callable, Optional

# Training data collector — optional. Only loaded when train/ package exists.
try:
    from train.collect import TrainingDataCollector
    _HAS_COLLECTOR = True
except (ImportError, ModuleNotFoundError):
    TrainingDataCollector = None  # type: ignore
    _HAS_COLLECTOR = False

FUNCAPTCHA_URL_PATTERNS = (
    "funcaptcha.com",
    "arkoselabs.com",
    "cdn.funcaptcha.com",
)

_SLIDER_KEYWORDS = ("slide", "drag", "puzzle", "align", "piece", "slider")
_TILE_KEYWORDS = ("click", "tap", "select", "ascending", "descending", "order")
_MATCH_KEYWORDS = ("match", "pair", "connect", "correspond")

# ── Vision prompts ───────────────────────────────────────────────

_SLIDER_PROMPT = (
    "You are solving a slider-puzzle captcha. The image shows a puzzle piece "
    "on a track and a background image with a cutout. "
    "The puzzle piece must be dragged to align with the cutout. "
    "Answer with ONLY a single integer between 0 and 100. "
    "This integer is the percentage of the track width to drag the piece. "
    "0 = all the way left, 100 = all the way right. "
    "Look at the piece and the cutout, decide the exact position, and "
    "output ONLY the number. Example: 42"
)

_SLIDER_LOCATE_PROMPT = (
    "This is a slider captcha. The image shows a track with a movable "
    "puzzle piece. "
    "Tell me: 1) Where is the puzzle piece? Output its LEFT EDGE X position "
    "as a percentage of the image width (0=left, 100=right edge). "
    "2) What percentage should it be dragged to align with the cutout? "
    "Answer ONLY two numbers separated by a space: handle_x_pct target_pct. "
    "Example: 5 42"
)

_TILE_PROMPT = (
    "This is a tile-click captcha with numbered or patterned tiles. "
    "Determine the correct order to click them. "
    "Answer with ONLY a JSON array of tile positions in click order. "
    "Position 1 = top-left, numbered left-to-right, top-to-bottom. "
    "Example: [3, 1, 4, 2]"
)

_MATCH_PROMPT = (
    "This is a drag-to-match captcha. Items must be dragged to targets. "
    "Answer with ONLY a JSON object with items and targets arrays. "
    'Example: {"items": [1, 3, 2], "targets": [3, 1, 2]}'
)


class DragSolver:
    """FunCAPTCHA solver using vision (works on canvas-based challenges)."""

    def __init__(self, page: Any, vision: Any,
                 log: Optional[Callable] = None):
        self._page = page
        self._vision = vision
        self._log = log or (lambda msg, level="info": None)
        self._challenge_type: str = ""
        self._funcaptcha_frame: Any = None
        self._funcaptcha_element: Any = None
        self._collector = TrainingDataCollector() if _HAS_COLLECTOR else None

    # ── Public API ─────────────────────────────────────────────────

    async def detect(self, timeout: float = 10.0) -> bool:
        """Find FunCAPTCHA iframe. Returns True when found."""
        for _ in range(int(timeout / 0.3)):
            frame, el = await self._find_funcaptcha()
            if frame is not None:
                self._funcaptcha_frame = frame
                self._funcaptcha_element = el
                src = "?"
                try:
                    src = (frame.url or "")[:80]
                except Exception:
                    pass
                self._log(f"[DragSolver] Frame: {src}", level="info")
                return True
            await asyncio.sleep(0.3)
        return False

    async def solve(self, timeout: float = 60.0) -> bool:
        """Solve the FunCAPTCHA challenge. Returns True when cleared."""
        if self._funcaptcha_frame is None:
            if not await self.detect():
                self._log("[DragSolver] No frame found", level="warn")
                return False

        deadline = time.monotonic() + timeout
        remaining = 3

        while time.monotonic() < deadline and remaining > 0:
            if await self._past_funcaptcha():
                self._log("[DragSolver] [OK] Cleared", level="info")
                return True

            frame = await self._locate_frame()
            if frame is None:
                self._log("[DragSolver] Frame gone", level="info")
                return await self._past_funcaptcha()

            # Detect canvas vs DOM challenge
            is_canvas = await self._is_canvas_challenge(frame)

            ctype = await self._detect_challenge_type(frame)
            if not ctype:
                ctype = "slider"

            self._challenge_type = ctype
            self._log(f"[DragSolver] Type: {ctype}"
                      f"{' [canvas]' if is_canvas else ' [dom]'}",
                      level="info")

            solved = False
            if ctype == "slider":
                solved = await self._solve_slider(frame, is_canvas)
            elif ctype == "tiles":
                solved = await self._solve_tiles(frame, is_canvas)
            elif ctype == "match":
                solved = await self._solve_match(frame, is_canvas)

            if solved:
                self._log("[DragSolver] [OK] Solved", level="info")
                for _ in range(15):
                    if await self._past_funcaptcha():
                        self._log("[DragSolver] [OK] Cleared!", level="info")
                        return True
                    await asyncio.sleep(0.5)
                self._log("[DragSolver] Solved but not cleared yet",
                          level="warn")
                remaining -= 1
                await asyncio.sleep(2.0)
            else:
                self._log("[DragSolver] Retrying", level="warn")
                remaining -= 1
                await asyncio.sleep(2.0)

        self._log("[DragSolver] [FAIL]", level="error")
        return False

    # ── Frame discovery ────────────────────────────────────────────

    async def _find_funcaptcha(self) -> tuple:
        """Find a FunCAPTCHA iframe by URL pattern.
        Returns (Playwright Frame, locator_element) or (None, None)."""
        for f in self._page.frames:
            try:
                furl = (f.url or "").lower()
            except Exception:
                continue
            if any(p in furl for p in FUNCAPTCHA_URL_PATTERNS):
                return f, None

        for pattern in FUNCAPTCHA_URL_PATTERNS:
            loc = self._page.locator(f'iframe[src*="{pattern}"]')
            try:
                count = await loc.count()
            except Exception:
                continue
            if count > 0:
                try:
                    cf = await loc.first.content_frame()
                    if cf is not None:
                        return cf, loc.first
                except Exception:
                    pass
                for f in self._page.frames:
                    try:
                        furl = (f.url or "").lower()
                    except Exception:
                        continue
                    if pattern in furl:
                        return f, loc.first
        return None, None

    async def _locate_frame(self):
        """Re-acquire reference to the FunCAPTCHA frame."""
        if self._funcaptcha_frame is not None:
            try:
                _ = self._funcaptcha_frame.url
                return self._funcaptcha_frame
            except Exception:
                pass
        frame, _ = await self._find_funcaptcha()
        self._funcaptcha_frame = frame
        return frame

    # ── Canvas detection ──────────────────────────────────────────

    async def _is_canvas_challenge(self, frame) -> bool:
        """Detect if the challenge renders via canvas (not DOM elements)."""
        try:
            has_canvas = await frame.evaluate("""() => {
                return document.querySelector('canvas') !== null;
            }""")
            return bool(has_canvas)
        except Exception:
            pass
        # Cross-origin: evaluate throws, assume canvas-based (common)
        return True

    # ── Challenge type detection ──────────────────────────────────

    async def _detect_challenge_type(self, frame) -> str:
        """Classify challenge: slider / tiles / match."""
        try:
            text = await frame.evaluate(
                "() => (document.body ? document.body.innerText : '')"
            )
            low = (text or "").lower()
        except Exception:
            low = ""

        if any(kw in low for kw in _SLIDER_KEYWORDS):
            return "slider"
        if any(kw in low for kw in _TILE_KEYWORDS):
            return "tiles"
        if any(kw in low for kw in _MATCH_KEYWORDS):
            return "match"

        fallback_prompt = (
            "Which type of captcha is this? One word only: "
            "slider (puzzle piece on track), "
            "tiles (grid of clickable items), "
            "or match (drag items to targets)."
        )
        try:
            img = await self._screenshot_frame(frame)
            if not img:
                return "slider"
            ans = await self._vision.solve(fallback_prompt, [img], timeout=30)
            if ans and ans.get("type") == "text":
                ta = ans["text"].strip().lower()
                if "tile" in ta:
                    return "tiles"
                if "match" in ta:
                    return "match"
        except Exception:
            pass
        return "slider"

    # ── SLIDER SOLVER ──────────────────────────────────────────────
    # FunCAPTCHA sliders are CANVAS-based. No DOM handles to grab.
    # Strategy: screenshot -> vision (locate handle + target %)
    #         -> drag from handle position to target position

    async def _solve_slider(self, frame, is_canvas: bool) -> bool:
        """Solve slider puzzle. Works on canvas or DOM."""
        try:
            img = await self._screenshot_frame(frame)
            if not img:
                self._log("[DragSolver] No screenshot", level="warn")
                return False

            # Step 1: Get the iframe bounding box ON THE PAGE
            fbox = await self._frame_box(frame)
            if not fbox:
                self._log("[DragSolver] No iframe box", level="warn")
                return False

            fx, fy = fbox["x"], fbox["y"]
            fw, fh = fbox["width"], fbox["height"]

            # Step 2: Ask vision model for drag percentage
            self._log("[DragSolver] Asking vision...", level="info")

            # Canvas challenges: ask for handle + target in one query
            if is_canvas:
                ans = await self._vision.solve(
                    _SLIDER_LOCATE_PROMPT, [img], timeout=60)
                handle_pct, target_pct = self._parse_dual_answer(ans)
                if handle_pct is not None and target_pct is not None:
                    handle_pct = max(0, min(100, handle_pct))
                    target_pct = max(0, min(100, target_pct))
                else:
                    # Fall back to single prompt
                    ans = await self._vision.solve(
                        _SLIDER_PROMPT, [img], timeout=60)
                    target_pct = self._parse_percentage(ans)
                    handle_pct = None  # will guess from position
            else:
                ans = await self._vision.solve(
                    _SLIDER_PROMPT, [img], timeout=60)
                target_pct = self._parse_percentage(ans)
                handle_pct = None

            if target_pct is None:
                self._log("[DragSolver] Vision: no target", level="warn")
                return False

            target_pct = max(0, min(100, target_pct))
            self._log(f"[DragSolver] Target: {target_pct}%", level="info")

            # Save training sample (optional, only when train/ package available)
            if img and self._collector:
                self._collector.save(
                    img, _SLIDER_PROMPT, str(int(target_pct)), "slider"
                )

            # Step 3: Find the handle's page coordinates
            if handle_pct is not None:
                # Vision told us where the handle is (as % of iframe width)
                sx = fx + (fw * handle_pct / 100.0)
            else:
                # Guess: handle is on the left side 5-15% in
                sx = fx + fw * 0.10

            # Step 4: Calculate target position
            # The track is typically full iframe width minus ~10% padding
            track_left = fx + fw * 0.05
            track_right = fx + fw - fw * 0.05
            track_width = track_right - track_left
            tx = track_left + track_width * (target_pct / 100.0)

            # Vertically: the slider is typically in the lower half
            sy = fy + fh * 0.65  # 65% down from top
            ty = sy  # same Y, horizontal drag only

            self._log(f"[DragSolver] Page coords: "
                      f"start=({sx:.0f},{sy:.0f}) target=({tx:.0f},{ty:.0f}) "
                      f"iframe=({fx:.0f},{fy:.0f},{fw:.0f}x{fh:.0f})",
                      level="debug")

            # Step 5: Execute drag with multiple strategies
            for strategy in ("human", "direct", "cdp"):
                if strategy == "human":
                    ok = await self._human_drag(sx, sy, tx, ty)
                elif strategy == "direct":
                    ok = await self._direct_drag(sx, sy, tx, ty)
                else:
                    ok = await self._cdp_drag(sx, sy, tx, ty)

                if ok:
                    # Verify
                    await asyncio.sleep(0.8)
                    for _ in range(8):
                        if await self._past_funcaptcha():
                            return True
                        await asyncio.sleep(0.4)
                    # Didn't clear but drag worked — might be wrong pct
                    self._log("[DragSolver] Drag ok but not cleared",
                              level="warn")
                    return True

            self._log("[DragSolver] All drag strategies failed",
                      level="warn")
            return False

        except Exception as e:
            self._log(f"[DragSolver] Slider error: {e}", level="error")
            return False

    # ── Three drag strategies ──────────────────────────────────────

    async def _human_drag(self, x1, y1, x2, y2) -> bool:
        """Human-like drag with easing and jitter."""
        try:
            await self._page.mouse.move(x1, y1)
            await asyncio.sleep(random.uniform(0.05, 0.15))
            await self._page.mouse.down()
            await asyncio.sleep(random.uniform(0.03, 0.08))

            dist = abs(x2 - x1) + abs(y2 - y1)
            steps = max(10, min(40, int(dist / 8)))

            for s in range(1, steps + 1):
                t = s / steps
                eased = 1 - (1 - t) ** 3  # cubic ease-out
                cx = x1 + (x2 - x1) * eased
                cy = y1 + (y2 - y1) * eased + random.uniform(-2, 2)
                await self._page.mouse.move(cx, cy)
                await asyncio.sleep(random.uniform(0.008, 0.025))

            await asyncio.sleep(random.uniform(0.03, 0.08))
            await self._page.mouse.up()
            self._log(f"[DragSolver] Human drag: {dist:.0f}px in {steps} steps",
                      level="debug")
            return True
        except Exception as e:
            self._log(f"[DragSolver] Human drag error: {e}", level="warn")
            return False

    async def _direct_drag(self, x1, y1, x2, y2) -> bool:
        """Direct Playwright drag (no easing, fast)."""
        try:
            await self._page.mouse.move(x1, y1)
            await self._page.mouse.down()
            await self._page.mouse.move(x2, y2, steps=1)  # single step
            await self._page.mouse.up()
            self._log("[DragSolver] Direct drag done", level="debug")
            return True
        except Exception as e:
            self._log(f"[DragSolver] Direct drag error: {e}", level="warn")
            return False

    async def _cdp_drag(self, x1, y1, x2, y2) -> bool:
        """CDP-level drag via Input.drag gesture."""
        try:
            cdp = await self._page.context.new_cdp_session(self._page)
            await cdp.send("Input.dispatchDragEvent", {
                "type": "dragStart",
                "x": x1, "y": y1,
                "data": {"items": [], "dragOperationsMask": 1},
            })
            await asyncio.sleep(0.02)
            await cdp.send("Input.dispatchDragEvent", {
                "type": "drag",
                "x": x2, "y": y2,
                "data": {"items": [], "dragOperationsMask": 1},
            })
            await asyncio.sleep(0.02)
            await cdp.send("Input.dispatchDragEvent", {
                "type": "dragEnd",
                "x": x2, "y": y2,
                "data": {"items": [], "dragOperationsMask": 1},
            })
            self._log("[DragSolver] CDP drag done", level="debug")
            return True
        except Exception as e:
            self._log(f"[DragSolver] CDP drag error: {e}", level="warn")
            return False

    # ── TILES solver ──────────────────────────────────────────────

    async def _solve_tiles(self, frame, is_canvas: bool) -> bool:
        """Click tiles in the correct order."""
        try:
            img = await self._screenshot_frame(frame)
            if not img:
                return False

            ans = await self._vision.solve(_TILE_PROMPT, [img], timeout=60)
            if not ans or ans.get("type") not in ("tiles", "text"):
                self._log("[DragSolver] Tiles: no answer", level="warn")
                return False

            indices = ans.get("indices", [])
            if not indices:
                text = ans.get("text", "")
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        indices = [int(i) for i in parsed
                                   if isinstance(i, (int, float))]
                except Exception:
                    pass

            if not indices:
                self._log("[DragSolver] Tiles: no indices", level="warn")
                return False

            self._log(f"[DragSolver] Tiles: order {indices}", level="info")

            clicked = 0
            for idx in indices:
                tile = await self._find_tile(frame, idx)
                if tile is None:
                    continue
                try:
                    box = await tile.bounding_box()
                    if box and box.get("width", 0) > 4:
                        cx = box["x"] + box["width"] / 2
                        cy = box["y"] + box["height"] / 2
                        await self._page.mouse.move(cx, cy)
                        await asyncio.sleep(random.uniform(0.1, 0.25))
                        await self._page.mouse.click(cx, cy)
                        clicked += 1
                        await asyncio.sleep(random.uniform(0.15, 0.35))
                except Exception:
                    continue

            self._log(f"[DragSolver] Tiles: {clicked}/{len(indices)}",
                      level="info")
            return clicked > 0

        except Exception as e:
            self._log(f"[DragSolver] Tiles error: {e}", level="error")
            return False

    async def _find_tile(self, frame, idx: int):
        """Return locator for nth tile (1-based)."""
        try:
            tiles = frame.locator(
                '[role="button"], [role="option"], button, '
                '[class*="tile"], [class*="grid"] img, '
                '[class*="option"], canvas'
            )
            n = await tiles.count()
            if n >= idx:
                return tiles.nth(idx - 1)
        except Exception:
            pass
        return None

    # ── MATCH solver ──────────────────────────────────────────────

    async def _solve_match(self, frame, is_canvas: bool) -> bool:
        """Solve drag-to-match."""
        try:
            img = await self._screenshot_frame(frame)
            if not img:
                return False

            ans = await self._vision.solve(_MATCH_PROMPT, [img], timeout=60)
            if not ans or ans.get("type") != "text":
                self._log("[DragSolver] Match: no answer", level="warn")
                return False

            mapping = None
            text = ans.get("text", "")
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    items = parsed.get("items", [])
                    targets = parsed.get("targets", [])
                    if items and targets and len(items) == len(targets):
                        mapping = list(zip(items, targets))
            except Exception:
                pass

            if not mapping:
                self._log("[DragSolver] Match: bad mapping", level="warn")
                return False

            self._log(f"[DragSolver] Match: {mapping}", level="info")

            for item_idx, target_idx in mapping:
                it_el = await self._find_tile(frame, item_idx)
                tg_el = await self._find_tile(frame, target_idx)
                if it_el is None or tg_el is None:
                    continue
                try:
                    ibox = await it_el.bounding_box()
                    tbox = await tg_el.bounding_box()
                    if not ibox or not tbox:
                        continue
                    sx = ibox["x"] + ibox["width"] / 2
                    sy = ibox["y"] + ibox["height"] / 2
                    tx = tbox["x"] + tbox["width"] / 2
                    ty = tbox["y"] + tbox["height"] / 2
                    await self._human_drag(sx, sy, tx, ty)
                    await asyncio.sleep(random.uniform(0.3, 0.6))
                except Exception:
                    continue

            return True

        except Exception as e:
            self._log(f"[DragSolver] Match error: {e}", level="error")
            return False

    # ── Utilities ─────────────────────────────────────────────────

    async def _screenshot_frame(self, frame) -> Optional[bytes]:
        """Screenshot the FunCAPTCHA iframe content."""
        try:
            return await frame.screenshot(timeout=10000)
        except Exception:
            pass
        if self._funcaptcha_element:
            try:
                return await self._funcaptcha_element.screenshot(
                    timeout=10000)
            except Exception:
                pass
        return None

    async def _frame_box(self, frame) -> Optional[dict]:
        """Get the iframe's bounding box (page coordinates)."""
        if self._funcaptcha_element:
            try:
                return await self._funcaptcha_element.bounding_box()
            except Exception:
                pass
        try:
            for f in self._page.frames:
                try:
                    if f.url and f.url == (frame.url or ""):
                        fe = f.frame_element()
                        box = await fe.bounding_box()
                        if box and box.get("width", 0) > 4:
                            return box
                except Exception:
                    continue
        except Exception:
            pass
        return None

    async def _past_funcaptcha(self) -> bool:
        """True when the captcha was cleared (frame gone or URL changed)."""
        try:
            frame, _ = await self._find_funcaptcha()
            if frame is None:
                return True
        except Exception:
            pass
        try:
            url = (self._page.url or "").lower()
            if any(k in url for k in ("/channels", "/verify", "/welcome",
                                      "discord.com/app")):
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _parse_percentage(ans: Optional[dict]) -> Optional[float]:
        """Extract a percentage (0-100) from the vision model's answer."""
        if not ans:
            return None
        text = ""
        if ans.get("type") == "text":
            text = ans["text"].strip()
        elif ans.get("type") == "tiles" and ans.get("indices"):
            val = ans["indices"][0]
            if 0 <= val <= 100:
                return float(val)

        if text:
            cleaned = re.sub(r"[^0-9.-]", "", text)
            try:
                val = float(cleaned)
                if 0 <= val <= 100:
                    return val
                if 100 < val <= 1000:
                    return val / 10.0
            except ValueError:
                pass
            nums = re.findall(r"(\d+\.?\d*)", text)
            if nums:
                val = float(nums[0])
                if 0 <= val <= 100:
                    return val
                if 100 < val <= 1000:
                    return val / 10.0
                # If > 1000, it's probably a pixel value - convert
                # Assume the image width is ~400-800px range
                if 1000 < val <= 8000:
                    return val / 100.0  # optimistic: assume 400px width
        return None

    @staticmethod
    def _parse_dual_answer(ans: Optional[dict]) -> tuple:
        """Parse 'handle_pct target_pct' from vision answer.
        Returns (handle_pct, target_pct) or (None, None)."""
        if not ans or ans.get("type") != "text":
            return None, None
        text = ans["text"].strip()
        nums = re.findall(r"(\d+\.?\d*)", text)
        if len(nums) >= 2:
            try:
                h = float(nums[0])
                t = float(nums[1])
                return h, t
            except (ValueError, IndexError):
                pass
        return None, None


async def _self_test() -> None:
    """Test parsers."""
    cases = [
        ({"type": "text", "text": "42"}, 42.0),
        ({"type": "text", "text": "75%"}, 75.0),
        ({"type": "text", "text": "about 63 percent"}, 63.0),
        ({"type": "text", "text": "250"}, 25.0),
        ({"type": "tiles", "indices": [50]}, 50.0),
        ({"type": "text", "text": "0"}, 0.0),
        ({"type": "text", "text": "100"}, 100.0),
        ({"type": "text", "text": "abc"}, None),
        (None, None),
    ]
    for ans, expected in cases:
        result = DragSolver._parse_percentage(ans)
        status = "OK" if result == expected else \
            f"FAIL (got {result}, expected {expected})"
        print(f"  parse_pct({ans}) = {result} {status}")

    # Dual parse tests
    dual_cases = [
        ({"type": "text", "text": "5 42"}, (5.0, 42.0)),
        ({"type": "text", "text": "10 50"}, (10.0, 50.0)),
        ({"type": "text", "text": "just 42"}, (None, None)),
        (None, (None, None)),
    ]
    for ans, expected in dual_cases:
        result = DragSolver._parse_dual_answer(ans)
        status = "OK" if result == expected else \
            f"FAIL (got {result}, expected {expected})"
        print(f"  parse_dual({ans}) = {result} {status}")

    print("[DragSolver] Self-test complete.")


if __name__ == "__main__":
    asyncio.run(_self_test())
#!/usr/bin/env python3
"""drag_solver.py — Vision-based solver for FunCAPTCHA / Arkose drag challenges.

FunCAPTCHA renders challenges inside a cross-origin iframe with CANVAS —
you CANNOT query the DOM for slider handles. This solver works entirely
through screenshots + vision model, then executes drags via coordinates.

Four challenge types:
  1. SLIDER  — jigsaw puzzle piece on a track (canvas-based)
  2. TILES   — clickable tiles (DOM elements, sometimes canvas)
  3. MATCH   — drag items to targets
  4. STACK   — stacks of blocks, drag blocks on top so every column is
               equally tall (the "3, 3, 3" boxes puzzle)
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
# Block-stacking game ("drag the boxes so every column is the same height").
# Checked BEFORE the generic drag wording above — "drag the block onto the
# stack" contains "drag" but is a STACK round, not a slider.
_STACK_KEYWORDS = ("same height", "equal height", "same number", "same amount",
                   "stack", "tower", "pile", "balance")

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

# Block-stacking puzzle: several vertical stacks of blocks (e.g. one 3 tall,
# one 2 tall, one 1 tall) plus loose draggable blocks; every stack must end
# up the SAME height. The vision call uses shape="stack" so the multi-drag
# plan comes back structured (see vision_solver._parse_stack_geometry);
# this prompt is the task description that rides along with it.
_STACK_PROMPT = (
    "The image shows a stacking puzzle: several vertical stacks (columns / "
    "towers) built from blocks, and one or more loose draggable blocks. "
    "Drag blocks on top of the stacks so that EVERY stack ends up with the "
    "SAME number of blocks (make all columns equally tall — e.g. if the "
    "stacks must be 3 blocks high, every column needs exactly 3). List "
    "EVERY drag needed, in order. Grab each block by its centre and drop "
    "it exactly on top of the target stack. Answer with ONLY the JSON "
    'object {"drags": [[sx, sy, tx, ty], ...]} where sx, sy is the centre '
    "of the block to grab and tx, ty is the drop point on the target "
    "stack, all four as integer PERCENTAGES of the image size (0-100, "
    "measured from the left edge / top edge). "
    'Example: {"drags": [[12, 80, 12, 40], [88, 82, 50, 55]]}'
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
            elif ctype == "stack":
                solved = await self._solve_stack(frame, is_canvas)

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
                    # Real Playwright: resolve the element handle first, then
                    # its async content_frame() (Locator.content_frame is a
                    # FrameLocator property, not the Frame we want here).
                    el = await loc.first.element_handle(timeout=3000)
                    cf = await el.content_frame() if el is not None else None
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
        """Classify challenge: slider / tiles / match / stack."""
        try:
            text = await frame.evaluate(
                "() => (document.body ? document.body.innerText : '')"
            )
            low = (text or "").lower()
        except Exception:
            low = ""

        # Stack wording first: it usually also contains "drag", which would
        # otherwise misroute the round to the slider solver.
        if any(kw in low for kw in _STACK_KEYWORDS):
            return "stack"
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
            "match (drag items to matching targets), or "
            "stack (columns/towers of stacked blocks — drag blocks on top "
            "so every column becomes the same height)."
        )
        try:
            img = await self._screenshot_frame(frame)
            if not img:
                return "slider"
            ans = await self._vision.solve(fallback_prompt, [img], timeout=30)
            if ans and ans.get("type") == "text":
                ta = ans["text"].strip().lower()
                if "stack" in ta:
                    return "stack"
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

    # ── STACK solver ──────────────────────────────────────────────

    async def _solve_stack(self, frame, is_canvas: bool) -> bool:
        """Solve the block-stacking puzzle ("make every column 3 tall").

        Vision returns the drag plan as iframe-RELATIVE percentages
        (0-100); we convert to absolute page coordinates using the iframe
        box and replay each drag humanised. Wrong/failed rounds simply
        return False — solve()'s retry loop re-screenshots and re-asks.
        """
        try:
            img = await self._screenshot_frame(frame)
            if not img:
                self._log("[DragSolver] Stack: no screenshot", level="warn")
                return False

            fbox = await self._frame_box(frame)
            if not fbox:
                self._log("[DragSolver] Stack: no iframe box", level="warn")
                return False
            fx, fy = fbox["x"], fbox["y"]
            fw, fh = fbox["width"], fbox["height"]

            self._log("[DragSolver] Stack: asking vision for the drag plan...",
                      level="info")
            ans = None
            try:
                ans = await self._vision.solve(
                    _STACK_PROMPT, [img], timeout=90, shape="stack")
            except TypeError:
                # Vision facade without the shape kwarg — plain call, the
                # multi-transport plan parser handles whatever comes back.
                ans = await self._vision.solve(_STACK_PROMPT, [img], timeout=90)

            plan = self._parse_stack_plan(ans)
            if not plan:
                self._log(f"[DragSolver] Stack: no plan from {ans}",
                          level="warn")
                return False

            self._log(f"[DragSolver] Stack: {len(plan)} drag(s): "
                      + " ".join(f"({s:.0f},{sy:.0f})->({t:.0f},{ty:.0f})"
                                 for s, sy, t, ty in plan), level="info")

            # Save training sample (optional, only when train/ is available)
            if img and self._collector:
                self._collector.save(
                    img, _STACK_PROMPT,
                    json.dumps([list(d) for d in plan]), "stack")

            # Percent -> page coords + a little human jitter that stays
            # well inside the grabbed block.
            for sx, sy, tx, ty in plan:
                jx = fw * 0.015
                jy = fh * 0.015
                x1 = fx + fw * sx / 100.0 + random.uniform(-jx, jx)
                y1 = fy + fh * sy / 100.0 + random.uniform(-jy, jy)
                x2 = fx + fw * tx / 100.0 + random.uniform(-jx, jx)
                y2 = fy + fh * ty / 100.0 + random.uniform(-jy, jy)
                ok = await self._human_drag(x1, y1, x2, y2)
                if not ok:
                    ok = await self._direct_drag(x1, y1, x2, y2)
                if not ok:
                    self._log("[DragSolver] Stack: drag failed mid-plan",
                              level="warn")
                    return False
                await asyncio.sleep(random.uniform(0.3, 0.7))

            # Give Arkose a moment to advance, then let solve()'s loop
            # verify via _past_funcaptcha (and retry when it did not).
            await asyncio.sleep(1.0)
            return True

        except Exception as e:
            self._log(f"[DragSolver] Stack error: {e}", level="error")
            return False

    @staticmethod
    def _parse_stack_plan(ans: Optional[dict]) -> Optional[list]:
        """Extract a drag plan [(sx, sy, tx, ty), ...] in 0-100 iframe
        percentages from a vision answer, tolerating every transport the
        answer pipeline can produce:

          {"type": "drags",  "drags": [(...), ...]}  — shape="stack" parse
          {"type": "drag",   "from": (x, y), "to": (x, y)} — 0-1 fractions
          {"type": "text",   "text": '{"drags": ...}' or "sx sy tx ty ..." }
          {"type": "tiles",  "indices": [sx, sy, tx, ty, ...]} — chunks of 4
          {"type": "points", "points": [(x, y), ...]} — 0-1 fractions,
                            pairs (grab, drop, grab, drop, ...)

        Values clamp to 0-100; a coordinate beyond -3..103 rejects its drag
        (hallucinated pixels); at most 12 drags; None when nothing usable.
        """
        if not isinstance(ans, dict):
            return None

        def _finish(raw: list) -> Optional[list]:
            out = []
            for item in raw:
                try:
                    a, b, c, d = (float(v) for v in item)
                except (TypeError, ValueError):
                    continue
                if any(v != v for v in (a, b, c, d)):  # NaN
                    continue
                if any(v < -3 or v > 103 for v in (a, b, c, d)):
                    continue
                out.append((max(0.0, min(100.0, a)), max(0.0, min(100.0, b)),
                            max(0.0, min(100.0, c)), max(0.0, min(100.0, d))))
            return [tuple(round(v, 3) for v in drag) for drag in out] or None

        # Primary: the structured multi-drag plan from shape="stack".
        if ans.get("type") == "drags" and ans.get("drags"):
            plan = _finish(ans["drags"])
            if plan:
                return plan[:12]

        # Squashed single drag (0-1 fractions) -> one percent drag.
        if ans.get("type") == "drag" and ans.get("from") and ans.get("to"):
            fx, fy = ans["from"]
            tx, ty = ans["to"]
            plan = _finish([[fx * 100.0, fy * 100.0, tx * 100.0, ty * 100.0]])
            if plan:
                return plan

        # Flat number lists (tiles indices / free text) -> chunks of four.
        nums: list = []
        if ans.get("type") == "tiles" and ans.get("indices"):
            nums = [float(v) for v in ans["indices"]]
        elif ans.get("type") == "text" and isinstance(ans.get("text"), str):
            text = ans["text"].strip()
            # fenced / embedded JSON first
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            candidate = m.group(1).strip() if m else text
            lo, hi = candidate.find("{"), candidate.rfind("}")
            obj = None
            if 0 <= lo < hi:
                try:
                    obj = json.loads(candidate[lo:hi + 1])
                except Exception:
                    obj = None
            if obj is None:
                lo, hi = candidate.find("["), candidate.rfind("]")
                if 0 <= lo < hi:
                    try:
                        obj = json.loads(candidate[lo:hi + 1])
                    except Exception:
                        obj = None
            if isinstance(obj, dict):
                raw = None
                for key in ("drags", "moves", "drag"):
                    v = obj.get(key)
                    if isinstance(v, list) and v:
                        raw = v
                        break
                if raw is None:
                    return None
                nums = []
                for d in raw:
                    vals = d if isinstance(d, (list, tuple)) else [d]
                    try:
                        nums.extend(float(x) for x in vals[:4])
                    except (TypeError, ValueError):
                        return None
            elif isinstance(obj, list):
                nums = [float(x) for d in obj for x in
                        (d if isinstance(d, (list, tuple)) else [d])]
            else:
                nums = [float(v) for v in re.findall(r"\d+\.?\d*", text)]
        if len(nums) >= 4:
            chunks = [nums[i:i + 4] for i in range(0, len(nums), 4)]
            plan = _finish(chunks)
            if plan:
                return plan[:12]

        # Points pairs (0-1 fractions): grab, drop, grab, drop, ...
        if ans.get("type") == "points" and ans.get("points"):
            pts = [(float(p[0]), float(p[1])) for p in ans["points"]]
            if len(pts) >= 4 and len(pts) % 2 == 0:
                plan = _finish([[pts[i][0] * 100.0, pts[i][1] * 100.0,
                                 pts[i + 1][0] * 100.0, pts[i + 1][1] * 100.0]
                                for i in range(0, len(pts), 2)])
                if plan:
                    return plan[:12]
        return None

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
                        fe = await f.frame_element()
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

    # Stack plan parse tests
    stack_cases = [
        # structured multi-drag plan (shape="stack" transport)
        ({"type": "drags", "drags": [[10, 80, 10, 45], [90, 85, 50, 60]]},
         [(10.0, 80.0, 10.0, 45.0), (90.0, 85.0, 50.0, 60.0)]),
        # squashed single drag, 0-1 fractions -> percent
        ({"type": "drag", "from": (0.12, 0.8), "to": (0.12, 0.4)},
         [(12.0, 80.0, 12.0, 40.0)]),
        # fenced JSON in a text transport
        ({"type": "text", "text":
          '```json\n{"drags": [[12, 80, 12, 40], [88, 82, 50, 55]]}\n```'},
         [(12.0, 80.0, 12.0, 40.0), (88.0, 82.0, 50.0, 55.0)]),
        # bare numbers line ("10 80 10 45 88 82 50 60")
        ({"type": "text", "text": "10 80 10 45 88 82 50 60"},
         [(10.0, 80.0, 10.0, 45.0), (88.0, 82.0, 50.0, 60.0)]),
        # flat tile indices -> chunks of four
        ({"type": "tiles", "indices": [12, 80, 12, 40]},
         [(12.0, 80.0, 12.0, 40.0)]),
        # points pairs (0-1) -> grab/drop pairs in percent
        ({"type": "points", "points": [(0.12, 0.8), (0.12, 0.4),
                                       (0.88, 0.82), (0.5, 0.55)]},
         [(12.0, 80.0, 12.0, 40.0), (88.0, 82.0, 50.0, 55.0)]),
        # out-of-range percent (pixel hallucination) rejects the drag
        ({"type": "drags", "drags": [[10, 80, 10, 450]]}, None),
        # unparseable prose
        ({"type": "text", "text": "I cannot see any blocks"}, None),
        (None, None),
    ]
    for ans, expected in stack_cases:
        result = DragSolver._parse_stack_plan(ans)
        status = "OK" if result == expected else \
            f"FAIL (got {result}, expected {expected})"
        print(f"  parse_stack({(ans or {}).get('type')}) = {result} {status}")

    # Stack wording routes to the stack solver, not the slider solver
    class _StubFrame:
        def __init__(self, text: str):
            self._text = text

        async def evaluate(self, js, *args):
            return self._text

    solver = DragSolver(page=None, vision=None)
    for wording, expected in (
        ("Drag the boxes so each column has the same height", "stack"),
        ("Stack the blocks until every tower is equal", "stack"),
        ("Slide the piece to align with the cutout", "slider"),
        ("Click each tile in ascending order", "tiles"),
    ):
        got = await solver._detect_challenge_type(_StubFrame(wording))
        status = "OK" if got == expected else \
            f"FAIL (got {got}, expected {expected})"
        print(f"  detect({wording[:38]!r}) = {got} {status}")

    print("[DragSolver] Self-test complete.")


if __name__ == "__main__":
    asyncio.run(_self_test())
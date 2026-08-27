#!/usr/bin/env python3
"""
brain_test.py — the TEST tab: live hCaptcha, solved by the Brain.

Runs the EXACT trainer flow (navigate the official hCaptcha demo, fill the
demo field, click the real checkbox, capture the challenge) — and when a
challenge round appears, the BRAIN (models/brain.pt, brain.py BrainSolver)
solves it for real:

  - routes the family from the live prompt
  - computes the answer (tiles / point / drag / box / count / text)
  - EXECUTES it on the live challenge (clicks the tiles, clicks the point,
    performs the drag) and submits
  - reports the answer + confidence in the Test tab, with an overlay image

models/brain.pt is reassembled automatically when missing: from loose part
files, from git (any ref), or downloaded from GitHub raw (works on any
machine with internet). No training, no sample collection.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import trainer
from trainer import TrainerEngine

import hcaptcha_types as hct

# Commit on arena/01a033e0-cs that carries the split brain parts — used as
# the GitHub raw fallback when neither loose files nor git have them.
_BRAIN_SHA = "5886f0e3c86ffc8cabbde701412f938bcecb9f5a"

# The live page the Test tab drives (override with BRAIN_TEST_URL).
TARGET_URL = os.environ.get(
    "BRAIN_TEST_URL",
    "https://democaptcha.com/demo-form-eng/hcaptcha.html")

_TILE_BOXES_JS = r"""() => {
    const nodes = document.querySelectorAll(
        'div.task-image, [class*="task-image" i]');
    const vw = window.innerWidth || 1, vh = window.innerHeight || 1;
    const boxes = [];
    nodes.forEach(el => {
        const r = el.getBoundingClientRect();
        if (r.width > 8 && r.height > 8)
            boxes.push({x: r.left / vw, y: r.top / vh,
                        w: r.width / vw, h: r.height / vh});
    });
    return JSON.stringify(boxes);
}"""


class BrainTestEngine(TrainerEngine):
    """Live hCaptcha demo rounds solved by the Brain (analysis + clicks)."""

    _PART_PATTERNS = (("brain_part_%02d", 0), ("brain.pt.part-%02d", 0),
                      ("brain.pt.part-%02d", 1))

    def __init__(self):
        super().__init__()
        self._solver = None
        self._solver_tried = False
        self._solver_lock = threading.Lock()
        self._brain_state = "not-loaded"   # not-loaded/loading/loaded/failed
        self._brain_error = ""
        self._tile_boxes: List[Dict[str, float]] = []
        self._challenge_iframe = None
        self._challenge_frame = None
        self._last_res: Optional[Dict[str, Any]] = None
        self.rounds: List[Dict[str, Any]] = []
        self.answered_count = 0
        self.deferred_count = 0
        self.executed_count = 0

    # ── brain file: reassemble from loose files / git / GitHub raw ───────
    def _write_brain(self, parts, js_path, ref="") -> None:
        root = os.path.dirname(os.path.abspath(__file__))
        pt_path = os.path.join(os.path.dirname(js_path), "brain.pt")
        os.makedirs(os.path.dirname(js_path), exist_ok=True)
        with open(pt_path, "wb") as f:
            for chunk in parts:
                f.write(chunk)
        d = {}
        if ref:
            import subprocess
            j = subprocess.run(["git", "show", "%s:brain.json" % ref],
                               cwd=root, capture_output=True, timeout=30)
            try:
                if j.returncode == 0 and j.stdout[:1] == b"{":
                    d = json.loads(j.stdout)
            except Exception:
                d = {}
        try:
            d.setdefault("classes", __import__("make_dataset").CLASSES)
        except Exception:
            d.setdefault("classes", [])
        d.setdefault("families", ["image_label_binary", "area_select_point",
                                  "area_select_bbox", "image_drag_drop",
                                  "multiple_choice", "text_entry", "counting",
                                  "pattern", "tower"])
        d.setdefault("scene_size", 96)
        d.setdefault("tile_size", 64)
        d.setdefault("arch", {}).setdefault("width", 48)
        d["arch"].setdefault("prompt_dim", 512)
        d["arch"].setdefault("prompt_layers", 8)
        d["arch"].setdefault("d_concept", 320)
        d["arch"].setdefault("pattern_d", 320)
        d["arch"].setdefault("pattern_layers", 4)
        d["arch"]["text_len"] = 5
        with open(js_path, "w") as f:
            json.dump(d, f, indent=2)

    def _ensure_brain_file(self) -> bool:
        """models/brain.pt is a ~149MB artifact, not committed. Reassemble it
        from (1) loose part files, (2) any git ref, (3) GitHub raw download."""
        root = os.path.dirname(os.path.abspath(__file__))
        pt = os.path.join(root, "models", "brain.pt")
        js = os.path.join(root, "models", "brain.json")
        if os.path.exists(pt) and os.path.exists(js):
            return True
        # 1) loose part files in the repo folder
        for pattern, offset in self._PART_PATTERNS:
            parts, i = [], 0
            while i < 12:
                path = os.path.join(root, pattern % (i + offset))
                if not os.path.isfile(path) or os.path.getsize(path) < 1000:
                    break
                with open(path, "rb") as f:
                    parts.append(f.read())
                i += 1
            if len(parts) >= 2:
                self._write_brain(parts, js)
                self._add_log("Reassembled brain.pt from %d loose parts."
                              % len(parts))
                return True
        # 2) part files stored in git (any ref)
        import subprocess
        for ref in ("HEAD", "origin/arena/01a033e0-cs",
                    "arena/01a033e0-cs", "origin/main", "main"):
            for pattern, offset in self._PART_PATTERNS:
                parts, i = [], 0
                while i < 12:
                    try:
                        p = subprocess.run(
                            ["git", "show",
                             "%s:%s" % (ref, pattern % (i + offset))],
                            cwd=root, capture_output=True, timeout=120)
                    except Exception:
                        break
                    if p.returncode != 0 or len(p.stdout) < 1000:
                        break
                    parts.append(p.stdout)
                    i += 1
                if len(parts) >= 2:
                    self._write_brain(parts, js, ref=ref)
                    self._add_log("Reassembled brain.pt from %d git parts "
                                  "(%s)." % (len(parts), ref))
                    return True
        # 3) download the parts from GitHub raw (any machine with internet)
        try:
            import requests
            parts = []
            for i in range(12):
                url = ("https://raw.githubusercontent.com/alistra742-source/"
                       "cs/%s/brain_part_%02d" % (_BRAIN_SHA, i))
                r = requests.get(url, timeout=120)
                if r.status_code != 200 or len(r.content) < 1000:
                    break
                parts.append(r.content)
            if len(parts) >= 2:
                self._write_brain(parts, js)
                self._add_log("Downloaded brain.pt as %d parts from GitHub."
                              % len(parts))
                return True
            self._brain_error = ("downloaded only %d parts from GitHub"
                                 % len(parts))
        except Exception as exc:
            self._brain_error = "GitHub download failed: %s" % exc
        self._add_log("Could not obtain brain parts anywhere: %s"
                      % self._brain_error)
        return False

    # ── solver (lazy, preloaded at launch on a background thread) ────────
    def _get_solver(self):
        with self._solver_lock:
            return self._get_solver_locked()

    def _get_solver_locked(self):
        if self._solver is None and not self._solver_tried:
            self._solver_tried = True
            self._brain_state = "loading"
            try:
                self._ensure_brain_file()
                import brain as _brain
                if not _brain._TORCH:
                    self._brain_state = "failed"
                    self._brain_error = ("torch is not installed on this "
                                         "machine - run: pip install torch")
                    self._add_log("Brain needs torch: pip install torch")
                    return None
                self._solver = _brain.BrainSolver()
                if not self._solver.available:
                    self._solver = None
                    self._brain_state = "failed"
                    self._brain_error = self._brain_error or \
                        "models/brain.pt not loadable"
                    self._add_log("Brain not loadable: %s" % self._brain_error)
                else:
                    self._brain_state = "loaded"
                    self._add_log("Brain loaded (%.0f MB) - live solving ON."
                                  % (sum(p.numel() for p in
                                         self._solver.model.parameters())
                                     * 4 / 1e6))
            except Exception as exc:
                self._solver = None
                self._brain_state = "failed"
                self._brain_error = str(exc)
                self._add_log("Brain import failed: %s" % exc)
        return self._solver

    def _preload_solver(self):
        threading.Thread(target=self._get_solver, daemon=True).start()

    def begin_launch(self) -> dict:
        queued = super().begin_launch()
        if queued.get("ok"):
            self._preload_solver()
        return queued

    # ── OWN cycle for the target page: checkbox -> 15s challenge wait
    # -> (no challenge) Next -> 5s -> checkbox again -> loop. When a
    # challenge DOES appear, the Brain solves it (analyze -> click -> submit)
    # and the loop continues. Heavily logged at every step.
    async def _find_widget_iframe_simple(self, page):
        """Find the hCaptcha CHECKBOX widget iframe element (no frame
        resolution). The challenge modal is excluded by its URL
        (frame=challenge / hcaptcha-challenge) - NEVER by its title, because
        the widget's own title is 'Widget containing checkbox for hCaptcha
        security challenge', which contains the word 'challenge' (excluding
        on the title made the checkbox unfindable)."""
        try:
            locs = await page.locator("iframe").all()
        except Exception:
            return None
        best = None
        for loc in locs:
            try:
                src = (await loc.get_attribute("src") or "").lower()
                title = (await loc.get_attribute("title") or "").lower()
            except Exception:
                continue
            if "challenge" in src or "frame=challenge" in src:
                continue                      # the challenge modal (by URL)
            is_widget = ("checkbox" in title or "frame=checkbox" in src
                         or "hcaptcha" in src or "hcaptcha" in title)
            if not is_widget:
                continue
            try:
                box = await loc.bounding_box()
            except Exception:
                box = None
            if box and float(box.get("width") or 0) > 20 \
                    and float(box.get("height") or 0) > 20:
                if "checkbox" in title:
                    return loc                # the checkbox widget itself
                best = best or loc
        return best

    async def _click_checkbox_patient(self, page, timeout=60.0) -> bool:
        """Find the hCaptcha widget iframe and click its left-center (where
        the checkbox is drawn). Waits patiently, logging progress."""
        waited = 0.0
        while waited < timeout and not self._stopped():
            loc = await self._find_widget_iframe_simple(page)
            if loc is not None:
                # strategy 1: click #checkbox INSIDE the widget frame
                try:
                    frame = await self._resolve_hcaptcha_frame(
                        page, loc, want_checkbox=True)
                    if frame is not None:
                        cb = frame.locator('#checkbox, [role="checkbox"]')
                        if await cb.count() > 0:
                            await cb.first.click(timeout=5000)
                            self._add_log("CLICKED #checkbox inside the "
                                          "widget frame.")
                            return True
                except Exception:
                    pass
                # strategy 2: page-coordinate click on the iframe's checkbox
                # area (left side, vertically centred)
                try:
                    box = await loc.bounding_box()
                    if box and float(box.get("width") or 0) > 20:
                        cx = box["x"] + min(38.0, box["width"] * 0.2)
                        cy = box["y"] + box["height"] / 2.0
                        await page.mouse.move(cx - 80, cy - 50)
                        await asyncio.sleep(0.3)
                        await page.mouse.move(cx, cy)
                        await asyncio.sleep(0.25)
                        await page.mouse.click(cx, cy)
                        self._add_log("CLICKED the checkbox at (%.0f, %.0f)."
                                      % (cx, cy))
                        return True
                except Exception as exc:
                    self._add_log("Checkbox click failed (%s) - retrying..."
                                  % exc)
            if int(waited) % 10 == 0 and waited > 0:
                self._add_log("Still waiting for the hCaptcha checkbox "
                              "(%.0fs)..." % waited)
            await asyncio.sleep(2.5)
            waited += 2.5
        return False

    async def _click_next(self, page) -> bool:
        """Click the page's Next button."""
        for sel in ('button:has-text("Next")',
                    'input[type="submit"][value*="Next" i]',
                    'a:has-text("Next")',
                    '[role="button"]:has-text("Next")',
                    'button:has-text("next")',
                    '[class*="next" i] button'):
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.click(timeout=8000)
                    self._add_log("CLICKED the Next button (%s)." % sel)
                    return True
            except Exception:
                continue
        self._add_log("No Next button found on the page.")
        return False

    async def _wait_challenge(self, page, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stopped():
            iframe, frame = await self._find_frame(
                page, trainer.CHALLENGE_IFRAME_SELECTOR)
            if iframe is not None and frame is not None:
                return iframe, frame
            await asyncio.sleep(0.5)
        return None, None

    async def _do_real_cycle(self) -> str:
        page = self._page
        if page is None:
            return "stopped"
        with self._lock:
            self.total_cycles += 1

        # 1) open the target page and let it render. A goto timeout is NOT
        #    fatal: the page is often usable anyway, so continue to the
        #    checkbox instead of bouncing back to the previous page forever.
        self._set_state("navigating", "Opening the page...")
        try:
            await page.goto(TARGET_URL,
                            wait_until="domcontentloaded", timeout=60000)
        except Exception as exc:
            self._add_log("Page load was slow (%s) - continuing on whatever "
                          "is rendered..." % exc)
        self._add_log("Page opened. Waiting 6s for render...")
        await asyncio.sleep(6.0)

        # 2) the loop: click the checkbox -> wait for the challenge ->
        #    (no challenge) refresh the page and try again
        for step in range(20):
            if self._stopped():
                return "stopped"
            self._set_state("clicking_checkbox",
                            "Clicking the hCaptcha checkbox...")
            if not await self._click_checkbox_patient(page, timeout=60.0):
                self._add_log("Checkbox never appeared - refreshing the "
                              "page for a fresh attempt.")
                await self._refresh_page(page)
                continue

            self._set_state("waiting_for_challenge",
                            "Waiting for the challenge...")
            iframe, frame = await self._wait_challenge(page, timeout=15.0)
            if iframe is not None and frame is not None:
                # a challenge appeared: the Brain solves it
                self._set_state("solving",
                                "The Brain is solving the challenge...")
                image, question, full_text = await self._capture_challenge(
                    iframe, frame)
                await self._record_challenge(image, question, full_text)
                await asyncio.sleep(5.0)
                await self._wait_for_human_completion(page)
                continue        # back to the checkbox (loop)

            # no challenge within 15s -> refresh the page and retry
            self._add_log("No challenge in 15s - refreshing the page...")
            await self._refresh_page(page)
        return "success"

    async def _refresh_page(self, page) -> None:
        """Refresh the page (fallback: re-navigate) and wait for render."""
        try:
            await page.reload(timeout=45000)
        except Exception:
            try:
                await page.goto(TARGET_URL,
                                wait_until="domcontentloaded", timeout=60000)
            except Exception:
                pass
        await asyncio.sleep(6.0)

    # ── capture: tile boxes + keep the live frame for executing answers ──
    async def _capture_challenge(self, iframe, frame):
        self._challenge_iframe = iframe
        self._challenge_frame = frame
        image, question, full_text = await super()._capture_challenge(
            iframe, frame)
        self._tile_boxes = []
        try:
            raw = await frame.evaluate(_TILE_BOXES_JS)
            boxes = json.loads(raw) if raw else []
            if isinstance(boxes, list) and boxes:
                self._tile_boxes = boxes
                self._add_log("Found %d tile nodes in the live challenge."
                              % len(boxes))
        except Exception:
            pass
        return image, question, full_text

    # ── the Brain solves: analyze, report, then EXECUTE the answer ────────
    async def _record_challenge(self, image: str, question: str,
                                full_text: str) -> None:
        try:
            self._analyze(image, question)
        except Exception as exc:
            self._add_log("Brain analysis error: %s" % exc)
            self._record_round(question, "error",
                               "analysis failed: %s" % exc, 0.0, None)
            return
        res = self._last_res
        if res is None:
            self._add_log("Not confident enough to click - deferring "
                          "(no wrong clicks).")
            return
        ok = await self._execute_answer(res)
        if ok:
            with self._lock:
                self.executed_count += 1

    # ── execute: actually click / drag the Brain's answer on the page ─────
    async def _execute_answer(self, res) -> bool:
        iframe, frame, page = (self._challenge_iframe, self._challenge_frame,
                               self._page)
        if iframe is None or frame is None or page is None:
            return False
        fam = res.get("family")
        a = res.get("answer")
        try:
            if (fam == hct.BINARY or res.get("family") == hct.BINARY) \
                    and isinstance(a, list):
                locs = frame.locator('div.task-image, [class*="task-image" i]')
                n = await locs.count()
                self._add_log("Clicking the Brain's tiles %s (%d tile nodes "
                              "available)…" % (a, n))
                for idx in a:
                    if 0 < idx <= n:
                        await locs.nth(idx - 1).click(timeout=5000)
                        await asyncio.sleep(0.4)
                await self._click_submit(frame)
                return True
            if isinstance(a, dict) and "from" in a:
                box = await iframe.bounding_box()
                if not box:
                    return False
                fx = box["x"] + a["from"][0] * box["width"]
                fy = box["y"] + a["from"][1] * box["height"]
                tx = box["x"] + a["to"][0] * box["width"]
                ty = box["y"] + a["to"][1] * box["height"]
                self._add_log("Dragging the Brain's answer (%.0f,%.0f) -> "
                              "(%.0f,%.0f)…" % (fx, fy, tx, ty))
                await page.mouse.move(fx, fy)
                await asyncio.sleep(0.15)
                await page.mouse.down()
                for t in range(1, 13):
                    await page.mouse.move(fx + (tx - fx) * t / 12.0,
                                          fy + (ty - fy) * t / 12.0)
                    await asyncio.sleep(0.035)
                await asyncio.sleep(0.12)
                await page.mouse.up()
                await asyncio.sleep(0.5)
                await self._click_submit(frame)
                return True
            if isinstance(a, tuple) and len(a) == 2:
                box = await iframe.bounding_box()
                if not box:
                    return False
                px = box["x"] + a[0] * box["width"]
                py = box["y"] + a[1] * box["height"]
                self._add_log("Clicking the Brain's point (%.0f,%.0f)…"
                              % (px, py))
                await page.mouse.click(px, py)
                await asyncio.sleep(0.4)
                await self._click_submit(frame)
                return True
            self._add_log("Answer type has no click path yet (%s) - reported "
                          "only." % fam)
            return False
        except Exception as exc:
            self._add_log("Executing the answer failed: %s" % exc)
            return False

    async def _click_submit(self, frame) -> None:
        for sel in ('.button-submit', '[class*="submit" i]',
                    'button[type="submit"]'):
            try:
                btn = frame.locator(sel)
                if await btn.count() > 0:
                    await btn.first.click(timeout=5000)
                    self._add_log("Submitted the round.")
                    return
            except Exception:
                continue

    # ── analysis (pure function of the screenshot + prompt) ──────────────
    def _analyze(self, image_b64: str, question: str) -> None:
        from PIL import Image, ImageDraw
        solver = self._get_solver()
        self._last_res = None
        if solver is None:
            self._record_round(question, "unavailable",
                               "Brain not loaded: %s" % (self._brain_error
                                                         or "missing"), 0.0,
                              None)
            return

        im = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        W, H = im.size
        fam = hct.classify(prompt=question) or "unknown"
        learned = None
        try:
            rr = solver.router_predict(question)
            learned = (rr or {}).get("family")
        except Exception:
            learned = None

        tiles, tile_boxes_px = self._extract_tiles(im)
        res = None
        reason = ""
        if fam == hct.BINARY and tiles:
            probs = solver.probabilities(tiles)
            labels = [max(p, key=p.get) for p in probs] if probs else []
            confs = [p.get(l, 0.0) for p, l in zip(probs, labels)]
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            idx = hct.resolve_semantic(question, labels)
            if idx is None:
                reason = "prompt not understood offline"
            elif mean_conf < solver.min_conf:
                reason = ("low confidence (mean %.2f < gate %.2f)"
                          % (mean_conf, solver.min_conf))
            else:
                res = {"family": hct.BINARY, "answer": idx,
                       "confidence": mean_conf, "labels": labels}
        else:
            try:
                res = solver.solve(im, prompt=question,
                                   tiles=tiles if tiles else None)
            except Exception:
                res = None
            if res is None:
                reason = "not confident - would defer to the vision model"

        overlay = im.copy()
        draw = ImageDraw.Draw(overlay)
        answer_text, conf = self._format_answer(fam, res, reason)
        self._draw_overlay(draw, fam, res, W, H, tile_boxes_px)

        buf = io.BytesIO()
        overlay.save(buf, "PNG")
        ov_b64 = base64.b64encode(buf.getvalue()).decode()
        overlay_url = ""
        try:
            import live_control
            rec = live_control.save_challenge_png(
                ov_b64, question, kind="brain-test") or {}
            overlay_url = rec.get("url") or ""
        except Exception:
            pass
        with self._lock:
            if ov_b64:
                self.latest_screenshot = ov_b64
            if overlay_url:
                self.latest_challenge_image = overlay_url
        self._last_res = res
        self._record_round(question, fam, answer_text, conf, overlay_url,
                           learned=learned)

    def _extract_tiles(self, im):
        W, H = im.size
        boxes_px = []
        if self._tile_boxes:
            for b in self._tile_boxes[:9]:
                boxes_px.append((int(b["x"] * W), int(b["y"] * H),
                                 int((b["x"] + b["w"]) * W),
                                 int((b["y"] + b["h"]) * H)))
        else:
            x0, x1 = int(W * 0.02), int(W * 0.98)
            y0, y1 = int(H * 0.20), int(H * 0.98)
            cw, ch = (x1 - x0) / 3.0, (y1 - y0) / 3.0
            for r in range(3):
                for c in range(3):
                    boxes_px.append((int(x0 + c * cw), int(y0 + r * ch),
                                     int(x0 + (c + 1) * cw),
                                     int(y0 + (r + 1) * ch)))
        tiles = []
        for (x0, y0, x1, y1) in boxes_px:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            tiles.append(im.crop((x0, y0, x1, y1)))
        return tiles, boxes_px

    def _format_answer(self, fam, res, reason):
        if res is None:
            return ("DEFERRED: %s" % (reason or "not confident"), 0.0)
        conf = float(res.get("confidence") or 0.0)
        a = res.get("answer")
        if fam == hct.BINARY or res.get("family") == hct.BINARY:
            labels = res.get("labels") or []
            lab = ", ".join(labels[i - 1] for i in a if 0 < i <= len(labels))
            return ("click tiles %s  (%s)" % (a, lab), conf)
        if isinstance(a, dict) and "from" in a:
            return ("drag (%.2f, %.2f) -> (%.2f, %.2f)"
                    % (a["from"][0], a["from"][1], a["to"][0], a["to"][1]),
                    conf)
        if isinstance(a, dict) and "w" in a:
            return ("box x=%.2f y=%.2f w=%.2f h=%.2f"
                    % (a["x"], a["y"], a["w"], a["h"]), conf)
        if isinstance(a, tuple):
            return ("click at (%.2f, %.2f)  [%s]"
                    % (a[0], a[1], res.get("label") or ""), conf)
        return ("answer: %s" % a, conf)

    def _draw_overlay(self, draw, fam, res, W, H, tile_boxes_px):
        if res is None:
            return
        a = res.get("answer")
        green, red = (52, 211, 153), (248, 113, 113)
        if (fam == hct.BINARY or res.get("family") == hct.BINARY) \
                and isinstance(a, list) and tile_boxes_px:
            for i in a:
                if 0 < i <= len(tile_boxes_px):
                    x0, y0, x1, y1 = tile_boxes_px[i - 1]
                    for w in range(4):
                        draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w],
                                       outline=green)
        elif isinstance(a, dict) and "from" in a:
            fx, fy = a["from"][0] * W, a["from"][1] * H
            tx, ty = a["to"][0] * W, a["to"][1] * H
            draw.line([fx, fy, tx, ty], fill=red, width=4)
            for (px, py) in ((fx, fy), (tx, ty)):
                draw.ellipse([px - 10, py - 10, px + 10, py + 10],
                             outline=red, width=3)
        elif isinstance(a, dict) and "w" in a:
            draw.rectangle([a["x"] * W, a["y"] * H,
                            (a["x"] + a["w"]) * W, (a["y"] + a["h"]) * H],
                           outline=red, width=4)
        elif isinstance(a, tuple):
            px, py = a[0] * W, a[1] * H
            draw.line([px - 14, py, px + 14, py], fill=red, width=3)
            draw.line([px, py - 14, px, py + 14], fill=red, width=3)
            draw.ellipse([px - 10, py - 10, px + 10, py + 10],
                         outline=red, width=3)

    def _record_round(self, question, fam, answer_text, conf, overlay_url,
                      learned=None):
        deferred = str(answer_text).startswith("DEFERRED")
        with self._lock:
            self.answered_count += 0 if deferred else 1
            self.deferred_count += 1 if deferred else 0
            self.rounds.append({
                "id": len(self.rounds) + 1,
                "t": time.strftime("%H:%M:%S"),
                "question": question,
                "family": fam,
                "learned_router": learned,
                "answer": answer_text,
                "confidence": round(float(conf or 0.0), 3),
                "deferred": deferred,
                "overlay_url": overlay_url or "",
            })
            if len(self.rounds) > 60:
                self.rounds = self.rounds[-60:]
            tag = "DEFERRED" if deferred else "answered"
            self._add_log("Brain %s #%d [%s]: %s"
                          % (tag, len(self.rounds), fam, answer_text))

    # ── after executing the answer, give hCaptcha time to react, then
    #    cycle to the next challenge automatically ─────────────────────────
    async def _wait_for_human_completion(self, page):
        try:
            return await asyncio.wait_for(
                super()._wait_for_human_completion(page), timeout=30.0)
        except asyncio.TimeoutError:
            self._add_log("Cycling to the next challenge…")
            return "success"

    def get_state(self) -> dict:
        st = super().get_state()
        st["cycles_count"] = st.pop("pass_count", 0)
        st["target_url"] = TARGET_URL
        st.update({
            "mode": "brain-test",
            "brain_state": self._brain_state,
            "brain_error": self._brain_error,
            "brain_loaded": self._brain_state == "loaded",
            "answered_count": self.answered_count,
            "deferred_count": self.deferred_count,
            "executed_count": self.executed_count,
            "rounds": [dict(r) for r in self.rounds[-25:]],
        })
        return st


brain_engine = BrainTestEngine()

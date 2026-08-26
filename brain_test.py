#!/usr/bin/env python3
"""
brain_test.py — the TEST tab engine: live hCaptcha, solved by the Brain.

Drives the official hCaptcha demo (accounts.hcaptcha.com/demo) in the shared
LIVE browser — checkbox, challenge, camera, the works — but instead of a
human answering, the BRAIN (models/brain.pt, brain.py BrainSolver) analyzes
every challenge round and reports exactly what it would answer:

  - the routed challenge family (from the real prompt text)
  - the Brain's answer (tile indices + labels / click point / drag from-to /
    box / count / text code) and its confidence
  - an overlay image showing WHERE the Brain would click
  - an honest "deferred (not confident)" when it would fall back to vision

It never trains and never collects samples: read-only analysis of live
rounds. Built as a TrainerEngine subclass so the whole battle-tested browser
flow (launch, checkbox, challenge capture, reload) is reused verbatim.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from typing import Any, Dict, List, Optional

import trainer
from trainer import TrainerEngine

import hcaptcha_types as hct

# Tile-node boxes as fractions of the challenge-iframe viewport. Returned by
# evaluating this inside the live challenge frame; the iframe screenshot maps
# 1:1 onto that viewport, so the fractions crop tiles exactly.
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
    """Live hCaptcha demo rounds answered by the Brain (analysis only)."""

    def __init__(self):
        super().__init__()
        self._solver = None
        self._solver_tried = False
        self._solver_lock = threading.Lock()
        self._brain_state = "not-loaded"   # not-loaded/loading/loaded/failed
        self._tile_boxes: List[Dict[str, float]] = []
        self.rounds: List[Dict[str, Any]] = []
        self.answered_count = 0
        self.deferred_count = 0

    # ── solver (lazy: torch/brain import only when first needed) ─────────
    def _get_solver(self):
        with self._solver_lock:
            return self._get_solver_locked()

    def _get_solver_locked(self):
        if self._solver is None and not self._solver_tried:
            self._solver_tried = True
            self._brain_state = "loading"
            try:
                import brain as _brain
                self._solver = _brain.BrainSolver()
                if not self._solver.available:
                    self._solver = None
                    self._brain_state = "failed"
                    self._add_log("models/brain.pt not loadable - train the "
                                  "Brain first (`python brain.py train`).")
                else:
                    self._brain_state = "loaded"
                    self._add_log("Brain loaded (%.0f MB) - live solving ON."
                                  % (sum(p.numel() for p in
                                         self._solver.model.parameters())
                                     * 4 / 1e6))
            except Exception as exc:
                self._solver = None
                self._brain_state = "failed"
                self._add_log("Brain import failed: %s" % exc)
        return self._solver

    def _preload_solver(self):
        """Load torch + the 149MB brain OFF the event loop, at launch time,
        so the first challenge analysis never stalls the browser flow."""
        threading.Thread(target=self._get_solver, daemon=True).start()

    def begin_launch(self) -> dict:
        queued = super().begin_launch()
        if queued.get("ok"):
            self._preload_solver()
        return queued

    # ── diagnostics: see exactly why form/checkbox steps fail ────────────
    async def _do_real_cycle(self) -> str:
        try:
            self._add_log("Cycle start: %s" % (self._page.url or "?"))
        except Exception:
            pass
        return await super()._do_real_cycle()

    async def _fill_demo_field(self, page, value: str) -> bool:
        ok = await super()._fill_demo_field(page, value)
        if ok:
            self._add_log("Demo form filled with sample text (%d chars)."
                          % len(value))
            await asyncio.sleep(1.2)     # let the camera capture the fill
            return True
        try:
            info = await page.evaluate(
                "() => JSON.stringify(Array.from("
                "document.querySelectorAll('input, textarea')).slice(0, 12)"
                ".map(e => ({tag: e.tagName, type: e.type || '', "
                "name: e.name || '', id: e.id || '', "
                "vis: !!(e.offsetWidth || e.offsetHeight)})))")
            self._add_log("DEMO FORM FIELD NOT FOUND. Inputs on the page: %s"
                          % info)
        except Exception as exc:
            self._add_log("DEMO FORM FIELD NOT FOUND (probe failed: %s)" % exc)
        return False

    async def _wait_for_checkbox(self, page, timeout: float = 60.0):
        # longer than the trainer's 35s: the widget can be slow behind proxies
        return await super()._wait_for_checkbox(page, timeout=timeout)

    async def _click_real_checkbox(self, page) -> bool:
        try:
            frames_info = await page.evaluate(
                "() => JSON.stringify(Array.from("
                "document.querySelectorAll('iframe')).slice(0, 10)"
                ".map(f => ({src: (f.src || '').slice(0, 90), "
                "w: f.offsetWidth, h: f.offsetHeight})))")
            self._add_log("Page iframes: %s" % frames_info)
        except Exception:
            pass
        ok = await super()._click_real_checkbox(page)
        if not ok:
            self._add_log(
                "CHECKBOX NOT CLICKED. If no hcaptcha iframe appears in the "
                "list above, the hCaptcha script did not load (check network/"
                "proxy) or the widget is still rendering - retrying the cycle.")
        return ok

    # ── capture: also grab exact tile boxes from the live DOM ────────────
    async def _capture_challenge(self, iframe, frame):
        image, question, full_text = await super()._capture_challenge(
            iframe, frame)
        self._tile_boxes = []
        try:
            raw = await frame.evaluate(_TILE_BOXES_JS)
            boxes = json.loads(raw) if raw else []
            if isinstance(boxes, list) and boxes:
                self._tile_boxes = boxes
        except Exception:
            pass
        return image, question, full_text

    # ── THE BRAIN ANSWERS HERE ────────────────────────────────────────────
    async def _record_challenge(self, image: str, question: str,
                                full_text: str) -> None:
        try:
            self._analyze(image, question)
        except Exception as exc:
            self._add_log("Brain analysis error: %s" % exc)
            self._record_round(question, "error",
                               "analysis failed: %s" % exc, 0.0, None)

    def _analyze(self, image_b64: str, question: str) -> None:
        from PIL import Image, ImageDraw
        solver = self._get_solver()
        if solver is None:
            self._record_round(question, "unavailable",
                               "Brain not loaded (models/brain.pt missing)",
                               0.0, None)
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
        self._record_round(question, fam, answer_text, conf, overlay_url,
                           learned=learned)

    def _extract_tiles(self, im):
        """Crop the 3x3 tile grid: exact DOM boxes when available, else a
        geometric split of the challenge area below the prompt bar."""
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
                tiles.append(None)
                continue
            tiles.append(im.crop((x0, y0, x1, y1)))
        tiles = [t for t in tiles if t is not None]
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

    # ── cycle control: show the answer, then move to the next challenge ──
    async def _wait_for_human_completion(self, page):
        """The Brain has already answered (reported in the Test tab). Give the
        operator a window to complete the round manually via the LIVE camera
        if they want; otherwise cycle to the next challenge automatically."""
        try:
            return await asyncio.wait_for(
                super()._wait_for_human_completion(page), timeout=18.0)
        except asyncio.TimeoutError:
            self._add_log("Cycling to the next challenge…")
            return "success"

    def get_state(self) -> dict:
        st = super().get_state()
        # pass_count here counts challenge CYCLES, not solved passes - the
        # Brain only reports answers, it does not click them.
        st["cycles_count"] = st.pop("pass_count", 0)
        st.update({
            "mode": "brain-test",
            "brain_state": self._brain_state,
            "brain_loaded": self._brain_state == "loaded",
            "answered_count": self.answered_count,
            "deferred_count": self.deferred_count,
            "rounds": [dict(r) for r in self.rounds[-25:]],
        })
        return st


brain_engine = BrainTestEngine()

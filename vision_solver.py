#!/usr/bin/env python3
"""vision_solver.py — local/remote vision solver for the hCaptcha image grid.

Replaces paid token APIs with a vision model you own and run. Instead of minting a token server-side, the bot:

  1. reads the challenge prompt ("Please select all images with a boat") from
     the hCaptcha challenge frame,
  2. screenshots every tile of the image grid,
  3. sends the prompt + tile images to a vision model,
  4. the model answers with the tile numbers that satisfy the task,
  5. the bot clicks those tiles + Verify (see server.py's
     ``_solve_hcaptcha_if_present``), and hCaptcha itself mints the token.

Configuration (env vars):

  VISION_API_BASE  base URL of the vision API endpoint. Overrides
                   OLLAMA_BASE when set. Points to an Ollama-compatible API
                   (default: http://localhost:11434 — the fallback when
                   neither VISION_API_BASE nor OLLAMA_BASE is set).
  OLLAMA_BASE      legacy alias — used only when VISION_API_BASE is empty.
  VISION_API_KEY   Bearer token for authenticated vision endpoints
                   (optional — added as ``Authorization: Bearer <key>``
                   header to every request when set).
  OLLAMA_MODEL     vision model to use (default qwen3-vl:2b)
  OLLAMA_TIMEOUT   per-request timeout in seconds (default 180)
  VISION_CHECK_TIMEOUT  reachability-probe timeout in seconds (default 60 —
                        hosted endpoints cold-start on first request)

Model recommendation (small, better than Moondream):

  qwen3-vl:2b       1.9 GB  ← default; newest Qwen vision model, far stronger
                             object recognition + OCR than Moondream, handles
                             multiple images in one prompt. Needs Ollama ≥ 0.12.7.
  qwen2.5vl:3b      3.2 GB  ← most proven small vision model; a bit more
                             accurate on hard grids if you have the room.
  granite3.2-vision:2b  2.4 GB  ← document/OCR-focused alternative.

There is no Ollama vision model under ~1 GB that is actually BETTER than
Moondream (Moondream itself is 1.7 GB) — qwen3-vl:2b at 1.9 GB is the
smallest model that clears that bar.

The client talks to Ollama's native HTTP API (POST /api/chat) with the
``format: json`` flag so answers come back structured, and falls back to
loose parsing when a model ignores the flag. Only aiohttp is used — no new
dependencies.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Callable, List, Optional

import aiohttp

# ── VISION_API_BASE is the canonical env var.  OLLAMA_BASE is the legacy
# fallback when only the old name is set.  Neither is required: the default
# http://localhost:11434 serves the local-Ollama development workflow.
VISION_API_BASE = os.environ.get("VISION_API_BASE", "").rstrip("/")
_OLLAMA_BASE_LEGACY = os.environ.get("OLLAMA_BASE", "").rstrip("/")
OLLAMA_BASE = VISION_API_BASE or _OLLAMA_BASE_LEGACY or "http://localhost:11434"

VISION_API_KEY = os.environ.get("VISION_API_KEY", "").strip()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:2b").strip()
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
# Reachability probe timeout (seconds). Hosted endpoints (Railway etc.)
# COLD-START on the first request after sleeping — the wake-up itself can
# take 30-90s, so a 10s probe marks a healthy service permanently down.
VISION_CHECK_TIMEOUT = float(os.environ.get("VISION_CHECK_TIMEOUT", "60"))

# Instruct the model to answer in reading order: image 1 = top-left tile,
# then left→right, top→bottom. JSON-only output (no markdown, no prose).
_SYSTEM_PROMPT = (
    "You are a precise image-selection solver for an hCaptcha challenge grid. "
    "You are given the challenge instruction and one image per grid tile, in "
    "reading order: image 1 is the top-left tile, image 2 is the tile to its "
    "right, and so on left-to-right, top-to-bottom. "
    "Look at EVERY tile carefully and decide which ones satisfy the instruction. "
    "THE ANSWER IS A SET: select EVERY tile that satisfies the instruction - "
    "the correct answer is OFTEN SEVERAL tiles, sometimes just ONE, and "
    "sometimes NONE. Never stop at the first match; check all tiles before "
    "answering, and do not return a single tile when several match. "
    "For attribute/material prompts ('select items that are primarily metal', "
    "'made of wood', 'has fur', 'is red', 'transparent', ...), judge the "
    "DOMINANT material or attribute of each tile's MAIN subject: a tile "
    "counts when its main subject is primarily that material / has that "
    "attribute. A butterfly is NOT 'primarily metal' even if it sits on a "
    "metal surface; a solid metal object IS. Judge the object itself, not "
    "the background. "
    "For visual-comparison prompts such as 'click the two elements/images that "
    "are identical', 'the same', 'matching', 'duplicates', 'similar', or 'most similar', "
    "compare all tiles against each other and return the matching tile numbers. "
    'Answer with ONLY a JSON object, never any other text: '
    '{"tiles": [1, 3, 7]} for a grid selection, or {"answer": "the text"} '
    "if the challenge asks you to type characters instead. "
    'Use [] when no tile matches. Tile numbers must be integers.'
)

# ── per-answer-shape system prompts (one hCaptcha family each) ───────────
# All coordinates are normalised 0.0-1.0, origin TOP-LEFT.

_SYSTEM_POINT = (
    "You are a precise point solver for an hCaptcha challenge. You are given "
    "ONE canvas image and an instruction asking you to click something in it "
    '(e.g. "click on the animal who jumps the highest"). '
    "Decide where the target is and answer with ONLY a JSON object: "
    '{"points": [[x, y]]} where x and y are NORMALISED coordinates between '
    "0.0 and 1.0 (x = fraction from the left edge, y = fraction from the top "
    "edge). Give one point at the CENTRE of the requested object unless the "
    "instruction asks for multiple targets. For prompts like 'click on the two "
    "elements that are identical', 'same', 'matching', 'duplicates', 'similar', or 'most "
    "similar', return one centre point for EACH requested matching element."
)

_SYSTEM_BBOX = (
    "You are a precise bounding-box solver for an hCaptcha challenge. You are "
    "given ONE canvas image and an instruction asking you to draw a box around "
    'something (e.g. "draw a box around the cat\'s head"). '
    "Answer with ONLY a JSON object: "
    '{"bbox": {"x1": a, "y1": b, "x2": c, "y2": d}} where all values are '
    "NORMALISED 0.0-1.0 fractions of the image size (origin top-left), "
    "x1 < x2 and y1 < y2, hugging the target tightly."
)

_SYSTEM_DRAG = (
    "You are a precise drag-and-drop solver for an hCaptcha challenge. You are "
    "given ONE canvas image containing a loose puzzle element (usually with a "
    '"Move" badge) and a matching empty slot silhouette. Answer with ONLY a '
    "JSON object: "
    '{"drag": {"from": [x1, y1], "to": [x2, y2]}} where "from" is the CENTRE '
    'of the draggable element and "to" is the CENTRE of the slot it fits '
    "into, both NORMALISED 0.0-1.0 fractions of the image size (origin "
    "top-left)."
)

_SYSTEM_STACK = (
    "You are a precise stacking-puzzle solver for a FunCAPTCHA/Arkose "
    "challenge. You are given ONE canvas image showing vertical stacks "
    "(columns/towers) of blocks and one or more loose draggable blocks. The "
    "goal is to drag blocks onto the stacks so that EVERY stack ends up with "
    "the SAME height (the number of blocks the puzzle demands — e.g. every "
    'column 3 blocks tall). Answer with ONLY a JSON object: {"drags": '
    "[[sx, sy, tx, ty], ...]} listing EVERY drag needed, in execution order. "
    "sx, sy = the CENTRE of the block to grab; tx, ty = the point ON TOP of "
    "the target stack where it must be dropped. All four numbers are integer "
    "PERCENTAGES of the image dimensions (0-100, sx/tx from the LEFT edge, "
    'sy/ty from the TOP edge). Example: {"drags": [[12, 80, 12, 40], '
    "[88, 82, 50, 55]]}"
)

_SYSTEM_CHOICE = (
    "You are a precise multiple-choice solver for an hCaptcha challenge. You "
    "are given an instruction, a list of numbered answer options and one "
    "reference image. Pick the single most accurate option and answer with "
    'ONLY a JSON object: {"choice": N} where N is the 1-based option number '
    "(integer)."
)

_SYSTEM_COUNT = (
    "You are a precise counting solver for an hCaptcha challenge. You are "
    "given ONE image and an instruction asking how many of something appear "
    'in it. Count carefully and answer with ONLY a JSON object: '
    '{"count": N} where N is the integer count (e.g. {"count": 3}).'
)

_SYSTEM_PATTERN = (
    "You are a precise pattern-completion solver for an hCaptcha drag "
    "challenge. You are given ONE image containing a grid of icons with "
    "ONE empty cell and a row of candidate icons. Determine which "
    "candidate completes the pattern (every row and every column of the "
    "grid must contain distinct icons — check both directions). Answer "
    "with ONLY a JSON object: "
    '{"drag": {"from": [x1, y1], "to": [x2, y2]}} where "from" is the '
    'CENTRE of the CORRECT candidate icon and "to" is the CENTRE of the '
    "empty cell, both NORMALISED 0.0-1.0 fractions of the image size "
    "(origin top-left)."
)

_SYSTEM_TOWER = (
    "You are a precise drag-and-drop solver for an hCaptcha wooden-block "
    "tower challenge. You are given ONE image showing several vertical "
    "wooden-block towers and a loose block SEGMENT (usually on the right, "
    "often with a Move badge). The instruction is to move the missing "
    "segment onto the INCOMPLETE tower — the stack that is shorter than "
    "the others, or that has a gap / missing block. Answer with ONLY a "
    "JSON object: "
    '{"drag": {"from": [x1, y1], "to": [x2, y2]}} where "from" is the '
    "CENTRE of the loose/Move piece and \"to\" is the drop point ON the "
    "incomplete tower (the gap, or the top of the shortest stack). "
    "Coordinates are NORMALISED 0.0-1.0 fractions of the image size "
    "(origin top-left). Do NOT click a finished tower. Do NOT treat this "
    "as a point-click — the piece must be dragged."
)

_SYSTEM_BY_SHAPE = {
    "tiles": _SYSTEM_PROMPT,
    "points": _SYSTEM_POINT,
    "bbox": _SYSTEM_BBOX,
    "drag": _SYSTEM_DRAG,
    "pattern": _SYSTEM_PATTERN,
    "tower": _SYSTEM_TOWER,
    "stack": _SYSTEM_STACK,
    "choice": _SYSTEM_CHOICE,
    "count": _SYSTEM_COUNT,
    "text": _SYSTEM_PROMPT,
}

_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]")
_JSON_STRING_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')


class OllamaVisionClient:
    """Async client for a vision model endpoint (Ollama or authenticated gateway).

    Connects to the endpoint defined by the env vars:
      - ``VISION_API_BASE`` (canonical) or ``OLLAMA_BASE`` (legacy)
      - ``VISION_API_KEY`` for Bearer auth when talking to an authenticated gateway
    """

    def __init__(self, log: Optional[Callable] = None,
                 base: str = OLLAMA_BASE, model: str = OLLAMA_MODEL):
        self._log = log or (lambda msg, level="info": None)
        self.base = base.rstrip("/")
        self.model = model or "qwen3-vl:2b"
        self._api_key = VISION_API_KEY
        self.stats = {"calls": 0, "ok": 0, "failed": 0}
        # Machine-readable result of the latest reachability probe.  Keep the
        # public ``check() -> (ok, models)`` contract for existing callers,
        # while letting them distinguish a transient connection failure from
        # deterministic configuration errors such as HTTP 401 or 404.
        self.last_check_error = ""
        self.last_check_http_status: Optional[int] = None
        # Log which env var supplied the base, so diagnostics are clear.
        src = "VISION_API_BASE" if os.environ.get("VISION_API_BASE", "").strip() else \
              ("OLLAMA_BASE" if os.environ.get("OLLAMA_BASE", "").strip() else "default")
        self._log(f"[Vision] API endpoint: {self.base} (from {src})"
                  f"{' [authenticated]' if self._api_key else ''}")

    @property
    def configured(self) -> bool:
        return bool(self.base)

    async def _headers(self) -> dict:
        """HTTP headers including optional Bearer auth."""
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def check(self) -> tuple:
        """Probe the Ollama server and list pulled models.

        Returns ``(ok, models)`` — ``models`` is a list of model names
        (including tags) or [] when the probe fails.  ``last_check_error``
        classifies failures so callers do not mistake an HTTP 401 (the server
        is up, but the credentials differ) for a cold start and retry it.
        """
        self.last_check_error = ""
        self.last_check_http_status = None
        try:
            timeout = aiohttp.ClientTimeout(total=VISION_CHECK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(f"{self.base}/api/tags",
                                 headers=await self._headers()) as r:
                    if r.status != 200:
                        self.last_check_http_status = int(r.status)
                        if r.status == 401:
                            self.last_check_error = "authentication"
                            key_state = ("the configured VISION_API_KEY was rejected"
                                         if self._api_key else
                                         "VISION_API_KEY is not configured in this service")
                            self._log(
                                f"[Vision] Authentication failed at {self.base} "
                                f"(HTTP 401): {key_state}", level="error")
                        elif r.status == 403:
                            self.last_check_error = "authorization"
                            self._log(
                                f"[Vision] Access forbidden at {self.base} (HTTP 403)",
                                level="error")
                        elif r.status == 404:
                            self.last_check_error = "protocol"
                            self._log(
                                f"[Vision] {self.base} does not expose the expected "
                                "Ollama /api/tags endpoint (HTTP 404)", level="error")
                        elif r.status == 429:
                            self.last_check_error = "rate_limit"
                            self._log(f"[Vision] /api/tags rate limited (HTTP 429)",
                                      level="warn")
                        elif r.status >= 500:
                            self.last_check_error = "server"
                            self._log(f"[Vision] /api/tags server error HTTP {r.status}",
                                      level="warn")
                        else:
                            self.last_check_error = "http"
                            self._log(f"[Vision] /api/tags HTTP {r.status}", level="warn")
                        return False, []
                    try:
                        data = await r.json()
                    except Exception as e:
                        self.last_check_error = "protocol"
                        self._log(
                            f"[Vision] /api/tags returned invalid JSON: "
                            f"{type(e).__name__}", level="error")
                        return False, []
            if not isinstance(data, dict) or not isinstance(data.get("models", []), list):
                self.last_check_error = "protocol"
                self._log("[Vision] /api/tags returned an invalid model-list payload",
                          level="error")
                return False, []
            models = [m.get("name") or m.get("model") or ""
                      for m in data.get("models", []) if isinstance(m, dict)]
            models = [m for m in models if m]
            self._log(f"[Ollama] Server OK at {self.base} ({len(models)} models pulled)")
            return True, models
        except asyncio.TimeoutError:
            self.last_check_error = "timeout"
            self._log(
                f"[Vision] Probe timed out after {VISION_CHECK_TIMEOUT:g}s at {self.base}",
                level="error")
            return False, []
        except aiohttp.ClientError as e:
            self.last_check_error = "connection"
            self._log(
                f"[Vision] Connection failed at {self.base}: {type(e).__name__}",
                level="error")
            return False, []
        except Exception as e:
            self.last_check_error = "connection"
            self._log(f"[Vision] Not reachable at {self.base}: {type(e).__name__}: {e}",
                      level="error")
            return False, []

    async def solve(self, prompt: str, images: List[bytes],
                    shape: str = "tiles", examples: Optional[List[bytes]] = None,
                    timeout: float = OLLAMA_TIMEOUT) -> Optional[dict]:
        """Ask the model to answer an hCaptcha round.

        ``images`` are the answer surfaces as PNG/JPEG bytes. For
        shape="tiles" they are the grid tiles in reading order (tile 1 =
        top-left); for "points"/"bbox"/"drag" it is the ONE big challenge
        canvas; for "choice" the reference image(s). ``examples`` are the
        prompt-header reference images (the "item shown"): they are
        PREPENDED to the message and the text explains which is which.

        Returns, by shape:

          tiles     {"type": "tiles",  "indices": [1, 3, 7]}
          points    {"type": "points", "points": [(x, y)]}         (0-1)
          bbox      {"type": "bbox",   "bbox": {x1,y1,x2,y2}}      (0-1)
          drag      {"type": "drag",   "from": (x,y), "to": (x,y)} (0-1)
          choice    {"type": "choice", "index": 2}                 (1-based)
          count     {"type": "count",  "count": 3}
          pattern   {"type": "drag",   "from": (x,y), "to": (x,y)} (0-1;
                    candidate centre -> empty-cell centre)
          stack     {"type": "drags",  "drags": [(sx, sy, tx, ty), ...]}
                    (0-100 percent coords; Arkose block-stacking plan)
          text      {"type": "text",   "text": "abc123"}
          None      model unreachable or answer unparseable
        """
        if not self.configured:
            self._log("[Ollama] No vision API base configured (set VISION_API_BASE or OLLAMA_BASE)", level="error")
            return None
        if not images:
            return None
        # Log solving with si vision (vision service / vision_solver)
        import time
        self._log(f"{time.strftime('%b %d %H:%M:%S.000 [info]')} solving with si vision")
        self.stats["calls"] += 1
        examples = list(examples or [])
        system = _SYSTEM_BY_SHAPE.get(shape, _SYSTEM_PROMPT)
        if examples:
            content = (
                f"REFERENCE IMAGES: the first {len(examples)} image(s) come "
                "from the challenge header and SHOW WHAT TO LOOK FOR — they "
                "are examples, NOT answer choices.\n"
                f"ANSWER SURFACES: the remaining {len(images)} image(s) are "
                "what you answer on (tiles in reading order, or the big "
                "canvas).\n"
                f"CHALLENGE TASK: {prompt}\n\n"
                "Answer with the JSON object only.")
        else:
            content = (f"CHALLENGE TASK: {prompt}\n\n"
                       f"There are {len(images)} image(s). "
                       "Answer with the JSON object only.")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": content,
                    "images": [base64.b64encode(b).decode("ascii")
                               for b in list(examples) + list(images)],
                },
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 256,
            },
            "keep_alive": "10m",
        }
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_cfg) as s:
                async with s.post(f"{self.base}/api/chat", json=payload,
                                  headers=await self._headers()) as r:
                    if r.status != 200:
                        body = await r.text()
                        self._log(
                            f"[Ollama] Solve rejected (HTTP {r.status}): {body[:200]}",
                            level="warn")
                        self.stats["failed"] += 1
                        return None
                    data = await r.json()
            content = ((data or {}).get("message") or {}).get("content") or ""
            if not content:
                self._log("[Ollama] Empty response from model", level="warn")
                self.stats["failed"] += 1
                return None
            parse_shape = "drag" if shape in ("pattern", "tower") else shape
            parsed = self._parse_geometry(content, parse_shape, len(images))
            if parsed is None and shape in ("tiles", "text", "count", "stack"):
                parsed = self._parse_answer(content, len(images), shape)
            if parsed is None:
                self._log(f"[Ollama] Unparseable model answer: {content[:160]}",
                          level="warn")
                self.stats["failed"] += 1
                return None
            self.stats["ok"] += 1
            return parsed
        except Exception as e:
            self._log(f"[Ollama] Solve error: {e}", level="error")
            self.stats["failed"] += 1
            return None

    # ── geometry answer parsing (points / bbox / drag / choice) ──────────

    @staticmethod
    def _loads_repaired(text: str):
        """json.loads with the repairs small vision models always need:
        bare-dot decimals (".8" -> "0.8"), trailing commas, ```json fences,
        prose around the object."""
        text = (text or "").strip()
        if not text:
            return None

        def repair(t):
            t = re.sub(r"(?<=[:\[,\s])\.(\d)", r"0.\1", t)   # .8 -> 0.8
            t = re.sub(r",\s*([}\]])", r"\1", t)             # trailing comma
            return t

        candidates = [text]
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.I)
        if fence:
            candidates.insert(0, fence.group(1).strip())
        lo, hi = text.find("{"), text.rfind("}")
        if 0 <= lo < hi:
            candidates.append(text[lo:hi + 1])
        lo, hi = text.find("["), text.rfind("]")
        if 0 <= lo < hi:
            candidates.append(text[lo:hi + 1])
        for cand in candidates:
            for variant in (cand, repair(cand)):
                try:
                    return json.loads(variant)
                except Exception:
                    continue
        return None

    @staticmethod
    def _num(v):
        """Coerce a coordinate: 0-1 floats stay, 0-100 read as percents,
        larger ("pixel-ish") values map through a nominal 500 px canvas.
        Always clamped to 0..1."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN
            return None
        if f < 0:
            f = 0.0
        if f > 100.0:
            f = f / 500.0
        elif f > 1.0:
            f = f / 100.0
        return max(0.0, min(1.0, f))

    @staticmethod
    def _point(p):
        """[x, y] / {"x","y"} / {"left","top"} / {"cx","cy"} -> (x, y) 0-1."""
        if isinstance(p, dict):
            for kx, ky in (("x", "y"), ("cx", "cy"), ("left", "top")):
                if kx in p and ky in p:
                    a, b = OllamaVisionClient._num(p[kx]), \
                        OllamaVisionClient._num(p[ky])
                    if a is not None and b is not None:
                        return (a, b)
            return None
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            a, b = OllamaVisionClient._num(p[0]), OllamaVisionClient._num(p[1])
            if a is not None and b is not None:
                return (a, b)
        return None

    @staticmethod
    def _parse_geometry(content: str, shape: str, tile_count: int = 0):
        """Parse points/bbox/drag/choice/tiles answers into structured form.

        Accepts the key variants every small model invents:
        points/point/clicks/coordinates, bbox/box/bounding_box (dict or
        4-list), drag/drags/path or a bare from/to pair, choice/option/
        answer_index, tiles/indices."""
        obj = OllamaVisionClient._loads_repaired(content)
        if obj is None:
            return None
        num = OllamaVisionClient._num
        pt = OllamaVisionClient._point

        # Stacking plans (FunCAPTCHA/Arkose "make every column equal"): the
        # generic branches below would squash a multi-drag {"drags": [...]}
        # into one drag / a points pair, so parse the FULL plan first.
        if shape == "stack":
            return OllamaVisionClient._parse_stack_geometry(obj)

        # bare top-level atoms
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if shape == "choice":
                return {"type": "choice", "index": int(round(obj))}
            if shape == "count":
                return {"type": "count", "count": int(round(obj))}
        if isinstance(obj, list):
            if obj and all(isinstance(x, (int, float))
                           and not isinstance(x, bool) for x in obj):
                if shape in ("points",):
                    p = pt(obj)
                    return {"type": "points", "points": [p]} if p else None
                return {"type": "tiles",
                        "indices": [int(x) for x in obj]}
            if obj and all(isinstance(x, (list, tuple, dict)) for x in obj):
                pts = [q for q in (pt(x) for x in obj) if q]
                if shape == "drag" and len(pts) >= 2:
                    return {"type": "drag", "from": pts[0], "to": pts[-1]}
                if pts:
                    return {"type": "points", "points": pts}
            return None
        if not isinstance(obj, dict):
            return None

        def first(keys):
            for k in keys:
                if k in obj and obj[k] is not None:
                    return obj[k]
            return None

        # tiles / choice are valid payloads regardless of requested shape
        raw_t = first(("tiles", "indices", "selection", "answers"))
        if isinstance(raw_t, list) and all(
                isinstance(x, (int, float)) and not isinstance(x, bool)
                for x in raw_t):
            return {"type": "tiles", "indices": [int(x) for x in raw_t]}
        raw_c = first(("choice", "option", "answer_index", "answerIndex"))
        if isinstance(raw_c, (int, float)) and not isinstance(raw_c, bool):
            return {"type": "choice", "index": int(round(raw_c))}

        raw_n = first(("count", "number", "total", "amount", "answer"))
        if isinstance(raw_n, (int, float)) and not isinstance(raw_n, bool) \
                and shape == "count":
            return {"type": "count", "count": int(round(raw_n))}

        raw_p = first(("points", "point", "clicks", "coordinates",
                       "locations"))
        if raw_p is not None:
            pts = []
            if isinstance(raw_p, (list, tuple)) and raw_p \
                    and all(isinstance(x, (list, tuple, dict)) for x in raw_p):
                pts = [q for q in (pt(x) for x in raw_p) if q]
            else:
                p = pt(raw_p)
                if p:
                    pts = [p]
            if pts:
                return {"type": "points", "points": pts}

        raw_b = first(("bbox", "box", "bounding_box", "boundingBox",
                       "rectangle", "rect"))
        if raw_b is not None:
            vals = None
            if isinstance(raw_b, dict):
                for keys in (("x1", "y1", "x2", "y2"),
                             ("left", "top", "right", "bottom")):
                    if all(k in raw_b for k in keys):
                        vals = [num(raw_b[k]) for k in keys]
                        break
                if vals is None and all(k in raw_b for k in
                                        ("x", "y", "width", "height")):
                    x, y = num(raw_b["x"]), num(raw_b["y"])
                    w, h = num(raw_b["width"]), num(raw_b["height"])
                    if None not in (x, y, w, h):
                        vals = [x, y, min(1.0, x + w), min(1.0, y + h)]
            elif isinstance(raw_b, (list, tuple)) and len(raw_b) >= 4:
                vals = [num(raw_b[i]) for i in range(4)]
            if vals and None not in vals:
                x1, y1, x2, y2 = vals
                return {"type": "bbox", "bbox": {
                    "x1": min(x1, x2), "y1": min(y1, y2),
                    "x2": max(x1, x2), "y2": max(y1, y2)}}

        raw_d = first(("drag", "drags", "path", "gesture", "pattern"))
        if isinstance(raw_d, dict):
            f = pt(raw_d.get("from") or raw_d.get("start"))
            t = pt(raw_d.get("to") or raw_d.get("end"))
            if f and t:
                return {"type": "drag", "from": f, "to": t}
        elif isinstance(raw_d, (list, tuple)) and len(raw_d) >= 2:
            pts = [q for q in (pt(x) for x in raw_d) if q]
            if len(pts) >= 2:
                return {"type": "drag", "from": pts[0], "to": pts[-1]}
        if "from" in obj and "to" in obj:
            f, t = pt(obj["from"]), pt(obj["to"])
            if f and t:
                return {"type": "drag", "from": f, "to": t}

        raw_txt = first(("answer", "text"))
        if isinstance(raw_txt, str) and raw_txt.strip():
            return {"type": "text", "text": raw_txt.strip()}
        return None

    @staticmethod
    def _parse_stack_geometry(obj) -> Optional[dict]:
        """Full stacking-puzzle drag plan, in 0-100 PERCENT coordinates.

        Accepts {"drags": [[sx, sy, tx, ty], ...]} (also "moves"/"drag"),
        {"drags": {"from": [..], "to": [..]}}, a bare [[sx, sy, tx, ty], ..]
        list, or per-drag {"from"/"to"} dicts — the shapes small models
        emit. Coordinates beyond -3..103 reject the drag (a hallucinated
        pixel coordinate, not a percent); the rest clamp to 0-100. When
        EVERY coordinate in the plan is <= 1.0 the model answered in 0-1
        fractions and the whole plan is scaled x100. Returns
        {"type": "drags", "drags": [(sx, sy, tx, ty), ...]} or None when no
        usable drag survives (max 12 kept).
        """
        raw = None
        if isinstance(obj, dict):
            for key in ("drags", "moves", "drag"):
                v = obj.get(key)
                if isinstance(v, (list, tuple)) and v:
                    raw = v
                    break
                if isinstance(v, dict):
                    f = OllamaVisionClient._point(
                        v.get("from") or v.get("start"))
                    t = OllamaVisionClient._point(
                        v.get("to") or v.get("end"))
                    if f and t:
                        raw = [list(f) + list(t)]
                    break
        elif isinstance(obj, (list, tuple)) and obj:
            raw = obj
        if not raw:
            return None
        drags = []
        for item in raw:
            if isinstance(item, dict):
                f = OllamaVisionClient._point(
                    item.get("from") or item.get("start")
                    or item.get("grab") or item.get("pick"))
                t = OllamaVisionClient._point(
                    item.get("to") or item.get("end")
                    or item.get("drop") or item.get("target"))
                if f is None or t is None:
                    continue
                vals = (f[0], f[1], t[0], t[1])
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                vals = item[:4]
            else:
                continue
            try:
                a, b, c, d = (float(v) for v in vals)
            except (TypeError, ValueError):
                continue
            if any(v != v for v in (a, b, c, d)):  # NaN guard
                continue
            if any(v < -3 or v > 103 for v in (a, b, c, d)):
                continue
            drags.append((max(0.0, min(100.0, a)), max(0.0, min(100.0, b)),
                          max(0.0, min(100.0, c)), max(0.0, min(100.0, d))))
        if not drags:
            return None
        flat = [v for drag in drags for v in drag]
        if flat and all(0 <= v <= 1.0 for v in flat) and any(flat):
            # Model answered in 0-1 fractions despite the percent prompt —
            # rescale the whole plan (a legit percent plan can never have
            # every coordinate inside the top-left 1% of the image).
            drags = [tuple(v * 100.0 for v in drag) for drag in drags]
        elif flat and any(v > 1.0 for v in flat) and any(
                0 < v <= 1.0 for v in flat):
            # Mixed units: the plan is in percents but some coordinates
            # came back as 0-1 fractions — scale just those.
            drags = [tuple(v * 100.0 if 0 < v <= 1.0 else v for v in drag)
                     for drag in drags]
        drags = [tuple(round(v, 3) for v in drag) for drag in drags]
        return {"type": "drags", "drags": drags[:12]}

    @staticmethod
    def _parse_answer(content: str, tile_count: int,
                      shape: str = "tiles") -> Optional[dict]:
        """Turn the model's raw answer into a structured result.

        Tries, in order: strict JSON (format:json), a bare JSON array, a
        quoted string (text challenge), then a loose int array (tiles) or
        a lone integer (count).
        """
        text = (content or "").strip()
        if not text:
            return None

        # 1) Strict JSON object / array.
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
        if obj is not None:
            if isinstance(obj, dict):
                for key in ("tiles", "indices", "selection", "answers"):
                    val = obj.get(key)
                    if isinstance(val, list) and all(
                            isinstance(x, (int, float)) and not isinstance(x, bool)
                            for x in val):
                        return {"type": "tiles",
                                "indices": [int(x) for x in val]}
                val = (obj.get("count") or obj.get("number")
                       or obj.get("total"))
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return {"type": "count", "count": int(round(val))}
                val = obj.get("answer") or obj.get("text")
                if isinstance(val, str) and val.strip():
                    return {"type": "text", "text": val.strip()}
            elif isinstance(obj, list) and all(
                    isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in obj):
                return {"type": "tiles", "indices": [int(x) for x in obj]}
            return None

        # 2) JSON inside markdown fences / prose.
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            try:
                obj = json.loads(fence.group(1).strip())
                if isinstance(obj, list):
                    return {"type": "tiles", "indices": [int(x) for x in obj
                                                         if isinstance(x, (int, float))]}
                if isinstance(obj, dict):
                    for key in ("tiles", "indices"):
                        val = obj.get(key)
                        if isinstance(val, list):
                            return {"type": "tiles",
                                    "indices": [int(x) for x in val]}
            except Exception:
                pass

        # 3) Bare array like [1, 3, 7].
        m = _JSON_ARRAY_RE.search(text)
        if m:
            nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
            if nums:
                return {"type": "tiles", "indices": nums}

        # 4) Counting: a lone integer anywhere ("the answer is 3", "3").
        if shape == "count":
            digits = re.findall(r"\d+", text)
            if digits:
                return {"type": "count", "count": int(digits[0])}

        # 4) Quoted string for a text challenge.
        m = _JSON_STRING_RE.search(text)
        if m:
            val = m.group(1).strip()
            if val and not val.startswith("{"):
                return {"type": "text", "text": val}

        # 5) Loose: a bare line of short tokens (text challenge fallback).
        line = text.strip().strip('"').strip()
        if line and len(line) <= 32 and not any(c in line for c in "{}[]"):
            return {"type": "text", "text": line}

        return None


async def _self_test() -> None:
    """Quick smoke test against a running Ollama server."""
    client = OllamaVisionClient()
    ok, models = await client.check()
    print(f"server ok={ok} models={models}")
    if not ok or client.model not in models:
        print(f"model {client.model} not pulled — run: ollama pull {client.model}")
        return
    # Two solid-color tiles: ask which one is red (image 2 should win).
    def _solid(rgb: tuple) -> bytes:
        from io import BytesIO
        import struct
        w = h = 64
        raw = b"".join(bytes(rgb) * w for _ in range(h))
        def _chunk(tag: bytes, data: bytes) -> bytes:
            return tag + struct.pack(">I", len(data)) + data + struct.pack(">I", 0)
        return (b"\x89PNG\r\n\x1a\n"
                + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + _chunk(b"IDAT", __import__("zlib").compress(raw))
                + _chunk(b"IEND", b""))
    ans = await client.solve(
        "Select all images that are red.",
        [_solid((30, 90, 200)), _solid((200, 30, 30))])
    print(f"answer: {ans}")


if __name__ == "__main__":
    asyncio.run(_self_test())

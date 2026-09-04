#!/usr/bin/env python3
"""vision_solver.py — Hugging Face vision solver for the hCaptcha image grid.

Every visual answer in this stack comes from a Hugging Face hosted vision
model. There is no local "brain" checkpoint, no CNN weights and no
self-hosted Ollama service any more: the bot

  1. reads the challenge prompt ("Please select all images with a boat")
     from the hCaptcha challenge frame,
  2. screenshots every tile of the image grid (or the whole canvas),
  3. sends the prompt + images to the Hugging Face Inference API,
  4. the model answers with the tiles / points / drag to perform,
  5. the bot clicks or drags, and hCaptcha itself mints the token.

Transport: the classic serverless Inference API,

    POST https://api-inference.huggingface.co/models/<HF_MODEL>/v1/chat/completions

which speaks the OpenAI chat schema with ``image_url`` data-URI parts.

Configuration (env vars):

  API_KEY     Hugging Face access token (hf_...). REQUIRED — sent as
              ``Authorization: Bearer <API_KEY>``. HF_TOKEN / HF_API_KEY
              are accepted as fallbacks.
  HF_MODEL    model repo id
              (default Qwen/Qwen2.5-VL-7B-Instruct)
  HF_API_BASE base URL of the Inference API
              (default https://api-inference.huggingface.co/models)
  HF_TIMEOUT  per-request timeout in seconds (default 60)
  HF_TILE_TIMEOUT  per-tile yes/no timeout when the model is asked one
              tile at a time (default 20)
  HF_IMAGE_SIDE    max image side in px sent to the model (default 512)
  HF_PER_TILE      set to 1 to force the per-tile yes/no path
  HF_CHECK_TIMEOUT reachability-probe timeout in seconds (default 60 —
              serverless models cold-start on the first request)

Only aiohttp is used — no new dependencies.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Callable, List, Optional

import aiohttp

# ── Hugging Face configuration ───────────────────────────────────────────
HF_API_BASE = (os.environ.get("HF_API_BASE", "").strip().rstrip("/")
               or "https://api-inference.huggingface.co/models")

# API_KEY is the canonical name (that is what the deploy sets); HF_TOKEN and
# HF_API_KEY are accepted so a standard HF environment also works.
API_KEY = (os.environ.get("API_KEY", "").strip()
           or os.environ.get("HF_TOKEN", "").strip()
           or os.environ.get("HF_API_KEY", "").strip())

_DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
HF_MODEL = os.environ.get("HF_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
HF_TIMEOUT = float(os.environ.get("HF_TIMEOUT", "60"))
HF_CHECK_TIMEOUT = float(os.environ.get("HF_CHECK_TIMEOUT", "60"))
_TILE_TIMEOUT = float(os.environ.get("HF_TILE_TIMEOUT", "20"))
_MAX_SIDE = int(os.environ.get("HF_IMAGE_SIDE", "512"))
_FORCE_PER_TILE = os.environ.get("HF_PER_TILE", "").strip() in ("1", "true", "yes")


def per_tile_mode(name: str) -> bool:
    """True when the grid must be asked one tile at a time.

    Small captioning VLMs (SmolVLM, moondream, …) cannot follow a 9-image
    JSON contract; HF_PER_TILE=1 forces the same path for any model.
    """
    if _FORCE_PER_TILE:
        return True
    n = (name or "").lower()
    return any(k in n for k in (
        "smolvlm", "smol-vlm", "smol_vlm", "moondream", "256m", "500m",
    ))


def shrink_image(data: bytes, max_side: int = _MAX_SIDE) -> bytes:
    """Downscale a PNG/JPEG to a JPEG small enough for a fast HF round trip."""
    if not data:
        return data
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((max(32, int(max_side)), max(32, int(max_side))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80, optimize=True)
        out = buf.getvalue()
        return out or data
    except Exception:
        return data


def _data_uri(data: bytes) -> str:
    """bytes -> data: URI the HF chat-completions schema accepts."""
    b64 = base64.b64encode(data).decode("ascii")
    kind = "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg"
    return f"data:image/{kind};base64,{b64}"


_YES_WORDS = ("yes", "y", "yeah", "yep", "yup", "true", "match", "matching")
_NO_WORDS = ("no", "n", "nope", "nah", "false", "none")


def parse_yesno(text: str):
    """Map a tiny-VLM reply to True/False/None.

    Small VLMs often echo the question (\"… Answer yes or no.\"). Strip
    that instruction and read the first/last token so an echoed prompt
    is not scored as a \"no\".
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    t = re.sub(r"\banswer(?:\s+with)?\s+yes\s+or\s+no\b[.?!:;]*", " ", t)
    t = re.sub(r"\byes\s*/\s*no\b", " ", t)
    t = re.sub(r"\byes\s+or\s+no\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    first = re.split(r"[\s,.;:!?]+", t, maxsplit=1)[0]
    if first in _YES_WORDS:
        return True
    if first in _NO_WORDS:
        return False
    head = t[:40]
    has_yes = bool(re.search(r"\byes\b", head))
    has_no = bool(re.search(r"\bno\b", head))
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    words = re.findall(r"[a-z]+", t)
    if words:
        if words[-1] in _YES_WORDS:
            return True
        if words[-1] in _NO_WORDS:
            return False
    return None


def tile_yes_question(prompt: str, has_ref: bool = False) -> str:
    """One-sentence yes/no the 256M model can actually answer."""
    p = (prompt or "").strip()
    try:
        import hcaptcha_types as hct
        if hct.is_setdown_prompt(p):
            q = ("Does this photo show a table, nightstand, bench, wooden "
                 "deck, counter or shelf a mug could sit on? Not a balloon, "
                 "ball, leaf or sky. Answer yes or no.")
            if has_ref:
                return ("First image is the item. Second is a tile. " + q)
            return q
    except Exception:
        pass
    q = ("Does this image match the task: %s? "
         "Answer only yes or no." % p[:160])
    if has_ref:
        return "First image is the example. Second is a tile. " + q
    return q


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
    "For spatial/reference prompts such as 'find places safe for setting down "
    "the item in the reference', 'where the item could be stored/used', or "
    "'match the reference item': look at the HEADER reference image FIRST "
    "(a mug, a tool, …). Then pick EVERY tile that is a PLACE or SURFACE "
    "where that object could safely rest or be used — a table, nightstand, "
    "bench, wooden deck, counter, shelf, floor. Do NOT pick the object "
    "itself, a ball, a leaf, a hot-air balloon, or anything that would "
    "tip/spill/break the item. Several tiles usually match. "
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

# Small captioning VLMs cannot follow the long JSON contracts above.
# Keep these to one or two sentences; the per-tile path asks yes/no instead.
_SMALL_SYSTEM = {
    "tiles": "Look at the photo. Answer the question with yes or no only.",
    "points": 'Look at the photo. Reply {"points": [[x, y]]} with x,y between 0 and 1.',
    "bbox": 'Look at the photo. Reply {"bbox": {"x1":a,"y1":b,"x2":c,"y2":d}} 0 to 1.',
    "drag": 'Look at the photo. Reply {"drag": {"from": [x,y], "to": [x,y]}} 0 to 1.',
    "pattern": 'Look at the photo. Reply {"drag": {"from": [x,y], "to": [x,y]}} 0 to 1.',
    "tower": 'Look at the wooden towers. Reply {"drag": {"from": [x,y], "to": [x,y]}} onto the short stack. 0 to 1.',
    "stack": 'Reply {"drags": [[sx,sy,tx,ty], ...]} in 0-100 percent.',
    "choice": 'Reply {"choice": N} with the option number.',
    "count": 'How many? Reply {"count": N}.',
    "text": "Read the characters. Reply with only the text.",
}

_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]")
_JSON_STRING_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')


class HFVisionClient:
    """Async client for the Hugging Face serverless Inference API.

    Configuration comes from the env vars ``API_KEY`` (Hugging Face token),
    ``HF_MODEL`` (model repo id) and ``HF_API_BASE``.
    """

    def __init__(self, log: Optional[Callable] = None,
                 base: str = HF_API_BASE, model: str = HF_MODEL,
                 api_key: str = ""):
        self._log = log or (lambda msg, level="info": None)
        self.base = (base or HF_API_BASE).rstrip("/")
        self.model = model or _DEFAULT_MODEL
        self._api_key = api_key or API_KEY
        self.stats = {"calls": 0, "ok": 0, "failed": 0}
        # Machine-readable result of the latest reachability probe. Keep the
        # public ``check() -> (ok, models)`` contract for existing callers,
        # while letting them distinguish a transient connection failure from
        # deterministic configuration errors such as HTTP 401 or 404.
        self.last_check_error = ""
        self.last_check_http_status: Optional[int] = None
        if not self._api_key:
            self._log("[Vision] API_KEY is not set — the Hugging Face "
                      "Inference API will reject every request", level="error")
        self._log(f"[Vision] Hugging Face model: {self.model} @ {self.base}"
                  f"{' [authenticated]' if self._api_key else ' [NO API_KEY]'}")

    @property
    def endpoint(self) -> str:
        """Chat-completions URL for the configured model."""
        return f"{self.base}/{self.model}/v1/chat/completions"

    @property
    def configured(self) -> bool:
        return bool(self.base and self.model and self._api_key)

    async def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def check(self) -> tuple:
        """Probe the Hugging Face model endpoint.

        Returns ``(ok, models)`` where ``models`` is ``[self.model]`` when the
        model is reachable (kept for callers written against the old
        contract). ``last_check_error`` classifies failures so callers do not
        mistake an HTTP 401 (bad API_KEY) for a cold start and retry it.
        """
        self.last_check_error = ""
        self.last_check_http_status = None
        if not self._api_key:
            self.last_check_error = "authentication"
            self._log("[Vision] No API_KEY configured for Hugging Face",
                      level="error")
            return False, []
        url = f"{self.base}/{self.model}"
        try:
            timeout = aiohttp.ClientTimeout(total=HF_CHECK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(url, headers=await self._headers()) as r:
                    status = int(r.status)
                    body = ""
                    try:
                        body = (await r.text())[:200]
                    except Exception:
                        pass
            # 200 = warm, 503 = cold-starting (still a healthy model), and a
            # GET on an inference route often answers 405 "use POST".
            if status in (200, 405, 503):
                if status == 503:
                    self._log(f"[Vision] {self.model} is loading on Hugging "
                              "Face (cold start) — first solve may be slow",
                              level="warn")
                else:
                    self._log(f"[Vision] Hugging Face OK: {self.model}")
                return True, [self.model]
            self.last_check_http_status = status
            if status == 401:
                self.last_check_error = "authentication"
                self._log("[Vision] Hugging Face rejected API_KEY (HTTP 401)",
                          level="error")
            elif status == 403:
                self.last_check_error = "authorization"
                self._log(f"[Vision] Access forbidden for {self.model} "
                          "(HTTP 403) — accept the model licence or use a "
                          "token with inference permission", level="error")
            elif status == 404:
                self.last_check_error = "protocol"
                self._log(f"[Vision] Model {self.model} not found on Hugging "
                          "Face (HTTP 404) — check HF_MODEL", level="error")
            elif status == 429:
                self.last_check_error = "rate_limit"
                self._log("[Vision] Hugging Face rate limited (HTTP 429)",
                          level="warn")
            elif status >= 500:
                self.last_check_error = "server"
                self._log(f"[Vision] Hugging Face server error HTTP {status}: "
                          f"{body}", level="warn")
            else:
                self.last_check_error = "http"
                self._log(f"[Vision] Hugging Face HTTP {status}: {body}",
                          level="warn")
            return False, []
        except asyncio.TimeoutError:
            self.last_check_error = "timeout"
            self._log(f"[Vision] Probe timed out after {HF_CHECK_TIMEOUT:g}s "
                      f"at {url}", level="error")
            return False, []
        except aiohttp.ClientError as e:
            self.last_check_error = "connection"
            self._log(f"[Vision] Connection failed at {url}: "
                      f"{type(e).__name__}", level="error")
            return False, []
        except Exception as e:
            self.last_check_error = "connection"
            self._log(f"[Vision] Not reachable at {url}: "
                      f"{type(e).__name__}: {e}", level="error")
            return False, []

    async def _chat(self, system: str, content: str, images: List[bytes],
                    timeout: float, *, want_json: bool,
                    num_predict: int) -> Optional[str]:
        """One chat-completions turn. Returns the raw message content or None."""
        parts: list = [{"type": "text", "text": content}]
        for b in images:
            if b:
                parts.append({"type": "image_url",
                              "image_url": {"url": _data_uri(b)}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": parts},
            ],
            "max_tokens": int(num_predict),
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False,
        }
        if want_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_cfg) as s:
                async with s.post(self.endpoint, json=payload,
                                  headers=await self._headers()) as r:
                    if r.status != 200:
                        body = await r.text()
                        self._log(f"[Vision] Solve rejected (HTTP {r.status}): "
                                  f"{body[:200]}", level="warn")
                        return None
                    data = await r.json()
            choices = (data or {}).get("choices") or []
            if not choices:
                return ""
            msg = (choices[0] or {}).get("message") or {}
            out = msg.get("content")
            if isinstance(out, list):      # some servers return content parts
                out = "".join(p.get("text", "") for p in out
                              if isinstance(p, dict))
            return out or ""
        except Exception as e:
            self._log(f"[Vision] Solve error: {e}", level="error")
            return None

    async def _solve_tiles_one_by_one(self, prompt: str, images: List[bytes],
                                      examples: List[bytes],
                                      timeout: float) -> Optional[dict]:
        """Ask the model yes/no on each tile (small-VLM path)."""
        ref = shrink_image(examples[0]) if examples else None
        q = tile_yes_question(prompt, has_ref=bool(ref))
        system = _SMALL_SYSTEM["tiles"]
        per = min(_TILE_TIMEOUT, max(6.0, timeout))
        hits = []
        answered = 0
        for i, raw in enumerate(images, 1):
            tile = shrink_image(raw)
            bundle = [b for b in ((ref, tile) if ref else (tile,)) if b]
            content = await self._chat(
                system, q, bundle, per, want_json=False, num_predict=16)
            if content is None:
                self._log(f"[Vision] tile {i}/{len(images)} -> error",
                          level="debug")
                continue
            answered += 1
            yn = parse_yesno(content)
            self._log(f"[Vision] tile {i}/{len(images)} -> "
                      f"{content[:40]!r} yes={yn}", level="debug")
            if yn is True:
                hits.append(i)
        if answered == 0:
            return None
        self._log(f"[Vision] per-tile yes: {hits} "
                  f"({answered}/{len(images)} answered)")
        return {"type": "tiles", "indices": hits}

    async def solve(self, prompt: str, images: List[bytes],
                    shape: str = "tiles", examples: Optional[List[bytes]] = None,
                    timeout: float = HF_TIMEOUT) -> Optional[dict]:
        """Ask the Hugging Face model to answer an hCaptcha round.

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
            self._log("[Vision] Hugging Face not configured — set API_KEY "
                      "(and optionally HF_MODEL)", level="error")
            return None
        if not images:
            return None
        import time
        small = per_tile_mode(self.model)
        self._log(f"{time.strftime('%b %d %H:%M:%S.000 [info]')} "
                  f"solving with hugging face ({self.model}"
                  f"{', per-tile' if small and shape == 'tiles' else ''})")
        self.stats["calls"] += 1
        examples = list(examples or [])
        if small and shape == "tiles" and len(images) >= 2:
            got = await self._solve_tiles_one_by_one(
                prompt, images, examples, timeout)
            if got is not None:
                self.stats["ok"] += 1
                return got
            self.stats["failed"] += 1
            return None
        if small:
            system = _SMALL_SYSTEM.get(shape, _SMALL_SYSTEM["tiles"])
            imgs = [shrink_image(b) for b in list(examples) + list(images)]
            if examples:
                content = (f"First {len(examples)} image(s) are the example. "
                           f"Then the answer image. Task: {prompt}")
            else:
                content = prompt
            want_json = False
            npred = 64
        else:
            system = _SYSTEM_BY_SHAPE.get(shape, _SYSTEM_PROMPT)
            imgs = [shrink_image(b) for b in list(examples) + list(images)]
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
            want_json = True
            npred = 256
        content_out = await self._chat(
            system, content, imgs, timeout,
            want_json=want_json, num_predict=npred)
        if content_out is None:
            self.stats["failed"] += 1
            return None
        if not content_out:
            self._log("[Vision] Empty response from model", level="warn")
            self.stats["failed"] += 1
            return None
        parse_shape = "drag" if shape in ("pattern", "tower") else shape
        parsed = self._parse_geometry(content_out, parse_shape, len(images))
        if parsed is None and shape in ("tiles", "text", "count", "stack"):
            parsed = self._parse_answer(content_out, len(images), shape)
        if parsed is None:
            self._log(f"[Vision] Unparseable model answer: {content_out[:160]}",
                      level="warn")
            self.stats["failed"] += 1
            return None
        self.stats["ok"] += 1
        return parsed

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
                    a, b = HFVisionClient._num(p[kx]), \
                        HFVisionClient._num(p[ky])
                    if a is not None and b is not None:
                        return (a, b)
            return None
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            a, b = HFVisionClient._num(p[0]), HFVisionClient._num(p[1])
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
        obj = HFVisionClient._loads_repaired(content)
        if obj is None:
            return None
        num = HFVisionClient._num
        pt = HFVisionClient._point

        # Stacking plans (FunCAPTCHA/Arkose "make every column equal"): the
        # generic branches below would squash a multi-drag {"drags": [...]}
        # into one drag / a points pair, so parse the FULL plan first.
        if shape == "stack":
            return HFVisionClient._parse_stack_geometry(obj)

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
                    f = HFVisionClient._point(
                        v.get("from") or v.get("start"))
                    t = HFVisionClient._point(
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
                f = HFVisionClient._point(
                    item.get("from") or item.get("start")
                    or item.get("grab") or item.get("pick"))
                t = HFVisionClient._point(
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

        # 5) Loose tile numbers: "1 3 5", "tiles 1, 3 and 5". Must run
        # BEFORE the text fallback — "1 3 5" is a 5-char line with no
        # brackets and used to be misread as a text challenge.
        if shape == "tiles":
            loose = HFVisionClient._parse_loose_tiles(text, tile_count)
            if loose is not None:
                return loose

        # 6) Quoted string for a text challenge.
        m = _JSON_STRING_RE.search(text)
        if m:
            val = m.group(1).strip()
            if val and not val.startswith("{"):
                return {"type": "text", "text": val}

        # 7) Loose: a bare line of short tokens (text challenge fallback).
        line = text.strip().strip('"').strip()
        if line and len(line) <= 32 and not any(c in line for c in "{}[]"):
            return {"type": "text", "text": line}

        return None

    @staticmethod
    def _parse_loose_tiles(text: str, tile_count: int) -> Optional[dict]:
        """Pull 1-based tile numbers out of prose.

        Accepts ``1 3 5``, ``tiles 1, 3 and 5``, ``pick 2, 4``. A cue
        word (tiles/select/pick/…) keeps the grid size out of
        ``I see 9 tiles, pick 1 and 3``.
        """
        t = (text or "").strip()
        if not t:
            return None
        cap = max(int(tile_count or 0), 12)
        cue = re.search(
            r"(?:tiles?|indices|indexes|select(?:ed)?|pick(?:ed)?|"
            r"choose|chosen|answers?)\s*[:#=-]?\s*(.+)$",
            t, re.I | re.S)
        blob = cue.group(1) if cue else t
        nums = [int(x) for x in re.findall(r"\d+", blob)]
        nums = [n for n in nums if 1 <= n <= cap]
        out: List[int] = []
        seen = set()
        for n in nums:
            if n not in seen:
                seen.add(n)
                out.append(n)
        if not out:
            return None
        return {"type": "tiles", "indices": out}


async def _self_test() -> None:
    """Quick smoke test against the Hugging Face Inference API."""
    client = HFVisionClient(log=lambda m, level="info": print(m, flush=True))
    ok, models = await client.check()
    print(f"endpoint ok={ok} model={models}")
    if not ok:
        print("Set API_KEY (Hugging Face token) and optionally HF_MODEL.")
        return

    # Two solid-color tiles: ask which one is red (image 2 should win).
    def _solid(rgb: tuple) -> bytes:
        from io import BytesIO
        import struct
        w = h = 64
        raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

        def _chunk(tag: bytes, data: bytes) -> bytes:
            import zlib
            return (struct.pack(">I", len(data)) + tag + data
                    + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
        import zlib
        return (b"\x89PNG\r\n\x1a\n"
                + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + _chunk(b"IDAT", zlib.compress(raw))
                + _chunk(b"IEND", b""))

    ans = await client.solve(
        "Select all images that are red.",
        [_solid((30, 90, 200)), _solid((200, 30, 30))])
    print(f"answer: {ans}")


if __name__ == "__main__":
    asyncio.run(_self_test())

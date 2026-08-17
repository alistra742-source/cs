#!/usr/bin/env python3
"""vision_solver.py — local Ollama vision solver for the hCaptcha image grid.

Replaces the paid NoneCap / Nopecha token APIs with a model you own and run
locally. Instead of minting a token server-side, the bot:

  1. reads the challenge prompt ("Please select all images with a boat") from
     the hCaptcha challenge frame,
  2. screenshots every tile of the image grid,
  3. sends the prompt + tile images to a local Ollama vision model,
  4. the model answers with the tile numbers that satisfy the task,
  5. the bot clicks those tiles + Verify (see server.py's
     ``_solve_hcaptcha_if_present``), and hCaptcha itself mints the token.

Configuration (env vars):

  OLLAMA_BASE   base URL of the Ollama server (default http://localhost:11434)
  OLLAMA_MODEL  vision model to use (default qwen3-vl:2b)
  OLLAMA_TIMEOUT  per-request timeout in seconds (default 180)

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

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:2b").strip()
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))

# Instruct the model to answer in reading order: image 1 = top-left tile,
# then left→right, top→bottom. JSON-only output (no markdown, no prose).
_SYSTEM_PROMPT = (
    "You are a precise image-selection solver for an hCaptcha challenge grid. "
    "You are given the challenge instruction and one image per grid tile, in "
    "reading order: image 1 is the top-left tile, image 2 is the tile to its "
    "right, and so on left-to-right, top-to-bottom. "
    "Look at EVERY tile carefully and decide which ones satisfy the instruction. "
    'Answer with ONLY a JSON object, never any other text: '
    '{"tiles": [1, 3, 7]} for a grid selection, or {"answer": "the text"} '
    "if the challenge asks you to type characters instead. "
    'Use [] when no tile matches. Tile numbers must be integers.'
)

_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]")
_JSON_STRING_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')


class OllamaVisionClient:
    """Async client for a local Ollama vision model (no paid API involved)."""

    def __init__(self, log: Optional[Callable] = None,
                 base: str = OLLAMA_BASE, model: str = OLLAMA_MODEL):
        self._log = log or (lambda msg, level="info": None)
        self.base = base.rstrip("/")
        self.model = model or "qwen3-vl:2b"
        self.stats = {"calls": 0, "ok": 0, "failed": 0}

    @property
    def configured(self) -> bool:
        return bool(self.base)

    async def check(self) -> tuple:
        """Probe the Ollama server and list pulled models.

        Returns ``(ok, models)`` — ``models`` is a list of model names
        (including tags) or [] when the server is unreachable.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(f"{self.base}/api/tags") as r:
                    if r.status != 200:
                        self._log(f"[Ollama] /api/tags HTTP {r.status}", level="warn")
                        return False, []
                    data = await r.json()
            models = [m.get("name") or m.get("model") or ""
                      for m in (data or {}).get("models", [])]
            models = [m for m in models if m]
            self._log(f"[Ollama] Server OK at {self.base} ({len(models)} models pulled)")
            return True, models
        except Exception as e:
            self._log(f"[Ollama] Not reachable at {self.base}: {e}", level="error")
            return False, []

    async def solve(self, prompt: str, images: List[bytes],
                    timeout: float = OLLAMA_TIMEOUT) -> Optional[dict]:
        """Ask the model which tiles match ``prompt``.

        ``images`` are the grid tiles as PNG/JPEG bytes in reading order
        (tile 1 = top-left). Returns one of:

          {"type": "tiles", "indices": [1, 3, 7]}
          {"type": "text",  "text": "abc123"}
          None  — model unreachable or answer unparseable
        """
        if not self.configured:
            self._log("[Ollama] No OLLAMA_BASE configured", level="error")
            return None
        if not images:
            return None
        self.stats["calls"] += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"CHALLENGE TASK: {prompt}\n\n"
                               f"There are {len(images)} tiles. "
                               f"Answer with the JSON object only.",
                    "images": [base64.b64encode(b).decode("ascii") for b in images],
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
                async with s.post(f"{self.base}/api/chat", json=payload) as r:
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
            parsed = self._parse_answer(content, len(images))
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

    @staticmethod
    def _parse_answer(content: str, tile_count: int) -> Optional[dict]:
        """Turn the model's raw answer into a structured result.

        Tries, in order: strict JSON (format:json), a bare JSON array, a
        quoted string (text challenge), then a loose int array.
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

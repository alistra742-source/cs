#!/usr/bin/env python3
"""vision_service.py — standalone 24/7 vision endpoint.

Runs as its OWN service (own Railway service, VPS, or `python
vision_service.py` anywhere). The bot POSTs an image + the challenge
prompt; this answers.

    POST /solve
    {
      "prompt": "click each image containing a boat",
      "shape":  "tiles" | "points" | "bbox" | "drag" | "count" | "text",
      "images": ["<base64>", ...],      # tiles, or one big canvas
      "examples": ["<base64>", ...]     # optional reference images
    }

    -> {"type":"tiles","indices":[1,4,7]}
       {"type":"points","points":[[0.42,0.31]]}
       {"type":"drag","from":[x,y],"to":[x,y]}
       {"type":"count","count":3}
       {"type":"text","text":"28"}

Backends, tried in order of what is configured:

  VISION_BACKEND=openai   OPENAI_API_KEY   (gpt-4o / any vision model)
  VISION_BACKEND=gemini   GEMINI_API_KEY
  VISION_BACKEND=ollama   OLLAMA_BASE      (self-hosted, e.g. gemma3:4b)
  VISION_BACKEND=auto     first one whose key is present  (default)

Any OpenAI-compatible endpoint works via OPENAI_BASE, so this also
covers Groq, OpenRouter, Together, a local vLLM, etc.

Health:  GET /health   -> {"ok":true,"backend":"openai","model":"gpt-4o"}
Test:    POST /solve with no images -> 400 with a clear message
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from flask import Flask, jsonify, request

app = Flask(__name__)

# ── configuration ───────────────────────────────────────────────────────
BACKEND = (os.environ.get("VISION_BACKEND") or "auto").strip().lower()

OPENAI_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_BASE = (os.environ.get("OPENAI_BASE")
               or "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = (os.environ.get("OPENAI_MODEL") or "gpt-4o").strip()

GEMINI_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL = (os.environ.get("GEMINI_MODEL")
                or "gemini-2.0-flash").strip()

OLLAMA_BASE = (os.environ.get("OLLAMA_BASE") or "").rstrip("/")
OLLAMA_MODEL = (os.environ.get("OLLAMA_MODEL") or "gemma3:4b").strip()

TIMEOUT = float(os.environ.get("VISION_TIMEOUT") or "60")
SERVICE_TOKEN = (os.environ.get("VISION_SERVICE_TOKEN") or "").strip()
PORT = int(os.environ.get("PORT") or "8099")


def active_backend() -> str:
    """Which backend will actually be used."""
    if BACKEND != "auto":
        return BACKEND
    if OPENAI_KEY:
        return "openai"
    if GEMINI_KEY:
        return "gemini"
    if OLLAMA_BASE:
        return "ollama"
    return "none"


def active_model() -> str:
    return {"openai": OPENAI_MODEL, "gemini": GEMINI_MODEL,
            "ollama": OLLAMA_MODEL}.get(active_backend(), "")


# ── prompting ───────────────────────────────────────────────────────────
# One system prompt per answer SHAPE. hCaptcha grades the shape as much as
# the content: answering a point round with tile indices can never pass.
_SYSTEM = {
    "tiles": (
        "You are looking at a numbered grid of images from a captcha. "
        "Number them 1..N in reading order (left to right, top to bottom). "
        "Reply ONLY with JSON: {\"indices\":[1,4,7]} listing every image "
        "that matches the instruction. Empty list if none match."),
    "points": (
        "You are looking at ONE image from a captcha. Reply ONLY with "
        "JSON: {\"points\":[[x,y]]} where x and y are NORMALISED 0.0-1.0 "
        "coordinates (0,0 = top-left, 1,1 = bottom-right) of the centre "
        "of each thing the instruction asks you to click."),
    "bbox": (
        "Reply ONLY with JSON: {\"bbox\":{\"x1\":..,\"y1\":..,\"x2\":..,"
        "\"y2\":..}} using NORMALISED 0.0-1.0 coordinates for the box the "
        "instruction asks for."),
    "drag": (
        "This is a drag puzzle. Identify the loose draggable piece (it "
        "sits apart from the main picture, often on a light panel, "
        "sometimes badged Move) and the place it belongs (the hole, "
        "socket, dashed outline or empty cell whose shape MATCHES it). "
        "Reply ONLY with JSON: {\"from\":[x,y],\"to\":[x,y]} in "
        "NORMALISED 0.0-1.0 coordinates."),
    "count": (
        "Reply ONLY with JSON: {\"count\":N} — how many of the thing the "
        "instruction names appear in the image."),
    "text": (
        "Read the image and answer the question. Reply ONLY with JSON: "
        "{\"text\":\"your answer\"} — a single word, number or short "
        "phrase, nothing else."),
    "choice": (
        "Reply ONLY with JSON: {\"choice\":N} — the 1-based index of the "
        "option that answers the instruction."),
}


def system_for(shape: str) -> str:
    return _SYSTEM.get(shape, _SYSTEM["tiles"])


# ── backends ────────────────────────────────────────────────────────────
def _openai_solve(system: str, prompt: str, images: List[str]) -> str:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                          "detail": "high"},
        })
    r = requests.post(
        f"{OPENAI_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_KEY}",
                 "Content-Type": "application/json"},
        json={"model": OPENAI_MODEL,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": content}],
              "temperature": 0.1, "max_tokens": 300},
        timeout=TIMEOUT)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _gemini_solve(system: str, prompt: str, images: List[str]) -> str:
    parts: List[Dict[str, Any]] = [{"text": f"{system}\n\n{prompt}"}]
    for b64 in images:
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": b64}})
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}",
        json={"contents": [{"parts": parts}],
              "generationConfig": {"temperature": 0.1,
                                   "maxOutputTokens": 300}},
        timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return (data["candidates"][0]["content"]["parts"][0]["text"] or "").strip()


def _ollama_solve(system: str, prompt: str, images: List[str]) -> str:
    r = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": OLLAMA_MODEL,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": prompt,
                            "images": images}],
              "stream": False, "format": "json",
              "options": {"temperature": 0.1, "num_predict": 300},
              "keep_alive": "10m"},
        timeout=TIMEOUT)
    r.raise_for_status()
    return ((r.json().get("message") or {}).get("content") or "").strip()


def call_backend(system: str, prompt: str, images: List[str]) -> str:
    be = active_backend()
    if be == "openai":
        return _openai_solve(system, prompt, images)
    if be == "gemini":
        return _gemini_solve(system, prompt, images)
    if be == "ollama":
        return _ollama_solve(system, prompt, images)
    raise RuntimeError(
        "No vision backend configured. Set OPENAI_API_KEY, GEMINI_API_KEY "
        "or OLLAMA_BASE.")


# ── answer parsing ──────────────────────────────────────────────────────
def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model reply.

    Models wrap answers in prose and code fences no matter how firmly you
    ask them not to, so never trust a bare json.loads.
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    depth, start = 0, -1
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    start = -1
    return None


def _clamp01(v) -> float:
    try:
        f = float(v)
    except Exception:
        return 0.5
    return 0.0 if f < 0 else (1.0 if f > 1 else f)


def to_answer(raw: str, shape: str, n_images: int) -> Optional[dict]:
    """Model reply -> the answer shape the bot expects."""
    obj = extract_json(raw)
    if obj is None:
        # last resort: bare digits for count/text rounds
        m = re.search(r"-?\d+", raw or "")
        if m and shape in ("count", "text"):
            return ({"type": "count", "count": int(m.group())}
                    if shape == "count"
                    else {"type": "text", "text": m.group()})
        return None

    if shape == "tiles":
        idx = obj.get("indices")
        if isinstance(idx, list):
            clean = sorted({int(i) for i in idx
                            if str(i).lstrip("-").isdigit()
                            and 1 <= int(i) <= max(n_images, 1)})
            return {"type": "tiles", "indices": clean}
        return None
    if shape == "points":
        pts = obj.get("points")
        if isinstance(pts, list) and pts:
            out = []
            for p in pts[:6]:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    out.append([_clamp01(p[0]), _clamp01(p[1])])
            if out:
                return {"type": "points", "points": out}
        return None
    if shape == "bbox":
        bb = obj.get("bbox")
        if isinstance(bb, dict) and all(k in bb for k in
                                        ("x1", "y1", "x2", "y2")):
            return {"type": "bbox", "bbox": {k: _clamp01(bb[k])
                                             for k in ("x1", "y1",
                                                       "x2", "y2")}}
        return None
    if shape == "drag":
        f, t = obj.get("from"), obj.get("to")
        if (isinstance(f, (list, tuple)) and len(f) >= 2
                and isinstance(t, (list, tuple)) and len(t) >= 2):
            return {"type": "drag",
                    "from": [_clamp01(f[0]), _clamp01(f[1])],
                    "to": [_clamp01(t[0]), _clamp01(t[1])]}
        return None
    if shape == "count":
        if "count" in obj:
            try:
                return {"type": "count", "count": int(obj["count"])}
            except Exception:
                return None
        return None
    if shape == "choice":
        if "choice" in obj:
            try:
                return {"type": "choice", "choice": int(obj["choice"])}
            except Exception:
                return None
        return None
    if shape == "text":
        if "text" in obj:
            return {"type": "text", "text": str(obj["text"]).strip()}
        return None
    return None


# ── routes ──────────────────────────────────────────────────────────────
def _authorised() -> bool:
    if not SERVICE_TOKEN:
        return True
    got = (request.headers.get("Authorization") or "").replace("Bearer ", "")
    return got.strip() == SERVICE_TOKEN


@app.route("/health")
def health():
    be = active_backend()
    return jsonify({"ok": be != "none", "backend": be,
                    "model": active_model(),
                    "configured": be != "none"})


@app.route("/solve", methods=["POST"])
def solve():
    if not _authorised():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt") or "").strip()
    shape = str(body.get("shape") or "tiles").strip().lower()
    images = [i for i in (body.get("images") or []) if i]
    examples = [i for i in (body.get("examples") or []) if i]
    if not images:
        return jsonify({"error": "no images"}), 400
    if active_backend() == "none":
        return jsonify({"error": "no vision backend configured"}), 503

    question = prompt or "Answer the captcha."
    if examples:
        question = ("The FIRST image is the reference/example. " + question)
    if shape == "tiles" and len(images) > 1:
        question += (f" There are {len(images)} images, numbered 1 to "
                     f"{len(images)} in reading order.")

    t0 = time.time()
    try:
        raw = call_backend(system_for(shape), question, examples + images)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 502
    answer = to_answer(raw, shape, len(images))
    took = round(time.time() - t0, 2)
    if answer is None:
        return jsonify({"error": "unparseable", "raw": raw[:400],
                        "took": took}), 200
    answer["took"] = took
    answer["model"] = active_model()
    return jsonify(answer)


if __name__ == "__main__":
    be = active_backend()
    print(f"[vision] backend={be} model={active_model()} port={PORT}",
          flush=True)
    if be == "none":
        print("[vision] WARNING: no backend configured — set "
              "OPENAI_API_KEY, GEMINI_API_KEY or OLLAMA_BASE", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)

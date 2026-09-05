#!/usr/bin/env python3
"""solver_chain.py — four captcha backends, tried in order.

    1  NoneCap        NONECAP_API   hosted enterprise token
    2  AZcaptcha      API_KEY2      hosted token (in.php / res.php)
    3  OpenRouter     API_KEY3      gemini-2.5-flash, VISION
    4  Google AI      API_KEY4      gemini-2.5-flash, VISION

Tiers 1-2 return a **token** — the site's own hCaptcha is satisfied by
submitting it.

Tiers 3-4 return **coordinates**. They cannot mint a token; they answer
"where do I click". The bot then clicks those points with its own
humanized mouse and lets the widget mint its own token — which for
enterprise rqdata is the only binding that can ever match, because the
widget knows its own challenge.

The vision tiers are told, in the strongest terms the prompt allows, to
reply with COORDINATES AND NOTHING ELSE — no prose, no explanation, no
markdown. Every reply is still parsed defensively, because models ignore
that instruction constantly.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore


def _env(*names: str) -> str:
    """First non-empty env var, tolerating a pasted ``NAME = value``."""
    for n in names:
        v = (os.environ.get(n) or "").strip()
        if not v:
            continue
        m = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.+)$", v)
        if m and ("://" in m.group(1) or len(m.group(1)) > 8):
            v = m.group(1).strip()
        return v.strip("'\"")
    return ""


AZCAPTCHA_KEY = lambda: _env("API_KEY2", "AZCAPTCHA_KEY")          # noqa: E731
OPENROUTER_KEY = lambda: _env("API_KEY3", "OPENROUTER_KEY")        # noqa: E731
GOOGLE_KEY = lambda: _env("API_KEY4", "GOOGLE_AI_KEY", "GEMINI_API_KEY")  # noqa: E731

AZCAPTCHA_BASE = (os.environ.get("AZCAPTCHA_BASE")
                  or "http://azcaptcha.com").rstrip("/")
OPENROUTER_MODEL = (os.environ.get("OPENROUTER_MODEL")
                    or "google/gemini-2.5-flash").strip()
GOOGLE_MODEL = (os.environ.get("GOOGLE_MODEL")
                or "gemini-2.5-flash").strip()
VISION_TIMEOUT = float(os.environ.get("VISION_TIMEOUT") or "45")


# ─────────────────────────────────────────────────────────────────────────
# Tier 2 — AZcaptcha (token)
# ─────────────────────────────────────────────────────────────────────────
class AZCaptcha:
    """in.php / res.php hCaptcha solver.

    Sends rqdata via the ``data`` parameter, which is how every
    2captcha-compatible API takes the enterprise blob.
    """

    def __init__(self, key: str = "", log=None):
        self._key = key or AZCAPTCHA_KEY()
        self._log = log or (lambda *a, **k: None)
        self.last_error = ""
        self.last_id = ""

    @property
    def enabled(self) -> bool:
        return bool(self._key) and aiohttp is not None

    async def solve(self, sitekey: str, url: str, rqdata: str = "",
                    invisible: bool = True, proxy: str = "",
                    timeout: float = 180.0) -> Optional[str]:
        if not self.enabled:
            self.last_error = "not configured"
            return None
        data = {
            "key": self._key,
            "method": "hcaptcha",
            "sitekey": sitekey,
            "pageurl": url,
            "json": "1",
        }
        if invisible:
            data["invisible"] = "1"
        if rqdata:
            data["data"] = rqdata          # the enterprise blob
        if proxy:
            data["proxy"] = proxy
            data["proxytype"] = "HTTP"
        try:
            cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=cfg) as s:
                self._log(f"[AZcaptcha] Submitting hcaptcha "
                          f"{sitekey[:8]}… (rqdata={'yes' if rqdata else 'no'})")
                async with s.post(f"{AZCAPTCHA_BASE}/in.php",
                                  data=data) as r:
                    body = await r.json(content_type=None)
                if str(body.get("status")) != "1":
                    self.last_error = str(body.get("request") or body)[:120]
                    self._log(f"[AZcaptcha] submit failed: "
                              f"{self.last_error}", level="warn")
                    return None
                cid = str(body.get("request"))
                self.last_id = cid
                # poll
                deadline = asyncio.get_event_loop().time() + timeout
                await asyncio.sleep(8)
                while asyncio.get_event_loop().time() < deadline:
                    async with s.get(
                            f"{AZCAPTCHA_BASE}/res.php",
                            params={"key": self._key, "action": "get",
                                    "id": cid, "json": "1"}) as r:
                        out = await r.json(content_type=None)
                    st = str(out.get("status"))
                    req = str(out.get("request") or "")
                    if st == "1":
                        self._log(f"[AZcaptcha] Token received "
                                  f"({len(req)} chars, {req[:3]}…)")
                        return req
                    if req and req != "CAPCHA_NOT_READY":
                        self.last_error = req[:120]
                        self._log(f"[AZcaptcha] {req}", level="warn")
                        return None
                    await asyncio.sleep(5)
            self.last_error = "timeout"
            return None
        except Exception as e:
            self.last_error = type(e).__name__
            self._log(f"[AZcaptcha] {type(e).__name__}: {e}", level="warn")
            return None


# ─────────────────────────────────────────────────────────────────────────
# Tiers 3-4 — vision (coordinates only)
# ─────────────────────────────────────────────────────────────────────────
# COORDINATES AND NOTHING ELSE. Repeated because models ignore it once.
_COORD_RULES = (
    "You are solving a captcha. Reply with COORDINATES ONLY.\n"
    "OUTPUT RULES — follow exactly:\n"
    "  * Reply with ONE line of raw JSON and NOTHING else.\n"
    "  * No prose. No explanation. No markdown. No code fences.\n"
    "  * Coordinates are NORMALISED floats 0.0-1.0, where (0,0) is the "
    "TOP-LEFT of the image and (1,1) is the BOTTOM-RIGHT.\n"
    "  * Never output pixels. Never output percentages.\n"
)

_SHAPE_RULE = {
    "tiles": (
        'The images are a grid, numbered 1..N in reading order '
        '(left to right, top to bottom).\n'
        'Output exactly: {"indices":[1,4,7]}\n'
        'List every image that matches. Empty list if none: {"indices":[]}'),
    "points": (
        'Output exactly: {"points":[[x,y]]}\n'
        'One [x,y] pair per thing to click, at its CENTRE.'),
    "bbox": (
        'Output exactly: {"bbox":{"x1":0.1,"y1":0.2,"x2":0.5,"y2":0.6}}'),
    "drag": (
        'Find the loose draggable piece (it sits apart from the main '
        'picture, often on a lighter panel, sometimes badged "Move") and '
        'the place it belongs (the hole, socket, dashed outline or empty '
        'cell whose SHAPE MATCHES the piece).\n'
        'Output exactly: {"from":[x,y],"to":[x,y]}\n'
        '"from" is the centre of the piece, "to" is the centre of the hole.'),
    "count": 'Output exactly: {"count":3}',
    "text": 'Output exactly: {"text":"your answer"}',
    "choice": 'Output exactly: {"choice":2}   (1-based index)',
}


def build_prompt(question: str, shape: str, n_images: int,
                 has_examples: bool = False) -> str:
    parts = [_COORD_RULES, _SHAPE_RULE.get(shape, _SHAPE_RULE["tiles"]), ""]
    if has_examples:
        parts.append("The FIRST image is the reference/example to match "
                     "against.")
    if shape == "tiles" and n_images > 1:
        parts.append(f"There are {n_images} images, numbered 1 to "
                     f"{n_images} in reading order.")
    parts.append("")
    parts.append(f"CHALLENGE: {question}")
    parts.append("")
    parts.append("Reply with the JSON only.")
    return "\n".join(parts)


def extract_json(text: str) -> Optional[dict]:
    """Dig the JSON out of a reply, however the model wrapped it."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    depth = 0
    start = -1
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


def _c01(v) -> float:
    try:
        f = float(v)
    except Exception:
        return 0.5
    if f > 1.0:                 # model gave a percentage or pixels
        f = f / 100.0 if f <= 100.0 else 1.0
    return 0.0 if f < 0 else (1.0 if f > 1 else f)


def parse_answer(raw: str, shape: str, n_images: int) -> Optional[dict]:
    obj = extract_json(raw)
    if obj is None:
        m = re.search(r"-?\d+", raw or "")
        if m and shape in ("count", "text"):
            return ({"type": "count", "count": int(m.group())}
                    if shape == "count"
                    else {"type": "text", "text": m.group()})
        return None
    if shape == "tiles":
        idx = obj.get("indices")
        if isinstance(idx, list):
            return {"type": "tiles",
                    "indices": sorted({int(i) for i in idx
                                       if str(i).lstrip("-").isdigit()
                                       and 1 <= int(i) <= max(n_images, 1)})}
        return None
    if shape == "points":
        pts = obj.get("points")
        if isinstance(pts, list) and pts:
            out = [[_c01(p[0]), _c01(p[1])] for p in pts[:6]
                   if isinstance(p, (list, tuple)) and len(p) >= 2]
            return {"type": "points", "points": out} if out else None
        return None
    if shape == "bbox":
        bb = obj.get("bbox")
        if isinstance(bb, dict) and all(k in bb for k in
                                        ("x1", "y1", "x2", "y2")):
            return {"type": "bbox",
                    "bbox": {k: _c01(bb[k]) for k in
                             ("x1", "y1", "x2", "y2")}}
        return None
    if shape == "drag":
        f, t = obj.get("from"), obj.get("to")
        if (isinstance(f, (list, tuple)) and len(f) >= 2
                and isinstance(t, (list, tuple)) and len(t) >= 2):
            return {"type": "drag", "from": [_c01(f[0]), _c01(f[1])],
                    "to": [_c01(t[0]), _c01(t[1])]}
        return None
    if shape == "count" and "count" in obj:
        try:
            return {"type": "count", "count": int(obj["count"])}
        except Exception:
            return None
    if shape == "choice" and "choice" in obj:
        try:
            return {"type": "choice", "choice": int(obj["choice"])}
        except Exception:
            return None
    if shape == "text" and "text" in obj:
        return {"type": "text", "text": str(obj["text"]).strip()}
    return None


class VisionSolver:
    """Coordinates from Gemini, via OpenRouter (tier 3) or Google (tier 4)."""

    def __init__(self, provider: str, key: str = "", log=None):
        self.provider = provider
        self._key = key or (OPENROUTER_KEY() if provider == "openrouter"
                            else GOOGLE_KEY())
        self._log = log or (lambda *a, **k: None)
        self.last_error = ""
        self.last_raw = ""

    @property
    def enabled(self) -> bool:
        return bool(self._key) and aiohttp is not None

    @property
    def model(self) -> str:
        return (OPENROUTER_MODEL if self.provider == "openrouter"
                else GOOGLE_MODEL)

    async def _openrouter(self, prompt: str, images: List[str]) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b in images:
            content.append({"type": "image_url",
                            "image_url": {"url":
                                          f"data:image/jpeg;base64,{b}"}})
        cfg = aiohttp.ClientTimeout(total=VISION_TIMEOUT)
        async with aiohttp.ClientSession(timeout=cfg) as s:
            async with s.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self._key}",
                             "Content-Type": "application/json"},
                    json={"model": OPENROUTER_MODEL,
                          "messages": [{"role": "user", "content": content}],
                          "temperature": 0.0, "max_tokens": 300}) as r:
                if r.status != 200:
                    raise RuntimeError(
                        f"HTTP {r.status}: {(await r.text())[:160]}")
                d = await r.json()
        return (d["choices"][0]["message"]["content"] or "").strip()

    async def _google(self, prompt: str, images: List[str]) -> str:
        parts: List[Dict[str, Any]] = [{"text": prompt}]
        for b in images:
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": b}})
        cfg = aiohttp.ClientTimeout(total=VISION_TIMEOUT)
        async with aiohttp.ClientSession(timeout=cfg) as s:
            async with s.post(
                    f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/{GOOGLE_MODEL}:generateContent",
                    headers={"x-goog-api-key": self._key,
                             "Content-Type": "application/json"},
                    json={"contents": [{"parts": parts}],
                          "generationConfig": {"temperature": 0.0,
                                               "maxOutputTokens": 300}}) as r:
                if r.status != 200:
                    raise RuntimeError(
                        f"HTTP {r.status}: {(await r.text())[:160]}")
                d = await r.json()
        return (d["candidates"][0]["content"]["parts"][0]["text"]
                or "").strip()

    async def solve(self, question: str, images: List[bytes],
                    shape: str = "tiles",
                    examples: Optional[List[bytes]] = None) -> Optional[dict]:
        if not self.enabled or not images:
            return None
        ex = examples or []
        b64 = [base64.b64encode(b).decode() for b in (ex + list(images))]
        prompt = build_prompt(question, shape, len(images), bool(ex))
        self._log(f"[{self.provider}] {self.model} solving shape={shape} "
                  f"({len(images)} image(s))")
        try:
            raw = (await self._openrouter(prompt, b64)
                   if self.provider == "openrouter"
                   else await self._google(prompt, b64))
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self._log(f"[{self.provider}] {self.last_error}", level="warn")
            return None
        self.last_raw = raw
        answer = parse_answer(raw, shape, len(images))
        if answer is None:
            self._log(f"[{self.provider}] unparseable reply: {raw[:160]!r}",
                      level="warn")
        else:
            self._log(f"[{self.provider}] -> {answer}")
        return answer


def available() -> Dict[str, bool]:
    return {
        "nonecap": bool(_env("NONECAP_API", "NONECAP_API_KEY")),
        "azcaptcha": bool(AZCAPTCHA_KEY()),
        "openrouter": bool(OPENROUTER_KEY()),
        "google": bool(GOOGLE_KEY()),
    }


if __name__ == "__main__":  # pragma: no cover
    print("configured backends:", json.dumps(available(), indent=2))
    print()
    print("--- prompt sent to the vision tiers (tiles) ---")
    print(build_prompt("click each image containing a boat", "tiles", 9))

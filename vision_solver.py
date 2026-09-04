#!/usr/bin/env python3
"""vision_solver.py — Roboflow (Gemini 3.6 Flash) vision solver for hCaptcha.

Every visual answer in this stack comes from a Roboflow Workflow running
Google Gemini 3.6 Flash. There is no local "brain" checkpoint, no CNN
weights and no self-hosted model server: the bot

  1. reads the challenge prompt ("Please select all images with a boat")
     from the hCaptcha challenge frame,
  2. screenshots every tile of the image grid (or the whole canvas),
  3. POSTs each image to the Roboflow workflow together with the
     challenge question,
  4. Gemini answers with the tiles / points / drag to perform,
  5. the bot clicks or drags, and hCaptcha itself mints the token.

Transport — Roboflow Serverless Workflows:

    POST https://serverless.roboflow.com/infer/workflows/<WS>/<WORKFLOW>
    {
      "api_key": "<API_KEY>",
      "inputs": {
        "image":   {"type": "base64", "value": "<b64 jpeg>"},
        "classes": ["boat"],          # object-detection workflows
        "prompt":  "<the question>"   # passed when the workflow takes it
      }
    }

The workflow is an **object-detection** workflow: it returns predictions
with `x`, `y`, `width`, `height`, `class` and `confidence` in PIXELS of
the submitted image. `_predictions_to_points()` normalises them to 0-1
so the rest of the solver is unchanged. The literal challenge question
is ALWAYS sent along with the image (as `prompt`, and mirrored into
`classes`), so Gemini is answering the captcha's own wording rather than
a generic label list.

Configuration (env vars):

  API_KEY             Roboflow API key. REQUIRED. Sent as `api_key` in the
                      request body (ROBOFLOW_API_KEY is accepted too).
  ROBOFLOW_WORKSPACE  workspace slug (default: alistra742-gmail-com)
  ROBOFLOW_WORKFLOW   workflow id (default: gemini-3-6-flash-object-detection)
  ROBOFLOW_API_BASE   serverless base URL
                      (default https://serverless.roboflow.com)
  GOOGLE_API_KEY      optional Google AI Studio key; when set it is sent
                      as `model_api_key` so inference bills your Google
                      account instead of Roboflow credits.
  ROBOFLOW_TIMEOUT    per-request timeout in seconds (default 60)
  ROBOFLOW_TILE_TIMEOUT  per-tile timeout for grid rounds (default 25)
  ROBOFLOW_IMAGE_SIDE    max image side in px sent up (default 640)
  ROBOFLOW_MIN_CONF   minimum prediction confidence to keep (default 0.30)
  ROBOFLOW_CHECK_TIMEOUT  readiness-probe timeout (default 60)

  RTDETR_ENABLED      set to 0 to disable the backup detector (default on)
  RTDETR_MODEL_ID     backup detector alias (default rfdetr-small — a
                      BUILT-IN COCO-pretrained model on the same host and
                      the same API_KEY; nothing to deploy or pull)
  RTDETR_TIMEOUT      per-image backup timeout in seconds (default 30)
  RTDETR_MIN_CONF     minimum backup detection confidence (default 0.35)

BACKUP PATH. Whenever the Gemini workflow fails — unreachable, rate
limited, out of credits, or an unparseable answer — ``solve()`` retries
the round against RF-DETR via ``solve_rtdetr()``. RF-DETR is COCO-80 and
cannot read a prompt, so the KNOWLEDGE BASE does that job:
``coco_targets()`` resolves the prompt through hcaptcha_types' ~1700-entry
alias table (helicopter -> airplane, nightstand -> dining table, "an
animal" -> every COCO animal) down to COCO labels. Prompts with no COCO
equivalent make it ABSTAIN rather than answer with the wrong object.

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

def _env_url(name: str, *fallbacks: str) -> str:
    """Read a URL env var, tolerating a pasted ``NAME = value`` line.

    Railway's variable editor makes it easy to paste the whole
    ``GEMMA_BASE = http://host:11434`` line into the VALUE box, which
    yields a string aiohttp cannot parse. Strip that prefix, surrounding
    quotes and trailing slashes so the tier still comes up.
    """
    raw = ""
    for key in (name,) + fallbacks:
        raw = os.environ.get(key, "").strip()
        if raw:
            break
    if not raw:
        return ""
    # "GEMMA_BASE = http://..." / "GEMMA_BASE=http://..."
    m = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.+)$", raw)
    if m and "://" in m.group(1):
        raw = m.group(1).strip()
    raw = raw.strip("'\"").strip()
    return raw.rstrip("/")


# ── Roboflow configuration ───────────────────────────────────────────────
ROBOFLOW_API_BASE = (_env_url("ROBOFLOW_API_BASE")
                     or "https://serverless.roboflow.com")

# API_KEY is the canonical name (that is what the deploy sets);
# ROBOFLOW_API_KEY is accepted so Roboflow's own convention also works.
API_KEY = (os.environ.get("API_KEY", "").strip()
           or os.environ.get("ROBOFLOW_API_KEY", "").strip())

# Optional BYO provider key: Gemini 3.6 Flash workflows accept a
# `model_api_key` runtime parameter (Google AI Studio key). Omitted when
# unset — inference then runs on Roboflow credits.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()

ROBOFLOW_WORKSPACE = (os.environ.get("ROBOFLOW_WORKSPACE", "").strip()
                      or "alistra742-gmail-com")
# Candidate workflow ids, tried in order until one probes healthy. Both
# of the user's workflows are listed because the Roboflow "solutions/chat"
# URLs do not make it obvious which slug the serverless API expects; the
# probe settles it at runtime instead of us guessing. ROBOFLOW_WORKFLOW
# pins a single id and skips the search.
ROBOFLOW_WORKFLOW_CANDIDATES = tuple(
    w.strip() for w in os.environ.get(
        "ROBOFLOW_WORKFLOWS",
        "gemini-3-6-flash-object-detection,"
        "coco-50-object-counter-1788536417919,"
        "gemini-3-6-flash,"
        "coco-50-object-counter").split(",") if w.strip())
ROBOFLOW_WORKFLOW = (os.environ.get("ROBOFLOW_WORKFLOW", "").strip()
                     or ROBOFLOW_WORKFLOW_CANDIDATES[0])

ROBOFLOW_TIMEOUT = float(os.environ.get("ROBOFLOW_TIMEOUT", "60"))
ROBOFLOW_CHECK_TIMEOUT = float(os.environ.get("ROBOFLOW_CHECK_TIMEOUT", "60"))
_TILE_TIMEOUT = float(os.environ.get("ROBOFLOW_TILE_TIMEOUT", "25"))
_MAX_SIDE = int(os.environ.get("ROBOFLOW_IMAGE_SIDE", "640"))
_MIN_CONF = float(os.environ.get("ROBOFLOW_MIN_CONF", "0.30"))

# Model label, for logs only — the workflow decides the real model.
MODEL_NAME = "gemini-3.6-flash (roboflow)"

# ── RT-DETR backup detector ──────────────────────────────────────────────
# When the Gemini workflow fails (down, rate limited, quota, unparseable),
# the solver retries the round against RF-DETR, Roboflow's real-time
# transformer detector. `rfdetr-small` is a BUILT-IN COCO-pretrained alias
# on the same serverless host and the same API_KEY: no service to create,
# no weights to pull, no workflow to build.
#
#     POST {base}/infer/object_detection
#     {"api_key": ..., "model_id": "rfdetr-small",
#      "image": {"type": "base64", "value": ...}}
#
# It is COCO-80 only, so it cannot read a prompt. The knowledge base
# (hcaptcha_types SYNONYMS/aliases, ~1700 nouns) is what makes it usable:
# `coco_targets()` walks the prompt noun -> canonical class -> COCO label,
# and a detection of any mapped label counts as a hit.
RTDETR_ENABLED = os.environ.get("RTDETR_ENABLED", "1").strip() not in (
    "0", "false", "no", "off")
RTDETR_MODEL_ID = (os.environ.get("RTDETR_MODEL_ID", "").strip()
                   or "rfdetr-small")
RTDETR_TIMEOUT = float(os.environ.get("RTDETR_TIMEOUT", "30"))
RTDETR_MIN_CONF = float(os.environ.get("RTDETR_MIN_CONF", "0.35"))

# The 80 COCO classes rfdetr-small can emit.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Canonical knowledge-base class -> COCO label(s) rfdetr-small can detect.
# Only visually defensible mappings: a "helicopter" tile is genuinely
# airplane-shaped to COCO, a "nightstand" reads as dining table. Anything
# not here has no COCO equivalent and RT-DETR abstains on it.
_KB_TO_COCO = {
    "airplane": ("airplane",), "bicycle": ("bicycle",), "car": ("car",),
    "motorcycle": ("motorcycle",), "bus": ("bus",), "train": ("train",),
    "truck": ("truck",), "boat": ("boat",), "traffic_light": ("traffic light",),
    "red_light": ("traffic light",), "fire_hydrant": ("fire hydrant",),
    "stop_sign": ("stop sign",), "parking_meter": ("parking meter",),
    "bird": ("bird",), "cat": ("cat",), "dog": ("dog",), "horse": ("horse",),
    "sheep": ("sheep",), "cow": ("cow",), "elephant": ("elephant",),
    "bear": ("bear", "teddy bear"), "zebra": ("zebra",), "giraffe": ("giraffe",),
    "umbrella": ("umbrella",), "backpack": ("backpack",), "tie": ("tie",),
    "suitcase": ("suitcase",), "frisbee": ("frisbee",), "ski": ("skis",),
    "snowboard": ("snowboard",), "ball": ("sports ball",),
    "tennis_ball": ("sports ball",), "basketball": ("sports ball",),
    "baseball": ("sports ball",), "football": ("sports ball",),
    "volleyball": ("sports ball",), "golf_ball": ("sports ball",),
    "kite": ("kite",), "skateboard": ("skateboard",),
    "surfboard": ("surfboard",), "racket": ("tennis racket",),
    "bottle": ("bottle",), "cup": ("cup", "wine glass"), "mug": ("cup",),
    "fork": ("fork",), "knife": ("knife",), "spoon": ("spoon",),
    "bowl": ("bowl",), "banana": ("banana",), "apple": ("apple",),
    "sandwich": ("sandwich",), "burger": ("sandwich",), "orange": ("orange",),
    "broccoli": ("broccoli",), "carrot": ("carrot",), "hotdog": ("hot dog",),
    "pizza": ("pizza",), "donut": ("donut",), "cake": ("cake",),
    "chair": ("chair", "couch", "bench"), "sofa": ("couch",),
    "bench": ("bench",), "bed": ("bed",),
    "table": ("dining table",), "nightstand": ("dining table",),
    "desk": ("dining table",), "tv": ("tv",), "monitor": ("tv",),
    "laptop": ("laptop",), "computer": ("laptop",), "mouse": ("mouse",),
    "keyboard": ("keyboard",), "phone": ("cell phone",),
    "remote_control": ("remote",), "microwave": ("microwave",),
    "oven": ("oven",), "sink": ("sink",), "fridge": ("refrigerator",),
    "book": ("book",), "clock": ("clock",), "watch": ("clock",),
    "vase": ("vase",), "scissors": ("scissors",),
    "toothbrush": ("toothbrush",), "flower": ("potted plant",),
    "tree": ("potted plant",), "person": ("person",),
}


def coco_targets(prompt: str) -> tuple:
    """COCO labels that satisfy ``prompt``, via the knowledge base.

    Resolution order: the canonical noun (extract_target walks the ~1700
    entry alias table), then the set predicates ("click each image with an
    animal" -> every COCO animal), then the set-down surfaces. Returns ()
    when the prompt has no COCO equivalent — the caller must then abstain
    rather than guess.
    """
    p = (prompt or "").strip()
    if not p:
        return ()
    labels: list = []

    def add(name):
        for lab in _KB_TO_COCO.get(name, ()):
            if lab not in labels:
                labels.append(lab)

    try:
        import hcaptcha_types as hct
    except Exception:
        return ()

    target = hct.extract_target(p)
    if target:
        add(target)
    if labels:
        return tuple(labels)

    # Set predicates: animals / edible / wheeled / motorised / surfaces.
    low = p.lower()
    groups = (
        (("animal", "animals"), getattr(hct, "ANIMALS", ())),
        (("eat", "edible", "food"), getattr(hct, "EDIBLE", ())),
        (("wheel", "wheels", "wheeled"), getattr(hct, "WHEELED", ())),
        (("motor", "motorised", "motorized", "engine"),
         getattr(hct, "MOTORISED", ())),
    )
    for words, members in groups:
        if any(re.search(r"\b%s\b" % w, low) for w in words):
            for name in members:
                add(name)
    if not labels and hct.is_setdown_prompt(p):
        for name in getattr(hct, "FLAT_SURFACES", ()):
            add(name)
    return tuple(labels)


def shrink_image(data: bytes, max_side: int = _MAX_SIDE) -> bytes:
    """Downscale a PNG/JPEG to a JPEG small enough for a fast round trip."""
    if not data:
        return data
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((max(32, int(max_side)), max(32, int(max_side))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85, optimize=True)
        out = buf.getvalue()
        return out or data
    except Exception:
        return data


def image_size(data: bytes):
    """(width, height) of an encoded image, or None."""
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return None


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ── Tier 3: Gemma vision fallback (Ollama-compatible) ────────────────────
# Last resort, used only when BOTH the Gemini workflow and rfdetr-small
# fail. Unlike RT-DETR this is a real VLM: it reads the prompt, so it can
# answer the reasoning rounds COCO has no class for ("odd one out", most
# trees/tools). Slower than either, hence last.
#
# Ollama cannot run inside the app container, so point GEMMA_BASE at
# wherever `ollama serve` actually lives:
#   GEMMA_BASE=http://ollama.railway.internal:11434   (Railway)
#   GEMMA_BASE=http://<host>:11434                    (self-hosted)
# On Railway the ollama service MUST set OLLAMA_HOST=[::]:11434 — the
# private network is IPv6-only and Ollama otherwise binds IPv4 loopback.
# Empty GEMMA_BASE disables the tier entirely.
GEMMA_BASE = _env_url("GEMMA_BASE", "OLLAMA_BASE")
GEMMA_MODEL = (os.environ.get("GEMMA_MODEL", "").strip() or "gemma3:4b")
GEMMA_TIMEOUT = float(os.environ.get("GEMMA_TIMEOUT", "45"))
GEMMA_TILE_TIMEOUT = float(os.environ.get("GEMMA_TILE_TIMEOUT", "30"))
# Geometry rounds (drag/points/bbox): a 4B VLM is slow and unreliable here.
GEMMA_GEOMETRY_TIMEOUT = float(os.environ.get("GEMMA_GEOMETRY_TIMEOUT", "20"))
GEMMA_ENABLED = (os.environ.get("GEMMA_ENABLED", "1").strip()
                 not in ("0", "false", "no", "off"))


# ── drag geometry ────────────────────────────────────────────────────────
# "Drag the icon to the place where it fits" is NOT a detection problem.
# A detector returns a bag of boxes with confidences; nothing in that
# ranking says which box is the loose piece and which is the hole. Picking
# pts[0] as source and pts[1] as destination — the old behaviour — is a
# coin flip that also silently swaps the two.
#
# The geometry does say, though. hCaptcha drag rounds are built the same
# way every time:
#   * the DRAGGABLE piece is small, high-contrast, and sits apart from the
#     scene — usually in a side rail or a corner, often badged "Move";
#   * the TARGET is a hole/socket/empty cell INSIDE the main composition,
#     and it is the odd region out: the one gap in an otherwise regular
#     layout.
# So: score every detection on "how much does this look like a loose
# piece" vs "how much does this look like a hole", then pair the best of
# each. That is deterministic, explainable, and needs no extra model.

_PIECE_WORDS = ("piece", "icon", "shape", "block", "tile", "puzzle", "move",
                "draggable", "loose", "item", "object", "fragment", "part")
_HOLE_WORDS = ("hole", "slot", "socket", "gap", "empty", "space", "target",
               "place", "outline", "silhouette", "missing", "cell", "blank",
               "dashed", "shadow", "well", "receptacle")


def _label_bias(label: str) -> float:
    """+1 -> looks like the loose piece, -1 -> looks like the hole."""
    lab = str(label or "").lower()
    score = 0.0
    if any(w in lab for w in _PIECE_WORDS):
        score += 1.0
    if any(w in lab for w in _HOLE_WORDS):
        score -= 1.0
    return score


def score_drag_roles(points):
    """Rank detections as (piece_score, hole_score) per detection.

    Pure geometry + label heuristics on the normalised
    ``(x, y, w, h, conf, label)`` tuples ``predictions_to_points`` yields.
    Returns ``[(idx, piece_score, hole_score), ...]``.
    """
    pts = list(points or [])
    if not pts:
        return []
    areas = [max(p[2] * p[3], 1e-6) for p in pts]
    median_area = sorted(areas)[len(areas) // 2]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    out = []
    for i, p in enumerate(pts):
        x, y, w, h, conf, label = p
        area = areas[i]
        # Distance from the crowd: a loose piece is set apart, a hole is
        # embedded in the composition.
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        # Edge proximity: draggables live in rails and margins.
        edge = min(x, 1.0 - x, y, 1.0 - y)
        small = median_area / (area + 1e-6)      # >1 when smaller than most
        bias = _label_bias(label)

        piece = (1.6 * dist + 1.2 * max(0.0, 0.25 - edge) * 4.0
                 + 0.6 * min(small, 3.0) + 0.8 * conf + 1.5 * bias)
        hole = (1.4 * (1.0 - min(dist * 2.0, 1.0))
                + 1.0 * min(edge * 4.0, 1.0)
                + 0.5 * conf - 1.5 * bias)
        out.append((i, piece, hole))
    return out


def pair_drag(points):
    """Choose (source, destination) centres for a drag round.

    Picks the detection that scores highest as a loose piece, then the
    best remaining hole candidate — never the same box twice. Returns
    ``((sx, sy), (dx, dy))`` or ``None``.
    """
    pts = list(points or [])
    if len(pts) < 2:
        return None
    scored = score_drag_roles(pts)
    src_i = max(scored, key=lambda t: t[1])[0]
    dst_i = max((t for t in scored if t[0] != src_i),
                key=lambda t: t[2])[0]
    src, dst = pts[src_i], pts[dst_i]
    return (src[0], src[1]), (dst[0], dst[1])


# ── question rewriting ───────────────────────────────────────────────────
# hCaptcha speaks to humans ("Please click on all the dogs"). A detector
# needs to be spoken to in boxes ("Box and coordinate every dog"). These
# rules do that rewrite: they strip the human-interface verb, keep the
# TARGET NOUN intact, and restate the task as localisation.
#
# Order matters — longest/most specific patterns first, so "click and drag"
# is not eaten by the plain "click" rule.
_VERB_REWRITES = (
    # drag / move phrasings -> two-point localisation
    (r"^\s*(?:please\s+)?(?:click\s+and\s+)?drag\s+(?:the|each|all|every)?\s*",
     "Box and coordinate the "),
    (r"^\s*(?:please\s+)?move\s+(?:the|each|all|every)?\s*",
     "Box and coordinate the "),
    # click / select / tap / choose / pick -> box every instance
    (r"^\s*(?:please\s+)?click\s+(?:on\s+)?(?:each|all|every)\s+(?:of\s+the\s+)?"
     r"(?:image[s]?\s+(?:that\s+)?(?:contain(?:ing|s)?|with|showing)\s+)?",
     "Box and coordinate every "),
    (r"^\s*(?:please\s+)?select\s+(?:each|all|every)\s+(?:of\s+the\s+)?"
     r"(?:image[s]?\s+(?:that\s+)?(?:contain(?:ing|s)?|with|showing)\s+)?",
     "Box and coordinate every "),
    (r"^\s*(?:please\s+)?(?:click|tap|touch|press)\s+(?:on\s+)?(?:the\s+)?",
     "Box and coordinate the "),
    (r"^\s*(?:please\s+)?(?:select|choose|pick|mark|identify|find|locate)"
     r"\s+(?:the\s+)?",
     "Box and coordinate the "),
)

# Task-shape suffixes: exactly what the model must return, in coordinates.
_SHAPE_INSTRUCTION = {
    "tiles": ("Report a bounding box and centre coordinate for every "
              "matching object you can see. If nothing in this image "
              "matches, return no detections."),
    "points": ("Report a tight bounding box and a centre coordinate for "
               "EACH matching object. Coordinates are normalised 0.0-1.0 "
               "fractions of the image, origin top-left."),
    "bbox": ("Report ONE tight bounding box around the single best match, "
             "with its centre coordinate, normalised 0.0-1.0."),
    "drag": ("This is a DRAG puzzle with exactly two answers. FIRST, box "
             "the loose draggable piece: it is the small icon or shape "
             "sitting APART from the main picture, usually in a side rail, "
             "a corner or a tray, sometimes marked Move. SECOND, box the "
             "place it belongs: the hole, socket, dashed outline, shadow "
             "or empty cell INSIDE the main picture whose shape MATCHES "
             "that piece. Do not box the finished/filled shapes. Label the "
             "first 'piece' and the second 'hole', and give the centre of "
             "each as normalised 0.0-1.0 coordinates."),
    "pattern": ("This is a PATTERN-COMPLETION puzzle. Box the candidate icon that completes the sequence (label it 'piece'), and box the EMPTY cell in the grid where it belongs (label it 'hole'). The empty cell is the one cell with no symbol in it. Centres normalised 0.0-1.0."),
    "tower": ("This is a BLOCK-STACK puzzle. Box the loose block segment that is not part of any tower (label it 'piece' — it usually sits to one side and may carry a Move badge), and box the INCOMPLETE tower it must be dropped on (label it 'hole' — the stack that is shorter than the others or has a visible gap). Centres normalised 0.0-1.0."),
    "stack": ("Box and coordinate every loose block and every stack top it "
              "must go to, as {\"drags\": [[sx, sy, tx, ty], ...]} in 0-100 "
              "percent coordinates."),
    "count": ("Box every matching object, then answer with the JSON object "
              "{\"count\": N} where N is how many you boxed."),
    "choice": ("Identify the correct option and answer with the JSON object "
               "{\"choice\": N} using the 1-based option number."),
    "text": ("Read the characters shown and answer with the text only."),
}


def rewrite_question(prompt: str, shape: str = "tiles") -> str:
    """Turn a human hCaptcha instruction into a detector instruction.

    hCaptcha writes for a mouse user: *"Please click on all the dogs"*.
    A detection model answers boxes, so it is asked instead for
    *"Box and coordinate every dog."* The target noun is never touched —
    only the interface verb around it — so the knowledge base and the
    workflow's ``classes`` list still see the real object.

    Examples
    --------
    >>> rewrite_question("Please click on all the dogs")
    'Box and coordinate every dogs. Report a bounding box ...'
    >>> rewrite_question("Drag the shape to where it fits", "drag")
    'Box and coordinate the shape to where it fits. Box and coordinate TWO ...'
    """
    p = " ".join((prompt or "").split())
    if not p:
        return ""
    body = p
    for pattern, replacement in _VERB_REWRITES:
        new_body, n = re.subn(pattern, replacement, body, count=1,
                              flags=re.I)
        if n:
            body = new_body
            break
    else:
        # No known verb: state the task explicitly rather than passing a
        # bare noun phrase the model might answer in prose.
        body = f"Box and coordinate the following in this image: {body}"

    # Clean up articles the verb strip leaves stranded: "every the dogs"
    # -> "every dog", "every a bus" -> "every bus".
    body = re.sub(r"\b(every|the)\s+(?:the|a|an|all|any)\b", r"\1", body,
                  flags=re.I)
    body = re.sub(r"\s{2,}", " ", body)
    body = body.rstrip(" .")
    # Re-capitalise and re-punctuate.
    if body:
        body = body[0].upper() + body[1:]
    suffix = _SHAPE_INSTRUCTION.get(shape, _SHAPE_INSTRUCTION["tiles"])
    return f"{body}. {suffix}"


def detection_classes(prompt: str) -> List[str]:
    """Classes to ask the object-detection workflow for.

    The captcha's own wording is always included so Gemini sees the real
    question; the canonical noun (when hcaptcha_types can extract one) is
    added as a short, detector-friendly alias.
    """
    p = (prompt or "").strip()
    out: List[str] = []
    try:
        import hcaptcha_types as hct
        target = hct.extract_target(p)
        if target:
            out.append(str(target).replace("_", " "))
    except Exception:
        pass
    noun = _prompt_noun(p)
    if noun and noun not in out:
        out.append(noun)
    if p and p not in out:
        out.append(p[:200])
    return out or ["object"]


_STOPWORDS = {
    "please", "click", "select", "choose", "pick", "check", "mark", "tap",
    "all", "each", "every", "the", "a", "an", "of", "with", "that", "which",
    "image", "images", "picture", "pictures", "photo", "photos", "tile",
    "tiles", "containing", "contain", "contains", "show", "shows", "showing",
    "in", "on", "is", "are", "to", "and", "or", "square", "squares", "this",
}


def _prompt_noun(prompt: str) -> str:
    """Best-effort short noun phrase from the prompt (detector-friendly)."""
    words = re.findall(r"[a-z]+", (prompt or "").lower())
    keep = [w for w in words if w not in _STOPWORDS]
    return " ".join(keep[:3])


_YES_WORDS = ("yes", "y", "yeah", "yep", "yup", "true", "match", "matching")

_NO_WORDS = ("no", "n", "nope", "nah", "false", "none")


def parse_yesno(text: str):
    """Map a yes/no reply to True/False/None.

    Models often echo the question (\"… Answer yes or no.\"). Strip that
    instruction and read the first/last token so an echoed prompt is not
    scored as a \"no\".
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
    """The question sent alongside a single grid tile.

    Phrased so a detection workflow has a concrete thing to find and a
    VLM step can still answer it with yes/no.
    """
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

_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]")
_JSON_STRING_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')


class RoboflowVisionClient:
    """Async client for a Roboflow Workflow running Gemini 3.6 Flash.

    One HTTP POST per image:

        POST {base}/infer/workflows/{workspace}/{workflow}
        {"api_key": ..., "inputs": {"image": {...}, "prompt": ..., ...}}

    The workflow is an object-detection workflow, so the reply carries
    `predictions` in pixel coordinates. Grid rounds ask ONE TILE AT A TIME
    (a detection on tile N means tile N matches); single-canvas rounds
    (points / bbox / drag / tower / pattern / count) map the detections
    straight onto the answer shape. When the workflow instead returns free
    text (a VLM/captioning step), the existing JSON parsers handle it.
    """

    def __init__(self, log: Optional[Callable] = None,
                 base: str = ROBOFLOW_API_BASE,
                 workspace: str = ROBOFLOW_WORKSPACE,
                 workflow: str = ROBOFLOW_WORKFLOW,
                 api_key: str = "",
                 google_api_key: str = "",
                 rtdetr_model_id: str = "",
                 rtdetr_enabled: Optional[bool] = None,
                 gemma_base: Optional[str] = None,
                 gemma_model: str = ""):
        self._log = log or (lambda msg, level="info": None)
        self.base = (base or ROBOFLOW_API_BASE).rstrip("/")
        self.workspace = workspace or ROBOFLOW_WORKSPACE
        self.workflow = workflow or ROBOFLOW_WORKFLOW
        # Every workflow id worth trying. If the caller pinned one
        # explicitly, honour it and do not search.
        if workflow or os.environ.get("ROBOFLOW_WORKFLOW", "").strip():
            self.workflow_candidates = (self.workflow,)
        else:
            seen, cands = set(), []
            for w in (self.workflow,) + tuple(ROBOFLOW_WORKFLOW_CANDIDATES):
                if w and w not in seen:
                    seen.add(w)
                    cands.append(w)
            self.workflow_candidates = tuple(cands)
        self._api_key = api_key or API_KEY
        self._google_api_key = google_api_key or GOOGLE_API_KEY
        self.model = MODEL_NAME
        # RT-DETR backup detector (COCO-80, knowledge-base driven).
        self.rtdetr_model_id = rtdetr_model_id or RTDETR_MODEL_ID
        self.rtdetr_enabled = (RTDETR_ENABLED if rtdetr_enabled is None
                               else bool(rtdetr_enabled))
        # Tier 3: Gemma VLM, only reachable when GEMMA_BASE is configured.
        self.gemma_base = (gemma_base if gemma_base is not None
                           else GEMMA_BASE).rstrip("/")
        self.gemma_model = gemma_model or GEMMA_MODEL
        self.gemma_enabled = bool(GEMMA_ENABLED and self.gemma_base)
        self.stats = {"calls": 0, "ok": 0, "failed": 0, "rtdetr": 0,
                      "gemma": 0}
        # Machine-readable result of the latest reachability probe. Keeps
        # the public ``check() -> (ok, models)`` contract, while letting
        # callers separate a transient connection failure from a
        # deterministic configuration error such as HTTP 401 or 404.
        self.last_check_error = ""
        self.last_check_http_status: Optional[int] = None
        if not self._api_key:
            self._log("[Vision] API_KEY is not set — Roboflow will reject "
                      "every request", level="error")
        self._log(f"[Vision] Roboflow workflow: {self.workspace}/"
                  f"{self.workflow} @ {self.base} ({self.model})"
                  f"{' [BYO google key]' if self._google_api_key else ''}")

    @property
    def endpoint(self) -> str:
        """Workflow run URL."""
        return f"{self.base}/infer/workflows/{self.workspace}/{self.workflow}"

    @property
    def configured(self) -> bool:
        return bool(self.base and self.workspace and self.workflow
                    and self._api_key)

    def _inputs(self, image: bytes, question: str,
                classes: Optional[List[str]] = None) -> dict:
        """Workflow inputs: the image AND the question, every call."""
        inputs = {
            "image": {"type": "base64", "value": _b64(image)},
            # The literal captcha question. Different Gemini workflow
            # templates name this input differently, so send the common
            # aliases — a workflow ignores inputs it does not declare.
            "prompt": question,
            "query": question,
            "question": question,
            "classes": list(classes or detection_classes(question)),
        }
        if self._google_api_key:
            inputs["model_api_key"] = self._google_api_key
        return inputs

    async def check(self) -> tuple:
        """Probe every candidate workflow and keep the first healthy one.

        The Roboflow console shows workflows under `solutions/chat` URLs
        whose slug is not necessarily the id the serverless API wants, so
        rather than guessing a single value we try each candidate and
        adopt whichever answers. Falls back to the RT-DETR backup when no
        workflow works but the key is still good.
        """
        candidates = getattr(self, "workflow_candidates", (self.workflow,))
        first_error = first_status = None
        for i, wf in enumerate(candidates):
            self.workflow = wf
            ok, models = await self._check_one()
            if ok:
                if i:
                    self._log(f"[Vision] Using workflow {wf!r} "
                              f"(candidate {i + 1}/{len(candidates)})")
                return True, models
            if first_error is None:
                first_error = self.last_check_error
                first_status = self.last_check_http_status
            # A bad key fails identically for every workflow — stop early.
            if self.last_check_error == "authentication":
                return False, []
            if len(candidates) > 1 and i < len(candidates) - 1:
                self._log(f"[Vision] Workflow {wf!r} not usable "
                          f"({self.last_check_error or 'error'}) — trying "
                          f"{candidates[i + 1]!r}", level="warn")
        # Nothing worked: restore the preferred id and report the first
        # failure, but let the backup detector rescue the round.
        self.workflow = candidates[0]
        self.last_check_error = first_error or "http"
        self.last_check_http_status = first_status
        if await self.check_rtdetr():
            self._log("[Vision] No workflow reachable — running in "
                      "BACKUP-ONLY mode on the RT-DETR detector",
                      level="warn")
            self.last_check_error = ""
            return True, [self.rtdetr_model_id]
        if await self.check_gemma():
            self._log(f"[Vision] Roboflow fully unavailable — running on "
                      f"the Gemma tier ({self.gemma_model})", level="warn")
            self.last_check_error = ""
            return True, [self.gemma_model]
        return False, []

    async def _check_one(self) -> tuple:
        """Probe the workflow endpoint.

        Returns ``(ok, models)`` where ``models`` is ``[self.model]`` when
        the workflow is reachable (kept for callers written against the old
        contract). ``last_check_error`` classifies failures so callers do
        not mistake an HTTP 401 (bad API_KEY) for a cold start.
        """
        self.last_check_error = ""
        self.last_check_http_status = None
        if not self._api_key:
            self.last_check_error = "authentication"
            self._log("[Vision] No API_KEY configured for Roboflow",
                      level="error")
            return False, []
        # Probe the RUN url itself, not describe_interface: the serverless
        # host does not expose describe_interface for every workflow and
        # answers 405 there, which tells us nothing about the workflow. A
        # run with no inputs returns 4xx-with-a-body when the workflow
        # exists and 404 when it does not.
        url = self.endpoint
        payload = {"api_key": self._api_key, "inputs": {}}
        try:
            timeout = aiohttp.ClientTimeout(total=ROBOFLOW_CHECK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.post(url, json=payload) as r:
                    status = int(r.status)
                    try:
                        body = (await r.text())[:200]
                    except Exception:
                        body = ""
            if status == 200:
                self._log(f"[Vision] Roboflow OK: {self.workspace}/"
                          f"{self.workflow}")
                return True, [self.model]
            self.last_check_http_status = status
            if status in (401, 403):
                self.last_check_error = "authentication"
                self._log(f"[Vision] Roboflow rejected API_KEY (HTTP {status}) "
                          "— check the key at app.roboflow.com/settings/api",
                          level="error")
            elif status == 404:
                self.last_check_error = "protocol"
                self._log(f"[Vision] Workflow {self.workspace}/{self.workflow} "
                          "not found (HTTP 404) — check ROBOFLOW_WORKSPACE / "
                          "ROBOFLOW_WORKFLOW", level="error")
            elif status in (400, 422):
                # The workflow RESOLVED and validated our (empty) inputs —
                # that is exactly what a healthy workflow does here.
                self._log(f"[Vision] Roboflow OK: {self.workspace}/"
                          f"{self.workflow} (validated inputs)")
                return True, [self.model]
            elif status == 405:
                self.last_check_error = "method"
                self._log(f"[Vision] {self.workflow!r} returned HTTP 405 "
                          "(method not allowed)", level="warn")
            elif status == 429:
                self.last_check_error = "rate_limit"
                self._log("[Vision] Roboflow rate limited (HTTP 429)",
                          level="warn")
            elif status >= 500:
                self.last_check_error = "server"
                self._log(f"[Vision] Roboflow server error HTTP {status}: "
                          f"{body}", level="warn")
            else:
                self.last_check_error = "http"
                self._log(f"[Vision] Roboflow HTTP {status}: {body}",
                          level="warn")
            return False, []
        except asyncio.TimeoutError:
            self.last_check_error = "timeout"
            self._log(f"[Vision] Probe timed out after "
                      f"{ROBOFLOW_CHECK_TIMEOUT:g}s at {url}", level="error")
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

    async def check_rtdetr(self) -> bool:
        """Is the rfdetr-small backup reachable with this API_KEY?

        Uses a 1x1 PNG so the probe is cheap. Any 200 means the alias is
        served and the key is accepted.
        """
        if not self._api_key:
            return False
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
        data = await self._rtdetr_post(png, RTDETR_TIMEOUT)
        return data is not None

    async def _rtdetr_post(self, image: bytes, timeout: float):
        """POST an image to the hosted alias. Returns parsed JSON or None.

        Tries the documented alias form first (base64 body, key in the
        query string), then the workflow-style JSON form as a fallback so
        a self-hosted inference server also works.
        """
        b64 = _b64(image)
        mid = self.rtdetr_model_id
        attempts = (
            # Documented alias form: base64 body, key in the query string.
            ("alias", f"{self.base}/{mid}?api_key={self._api_key}",
             {"data": b64,
              "headers": {"Content-Type":
                          "application/x-www-form-urlencoded"}}),
            # Some accounts serve aliases under /infer/<model_id>.
            ("infer-alias", f"{self.base}/infer/{mid}?api_key={self._api_key}",
             {"data": b64,
              "headers": {"Content-Type":
                          "application/x-www-form-urlencoded"}}),
            # Self-hosted inference servers take the JSON form.
            ("json", f"{self.base}/infer/object_detection",
             {"json": {"api_key": self._api_key,
                       "model_id": mid,
                       "image": {"type": "base64", "value": b64}}}),
        )
        last = ""
        for name, url, kw in attempts:
            try:
                cfg = aiohttp.ClientTimeout(total=timeout)
                async with aiohttp.ClientSession(timeout=cfg) as s:
                    async with s.post(url, **kw) as r:
                        if r.status == 200:
                            return await r.json()
                        last = f"{name} HTTP {r.status}: {(await r.text())[:160]}"
            except Exception as e:
                last = f"{name} {type(e).__name__}"
        if last:
            self._log(f"[RT-DETR] {last}", level="warn")
        return None

    async def _run(self, image: bytes, question: str, timeout: float,
                   classes: Optional[List[str]] = None) -> Optional[dict]:
        """One workflow run. Returns the parsed JSON body or None."""
        payload = {"api_key": self._api_key,
                   "inputs": self._inputs(image, question, classes)}
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_cfg) as s:
                async with s.post(self.endpoint, json=payload) as r:
                    if r.status != 200:
                        body = await r.text()
                        self._log(f"[Vision] Workflow rejected "
                                  f"(HTTP {r.status}): {body[:200]}",
                                  level="warn")
                        return None
                    return await r.json()
        except Exception as e:
            self._log(f"[Vision] Workflow error: {e}", level="error")
            return None

    # ── response readers ────────────────────────────────────────────────

    @staticmethod
    def _walk(node, out_pred: list, out_text: list) -> None:
        """Collect every predictions[] list and every string leaf."""
        if isinstance(node, dict):
            preds = node.get("predictions")
            if isinstance(preds, list):
                out_pred.extend(p for p in preds if isinstance(p, dict))
            elif isinstance(preds, dict) and isinstance(
                    preds.get("predictions"), list):
                out_pred.extend(p for p in preds["predictions"]
                                if isinstance(p, dict))
            for k, v in node.items():
                if k == "predictions":
                    continue
                if isinstance(v, str):
                    if len(v) <= 4000 and k not in ("type", "format",
                                                    "parent_id", "image_id"):
                        out_text.append(v)
                else:
                    RoboflowVisionClient._walk(v, out_pred, out_text)
        elif isinstance(node, list):
            for v in node:
                RoboflowVisionClient._walk(v, out_pred, out_text)

    @classmethod
    def read_response(cls, data) -> tuple:
        """(predictions, texts) from any Roboflow workflow response shape."""
        preds: list = []
        texts: list = []
        cls._walk(data, preds, texts)
        return preds, texts

    @staticmethod
    def predictions_to_points(preds: list, size, min_conf: float = _MIN_CONF):
        """Pixel detections -> [(x, y, w, h, conf, label)] normalised 0-1.

        Roboflow gives `x`/`y` as the box CENTRE in pixels. Boxes already
        expressed in 0-1 (some VLM blocks do) are passed through.
        """
        w = float((size or (0, 0))[0]) or 0.0
        h = float((size or (0, 0))[1]) or 0.0
        out = []
        for p in preds or []:
            try:
                conf = float(p.get("confidence", p.get("score", 1.0)))
            except (TypeError, ValueError):
                conf = 1.0
            if conf < min_conf:
                continue
            try:
                px = float(p.get("x"))
                py = float(p.get("y"))
            except (TypeError, ValueError):
                continue
            pw = float(p.get("width", 0) or 0)
            ph = float(p.get("height", 0) or 0)
            if w > 0 and h > 0 and (px > 1.0 or py > 1.0):
                px, py, pw, ph = px / w, py / h, pw / w, ph / h
            if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
                continue
            label = str(p.get("class", p.get("class_name", "")) or "")
            out.append((px, py, max(0.0, pw), max(0.0, ph), conf, label))
        out.sort(key=lambda t: -t[4])
        return out

    # ── per-shape solving ───────────────────────────────────────────────

    async def _detect(self, image: bytes, question: str, timeout: float,
                      classes: Optional[List[str]] = None):
        """Run the workflow and return (points, texts, raw)."""
        img = shrink_image(image)
        size = image_size(img) or (0, 0)
        data = await self._run(img, question, timeout, classes)
        if data is None:
            return None, None, None
        preds, texts = self.read_response(data)
        return self.predictions_to_points(preds, size), texts, data

    async def _solve_tiles(self, prompt: str, images: List[bytes],
                           examples: List[bytes],
                           timeout: float) -> Optional[dict]:
        """Grid rounds: one detection call per tile, in reading order.

        A tile counts as a match when the workflow returns at least one
        prediction on it (or answers the question with "yes").
        """
        question = tile_yes_question(prompt, has_ref=bool(examples))
        classes = detection_classes(prompt)
        per = min(_TILE_TIMEOUT, max(8.0, timeout))
        hits: List[int] = []
        answered = 0
        for i, raw in enumerate(images, 1):
            points, texts, data = await self._detect(raw, question, per,
                                                     classes)
            if data is None:
                self._log(f"[Vision] tile {i}/{len(images)} -> error",
                          level="debug")
                continue
            answered += 1
            hit = bool(points)
            if not hit and texts:
                yn = parse_yesno(" ".join(texts))
                hit = yn is True
            self._log(f"[Vision] tile {i}/{len(images)} -> "
                      f"{len(points or [])} detection(s) hit={hit}",
                      level="debug")
            if hit:
                hits.append(i)
        if answered == 0:
            return None
        self._log(f"[Vision] grid detections: {hits} "
                  f"({answered}/{len(images)} answered)")
        return {"type": "tiles", "indices": hits}

    # ── RT-DETR backup detector ─────────────────────────────────────────

    @property
    def rtdetr_endpoint(self) -> str:
        """Hosted alias URL.

        COCO aliases are served at ``POST {base}/{model_id}?api_key=...``
        with a base64 body — NOT at /infer/object_detection, which expects
        a registered workspace project and 404s for aliases.
        """
        return f"{self.base}/{self.rtdetr_model_id}"

    async def _rtdetr_infer(self, image: bytes, timeout: float):
        """One rfdetr-small call. Returns normalised detections or None."""
        img = shrink_image(image)
        size = image_size(img) or (0, 0)
        data = await self._rtdetr_post(img, timeout)
        if data is None:
            return None
        preds, _ = self.read_response(data)
        return self.predictions_to_points(preds, size,
                                          min_conf=RTDETR_MIN_CONF)

    @staticmethod
    def filter_by_labels(points, labels) -> list:
        """Keep detections whose class is one of ``labels`` (case-loose)."""
        want = {str(l).lower().strip() for l in (labels or ())}
        if not want:
            return []
        return [p for p in (points or []) if str(p[5]).lower().strip() in want]

    async def solve_rtdetr(self, prompt: str, images: List[bytes],
                           shape: str = "tiles",
                           timeout: float = RTDETR_TIMEOUT) -> Optional[dict]:
        """Backup solve with rfdetr-small + the knowledge base.

        COCO-80 cannot read a prompt, so the prompt is resolved to COCO
        labels through the alias table first. No mapping -> abstain
        (returning None) rather than answer with the wrong object.
        """
        if not self.rtdetr_enabled or not self._api_key or not images:
            return None
        labels = coco_targets(prompt)
        if not labels:
            self._log(f"[RT-DETR] no COCO class for prompt {prompt[:60]!r} "
                      "— abstaining", level="debug")
            return None
        if shape not in ("tiles", "points", "bbox", "count"):
            self._log(f"[RT-DETR] shape={shape} needs reasoning a detector "
                      "cannot do — abstaining", level="debug")
            return None
        self._log(f"[RT-DETR] backup solve with {self.rtdetr_model_id} "
                  f"looking for {list(labels)[:6]}")
        self.stats["rtdetr"] += 1

        if shape == "tiles" and len(images) >= 2:
            hits, answered = [], 0
            for i, raw in enumerate(images, 1):
                pts = await self._rtdetr_infer(raw, timeout)
                if pts is None:
                    continue
                answered += 1
                if self.filter_by_labels(pts, labels):
                    hits.append(i)
            if answered == 0:
                return None
            self._log(f"[RT-DETR] grid hits: {hits} "
                      f"({answered}/{len(images)} answered)")
            return {"type": "tiles", "indices": hits}

        pts = await self._rtdetr_infer(images[0], timeout)
        if pts is None:
            return None
        keep = self.filter_by_labels(pts, labels)
        if not keep:
            return None
        return self.detections_to_answer(keep, shape, len(images))

    async def solve(self, prompt: str, images: List[bytes],
                    shape: str = "tiles", examples: Optional[List[bytes]] = None,
                    timeout: float = ROBOFLOW_TIMEOUT) -> Optional[dict]:
        """Ask the Roboflow/Gemini workflow to answer an hCaptcha round.

        ``images`` are the answer surfaces as PNG/JPEG bytes. For
        shape="tiles" they are the grid tiles in reading order (tile 1 =
        top-left); for "points"/"bbox"/"drag" it is the ONE big challenge
        canvas; for "choice" the reference image(s). ``examples`` are the
        prompt-header reference images (the "item shown").

        Returns, by shape:

          tiles     {"type": "tiles",  "indices": [1, 3, 7]}
          points    {"type": "points", "points": [(x, y)]}         (0-1)
          bbox      {"type": "bbox",   "bbox": {x1,y1,x2,y2}}      (0-1)
          drag      {"type": "drag",   "from": (x,y), "to": (x,y)} (0-1)
          choice    {"type": "choice", "index": 2}                 (1-based)
          count     {"type": "count",  "count": 3}
          pattern   {"type": "drag",   "from": (x,y), "to": (x,y)} (0-1)
          stack     {"type": "drags",  "drags": [(sx, sy, tx, ty), ...]}
          text      {"type": "text",   "text": "abc123"}
          None      workflow unreachable or answer unusable
        """
        if not self.configured:
            self._log("[Vision] Roboflow not configured — set API_KEY "
                      "(and ROBOFLOW_WORKSPACE / ROBOFLOW_WORKFLOW)",
                      level="error")
            return None
        if not images:
            return None
        import time
        self._log(f"{time.strftime('%b %d %H:%M:%S.000 [info]')} "
                  f"solving with roboflow {self.model} "
                  f"({self.workspace}/{self.workflow}, shape={shape})")
        self.stats["calls"] += 1
        examples = list(examples or [])

        if shape == "tiles" and len(images) >= 2:
            got = await self._solve_tiles(prompt, images, examples, timeout)
            if got is not None:
                self.stats["ok"] += 1
                return got
            self.stats["failed"] += 1
            return await self._fallback(prompt, images, shape)

        question = self.shape_question(prompt, shape)
        points, texts, data = await self._detect(images[0], question, timeout)
        if data is None:
            self.stats["failed"] += 1
            return await self._fallback(prompt, images, shape)

        parsed = self.detections_to_answer(points, shape, len(images))
        if parsed is None and texts:
            blob = "\n".join(texts)
            parse_shape = "drag" if shape in ("pattern", "tower") else shape
            parsed = self._parse_geometry(blob, parse_shape, len(images))
            if parsed is None and shape in ("tiles", "text", "count", "stack"):
                parsed = self._parse_answer(blob, len(images), shape)
        if parsed is None:
            self._log(f"[Vision] No usable answer for shape={shape} "
                      f"({len(points or [])} detections, "
                      f"{len(texts or [])} text field(s))", level="warn")
            self.stats["failed"] += 1
            return await self._fallback(prompt, images, shape)
        self.stats["ok"] += 1
        return parsed

    async def _fallback(self, prompt: str, images: List[bytes],
                        shape: str) -> Optional[dict]:
        """Gemini failed. Try tier 2 (RT-DETR), then tier 3 (Gemma)."""
        if self.rtdetr_enabled:
            got = await self.solve_rtdetr(prompt, images, shape)
            if got is not None:
                self._log(f"[Vision] RT-DETR backup answered: {got}")
                return got
        if self.gemma_enabled:
            got = await self.solve_gemma(prompt, images, shape)
            if got is not None:
                self._log(f"[Vision] Gemma tier-3 answered: {got}")
                return got
        return None

    # ── Tier 3: Gemma VLM fallback ──────────────────────────────────────

    async def check_gemma(self) -> bool:
        """Is the Gemma host up and does it have the model pulled?"""
        if not self.gemma_enabled:
            return False
        try:
            cfg = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=cfg) as s:
                async with s.get(f"{self.gemma_base}/api/tags") as r:
                    if r.status != 200:
                        self._log(f"[Gemma] host HTTP {r.status}",
                                  level="warn")
                        return False
                    data = await r.json()
            names = [m.get("name", "") for m in (data or {}).get("models", [])]
            base = self.gemma_model.split(":")[0]
            if names and not any(n.split(":")[0] == base for n in names):
                self._log(f"[Gemma] {self.gemma_model} not pulled on "
                          f"{self.gemma_base} (have: {names[:4]}) — run "
                          f"`ollama pull {self.gemma_model}`", level="warn")
                return False
            return True
        except Exception as e:
            hint = ""
            if ".railway.internal" in self.gemma_base:
                # Railway's private network is IPv6-only; Ollama defaults
                # to 127.0.0.1 and never appears on it.
                hint = (" — set OLLAMA_HOST=[::]:11434 on the ollama "
                        "service (Railway private networking is IPv6-only)")
            self._log(f"[Gemma] unreachable at {self.gemma_base}: "
                      f"{type(e).__name__}{hint}", level="warn")
            return False

    async def _gemma_chat(self, system: str, question: str,
                          images: List[bytes], timeout: float,
                          want_json: bool = True) -> Optional[str]:
        """One Ollama /api/chat turn. Returns the message text or None."""
        payload = {
            "model": self.gemma_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question,
                 "images": [_b64(b) for b in images if b]},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9,
                        "num_predict": 256},
            "keep_alive": "10m",
        }
        if want_json:
            payload["format"] = "json"
        try:
            cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=cfg) as s:
                async with s.post(f"{self.gemma_base}/api/chat",
                                  json=payload) as r:
                    if r.status != 200:
                        body = (await r.text())[:200]
                        self._log(f"[Gemma] HTTP {r.status}: {body}",
                                  level="warn")
                        return None
                    data = await r.json()
            return ((data or {}).get("message") or {}).get("content") or ""
        except Exception as e:
            self._log(f"[Gemma] error: {type(e).__name__}", level="warn")
            return None

    async def solve_gemma(self, prompt: str, images: List[bytes],
                          shape: str = "tiles",
                          examples: Optional[List[bytes]] = None,
                          timeout: float = GEMMA_TIMEOUT) -> Optional[dict]:
        """Last-resort solve with the Gemma VLM.

        Gemma reads the prompt, so unlike rfdetr-small it can answer the
        rounds that have no COCO class at all. Grid rounds ask one tile at
        a time (yes/no); every other shape uses the JSON contracts the
        parsers already understand.
        """
        if not self.gemma_enabled or not images:
            return None
        self._log(f"[Gemma] tier-3 solve with {self.gemma_model} "
                  f"@ {self.gemma_base} (shape={shape})")
        self.stats["gemma"] += 1

        if shape == "tiles" and len(images) >= 2:
            ref = shrink_image(examples[0]) if examples else None
            q = tile_yes_question(prompt, has_ref=bool(ref))
            per = min(GEMMA_TILE_TIMEOUT, max(8.0, timeout))
            hits, answered = [], 0
            for i, raw in enumerate(images, 1):
                tile = shrink_image(raw)
                bundle = [b for b in ((ref, tile) if ref else (tile,)) if b]
                text = await self._gemma_chat(
                    "Look at the image and answer the question with yes "
                    "or no only.", q, bundle, per, want_json=False)
                if text is None:
                    continue
                answered += 1
                if parse_yesno(text) is True:
                    hits.append(i)
            if answered == 0:
                return None
            self._log(f"[Gemma] grid hits: {hits} "
                      f"({answered}/{len(images)} answered)")
            return {"type": "tiles", "indices": hits}

        system = _SYSTEM_BY_SHAPE.get(shape, _SYSTEM_BY_SHAPE["tiles"])
        question = self.shape_question(prompt, shape)
        bundle = [shrink_image(b) for b in (list(examples or []) + images)]
        # Geometry shapes are where a 4B model is weakest AND slowest — on
        # drag rounds it has timed out every time, stalling the challenge
        # for the full budget. Cap them hard so the round moves on.
        if shape in ("drag", "pattern", "tower", "bbox", "points"):
            timeout = min(timeout, GEMMA_GEOMETRY_TIMEOUT)
        text = await self._gemma_chat(system, question, bundle, timeout)
        if not text:
            return None
        parse_shape = shape if shape != "text" else "tiles"
        parsed = self._parse_geometry(text, parse_shape, len(images))
        if parsed is None:
            parsed = self._parse_answer(text, len(images), shape)
        return parsed

    @staticmethod
    def shape_question(prompt: str, shape: str) -> str:
        """The question sent with the image, tuned per answer shape.

        Delegates to :func:`rewrite_question`, which restates hCaptcha's
        human "click on ..." wording as detector wording ("box and
        coordinate ...") and appends the per-shape output contract.
        """
        return rewrite_question(prompt, shape)

    @staticmethod
    def _legacy_shape_question(prompt: str, shape: str) -> str:
        """Previous plain-passthrough phrasing, kept for reference."""
        p = (prompt or "").strip()
        if shape == "count":
            return (f"{p} Count every matching object and answer with the "
                    'JSON object {"count": N}.')
        if shape == "bbox":
            return (f"{p} Return one tight bounding box around the target.")
        if shape in ("drag", "pattern"):
            return (f"{p} Detect the loose draggable piece and the empty "
                    "slot it belongs in.")
        if shape == "tower":
            return (f"{p} Detect the loose block segment with the Move badge "
                    "and the incomplete tower it must be dropped on.")
        if shape == "stack":
            return (f"{p} Detect every loose block and every stack top, and "
                    'answer with {"drags": [[sx, sy, tx, ty], ...]} in '
                    "0-100 percent coordinates.")
        if shape == "choice":
            return (f"{p} Answer with the JSON object "
                    '{"choice": N} using the 1-based option number.')
        if shape == "text":
            return f"{p} Read the characters and answer with the text only."
        return p

    @staticmethod
    def detections_to_answer(points, shape: str, image_count: int = 1):
        """Map normalised detections onto the answer shape."""
        pts = list(points or [])
        if not pts:
            return None
        if shape == "points":
            return {"type": "points", "points": [(p[0], p[1]) for p in pts[:4]]}
        if shape == "bbox":
            x, y, w, h = pts[0][0], pts[0][1], pts[0][2], pts[0][3]
            if w <= 0 or h <= 0:
                return None
            return {"type": "bbox", "bbox": {
                "x1": max(0.0, x - w / 2), "y1": max(0.0, y - h / 2),
                "x2": min(1.0, x + w / 2), "y2": min(1.0, y + h / 2)}}
        if shape in ("drag", "pattern", "tower"):
            # Confidence order says nothing about WHICH box is the loose
            # piece and which is the hole — score the roles geometrically.
            pair = pair_drag(pts)
            if pair is None:
                return None
            src, dst = pair
            return {"type": "drag", "from": src, "to": dst}
        if shape == "count":
            return {"type": "count", "count": len(pts)}
        if shape == "tiles":
            return {"type": "tiles", "indices": [1]} if image_count == 1 else None
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
                    a, b = RoboflowVisionClient._num(p[kx]), \
                        RoboflowVisionClient._num(p[ky])
                    if a is not None and b is not None:
                        return (a, b)
            return None
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            a, b = RoboflowVisionClient._num(p[0]), RoboflowVisionClient._num(p[1])
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
        obj = RoboflowVisionClient._loads_repaired(content)
        if obj is None:
            return None
        num = RoboflowVisionClient._num
        pt = RoboflowVisionClient._point

        # Stacking plans (FunCAPTCHA/Arkose "make every column equal"): the
        # generic branches below would squash a multi-drag {"drags": [...]}
        # into one drag / a points pair, so parse the FULL plan first.
        if shape == "stack":
            return RoboflowVisionClient._parse_stack_geometry(obj)

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
                    f = RoboflowVisionClient._point(
                        v.get("from") or v.get("start"))
                    t = RoboflowVisionClient._point(
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
                f = RoboflowVisionClient._point(
                    item.get("from") or item.get("start")
                    or item.get("grab") or item.get("pick"))
                t = RoboflowVisionClient._point(
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
            loose = RoboflowVisionClient._parse_loose_tiles(text, tile_count)
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
    """Quick smoke test against the Roboflow workflow."""
    client = RoboflowVisionClient(log=lambda m, level="info": print(m, flush=True))
    ok, models = await client.check()
    print(f"workflow ok={ok} model={models}")
    if not ok:
        print("Set API_KEY (Roboflow) and, if needed, ROBOFLOW_WORKSPACE / "
              "ROBOFLOW_WORKFLOW.")
        return

    def _solid(rgb: tuple) -> bytes:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (96, 96), rgb).save(buf, format="JPEG")
        return buf.getvalue()

    ans = await client.solve(
        "Select all images that are red.",
        [_solid((30, 90, 200)), _solid((200, 30, 30))])
    print(f"answer: {ans}")


if __name__ == "__main__":
    asyncio.run(_self_test())

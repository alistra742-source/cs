#!/usr/bin/env python3
"""
hcaptcha_types.py — challenge-family router + semantic knowledge base.

hCaptcha serves five challenge families, not one:

  ============================  =======================================
  image_label_binary            "click each image containing a bus"
  binary + reference image      "pick all things you can work on with
                                 the item shown" (tool -> materials)
  area_select (point)           "click on the animal who jumps highest"
  area_select (bbox)            "draw a box around the cat's head"
  image_drag_drop               "drag the element to the place it fits"
  multiple_choice               "select the most accurate description"
  ============================  =======================================

The old solver assumed every round was a tile grid and answered with tile
indices — only family #1 worked. This module decides WHICH family a round
is (three independent tiers, highest-signal first):

  1. classify_from_payload  — request_type inside hCaptcha's /getcaptcha JSON
  2. classify_from_dom      — visible tiles/canvases/draggables/choices
  3. classify_from_prompt   — wording regexes (last resort)

``classify(payload, dom, prompt)`` tries them in that order.

It also carries the small knowledge base that makes relational prompts
solvable *offline*: superlative tables (size/jump/speed/temperature), a
tool-affordance map ("what can you work on with a drill") and category
sets (animals/edible/wheeled/motorised). ``resolve_semantic`` turns a
prompt + per-tile labels into 1-based tile indices without a vision model.
"""

from __future__ import annotations

import re

# ── families + answer shapes ──────────────────────────────────────────────

BINARY = "image_label_binary"
AREA_POINT = "area_select_point"
AREA_BBOX = "area_select_bbox"
DRAG_DROP = "image_drag_drop"
MULTIPLE_CHOICE = "multiple_choice"
TEXT_ENTRY = "text_entry"
COUNT = "counting"
UNKNOWN = "unknown"

ANSWER_SHAPE = {
    BINARY: "tiles",
    AREA_POINT: "points",
    AREA_BBOX: "bbox",
    DRAG_DROP: "drag",
    MULTIPLE_CHOICE: "choice",
    TEXT_ENTRY: "text",
    COUNT: "count",
    UNKNOWN: "tiles",  # safest fallback: the grid clicker also no-ops cleanly
}


def answer_shape(family: str) -> str:
    return ANSWER_SHAPE.get(family, "tiles")


# ── payload helpers (hCaptcha /getcaptcha JSON) ───────────────────────────

def question_text(payload: dict) -> str:
    """requester_question: {"en": "..."} | str | missing."""
    if not isinstance(payload, dict):
        return ""
    rq = payload.get("requester_question")
    if isinstance(rq, dict):
        for key in ("en", "en-US", "en_US"):
            if rq.get(key):
                return str(rq[key])
        for val in rq.values():
            if isinstance(val, str) and val.strip():
                return val
        return ""
    if isinstance(rq, str):
        return rq
    return ""


def example_urls(payload: dict) -> list:
    """requester_question_example: header reference image URL(s), 0..2."""
    if not isinstance(payload, dict):
        return []
    ex = payload.get("requester_question_example")
    out = []
    if isinstance(ex, str):
        out = [ex]
    elif isinstance(ex, (list, tuple)):
        out = [u for u in ex if isinstance(u, str)]
    elif isinstance(ex, dict):
        for val in ex.values():
            if isinstance(val, str):
                out.append(val)
            elif isinstance(val, (list, tuple)):
                out.extend(u for u in val if isinstance(u, str))
    return [u for u in out if u.startswith(("http", "/", "data:"))]


def task_urls(payload: dict) -> list:
    """tasklist[].datapoint_uri — the tile image URLs in reading order."""
    if not isinstance(payload, dict):
        return []
    tasks = payload.get("tasklist") or payload.get("tasks") or []
    out = []
    for t in tasks if isinstance(tasks, (list, tuple)) else []:
        if isinstance(t, dict) and isinstance(t.get("datapoint_uri"), str):
            out.append(t["datapoint_uri"])
    return out


# ── tier 1: classify from the /getcaptcha payload ────────────────────────

_BBOX_WORD_RE = re.compile(
    r"draw (a|the) (bounding )?(box|rectangle)|bounding box|"
    r"box around|drag a box|draw around", re.I)

# Binary-stage wording inside an image_label_area_select payload. hCaptcha
# serves MIXED rounds under that request_type: a tile-grid binary stage
# ("click each image containing a bus") followed by an area stage ("then
# click on the car"). The payload only carries the stage-1 question, so the
# payload tier must NOT commit to area_select for these — it defers to the
# live DOM/prompt tiers, which classify each stage as it appears.
_BINARY_GRID_Q_RE = re.compile(
    r"click (on )?(each|every|all (the )?) (image|photo|picture|tile)s?|"
    r"select all (the )?(images|photos|pictures|tiles)|"
    r"choose all (the )?(images|photos|pictures|tiles)|"
    r"pick (all|every) (the )?(images?|photos?|pictures?)", re.I)


# Counting tasks ("How many X are in this image?") — a photo + numeric
# answer options. Routed from the payload, from counting wording with
# option buttons in the DOM, or from the prompt alone.
_COUNT_WORD_RE = re.compile(
    r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal (number|amount)\b|"
    r"\bselect (the )?(correct )?number\b", re.I)

# Pattern-completion drag rounds ("Put one of the animals into the empty
# spot to complete the pattern"): a 3x3 icon grid with ONE empty cell and
# a row of candidate elements; the right candidate completes every row and
# column. These are image_drag_drop under the hood, but the piece is not a
# punched silhouette — it is chosen by the PATTERN, so the router flags
# them and the round solver applies semantic logic instead of the pure
# geometric DragLocator.
_PATTERN_PHRASE_RE = re.compile(
    r"complete the pattern|finish the pattern|"
    r"empty (spot|space|cell)|blank (spot|space|cell)|"
    r"missing (spot|space|cell)|"
    r"fill (the |in )?(the )?(empty|missing|blank) (spot|space|cell)|"
    r"put (one of )?the .{0,40}? into the (empty|blank|missing) "
    r"(spot|space|cell)|"
    r"to (complete|finish) the (pattern|row|sequence|grid)|"
    r"which (one )?(belongs|goes|fits) in the (empty|blank|missing) "
    r"(spot|space|cell)", re.I)


def is_pattern_prompt(prompt: str) -> bool:
    """True when the prompt is a pattern-completion drag round."""
    return bool(_PATTERN_PHRASE_RE.search(prompt or ""))


def _bbox_config(payload: dict) -> bool:
    cfg = payload.get("request_config")
    if not isinstance(cfg, dict):
        return False
    blob = " ".join(str(v) for v in cfg.values()).lower()
    return "bbox" in blob or "bounding" in blob or "rectangle" in blob or \
        "\"box\"" in blob


def classify_from_payload(payload: dict) -> str:
    if not isinstance(payload, dict):
        return UNKNOWN
    rt = str(payload.get("request_type") or payload.get("type") or "").lower()
    if not rt:
        return UNKNOWN
    q = question_text(payload)
    if "drag" in rt or "drop" in rt:
        return DRAG_DROP
    if "multiple" in rt or "choice" in rt:
        return MULTIPLE_CHOICE
    if "count" in rt or "number" in rt:
        return COUNT
    if "area_select" in rt or ("area" in rt and "select" in rt):
        # MIXED rounds (binary grid stage, then an area stage) share this
        # request_type; the payload question describes stage 1. When it is
        # binary-grid wording, defer to the live DOM/prompt tiers so each
        # stage classifies correctly as it renders.
        if _BINARY_GRID_Q_RE.search(q):
            return UNKNOWN
        if _BBOX_WORD_RE.search(q) or _bbox_config(payload):
            return AREA_BBOX
        return AREA_POINT
    if "binary" in rt:
        return BINARY
    if "text" in rt or "entry" in rt or "ocr" in rt:
        return TEXT_ENTRY
    return UNKNOWN


# ── tier 2: classify from the live DOM (DOM_PROBE_JS facts) ──────────────

# JS evaluated inside the challenge frame by server._probe_challenge_dom.
# Counts what's actually visible; the rules below map counts to a family.
DOM_PROBE_JS = r"""() => {
    const vis = (el) => !!(el) &&
        (el.offsetParent !== null || el.getClientRects().length > 0);
    const q = (sel) => Array.from(document.querySelectorAll(sel));
    const count = (sel) => q(sel).filter(vis).length;
    const tiles = count('div.task-image, [class*="task-image" i]');
    const examples = count(
        '.challenge-example, .prompt-image, .challenge-prompt img, ' +
        '[class*="example" i] img, [class*="example" i] div[style*="background"]');
    const canvases = q('canvas').filter(c => vis(c) && c.width > 50).length;
    const bigImages = q('img, [class*="image" i]')
        .filter(el => vis(el) &&
            (el.clientWidth >= 180 || el.naturalWidth >= 180)).length;
    const draggables = count('[draggable="true"], [class*="drag" i]');
    let moveBadge = false;
    for (const el of q('div, span, p')) {
        const t = (el.textContent || '').trim();
        if (vis(el) && el.children.length === 0 && /^move$/i.test(t)) {
            moveBadge = true; break;
        }
    }
    const choices = count(
        '.answer-option, [class*="answer-option" i], [class*="choice" i] button, ' +
        '[class*="multiple" i] [role="button"], .options [role="button"]');
    const inputs = count('input[type="text"], input:not([type]), textarea');
    return {
        tiles, examples, canvases, images: bigImages,
        draggables, move_badge: moveBadge, choices, inputs,
    };
}"""


def classify_from_dom(facts: dict, prompt: str = "") -> str:
    """Map DOM fact counts to a family. Rules ordered by confidence."""
    if not isinstance(facts, dict) or not facts:
        return UNKNOWN

    def n(key):
        try:
            return max(0, int(facts.get(key, 0) or 0))
        except Exception:
            return 0

    tiles = n("tiles")
    # A draggable element / "Move" badge with at most one tile is the
    # drag-drop challenge (the piece is the draggable).
    if (n("draggables") > 0 or facts.get("move_badge")) and tiles <= 1:
        return DRAG_DROP
    # Pattern-completion rounds: a 3x3 grid of tiles PLUS a row of
    # draggable candidates — the DOM shows many tiles, which would
    # otherwise fall through to binary. The wording disambiguates.
    if _PATTERN_PHRASE_RE.search(prompt or "") and (
            n("draggables") > 0 or facts.get("move_badge") or tiles >= 6):
        return DRAG_DROP
    # A genuine tile grid is the binary family (hCaptcha grids are 9+ tiles;
    # 4 tolerated for odd layouts).
    if tiles >= 4:
        return BINARY
    # Several option buttons with no grid is multiple choice — unless the
    # wording asks "how many", which is a counting task (numeric options).
    if n("choices") >= 2 and tiles <= 1:
        if _COUNT_WORD_RE.search(prompt or ""):
            return COUNT
        return MULTIPLE_CHOICE
    # A bare text field is a text-entry (OCR) challenge.
    if n("inputs") >= 1 and tiles == 0 and n("canvases") == 0 \
            and n("images") == 0:
        return TEXT_ENTRY
    # One big surface to click on: area_select. Wording splits point vs bbox.
    if tiles == 1 or n("canvases") >= 1 or n("images") >= 1:
        if _BBOX_WORD_RE.search(prompt or ""):
            return AREA_BBOX
        return AREA_POINT
    return UNKNOWN


# ── tier 3: classify from the prompt wording ─────────────────────────────

_PROMPT_RULES = (
    (DRAG_DROP, re.compile(
        r"\bdrag\b|where it fits|place where it belongs|puzzle piece|"
        r"complete the puzzle|missing piece|empty space|matching slot|"
        r"matching outline|move the (piece|element|shape|tile)|"
        r"complete the pattern|empty (spot|cell)|into the empty "
        r"(spot|space|cell)|fill the (empty|missing|blank) "
        r"(spot|space|cell)", re.I)),
    (BINARY, re.compile(
        r"click each image|click (on )?every (image|photo|picture|tile)|"
        r"select all (the )?(images|pictures|photos|tiles)|"
        r"choose all (the )?(images|pictures|photos|tiles)|"
        r"pick (all|every) (the )?(images?|pictures?|photos?|tiles?)|"
        r"check all (the )?(images|pictures|photos|tiles)|"
        r"mark (all|every) (the )?(images|pictures|photos|tiles)|"
        r"click (all )?(the )?images (containing|with)|"
        r"images? (containing|with|of) a|all (the )?squares with", re.I)),
    (COUNT, _COUNT_WORD_RE),
    (AREA_BBOX, _BBOX_WORD_RE),
    (MULTIPLE_CHOICE, re.compile(
        r"most accurate|best (describes|description)|which (of these|one)|"
        r"(choose|select) the (correct|best|right) (answer|option|description)", re.I)),
    (TEXT_ENTRY, re.compile(
        r"\btype\b|enter the (characters|text|letters)|"
        r"characters (you see|in the image)", re.I)),
    (AREA_POINT, re.compile(
        r"\bclick on\b|\btap on\b|click the (animal|object|item|largest|"
        r"smallest|fastest|slowest|highest)|click anywhere on|"
        r"place (a|the) (point|dot)", re.I)),
)


def classify_from_prompt(prompt: str) -> str:
    p = (prompt or "").strip()
    if not p:
        return UNKNOWN
    for family, rx in _PROMPT_RULES:
        if rx.search(p):
            return family
    return UNKNOWN


def classify(payload: dict = None, dom: dict = None, prompt: str = "") -> str:
    """All three tiers in order; first non-UNKNOWN wins."""
    for result in (classify_from_payload(payload),
                   classify_from_dom(dom, prompt),
                   classify_from_prompt(prompt)):
        if result != UNKNOWN:
            return result
    return UNKNOWN


# ── knowledge base ────────────────────────────────────────────────────────
#
# Ranks are arbitrary-but-consistent scales. The synthetic challenge
# generator (make_challenges.py) labels relational rounds from THE SAME
# tables, so the offline models and the router stay in lockstep by
# construction.

SIZE_RANK = {
    "nail": 1, "screw": 2, "bolt": 3, "butterfly": 3, "snail": 4, "flower": 5,
    "fish": 5, "apple": 6, "banana": 7, "cup": 7, "frog": 8, "bird": 9,
    "duck": 10, "rabbit": 10, "book": 11, "clock": 12, "turtle": 13,
    "cactus": 14, "boot": 14, "cat": 15, "umbrella": 16, "pizza": 17,
    "dog": 18, "guitar": 19, "chair": 19, "kangaroo": 20, "bicycle": 21,
    "table": 22, "sheep": 23, "motorcycle": 23, "cow": 24, "zebra": 25,
    "car": 25, "lion": 26, "bear": 27, "horse": 26, "boat": 27, "truck": 28,
    "bus": 29, "giraffe": 30, "train": 30, "airplane": 31, "elephant": 32,
    "tree": 33, "house": 34, "mountain": 35,
    # tools / street furniture / surfaces (relational "largest/smallest"
    # rounds over tools or urban objects)
    "screwdriver": 5, "paintbrush": 6, "wrench": 7, "hammer": 8,
    "drill": 9, "saw": 10, "wood": 11, "fire_hydrant": 13,
    "parking_meter": 14, "traffic_light": 15, "red_light": 15,
    "canvas": 17, "crosswalk": 21, "wall": 31,
}

JUMP_RANK = {
    "snail": 1, "fish": 1, "turtle": 2, "elephant": 3, "cow": 4, "sheep": 5,
    "duck": 5, "horse": 6, "bear": 6, "dog": 7, "giraffe": 7, "butterfly": 8,
    "cat": 8, "bird": 8, "rabbit": 9, "zebra": 9, "frog": 10, "lion": 10,
    "kangaroo": 11,
}

SPEED_RANK = {
    "snail": 1, "turtle": 2, "frog": 3, "butterfly": 4, "elephant": 5,
    "cow": 5, "fish": 6, "rabbit": 6, "dog": 7, "kangaroo": 8, "duck": 8,
    "bird": 9, "cat": 9, "horse": 10, "bicycle": 11, "sheep": 11, "motorcycle": 12,
    "bear": 12, "car": 13, "giraffe": 13, "bus": 14, "truck": 15, "zebra": 15,
    "boat": 16, "lion": 16, "train": 17, "airplane": 18,
}

TEMP_RANK = {
    "mountain": 1, "fish": 2, "flower": 3, "apple": 4, "tree": 4, "banana": 5,
    "boot": 5, "house": 5, "umbrella": 5, "cactus": 9, "cup": 8, "pizza": 9,
    # animals by habitat warmth ("coldest/hottest place" rounds)
    "snail": 2, "turtle": 3, "frog": 4, "bear": 5, "sheep": 6, "cow": 6,
    "horse": 6, "rabbit": 6, "duck": 6, "dog": 7, "bird": 7,
    "butterfly": 7, "cat": 8, "kangaroo": 9, "zebra": 9, "giraffe": 9,
    "elephant": 9, "lion": 10,
}

# What you can "work on" with each tool (reference-image affordance grids).
TOOL_AFFORDANCE = {
    "drill": {"wood", "wall", "table", "chair", "house"},
    "hammer": {"nail", "wood", "wall"},
    "saw": {"wood", "tree"},
    "wrench": {"bolt", "bicycle", "car", "truck"},
    "paintbrush": {"wall", "canvas", "house"},
    "screwdriver": {"screw", "wood"},
}

ANIMALS = {"dog", "cat", "rabbit", "horse", "elephant", "cow", "bird",
           "frog", "turtle", "snail", "kangaroo", "zebra", "giraffe", "lion",
           "bear", "sheep", "duck", "fish", "butterfly"}
EDIBLE = {"apple", "pizza", "banana", "fish"}
WHEELED = {"car", "bus", "truck", "bicycle", "motorcycle"}
MOTORISED = {"car", "bus", "truck", "motorcycle", "boat", "airplane", "train"}
TOOLS = set(TOOL_AFFORDANCE)
MATERIALS = {"wood", "nail", "screw", "bolt", "wall", "canvas"}

# Synonyms -> canonical class name. Keys are lowercase, spaces as "_".
# NB: "red_light" and "traffic_light" are deliberately kept exclusive —
# red_light must never canonicalise to traffic_light (opposite labels).
SYNONYMS = {
    "bus": "bus", "car": "car", "auto": "car", "automobile": "car",
    "truck": "truck", "lorry": "truck", "train": "train",
    "bicycle": "bicycle", "bike": "bicycle", "motorcycle": "motorcycle",
    "motorbike": "motorcycle", "scooter": "motorcycle",
    "boat": "boat", "ship": "boat", "airplane": "airplane",
    "aeroplane": "airplane", "plane": "airplane", "jet": "airplane",
    "traffic_light": "traffic_light", "stoplight": "traffic_light",
    "red_light": "red_light", "crosswalk": "crosswalk",
    "zebra_crossing": "crosswalk",
    "fire_hydrant": "fire_hydrant", "hydrant": "fire_hydrant",
    "parking_meter": "parking_meter",
    "dog": "dog", "puppy": "dog", "doggy": "dog",
    "cat": "cat", "kitten": "cat", "kitty": "cat",
    "rabbit": "rabbit", "bunny": "rabbit", "hare": "rabbit",
    "horse": "horse", "pony": "horse", "elephant": "elephant",
    "cow": "cow", "bird": "bird", "frog": "frog", "toad": "frog",
    "turtle": "turtle", "tortoise": "turtle", "snail": "snail",
    "kangaroo": "kangaroo", "roo": "kangaroo",
    "hammer": "hammer", "drill": "drill", "saw": "saw",
    "paintbrush": "paintbrush", "brush": "paintbrush",
    "wrench": "wrench", "spanner": "wrench", "screwdriver": "screwdriver",
    "wood": "wood", "plank": "wood", "lumber": "wood",
    "nail": "nail", "screw": "screw", "bolt": "bolt",
    "wall": "wall", "wallpaper": "wall", "canvas": "canvas",
    "apple": "apple", "pizza": "pizza", "table": "table",
    "chair": "chair", "cup": "cup", "mug": "cup", "book": "book",
    "clock": "clock", "umbrella": "umbrella", "tree": "tree",
    "flower": "flower", "house": "house", "home": "house",
    "mountain": "mountain", "hill": "mountain", "boot": "boot",
    "shoe": "boot",
    # batch 3 (49 -> 60)
    "zebra": "zebra",
    "giraffe": "giraffe",
    "lion": "lion",
    "bear": "bear", "teddy": "bear",
    "sheep": "sheep", "lamb": "sheep", "ram": "sheep", "ewe": "sheep",
    "duck": "duck", "duckling": "duck", "mallard": "duck",
    "fish": "fish", "goldfish": "fish",
    "butterfly": "butterfly", "monarch": "butterfly",
    "banana": "banana", "plantain": "banana",
    "guitar": "guitar", "acoustic_guitar": "guitar",
    "cactus": "cactus", "cacti": "cactus", "saguaro": "cactus",
    # ── wide coverage aliases ────────────────────────────────────────────
    # These map the long tail of hCaptcha prompt nouns onto the 60 classes
    # the offline models can actually emit, ONLY where the visuals are
    # defensible at tile resolution. Everything else deliberately stays
    # unmapped so the server falls back to the vision model (which reads
    # arbitrary prompt text) instead of trusting a wrong offline answer.
    # aircraft / watercraft
    "seaplane": "airplane", "hydroplane": "airplane",
    "helicopter": "airplane", "copter": "airplane", "aircraft": "airplane",
    "airliner": "airplane",
    "subway": "train", "metro": "train", "tram": "train",
    "streetcar": "train", "locomotive": "train", "monorail": "train",
    "railway": "train", "railcar": "train",
    "sailboat": "boat", "yacht": "boat", "ferry": "boat",
    "canoe": "boat", "kayak": "boat", "rowboat": "boat",
    "speedboat": "boat", "barge": "boat", "catamaran": "boat",
    
    "pickup": "truck", "pickup_truck": "truck", "semi": "truck",
    "semi_truck": "truck", "dump_truck": "truck",
    "garbage_truck": "truck", "fire_truck": "truck",
    "tow_truck": "truck",  "van": "truck",
    "ambulance": "truck",
    "taxi": "car", "police_car": "car", "race_car": "car",
    "racecar": "car", "sedan": "car", "convertible": "car",
    "minivan": "car", 
    "school_bus": "bus", "shuttle": "bus",
    "moped": "motorcycle", 
    # birds (the bird class is a generic bird — fine at tile scale)
    "owl": "bird", "parrot": "bird", "penguin": "bird",
    "chicken": "bird", "rooster": "bird", "hen": "bird",
    "eagle": "bird", "sparrow": "bird", "seagull": "bird",
    "pigeon": "bird", "robin": "bird", "peacock": "bird",
    "flamingo": "bird", "toucan": "bird", "hummingbird": "bird",
    "woodpecker": "bird", "swan": "duck", "goose": "duck",
    # water animals
    "shark": "fish", "dolphin": "fish", "whale": "fish",
    "clownfish": "fish", "salmon": "fish", "trout": "fish",
    "tuna": "fish", 
    # hooved / farm animals
    "deer": "horse", "stag": "horse", "reindeer": "horse",
    "moose": "horse", "elk": "horse", "donkey": "horse",
    "mule": "horse", "camel": "horse",
    "goat": "sheep", "alpaca": "sheep", "llama": "sheep",
    "yak": "cow", "buffalo": "cow", "bison": "cow", "ox": "cow",
    "calf": "cow",
    "panda": "bear", "koala": "bear", "grizzly": "bear",
    "polar_bear": "bear", "geese": "duck",
    "tadpole": "frog",
    # plants / nature
    "palm_tree": "tree", "pine_tree": "tree", "oak": "tree",
    "maple": "tree", "willow": "tree", "birch": "tree",
    "evergreen": "tree", "forest": "tree", "woodland": "tree",
    "jungle": "tree", "rainforest": "tree", "grove": "tree",
    "woods": "tree",
    "rose": "flower", "sunflower": "flower", "tulip": "flower",
    "daisy": "flower", "dandelion": "flower", "lily": "flower",
    "orchid": "flower", "blossom": "flower",
    "volcano": "mountain", "cliff": "mountain", "peak": "mountain",
    "ridge": "mountain", "hill": "mountain",
    # buildings / urban
    "building": "house", "barn": "house", "cabin": "house",
    "cottage": "house", "hut": "house", "shed": "house",
    "apartment": "house", "bungalow": "house", "villa": "house",
    "traffic_signal": "traffic_light", "traffic_lights": "traffic_light",
    "pedestrian_crossing": "crosswalk", "crossing": "crosswalk",
    "zebra_crossing": "crosswalk", "fireplug": "fire_hydrant",
    # furniture / home
    "bench": "chair", "stool": "chair", "sofa": "chair",
    "couch": "chair", "armchair": "chair", "seat": "chair",
    "desk": "table", "counter": "table",
    "pie": "pizza", "quiche": "pizza",
    "teacup": "cup", "drinking_glass": "cup", "bucket": "cup",
    "magazine": "book", "notebook": "book",
    "watch": "clock", "alarm_clock": "clock", "wall_clock": "clock",
    "grandfather_clock": "clock", "cuckoo_clock": "clock",
    "stopwatch": "clock", "timer": "clock",
    "nut": "bolt",
}


def canonical(word: str):
    """Map a surface word/phrase to the canonical class name (or None)."""
    if not word:
        return None
    key = re.sub(r"[^a-z]+", "_", word.lower()).strip("_")
    if key in SYNONYMS:
        return SYNONYMS[key]
    # naive singular/plural
    if key.endswith("s") and key[:-1] in SYNONYMS:
        return SYNONYMS[key[:-1]]
    return None


_PHRASES = sorted(SYNONYMS.keys(), key=len, reverse=True)


def extract_target(prompt: str):
    """Pull the canonical noun a prompt is about (longest match wins).

    Matches phrases in the prompt text with the common plural endings
    ("pandas" → panda, "boxes" → box, "buses" → bus)."""
    p = " %s " % re.sub(r"[^a-z ]+", " ", (prompt or "").lower())
    for phrase in _PHRASES:
        stem = " %s" % phrase.replace("_", " ")
        for form in ("%s " % stem, "%ss " % stem, "%ses " % stem):
            if form in p:
                return SYNONYMS[phrase]
    return None


def category_of(name: str):
    c = canonical(name) or name
    for label, s in (("animals", ANIMALS), ("edible", EDIBLE),
                     ("wheeled", WHEELED), ("motorised", MOTORISED),
                     ("tools", TOOLS), ("materials", MATERIALS)):
        if c in s:
            return label
    return None


def resolve_pattern(grid_labels, hole_index, candidates):
    """Pattern completion: which candidate completes the 3x3 grid?

    ``grid_labels``: 9 labels in reading order; ``grid_labels[hole_index]``
    is the empty cell (its value is ignored). ``candidates``: list of
    labels for the candidate elements (reading order). Returns the index
    INTO ``candidates`` that makes every row and every column contain
    three distinct labels (a Latin square), or None when no candidate or
    more than one does. Tries the full Latin-square rule first, then the
    rows-only rule (some hCaptcha patterns only constrain rows).

    This is the semantic core of "put one of the animals into the empty
    spot to complete the pattern": it never trusts a guess — ambiguity
    returns None so the caller falls back to the vision model.
    """
    if not isinstance(grid_labels, (list, tuple)) or len(grid_labels) != 9:
        return None
    if not isinstance(hole_index, int) or not (0 <= hole_index < 9):
        return None
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return None
    winners_full, winners_rows = [], []
    for ci, cand in enumerate(candidates):
        if cand is None:
            continue
        labels = list(grid_labels)
        labels[hole_index] = cand
        rows = [labels[r * 3:(r + 1) * 3] for r in range(3)]
        cols = [[labels[r * 3 + c] for r in range(3)] for c in range(3)]
        if None in labels:
            continue
        rows_ok = all(len(set(r)) == 3 for r in rows)
        if rows_ok and all(len(set(c)) == 3 for c in cols):
            winners_full.append(ci)
        elif rows_ok:
            winners_rows.append(ci)
    winners = winners_full or winners_rows
    return winners[0] if len(winners) == 1 else None


_SUPERLATIVE_RULES = (
    (re.compile(r"jumps?\b.*\b(lowest|least)", re.I), "JUMP", "min"),
    (re.compile(r"jumps?\b.*\b(highest|best)|best jumper|highest jump", re.I),
     "JUMP", "max"),
    (re.compile(r"\b(largest|biggest|greatest)\b", re.I), "SIZE", "max"),
    (re.compile(r"\b(smallest|tiniest|littlest)\b", re.I), "SIZE", "min"),
    (re.compile(r"\b(fastest|quickest|speediest)\b", re.I), "SPEED", "max"),
    (re.compile(r"\b(slowest|most slowly)\b", re.I), "SPEED", "min"),
    (re.compile(r"\b(hottest|warmest)\b", re.I), "TEMP", "max"),
    (re.compile(r"\b(coldest|coolest)\b", re.I), "TEMP", "min"),
)

_TABLES = {"JUMP": JUMP_RANK, "SIZE": SIZE_RANK,
           "SPEED": SPEED_RANK, "TEMP": TEMP_RANK}


def superlative_table(prompt: str):
    """"largest"/"jumps the highest"/"slowest" -> (rank_table, 'max'|'min')."""
    p = prompt or ""
    for rx, table, direction in _SUPERLATIVE_RULES:
        if rx.search(p):
            return _TABLES[table], direction
    return None


# ── offline answer resolution ────────────────────────────────────────────

_AFFORDANCE_PHRASE = re.compile(
    r"work on|use (the )?(item|tool|object)|item shown|shown in the "
    r"(image|picture|example)|goes with|used (with|on|together)|"
    r"fit(s)? with|belongs? with|associated with", re.I)

_EXAMPLE_CAT_PHRASE = re.compile(
    r"same (kind|category|type|group)|similar(ly)?|like the (one|item) shown|"
    r"belong(s|ing)? (with|together)|matching the (item|image|example)", re.I)

_SET_PREDICATES = (
    (re.compile(r"\banimals?\b", re.I), ANIMALS),
    (re.compile(r"\b(food|edible|eat|snack)\w*\b", re.I), EDIBLE),
    (re.compile(r"\bwheels?\b|\bwheeled\b", re.I), WHEELED),
    (re.compile(r"\bmotor\w*\b|\bengine\w*\b", re.I), MOTORISED),
    (re.compile(r"\b(vehicles?|transport(ation)?)\b", re.I),
     WHEELED | MOTORISED),
    (re.compile(r"\btools?\b", re.I), TOOLS),
)


def resolve_semantic(prompt: str, tile_labels, example_label=None):
    """Prompt + per-tile canonical labels -> 1-based indices, offline.

    Returns:
      list[int]  understood prompt (possibly legitimately empty — a real
                 zero-match round)
      None       not understood; caller should fall back to the vision model
    """
    p = (prompt or "").strip()
    labels = [canonical(l) or str(l or "").strip().lower()
              for l in (tile_labels or [])]
    example = canonical(example_label) if example_label else None

    if not p and example is None:
        return None

    # 1) superlatives ("click on the animal who jumps the highest")
    sup = superlative_table(p)
    if sup is not None:
        table, direction = sup
        best_i, best_v = None, None
        for i, lab in enumerate(labels, 1):
            v = table.get(lab)
            if v is None:
                continue
            if best_v is None or (v > best_v if direction == "max"
                                  else v < best_v):
                best_i, best_v = i, v
        return [best_i] if best_i is not None else []

    # 2) tool affordance ("things you can work on with the item shown")
    if _AFFORDANCE_PHRASE.search(p):
        tool = extract_target(p)
        if tool not in TOOL_AFFORDANCE and example in TOOL_AFFORDANCE:
            tool = example
        if tool in TOOL_AFFORDANCE:
            allowed = TOOL_AFFORDANCE[tool]
            return [i for i, lab in enumerate(labels, 1) if lab in allowed]
        return None  # an affordance phrasing with no known tool: defer

    # 3) same category as the reference example
    if example and _EXAMPLE_CAT_PHRASE.search(p):
        wanted = category_of(example)
        if wanted:
            members = {"animals": ANIMALS, "edible": EDIBLE,
                       "wheeled": WHEELED, "motorised": MOTORISED,
                       "tools": TOOLS, "materials": MATERIALS}[wanted]
            return [i for i, lab in enumerate(labels, 1) if lab in members]

    # 4) set predicates ("click each image containing an animal")
    for rx, s in _SET_PREDICATES:
        if rx.search(p):
            idx = [i for i, lab in enumerate(labels, 1) if lab in s]
            return idx

    # 5) plain noun target ("each image containing a bus")
    target = extract_target(p)
    if target:
        return [i for i, lab in enumerate(labels, 1) if lab == target]

    return None


# ── geometry helpers shared with the server ───────────────────────────────

def denorm(point, box):
    """Normalised (0..1) point -> page coordinates inside a bounding box.

    ``box`` is a Playwright-style {"x","y","width","height"} dict (or a
    4-tuple). Values outside 0..1 are clamped INTO the box.
    """
    if isinstance(box, dict):
        left = float(box.get("x", 0.0))
        top = float(box.get("y", 0.0))
        w = float(box.get("width", 0.0))
        h = float(box.get("height", 0.0))
    else:
        left, top, w, h = (float(v) for v in box)
    try:
        x, y = float(point[0]), float(point[1])
    except Exception:
        x, y = 0.5, 0.5
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return (left + x * w, top + y * h)

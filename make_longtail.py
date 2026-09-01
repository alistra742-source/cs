#!/usr/bin/env python3
"""
make_longtail.py — the Brain's 1000-class vocabulary.

The shipped solver knows 60 painted classes. hCaptcha's prompt catalog is
~1000 prompts long, and the live vehicle/tile families use nouns the 60-class
model can't emit (tiger, fork, burger, red car, ...). This module extends
the vocabulary to exactly 1000 classes, all deterministic and network-free:

    60   core classes        (make_dataset.py painters — unchanged)
   454  long-tail base classes (this file: compact recipe painters over
        Pillow primitives + shared archetypes, one recipe per class)
   486  colour compounds     (54 core classes x 9 colours — "red_car":
        hCaptcha serves colour grids, "click each image containing a red
        car", on the same binary family)

        = 1000 classes

Each long-tail class carries:
  * a RECIPE — a list of primitive ops in unit coordinates (0..1) rendered
    into the object layer exactly like a core painter (see render_recipe);
  * a CATEGORY — one of the Brain ontology categories (vehicle, animal,
    food, ...); the Brain seeds its KnowledgeBank from this;
  * a SIZE — a coarse SIZE_RANK value for relational ("largest/smallest")
    rounds;
  * ALIASES — 1-3 common synonyms, merged into hcaptcha_types.SYNONYMS so
    prompts like "bike", "soccer ball", "navy van" canonicalise onto the
    class.

Colour compounds are rendered by painting the core object into an RGBA
layer and RECOLOURING the opaque pixels to the target colour (hue shift +
fixed lightness range), keeping the original shading — "red car" is a real
red car, not a flat disc.

Usage (through make_dataset, which merges everything into CLASSES):
    python make_longtail.py            # self-test: render a contact sheet
"""

import math
import random

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:  # pragma: no cover
    raise SystemExit("Pillow is required:  pip install Pillow")


# ═══════════════════════════════════════════════════════════════════════════
#  Recipe op language
#
#  Opcodes (unit coordinates 0..1, scaled by the render size S):
#    r   (x0, y0, x1, y1, C)                rectangle
#    rr  (x0, y0, x1, y1, rad, C)           rounded rectangle
#    e   (cx, cy, rx, ry, C)                ellipse
#    c   (cx, cy, r, C)                     circle
#    p   [(x, y), ..., C]                   polygon
#    l   (x0, y0, x1, y1, w, C)             line
#    a   (cx, cy, r, a0deg, a1deg, w, C)    arc
#    t   (x, y, "text", size, C)            text
#
#  C is a palette key: "A" primary, "B" secondary, "C" dark, "D" accent,
#  "W" white, "Bk" black. The renderer picks ONE palette from the class's
#  palette list per image, so each class renders in many colour variants
#  (the same trick the core painters use with BODY_COLORS).
# ═══════════════════════════════════════════════════════════════════════════

# shared palette choices (A, B, C, D) — (primary, secondary, dark, accent)
PALETTES = [
    ((196, 60, 55), (232, 174, 44), (90, 24, 20), (245, 245, 248)),
    ((44, 82, 158), (90, 140, 210), (20, 40, 90), (245, 245, 248)),
    ((36, 120, 90), (110, 190, 150), (16, 60, 44), (245, 245, 248)),
    ((178, 106, 32), (222, 160, 84), (96, 56, 16), (245, 245, 248)),
    ((120, 66, 156), (170, 130, 210), (60, 30, 90), (245, 245, 248)),
    ((214, 116, 160), (240, 170, 200), (120, 50, 90), (245, 245, 248)),
    ((70, 72, 80), (130, 134, 145), (30, 32, 38), (245, 245, 248)),
    ((222, 120, 48), (245, 180, 120), (120, 60, 20), (245, 245, 248)),
    ((86, 140, 60), (150, 200, 110), (40, 70, 28), (245, 245, 248)),
    ((58, 132, 170), (120, 180, 215), (26, 66, 96), (245, 245, 248)),
]

COLOUR_KEYS = {"W": (250, 250, 252), "Bk": (24, 24, 28)}

# Species-true palettes for the hCaptcha object-roster animals: hCaptcha's
# own illustrations are colour-consistent (the red panda is always
# rust/cream, the boar brown, the warthog grey-brown), so these classes
# render in a FIXED palette instead of the random per-image pick.
# (A primary, B secondary, C dark, D accent)
PALETTE_OVERRIDES = {
    "red_panda": ((204, 96, 44), (244, 232, 208), (96, 40, 18), (250, 250, 252)),
    "boar": ((146, 98, 66), (196, 156, 116), (74, 48, 32), (250, 250, 252)),
    "warthog": ((148, 128, 112), (188, 164, 150), (72, 58, 50), (250, 250, 252)),
}


def _poly_smoke(d, pts, S, fill, outline=None, width=0):
    d.polygon([(x * S, y * S) for x, y in pts], fill=fill,
              outline=outline, width=max(1, int(width * S)) if width else 0)


def render_recipe(ops, S, rng, palette=None):
    """Render a recipe into an RGBA layer of size S x S (transparent bg)."""
    if palette is None:
        palette = PALETTES[rng.randrange(len(PALETTES))]
    pal = {"A": palette[0], "B": palette[1], "C": palette[2], "D": palette[3],
           **COLOUR_KEYS}
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for op in ops:
        kind = op[0]
        try:
            if kind == "r":
                x0, y0, x1, y1, c = op[1:6]
                d.rectangle([x0 * S, y0 * S, x1 * S, y1 * S],
                            fill=pal[c] + (255,))
            elif kind == "rr":
                x0, y0, x1, y1, rad, c = op[1:7]
                d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S],
                                    radius=max(1, rad * S), fill=pal[c] + (255,))
            elif kind == "e":
                cx, cy, rx, ry, c = op[1:6]
                d.ellipse([(cx - rx) * S, (cy - ry) * S,
                           (cx + rx) * S, (cy + ry) * S], fill=pal[c] + (255,))
            elif kind == "c":
                cx, cy, r, c = op[1:5]
                d.ellipse([(cx - r) * S, (cy - r) * S,
                           (cx + r) * S, (cy + r) * S], fill=pal[c] + (255,))
            elif kind == "p":
                pts, c = op[1], op[2]
                outline = op[3] if len(op) > 3 else None
                _poly_smoke(d, pts, S, pal[c] + (255,),
                            None if outline is None else pal[outline] + (255,))
            elif kind == "l":
                x0, y0, x1, y1, w, c = op[1:7]
                d.line([x0 * S, y0 * S, x1 * S, y1 * S], fill=pal[c] + (255,),
                       width=max(1, int(w * S)))
            elif kind == "a":
                cx, cy, r, a0, a1, w, c = op[1:8]
                d.arc([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
                      a0, a1, fill=pal[c] + (255,),
                      width=max(1, int(w * S)))
            elif kind == "t":
                x, y, s, sz, c = op[1:6]
                try:
                    from PIL import ImageFont
                    font = ImageFont.load_default(size=max(4, int(sz * S)))
                except Exception:
                    font = None
                d.text((x * S, y * S), s, font=font, fill=pal[c] + (255,))
        except Exception:
            continue
    return layer


# ═══════════════════════════════════════════════════════════════════════════
#  Archetypes — shared silhouettes that keep 454 recipes compact while
#  staying visually distinct at 64 px (silhouette is what matters at tile
#  scale; the palette + small per-class op differences carry identity).
# ═══════════════════════════════════════════════════════════════════════════

def quaduped(ear="flat", tail="mid", legs=4, snout=0.30, hump=0.0):
    """A four-legged animal: body oval + head + snout + ears + legs + tail.
    Params pick the silhouette flavour (ear shape, tail, hump for rhino/
    bull, long snout for rhino/hippo)."""
    ops = []
    ops.append(("e", 0.46, 0.52, 0.24, 0.15, "A"))            # body
    if hump:
        ops.append(("e", 0.56, 0.40, 0.12, hump, "A"))        # shoulder hump
    ops.append(("e", 0.74, 0.42, 0.10, 0.09, "A"))            # head
    ops.append(("e", 0.82 + snout * 0.08, 0.46, 0.05 + snout * 0.05,
                0.035, "B"))                                   # snout
    if ear == "flat":
        ops.append(("p", [(0.70, 0.34), (0.74, 0.22), (0.78, 0.34)], "A"))
    elif ear == "big":
        ops.append(("e", 0.72, 0.28, 0.045, 0.075, "A"))
        ops.append(("e", 0.78, 0.28, 0.045, 0.075, "A"))
    elif ear == "tall":
        ops.append(("p", [(0.71, 0.35), (0.72, 0.18), (0.76, 0.33)], "A"))
        ops.append(("p", [(0.77, 0.33), (0.81, 0.18), (0.82, 0.35)], "A"))
    else:  # "bald"
        pass
    ops.append(("c", 0.77, 0.40, 0.012, "Bk"))                # eye
    for lx in (0.30, 0.40, 0.55, 0.65):                       # legs
        if legs < 4 and lx in (0.40, 0.55):
            continue
        ops.append(("r", lx - 0.025, 0.58, lx + 0.025, 0.80, "A"))
    if tail == "mid":
        ops.append(("l", 0.23, 0.50, 0.14, 0.42, 0.02, "A"))
    elif tail == "up":
        ops.append(("l", 0.23, 0.48, 0.12, 0.30, 0.025, "A"))
    elif tail == "long":
        ops.append(("a", 0.20, 0.45, 0.10, 180, 340, 0.03, "A"))
    elif tail == "none":
        pass
    else:  # "bushy"
        ops.append(("e", 0.16, 0.42, 0.05, 0.035, "B"))
    return ops


def bird(beak="mid", wings="up", tail="mid", legs=0.20, crest=0.0):
    """A bird: body teardrop + head + beak + wing + tail + legs."""
    ops = []
    ops.append(("e", 0.48, 0.52, 0.17, 0.12, "A"))            # body
    ops.append(("c", 0.68, 0.40, 0.075, "A"))                 # head
    if crest:
        for i in range(3):
            ops.append(("l", 0.66 + i * 0.03, 0.33, 0.64 + i * 0.04,
                        0.33 - crest, 0.012, "D"))
    if beak == "mid":
        ops.append(("p", [(0.75, 0.38), (0.88, 0.42), (0.75, 0.45)], "D"))
    elif beak == "long":
        ops.append(("l", 0.75, 0.40, 0.94, 0.46, 0.02, "D"))
        ops.append(("l", 0.75, 0.44, 0.94, 0.50, 0.02, "D"))
    elif beak == "hook":
        ops.append(("p", [(0.75, 0.38), (0.90, 0.40), (0.86, 0.50),
                          (0.75, 0.45)], "D"))
    else:  # "duck" (broad bill)
        ops.append(("e", 0.84, 0.43, 0.08, 0.035, "D"))
    ops.append(("c", 0.70, 0.38, 0.014, "Bk"))                # eye
    if wings == "up":
        ops.append(("p", [(0.40, 0.50), (0.55, 0.30), (0.62, 0.48)], "B"))
    elif wings == "down":
        ops.append(("p", [(0.36, 0.52), (0.55, 0.62), (0.62, 0.52)], "B"))
    else:  # "flat"
        ops.append(("e", 0.46, 0.54, 0.12, 0.06, "B"))
    if tail == "mid":
        ops.append(("p", [(0.32, 0.50), (0.14, 0.56), (0.32, 0.60)], "B"))
    elif tail == "long":
        ops.append(("p", [(0.32, 0.48), (0.08, 0.60), (0.32, 0.60)], "B"))
    elif tail == "fan":
        for i in range(5):
            ops.append(("l", 0.33, 0.54, 0.33 - 0.16 * (0.5 + 0.5 * math.cos(
                math.radians(20 * i - 40))), 0.54 + 0.14 * math.sin(
                math.radians(20 * i - 40)), 0.02, "B"))
    else:  # "short"
        ops.append(("p", [(0.33, 0.52), (0.24, 0.55), (0.33, 0.58)], "B"))
    ly = 0.66
    ops.append(("l", 0.44, ly, 0.44, ly + legs, 0.015, "D"))
    ops.append(("l", 0.52, ly, 0.52, ly + legs, 0.015, "D"))
    return ops


def fish(tail="mid", fins=1):
    """A fish: body + tail + dorsal fin + eye."""
    ops = []
    ops.append(("p", [(0.25, 0.50), (0.45, 0.30), (0.72, 0.38), (0.78, 0.50),
                      (0.72, 0.62), (0.45, 0.70), (0.25, 0.50)], "A"))
    if tail == "mid":
        ops.append(("p", [(0.25, 0.50), (0.10, 0.36), (0.14, 0.50),
                          (0.10, 0.64)], "B"))
    elif tail == "fan":
        for i in range(4):
            ops.append(("l", 0.26, 0.50, 0.10, 0.34 + i * 0.11, 0.025, "B"))
    else:  # "round"
        ops.append(("e", 0.16, 0.50, 0.06, 0.10, "B"))
    if fins:
        ops.append(("p", [(0.44, 0.31), (0.52, 0.18), (0.62, 0.33)], "B"))
    ops.append(("c", 0.68, 0.46, 0.025, "W"))
    ops.append(("c", 0.68, 0.46, 0.013, "Bk"))
    for i in range(3):                                        # gill lines
        ops.append(("l", 0.58 + i * 0.03, 0.38, 0.55 + i * 0.03,
                    0.60, 0.008, "C"))
    return ops


def insect(body="oval", wings=1, legs=6):
    """A bug: body + head + wings + legs + antennae."""
    ops = []
    if body == "oval":
        ops.append(("e", 0.50, 0.55, 0.10, 0.16, "A"))
        ops.append(("e", 0.50, 0.34, 0.07, 0.07, "C"))
    else:  # "long"
        ops.append(("rr", 0.42, 0.40, 0.58, 0.74, 0.06, "A"))
        ops.append(("c", 0.50, 0.33, 0.06, "C"))
    if wings:
        ops.append(("e", 0.40, 0.48, 0.05, 0.14, "D"))
        ops.append(("e", 0.60, 0.48, 0.05, 0.14, "D"))
    for side in (-1, 1):
        for i in range(legs // 2):
            y = 0.48 + i * 0.09
            x0 = 0.50 + side * 0.08
            ops.append(("l", x0, y, x0 + side * 0.12, y + 0.06, 0.012, "Bk"))
    ops.append(("l", 0.47, 0.28, 0.42, 0.18, 0.01, "Bk"))
    ops.append(("l", 0.53, 0.28, 0.58, 0.18, 0.01, "Bk"))
    return ops


def ball(markings="spots", colour="A"):
    """A round ball with a marking pattern (identity at tile scale)."""
    ops = [("c", 0.5, 0.5, 0.26, colour)]
    if markings == "spots":
        for (sx, sy) in [(0.42, 0.42), (0.58, 0.46), (0.48, 0.60),
                         (0.62, 0.58), (0.40, 0.55)]:
            ops.append(("c", sx, sy, 0.028, "Bk"))
    elif markings == "panel":
        ops.append(("p", [(0.32, 0.50), (0.42, 0.36), (0.52, 0.50),
                          (0.42, 0.66)], "Bk"))
        ops.append(("a", 0.5, 0.5, 0.26, 200, 340, 0.02, "Bk"))
        ops.append(("a", 0.5, 0.5, 0.26, 20, 160, 0.02, "Bk"))
    elif markings == "stripes":
        for i in range(3):
            ops.append(("l", 0.28, 0.40 + i * 0.10, 0.72,
                        0.40 + i * 0.10, 0.03, "Bk"))
    elif markings == "cross":
        ops.append(("l", 0.30, 0.50, 0.70, 0.50, 0.035, "Bk"))
        ops.append(("l", 0.50, 0.30, 0.50, 0.70, 0.035, "Bk"))
    elif markings == "zigzag":
        pts = [(0.28, 0.55)]
        for i in range(5):
            pts.append((0.28 + 0.088 * i, 0.45 if i % 2 == 0 else 0.55))
        ops.append(("p", pts + [(0.72, 0.62), (0.28, 0.62)], "Bk"))
    else:  # "plain"
        ops.append(("a", 0.5, 0.5, 0.20, 180, 300, 0.02, "W"))
    return ops


def tool_handle(head_ops, handle_col="B"):
    """A handheld tool: D-grip handle + per-class head ops on top."""
    ops = [
        ("rr", 0.455, 0.42, 0.545, 0.84, 0.05, handle_col),   # grip
        ("l", 0.42, 0.46, 0.58, 0.46, 0.03, handle_col),      # neck
    ]
    return list(head_ops) + ops


def vehicle(cab="left", wheels=2, long=0.0):
    """A small road vehicle silhouette: body + cab + wheels."""
    ops = []
    y0, y1 = 0.42, 0.62
    ops.append(("rr", 0.10, y0, 0.90 - long, y1, 0.05, "A"))
    if cab == "left":
        ops.append(("rr", 0.16, 0.28, 0.42, 0.46, 0.04, "A"))
        ops.append(("e", 0.22, 0.36, 0.045, 0.05, "D"))
        ops.append(("e", 0.34, 0.36, 0.045, 0.05, "D"))
    elif cab == "right":
        ops.append(("rr", 0.55, 0.28, 0.80, 0.46, 0.04, "A"))
        ops.append(("e", 0.62, 0.36, 0.045, 0.05, "D"))
        ops.append(("e", 0.74, 0.36, 0.045, 0.05, "D"))
    elif cab == "truck":
        ops.append(("rr", 0.62, 0.26, 0.86, 0.46, 0.03, "B"))
        ops.append(("e", 0.70, 0.36, 0.04, 0.05, "D"))
    else:  # "van" (single cab line)
        ops.append(("l", 0.46, 0.30, 0.46, 0.44, 0.02, "C"))
        ops.append(("e", 0.24, 0.36, 0.05, 0.05, "D"))
    for i in range(wheels):
        wx = 0.24 + i * (0.52 if wheels == 2 else 0.26)
        ops.append(("c", wx, 0.66, 0.075, "Bk"))
        ops.append(("c", wx, 0.66, 0.035, "D"))
    return ops


def fruit(top="stem", leafy=False):
    """A round fruit: body + stem/leaf + shine."""
    ops = [("c", 0.5, 0.56, 0.24, "A")]
    if top == "stem":
        ops.append(("l", 0.5, 0.33, 0.53, 0.24, 0.02, "C"))
    elif top == "cap":
        ops.append(("p", [(0.40, 0.34), (0.45, 0.24), (0.50, 0.33),
                          (0.55, 0.24), (0.60, 0.34)], "C"))
    elif top == "bumpy":
        for i in range(6):
            a = math.radians(i * 60)
            ops.append(("c", 0.5 + 0.24 * math.cos(a) * 0.5,
                        0.56 + 0.24 * math.sin(a) * 0.5, 0.05, "B"))
    elif top == "spikes":
        for i in range(8):
            a = math.radians(i * 45 - 90)
            ops.append(("p", [(0.5 + 0.20 * math.cos(a - 0.18),
                               0.56 + 0.20 * math.sin(a - 0.18)),
                              (0.5 + 0.30 * math.cos(a),
                               0.56 + 0.30 * math.sin(a)),
                              (0.5 + 0.20 * math.cos(a + 0.18),
                               0.56 + 0.20 * math.sin(a + 0.18))], "A"))
    if leafy:
        ops.append(("e", 0.58, 0.30, 0.07, 0.035, "B"))
    ops.append(("a", 0.42, 0.46, 0.08, 200, 300, 0.02, "W"))
    return ops


def plant(stem_h=0.30, leaves=5, flower=False, cluster=False):
    """A plant: pot-less stem + leaves/flower head."""
    ops = []
    x0, y1 = 0.5, 0.86
    ops.append(("l", x0, y1, x0, y1 - stem_h, 0.025, "B"))
    for i in range(leaves):
        y = y1 - stem_h * (0.25 + 0.6 * i / max(1, leaves - 1))
        for side in (-1, 1):
            ops.append(("e", x0 + side * 0.09, y, 0.08, 0.035, "B"))
            ops[-1] = ("p", [(x0, y), (x0 + side * 0.16, y - 0.03),
                             (x0 + side * 0.13, y + 0.04)], "B")
    if flower:
        for i in range(6):
            a = math.radians(i * 60)
            ops.append(("e", x0 + 0.06 * math.cos(a),
                        y1 - stem_h - 0.04 + 0.06 * math.sin(a), 0.045, "A"))
        ops.append(("c", x0, y1 - stem_h - 0.04, 0.035, "D"))
    if cluster:
        for (bx, by, r) in [(0.36, 0.5, 0.07), (0.5, 0.44, 0.08),
                            (0.64, 0.5, 0.07), (0.43, 0.6, 0.07),
                            (0.57, 0.6, 0.07)]:
            ops.append(("c", bx, by, r, "A"))
    return ops


def building(floors=3, door=True, sign=False):
    """A building: slab + window grid + door + optional sign."""
    ops = [("r", 0.25, 0.16, 0.75, 0.86, "A")]
    ops.append(("r", 0.23, 0.12, 0.77, 0.18, "C"))
    for f in range(floors):
        for c in range(3):
            ops.append(("e", 0.35 + c * 0.125, 0.28 + f * 0.17, 0.04,
                        0.055, "D"))
    if door:
        ops.append(("rr", 0.45, 0.66, 0.55, 0.86, 0.02, "C"))
    if sign:
        ops.append(("r", 0.42, 0.05, 0.58, 0.12, "D"))
    return ops


def street_pole(head="light", cross=False):
    """A street fixture: pole + head (light / sign / box / camera)."""
    ops = [("r", 0.475, 0.28, 0.525, 0.88, 0.02, "C")]
    ops.append(("r", 0.40, 0.86, 0.60, 0.92, "C"))
    if head == "light":
        ops.append(("r", 0.30, 0.20, 0.70, 0.28, "C"))
        ops.append(("e", 0.36, 0.24, 0.04, 0.035, "A"))
        ops.append(("e", 0.50, 0.24, 0.04, 0.035, "B"))
        ops.append(("e", 0.64, 0.24, 0.04, 0.035, "D"))
    elif head == "sign":
        ops.append(("c", 0.5, 0.16, 0.11, "A"))
        ops.append(("l", 0.5, 0.10, 0.5, 0.22, 0.02, "W"))
        ops.append(("l", 0.44, 0.16, 0.56, 0.16, 0.02, "W"))
    elif head == "box":
        ops.append(("r", 0.34, 0.14, 0.66, 0.30, "A"))
        ops.append(("r", 0.40, 0.20, 0.60, 0.24, "W"))
    elif head == "camera":
        ops.append(("r", 0.34, 0.16, 0.62, 0.28, "A"))
        ops.append(("c", 0.56, 0.22, 0.03, "Bk"))
    else:  # "flag"
        ops.append(("p", [(0.52, 0.14), (0.74, 0.20), (0.52, 0.26)], "A"))
    if cross:
        ops.append(("l", 0.30, 0.34, 0.70, 0.34, 0.02, "C"))
    return ops


def garment(shape="top"):
    """A garment silhouette (top / dress / pants / shoe / hat)."""
    if shape == "top":
        return [("p", [(0.30, 0.28), (0.42, 0.22), (0.58, 0.22), (0.70, 0.28),
                       (0.64, 0.40), (0.58, 0.36), (0.58, 0.78), (0.42, 0.78),
                       (0.42, 0.36), (0.36, 0.40)], "A"),
                ("a", 0.5, 0.235, 0.05, 180, 360, 0.02, "C")]
    if shape == "dress":
        return [("p", [(0.40, 0.20), (0.60, 0.20), (0.58, 0.42), (0.74, 0.82),
                       (0.26, 0.82), (0.42, 0.42)], "A"),
                ("a", 0.5, 0.215, 0.05, 180, 360, 0.02, "C")]
    if shape == "pants":
        return [("p", [(0.36, 0.20), (0.64, 0.20), (0.66, 0.42), (0.60, 0.84),
                       (0.52, 0.84), (0.50, 0.46), (0.48, 0.84), (0.40, 0.84),
                       (0.34, 0.42)], "A"),
                ("l", 0.36, 0.26, 0.64, 0.26, 0.025, "C")]
    if shape == "shoe":
        return [("p", [(0.20, 0.55), (0.42, 0.55), (0.52, 0.42), (0.62, 0.50),
                       (0.82, 0.58), (0.82, 0.68), (0.20, 0.68)], "A"),
                ("l", 0.20, 0.64, 0.82, 0.64, 0.02, "C"),
                ("l", 0.44, 0.56, 0.50, 0.50, 0.012, "W"),
                ("l", 0.50, 0.54, 0.56, 0.48, 0.012, "W")]
    if shape == "hat":
        return [("p", [(0.24, 0.55), (0.36, 0.30), (0.64, 0.30), (0.76, 0.55),
                       (0.86, 0.60), (0.14, 0.60)], "A"),
                ("r", 0.24, 0.52, 0.76, 0.60, "C")]
    if shape == "cap":
        return [("p", [(0.26, 0.50), (0.34, 0.30), (0.62, 0.30), (0.70, 0.50)],
                 "A"),
                ("p", [(0.26, 0.50), (0.84, 0.50), (0.80, 0.58), (0.26, 0.58)],
                 "C")]
    if shape == "glove":
        return [("p", [(0.40, 0.30), (0.60, 0.30), (0.62, 0.62), (0.74, 0.52),
                       (0.78, 0.60), (0.66, 0.72), (0.62, 0.80), (0.40, 0.80),
                       (0.38, 0.62)], "A"),
                ("l", 0.46, 0.34, 0.46, 0.52, 0.012, "C"),
                ("l", 0.53, 0.34, 0.53, 0.52, 0.012, "C"),
                ("l", 0.59, 0.34, 0.59, 0.52, 0.012, "C")]
    if shape == "scarf":
        return [("rr", 0.38, 0.20, 0.62, 0.44, 0.06, "A"),
                ("r", 0.44, 0.40, 0.54, 0.82, "A"),
                ("l", 0.44, 0.78, 0.54, 0.78, 0.02, "D"),
                ("l", 0.38, 0.30, 0.62, 0.30, 0.02, "D")]
    if shape == "belt":
        return [("p", [(0.16, 0.44), (0.84, 0.44), (0.84, 0.58), (0.16, 0.58)],
                 "A"),
                ("rr", 0.44, 0.38, 0.56, 0.64, 0.02, "D"),
                ("l", 0.44, 0.51, 0.56, 0.51, 0.015, "Bk")]
    if shape == "sock":
        return [("p", [(0.42, 0.20), (0.60, 0.20), (0.60, 0.58), (0.72, 0.72),
                       (0.56, 0.84), (0.36, 0.84), (0.36, 0.66), (0.42, 0.60)],
                 "A"),
                ("r", 0.40, 0.20, 0.62, 0.28, "D")]
    return [("e", 0.5, 0.5, 0.2, "A")]


def small_object(shape="key"):
    """Small household items (key, padlock, bottle, jar, ring, ...)."""
    if shape == "key":
        return [("c", 0.32, 0.5, 0.13, "A"), ("c", 0.32, 0.5, 0.05, "W"),
                ("r", 0.42, 0.47, 0.80, 0.53, "A"),
                ("r", 0.66, 0.53, 0.70, 0.62, "A"),
                ("r", 0.74, 0.53, 0.78, 0.64, "A")]
    if shape == "padlock":
        return [("rr", 0.34, 0.44, 0.66, 0.80, 0.05, "A"),
                ("a", 0.5, 0.44, 0.13, 180, 360, 0.035, "C"),
                ("c", 0.5, 0.58, 0.035, "Bk"),
                ("l", 0.5, 0.58, 0.5, 0.66, 0.02, "Bk")]
    if shape == "bottle":
        return [("rr", 0.42, 0.34, 0.58, 0.82, 0.04, "A"),
                ("r", 0.46, 0.20, 0.54, 0.34, "A"),
                ("r", 0.45, 0.16, 0.55, 0.22, "D"),
                ("r", 0.42, 0.50, 0.58, 0.68, "D")]
    if shape == "jar":
        return [("rr", 0.36, 0.34, 0.64, 0.82, 0.05, "A"),
                ("r", 0.38, 0.24, 0.62, 0.34, "D"),
                ("l", 0.38, 0.30, 0.62, 0.30, 0.015, "C")]
    if shape == "ring":
        return [("a", 0.5, 0.58, 0.18, 0, 360, 0.06, "A"),
                ("c", 0.5, 0.34, 0.05, "D")]
    if shape == "watch":
        return [("rr", 0.44, 0.20, 0.56, 0.38, 0.02, "C"),
                ("rr", 0.44, 0.62, 0.56, 0.80, 0.02, "C"),
                ("c", 0.5, 0.5, 0.15, "A"),
                ("c", 0.5, 0.5, 0.11, "D"),
                ("l", 0.5, 0.5, 0.5, 0.43, 0.015, "Bk"),
                ("l", 0.5, 0.5, 0.55, 0.5, 0.015, "Bk")]
    if shape == "backpack":
        return [("rr", 0.34, 0.28, 0.66, 0.82, 0.08, "A"),
                ("rr", 0.40, 0.54, 0.60, 0.80, 0.05, "B"),
                ("a", 0.44, 0.28, 0.08, 180, 360, 0.025, "C"),
                ("a", 0.56, 0.28, 0.08, 180, 360, 0.025, "C")]
    if shape == "suitcase":
        return [("rr", 0.30, 0.34, 0.70, 0.80, 0.05, "A"),
                ("r", 0.44, 0.24, 0.56, 0.36, "C"),
                ("l", 0.42, 0.34, 0.42, 0.80, 0.02, "C"),
                ("l", 0.58, 0.34, 0.58, 0.80, 0.02, "C")]
    if shape == "wallet":
        return [("rr", 0.28, 0.38, 0.72, 0.66, 0.03, "A"),
                ("c", 0.64, 0.52, 0.025, "D")]
    if shape == "cup":
        return [("rr", 0.38, 0.34, 0.62, 0.78, 0.03, "A"),
                ("a", 0.62, 0.56, 0.10, -80, 80, 0.03, "A")]
    if shape == "plate":
        return [("e", 0.5, 0.5, 0.28, 0.09, "A"),
                ("e", 0.5, 0.5, 0.18, 0.06, "D")]
    if shape == "bowl":
        return [("p", [(0.24, 0.44), (0.76, 0.44), (0.66, 0.74), (0.34, 0.74)],
                 "A"),
                ("e", 0.5, 0.44, 0.26, 0.07, "D")]
    if shape == "mug":
        return [("rr", 0.36, 0.36, 0.60, 0.78, 0.03, "A"),
                ("a", 0.60, 0.57, 0.09, -90, 90, 0.035, "A")]
    if shape == "vase":
        return [("p", [(0.44, 0.24), (0.56, 0.24), (0.54, 0.36), (0.64, 0.52),
                       (0.62, 0.74), (0.50, 0.82), (0.38, 0.74), (0.36, 0.52),
                       (0.46, 0.36)], "A"),
                ("l", 0.52, 0.24, 0.58, 0.12, 0.015, "B"),
                ("l", 0.46, 0.24, 0.40, 0.10, 0.015, "B")]
    if shape == "lamp":
        return [("p", [(0.38, 0.20), (0.62, 0.20), (0.70, 0.44), (0.30, 0.44)],
                 "A"),
                ("r", 0.48, 0.44, 0.52, 0.76, "C"),
                ("r", 0.38, 0.76, 0.62, 0.84, "C")]
    if shape == "fireplace":
        return [("r", 0.22, 0.30, 0.78, 0.82, "C"),
                ("rr", 0.36, 0.46, 0.64, 0.82, 0.04, "Bk"),
                ("p", [(0.44, 0.80), (0.50, 0.56), (0.56, 0.80)], "A"),
                ("p", [(0.50, 0.80), (0.55, 0.64), (0.60, 0.80)], "D")]
    if shape == "bed":
        return [("rr", 0.20, 0.40, 0.80, 0.72, 0.04, "A"),
                ("rr", 0.24, 0.34, 0.44, 0.50, 0.03, "D"),
                ("r", 0.22, 0.70, 0.28, 0.82, "C"),
                ("r", 0.72, 0.70, 0.78, 0.82, "C")]
    if shape == "sofa":
        return [("rr", 0.20, 0.44, 0.80, 0.72, 0.05, "A"),
                ("rr", 0.16, 0.34, 0.30, 0.72, 0.05, "B"),
                ("rr", 0.70, 0.34, 0.84, 0.72, 0.05, "B"),
                ("rr", 0.28, 0.40, 0.48, 0.56, 0.03, "D"),
                ("rr", 0.52, 0.40, 0.72, 0.56, 0.03, "D"),
                ("r", 0.24, 0.72, 0.30, 0.82, "C"),
                ("r", 0.70, 0.72, 0.76, 0.82, "C")]
    return [("e", 0.5, 0.5, 0.18, "A")]


def nature(kind="cloud"):
    """Sky / landscape icons."""
    if kind == "cloud":
        return [("e", 0.38, 0.52, 0.14, 0.10, "A"),
                ("e", 0.54, 0.46, 0.16, 0.13, "A"),
                ("e", 0.68, 0.54, 0.13, 0.09, "A"),
                ("r", 0.30, 0.52, 0.78, 0.63, "A")]
    if kind == "sun":
        ops = [("c", 0.5, 0.5, 0.16, "A")]
        for i in range(8):
            a = math.radians(i * 45)
            ops.append(("l", 0.5 + 0.21 * math.cos(a), 0.5 + 0.21 * math.sin(a),
                        0.5 + 0.30 * math.cos(a), 0.5 + 0.30 * math.sin(a),
                        0.035, "A"))
        return ops
    if kind == "moon":
        return [("c", 0.5, 0.5, 0.22, "A"),
                ("c", 0.58, 0.44, 0.18, "B")]
    if kind == "star":
        pts = []
        for i in range(10):
            r = 0.28 if i % 2 == 0 else 0.12
            a = math.radians(i * 36 - 90)
            pts.append((0.5 + r * math.cos(a), 0.5 + r * math.sin(a)))
        return [("p", pts, "A")]
    if kind == "rainbow":
        ops = []
        cols = ["A", "B", "D", "C"]
        for i, c in enumerate(cols):
            ops.append(("a", 0.5, 0.78, 0.26 - i * 0.045, 180, 360, 0.04, c))
        return ops
    if kind == "snowflake":
        ops = []
        for i in range(6):
            a = math.radians(i * 60)
            ops.append(("l", 0.5, 0.5, 0.5 + 0.28 * math.cos(a),
                        0.5 + 0.28 * math.sin(a), 0.025, "A"))
            ops.append(("l", 0.5 + 0.18 * math.cos(a - 0.5),
                        0.5 + 0.18 * math.sin(a - 0.5), 0.5 + 0.24 * math.cos(a),
                        0.5 + 0.24 * math.sin(a), 0.018, "A"))
            ops.append(("l", 0.5 + 0.18 * math.cos(a + 0.5),
                        0.5 + 0.18 * math.sin(a + 0.5), 0.5 + 0.24 * math.cos(a),
                        0.5 + 0.24 * math.sin(a), 0.018, "A"))
        return ops
    if kind == "snowman":
        return [("c", 0.5, 0.68, 0.18, "A"), ("c", 0.5, 0.38, 0.12, "A"),
                ("c", 0.46, 0.35, 0.012, "Bk"), ("c", 0.54, 0.35, 0.012, "Bk"),
                ("l", 0.5, 0.38, 0.62, 0.36, 0.015, "A"),
                ("l", 0.38, 0.52, 0.26, 0.44, 0.015, "C"),
                ("l", 0.62, 0.52, 0.74, 0.44, 0.015, "C"),
                ("c", 0.5, 0.62, 0.012, "Bk"), ("c", 0.5, 0.70, 0.012, "Bk")]
    if kind == "volcano":
        return [("p", [(0.24, 0.80), (0.44, 0.34), (0.48, 0.30), (0.52, 0.30),
                       (0.56, 0.34), (0.76, 0.80)], "A"),
                ("p", [(0.44, 0.34), (0.50, 0.14), (0.56, 0.34)], "D")]
    if kind == "waterfall":
        return [("r", 0.20, 0.24, 0.44, 0.84, "A"),
                ("r", 0.56, 0.24, 0.80, 0.84, "A"),
                ("r", 0.44, 0.30, 0.56, 0.84, "D"),
                ("l", 0.47, 0.34, 0.47, 0.82, 0.012, "W"),
                ("l", 0.53, 0.34, 0.53, 0.82, 0.012, "W"),
                ("r", 0.36, 0.84, 0.64, 0.90, "B")]
    if kind == "palm":
        return [("l", 0.5, 0.84, 0.46, 0.42, 0.045, "A"),
                ("p", [(0.46, 0.44), (0.28, 0.34), (0.44, 0.40)], "B"),
                ("p", [(0.46, 0.44), (0.34, 0.26), (0.48, 0.38)], "B"),
                ("p", [(0.46, 0.44), (0.52, 0.24), (0.50, 0.40)], "B"),
                ("p", [(0.46, 0.44), (0.64, 0.28), (0.50, 0.40)], "B"),
                ("p", [(0.46, 0.44), (0.70, 0.40), (0.50, 0.46)], "B"),
                ("c", 0.44, 0.48, 0.03, "C"), ("c", 0.50, 0.50, 0.03, "C")]
    if kind == "leaf":
        return [("p", [(0.5, 0.18), (0.74, 0.44), (0.5, 0.82), (0.26, 0.44)],
                 "A"),
                ("l", 0.5, 0.22, 0.5, 0.78, 0.015, "C"),
                ("l", 0.5, 0.40, 0.62, 0.34, 0.012, "C"),
                ("l", 0.5, 0.56, 0.38, 0.50, 0.012, "C")]
    if kind == "mushroom":
        return [("r", 0.44, 0.50, 0.56, 0.82, "A"),
                ("p", [(0.26, 0.50), (0.5, 0.22), (0.74, 0.50)], "D"),
                ("c", 0.42, 0.40, 0.03, "W"), ("c", 0.58, 0.38, 0.025, "W")]
    return [("e", 0.5, 0.5, 0.2, "A")]

# ═══════════════════════════════════════════════════════════════════
#  The 454 long-tail base classes
#
#  (name, category, size, recipe) — size is a coarse SIZE_RANK
#  (1..35) used for relational rounds and the ontology.
# ═══════════════════════════════════════════════════════════════════

LONGTAIL = [
    ('tiger', 'animal', 26, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.844, 0.46, 0.065, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('a', 0.2, 0.45, 0.1, 180, 340, 0.03, 'A'), ('l', 0.36, 0.46, 0.4, 0.58, 0.02, 'C'), ('l', 0.48, 0.44, 0.52, 0.6, 0.02, 'C'), ('l', 0.58, 0.46, 0.62, 0.58, 0.02, 'C'), ('l', 0.66, 0.46, 0.68, 0.56, 0.02, 'C')]),
    ('leopard', 'animal', 25, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.844, 0.46, 0.065, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('a', 0.2, 0.45, 0.1, 180, 340, 0.03, 'A'), ('c', 0.4, 0.5, 0.02, 'Bk'), ('c', 0.52, 0.48, 0.02, 'Bk'), ('c', 0.6, 0.54, 0.02, 'Bk'), ('c', 0.46, 0.58, 0.02, 'Bk')]),
    ('cheetah', 'animal', 24, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.844, 0.46, 0.065, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('a', 0.2, 0.45, 0.1, 180, 340, 0.03, 'A'), ('c', 0.42, 0.5, 0.016, 'Bk'), ('c', 0.54, 0.52, 0.016, 'Bk'), ('c', 0.62, 0.5, 0.016, 'Bk')]),
    ('fox', 'animal', 16, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.852, 0.46, 0.07, 0.035, 'B'), ('p', [(0.71, 0.35), (0.72, 0.18), (0.76, 0.33)], 'A'), ('p', [(0.77, 0.33), (0.81, 0.18), (0.82, 0.35)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('e', 0.16, 0.42, 0.05, 0.035, 'B')]),
    ('wolf', 'animal', 22, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8503999999999999, 0.46, 0.069, 0.035, 'B'), ('p', [(0.71, 0.35), (0.72, 0.18), (0.76, 0.33)], 'A'), ('p', [(0.77, 0.33), (0.81, 0.18), (0.82, 0.35)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('l', 0.23, 0.5, 0.14, 0.42, 0.02, 'A')]),
    ('skunk', 'animal', 14, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8535999999999999, 0.46, 0.07100000000000001, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('e', 0.16, 0.42, 0.05, 0.035, 'B'), ('l', 0.34, 0.44, 0.68, 0.44, 0.035, 'W')]),
    ('raccoon', 'animal', 15, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8472, 0.46, 0.067, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('e', 0.16, 0.42, 0.05, 0.035, 'B'), ('e', 0.72, 0.4, 0.03, 0.022, 'Bk'), ('e', 0.79, 0.4, 0.03, 0.022, 'Bk')]),
    ('deer', 'animal', 24, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8535999999999999, 0.46, 0.07100000000000001, 0.035, 'B'), ('p', [(0.71, 0.35), (0.72, 0.18), (0.76, 0.33)], 'A'), ('p', [(0.77, 0.33), (0.81, 0.18), (0.82, 0.35)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('l', 0.73, 0.3, 0.71, 0.18, 0.015, 'C'), ('l', 0.76, 0.28, 0.78, 0.16, 0.015, 'C'), ('l', 0.74, 0.22, 0.7, 0.2, 0.012, 'C')]),
    ('moose', 'animal', 30, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.86, 0.46, 0.07500000000000001, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('p', [(0.68, 0.3), (0.7, 0.14), (0.74, 0.22), (0.72, 0.14), (0.76, 0.3)], 'C')]),
    ('rhino', 'animal', 31, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.864, 0.46, 0.07750000000000001, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('l', 0.23, 0.5, 0.14, 0.42, 0.02, 'A'), ('p', [(0.84, 0.44), (0.92, 0.36), (0.86, 0.48)], 'W')]),
    ('hippo', 'animal', 30, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.868, 0.46, 0.08, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('l', 0.23, 0.5, 0.14, 0.42, 0.02, 'A'), ('e', 0.86, 0.47, 0.04, 0.02, 'Bk')]),
    ('gorilla', 'animal', 27, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8488, 0.46, 0.068, 0.035, 'B'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('l', 0.4, 0.78, 0.36, 0.9, 0.03, 'A'), ('l', 0.6, 0.78, 0.64, 0.9, 0.03, 'A')]),
    ('monkey', 'animal', 14, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.844, 0.46, 0.065, 0.035, 'B'), ('e', 0.72, 0.28, 0.045, 0.075, 'A'), ('e', 0.78, 0.28, 0.045, 0.075, 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('a', 0.2, 0.45, 0.1, 180, 340, 0.03, 'A'), ('e', 0.74, 0.44, 0.05, 0.04, 'D')]),
    ('koala', 'animal', 13, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.844, 0.46, 0.065, 0.035, 'B'), ('e', 0.72, 0.28, 0.045, 0.075, 'A'), ('e', 0.78, 0.28, 0.045, 0.075, 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('e', 0.78, 0.46, 0.03, 0.025, 'Bk')]),
    ('panda', 'animal', 22, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8488, 0.46, 0.068, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('e', 0.72, 0.36, 0.035, 0.04, 'Bk'), ('e', 0.79, 0.36, 0.035, 0.04, 'Bk'), ('r', 0.34, 0.48, 0.44, 0.58, 'Bk'), ('r', 0.58, 0.48, 0.68, 0.58, 'Bk')]),
    ('sloth', 'animal', 18, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8472, 0.46, 0.067, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A')]),
    ('beaver', 'animal', 15, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8488, 0.46, 0.068, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('e', 0.18, 0.6, 0.08, 0.045, 'C')]),
    ('squirrel', 'animal', 11, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8472, 0.46, 0.067, 0.035, 'B'), ('p', [(0.71, 0.35), (0.72, 0.18), (0.76, 0.33)], 'A'), ('p', [(0.77, 0.33), (0.81, 0.18), (0.82, 0.35)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('e', 0.16, 0.42, 0.05, 0.035, 'B')]),
    ('mouse', 'animal', 8, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8503999999999999, 0.46, 0.069, 0.035, 'B'), ('e', 0.72, 0.28, 0.045, 0.075, 'A'), ('e', 0.78, 0.28, 0.045, 0.075, 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('a', 0.2, 0.45, 0.1, 180, 340, 0.03, 'A'), ('c', 0.88, 0.44, 0.012, 'D')]),
    ('hamster', 'animal', 9, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8488, 0.46, 0.068, 0.035, 'B'), ('e', 0.72, 0.28, 0.045, 0.075, 'A'), ('e', 0.78, 0.28, 0.045, 0.075, 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A')]),
    ('penguin', 'animal', 12, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('p', [(0.75, 0.38), (0.88, 0.42), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.36, 0.52), (0.55, 0.62), (0.62, 0.52)], 'B'), ('p', [(0.33, 0.52), (0.24, 0.55), (0.33, 0.58)], 'B'), ('l', 0.44, 0.66, 0.44, 0.8200000000000001, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.8200000000000001, 0.015, 'D'), ('r', 0.38, 0.46, 0.58, 0.62, 'W')]),
    ('owl', 'animal', 11, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('l', 0.66, 0.33, 0.64, 0.23, 0.012, 'D'), ('l', 0.6900000000000001, 0.33, 0.68, 0.23, 0.012, 'D'), ('l', 0.72, 0.33, 0.72, 0.23, 0.012, 'D'), ('p', [(0.75, 0.38), (0.9, 0.4), (0.86, 0.5), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.36, 0.52), (0.55, 0.62), (0.62, 0.52)], 'B'), ('p', [(0.33, 0.52), (0.24, 0.55), (0.33, 0.58)], 'B'), ('l', 0.44, 0.66, 0.44, 0.74, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.74, 0.015, 'D'), ('c', 0.65, 0.38, 0.025, 'W'), ('c', 0.72, 0.38, 0.025, 'W')]),
    ('parrot', 'animal', 12, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('p', [(0.75, 0.38), (0.9, 0.4), (0.86, 0.5), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.36, 0.52), (0.55, 0.62), (0.62, 0.52)], 'B'), ('p', [(0.32, 0.48), (0.08, 0.6), (0.32, 0.6)], 'B'), ('l', 0.44, 0.66, 0.44, 0.8, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.8, 0.015, 'D')]),
    ('flamingo', 'animal', 17, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('p', [(0.75, 0.38), (0.9, 0.4), (0.86, 0.5), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.36, 0.52), (0.55, 0.62), (0.62, 0.52)], 'B'), ('p', [(0.33, 0.52), (0.24, 0.55), (0.33, 0.58)], 'B'), ('l', 0.44, 0.66, 0.44, 1.0, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 1.0, 0.015, 'D')]),
    ('eagle', 'animal', 16, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('p', [(0.75, 0.38), (0.9, 0.4), (0.86, 0.5), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.4, 0.5), (0.55, 0.3), (0.62, 0.48)], 'B'), ('p', [(0.32, 0.5), (0.14, 0.56), (0.32, 0.6)], 'B'), ('l', 0.44, 0.66, 0.44, 0.76, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.76, 0.015, 'D')]),
    ('peacock', 'animal', 15, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('l', 0.66, 0.33, 0.64, 0.25, 0.012, 'D'), ('l', 0.6900000000000001, 0.33, 0.68, 0.25, 0.012, 'D'), ('l', 0.72, 0.33, 0.72, 0.25, 0.012, 'D'), ('p', [(0.75, 0.38), (0.88, 0.42), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.36, 0.52), (0.55, 0.62), (0.62, 0.52)], 'B'), ('l', 0.33, 0.54, 0.18871644455048178, 0.45000973464388455, 0.02, 'B'), ('l', 0.33, 0.54, 0.17482459033712733, 0.4921171799344064, 0.02, 'B'), ('l', 0.33, 0.54, 0.17, 0.54, 0.02, 'B'), ('l', 0.33, 0.54, 0.17482459033712733, 0.5878828200655937, 0.02, 'B'), ('l', 0.33, 0.54, 0.18871644455048178, 0.6299902653561156, 0.02, 'B'), ('l', 0.44, 0.66, 0.44, 0.8600000000000001, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.8600000000000001, 0.015, 'D')]),
    ('ostrich', 'animal', 26, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('p', [(0.75, 0.38), (0.88, 0.42), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('e', 0.46, 0.54, 0.12, 0.06, 'B'), ('p', [(0.33, 0.52), (0.24, 0.55), (0.33, 0.58)], 'B'), ('l', 0.44, 0.66, 0.44, 1.0, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 1.0, 0.015, 'D')]),
    ('swan', 'animal', 16, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('e', 0.84, 0.43, 0.08, 0.035, 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('e', 0.46, 0.54, 0.12, 0.06, 'B'), ('p', [(0.33, 0.52), (0.24, 0.55), (0.33, 0.58)], 'B'), ('l', 0.44, 0.66, 0.44, 0.8200000000000001, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.8200000000000001, 0.015, 'D'), ('l', 0.6, 0.36, 0.68, 0.28, 0.03, 'A')]),
    ('goose', 'animal', 14, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('e', 0.84, 0.43, 0.08, 0.035, 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('e', 0.46, 0.54, 0.12, 0.06, 'B'), ('p', [(0.32, 0.5), (0.14, 0.56), (0.32, 0.6)], 'B'), ('l', 0.44, 0.66, 0.44, 0.8600000000000001, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.8600000000000001, 0.015, 'D')]),
    ('seagull', 'animal', 11, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('p', [(0.75, 0.38), (0.88, 0.42), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.4, 0.5), (0.55, 0.3), (0.62, 0.48)], 'B'), ('p', [(0.33, 0.52), (0.24, 0.55), (0.33, 0.58)], 'B'), ('l', 0.44, 0.66, 0.44, 0.8, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.8, 0.015, 'D')]),
    ('whale', 'animal', 35, [('p', [(0.14, 0.52), (0.34, 0.36), (0.66, 0.34), (0.84, 0.46), (0.8, 0.58), (0.56, 0.68), (0.3, 0.66), (0.18, 0.58)], 'A'), ('e', 0.16, 0.46, 0.05, 0.06, 'A'), ('l', 0.4, 0.72, 0.46, 0.84, 0.03, 'A'), ('c', 0.7, 0.44, 0.02, 'Bk')]),
    ('dolphin', 'animal', 22, [('p', [(0.16, 0.48), (0.4, 0.32), (0.68, 0.38), (0.86, 0.52), (0.72, 0.6), (0.44, 0.62), (0.22, 0.58)], 'A'), ('p', [(0.2, 0.42), (0.1, 0.3), (0.26, 0.4)], 'A'), ('c', 0.7, 0.46, 0.018, 'Bk')]),
    ('shark', 'animal', 27, [('p', [(0.12, 0.52), (0.36, 0.4), (0.64, 0.38), (0.86, 0.52), (0.64, 0.64), (0.36, 0.62), (0.12, 0.52)], 'A'), ('p', [(0.4, 0.38), (0.5, 0.22), (0.58, 0.38)], 'A'), ('l', 0.66, 0.44, 0.66, 0.6, 0.015, 'Bk'), ('c', 0.76, 0.47, 0.018, 'Bk')]),
    ('seal', 'animal', 18, [('e', 0.48, 0.54, 0.26, 0.14, 'A'), ('c', 0.72, 0.48, 0.09, 'A'), ('l', 0.66, 0.72, 0.62, 0.82, 0.03, 'A'), ('l', 0.56, 0.74, 0.54, 0.84, 0.03, 'A'), ('c', 0.76, 0.44, 0.015, 'Bk')]),
    ('otter', 'animal', 14, [('e', 0.46, 0.54, 0.24, 0.12, 'A'), ('c', 0.7, 0.48, 0.08, 'A'), ('e', 0.22, 0.5, 0.06, 0.09, 'B'), ('c', 0.74, 0.44, 0.015, 'Bk')]),
    ('walrus', 'animal', 29, [('e', 0.46, 0.54, 0.24, 0.13, 'A'), ('c', 0.72, 0.5, 0.09, 'A'), ('l', 0.8, 0.56, 0.84, 0.72, 0.02, 'W'), ('l', 0.86, 0.56, 0.9, 0.72, 0.02, 'W'), ('c', 0.76, 0.44, 0.015, 'Bk')]),
    ('crab', 'animal', 10, [('e', 0.5, 0.54, 0.2, 0.14, 'A'), ('c', 0.38, 0.42, 0.05, 'A'), ('c', 0.62, 0.42, 0.05, 'A'), ('p', [(0.34, 0.4), (0.26, 0.3), (0.3, 0.44)], 'A'), ('p', [(0.66, 0.4), (0.74, 0.3), (0.7, 0.44)], 'A'), ('l', 0.32, 0.62, 0.2, 0.72, 0.02, 'A'), ('l', 0.68, 0.62, 0.8, 0.72, 0.02, 'A'), ('c', 0.38, 0.4, 0.015, 'Bk'), ('c', 0.62, 0.4, 0.015, 'Bk')]),
    ('lobster', 'animal', 12, [('e', 0.5, 0.52, 0.16, 0.24, 'A'), ('e', 0.36, 0.44, 0.05, 0.09, 'A'), ('e', 0.64, 0.44, 0.05, 0.09, 'A'), ('l', 0.44, 0.76, 0.4, 0.86, 0.02, 'A'), ('l', 0.56, 0.76, 0.6, 0.86, 0.02, 'A'), ('l', 0.46, 0.4, 0.38, 0.28, 0.015, 'A'), ('l', 0.54, 0.4, 0.62, 0.28, 0.015, 'A')]),
    ('octopus', 'animal', 13, [('e', 0.5, 0.38, 0.16, 0.17, 'A'), ('l', 0.4, 0.52, 0.3, 0.72, 0.03, 'A'), ('l', 0.47, 0.54, 0.44, 0.78, 0.03, 'A'), ('l', 0.53, 0.54, 0.56, 0.78, 0.03, 'A'), ('l', 0.6, 0.52, 0.7, 0.72, 0.03, 'A'), ('c', 0.44, 0.36, 0.02, 'Bk'), ('c', 0.56, 0.36, 0.02, 'Bk')]),
    ('jellyfish', 'animal', 12, [('p', [(0.3, 0.46), (0.36, 0.26), (0.64, 0.26), (0.7, 0.46), (0.5, 0.52)], 'A'), ('l', 0.38, 0.5, 0.34, 0.74, 0.018, 'D'), ('l', 0.46, 0.52, 0.46, 0.8, 0.018, 'D'), ('l', 0.54, 0.52, 0.54, 0.8, 0.018, 'D'), ('l', 0.62, 0.5, 0.66, 0.74, 0.018, 'D')]),
    ('seahorse', 'animal', 10, [('e', 0.52, 0.4, 0.1, 0.16, 'A'), ('a', 0.5, 0.62, 0.1, 90, 270, 0.04, 'A'), ('l', 0.44, 0.26, 0.5, 0.3, 0.02, 'A'), ('c', 0.48, 0.32, 0.015, 'Bk')]),
    ('snake', 'animal', 14, [('a', 0.34, 0.62, 0.16, 180, 360, 0.055, 'A'), ('a', 0.58, 0.44, 0.16, 180, 40, 0.055, 'A'), ('e', 0.74, 0.42, 0.07, 0.05, 'A'), ('l', 0.8, 0.42, 0.88, 0.4, 0.012, 'D'), ('c', 0.74, 0.4, 0.015, 'Bk')]),
    ('cobra', 'animal', 16, [('a', 0.4, 0.7, 0.18, 180, 360, 0.06, 'A'), ('l', 0.4, 0.52, 0.48, 0.34, 0.06, 'A'), ('p', [(0.4, 0.34), (0.56, 0.34), (0.48, 0.22)], 'A'), ('c', 0.47, 0.3, 0.015, 'Bk')]),
    ('crocodile', 'animal', 28, [('e', 0.44, 0.56, 0.26, 0.12, 'A'), ('e', 0.74, 0.54, 0.1, 0.05, 'A'), ('l', 0.84, 0.52, 0.94, 0.52, 0.015, 'Bk'), ('l', 0.2, 0.52, 0.1, 0.46, 0.04, 'A'), ('c', 0.78, 0.5, 0.015, 'Bk')]),
    ('alligator', 'animal', 26, [('e', 0.44, 0.56, 0.24, 0.12, 'A'), ('e', 0.72, 0.54, 0.09, 0.05, 'A'), ('e', 0.82, 0.5, 0.06, 0.03, 'B'), ('l', 0.2, 0.52, 0.1, 0.46, 0.04, 'A'), ('c', 0.75, 0.5, 0.015, 'Bk')]),
    ('iguana', 'animal', 14, [('e', 0.44, 0.54, 0.22, 0.1, 'A'), ('c', 0.7, 0.48, 0.07, 'A'), ('l', 0.78, 0.46, 0.88, 0.44, 0.015, 'Bk'), ('l', 0.24, 0.5, 0.12, 0.42, 0.035, 'A'), ('l', 0.4, 0.44, 0.44, 0.38, 0.015, 'B'), ('l', 0.5, 0.44, 0.54, 0.38, 0.015, 'B')]),
    ('lizard', 'animal', 10, [('e', 0.44, 0.52, 0.18, 0.08, 'A'), ('c', 0.68, 0.48, 0.055, 'A'), ('l', 0.74, 0.46, 0.82, 0.42, 0.012, 'D'), ('l', 0.28, 0.48, 0.14, 0.4, 0.03, 'A'), ('l', 0.36, 0.6, 0.32, 0.7, 0.02, 'A'), ('l', 0.52, 0.6, 0.56, 0.7, 0.02, 'A')]),
    ('bat', 'animal', 9, [('p', [(0.22, 0.44), (0.42, 0.56), (0.5, 0.5), (0.58, 0.56), (0.78, 0.44), (0.66, 0.58), (0.58, 0.52), (0.5, 0.58), (0.42, 0.52), (0.34, 0.58)], 'A'), ('e', 0.5, 0.48, 0.08, 0.07, 'A'), ('c', 0.47, 0.44, 0.018, 'W'), ('c', 0.53, 0.44, 0.018, 'W')]),
    ('bee', 'animal', 4, [('e', 0.5, 0.55, 0.1, 0.16, 'A'), ('e', 0.5, 0.34, 0.07, 0.07, 'C'), ('e', 0.4, 0.48, 0.05, 0.14, 'D'), ('e', 0.6, 0.48, 0.05, 0.14, 'D'), ('l', 0.42, 0.48, 0.3, 0.54, 0.012, 'Bk'), ('l', 0.42, 0.57, 0.3, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.42, 0.6599999999999999, 0.3, 0.72, 0.012, 'Bk'), ('l', 0.58, 0.48, 0.7, 0.54, 0.012, 'Bk'), ('l', 0.58, 0.57, 0.7, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.58, 0.6599999999999999, 0.7, 0.72, 0.012, 'Bk'), ('l', 0.47, 0.28, 0.42, 0.18, 0.01, 'Bk'), ('l', 0.53, 0.28, 0.58, 0.18, 0.01, 'Bk'), ('l', 0.43, 0.56, 0.57, 0.56, 0.025, 'Bk'), ('l', 0.42, 0.64, 0.58, 0.64, 0.025, 'Bk')]),
    ('ladybug', 'animal', 4, [('e', 0.5, 0.55, 0.1, 0.16, 'A'), ('e', 0.5, 0.34, 0.07, 0.07, 'C'), ('l', 0.42, 0.48, 0.3, 0.54, 0.012, 'Bk'), ('l', 0.42, 0.57, 0.3, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.42, 0.6599999999999999, 0.3, 0.72, 0.012, 'Bk'), ('l', 0.58, 0.48, 0.7, 0.54, 0.012, 'Bk'), ('l', 0.58, 0.57, 0.7, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.58, 0.6599999999999999, 0.7, 0.72, 0.012, 'Bk'), ('l', 0.47, 0.28, 0.42, 0.18, 0.01, 'Bk'), ('l', 0.53, 0.28, 0.58, 0.18, 0.01, 'Bk'), ('l', 0.5, 0.44, 0.5, 0.72, 0.015, 'Bk'), ('c', 0.44, 0.54, 0.016, 'Bk'), ('c', 0.56, 0.54, 0.016, 'Bk'), ('c', 0.44, 0.64, 0.016, 'Bk'), ('c', 0.56, 0.64, 0.016, 'Bk')]),
    ('dragonfly', 'animal', 5, [('rr', 0.42, 0.4, 0.58, 0.74, 0.06, 'A'), ('c', 0.5, 0.33, 0.06, 'C'), ('e', 0.4, 0.48, 0.05, 0.14, 'D'), ('e', 0.6, 0.48, 0.05, 0.14, 'D'), ('l', 0.42, 0.48, 0.3, 0.54, 0.012, 'Bk'), ('l', 0.42, 0.57, 0.3, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.42, 0.6599999999999999, 0.3, 0.72, 0.012, 'Bk'), ('l', 0.58, 0.48, 0.7, 0.54, 0.012, 'Bk'), ('l', 0.58, 0.57, 0.7, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.58, 0.6599999999999999, 0.7, 0.72, 0.012, 'Bk'), ('l', 0.47, 0.28, 0.42, 0.18, 0.01, 'Bk'), ('l', 0.53, 0.28, 0.58, 0.18, 0.01, 'Bk'), ('c', 0.47, 0.3, 0.02, 'Bk'), ('c', 0.53, 0.3, 0.02, 'Bk')]),
    ('caterpillar', 'animal', 5, [('c', 0.28, 0.55, 0.06, 'A'), ('c', 0.38, 0.55, 0.06, 'A'), ('c', 0.48, 0.55, 0.06, 'A'), ('c', 0.58, 0.55, 0.06, 'A'), ('c', 0.68, 0.55, 0.07, 'B'), ('l', 0.3, 0.48, 0.28, 0.42, 0.01, 'Bk'), ('l', 0.7, 0.48, 0.72, 0.42, 0.01, 'Bk'), ('c', 0.66, 0.52, 0.012, 'Bk')]),
    ('spider', 'animal', 6, [('c', 0.5, 0.52, 0.08, 'A'), ('e', 0.5, 0.38, 0.05, 0.05, 'A'), ('l', 0.44, 0.48, 0.26, 0.36, 0.015, 'Bk'), ('l', 0.44, 0.54, 0.24, 0.54, 0.015, 'Bk'), ('l', 0.44, 0.6, 0.26, 0.7, 0.015, 'Bk'), ('l', 0.56, 0.48, 0.74, 0.36, 0.015, 'Bk'), ('l', 0.56, 0.54, 0.76, 0.54, 0.015, 'Bk'), ('l', 0.56, 0.6, 0.74, 0.7, 0.015, 'Bk')]),
    ('scorpion', 'animal', 7, [('e', 0.5, 0.6, 0.1, 0.13, 'A'), ('a', 0.44, 0.42, 0.1, 90, 250, 0.035, 'A'), ('l', 0.38, 0.36, 0.44, 0.26, 0.03, 'A'), ('p', [(0.42, 0.26), (0.38, 0.16), (0.46, 0.22)], 'A'), ('l', 0.46, 0.68, 0.4, 0.78, 0.015, 'Bk'), ('l', 0.54, 0.68, 0.6, 0.78, 0.015, 'Bk'), ('l', 0.48, 0.5, 0.36, 0.44, 0.012, 'Bk')]),
    ('burger', 'food', 9, [('rr', 0.24, 0.4, 0.76, 0.52, 0.06, 'A'), ('r', 0.26, 0.52, 0.74, 0.6, 'B'), ('r', 0.26, 0.6, 0.74, 0.68, 'C'), ('rr', 0.24, 0.68, 0.76, 0.76, 0.04, 'A'), ('e', 0.34, 0.46, 0.015, 0.02, 'D'), ('e', 0.44, 0.44, 0.015, 0.02, 'D'), ('e', 0.54, 0.46, 0.015, 0.02, 'D'), ('e', 0.64, 0.44, 0.015, 0.02, 'D')]),
    ('sandwich', 'food', 9, [('p', [(0.2, 0.56), (0.5, 0.28), (0.8, 0.56)], 'A'), ('r', 0.26, 0.56, 0.74, 0.64, 'B'), ('r', 0.28, 0.64, 0.72, 0.72, 'C')]),
    ('hotdog', 'food', 9, [('e', 0.5, 0.56, 0.28, 0.11, 'A'), ('e', 0.5, 0.5, 0.24, 0.09, 'B'), ('l', 0.32, 0.48, 0.68, 0.52, 0.02, 'D')]),
    ('french_fries', 'food', 8, [('p', [(0.34, 0.44), (0.66, 0.44), (0.6, 0.8), (0.4, 0.8)], 'A'), ('r', 0.42, 0.22, 0.47, 0.5, 'B'), ('r', 0.5, 0.18, 0.55, 0.48, 'B'), ('r', 0.58, 0.24, 0.63, 0.5, 'B')]),
    ('donut', 'food', 8, [('a', 0.5, 0.52, 0.24, 0, 360, 0.14, 'A'), ('a', 0.5, 0.52, 0.24, 0, 360, 0.14, 'B'), ('c', 0.4, 0.46, 0.012, 'D'), ('c', 0.52, 0.42, 0.012, 'C'), ('c', 0.6, 0.48, 0.012, 'A'), ('c', 0.46, 0.58, 0.012, 'W')]),
    ('cake', 'food', 10, [('rr', 0.24, 0.48, 0.76, 0.78, 0.03, 'A'), ('rr', 0.3, 0.36, 0.7, 0.5, 0.03, 'B'), ('l', 0.5, 0.3, 0.5, 0.38, 0.02, 'D'), ('c', 0.5, 0.28, 0.02, 'A')]),
    ('cookie', 'food', 8, [('c', 0.5, 0.54, 0.24, 'A'), ('c', 0.42, 0.48, 0.022, 'Bk'), ('c', 0.54, 0.5, 0.022, 'Bk'), ('c', 0.46, 0.6, 0.022, 'Bk'), ('c', 0.58, 0.6, 0.022, 'Bk')]),
    ('ice_cream', 'food', 8, [('p', [(0.4, 0.44), (0.6, 0.44), (0.5, 0.84)], 'A'), ('c', 0.5, 0.4, 0.11, 'B'), ('c', 0.46, 0.3, 0.09, 'A'), ('c', 0.52, 0.22, 0.05, 'D')]),
    ('cheese', 'food', 8, [('p', [(0.2, 0.6), (0.8, 0.44), (0.8, 0.64), (0.2, 0.76)], 'A'), ('c', 0.42, 0.6, 0.02, 'D'), ('c', 0.62, 0.54, 0.025, 'D')]),
    ('bread', 'food', 8, [('e', 0.5, 0.54, 0.28, 0.16, 'A'), ('a', 0.38, 0.46, 0.05, 120, 300, 0.018, 'C'), ('a', 0.5, 0.42, 0.05, 120, 300, 0.018, 'C'), ('a', 0.62, 0.46, 0.05, 120, 300, 0.018, 'C')]),
    ('egg', 'food', 6, [('e', 0.5, 0.54, 0.16, 0.2, 'A')]),
    ('carrot', 'food', 6, [('p', [(0.56, 0.24), (0.64, 0.3), (0.48, 0.78), (0.38, 0.7)], 'A'), ('l', 0.6, 0.22, 0.68, 0.14, 0.02, 'B'), ('l', 0.64, 0.26, 0.74, 0.22, 0.02, 'B'), ('l', 0.5, 0.48, 0.58, 0.5, 0.012, 'C')]),
    ('potato', 'food', 7, [('e', 0.5, 0.54, 0.24, 0.17, 'A'), ('c', 0.42, 0.5, 0.014, 'C'), ('c', 0.56, 0.58, 0.014, 'C')]),
    ('tomato', 'food', 7, [('c', 0.5, 0.56, 0.24, 'A'), ('p', [(0.4, 0.34), (0.45, 0.24), (0.5, 0.33), (0.55, 0.24), (0.6, 0.34)], 'C'), ('a', 0.42, 0.46, 0.08, 200, 300, 0.02, 'W'), ('e', 0.58, 0.34, 0.05, 0.02, 'B')]),
    ('onion', 'food', 7, [('c', 0.5, 0.56, 0.2, 'A'), ('l', 0.5, 0.36, 0.52, 0.24, 0.015, 'B'), ('a', 0.44, 0.56, 0.12, 200, 340, 0.015, 'C'), ('a', 0.56, 0.56, 0.12, 200, 340, 0.015, 'C')]),
    ('garlic', 'food', 6, [('p', [(0.5, 0.24), (0.66, 0.44), (0.6, 0.72), (0.4, 0.72), (0.34, 0.44)], 'A'), ('l', 0.5, 0.3, 0.5, 0.7, 0.015, 'C')]),
    ('lemon', 'food', 6, [('c', 0.5, 0.56, 0.24, 'A'), ('l', 0.5, 0.33, 0.53, 0.24, 0.02, 'C'), ('a', 0.42, 0.46, 0.08, 200, 300, 0.02, 'W')]),
    ('orange', 'food', 7, [('c', 0.5, 0.56, 0.24, 'A'), ('l', 0.5, 0.33, 0.53, 0.24, 0.02, 'C'), ('e', 0.58, 0.3, 0.07, 0.035, 'B'), ('a', 0.42, 0.46, 0.08, 200, 300, 0.02, 'W')]),
    ('grape', 'food', 6, [('l', 0.5, 0.86, 0.5, 0.7, 0.025, 'B'), ('c', 0.36, 0.5, 0.07, 'A'), ('c', 0.5, 0.44, 0.08, 'A'), ('c', 0.64, 0.5, 0.07, 'A'), ('c', 0.43, 0.6, 0.07, 'A'), ('c', 0.57, 0.6, 0.07, 'A')]),
    ('peach', 'food', 7, [('c', 0.5, 0.56, 0.24, 'A'), ('l', 0.5, 0.33, 0.53, 0.24, 0.02, 'C'), ('e', 0.58, 0.3, 0.07, 0.035, 'B'), ('a', 0.42, 0.46, 0.08, 200, 300, 0.02, 'W'), ('l', 0.5, 0.34, 0.5, 0.8, 0.012, 'C')]),
    ('pear', 'food', 7, [('e', 0.5, 0.6, 0.18, 0.22, 'A'), ('e', 0.5, 0.34, 0.1, 0.12, 'A'), ('l', 0.5, 0.22, 0.53, 0.14, 0.015, 'C')]),
    ('cherry', 'food', 5, [('c', 0.4, 0.6, 0.1, 'A'), ('c', 0.6, 0.56, 0.1, 'A'), ('l', 0.42, 0.5, 0.5, 0.28, 0.015, 'C'), ('l', 0.58, 0.46, 0.5, 0.28, 0.015, 'C'), ('e', 0.52, 0.26, 0.04, 0.02, 'B')]),
    ('watermelon', 'food', 14, [('c', 0.5, 0.56, 0.26, 'B'), ('c', 0.5, 0.56, 0.2, 'A'), ('l', 0.38, 0.44, 0.42, 0.68, 0.025, 'B'), ('l', 0.5, 0.4, 0.5, 0.76, 0.025, 'B'), ('l', 0.62, 0.44, 0.58, 0.68, 0.025, 'B')]),
    ('corn', 'food', 7, [('e', 0.5, 0.54, 0.1, 0.24, 'A'), ('p', [(0.42, 0.34), (0.4, 0.18), (0.5, 0.32)], 'B'), ('p', [(0.58, 0.34), (0.6, 0.18), (0.5, 0.32)], 'B'), ('l', 0.46, 0.36, 0.46, 0.76, 0.012, 'C'), ('l', 0.54, 0.36, 0.54, 0.76, 0.012, 'C')]),
    ('rice', 'food', 7, [('p', [(0.3, 0.44), (0.7, 0.44), (0.62, 0.78), (0.38, 0.78)], 'A'), ('c', 0.44, 0.56, 0.015, 'W'), ('c', 0.52, 0.6, 0.015, 'W'), ('c', 0.58, 0.54, 0.015, 'W'), ('c', 0.48, 0.66, 0.015, 'W')]),
    ('pasta', 'food', 8, [('c', 0.5, 0.54, 0.22, 'A'), ('a', 0.5, 0.54, 0.16, 40, 200, 0.015, 'C'), ('a', 0.5, 0.54, 0.1, 220, 340, 0.015, 'C')]),
    ('soup', 'food', 8, [('p', [(0.26, 0.48), (0.74, 0.48), (0.64, 0.8), (0.36, 0.8)], 'A'), ('e', 0.5, 0.48, 0.24, 0.05, 'B'), ('a', 0.44, 0.3, 0.06, 20, 160, 0.015, 'D'), ('a', 0.56, 0.3, 0.06, 20, 160, 0.015, 'D')]),
    ('coffee', 'food', 7, [('rr', 0.36, 0.38, 0.6, 0.78, 0.03, 'A'), ('a', 0.6, 0.58, 0.08, -90, 90, 0.03, 'A'), ('a', 0.46, 0.3, 0.05, 0, 180, 0.015, 'D'), ('a', 0.54, 0.28, 0.05, 0, 180, 0.015, 'D')]),
    ('tea', 'food', 7, [('rr', 0.34, 0.4, 0.62, 0.78, 0.04, 'A'), ('a', 0.62, 0.59, 0.08, -90, 90, 0.03, 'A'), ('e', 0.44, 0.34, 0.05, 0.03, 'B'), ('e', 0.54, 0.3, 0.05, 0.03, 'B')]),
    ('juice', 'food', 7, [('rr', 0.42, 0.34, 0.58, 0.82, 0.04, 'A'), ('r', 0.46, 0.2, 0.54, 0.34, 'A'), ('r', 0.45, 0.16, 0.55, 0.22, 'D'), ('r', 0.42, 0.5, 0.58, 0.68, 'D')]),
    ('chocolate', 'food', 7, [('rr', 0.26, 0.4, 0.74, 0.72, 0.02, 'A'), ('l', 0.42, 0.4, 0.42, 0.72, 0.015, 'C'), ('l', 0.58, 0.4, 0.58, 0.72, 0.015, 'C'), ('l', 0.26, 0.56, 0.74, 0.56, 0.015, 'C')]),
    ('honey', 'food', 7, [('rr', 0.36, 0.34, 0.64, 0.8, 0.05, 'A'), ('r', 0.38, 0.24, 0.62, 0.34, 'Bk')]),
    ('popcorn', 'food', 8, [('p', [(0.32, 0.46), (0.68, 0.46), (0.6, 0.8), (0.4, 0.8)], 'A'), ('c', 0.42, 0.42, 0.05, 'W'), ('c', 0.52, 0.38, 0.05, 'W'), ('c', 0.6, 0.44, 0.05, 'W'), ('c', 0.48, 0.32, 0.04, 'W'), ('c', 0.56, 0.3, 0.04, 'W')]),
    ('pretzel', 'food', 8, [('a', 0.5, 0.54, 0.22, 20, 140, 0.07, 'A'), ('a', 0.42, 0.48, 0.12, 90, 300, 0.07, 'A'), ('a', 0.58, 0.48, 0.12, 60, 270, 0.07, 'A')]),
    ('yogurt', 'food', 6, [('rr', 0.36, 0.34, 0.64, 0.82, 0.05, 'A'), ('r', 0.38, 0.24, 0.62, 0.34, 'D'), ('l', 0.38, 0.3, 0.62, 0.3, 0.015, 'C')]),
    ('butter', 'food', 6, [('rr', 0.28, 0.44, 0.72, 0.68, 0.02, 'A'), ('rr', 0.34, 0.34, 0.66, 0.52, 0.02, 'B')]),
    ('milk', 'food', 8, [('rr', 0.42, 0.34, 0.58, 0.82, 0.04, 'A'), ('r', 0.46, 0.2, 0.54, 0.34, 'A'), ('r', 0.45, 0.16, 0.55, 0.22, 'D'), ('r', 0.42, 0.5, 0.58, 0.68, 'D'), ('r', 0.42, 0.48, 0.58, 0.66, 'W')]),
    ('candy', 'food', 5, [('c', 0.5, 0.54, 0.1, 'A'), ('p', [(0.4, 0.48), (0.32, 0.42), (0.36, 0.54)], 'A'), ('p', [(0.6, 0.48), (0.68, 0.42), (0.64, 0.54)], 'A')]),
    ('pie', 'food', 10, [('e', 0.5, 0.54, 0.28, 0.11, 'A'), ('a', 0.5, 0.48, 0.24, 180, 360, 0.05, 'B'), ('c', 0.5, 0.4, 0.03, 'C')]),
    ('bagel', 'food', 7, [('a', 0.5, 0.54, 0.22, 0, 360, 0.12, 'A'), ('l', 0.36, 0.5, 0.64, 0.5, 0.02, 'W')]),
    ('waffle', 'food', 9, [('r', 0.28, 0.34, 0.72, 0.78, 'A'), ('l', 0.42, 0.34, 0.42, 0.78, 0.015, 'C'), ('l', 0.58, 0.34, 0.58, 0.78, 0.015, 'C'), ('l', 0.28, 0.48, 0.72, 0.48, 0.015, 'C'), ('l', 0.28, 0.64, 0.72, 0.64, 0.015, 'C')]),
    ('pancake', 'food', 8, [('e', 0.5, 0.6, 0.24, 0.08, 'A'), ('e', 0.5, 0.5, 0.2, 0.07, 'B'), ('e', 0.5, 0.4, 0.14, 0.06, 'A'), ('r', 0.44, 0.3, 0.56, 0.36, 'C')]),
    ('sushi', 'food', 7, [('rr', 0.28, 0.5, 0.72, 0.68, 0.03, 'A'), ('e', 0.5, 0.46, 0.16, 0.09, 'B'), ('r', 0.42, 0.44, 0.58, 0.5, 'C')]),
    ('dumpling', 'food', 7, [('e', 0.5, 0.56, 0.22, 0.13, 'A'), ('a', 0.5, 0.46, 0.14, 180, 360, 0.02, 'C'), ('l', 0.42, 0.47, 0.58, 0.47, 0.015, 'C')]),
    ('taco', 'food', 9, [('a', 0.5, 0.6, 0.24, 180, 360, 0.14, 'A'), ('a', 0.5, 0.6, 0.24, 200, 340, 0.14, 'B'), ('c', 0.44, 0.42, 0.02, 'C'), ('c', 0.54, 0.4, 0.02, 'C')]),
    ('burrito', 'food', 9, [('e', 0.5, 0.54, 0.28, 0.13, 'A'), ('l', 0.28, 0.54, 0.72, 0.54, 0.02, 'C')]),
    ('salad', 'food', 8, [('p', [(0.26, 0.48), (0.74, 0.48), (0.62, 0.78), (0.38, 0.78)], 'A'), ('e', 0.42, 0.44, 0.07, 0.04, 'B'), ('e', 0.56, 0.42, 0.07, 0.04, 'B'), ('c', 0.62, 0.48, 0.03, 'C')]),
    ('lettuce', 'food', 7, [('e', 0.4, 0.56, 0.12, 0.14, 'A'), ('e', 0.58, 0.56, 0.12, 0.14, 'A'), ('e', 0.5, 0.48, 0.12, 0.16, 'B')]),
    ('broccoli', 'food', 7, [('r', 0.44, 0.56, 0.56, 0.82, 'B'), ('c', 0.38, 0.48, 0.08, 'A'), ('c', 0.5, 0.42, 0.09, 'A'), ('c', 0.62, 0.48, 0.08, 'A')]),
    ('cabbage', 'food', 8, [('c', 0.5, 0.54, 0.22, 'A'), ('a', 0.5, 0.54, 0.16, 100, 260, 0.02, 'C'), ('a', 0.5, 0.54, 0.1, 100, 260, 0.02, 'C')]),
    ('mushroom', 'food', 7, [('r', 0.44, 0.5, 0.56, 0.82, 'A'), ('p', [(0.26, 0.5), (0.5, 0.22), (0.74, 0.5)], 'D'), ('c', 0.42, 0.4, 0.03, 'W'), ('c', 0.58, 0.38, 0.025, 'W')]),
    ('pumpkin', 'food', 12, [('c', 0.5, 0.56, 0.24, 'A'), ('l', 0.38, 0.36, 0.38, 0.76, 0.02, 'C'), ('l', 0.5, 0.32, 0.5, 0.8, 0.02, 'C'), ('l', 0.62, 0.36, 0.62, 0.76, 0.02, 'C'), ('r', 0.48, 0.26, 0.52, 0.34, 'B')]),
    ('pineapple', 'food', 10, [('e', 0.5, 0.58, 0.16, 0.22, 'A'), ('p', [(0.4, 0.36), (0.44, 0.2), (0.5, 0.34)], 'B'), ('p', [(0.56, 0.36), (0.6, 0.2), (0.5, 0.34)], 'B'), ('p', [(0.48, 0.34), (0.5, 0.16), (0.52, 0.34)], 'B'), ('l', 0.4, 0.5, 0.6, 0.66, 0.012, 'C'), ('l', 0.4, 0.66, 0.6, 0.5, 0.012, 'C')]),
    ('coconut', 'food', 8, [('c', 0.5, 0.56, 0.22, 'C'), ('c', 0.44, 0.48, 0.015, 'Bk'), ('c', 0.56, 0.48, 0.015, 'Bk')]),
    ('nuts', 'food', 5, [('e', 0.4, 0.56, 0.08, 0.1, 'A'), ('e', 0.6, 0.52, 0.08, 0.1, 'B'), ('e', 0.5, 0.68, 0.07, 0.09, 'A')]),
    ('jam', 'food', 6, [('rr', 0.36, 0.34, 0.64, 0.82, 0.05, 'A'), ('r', 0.38, 0.24, 0.62, 0.34, 'D'), ('l', 0.38, 0.3, 0.62, 0.3, 0.015, 'C'), ('r', 0.4, 0.44, 0.6, 0.62, 'A')]),
    ('van', 'vehicle', 26, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('l', 0.46, 0.3, 0.46, 0.44, 0.02, 'C'), ('e', 0.24, 0.36, 0.05, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D')]),
    ('jeep', 'vehicle', 24, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('r', 0.3, 0.2, 0.34, 0.3, 'C')]),
    ('taxi', 'vehicle', 23, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('r', 0.44, 0.16, 0.56, 0.24, 'D')]),
    ('ambulance', 'vehicle', 27, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.55, 0.28, 0.8, 0.46, 0.04, 'A'), ('e', 0.62, 0.36, 0.045, 0.05, 'D'), ('e', 0.74, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('c', 0.28, 0.36, 0.04, 'W'), ('l', 0.28, 0.32, 0.28, 0.4, 0.02, 'A'), ('l', 0.24, 0.36, 0.32, 0.36, 0.02, 'A')]),
    ('police_car', 'vehicle', 24, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('rr', 0.42, 0.18, 0.58, 0.26, 0.02, 'Bk'), ('c', 0.47, 0.22, 0.02, 'A'), ('c', 0.53, 0.22, 0.02, 'B')]),
    ('fire_engine', 'vehicle', 28, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.62, 0.26, 0.86, 0.46, 0.03, 'B'), ('e', 0.7, 0.36, 0.04, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D')]),
    ('tractor', 'vehicle', 26, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('c', 0.3, 0.68, 0.12, 'Bk'), ('c', 0.3, 0.68, 0.05, 'D')]),
    ('forklift', 'vehicle', 24, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.55, 0.28, 0.8, 0.46, 0.04, 'A'), ('e', 0.62, 0.36, 0.045, 0.05, 'D'), ('e', 0.74, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.1, 0.24, 0.1, 0.72, 0.03, 'C'), ('l', 0.1, 0.7, 0.22, 0.7, 0.03, 'C')]),
    ('excavator', 'vehicle', 26, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.55, 0.28, 0.8, 0.46, 0.04, 'A'), ('e', 0.62, 0.36, 0.045, 0.05, 'D'), ('e', 0.74, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.2, 0.44, 0.08, 0.3, 0.04, 'A'), ('p', [(0.04, 0.28), (0.12, 0.26), (0.08, 0.38)], 'A')]),
    ('bulldozer', 'vehicle', 25, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.55, 0.28, 0.8, 0.46, 0.04, 'A'), ('e', 0.62, 0.36, 0.045, 0.05, 'D'), ('e', 0.74, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('r', 0.08, 0.52, 0.2, 0.64, 'A')]),
    ('crane', 'vehicle', 30, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.55, 0.28, 0.8, 0.46, 0.04, 'A'), ('e', 0.62, 0.36, 0.045, 0.05, 'D'), ('e', 0.74, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.24, 0.42, 0.06, 0.1, 0.035, 'A'), ('l', 0.06, 0.12, 0.3, 0.3, 0.025, 'A'), ('l', 0.1, 0.2, 0.1, 0.34, 0.012, 'Bk')]),
    ('garbage_truck', 'vehicle', 27, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.56, 0.3, 0.82, 0.3, 0.02, 'C'), ('l', 0.56, 0.38, 0.82, 0.38, 0.02, 'C'), ('l', 0.56, 0.46, 0.82, 0.46, 0.02, 'C')]),
    ('trailer', 'vehicle', 28, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('l', 0.46, 0.3, 0.46, 0.44, 0.02, 'C'), ('e', 0.24, 0.36, 0.05, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.06, 0.66, 0.14, 0.66, 0.03, 'C')]),
    ('ferry', 'vehicle', 29, [('p', [(0.14, 0.54), (0.86, 0.54), (0.74, 0.72), (0.26, 0.72)], 'A'), ('rr', 0.28, 0.4, 0.72, 0.54, 0.02, 'D'), ('r', 0.5, 0.26, 0.54, 0.4, 'C'), ('e', 0.34, 0.46, 0.02, 0.025, 'W'), ('e', 0.44, 0.46, 0.02, 0.025, 'W'), ('e', 0.54, 0.46, 0.02, 0.025, 'W'), ('e', 0.64, 0.46, 0.02, 0.025, 'W')]),
    ('yacht', 'vehicle', 25, [('p', [(0.18, 0.6), (0.82, 0.6), (0.68, 0.74), (0.3, 0.74)], 'A'), ('l', 0.5, 0.6, 0.5, 0.18, 0.02, 'C'), ('p', [(0.5, 0.2), (0.72, 0.56), (0.5, 0.56)], 'D'), ('p', [(0.5, 0.24), (0.34, 0.56), (0.5, 0.56)], 'W')]),
    ('kayak', 'vehicle', 12, [('e', 0.5, 0.54, 0.34, 0.06, 'A'), ('e', 0.5, 0.54, 0.1, 0.035, 'D')]),
    ('canoe', 'vehicle', 12, [('e', 0.5, 0.54, 0.32, 0.07, 'A'), ('l', 0.24, 0.54, 0.76, 0.54, 0.015, 'C')]),
    ('sailboat', 'vehicle', 20, [('p', [(0.24, 0.62), (0.76, 0.62), (0.64, 0.74), (0.34, 0.74)], 'A'), ('l', 0.5, 0.62, 0.5, 0.16, 0.02, 'C'), ('p', [(0.5, 0.18), (0.7, 0.58), (0.5, 0.58)], 'D'), ('p', [(0.5, 0.22), (0.32, 0.58), (0.5, 0.58)], 'W')]),
    ('hot_air_balloon', 'vehicle', 24, [('c', 0.5, 0.4, 0.2, 'A'), ('l', 0.38, 0.26, 0.44, 0.56, 0.015, 'C'), ('l', 0.5, 0.2, 0.5, 0.6, 0.015, 'C'), ('l', 0.62, 0.26, 0.56, 0.56, 0.015, 'C'), ('l', 0.42, 0.6, 0.46, 0.68, 0.012, 'C'), ('l', 0.58, 0.6, 0.54, 0.68, 0.012, 'C'), ('r', 0.44, 0.68, 0.56, 0.78, 'B')]),
    ('helicopter', 'vehicle', 22, [('e', 0.44, 0.54, 0.22, 0.12, 'A'), ('rr', 0.58, 0.44, 0.74, 0.58, 0.05, 'A'), ('e', 0.68, 0.5, 0.03, 0.035, 'D'), ('l', 0.24, 0.4, 0.66, 0.4, 0.025, 'C'), ('l', 0.45, 0.4, 0.45, 0.44, 0.03, 'C'), ('l', 0.3, 0.68, 0.6, 0.68, 0.02, 'C')]),
    ('rocket', 'vehicle', 16, [('p', [(0.5, 0.14), (0.58, 0.34), (0.58, 0.62), (0.5, 0.74), (0.42, 0.62), (0.42, 0.34)], 'A'), ('c', 0.5, 0.42, 0.04, 'D'), ('p', [(0.42, 0.62), (0.34, 0.76), (0.42, 0.7)], 'B'), ('p', [(0.58, 0.62), (0.66, 0.76), (0.58, 0.7)], 'B'), ('p', [(0.46, 0.74), (0.5, 0.88), (0.54, 0.74)], 'D')]),
    ('submarine', 'vehicle', 27, [('e', 0.5, 0.54, 0.32, 0.14, 'A'), ('e', 0.22, 0.48, 0.05, 0.08, 'A'), ('r', 0.46, 0.36, 0.54, 0.44, 'C'), ('c', 0.66, 0.5, 0.025, 'D'), ('c', 0.58, 0.5, 0.025, 'D')]),
    ('tram', 'vehicle', 27, [('rr', 0.1, 0.34, 0.9, 0.64, 0.03, 'A'), ('l', 0.16, 0.2, 0.84, 0.2, 0.02, 'C'), ('l', 0.5, 0.2, 0.5, 0.34, 0.02, 'C'), ('e', 0.24, 0.48, 0.05, 0.07, 'D'), ('e', 0.44, 0.48, 0.05, 0.07, 'D'), ('e', 0.64, 0.48, 0.05, 0.07, 'D'), ('c', 0.28, 0.68, 0.06, 'Bk'), ('c', 0.72, 0.68, 0.06, 'Bk')]),
    ('rickshaw', 'vehicle', 16, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.1, 0.3, 0.1, 0.66, 0.025, 'C'), ('l', 0.1, 0.3, 0.24, 0.3, 0.025, 'C'), ('p', [(0.06, 0.3), (0.14, 0.22), (0.14, 0.3)], 'D')]),
    ('cart', 'vehicle', 14, [('r', 0.24, 0.4, 0.7, 0.62, 'A'), ('l', 0.24, 0.4, 0.1, 0.3, 0.03, 'C'), ('c', 0.36, 0.7, 0.07, 'Bk'), ('c', 0.58, 0.7, 0.07, 'Bk')]),
    ('sled', 'vehicle', 13, [('rr', 0.28, 0.44, 0.72, 0.6, 0.04, 'A'), ('a', 0.28, 0.6, 0.1, 0, 180, 0.025, 'C'), ('a', 0.72, 0.6, 0.1, 0, 180, 0.025, 'C'), ('l', 0.22, 0.7, 0.78, 0.7, 0.025, 'C')]),
    ('snowmobile', 'vehicle', 19, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.14, 0.74, 0.86, 0.74, 0.04, 'Bk'), ('l', 0.14, 0.68, 0.14, 0.78, 0.04, 'Bk'), ('l', 0.86, 0.68, 0.86, 0.78, 0.04, 'Bk')]),
    ('skateboard', 'vehicle', 11, [('rr', 0.18, 0.5, 0.82, 0.58, 0.03, 'A'), ('c', 0.32, 0.64, 0.05, 'Bk'), ('c', 0.68, 0.64, 0.05, 'Bk')]),
    ('surfboard', 'vehicle', 13, [('e', 0.5, 0.54, 0.1, 0.3, 'A'), ('l', 0.5, 0.3, 0.5, 0.78, 0.015, 'C'), ('l', 0.5, 0.56, 0.5, 0.62, 0.03, 'D')]),
    ('scooter', 'vehicle', 14, [('r', 0.2, 0.48, 0.42, 0.54, 'A'), ('l', 0.42, 0.3, 0.42, 0.52, 0.03, 'C'), ('l', 0.36, 0.3, 0.48, 0.3, 0.03, 'C'), ('a', 0.24, 0.56, 0.06, 0, 360, 0.03, 'Bk'), ('a', 0.72, 0.56, 0.08, 0, 360, 0.03, 'Bk'), ('l', 0.3, 0.56, 0.66, 0.56, 0.03, 'C')]),
    ('stroller', 'vehicle', 15, [('p', [(0.34, 0.34), (0.66, 0.34), (0.6, 0.56), (0.36, 0.56)], 'A'), ('l', 0.64, 0.34, 0.78, 0.28, 0.03, 'C'), ('a', 0.38, 0.64, 0.07, 0, 360, 0.03, 'Bk'), ('a', 0.62, 0.64, 0.07, 0, 360, 0.03, 'Bk'), ('l', 0.4, 0.56, 0.38, 0.58, 0.03, 'C'), ('l', 0.6, 0.56, 0.62, 0.58, 0.03, 'C')]),
    ('wheelbarrow', 'vehicle', 14, [('p', [(0.3, 0.44), (0.66, 0.44), (0.6, 0.62), (0.36, 0.62)], 'A'), ('l', 0.62, 0.52, 0.84, 0.6, 0.03, 'C'), ('l', 0.62, 0.58, 0.84, 0.52, 0.03, 'C'), ('a', 0.3, 0.66, 0.06, 0, 360, 0.03, 'Bk')]),
    ('tricycle', 'vehicle', 14, [('p', [(0.44, 0.34), (0.6, 0.34), (0.64, 0.5), (0.48, 0.5)], 'A'), ('a', 0.34, 0.58, 0.09, 0, 360, 0.03, 'Bk'), ('a', 0.56, 0.58, 0.09, 0, 360, 0.03, 'Bk'), ('a', 0.72, 0.58, 0.09, 0, 360, 0.03, 'Bk'), ('l', 0.48, 0.5, 0.34, 0.58, 0.025, 'C'), ('l', 0.56, 0.5, 0.72, 0.58, 0.025, 'C'), ('l', 0.52, 0.34, 0.5, 0.28, 0.025, 'C')]),
    ('go_kart', 'vehicle', 18, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.44, 0.24, 0.44, 0.34, 0.03, 'C'), ('l', 0.38, 0.24, 0.5, 0.24, 0.03, 'C')]),
    ('fork', 'tool', 6, [('r', 0.4, 0.18, 0.44, 0.42, 'A'), ('r', 0.48, 0.18, 0.52, 0.42, 'A'), ('r', 0.56, 0.18, 0.6, 0.42, 'A'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'A'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'A')]),
    ('spoon', 'tool', 6, [('e', 0.5, 0.28, 0.07, 0.1, 'A'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'A'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'A')]),
    ('knife', 'tool', 6, [('p', [(0.44, 0.42), (0.44, 0.16), (0.56, 0.16), (0.56, 0.42)], 'D'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'A'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'A')]),
    ('scissors', 'tool', 6, [('l', 0.34, 0.26, 0.66, 0.6, 0.035, 'A'), ('l', 0.66, 0.26, 0.34, 0.6, 0.035, 'A'), ('c', 0.36, 0.66, 0.06, 'B'), ('c', 0.64, 0.66, 0.06, 'B'), ('c', 0.5, 0.43, 0.02, 'Bk')]),
    ('glue', 'tool', 5, [('rr', 0.42, 0.34, 0.58, 0.82, 0.04, 'A'), ('r', 0.46, 0.2, 0.54, 0.34, 'A'), ('r', 0.45, 0.16, 0.55, 0.22, 'D'), ('r', 0.42, 0.5, 0.58, 0.68, 'D'), ('p', [(0.46, 0.16), (0.54, 0.16), (0.5, 0.08)], 'A')]),
    ('tape', 'tool', 5, [('a', 0.5, 0.5, 0.22, 0, 360, 0.09, 'A'), ('l', 0.72, 0.5, 0.84, 0.44, 0.09, 'A')]),
    ('ruler', 'tool', 8, [('rr', 0.14, 0.44, 0.86, 0.58, 0.02, 'A'), ('l', 0.24, 0.44, 0.24, 0.5, 0.012, 'Bk'), ('l', 0.34, 0.44, 0.34, 0.52, 0.012, 'Bk'), ('l', 0.44, 0.44, 0.44, 0.5, 0.012, 'Bk'), ('l', 0.54, 0.44, 0.54, 0.52, 0.012, 'Bk'), ('l', 0.64, 0.44, 0.64, 0.5, 0.012, 'Bk'), ('l', 0.74, 0.44, 0.74, 0.52, 0.012, 'Bk')]),
    ('pencil', 'tool', 7, [('l', 0.2, 0.7, 0.74, 0.3, 0.05, 'A'), ('p', [(0.74, 0.3), (0.84, 0.22), (0.8, 0.38)], 'B'), ('p', [(0.8, 0.34), (0.86, 0.28), (0.84, 0.34)], 'Bk'), ('r', 0.14, 0.66, 0.24, 0.74, 'D')]),
    ('pen', 'tool', 6, [('l', 0.24, 0.68, 0.72, 0.32, 0.045, 'A'), ('p', [(0.72, 0.32), (0.82, 0.24), (0.78, 0.38)], 'C'), ('l', 0.3, 0.62, 0.42, 0.52, 0.012, 'W')]),
    ('eraser', 'tool', 4, [('rr', 0.3, 0.42, 0.64, 0.6, 0.03, 'A'), ('rr', 0.3, 0.5, 0.44, 0.6, 0.03, 'B')]),
    ('stapler', 'tool', 6, [('rr', 0.22, 0.54, 0.78, 0.68, 0.04, 'A'), ('p', [(0.24, 0.54), (0.7, 0.42), (0.74, 0.5), (0.28, 0.58)], 'C')]),
    ('calculator', 'tool', 7, [('rr', 0.32, 0.26, 0.68, 0.8, 0.03, 'A'), ('r', 0.38, 0.32, 0.62, 0.44, 'D'), ('c', 0.42, 0.54, 0.02, 'Bk'), ('c', 0.5, 0.54, 0.02, 'Bk'), ('c', 0.58, 0.54, 0.02, 'Bk'), ('c', 0.42, 0.64, 0.02, 'Bk'), ('c', 0.5, 0.64, 0.02, 'Bk'), ('c', 0.58, 0.64, 0.02, 'Bk'), ('c', 0.42, 0.74, 0.02, 'D'), ('c', 0.5, 0.74, 0.02, 'D'), ('c', 0.58, 0.74, 0.02, 'A')]),
    ('light_bulb', 'tool', 5, [('c', 0.5, 0.44, 0.14, 'D'), ('r', 0.44, 0.56, 0.56, 0.72, 'B'), ('l', 0.44, 0.62, 0.56, 0.62, 0.012, 'C'), ('l', 0.44, 0.68, 0.56, 0.68, 0.012, 'C'), ('a', 0.46, 0.4, 0.05, 200, 300, 0.012, 'W')]),
    ('candle', 'tool', 6, [('rr', 0.44, 0.4, 0.56, 0.78, 0.02, 'A'), ('l', 0.5, 0.34, 0.5, 0.4, 0.012, 'C'), ('e', 0.5, 0.28, 0.03, 0.06, 'D')]),
    ('match', 'tool', 4, [('l', 0.3, 0.64, 0.58, 0.42, 0.03, 'B'), ('e', 0.62, 0.38, 0.04, 0.06, 'A')]),
    ('key', 'tool', 4, [('c', 0.32, 0.5, 0.13, 'A'), ('c', 0.32, 0.5, 0.05, 'W'), ('r', 0.42, 0.47, 0.8, 0.53, 'A'), ('r', 0.66, 0.53, 0.7, 0.62, 'A'), ('r', 0.74, 0.53, 0.78, 0.64, 'A')]),
    ('padlock', 'tool', 5, [('rr', 0.34, 0.44, 0.66, 0.8, 0.05, 'A'), ('a', 0.5, 0.44, 0.13, 180, 360, 0.035, 'C'), ('c', 0.5, 0.58, 0.035, 'Bk'), ('l', 0.5, 0.58, 0.5, 0.66, 0.02, 'Bk')]),
    ('chain', 'tool', 5, [('a', 0.4, 0.4, 0.09, 0, 360, 0.03, 'A'), ('a', 0.52, 0.52, 0.09, 0, 360, 0.03, 'A'), ('a', 0.64, 0.64, 0.09, 0, 360, 0.03, 'A')]),
    ('rope', 'tool', 6, [('a', 0.4, 0.46, 0.14, 40, 300, 0.05, 'A'), ('l', 0.5, 0.6, 0.62, 0.72, 0.05, 'A'), ('c', 0.66, 0.76, 0.04, 'B')]),
    ('shovel', 'tool', 8, [('p', [(0.42, 0.14), (0.58, 0.14), (0.6, 0.42), (0.4, 0.42)], 'A'), ('a', 0.5, 0.42, 0.1, 0, 180, 0.03, 'A'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'A'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'A')]),
    ('broom', 'tool', 9, [('r', 0.36, 0.14, 0.64, 0.24, 'A'), ('l', 0.38, 0.24, 0.36, 0.4, 0.02, 'B'), ('l', 0.46, 0.24, 0.46, 0.42, 0.02, 'B'), ('l', 0.54, 0.24, 0.54, 0.42, 0.02, 'B'), ('l', 0.62, 0.24, 0.64, 0.4, 0.02, 'B'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'B'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'B')]),
    ('mop', 'tool', 9, [('rr', 0.4, 0.16, 0.6, 0.42, 0.03, 'A'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'D'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'D')]),
    ('bucket', 'tool', 8, [('p', [(0.3, 0.36), (0.7, 0.36), (0.62, 0.8), (0.38, 0.8)], 'A'), ('a', 0.5, 0.36, 0.2, 180, 360, 0.03, 'C'), ('a', 0.5, 0.3, 0.16, 200, 340, 0.02, 'C')]),
    ('ladder', 'tool', 12, [('l', 0.38, 0.16, 0.46, 0.84, 0.03, 'A'), ('l', 0.62, 0.16, 0.54, 0.84, 0.03, 'A'), ('l', 0.4, 0.3, 0.6, 0.3, 0.025, 'A'), ('l', 0.42, 0.46, 0.58, 0.46, 0.025, 'A'), ('l', 0.43, 0.62, 0.57, 0.62, 0.025, 'A'), ('l', 0.44, 0.76, 0.56, 0.76, 0.025, 'A')]),
    ('axe', 'tool', 7, [('p', [(0.38, 0.18), (0.62, 0.14), (0.64, 0.3), (0.42, 0.34)], 'A'), ('p', [(0.4, 0.2), (0.46, 0.14), (0.46, 0.3)], 'W'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'B'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'B')]),
    ('pickaxe', 'tool', 8, [('l', 0.32, 0.28, 0.68, 0.18, 0.04, 'A'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'A'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'A')]),
    ('pliers', 'tool', 5, [('p', [(0.4, 0.18), (0.48, 0.18), (0.5, 0.44), (0.44, 0.44)], 'A'), ('p', [(0.52, 0.18), (0.6, 0.18), (0.56, 0.44), (0.5, 0.44)], 'A'), ('c', 0.5, 0.46, 0.025, 'Bk'), ('rr', 0.38, 0.46, 0.48, 0.78, 0.04, 'B'), ('rr', 0.52, 0.46, 0.62, 0.78, 0.04, 'B')]),
    ('clamp', 'tool', 6, [('r', 0.3, 0.3, 0.38, 0.7, 'A'), ('r', 0.62, 0.3, 0.7, 0.7, 'A'), ('l', 0.3, 0.5, 0.7, 0.5, 0.04, 'C'), ('l', 0.46, 0.5, 0.54, 0.42, 0.04, 'C'), ('c', 0.5, 0.4, 0.03, 'B')]),
    ('chisel', 'tool', 6, [('l', 0.44, 0.18, 0.44, 0.42, 0.05, 'D'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'B'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'B')]),
    ('trowel', 'tool', 6, [('p', [(0.4, 0.14), (0.6, 0.14), (0.5, 0.42)], 'A'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'A'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'A')]),
    ('level', 'tool', 7, [('rr', 0.16, 0.44, 0.84, 0.58, 0.02, 'A'), ('rr', 0.46, 0.46, 0.54, 0.56, 0.02, 'D'), ('c', 0.5, 0.51, 0.012, 'B')]),
    ('tape_measure', 'tool', 5, [('c', 0.5, 0.46, 0.16, 'A'), ('c', 0.5, 0.46, 0.04, 'Bk'), ('l', 0.64, 0.54, 0.8, 0.66, 0.05, 'B'), ('l', 0.7, 0.56, 0.7, 0.62, 0.012, 'Bk'), ('l', 0.76, 0.6, 0.76, 0.66, 0.012, 'Bk')]),
    ('work_glove', 'tool', 6, [('p', [(0.4, 0.3), (0.6, 0.3), (0.62, 0.62), (0.74, 0.52), (0.78, 0.6), (0.66, 0.72), (0.62, 0.8), (0.4, 0.8), (0.38, 0.62)], 'A'), ('l', 0.46, 0.34, 0.46, 0.52, 0.012, 'C'), ('l', 0.53, 0.34, 0.53, 0.52, 0.012, 'C'), ('l', 0.59, 0.34, 0.59, 0.52, 0.012, 'C')]),
    ('hard_hat', 'tool', 5, [('p', [(0.26, 0.5), (0.34, 0.3), (0.62, 0.3), (0.7, 0.5)], 'A'), ('p', [(0.26, 0.5), (0.84, 0.5), (0.8, 0.58), (0.26, 0.58)], 'C')]),
    ('sofa', 'furniture', 22, [('rr', 0.2, 0.44, 0.8, 0.72, 0.05, 'A'), ('rr', 0.16, 0.34, 0.3, 0.72, 0.05, 'B'), ('rr', 0.7, 0.34, 0.84, 0.72, 0.05, 'B'), ('rr', 0.28, 0.4, 0.48, 0.56, 0.03, 'D'), ('rr', 0.52, 0.4, 0.72, 0.56, 0.03, 'D'), ('r', 0.24, 0.72, 0.3, 0.82, 'C'), ('r', 0.7, 0.72, 0.76, 0.82, 'C')]),
    ('bed', 'furniture', 26, [('rr', 0.2, 0.4, 0.8, 0.72, 0.04, 'A'), ('rr', 0.24, 0.34, 0.44, 0.5, 0.03, 'D'), ('r', 0.22, 0.7, 0.28, 0.82, 'C'), ('r', 0.72, 0.7, 0.78, 0.82, 'C')]),
    ('pillow', 'furniture', 12, [('rr', 0.24, 0.4, 0.76, 0.64, 0.1, 'A'), ('rr', 0.3, 0.46, 0.7, 0.58, 0.08, 'D')]),
    ('blanket', 'furniture', 16, [('rr', 0.26, 0.3, 0.74, 0.7, 0.04, 'A'), ('l', 0.26, 0.44, 0.74, 0.44, 0.02, 'C'), ('l', 0.26, 0.58, 0.74, 0.58, 0.02, 'C'), ('l', 0.26, 0.64, 0.74, 0.64, 0.03, 'B')]),
    ('rug', 'furniture', 18, [('rr', 0.22, 0.36, 0.78, 0.68, 0.03, 'A'), ('rr', 0.3, 0.42, 0.7, 0.62, 0.03, 'B'), ('e', 0.5, 0.52, 0.06, 0.04, 'D')]),
    ('vase', 'furniture', 10, [('p', [(0.44, 0.24), (0.56, 0.24), (0.54, 0.36), (0.64, 0.52), (0.62, 0.74), (0.5, 0.82), (0.38, 0.74), (0.36, 0.52), (0.46, 0.36)], 'A'), ('l', 0.52, 0.24, 0.58, 0.12, 0.015, 'B'), ('l', 0.46, 0.24, 0.4, 0.1, 0.015, 'B')]),
    ('pot', 'furniture', 8, [('rr', 0.36, 0.34, 0.64, 0.82, 0.05, 'A'), ('r', 0.38, 0.24, 0.62, 0.34, 'D'), ('l', 0.38, 0.3, 0.62, 0.3, 0.015, 'C'), ('l', 0.36, 0.3, 0.3, 0.24, 0.025, 'C'), ('l', 0.64, 0.3, 0.7, 0.24, 0.025, 'C')]),
    ('kettle', 'furniture', 8, [('e', 0.52, 0.54, 0.18, 0.16, 'A'), ('a', 0.52, 0.54, 0.18, 200, 340, 0.03, 'C'), ('l', 0.34, 0.5, 0.26, 0.4, 0.03, 'A'), ('c', 0.52, 0.34, 0.025, 'C')]),
    ('fridge', 'furniture', 28, [('rr', 0.32, 0.18, 0.68, 0.84, 0.03, 'A'), ('l', 0.32, 0.44, 0.68, 0.44, 0.02, 'C'), ('l', 0.4, 0.28, 0.4, 0.38, 0.02, 'C'), ('l', 0.4, 0.54, 0.4, 0.72, 0.02, 'C')]),
    ('oven', 'furniture', 20, [('rr', 0.26, 0.3, 0.74, 0.8, 0.03, 'A'), ('rr', 0.32, 0.46, 0.68, 0.66, 0.02, 'Bk'), ('r', 0.32, 0.36, 0.68, 0.4, 'C'), ('c', 0.36, 0.42, 0.015, 'D'), ('c', 0.46, 0.42, 0.015, 'D'), ('c', 0.56, 0.42, 0.015, 'D'), ('c', 0.66, 0.42, 0.015, 'A')]),
    ('microwave', 'furniture', 16, [('rr', 0.22, 0.34, 0.78, 0.68, 0.03, 'A'), ('rr', 0.28, 0.4, 0.62, 0.62, 0.02, 'Bk'), ('c', 0.7, 0.44, 0.015, 'D'), ('c', 0.7, 0.54, 0.015, 'D')]),
    ('sink', 'furniture', 18, [('rr', 0.24, 0.4, 0.76, 0.72, 0.05, 'A'), ('e', 0.5, 0.56, 0.18, 0.1, 'Bk'), ('l', 0.5, 0.34, 0.5, 0.44, 0.03, 'C'), ('c', 0.5, 0.32, 0.02, 'C')]),
    ('towel', 'furniture', 10, [('rr', 0.34, 0.28, 0.66, 0.72, 0.03, 'A'), ('l', 0.34, 0.38, 0.66, 0.38, 0.025, 'D'), ('l', 0.34, 0.62, 0.66, 0.62, 0.025, 'D')]),
    ('soap', 'furniture', 4, [('rr', 0.36, 0.42, 0.64, 0.62, 0.08, 'A'), ('c', 0.46, 0.5, 0.015, 'W')]),
    ('toothbrush', 'furniture', 5, [('l', 0.24, 0.64, 0.66, 0.4, 0.035, 'A'), ('r', 0.66, 0.34, 0.78, 0.4, 'B')]),
    ('comb', 'furniture', 4, [('rr', 0.26, 0.44, 0.74, 0.52, 0.02, 'A'), ('l', 0.3, 0.52, 0.3, 0.62, 0.018, 'A'), ('l', 0.38, 0.52, 0.38, 0.62, 0.018, 'A'), ('l', 0.46, 0.52, 0.46, 0.62, 0.018, 'A'), ('l', 0.54, 0.52, 0.54, 0.62, 0.018, 'A'), ('l', 0.62, 0.52, 0.62, 0.62, 0.018, 'A'), ('l', 0.7, 0.52, 0.7, 0.62, 0.018, 'A')]),
    ('mirror', 'furniture', 14, [('e', 0.5, 0.48, 0.2, 0.26, 'C'), ('e', 0.5, 0.48, 0.17, 0.23, 'D'), ('l', 0.44, 0.36, 0.4, 0.3, 0.015, 'W')]),
    ('lamp', 'furniture', 12, [('p', [(0.38, 0.2), (0.62, 0.2), (0.7, 0.44), (0.3, 0.44)], 'A'), ('r', 0.48, 0.44, 0.52, 0.76, 'C'), ('r', 0.38, 0.76, 0.62, 0.84, 'C')]),
    ('shelf', 'furniture', 20, [('l', 0.24, 0.34, 0.76, 0.34, 0.03, 'A'), ('l', 0.24, 0.54, 0.76, 0.54, 0.03, 'A'), ('l', 0.24, 0.74, 0.76, 0.74, 0.03, 'A'), ('r', 0.3, 0.24, 0.36, 0.34, 'B'), ('r', 0.52, 0.44, 0.58, 0.54, 'C'), ('r', 0.44, 0.64, 0.5, 0.74, 'D')]),
    ('closet', 'furniture', 24, [('r', 0.25, 0.16, 0.75, 0.86, 'A'), ('r', 0.23, 0.12, 0.77, 0.18, 'C'), ('rr', 0.45, 0.66, 0.55, 0.86, 0.02, 'C'), ('l', 0.5, 0.16, 0.5, 0.86, 0.02, 'C'), ('c', 0.47, 0.52, 0.015, 'Bk'), ('c', 0.53, 0.52, 0.015, 'Bk')]),
    ('curtain', 'furniture', 16, [('l', 0.24, 0.2, 0.76, 0.2, 0.03, 'C'), ('l', 0.3, 0.2, 0.28, 0.84, 0.03, 'A'), ('l', 0.38, 0.2, 0.38, 0.84, 0.03, 'A'), ('l', 0.62, 0.2, 0.62, 0.84, 0.03, 'A'), ('l', 0.7, 0.2, 0.72, 0.84, 0.03, 'A')]),
    ('mattress', 'furniture', 20, [('rr', 0.18, 0.42, 0.82, 0.68, 0.05, 'A'), ('rr', 0.22, 0.38, 0.42, 0.52, 0.03, 'D')]),
    ('basket', 'furniture', 9, [('p', [(0.28, 0.44), (0.72, 0.44), (0.64, 0.78), (0.36, 0.78)], 'A'), ('l', 0.32, 0.52, 0.68, 0.52, 0.015, 'C'), ('l', 0.34, 0.6, 0.66, 0.6, 0.015, 'C'), ('l', 0.36, 0.68, 0.64, 0.68, 0.015, 'C'), ('a', 0.5, 0.44, 0.18, 200, 340, 0.02, 'C')]),
    ('jar', 'furniture', 7, [('rr', 0.36, 0.34, 0.64, 0.82, 0.05, 'A'), ('r', 0.38, 0.24, 0.62, 0.34, 'D'), ('l', 0.38, 0.3, 0.62, 0.3, 0.015, 'C')]),
    ('bottle', 'furniture', 8, [('rr', 0.42, 0.34, 0.58, 0.82, 0.04, 'A'), ('r', 0.46, 0.2, 0.54, 0.34, 'A'), ('r', 0.45, 0.16, 0.55, 0.22, 'D'), ('r', 0.42, 0.5, 0.58, 0.68, 'D')]),
    ('plate', 'furniture', 10, [('e', 0.5, 0.5, 0.28, 0.09, 'A'), ('e', 0.5, 0.5, 0.18, 0.06, 'D')]),
    ('bowl', 'furniture', 9, [('p', [(0.24, 0.44), (0.76, 0.44), (0.66, 0.74), (0.34, 0.74)], 'A'), ('e', 0.5, 0.44, 0.26, 0.07, 'D')]),
    ('mug', 'furniture', 7, [('rr', 0.36, 0.36, 0.6, 0.78, 0.03, 'A'), ('a', 0.6, 0.57, 0.09, -90, 90, 0.035, 'A')]),
    ('napkin', 'furniture', 6, [('p', [(0.3, 0.3), (0.7, 0.3), (0.64, 0.7), (0.36, 0.7)], 'A'), ('l', 0.34, 0.44, 0.66, 0.44, 0.015, 'C'), ('l', 0.35, 0.56, 0.65, 0.56, 0.015, 'C')]),
    ('candlestick', 'furniture', 9, [('r', 0.46, 0.4, 0.54, 0.74, 'A'), ('r', 0.38, 0.74, 0.62, 0.82, 'A'), ('r', 0.44, 0.32, 0.56, 0.4, 'A'), ('e', 0.5, 0.26, 0.025, 0.05, 'D')]),
    ('fireplace', 'furniture', 22, [('r', 0.22, 0.3, 0.78, 0.82, 'C'), ('rr', 0.36, 0.46, 0.64, 0.82, 0.04, 'Bk'), ('p', [(0.44, 0.8), (0.5, 0.56), (0.56, 0.8)], 'A'), ('p', [(0.5, 0.8), (0.55, 0.64), (0.6, 0.8)], 'D')]),
    ('wardrobe', 'furniture', 26, [('r', 0.25, 0.16, 0.75, 0.86, 'A'), ('r', 0.23, 0.12, 0.77, 0.18, 'C'), ('rr', 0.45, 0.66, 0.55, 0.86, 0.02, 'C'), ('l', 0.5, 0.18, 0.5, 0.86, 0.02, 'C'), ('c', 0.46, 0.5, 0.015, 'Bk'), ('c', 0.54, 0.5, 0.015, 'Bk')]),
    ('bookshelf', 'furniture', 24, [('r', 0.25, 0.16, 0.75, 0.86, 'A'), ('r', 0.23, 0.12, 0.77, 0.18, 'C'), ('e', 0.35, 0.28, 0.04, 0.055, 'D'), ('e', 0.475, 0.28, 0.04, 0.055, 'D'), ('e', 0.6, 0.28, 0.04, 0.055, 'D'), ('e', 0.35, 0.45000000000000007, 0.04, 0.055, 'D'), ('e', 0.475, 0.45000000000000007, 0.04, 0.055, 'D'), ('e', 0.6, 0.45000000000000007, 0.04, 0.055, 'D'), ('e', 0.35, 0.6200000000000001, 0.04, 0.055, 'D'), ('e', 0.475, 0.6200000000000001, 0.04, 0.055, 'D'), ('e', 0.6, 0.6200000000000001, 0.04, 0.055, 'D'), ('r', 0.28, 0.24, 0.34, 0.32, 'B'), ('r', 0.56, 0.24, 0.62, 0.32, 'D'), ('r', 0.3, 0.44, 0.36, 0.52, 'A'), ('r', 0.6, 0.64, 0.66, 0.72, 'B')]),
    ('laundry_basket', 'furniture', 10, [('e', 0.5, 0.5, 0.18, 'A')]),
    ('hanger', 'furniture', 6, [('a', 0.5, 0.3, 0.05, 180, 360, 0.02, 'C'), ('p', [(0.5, 0.32), (0.26, 0.56), (0.74, 0.56)], 'A')]),
    ('remote_control', 'furniture', 5, [('rr', 0.42, 0.24, 0.58, 0.78, 0.04, 'A'), ('c', 0.5, 0.34, 0.02, 'D'), ('c', 0.46, 0.48, 0.015, 'C'), ('c', 0.54, 0.48, 0.015, 'C'), ('c', 0.46, 0.62, 0.015, 'C'), ('c', 0.54, 0.62, 0.015, 'C')]),
    ('tv', 'electronics', 22, [('rr', 0.18, 0.3, 0.82, 0.68, 0.02, 'A'), ('rr', 0.22, 0.34, 0.78, 0.64, 0.02, 'D'), ('l', 0.42, 0.74, 0.58, 0.74, 0.03, 'A'), ('r', 0.38, 0.74, 0.62, 0.8, 'A')]),
    ('radio', 'electronics', 12, [('rr', 0.2, 0.36, 0.8, 0.66, 0.04, 'A'), ('c', 0.34, 0.51, 0.09, 'D'), ('l', 0.52, 0.42, 0.72, 0.42, 0.02, 'C'), ('l', 0.52, 0.5, 0.72, 0.5, 0.02, 'C'), ('l', 0.52, 0.58, 0.72, 0.58, 0.02, 'C'), ('l', 0.28, 0.36, 0.2, 0.2, 0.02, 'C')]),
    ('computer', 'electronics', 18, [('rr', 0.28, 0.24, 0.72, 0.58, 0.02, 'A'), ('rr', 0.32, 0.28, 0.68, 0.54, 0.02, 'D'), ('r', 0.46, 0.58, 0.54, 0.64, 'A'), ('rr', 0.36, 0.66, 0.64, 0.74, 0.02, 'A')]),
    ('laptop', 'electronics', 14, [('p', [(0.26, 0.3), (0.74, 0.3), (0.7, 0.58), (0.3, 0.58)], 'A'), ('p', [(0.3, 0.34), (0.7, 0.34), (0.67, 0.54), (0.33, 0.54)], 'D'), ('p', [(0.2, 0.58), (0.8, 0.58), (0.86, 0.7), (0.14, 0.7)], 'A')]),
    ('phone', 'electronics', 6, [('rr', 0.4, 0.22, 0.6, 0.78, 0.05, 'A'), ('rr', 0.44, 0.3, 0.56, 0.64, 0.02, 'D'), ('c', 0.5, 0.72, 0.015, 'C')]),
    ('camera', 'electronics', 9, [('rr', 0.24, 0.38, 0.76, 0.68, 0.04, 'A'), ('rr', 0.3, 0.3, 0.44, 0.4, 0.02, 'A'), ('c', 0.5, 0.53, 0.1, 'Bk'), ('c', 0.5, 0.53, 0.06, 'D'), ('c', 0.68, 0.44, 0.02, 'B')]),
    ('speaker', 'electronics', 12, [('rr', 0.34, 0.26, 0.66, 0.78, 0.04, 'A'), ('c', 0.5, 0.44, 0.07, 'C'), ('c', 0.5, 0.64, 0.1, 'C')]),
    ('headphones', 'electronics', 8, [('a', 0.5, 0.44, 0.22, 180, 360, 0.03, 'A'), ('rr', 0.24, 0.44, 0.34, 0.66, 0.04, 'B'), ('rr', 0.66, 0.44, 0.76, 0.66, 0.04, 'B')]),
    ('battery', 'electronics', 4, [('rr', 0.34, 0.38, 0.62, 0.66, 0.03, 'A'), ('r', 0.62, 0.46, 0.68, 0.58, 'A'), ('l', 0.42, 0.46, 0.42, 0.58, 0.02, 'Bk'), ('l', 0.38, 0.52, 0.46, 0.52, 0.02, 'Bk')]),
    ('charger', 'electronics', 4, [('rr', 0.36, 0.38, 0.64, 0.64, 0.04, 'A'), ('l', 0.44, 0.3, 0.44, 0.38, 0.025, 'C'), ('l', 0.56, 0.3, 0.56, 0.38, 0.025, 'C'), ('l', 0.5, 0.64, 0.5, 0.76, 0.025, 'C')]),
    ('keyboard', 'electronics', 14, [('rr', 0.16, 0.4, 0.84, 0.64, 0.03, 'A'), ('c', 0.28, 0.48, 0.02, 'C'), ('c', 0.4, 0.48, 0.02, 'C'), ('c', 0.52, 0.48, 0.02, 'C'), ('c', 0.64, 0.48, 0.02, 'C'), ('c', 0.76, 0.48, 0.02, 'C'), ('r', 0.36, 0.55, 0.64, 0.59, 'C')]),
    ('joystick', 'electronics', 8, [('rr', 0.32, 0.54, 0.68, 0.78, 0.05, 'A'), ('l', 0.5, 0.54, 0.5, 0.36, 0.03, 'C'), ('c', 0.5, 0.32, 0.07, 'B'), ('c', 0.4, 0.66, 0.02, 'D'), ('c', 0.48, 0.66, 0.02, 'C')]),
    ('gamepad', 'electronics', 9, [('p', [(0.24, 0.44), (0.76, 0.44), (0.82, 0.58), (0.68, 0.68), (0.32, 0.68), (0.18, 0.58)], 'A'), ('c', 0.36, 0.56, 0.03, 'C'), ('c', 0.4, 0.52, 0.03, 'C'), ('c', 0.64, 0.54, 0.02, 'D'), ('c', 0.7, 0.58, 0.02, 'B')]),
    ('console', 'electronics', 12, [('rr', 0.28, 0.34, 0.72, 0.62, 0.05, 'A'), ('l', 0.28, 0.48, 0.72, 0.48, 0.02, 'C'), ('c', 0.64, 0.56, 0.02, 'D')]),
    ('drone', 'electronics', 10, [('rr', 0.42, 0.44, 0.58, 0.58, 0.03, 'A'), ('l', 0.3, 0.36, 0.7, 0.36, 0.02, 'C'), ('l', 0.3, 0.68, 0.7, 0.68, 0.02, 'C'), ('a', 0.26, 0.36, 0.08, 0, 360, 0.02, 'D'), ('a', 0.74, 0.36, 0.08, 0, 360, 0.02, 'D'), ('a', 0.26, 0.68, 0.08, 0, 360, 0.02, 'D'), ('a', 0.74, 0.68, 0.08, 0, 360, 0.02, 'D'), ('c', 0.5, 0.51, 0.02, 'Bk')]),
    ('satellite', 'electronics', 14, [('rr', 0.44, 0.42, 0.56, 0.6, 0.02, 'A'), ('r', 0.16, 0.46, 0.44, 0.56, 'B'), ('r', 0.56, 0.46, 0.84, 0.56, 'B'), ('a', 0.5, 0.7, 0.08, 0, 360, 0.02, 'C')]),
    ('antenna', 'electronics', 10, [('l', 0.5, 0.84, 0.5, 0.34, 0.03, 'C'), ('a', 0.5, 0.34, 0.1, 180, 360, 0.015, 'A'), ('a', 0.5, 0.34, 0.06, 180, 360, 0.015, 'A'), ('c', 0.5, 0.3, 0.02, 'D'), ('r', 0.42, 0.84, 0.58, 0.9, 'C')]),
    ('solar_panel', 'electronics', 16, [('p', [(0.24, 0.36), (0.76, 0.36), (0.68, 0.7), (0.32, 0.7)], 'A'), ('l', 0.34, 0.36, 0.42, 0.7, 0.012, 'D'), ('l', 0.48, 0.36, 0.48, 0.7, 0.012, 'D'), ('l', 0.62, 0.36, 0.56, 0.7, 0.012, 'D'), ('l', 0.28, 0.52, 0.72, 0.52, 0.012, 'D')]),
    ('monitor', 'electronics', 16, [('rr', 0.24, 0.28, 0.76, 0.62, 0.02, 'A'), ('rr', 0.28, 0.32, 0.72, 0.58, 0.02, 'D'), ('r', 0.46, 0.62, 0.54, 0.7, 'A'), ('rr', 0.36, 0.7, 0.64, 0.78, 0.02, 'A')]),
    ('hat', 'clothing', 8, [('p', [(0.24, 0.55), (0.36, 0.3), (0.64, 0.3), (0.76, 0.55), (0.86, 0.6), (0.14, 0.6)], 'A'), ('r', 0.24, 0.52, 0.76, 0.6, 'C')]),
    ('cap', 'clothing', 7, [('p', [(0.26, 0.5), (0.34, 0.3), (0.62, 0.3), (0.7, 0.5)], 'A'), ('p', [(0.26, 0.5), (0.84, 0.5), (0.8, 0.58), (0.26, 0.58)], 'C')]),
    ('shirt', 'clothing', 10, [('p', [(0.3, 0.28), (0.42, 0.22), (0.58, 0.22), (0.7, 0.28), (0.64, 0.4), (0.58, 0.36), (0.58, 0.78), (0.42, 0.78), (0.42, 0.36), (0.36, 0.4)], 'A'), ('a', 0.5, 0.235, 0.05, 180, 360, 0.02, 'C')]),
    ('pants', 'clothing', 10, [('p', [(0.36, 0.2), (0.64, 0.2), (0.66, 0.42), (0.6, 0.84), (0.52, 0.84), (0.5, 0.46), (0.48, 0.84), (0.4, 0.84), (0.34, 0.42)], 'A'), ('l', 0.36, 0.26, 0.64, 0.26, 0.025, 'C')]),
    ('dress', 'clothing', 12, [('p', [(0.4, 0.2), (0.6, 0.2), (0.58, 0.42), (0.74, 0.82), (0.26, 0.82), (0.42, 0.42)], 'A'), ('a', 0.5, 0.215, 0.05, 180, 360, 0.02, 'C')]),
    ('shoes', 'clothing', 8, [('p', [(0.2, 0.55), (0.42, 0.55), (0.52, 0.42), (0.62, 0.5), (0.82, 0.58), (0.82, 0.68), (0.2, 0.68)], 'A'), ('l', 0.2, 0.64, 0.82, 0.64, 0.02, 'C'), ('l', 0.44, 0.56, 0.5, 0.5, 0.012, 'W'), ('l', 0.5, 0.54, 0.56, 0.48, 0.012, 'W')]),
    ('glove', 'clothing', 5, [('p', [(0.4, 0.3), (0.6, 0.3), (0.62, 0.62), (0.74, 0.52), (0.78, 0.6), (0.66, 0.72), (0.62, 0.8), (0.4, 0.8), (0.38, 0.62)], 'A'), ('l', 0.46, 0.34, 0.46, 0.52, 0.012, 'C'), ('l', 0.53, 0.34, 0.53, 0.52, 0.012, 'C'), ('l', 0.59, 0.34, 0.59, 0.52, 0.012, 'C')]),
    ('scarf', 'clothing', 6, [('rr', 0.38, 0.2, 0.62, 0.44, 0.06, 'A'), ('r', 0.44, 0.4, 0.54, 0.82, 'A'), ('l', 0.44, 0.78, 0.54, 0.78, 0.02, 'D'), ('l', 0.38, 0.3, 0.62, 0.3, 0.02, 'D')]),
    ('belt', 'clothing', 6, [('p', [(0.16, 0.44), (0.84, 0.44), (0.84, 0.58), (0.16, 0.58)], 'A'), ('rr', 0.44, 0.38, 0.56, 0.64, 0.02, 'D'), ('l', 0.44, 0.51, 0.56, 0.51, 0.015, 'Bk')]),
    ('sock', 'clothing', 5, [('p', [(0.42, 0.2), (0.6, 0.2), (0.6, 0.58), (0.72, 0.72), (0.56, 0.84), (0.36, 0.84), (0.36, 0.66), (0.42, 0.6)], 'A'), ('r', 0.4, 0.2, 0.62, 0.28, 'D')]),
    ('swimsuit', 'clothing', 8, [('p', [(0.4, 0.2), (0.6, 0.2), (0.58, 0.42), (0.74, 0.82), (0.26, 0.82), (0.42, 0.42)], 'A'), ('a', 0.5, 0.215, 0.05, 180, 360, 0.02, 'C'), ('l', 0.4, 0.46, 0.6, 0.46, 0.02, 'C')]),
    ('crown', 'clothing', 5, [('p', [(0.28, 0.62), (0.28, 0.4), (0.38, 0.52), (0.44, 0.32), (0.5, 0.5), (0.56, 0.32), (0.62, 0.52), (0.72, 0.4), (0.72, 0.62)], 'A'), ('c', 0.36, 0.56, 0.015, 'D'), ('c', 0.5, 0.56, 0.015, 'B'), ('c', 0.64, 0.56, 0.015, 'D')]),
    ('ring', 'clothing', 3, [('a', 0.5, 0.58, 0.18, 0, 360, 0.06, 'A'), ('c', 0.5, 0.34, 0.05, 'D')]),
    ('necklace', 'clothing', 4, [('a', 0.5, 0.44, 0.2, 0, 180, 0.02, 'A'), ('c', 0.5, 0.64, 0.035, 'D')]),
    ('bracelet', 'clothing', 3, [('a', 0.5, 0.52, 0.16, 0, 360, 0.035, 'A'), ('c', 0.5, 0.36, 0.02, 'D')]),
    ('earring', 'clothing', 2, [('l', 0.5, 0.3, 0.5, 0.44, 0.012, 'A'), ('c', 0.5, 0.54, 0.05, 'A'), ('c', 0.5, 0.54, 0.02, 'D')]),
    ('watch', 'clothing', 4, [('rr', 0.44, 0.2, 0.56, 0.38, 0.02, 'C'), ('rr', 0.44, 0.62, 0.56, 0.8, 0.02, 'C'), ('c', 0.5, 0.5, 0.15, 'A'), ('c', 0.5, 0.5, 0.11, 'D'), ('l', 0.5, 0.5, 0.5, 0.43, 0.015, 'Bk'), ('l', 0.5, 0.5, 0.55, 0.5, 0.015, 'Bk')]),
    ('backpack', 'clothing', 12, [('rr', 0.34, 0.28, 0.66, 0.82, 0.08, 'A'), ('rr', 0.4, 0.54, 0.6, 0.8, 0.05, 'B'), ('a', 0.44, 0.28, 0.08, 180, 360, 0.025, 'C'), ('a', 0.56, 0.28, 0.08, 180, 360, 0.025, 'C')]),
    ('suitcase', 'clothing', 16, [('rr', 0.3, 0.34, 0.7, 0.8, 0.05, 'A'), ('r', 0.44, 0.24, 0.56, 0.36, 'C'), ('l', 0.42, 0.34, 0.42, 0.8, 0.02, 'C'), ('l', 0.58, 0.34, 0.58, 0.8, 0.02, 'C')]),
    ('wallet', 'clothing', 5, [('rr', 0.28, 0.38, 0.72, 0.66, 0.03, 'A'), ('c', 0.64, 0.52, 0.025, 'D')]),
    ('purse', 'clothing', 7, [('rr', 0.32, 0.42, 0.68, 0.7, 0.04, 'A'), ('a', 0.5, 0.42, 0.12, 180, 360, 0.03, 'C'), ('c', 0.5, 0.56, 0.02, 'D')]),
    ('sandal', 'clothing', 5, [('p', [(0.2, 0.55), (0.42, 0.55), (0.52, 0.42), (0.62, 0.5), (0.82, 0.58), (0.82, 0.68), (0.2, 0.68)], 'A'), ('l', 0.2, 0.64, 0.82, 0.64, 0.02, 'C'), ('l', 0.44, 0.56, 0.5, 0.5, 0.012, 'W'), ('l', 0.5, 0.54, 0.56, 0.48, 0.012, 'W'), ('l', 0.44, 0.5, 0.56, 0.5, 0.015, 'C')]),
    ('slipper', 'clothing', 5, [('p', [(0.24, 0.56), (0.5, 0.52), (0.76, 0.58), (0.76, 0.7), (0.24, 0.7)], 'A'), ('e', 0.44, 0.54, 0.1, 0.04, 'D')]),
    ('jacket', 'clothing', 12, [('p', [(0.3, 0.28), (0.42, 0.22), (0.58, 0.22), (0.7, 0.28), (0.64, 0.4), (0.58, 0.36), (0.58, 0.78), (0.42, 0.78), (0.42, 0.36), (0.36, 0.4)], 'A'), ('a', 0.5, 0.235, 0.05, 180, 360, 0.02, 'C'), ('l', 0.5, 0.24, 0.5, 0.78, 0.02, 'C')]),
    ('jeans', 'clothing', 10, [('p', [(0.36, 0.2), (0.64, 0.2), (0.66, 0.42), (0.6, 0.84), (0.52, 0.84), (0.5, 0.46), (0.48, 0.84), (0.4, 0.84), (0.34, 0.42)], 'A'), ('l', 0.36, 0.26, 0.64, 0.26, 0.025, 'C')]),
    ('shorts', 'clothing', 8, [('p', [(0.36, 0.2), (0.64, 0.2), (0.66, 0.42), (0.6, 0.84), (0.52, 0.84), (0.5, 0.46), (0.48, 0.84), (0.4, 0.84), (0.34, 0.42)], 'A'), ('l', 0.36, 0.26, 0.64, 0.26, 0.025, 'C'), ('l', 0.4, 0.56, 0.6, 0.56, 0.02, 'C')]),
    ('tie', 'clothing', 4, [('p', [(0.46, 0.26), (0.54, 0.26), (0.58, 0.44), (0.5, 0.8), (0.42, 0.44)], 'A'), ('r', 0.44, 0.22, 0.56, 0.28, 'C')]),
    ('bow_tie', 'clothing', 4, [('p', [(0.3, 0.42), (0.48, 0.52), (0.3, 0.62)], 'A'), ('p', [(0.7, 0.42), (0.52, 0.52), (0.7, 0.62)], 'A'), ('c', 0.5, 0.52, 0.03, 'C')]),
    ('swim_cap', 'clothing', 5, [('e', 0.5, 0.5, 0.18, 0.16, 'A'), ('l', 0.34, 0.58, 0.66, 0.58, 0.02, 'C')]),
    ('football', 'sports', 9, [('c', 0.5, 0.5, 0.26, 'A'), ('p', [(0.32, 0.5), (0.42, 0.36), (0.52, 0.5), (0.42, 0.66)], 'Bk'), ('a', 0.5, 0.5, 0.26, 200, 340, 0.02, 'Bk'), ('a', 0.5, 0.5, 0.26, 20, 160, 0.02, 'Bk')]),
    ('basketball', 'sports', 9, [('c', 0.5, 0.5, 0.26, 'A'), ('l', 0.28, 0.4, 0.72, 0.4, 0.03, 'Bk'), ('l', 0.28, 0.5, 0.72, 0.5, 0.03, 'Bk'), ('l', 0.28, 0.6000000000000001, 0.72, 0.6000000000000001, 0.03, 'Bk')]),
    ('baseball', 'sports', 6, [('c', 0.5, 0.5, 0.26, 'A'), ('p', [(0.28, 0.55), (0.28, 0.45), (0.368, 0.55), (0.456, 0.45), (0.544, 0.55), (0.632, 0.45), (0.72, 0.62), (0.28, 0.62)], 'Bk')]),
    ('tennis_ball', 'sports', 6, [('c', 0.5, 0.5, 0.26, 'A'), ('l', 0.3, 0.5, 0.7, 0.5, 0.035, 'Bk'), ('l', 0.5, 0.3, 0.5, 0.7, 0.035, 'Bk')]),
    ('golf_ball', 'sports', 4, [('c', 0.5, 0.5, 0.26, 'A'), ('c', 0.42, 0.42, 0.028, 'Bk'), ('c', 0.58, 0.46, 0.028, 'Bk'), ('c', 0.48, 0.6, 0.028, 'Bk'), ('c', 0.62, 0.58, 0.028, 'Bk'), ('c', 0.4, 0.55, 0.028, 'Bk')]),
    ('hockey_puck', 'sports', 4, [('rr', 0.32, 0.44, 0.68, 0.58, 0.03, 'Bk')]),
    ('dart', 'sports', 5, [('l', 0.24, 0.62, 0.62, 0.4, 0.02, 'A'), ('p', [(0.62, 0.4), (0.78, 0.3), (0.7, 0.48)], 'D'), ('l', 0.24, 0.66, 0.24, 0.58, 0.04, 'C')]),
    ('bow', 'sports', 10, [('a', 0.4, 0.5, 0.24, 60, 300, 0.03, 'A'), ('l', 0.3, 0.29, 0.3, 0.71, 0.015, 'C'), ('l', 0.3, 0.3, 0.78, 0.3, 0.015, 'D')]),
    ('arrow', 'sports', 8, [('l', 0.2, 0.5, 0.72, 0.5, 0.02, 'A'), ('p', [(0.72, 0.42), (0.86, 0.5), (0.72, 0.58)], 'D'), ('p', [(0.2, 0.44), (0.3, 0.5), (0.2, 0.56)], 'C')]),
    ('sword', 'sports', 9, [('l', 0.28, 0.7, 0.68, 0.3, 0.025, 'D'), ('l', 0.42, 0.56, 0.5, 0.64, 0.03, 'C'), ('l', 0.24, 0.66, 0.3, 0.74, 0.04, 'A')]),
    ('shield', 'sports', 10, [('p', [(0.32, 0.3), (0.68, 0.3), (0.68, 0.54), (0.5, 0.74), (0.32, 0.54)], 'A'), ('l', 0.5, 0.34, 0.5, 0.66, 0.02, 'C'), ('l', 0.38, 0.42, 0.62, 0.42, 0.02, 'C')]),
    ('ball', 'sports', 9, [('c', 0.5, 0.5, 0.26, 'A'), ('a', 0.5, 0.5, 0.2, 180, 300, 0.02, 'W')]),
    ('yo_yo', 'sports', 4, [('c', 0.5, 0.54, 0.16, 'A'), ('c', 0.5, 0.54, 0.05, 'Bk'), ('l', 0.5, 0.38, 0.5, 0.2, 0.012, 'D')]),
    ('kite', 'sports', 8, [('p', [(0.5, 0.24), (0.7, 0.48), (0.5, 0.7), (0.3, 0.48)], 'A'), ('l', 0.5, 0.24, 0.5, 0.7, 0.015, 'C'), ('l', 0.3, 0.48, 0.7, 0.48, 0.015, 'C'), ('l', 0.5, 0.7, 0.62, 0.84, 0.012, 'D')]),
    ('balloon', 'sports', 8, [('e', 0.5, 0.44, 0.16, 0.18, 'A'), ('p', [(0.48, 0.62), (0.52, 0.62), (0.5, 0.66)], 'A'), ('l', 0.5, 0.66, 0.58, 0.82, 0.012, 'D')]),
    ('chess', 'sports', 7, [('rr', 0.3, 0.36, 0.7, 0.42, 0.01, 'A'), ('rr', 0.36, 0.44, 0.64, 0.62, 0.02, 'A'), ('e', 0.5, 0.52, 0.1, 0.12, 'A'), ('rr', 0.34, 0.62, 0.66, 0.74, 0.01, 'A')]),
    ('dice', 'sports', 5, [('rr', 0.32, 0.32, 0.68, 0.68, 0.08, 'A'), ('c', 0.42, 0.42, 0.022, 'Bk'), ('c', 0.58, 0.42, 0.022, 'Bk'), ('c', 0.5, 0.5, 0.022, 'Bk'), ('c', 0.42, 0.58, 0.022, 'Bk'), ('c', 0.58, 0.58, 0.022, 'Bk')]),
    ('domino', 'sports', 5, [('rr', 0.3, 0.34, 0.7, 0.66, 0.03, 'A'), ('l', 0.5, 0.34, 0.5, 0.66, 0.015, 'C'), ('c', 0.4, 0.44, 0.02, 'Bk'), ('c', 0.4, 0.56, 0.02, 'Bk'), ('c', 0.6, 0.5, 0.02, 'Bk')]),
    ('playing_card', 'sports', 6, [('rr', 0.36, 0.28, 0.64, 0.72, 0.03, 'A'), ('c', 0.46, 0.42, 0.03, 'A'), ('p', [(0.54, 0.54), (0.62, 0.62), (0.46, 0.62)], 'A')]),
    ('puzzle', 'sports', 8, [('r', 0.28, 0.32, 0.72, 0.68, 'A'), ('c', 0.5, 0.32, 0.05, 'A'), ('c', 0.28, 0.5, 0.05, 'A'), ('c', 0.72, 0.5, 0.05, 'A'), ('l', 0.44, 0.5, 0.56, 0.5, 0.02, 'C'), ('l', 0.5, 0.44, 0.5, 0.56, 0.02, 'C')]),
    ('ski', 'sports', 12, [('l', 0.2, 0.64, 0.8, 0.36, 0.025, 'A'), ('l', 0.2, 0.7, 0.8, 0.42, 0.025, 'B'), ('c', 0.8, 0.36, 0.02, 'A'), ('c', 0.8, 0.42, 0.02, 'B')]),
    ('snowboard', 'sports', 10, [('rr', 0.22, 0.44, 0.78, 0.58, 0.06, 'A'), ('c', 0.38, 0.51, 0.02, 'C'), ('c', 0.62, 0.51, 0.02, 'C')]),
    ('golf_club', 'sports', 8, [('r', 0.42, 0.16, 0.58, 0.24, 'A'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'C'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'C')]),
    ('racket', 'sports', 7, [('e', 0.5, 0.36, 0.14, 0.16, 'D'), ('l', 0.5, 0.52, 0.5, 0.82, 0.03, 'A'), ('l', 0.44, 0.3, 0.56, 0.3, 0.01, 'C'), ('l', 0.44, 0.38, 0.56, 0.38, 0.01, 'C')]),
    ('trophy', 'sports', 8, [('rr', 0.38, 0.24, 0.62, 0.52, 0.04, 'A'), ('a', 0.38, 0.36, 0.1, 90, 270, 0.025, 'A'), ('a', 0.62, 0.36, 0.1, -90, 90, 0.025, 'A'), ('r', 0.46, 0.52, 0.54, 0.68, 'A'), ('r', 0.4, 0.68, 0.6, 0.78, 'C')]),
    ('medal', 'sports', 5, [('a', 0.4, 0.34, 0.1, 0, 360, 0.03, 'A'), ('a', 0.6, 0.34, 0.1, 0, 360, 0.03, 'B'), ('c', 0.5, 0.56, 0.12, 'A'), ('c', 0.5, 0.56, 0.06, 'B')]),
    ('whistle', 'sports', 5, [('e', 0.42, 0.54, 0.14, 0.1, 'A'), ('l', 0.54, 0.5, 0.74, 0.46, 0.04, 'A'), ('c', 0.36, 0.52, 0.02, 'Bk')]),
    ('jersey', 'sports', 10, [('p', [(0.3, 0.28), (0.42, 0.22), (0.58, 0.22), (0.7, 0.28), (0.64, 0.4), (0.58, 0.36), (0.58, 0.78), (0.42, 0.78), (0.42, 0.36), (0.36, 0.4)], 'A'), ('a', 0.5, 0.235, 0.05, 180, 360, 0.02, 'C'), ('t', 0.42, 0.48, '7', 0.16, 'W')]),
    ('goggles', 'sports', 6, [('rr', 0.24, 0.42, 0.76, 0.6, 0.05, 'A'), ('c', 0.38, 0.51, 0.06, 'D'), ('c', 0.62, 0.51, 0.06, 'D')]),
    ('helmet', 'sports', 8, [('a', 0.5, 0.54, 0.2, 180, 360, 0.08, 'A'), ('r', 0.3, 0.52, 0.7, 0.6, 'A'), ('l', 0.5, 0.52, 0.5, 0.64, 0.02, 'C'), ('l', 0.38, 0.54, 0.62, 0.54, 0.02, 'C')]),
    ('paddle', 'sports', 8, [('e', 0.44, 0.32, 0.1, 0.12, 'A'), ('l', 0.46, 0.44, 0.58, 0.8, 0.03, 'B')]),
    ('cloud', 'nature', 25, [('e', 0.38, 0.52, 0.14, 0.1, 'A'), ('e', 0.54, 0.46, 0.16, 0.13, 'A'), ('e', 0.68, 0.54, 0.13, 0.09, 'A'), ('r', 0.3, 0.52, 0.78, 0.63, 'A')]),
    ('sun', 'nature', 30, [('c', 0.5, 0.5, 0.16, 'A'), ('l', 0.71, 0.5, 0.8, 0.5, 0.035, 'A'), ('l', 0.648492424049175, 0.648492424049175, 0.7121320343559643, 0.7121320343559643, 0.035, 'A'), ('l', 0.5, 0.71, 0.5, 0.8, 0.035, 'A'), ('l', 0.35150757595082505, 0.648492424049175, 0.2878679656440358, 0.7121320343559643, 0.035, 'A'), ('l', 0.29000000000000004, 0.5, 0.2, 0.5, 0.035, 'A'), ('l', 0.351507575950825, 0.35150757595082505, 0.2878679656440357, 0.2878679656440358, 0.035, 'A'), ('l', 0.49999999999999994, 0.29000000000000004, 0.49999999999999994, 0.2, 0.035, 'A'), ('l', 0.648492424049175, 0.351507575950825, 0.7121320343559642, 0.2878679656440357, 0.035, 'A')]),
    ('moon', 'nature', 28, [('c', 0.5, 0.5, 0.22, 'A'), ('c', 0.58, 0.44, 0.18, 'B')]),
    ('star', 'nature', 5, [('p', [(0.5, 0.21999999999999997), (0.5705342302750968, 0.4029179606750063), (0.766295824562643, 0.4134752415750147), (0.6141267819554184, 0.5370820393249937), (0.6645798706418925, 0.7265247584249853), (0.5, 0.62), (0.3354201293581075, 0.7265247584249853), (0.3858732180445816, 0.5370820393249937), (0.23370417543735694, 0.41347524157501475), (0.42946576972490325, 0.4029179606750063)], 'A')]),
    ('rainbow', 'nature', 35, [('a', 0.5, 0.78, 0.26, 180, 360, 0.04, 'A'), ('a', 0.5, 0.78, 0.21500000000000002, 180, 360, 0.04, 'B'), ('a', 0.5, 0.78, 0.17, 180, 360, 0.04, 'D'), ('a', 0.5, 0.78, 0.125, 180, 360, 0.04, 'C')]),
    ('snowflake', 'nature', 4, [('l', 0.5, 0.5, 0.78, 0.5, 0.025, 'A'), ('l', 0.6579648611402671, 0.4137034030512435, 0.74, 0.5, 0.018, 'A'), ('l', 0.6579648611402671, 0.5862965969487566, 0.74, 0.5, 0.018, 'A'), ('l', 0.5, 0.5, 0.64, 0.7424871130596429, 0.025, 'A'), ('l', 0.6537174757879034, 0.5936532841783743, 0.62, 0.7078460969082653, 0.018, 'A'), ('l', 0.5042473853523637, 0.6799498811271308, 0.62, 0.7078460969082653, 0.018, 'A'), ('l', 0.5, 0.5, 0.36000000000000004, 0.7424871130596429, 0.025, 'A'), ('l', 0.4957526146476363, 0.6799498811271308, 0.38000000000000006, 0.7078460969082653, 0.018, 'A'), ('l', 0.34628252421209665, 0.5936532841783744, 0.38000000000000006, 0.7078460969082653, 0.018, 'A'), ('l', 0.5, 0.5, 0.21999999999999997, 0.5, 0.025, 'A'), ('l', 0.3420351388597329, 0.5862965969487566, 0.26, 0.5, 0.018, 'A'), ('l', 0.3420351388597329, 0.4137034030512435, 0.26, 0.5, 0.018, 'A'), ('l', 0.5, 0.5, 0.3599999999999999, 0.25751288694035723, 0.025, 'A'), ('l', 0.3462825242120966, 0.40634671582162574, 0.3799999999999999, 0.29215390309173483, 0.018, 'A'), ('l', 0.4957526146476362, 0.32005011887286916, 0.3799999999999999, 0.29215390309173483, 0.018, 'A'), ('l', 0.5, 0.5, 0.64, 0.2575128869403572, 0.025, 'A'), ('l', 0.5042473853523637, 0.32005011887286916, 0.62, 0.2921539030917347, 0.018, 'A'), ('l', 0.6537174757879034, 0.4063467158216257, 0.62, 0.2921539030917347, 0.018, 'A')]),
    ('snowman', 'nature', 18, [('c', 0.5, 0.68, 0.18, 'A'), ('c', 0.5, 0.38, 0.12, 'A'), ('c', 0.46, 0.35, 0.012, 'Bk'), ('c', 0.54, 0.35, 0.012, 'Bk'), ('l', 0.5, 0.38, 0.62, 0.36, 0.015, 'A'), ('l', 0.38, 0.52, 0.26, 0.44, 0.015, 'C'), ('l', 0.62, 0.52, 0.74, 0.44, 0.015, 'C'), ('c', 0.5, 0.62, 0.012, 'Bk'), ('c', 0.5, 0.7, 0.012, 'Bk')]),
    ('leaf', 'nature', 4, [('p', [(0.5, 0.18), (0.74, 0.44), (0.5, 0.82), (0.26, 0.44)], 'A'), ('l', 0.5, 0.22, 0.5, 0.78, 0.015, 'C'), ('l', 0.5, 0.4, 0.62, 0.34, 0.012, 'C'), ('l', 0.5, 0.56, 0.38, 0.5, 0.012, 'C')]),
    ('branch', 'nature', 8, [('l', 0.2, 0.62, 0.8, 0.34, 0.03, 'A'), ('l', 0.44, 0.5, 0.48, 0.38, 0.015, 'A'), ('l', 0.6, 0.42, 0.56, 0.32, 0.015, 'A')]),
    ('grass', 'nature', 6, [('l', 0.3, 0.72, 0.26, 0.44, 0.02, 'B'), ('l', 0.38, 0.72, 0.4, 0.38, 0.02, 'B'), ('l', 0.46, 0.72, 0.5, 0.42, 0.02, 'B'), ('l', 0.54, 0.72, 0.56, 0.36, 0.02, 'B'), ('l', 0.62, 0.72, 0.66, 0.44, 0.02, 'B'), ('l', 0.7, 0.72, 0.76, 0.48, 0.02, 'B'), ('l', 0.24, 0.72, 0.76, 0.72, 0.03, 'A')]),
    ('bush', 'nature', 10, [('c', 0.4, 0.56, 0.12, 'A'), ('c', 0.54, 0.5, 0.13, 'A'), ('c', 0.64, 0.58, 0.11, 'A'), ('r', 0.3, 0.6, 0.74, 0.66, 'A')]),
    ('field', 'nature', 35, [('r', 0.16, 0.44, 0.84, 0.8, 'B'), ('l', 0.16, 0.56, 0.84, 0.56, 0.015, 'A'), ('l', 0.16, 0.68, 0.84, 0.68, 0.015, 'A'), ('l', 0.3, 0.44, 0.26, 0.8, 0.012, 'A'), ('l', 0.5, 0.44, 0.5, 0.8, 0.012, 'A'), ('l', 0.7, 0.44, 0.74, 0.8, 0.012, 'A')]),
    ('beach', 'nature', 35, [('r', 0.16, 0.4, 0.84, 0.54, 'B'), ('r', 0.16, 0.54, 0.84, 0.82, 'D'), ('l', 0.16, 0.5, 0.84, 0.5, 0.015, 'W')]),
    ('island', 'nature', 24, [('e', 0.5, 0.66, 0.3, 0.1, 'D'), ('l', 0.5, 0.58, 0.46, 0.3, 0.04, 'A'), ('p', [(0.46, 0.32), (0.3, 0.22), (0.44, 0.28)], 'B'), ('p', [(0.46, 0.32), (0.36, 0.14), (0.5, 0.26)], 'B'), ('p', [(0.46, 0.32), (0.52, 0.12), (0.5, 0.28)], 'B'), ('p', [(0.46, 0.32), (0.64, 0.18), (0.5, 0.28)], 'B'), ('p', [(0.46, 0.32), (0.7, 0.3), (0.5, 0.36)], 'B')]),
    ('desert', 'nature', 35, [('e', 0.34, 0.7, 0.26, 0.09, 'A'), ('e', 0.68, 0.74, 0.22, 0.08, 'A'), ('c', 0.76, 0.34, 0.08, 'D')]),
    ('canyon', 'nature', 33, [('p', [(0.16, 0.34), (0.4, 0.34), (0.44, 0.54), (0.36, 0.54), (0.32, 0.76), (0.2, 0.76)], 'A'), ('p', [(0.84, 0.34), (0.6, 0.34), (0.56, 0.54), (0.64, 0.54), (0.68, 0.76), (0.8, 0.76)], 'B')]),
    ('cave', 'nature', 30, [('e', 0.5, 0.6, 0.3, 0.22, 'A'), ('e', 0.5, 0.64, 0.18, 0.16, 'Bk')]),
    ('volcano', 'nature', 34, [('p', [(0.24, 0.8), (0.44, 0.34), (0.48, 0.3), (0.52, 0.3), (0.56, 0.34), (0.76, 0.8)], 'A'), ('p', [(0.44, 0.34), (0.5, 0.14), (0.56, 0.34)], 'D')]),
    ('waterfall', 'nature', 33, [('r', 0.2, 0.24, 0.44, 0.84, 'A'), ('r', 0.56, 0.24, 0.8, 0.84, 'A'), ('r', 0.44, 0.3, 0.56, 0.84, 'D'), ('l', 0.47, 0.34, 0.47, 0.82, 0.012, 'W'), ('l', 0.53, 0.34, 0.53, 0.82, 0.012, 'W'), ('r', 0.36, 0.84, 0.64, 0.9, 'B')]),
    ('river', 'nature', 33, [('e', 0.5, 0.56, 0.34, 0.12, 'A'), ('l', 0.28, 0.5, 0.72, 0.5, 0.012, 'W'), ('l', 0.3, 0.6, 0.7, 0.6, 0.012, 'W')]),
    ('lake', 'nature', 33, [('e', 0.5, 0.56, 0.34, 0.14, 'A'), ('a', 0.42, 0.5, 0.06, 200, 320, 0.012, 'W'), ('a', 0.58, 0.58, 0.05, 200, 320, 0.012, 'W')]),
    ('bridge', 'nature', 34, [('l', 0.16, 0.52, 0.84, 0.52, 0.04, 'A'), ('l', 0.3, 0.52, 0.3, 0.76, 0.04, 'A'), ('l', 0.7, 0.52, 0.7, 0.76, 0.04, 'A'), ('a', 0.5, 0.52, 0.2, 0, 180, 0.025, 'A')]),
    ('tunnel', 'nature', 30, [('e', 0.5, 0.56, 0.3, 0.2, 'A'), ('e', 0.5, 0.58, 0.2, 0.14, 'Bk')]),
    ('dam', 'nature', 34, [('r', 0.24, 0.3, 0.76, 0.8, 'A'), ('r', 0.28, 0.36, 0.72, 0.44, 'B'), ('l', 0.32, 0.6, 0.32, 0.76, 0.015, 'C'), ('l', 0.44, 0.6, 0.44, 0.76, 0.015, 'C'), ('l', 0.56, 0.6, 0.56, 0.76, 0.015, 'C'), ('l', 0.68, 0.6, 0.68, 0.76, 0.015, 'C')]),
    ('lighthouse', 'structure', 30, [('r', 0.25, 0.16, 0.75, 0.86, 'A'), ('r', 0.23, 0.12, 0.77, 0.18, 'C'), ('e', 0.35, 0.28, 0.04, 0.055, 'D'), ('e', 0.475, 0.28, 0.04, 0.055, 'D'), ('e', 0.6, 0.28, 0.04, 0.055, 'D'), ('e', 0.35, 0.45000000000000007, 0.04, 0.055, 'D'), ('e', 0.475, 0.45000000000000007, 0.04, 0.055, 'D'), ('e', 0.6, 0.45000000000000007, 0.04, 0.055, 'D'), ('rr', 0.45, 0.66, 0.55, 0.86, 0.02, 'C'), ('r', 0.42, 0.05, 0.58, 0.12, 'D'), ('a', 0.42, 0.28, 0.03, 200, 340, 0.012, 'W'), ('a', 0.58, 0.28, 0.03, 200, 340, 0.012, 'W')]),
    ('windmill', 'structure', 28, [('r', 0.25, 0.16, 0.75, 0.86, 'A'), ('r', 0.23, 0.12, 0.77, 0.18, 'C'), ('e', 0.35, 0.28, 0.04, 0.055, 'D'), ('e', 0.475, 0.28, 0.04, 0.055, 'D'), ('e', 0.6, 0.28, 0.04, 0.055, 'D'), ('rr', 0.45, 0.66, 0.55, 0.86, 0.02, 'C'), ('l', 0.5, 0.22, 0.5, 0.1, 0.02, 'C'), ('p', [(0.5, 0.16), (0.6, 0.08), (0.56, 0.24)], 'A'), ('p', [(0.5, 0.16), (0.4, 0.08), (0.44, 0.24)], 'A'), ('p', [(0.5, 0.16), (0.58, 0.28), (0.44, 0.24)], 'B')]),
    ('palm_tree', 'plant', 22, [('l', 0.5, 0.84, 0.46, 0.42, 0.045, 'A'), ('p', [(0.46, 0.44), (0.28, 0.34), (0.44, 0.4)], 'B'), ('p', [(0.46, 0.44), (0.34, 0.26), (0.48, 0.38)], 'B'), ('p', [(0.46, 0.44), (0.52, 0.24), (0.5, 0.4)], 'B'), ('p', [(0.46, 0.44), (0.64, 0.28), (0.5, 0.4)], 'B'), ('p', [(0.46, 0.44), (0.7, 0.4), (0.5, 0.46)], 'B'), ('c', 0.44, 0.48, 0.03, 'C'), ('c', 0.5, 0.5, 0.03, 'C')]),
    ('oak_tree', 'plant', 30, [('l', 0.5, 0.84, 0.5, 0.52, 0.05, 'A'), ('c', 0.4, 0.42, 0.1, 'B'), ('c', 0.52, 0.34, 0.11, 'B'), ('c', 0.62, 0.44, 0.09, 'B'), ('c', 0.48, 0.5, 0.09, 'B')]),
    ('mailbox', 'street', 8, [('r', 0.475, 0.28, 0.525, 0.88, 0.02, 'C'), ('r', 0.4, 0.86, 0.6, 0.92, 'C'), ('r', 0.34, 0.14, 0.66, 0.3, 'A'), ('r', 0.4, 0.2, 0.6, 0.24, 'W')]),
    ('street_lamp', 'street', 24, [('r', 0.475, 0.28, 0.525, 0.88, 0.02, 'C'), ('r', 0.4, 0.86, 0.6, 0.92, 'C'), ('r', 0.3, 0.2, 0.7, 0.28, 'C'), ('e', 0.36, 0.24, 0.04, 0.035, 'A'), ('e', 0.5, 0.24, 0.04, 0.035, 'B'), ('e', 0.64, 0.24, 0.04, 0.035, 'D')]),
    ('bus_stop', 'street', 20, [('r', 0.475, 0.28, 0.525, 0.88, 0.02, 'C'), ('r', 0.4, 0.86, 0.6, 0.92, 'C'), ('c', 0.5, 0.16, 0.11, 'A'), ('l', 0.5, 0.1, 0.5, 0.22, 0.02, 'W'), ('l', 0.44, 0.16, 0.56, 0.16, 0.02, 'W'), ('r', 0.26, 0.44, 0.34, 0.62, 'C'), ('r', 0.66, 0.44, 0.74, 0.62, 'C'), ('r', 0.24, 0.4, 0.76, 0.46, 'C')]),
    ('trash_can', 'street', 10, [('r', 0.475, 0.28, 0.525, 0.88, 0.02, 'C'), ('r', 0.4, 0.86, 0.6, 0.92, 'C'), ('r', 0.34, 0.14, 0.66, 0.3, 'A'), ('r', 0.4, 0.2, 0.6, 0.24, 'W'), ('r', 0.36, 0.12, 0.64, 0.16, 'C')]),
    ('recycling_bin', 'street', 10, [('r', 0.475, 0.28, 0.525, 0.88, 0.02, 'C'), ('r', 0.4, 0.86, 0.6, 0.92, 'C'), ('r', 0.34, 0.14, 0.66, 0.3, 'A'), ('r', 0.4, 0.2, 0.6, 0.24, 'W'), ('p', [(0.42, 0.2), (0.48, 0.12), (0.5, 0.2)], 'Bk'), ('p', [(0.54, 0.2), (0.6, 0.12), (0.58, 0.2)], 'Bk')]),
    ('manhole', 'street', 6, [('e', 0.5, 0.54, 0.24, 0.1, 'A'), ('l', 0.36, 0.5, 0.64, 0.5, 0.012, 'C'), ('l', 0.36, 0.58, 0.64, 0.58, 0.012, 'C')]),
    ('stop_sign', 'street', 8, [('r', 0.475, 0.28, 0.525, 0.88, 0.02, 'C'), ('r', 0.4, 0.86, 0.6, 0.92, 'C'), ('c', 0.5, 0.16, 0.11, 'A'), ('l', 0.5, 0.1, 0.5, 0.22, 0.02, 'W'), ('l', 0.44, 0.16, 0.56, 0.16, 0.02, 'W'), ('p', [(0.42, 0.1), (0.58, 0.1), (0.64, 0.16), (0.64, 0.22), (0.58, 0.28), (0.42, 0.28), (0.36, 0.22), (0.36, 0.16)], 'A')]),
    ('yield_sign', 'street', 8, [('r', 0.475, 0.28, 0.525, 0.88, 0.02, 'C'), ('r', 0.4, 0.86, 0.6, 0.92, 'C'), ('c', 0.5, 0.16, 0.11, 'A'), ('l', 0.5, 0.1, 0.5, 0.22, 0.02, 'W'), ('l', 0.44, 0.16, 0.56, 0.16, 0.02, 'W'), ('p', [(0.5, 0.08), (0.64, 0.28), (0.36, 0.28)], 'D')]),
    ('speed_cam', 'street', 10, [('r', 0.475, 0.28, 0.525, 0.88, 0.02, 'C'), ('r', 0.4, 0.86, 0.6, 0.92, 'C'), ('r', 0.34, 0.16, 0.62, 0.28, 'A'), ('c', 0.56, 0.22, 0.03, 'Bk')]),
    ('barricade', 'street', 12, [('l', 0.2, 0.36, 0.8, 0.7, 0.05, 'A'), ('l', 0.2, 0.7, 0.8, 0.36, 0.05, 'A'), ('l', 0.3, 0.44, 0.3, 0.62, 0.04, 'C'), ('l', 0.7, 0.38, 0.7, 0.56, 0.04, 'C')]),
    ('traffic_cone', 'street', 8, [('p', [(0.44, 0.28), (0.56, 0.28), (0.66, 0.72), (0.34, 0.72)], 'A'), ('l', 0.38, 0.56, 0.62, 0.56, 0.025, 'W'), ('r', 0.3, 0.72, 0.7, 0.8, 'A')]),
    ('road_barrel', 'street', 10, [('r', 0.36, 0.32, 0.64, 0.76, 'A'), ('l', 0.36, 0.44, 0.64, 0.44, 0.03, 'W'), ('l', 0.36, 0.62, 0.64, 0.62, 0.03, 'W')]),
    ('fence', 'street', 16, [('l', 0.24, 0.36, 0.24, 0.76, 0.035, 'A'), ('l', 0.44, 0.36, 0.44, 0.76, 0.035, 'A'), ('l', 0.64, 0.36, 0.64, 0.76, 0.035, 'A'), ('l', 0.2, 0.44, 0.8, 0.44, 0.03, 'A'), ('l', 0.2, 0.62, 0.8, 0.62, 0.03, 'A')]),
    ('gate', 'street', 16, [('l', 0.3, 0.3, 0.3, 0.78, 0.04, 'A'), ('l', 0.7, 0.3, 0.7, 0.78, 0.04, 'A'), ('l', 0.3, 0.4, 0.7, 0.4, 0.025, 'A'), ('l', 0.3, 0.54, 0.7, 0.54, 0.025, 'A'), ('l', 0.3, 0.68, 0.7, 0.68, 0.025, 'A'), ('c', 0.5, 0.54, 0.02, 'Bk')]),
    ('doorknob', 'street', 3, [('c', 0.32, 0.5, 0.13, 'A'), ('c', 0.32, 0.5, 0.05, 'W'), ('r', 0.42, 0.47, 0.8, 0.53, 'A'), ('r', 0.66, 0.53, 0.7, 0.62, 'A'), ('r', 0.74, 0.53, 0.78, 0.64, 'A')]),
    ('door', 'structure', 24, [('rr', 0.3, 0.2, 0.7, 0.84, 0.02, 'A'), ('rr', 0.38, 0.3, 0.62, 0.54, 0.02, 'D'), ('rr', 0.38, 0.6, 0.62, 0.78, 0.02, 'D'), ('c', 0.64, 0.56, 0.02, 'Bk')]),
    ('window', 'structure', 14, [('rr', 0.28, 0.26, 0.72, 0.74, 0.02, 'A'), ('rr', 0.34, 0.32, 0.66, 0.68, 0.02, 'D'), ('l', 0.5, 0.32, 0.5, 0.68, 0.02, 'A'), ('l', 0.34, 0.5, 0.66, 0.5, 0.02, 'A')]),
    ('globe', 'household', 8, [('c', 0.5, 0.5, 0.22, 'A'), ('l', 0.5, 0.28, 0.5, 0.72, 0.015, 'C'), ('a', 0.5, 0.5, 0.22, 90, 270, 0.015, 'C'), ('a', 0.5, 0.5, 0.12, 90, 270, 0.015, 'C'), ('l', 0.28, 0.5, 0.72, 0.5, 0.015, 'C'), ('l', 0.5, 0.72, 0.5, 0.8, 0.025, 'C'), ('l', 0.42, 0.8, 0.58, 0.8, 0.025, 'C')]),
    ('telescope', 'household', 10, [('l', 0.3, 0.74, 0.7, 0.4, 0.04, 'A'), ('l', 0.44, 0.78, 0.4, 0.86, 0.03, 'C'), ('l', 0.56, 0.72, 0.6, 0.82, 0.03, 'C'), ('c', 0.72, 0.38, 0.02, 'Bk')]),
    ('microscope', 'household', 8, [('r', 0.34, 0.7, 0.66, 0.8, 'A'), ('l', 0.5, 0.7, 0.5, 0.44, 0.035, 'C'), ('l', 0.5, 0.44, 0.58, 0.34, 0.035, 'C'), ('c', 0.46, 0.52, 0.02, 'B'), ('l', 0.44, 0.62, 0.58, 0.62, 0.025, 'Bk')]),
    ('magnet', 'household', 5, [('a', 0.5, 0.54, 0.18, 180, 360, 0.09, 'A'), ('r', 0.32, 0.54, 0.41, 0.63, 'A'), ('r', 0.59, 0.54, 0.68, 0.63, 'A')]),
    ('gear', 'household', 7, [('a', 0.5, 0.5, 0.2, 0, 360, 0.08, 'A'), ('r', 0.46, 0.24, 0.54, 0.32, 'A'), ('r', 0.46, 0.68, 0.54, 0.76, 'A'), ('r', 0.24, 0.46, 0.32, 0.54, 'A'), ('r', 0.68, 0.46, 0.76, 0.54, 'A'), ('r', 0.29, 0.31, 0.36, 0.38, 'A'), ('r', 0.64, 0.62, 0.71, 0.69, 'A'), ('r', 0.29, 0.62, 0.36, 0.69, 'A'), ('r', 0.64, 0.31, 0.71, 0.38, 'A'), ('c', 0.5, 0.5, 0.06, 'C')]),
    ('coin', 'household', 3, [('c', 0.5, 0.52, 0.18, 'A'), ('t', 0.46, 0.44, '$', 0.14, 'C')]),
    ('banknote', 'household', 5, [('rr', 0.22, 0.4, 0.78, 0.62, 0.02, 'A'), ('c', 0.5, 0.51, 0.06, 'D'), ('l', 0.28, 0.46, 0.4, 0.46, 0.012, 'C'), ('l', 0.6, 0.56, 0.72, 0.56, 0.012, 'C')]),
    ('gem', 'household', 3, [('p', [(0.34, 0.36), (0.66, 0.36), (0.76, 0.48), (0.5, 0.78), (0.24, 0.48)], 'A'), ('l', 0.34, 0.36, 0.5, 0.48, 0.012, 'C'), ('l', 0.66, 0.36, 0.5, 0.48, 0.012, 'C'), ('l', 0.5, 0.48, 0.5, 0.78, 0.012, 'C')]),
    ('feather', 'household', 4, [('p', [(0.5, 0.16), (0.66, 0.44), (0.56, 0.72), (0.44, 0.72), (0.34, 0.44)], 'A'), ('l', 0.5, 0.2, 0.5, 0.82, 0.015, 'C')]),
    ('pig', 'animal', 16, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.86, 0.46, 0.07500000000000001, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('a', 0.2, 0.45, 0.1, 180, 340, 0.03, 'A'), ('e', 0.86, 0.48, 0.035, 0.028, 'D'), ('c', 0.85, 0.47, 0.008, 'Bk'), ('c', 0.88, 0.47, 0.008, 'Bk')]),
    ('goat', 'animal', 18, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8552, 0.46, 0.07200000000000001, 0.035, 'B'), ('p', [(0.71, 0.35), (0.72, 0.18), (0.76, 0.33)], 'A'), ('p', [(0.77, 0.33), (0.81, 0.18), (0.82, 0.35)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('l', 0.23, 0.48, 0.12, 0.3, 0.025, 'A'), ('l', 0.76, 0.5, 0.72, 0.56, 0.015, 'Bk')]),
    ('camel', 'animal', 28, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8568, 0.46, 0.07300000000000001, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('l', 0.23, 0.5, 0.14, 0.42, 0.02, 'A'), ('e', 0.42, 0.36, 0.09, 0.07, 'A'), ('e', 0.56, 0.36, 0.09, 0.07, 'A')]),
    ('hedgehog', 'animal', 8, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8568, 0.46, 0.07300000000000001, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('l', 0.36, 0.4, 0.34, 0.32, 0.015, 'A'), ('l', 0.46, 0.38, 0.46, 0.28, 0.015, 'A'), ('l', 0.56, 0.38, 0.58, 0.3, 0.015, 'A'), ('l', 0.64, 0.4, 0.68, 0.32, 0.015, 'A')]),
    ('weasel', 'animal', 10, [('e', 0.46, 0.52, 0.24, 0.15, 'A'), ('e', 0.74, 0.42, 0.1, 0.09, 'A'), ('e', 0.8568, 0.46, 0.07300000000000001, 0.035, 'B'), ('p', [(0.7, 0.34), (0.74, 0.22), (0.78, 0.34)], 'A'), ('c', 0.77, 0.4, 0.012, 'Bk'), ('r', 0.27499999999999997, 0.58, 0.325, 0.8, 'A'), ('r', 0.375, 0.58, 0.42500000000000004, 0.8, 'A'), ('r', 0.525, 0.58, 0.5750000000000001, 0.8, 'A'), ('r', 0.625, 0.58, 0.675, 0.8, 'A'), ('a', 0.2, 0.45, 0.1, 180, 340, 0.03, 'A'), ('c', 0.74, 0.34, 0.03, 'Bk')]),
    ('shrimp', 'animal', 5, [('a', 0.5, 0.5, 0.18, 300, 120, 0.07, 'A'), ('l', 0.38, 0.38, 0.32, 0.3, 0.015, 'A'), ('l', 0.36, 0.42, 0.28, 0.38, 0.012, 'A'), ('c', 0.64, 0.4, 0.015, 'Bk')]),
    ('hawk', 'animal', 14, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('p', [(0.75, 0.38), (0.9, 0.4), (0.86, 0.5), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.4, 0.5), (0.55, 0.3), (0.62, 0.48)], 'B'), ('p', [(0.32, 0.5), (0.14, 0.56), (0.32, 0.6)], 'B'), ('l', 0.44, 0.66, 0.44, 0.76, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.76, 0.015, 'D')]),
    ('pigeon', 'animal', 10, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('p', [(0.75, 0.38), (0.88, 0.42), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('e', 0.46, 0.54, 0.12, 0.06, 'B'), ('p', [(0.33, 0.52), (0.24, 0.55), (0.33, 0.58)], 'B'), ('l', 0.44, 0.66, 0.44, 0.78, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.78, 0.015, 'D')]),
    ('chicken', 'animal', 10, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('p', [(0.75, 0.38), (0.88, 0.42), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.36, 0.52), (0.55, 0.62), (0.62, 0.52)], 'B'), ('p', [(0.33, 0.52), (0.24, 0.55), (0.33, 0.58)], 'B'), ('l', 0.44, 0.66, 0.44, 0.8, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.8, 0.015, 'D'), ('p', [(0.66, 0.3), (0.7, 0.22), (0.74, 0.3)], 'D')]),
    ('rooster', 'animal', 12, [('e', 0.48, 0.52, 0.17, 0.12, 'A'), ('c', 0.68, 0.4, 0.075, 'A'), ('l', 0.66, 0.33, 0.64, 0.23, 0.012, 'D'), ('l', 0.6900000000000001, 0.33, 0.68, 0.23, 0.012, 'D'), ('l', 0.72, 0.33, 0.72, 0.23, 0.012, 'D'), ('p', [(0.75, 0.38), (0.88, 0.42), (0.75, 0.45)], 'D'), ('c', 0.7, 0.38, 0.014, 'Bk'), ('p', [(0.36, 0.52), (0.55, 0.62), (0.62, 0.52)], 'B'), ('p', [(0.32, 0.48), (0.08, 0.6), (0.32, 0.6)], 'B'), ('l', 0.44, 0.66, 0.44, 0.8200000000000001, 0.015, 'D'), ('l', 0.52, 0.66, 0.52, 0.8200000000000001, 0.015, 'D'), ('p', [(0.76, 0.34), (0.8, 0.4), (0.74, 0.4)], 'D')]),
    ('pickles', 'food', 6, [('e', 0.4, 0.54, 0.07, 0.2, 'A'), ('e', 0.56, 0.56, 0.07, 0.2, 'A'), ('l', 0.38, 0.46, 0.42, 0.62, 0.012, 'C'), ('l', 0.54, 0.48, 0.58, 0.64, 0.012, 'C')]),
    ('olive', 'food', 4, [('c', 0.5, 0.54, 0.12, 'A'), ('c', 0.5, 0.54, 0.03, 'W'), ('e', 0.5, 0.4, 0.02, 0.04, 'B')]),
    ('cereal', 'food', 8, [('p', [(0.32, 0.44), (0.68, 0.44), (0.6, 0.78), (0.4, 0.78)], 'A'), ('c', 0.44, 0.42, 0.02, 'D'), ('c', 0.52, 0.4, 0.02, 'D'), ('c', 0.58, 0.44, 0.02, 'D'), ('e', 0.5, 0.36, 0.1, 0.04, 'W')]),
    ('syrup', 'food', 6, [('rr', 0.42, 0.34, 0.58, 0.82, 0.04, 'A'), ('r', 0.46, 0.2, 0.54, 0.34, 'A'), ('r', 0.45, 0.16, 0.55, 0.22, 'D'), ('r', 0.42, 0.5, 0.58, 0.68, 'D'), ('l', 0.5, 0.7, 0.5, 0.8, 0.03, 'A')]),
    ('ketchup', 'food', 7, [('rr', 0.42, 0.34, 0.58, 0.82, 0.04, 'A'), ('r', 0.46, 0.2, 0.54, 0.34, 'A'), ('r', 0.45, 0.16, 0.55, 0.22, 'D'), ('r', 0.42, 0.5, 0.58, 0.68, 'D'), ('p', [(0.46, 0.16), (0.54, 0.16), (0.5, 0.06)], 'A'), ('r', 0.42, 0.46, 0.58, 0.62, 'D')]),
    ('mustard', 'food', 7, [('rr', 0.42, 0.34, 0.58, 0.82, 0.04, 'A'), ('r', 0.46, 0.2, 0.54, 0.34, 'A'), ('r', 0.45, 0.16, 0.55, 0.22, 'D'), ('r', 0.42, 0.5, 0.58, 0.68, 'D'), ('r', 0.42, 0.46, 0.58, 0.62, 'D')]),
    ('eggplant', 'food', 8, [('e', 0.5, 0.58, 0.13, 0.2, 'A'), ('p', [(0.44, 0.38), (0.5, 0.26), (0.56, 0.38), (0.52, 0.34), (0.48, 0.34)], 'B')]),
    ('asparagus', 'food', 6, [('l', 0.4, 0.76, 0.42, 0.36, 0.025, 'A'), ('l', 0.5, 0.78, 0.5, 0.34, 0.025, 'A'), ('l', 0.6, 0.76, 0.58, 0.38, 0.025, 'A'), ('e', 0.42, 0.32, 0.02, 0.05, 'B'), ('e', 0.5, 0.3, 0.02, 0.05, 'B'), ('e', 0.58, 0.34, 0.02, 0.05, 'B')]),
    ('zucchini', 'food', 7, [('e', 0.5, 0.54, 0.1, 0.26, 'A'), ('l', 0.5, 0.28, 0.52, 0.22, 0.02, 'C')]),
    ('bell_pepper', 'food', 7, [('e', 0.5, 0.56, 0.18, 0.22, 'A'), ('l', 0.46, 0.34, 0.44, 0.24, 0.025, 'B'), ('l', 0.5, 0.32, 0.5, 0.24, 0.025, 'B'), ('l', 0.54, 0.34, 0.56, 0.24, 0.025, 'B'), ('a', 0.44, 0.54, 0.06, 200, 320, 0.015, 'W')]),
    ('mango', 'food', 7, [('e', 0.5, 0.56, 0.22, 0.16, 'A'), ('l', 0.5, 0.34, 0.53, 0.28, 0.015, 'C')]),
    ('kiwi', 'food', 6, [('e', 0.5, 0.54, 0.2, 0.14, 'C'), ('e', 0.5, 0.54, 0.13, 0.08, 'B'), ('c', 0.5, 0.54, 0.02, 'W'), ('c', 0.44, 0.52, 0.01, 'Bk'), ('c', 0.56, 0.52, 0.01, 'Bk'), ('c', 0.46, 0.58, 0.01, 'Bk'), ('c', 0.54, 0.58, 0.01, 'Bk')]),
    ('plum', 'food', 6, [('c', 0.5, 0.56, 0.24, 'A'), ('l', 0.5, 0.33, 0.53, 0.24, 0.02, 'C'), ('a', 0.42, 0.46, 0.08, 200, 300, 0.02, 'W')]),
    ('strawberry', 'food', 5, [('p', [(0.36, 0.38), (0.64, 0.38), (0.58, 0.66), (0.5, 0.78), (0.42, 0.66)], 'A'), ('p', [(0.4, 0.36), (0.46, 0.26), (0.52, 0.34), (0.58, 0.26), (0.62, 0.36)], 'B'), ('c', 0.46, 0.48, 0.01, 'W'), ('c', 0.54, 0.52, 0.01, 'W'), ('c', 0.5, 0.62, 0.01, 'W')]),
    ('blueberry', 'food', 4, [('c', 0.44, 0.54, 0.08, 'A'), ('c', 0.6, 0.5, 0.08, 'B'), ('c', 0.52, 0.64, 0.07, 'A'), ('a', 0.44, 0.5, 0.02, 200, 340, 0.01, 'W')]),
    ('melon', 'food', 12, [('c', 0.5, 0.56, 0.24, 'A'), ('l', 0.5, 0.33, 0.53, 0.24, 0.02, 'C'), ('a', 0.42, 0.46, 0.08, 200, 300, 0.02, 'W'), ('l', 0.38, 0.44, 0.62, 0.68, 0.02, 'C'), ('l', 0.62, 0.44, 0.38, 0.68, 0.02, 'C')]),
    ('cargo_ship', 'vehicle', 32, [('p', [(0.1, 0.56), (0.9, 0.56), (0.78, 0.76), (0.22, 0.76)], 'A'), ('rr', 0.24, 0.42, 0.5, 0.56, 0.01, 'B'), ('rr', 0.66, 0.42, 0.8, 0.56, 0.01, 'D'), ('r', 0.54, 0.3, 0.58, 0.42, 'C')]),
    ('tugboat', 'vehicle', 20, [('p', [(0.2, 0.56), (0.8, 0.56), (0.7, 0.72), (0.3, 0.72)], 'A'), ('rr', 0.38, 0.42, 0.62, 0.56, 0.02, 'D'), ('r', 0.52, 0.28, 0.56, 0.42, 'C')]),
    ('lifeboat', 'vehicle', 16, [('p', [(0.22, 0.54), (0.78, 0.54), (0.66, 0.7), (0.34, 0.7)], 'A'), ('l', 0.22, 0.54, 0.78, 0.54, 0.03, 'D')]),
    ('limo', 'vehicle', 28, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('l', 0.46, 0.3, 0.46, 0.44, 0.02, 'C'), ('e', 0.24, 0.36, 0.05, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('e', 0.26, 0.34, 0.04, 0.05, 'D'), ('e', 0.4, 0.34, 0.04, 0.05, 'D'), ('e', 0.54, 0.34, 0.04, 0.05, 'D'), ('e', 0.68, 0.34, 0.04, 0.05, 'D')]),
    ('convertible', 'vehicle', 23, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('l', 0.46, 0.3, 0.46, 0.44, 0.02, 'C'), ('e', 0.24, 0.36, 0.05, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D')]),
    ('double_decker', 'vehicle', 29, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('l', 0.46, 0.3, 0.46, 0.44, 0.02, 'C'), ('e', 0.24, 0.36, 0.05, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('rr', 0.12, 0.18, 0.88, 0.32, 0.02, 'A'), ('e', 0.24, 0.25, 0.04, 0.04, 'D'), ('e', 0.4, 0.25, 0.04, 0.04, 'D'), ('e', 0.56, 0.25, 0.04, 0.04, 'D'), ('e', 0.72, 0.25, 0.04, 0.04, 'D')]),
    ('golf_cart', 'vehicle', 14, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('l', 0.46, 0.3, 0.46, 0.44, 0.02, 'C'), ('e', 0.24, 0.36, 0.05, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.4, 0.2, 0.66, 0.2, 0.025, 'C'), ('l', 0.4, 0.2, 0.4, 0.32, 0.025, 'C'), ('l', 0.66, 0.2, 0.66, 0.32, 0.025, 'C')]),
    ('sedan', 'vehicle', 23, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D')]),
    ('pickup', 'vehicle', 25, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('r', 0.56, 0.34, 0.88, 0.46, 'C')]),
    ('minivan', 'vehicle', 26, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('l', 0.46, 0.3, 0.46, 0.44, 0.02, 'C'), ('e', 0.24, 0.36, 0.05, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D')]),
    ('moped', 'vehicle', 12, [('c', 0.28, 0.62, 0.09, 'Bk'), ('c', 0.72, 0.62, 0.09, 'Bk'), ('l', 0.28, 0.62, 0.5, 0.5, 0.03, 'A'), ('l', 0.5, 0.5, 0.72, 0.62, 0.03, 'A'), ('l', 0.5, 0.5, 0.46, 0.34, 0.025, 'C'), ('l', 0.4, 0.34, 0.52, 0.34, 0.03, 'C'), ('l', 0.62, 0.38, 0.72, 0.62, 0.025, 'C')]),
    ('monorail', 'vehicle', 28, [('rr', 0.1, 0.34, 0.9, 0.58, 0.03, 'A'), ('e', 0.24, 0.46, 0.05, 0.06, 'D'), ('e', 0.44, 0.46, 0.05, 0.06, 'D'), ('e', 0.64, 0.46, 0.05, 0.06, 'D'), ('l', 0.1, 0.58, 0.9, 0.58, 0.03, 'C'), ('l', 0.5, 0.24, 0.5, 0.34, 0.025, 'C')]),
    ('trolleybus', 'vehicle', 27, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('l', 0.46, 0.3, 0.46, 0.44, 0.02, 'C'), ('e', 0.24, 0.36, 0.05, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('l', 0.3, 0.22, 0.7, 0.22, 0.02, 'C'), ('l', 0.44, 0.22, 0.44, 0.3, 0.02, 'C'), ('l', 0.56, 0.22, 0.56, 0.3, 0.02, 'C')]),
    ('tanker', 'vehicle', 27, [('rr', 0.1, 0.42, 0.9, 0.62, 0.05, 'A'), ('rr', 0.16, 0.28, 0.42, 0.46, 0.04, 'A'), ('e', 0.22, 0.36, 0.045, 0.05, 'D'), ('e', 0.34, 0.36, 0.045, 0.05, 'D'), ('c', 0.24, 0.66, 0.075, 'Bk'), ('c', 0.24, 0.66, 0.035, 'D'), ('c', 0.5, 0.66, 0.075, 'Bk'), ('c', 0.5, 0.66, 0.035, 'D'), ('c', 0.76, 0.66, 0.075, 'Bk'), ('c', 0.76, 0.66, 0.035, 'D'), ('rr', 0.5, 0.32, 0.88, 0.5, 0.08, 'B')]),
    ('crowbar', 'tool', 8, [('l', 0.24, 0.68, 0.72, 0.34, 0.035, 'A'), ('a', 0.74, 0.36, 0.05, 180, 320, 0.035, 'A'), ('a', 0.22, 0.7, 0.05, 20, 140, 0.035, 'A')]),
    ('sledgehammer', 'tool', 9, [('rr', 0.36, 0.14, 0.64, 0.28, 0.03, 'C'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'B'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'B')]),
    ('desk', 'furniture', 22, [('r', 0.2, 0.42, 0.8, 0.5, 'A'), ('l', 0.26, 0.5, 0.26, 0.8, 0.035, 'C'), ('l', 0.74, 0.5, 0.74, 0.8, 0.035, 'C'), ('r', 0.56, 0.5, 0.74, 0.58, 'C')]),
    ('stool', 'furniture', 10, [('e', 0.5, 0.4, 0.16, 0.06, 'A'), ('l', 0.38, 0.44, 0.32, 0.8, 0.025, 'C'), ('l', 0.5, 0.46, 0.5, 0.82, 0.025, 'C'), ('l', 0.62, 0.44, 0.68, 0.8, 0.025, 'C')]),
    ('dresser', 'furniture', 22, [('rr', 0.26, 0.28, 0.74, 0.8, 0.02, 'A'), ('l', 0.26, 0.45, 0.74, 0.45, 0.015, 'C'), ('l', 0.26, 0.63, 0.74, 0.63, 0.015, 'C'), ('c', 0.5, 0.37, 0.02, 'Bk'), ('c', 0.5, 0.55, 0.02, 'Bk'), ('c', 0.5, 0.71, 0.02, 'Bk')]),
    ('nightstand', 'furniture', 12, [('rr', 0.32, 0.36, 0.68, 0.78, 0.02, 'A'), ('l', 0.32, 0.57, 0.68, 0.57, 0.015, 'C'), ('c', 0.5, 0.46, 0.018, 'Bk'), ('c', 0.5, 0.68, 0.018, 'Bk')]),
    ('hammock', 'furniture', 14, [('a', 0.3, 0.5, 0.2, 0, 180, 0.04, 'A'), ('a', 0.7, 0.5, 0.2, 0, 180, 0.04, 'B'), ('l', 0.24, 0.48, 0.16, 0.24, 0.03, 'C'), ('l', 0.76, 0.48, 0.84, 0.24, 0.03, 'C')]),
    ('cradle', 'furniture', 16, [('rr', 0.28, 0.44, 0.72, 0.66, 0.04, 'A'), ('rr', 0.34, 0.38, 0.5, 0.5, 0.02, 'D'), ('l', 0.32, 0.66, 0.3, 0.8, 0.03, 'C'), ('l', 0.68, 0.66, 0.7, 0.8, 0.03, 'C')]),
    ('cabinet', 'furniture', 24, [('r', 0.25, 0.16, 0.75, 0.86, 'A'), ('r', 0.23, 0.12, 0.77, 0.18, 'C'), ('rr', 0.45, 0.66, 0.55, 0.86, 0.02, 'C'), ('l', 0.5, 0.18, 0.5, 0.86, 0.02, 'C')]),
    ('tablet', 'electronics', 10, [('rr', 0.34, 0.24, 0.66, 0.76, 0.04, 'A'), ('rr', 0.38, 0.3, 0.62, 0.66, 0.02, 'D'), ('c', 0.5, 0.72, 0.015, 'C')]),
    ('projector', 'electronics', 10, [('rr', 0.28, 0.4, 0.72, 0.64, 0.04, 'A'), ('c', 0.4, 0.52, 0.08, 'D'), ('c', 0.4, 0.52, 0.04, 'W'), ('c', 0.62, 0.52, 0.02, 'Bk')]),
    ('modem', 'electronics', 8, [('rr', 0.26, 0.48, 0.74, 0.68, 0.04, 'A'), ('l', 0.34, 0.48, 0.34, 0.3, 0.025, 'C'), ('l', 0.44, 0.48, 0.44, 0.24, 0.025, 'C'), ('c', 0.64, 0.58, 0.015, 'D')]),
    ('router', 'electronics', 8, [('rr', 0.26, 0.48, 0.74, 0.68, 0.04, 'A'), ('l', 0.32, 0.48, 0.28, 0.32, 0.025, 'C'), ('l', 0.44, 0.48, 0.42, 0.28, 0.025, 'C'), ('l', 0.56, 0.48, 0.58, 0.28, 0.025, 'C'), ('l', 0.68, 0.48, 0.72, 0.32, 0.025, 'C'), ('c', 0.34, 0.58, 0.015, 'D'), ('c', 0.42, 0.58, 0.015, 'A')]),
    ('earbuds', 'electronics', 4, [('e', 0.4, 0.48, 0.07, 0.09, 'A'), ('e', 0.6, 0.48, 0.07, 0.09, 'A'), ('l', 0.4, 0.57, 0.4, 0.7, 0.015, 'A'), ('l', 0.6, 0.57, 0.6, 0.7, 0.015, 'A')]),
    ('webcam', 'electronics', 6, [('rr', 0.34, 0.36, 0.66, 0.62, 0.08, 'A'), ('c', 0.5, 0.49, 0.1, 'Bk'), ('c', 0.5, 0.49, 0.06, 'D'), ('r', 0.44, 0.62, 0.56, 0.72, 'C')]),
    ('hoodie', 'clothing', 12, [('p', [(0.3, 0.28), (0.42, 0.22), (0.58, 0.22), (0.7, 0.28), (0.64, 0.4), (0.58, 0.36), (0.58, 0.78), (0.42, 0.78), (0.42, 0.36), (0.36, 0.4)], 'A'), ('a', 0.5, 0.235, 0.05, 180, 360, 0.02, 'C'), ('a', 0.5, 0.26, 0.09, 180, 360, 0.03, 'C'), ('l', 0.47, 0.3, 0.47, 0.4, 0.015, 'Bk'), ('l', 0.53, 0.3, 0.53, 0.4, 0.015, 'Bk')]),
    ('apron', 'clothing', 10, [('p', [(0.42, 0.26), (0.58, 0.26), (0.58, 0.4), (0.7, 0.46), (0.7, 0.8), (0.3, 0.8), (0.3, 0.46), (0.42, 0.4)], 'A'), ('r', 0.4, 0.56, 0.6, 0.72, 'D')]),
    ('vest', 'clothing', 10, [('p', [(0.34, 0.26), (0.46, 0.22), (0.54, 0.22), (0.66, 0.26), (0.62, 0.8), (0.38, 0.8)], 'A'), ('l', 0.5, 0.24, 0.5, 0.8, 0.025, 'C')]),
    ('bib', 'clothing', 8, [('p', [(0.4, 0.34), (0.6, 0.34), (0.62, 0.72), (0.38, 0.72)], 'A'), ('l', 0.4, 0.4, 0.46, 0.3, 0.02, 'A'), ('l', 0.6, 0.4, 0.54, 0.3, 0.02, 'A'), ('r', 0.44, 0.48, 0.56, 0.6, 'D')]),
    ('raincoat', 'clothing', 12, [('p', [(0.4, 0.2), (0.6, 0.2), (0.58, 0.42), (0.74, 0.82), (0.26, 0.82), (0.42, 0.42)], 'A'), ('a', 0.5, 0.215, 0.05, 180, 360, 0.02, 'C'), ('a', 0.5, 0.215, 0.1, 180, 360, 0.025, 'C'), ('l', 0.5, 0.26, 0.5, 0.8, 0.015, 'C')]),
    ('golf_bag', 'sports', 12, [('rr', 0.38, 0.3, 0.62, 0.8, 0.06, 'A'), ('l', 0.46, 0.3, 0.42, 0.14, 0.02, 'C'), ('l', 0.54, 0.3, 0.58, 0.14, 0.02, 'C'), ('l', 0.5, 0.3, 0.5, 0.1, 0.02, 'C')]),
    ('baseball_glove', 'sports', 8, [('e', 0.5, 0.5, 0.18, 'A'), ('a', 0.5, 0.6, 0.1, 0, 180, 0.03, 'C')]),
    ('bowling_pin', 'sports', 8, [('p', [(0.44, 0.28), (0.56, 0.28), (0.52, 0.46), (0.6, 0.58), (0.6, 0.76), (0.4, 0.76), (0.4, 0.58), (0.48, 0.46)], 'A'), ('l', 0.45, 0.4, 0.55, 0.4, 0.015, 'D'), ('l', 0.45, 0.45, 0.55, 0.45, 0.015, 'D')]),
    ('bowling_ball', 'sports', 8, [('c', 0.5, 0.5, 0.26, 'A'), ('c', 0.42, 0.42, 0.028, 'Bk'), ('c', 0.58, 0.46, 0.028, 'Bk'), ('c', 0.48, 0.6, 0.028, 'Bk'), ('c', 0.62, 0.58, 0.028, 'Bk'), ('c', 0.4, 0.55, 0.028, 'Bk')]),
    ('boxing_glove', 'sports', 6, [('e', 0.5, 0.5, 0.16, 0.18, 'A'), ('e', 0.62, 0.58, 0.07, 0.06, 'B'), ('r', 0.4, 0.64, 0.6, 0.72, 'C')]),
    ('golf_tee', 'sports', 3, [('l', 0.5, 0.4, 0.5, 0.74, 0.03, 'A'), ('l', 0.44, 0.74, 0.56, 0.74, 0.035, 'A'), ('c', 0.5, 0.34, 0.05, 'D')]),
    ('rock', 'nature', 10, [('p', [(0.3, 0.66), (0.36, 0.46), (0.52, 0.38), (0.68, 0.48), (0.7, 0.66)], 'A'), ('l', 0.44, 0.5, 0.52, 0.62, 0.015, 'C')]),
    ('snow', 'nature', 30, [('l', 0.5, 0.5, 0.78, 0.5, 0.025, 'A'), ('l', 0.6579648611402671, 0.4137034030512435, 0.74, 0.5, 0.018, 'A'), ('l', 0.6579648611402671, 0.5862965969487566, 0.74, 0.5, 0.018, 'A'), ('l', 0.5, 0.5, 0.64, 0.7424871130596429, 0.025, 'A'), ('l', 0.6537174757879034, 0.5936532841783743, 0.62, 0.7078460969082653, 0.018, 'A'), ('l', 0.5042473853523637, 0.6799498811271308, 0.62, 0.7078460969082653, 0.018, 'A'), ('l', 0.5, 0.5, 0.36000000000000004, 0.7424871130596429, 0.025, 'A'), ('l', 0.4957526146476363, 0.6799498811271308, 0.38000000000000006, 0.7078460969082653, 0.018, 'A'), ('l', 0.34628252421209665, 0.5936532841783744, 0.38000000000000006, 0.7078460969082653, 0.018, 'A'), ('l', 0.5, 0.5, 0.21999999999999997, 0.5, 0.025, 'A'), ('l', 0.3420351388597329, 0.5862965969487566, 0.26, 0.5, 0.018, 'A'), ('l', 0.3420351388597329, 0.4137034030512435, 0.26, 0.5, 0.018, 'A'), ('l', 0.5, 0.5, 0.3599999999999999, 0.25751288694035723, 0.025, 'A'), ('l', 0.3462825242120966, 0.40634671582162574, 0.3799999999999999, 0.29215390309173483, 0.018, 'A'), ('l', 0.4957526146476362, 0.32005011887286916, 0.3799999999999999, 0.29215390309173483, 0.018, 'A'), ('l', 0.5, 0.5, 0.64, 0.2575128869403572, 0.025, 'A'), ('l', 0.5042473853523637, 0.32005011887286916, 0.62, 0.2921539030917347, 0.018, 'A'), ('l', 0.6537174757879034, 0.4063467158216257, 0.62, 0.2921539030917347, 0.018, 'A'), ('e', 0.3, 0.72, 0.16, 0.06, 'W'), ('e', 0.62, 0.74, 0.2, 0.06, 'W')]),
    ('rain', 'nature', 24, [('a', 0.36, 0.46, 0.16, 20, 160, 0.035, 'A'), ('a', 0.6, 0.4, 0.16, 20, 160, 0.035, 'A'), ('l', 0.4, 0.58, 0.38, 0.7, 0.02, 'D'), ('l', 0.52, 0.62, 0.5, 0.74, 0.02, 'D'), ('l', 0.64, 0.54, 0.62, 0.66, 0.02, 'D')]),
    ('fog', 'nature', 24, [('a', 0.4, 0.44, 0.18, 20, 160, 0.03, 'A'), ('l', 0.3, 0.56, 0.7, 0.56, 0.03, 'W'), ('l', 0.32, 0.66, 0.68, 0.66, 0.03, 'W'), ('l', 0.36, 0.76, 0.64, 0.76, 0.03, 'W')]),
    ('sidewalk', 'street', 35, [('r', 0.16, 0.4, 0.84, 0.8, 'C'), ('l', 0.32, 0.4, 0.3, 0.8, 0.015, 'W'), ('l', 0.5, 0.4, 0.5, 0.8, 0.015, 'W'), ('l', 0.68, 0.4, 0.7, 0.8, 0.015, 'W')]),
    ('speed_bump', 'street', 12, [('p', [(0.2, 0.56), (0.4, 0.44), (0.6, 0.44), (0.8, 0.56), (0.8, 0.62), (0.2, 0.62)], 'A'), ('l', 0.44, 0.47, 0.47, 0.58, 0.015, 'D'), ('l', 0.56, 0.47, 0.53, 0.58, 0.015, 'D')]),
    ('ant', 'animal', 3, [('rr', 0.42, 0.4, 0.58, 0.74, 0.06, 'A'), ('c', 0.5, 0.33, 0.06, 'C'), ('e', 0.4, 0.48, 0.05, 0.14, 'D'), ('e', 0.6, 0.48, 0.05, 0.14, 'D'), ('l', 0.42, 0.48, 0.3, 0.54, 0.012, 'Bk'), ('l', 0.42, 0.57, 0.3, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.42, 0.6599999999999999, 0.3, 0.72, 0.012, 'Bk'), ('l', 0.58, 0.48, 0.7, 0.54, 0.012, 'Bk'), ('l', 0.58, 0.57, 0.7, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.58, 0.6599999999999999, 0.7, 0.72, 0.012, 'Bk'), ('l', 0.47, 0.28, 0.42, 0.18, 0.01, 'Bk'), ('l', 0.53, 0.28, 0.58, 0.18, 0.01, 'Bk'), ('c', 0.44, 0.6, 0.05, 'Bk'), ('c', 0.62, 0.46, 0.07, 'Bk')]),
    ('hornet', 'animal', 5, [('e', 0.5, 0.55, 0.1, 0.16, 'A'), ('e', 0.5, 0.34, 0.07, 0.07, 'C'), ('e', 0.4, 0.48, 0.05, 0.14, 'D'), ('e', 0.6, 0.48, 0.05, 0.14, 'D'), ('l', 0.42, 0.48, 0.3, 0.54, 0.012, 'Bk'), ('l', 0.42, 0.57, 0.3, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.42, 0.6599999999999999, 0.3, 0.72, 0.012, 'Bk'), ('l', 0.58, 0.48, 0.7, 0.54, 0.012, 'Bk'), ('l', 0.58, 0.57, 0.7, 0.6299999999999999, 0.012, 'Bk'), ('l', 0.58, 0.6599999999999999, 0.7, 0.72, 0.012, 'Bk'), ('l', 0.47, 0.28, 0.42, 0.18, 0.01, 'Bk'), ('l', 0.53, 0.28, 0.58, 0.18, 0.01, 'Bk'), ('l', 0.42, 0.56, 0.58, 0.56, 0.03, 'Bk'), ('l', 0.42, 0.64, 0.58, 0.64, 0.03, 'Bk'), ('e', 0.5, 0.34, 0.075, 0.075, 'C')]),
    ('chameleon', 'animal', 10, [('e', 0.44, 0.52, 0.18, 0.08, 'A'), ('c', 0.68, 0.48, 0.06, 'A'), ('a', 0.26, 0.46, 0.08, 0, 260, 0.03, 'A'), ('l', 0.74, 0.46, 0.82, 0.42, 0.012, 'D'), ('c', 0.7, 0.45, 0.02, 'Bk'), ('l', 0.36, 0.6, 0.32, 0.7, 0.02, 'A'), ('l', 0.52, 0.6, 0.56, 0.7, 0.02, 'A')]),
    ('starfish', 'animal', 8, [('p', [(0.5, 0.24), (0.55, 0.42), (0.72, 0.34), (0.59, 0.5), (0.76, 0.62), (0.57, 0.6), (0.62, 0.78), (0.5, 0.66), (0.38, 0.78), (0.43, 0.6), (0.24, 0.62), (0.41, 0.5), (0.28, 0.34), (0.45, 0.42)], 'A'), ('c', 0.5, 0.52, 0.03, 'D')]),
    ('squid', 'animal', 12, [('p', [(0.42, 0.28), (0.58, 0.28), (0.54, 0.58), (0.46, 0.58)], 'A'), ('p', [(0.42, 0.28), (0.34, 0.18), (0.46, 0.24)], 'A'), ('p', [(0.58, 0.28), (0.66, 0.18), (0.54, 0.24)], 'A'), ('l', 0.46, 0.58, 0.42, 0.78, 0.02, 'B'), ('l', 0.5, 0.58, 0.5, 0.8, 0.02, 'B'), ('l', 0.54, 0.58, 0.58, 0.78, 0.02, 'B'), ('c', 0.46, 0.34, 0.018, 'Bk'), ('c', 0.54, 0.34, 0.018, 'Bk')]),
    ('salmon', 'animal', 16, [('p', [(0.25, 0.5), (0.45, 0.3), (0.72, 0.38), (0.78, 0.5), (0.72, 0.62), (0.45, 0.7), (0.25, 0.5)], 'A'), ('l', 0.26, 0.5, 0.1, 0.34, 0.025, 'B'), ('l', 0.26, 0.5, 0.1, 0.45, 0.025, 'B'), ('l', 0.26, 0.5, 0.1, 0.56, 0.025, 'B'), ('l', 0.26, 0.5, 0.1, 0.67, 0.025, 'B'), ('p', [(0.44, 0.31), (0.52, 0.18), (0.62, 0.33)], 'B'), ('c', 0.68, 0.46, 0.025, 'W'), ('c', 0.68, 0.46, 0.013, 'Bk'), ('l', 0.58, 0.38, 0.55, 0.6, 0.008, 'C'), ('l', 0.61, 0.38, 0.5800000000000001, 0.6, 0.008, 'C'), ('l', 0.6399999999999999, 0.38, 0.6100000000000001, 0.6, 0.008, 'C'), ('c', 0.44, 0.44, 0.014, 'C'), ('c', 0.52, 0.46, 0.014, 'C'), ('c', 0.6, 0.44, 0.014, 'C')]),
    ('toast', 'food', 8, [('rr', 0.28, 0.34, 0.66, 0.74, 0.06, 'A'), ('rr', 0.34, 0.4, 0.6, 0.68, 0.05, 'D'), ('c', 0.44, 0.48, 0.012, 'C'), ('c', 0.54, 0.56, 0.012, 'C')]),
    ('croissant', 'food', 8, [('a', 0.5, 0.52, 0.24, 20, 160, 0.14, 'A'), ('l', 0.36, 0.56, 0.32, 0.64, 0.04, 'A'), ('l', 0.64, 0.56, 0.68, 0.64, 0.04, 'A'), ('l', 0.42, 0.58, 0.44, 0.66, 0.02, 'C'), ('l', 0.58, 0.58, 0.56, 0.66, 0.02, 'C')]),
    ('baguette', 'food', 9, [('rr', 0.2, 0.48, 0.8, 0.6, 0.05, 'A'), ('l', 0.36, 0.5, 0.3, 0.58, 0.02, 'C'), ('l', 0.5, 0.5, 0.44, 0.58, 0.02, 'C'), ('l', 0.64, 0.5, 0.58, 0.58, 0.02, 'C')]),
    ('sausage', 'food', 7, [('e', 0.4, 0.5, 0.18, 0.07, 'A'), ('e', 0.58, 0.6, 0.18, 0.07, 'A'), ('l', 0.22, 0.5, 0.18, 0.48, 0.02, 'Bk'), ('l', 0.76, 0.6, 0.8, 0.62, 0.02, 'Bk')]),
    ('bacon', 'food', 8, [('l', 0.26, 0.42, 0.74, 0.42, 0.045, 'A'), ('l', 0.26, 0.56, 0.74, 0.56, 0.045, 'A'), ('l', 0.26, 0.7, 0.74, 0.7, 0.045, 'A'), ('l', 0.3, 0.46, 0.4, 0.42, 0.02, 'D'), ('l', 0.52, 0.6, 0.62, 0.56, 0.02, 'D'), ('l', 0.36, 0.74, 0.46, 0.7, 0.02, 'D')]),
    ('salsa', 'food', 8, [('p', [(0.26, 0.48), (0.74, 0.48), (0.62, 0.78), (0.38, 0.78)], 'A'), ('e', 0.5, 0.48, 0.23, 0.05, 'D'), ('c', 0.44, 0.46, 0.02, 'B'), ('c', 0.56, 0.47, 0.02, 'W')]),
    ('hummus', 'food', 8, [('p', [(0.26, 0.48), (0.74, 0.48), (0.62, 0.78), (0.38, 0.78)], 'A'), ('e', 0.5, 0.48, 0.23, 0.05, 'D'), ('a', 0.5, 0.48, 0.14, 40, 300, 0.015, 'Bk')]),
    ('speedboat', 'vehicle', 18, [('p', [(0.16, 0.54), (0.84, 0.54), (0.66, 0.7), (0.26, 0.7)], 'A'), ('rr', 0.4, 0.42, 0.6, 0.54, 0.03, 'D'), ('l', 0.1, 0.74, 0.24, 0.74, 0.015, 'W'), ('l', 0.14, 0.78, 0.28, 0.78, 0.015, 'W')]),
    ('rowboat', 'vehicle', 14, [('p', [(0.18, 0.52), (0.82, 0.52), (0.66, 0.68), (0.34, 0.68)], 'A'), ('l', 0.1, 0.44, 0.9, 0.58, 0.02, 'C'), ('l', 0.9, 0.44, 0.1, 0.58, 0.02, 'C')]),
    ('battleship', 'vehicle', 32, [('p', [(0.08, 0.56), (0.92, 0.56), (0.78, 0.76), (0.22, 0.76)], 'A'), ('rr', 0.36, 0.4, 0.6, 0.56, 0.02, 'C'), ('r', 0.48, 0.28, 0.36, 0.14, 'C'), ('l', 0.44, 0.34, 0.28, 0.3, 0.02, 'Bk'), ('l', 0.44, 0.4, 0.3, 0.38, 0.02, 'Bk')]),
    ('caravan', 'vehicle', 24, [('rr', 0.14, 0.36, 0.78, 0.66, 0.06, 'A'), ('e', 0.28, 0.48, 0.05, 0.06, 'D'), ('rr', 0.54, 0.44, 0.68, 0.66, 0.02, 'C'), ('c', 0.56, 0.7, 0.06, 'Bk'), ('c', 0.7, 0.7, 0.06, 'Bk')]),
    ('airship', 'vehicle', 26, [('e', 0.5, 0.42, 0.34, 0.14, 'A'), ('p', [(0.16, 0.38), (0.1, 0.42), (0.16, 0.46)], 'B'), ('rr', 0.4, 0.58, 0.6, 0.68, 0.03, 'C'), ('l', 0.46, 0.54, 0.46, 0.58, 0.015, 'C'), ('l', 0.54, 0.54, 0.54, 0.58, 0.015, 'C')]),
    ('soldering_iron', 'tool', 6, [('p', [(0.46, 0.14), (0.54, 0.14), (0.5, 0.42)], 'D'), ('rr', 0.455, 0.42, 0.545, 0.84, 0.05, 'A'), ('l', 0.42, 0.46, 0.58, 0.46, 0.03, 'A')]),
    ('armchair', 'furniture', 18, [('rr', 0.28, 0.4, 0.72, 0.7, 0.05, 'A'), ('rr', 0.22, 0.32, 0.36, 0.7, 0.05, 'B'), ('rr', 0.64, 0.32, 0.78, 0.7, 0.05, 'B'), ('rr', 0.36, 0.44, 0.64, 0.58, 0.03, 'D'), ('r', 0.32, 0.7, 0.38, 0.8, 'C'), ('r', 0.62, 0.7, 0.68, 0.8, 'C')]),
    ('chest', 'furniture', 16, [('rr', 0.26, 0.4, 0.74, 0.76, 0.03, 'A'), ('a', 0.5, 0.44, 0.22, 180, 360, 0.03, 'B'), ('r', 0.46, 0.5, 0.54, 0.6, 'D'), ('c', 0.5, 0.55, 0.015, 'Bk')]),
    ('cushion', 'furniture', 10, [('rr', 0.26, 0.3, 0.74, 0.72, 0.12, 'A'), ('c', 0.5, 0.51, 0.015, 'C'), ('l', 0.34, 0.4, 0.66, 0.4, 0.012, 'C'), ('l', 0.34, 0.62, 0.66, 0.62, 0.012, 'C')]),
    ('e_reader', 'electronics', 9, [('rr', 0.36, 0.24, 0.64, 0.76, 0.04, 'A'), ('rr', 0.4, 0.3, 0.6, 0.66, 0.02, 'W'), ('l', 0.44, 0.4, 0.56, 0.4, 0.015, 'C'), ('l', 0.44, 0.48, 0.56, 0.48, 0.015, 'C'), ('l', 0.44, 0.56, 0.52, 0.56, 0.015, 'C')]),
    ('vr_headset', 'electronics', 8, [('rr', 0.28, 0.4, 0.72, 0.64, 0.06, 'A'), ('c', 0.4, 0.52, 0.05, 'Bk'), ('c', 0.6, 0.52, 0.05, 'Bk'), ('l', 0.28, 0.44, 0.16, 0.4, 0.03, 'C'), ('l', 0.72, 0.44, 0.84, 0.4, 0.03, 'C')]),
    ('walkie_talkie', 'electronics', 6, [('rr', 0.4, 0.34, 0.6, 0.8, 0.04, 'A'), ('l', 0.44, 0.34, 0.44, 0.18, 0.02, 'C'), ('r', 0.46, 0.42, 0.54, 0.5, 'D'), ('c', 0.5, 0.6, 0.025, 'Bk'), ('l', 0.44, 0.68, 0.56, 0.68, 0.015, 'C'), ('l', 0.44, 0.73, 0.56, 0.73, 0.015, 'C')]),
    ('beanie', 'clothing', 6, [('a', 0.5, 0.56, 0.18, 180, 360, 0.07, 'A'), ('r', 0.3, 0.54, 0.7, 0.64, 'C'), ('c', 0.5, 0.36, 0.035, 'D')]),
    ('overalls', 'clothing', 10, [('p', [(0.38, 0.34), (0.62, 0.34), (0.64, 0.48), (0.6, 0.84), (0.52, 0.84), (0.5, 0.52), (0.48, 0.84), (0.4, 0.84), (0.36, 0.48)], 'A'), ('r', 0.4, 0.24, 0.6, 0.4, 'A'), ('l', 0.42, 0.24, 0.42, 0.34, 0.03, 'C'), ('l', 0.58, 0.24, 0.58, 0.34, 0.03, 'C'), ('r', 0.44, 0.38, 0.56, 0.48, 'D')]),
    ('volleyball', 'sports', 9, [('c', 0.5, 0.5, 0.26, 'A'), ('p', [(0.32, 0.5), (0.42, 0.36), (0.52, 0.5), (0.42, 0.66)], 'Bk'), ('a', 0.5, 0.5, 0.26, 200, 340, 0.02, 'Bk'), ('a', 0.5, 0.5, 0.26, 20, 160, 0.02, 'Bk')]),
    ('frisbee', 'sports', 7, [('e', 0.5, 0.52, 0.26, 0.09, 'A'), ('e', 0.5, 0.5, 0.22, 0.07, 'D'), ('a', 0.42, 0.47, 0.05, 200, 320, 0.012, 'W')]),
    ('javelin', 'sports', 9, [('l', 0.18, 0.7, 0.74, 0.32, 0.015, 'A'), ('p', [(0.74, 0.32), (0.84, 0.24), (0.76, 0.4)], 'D')]),
    ('ping_pong_ball', 'sports', 4, [('c', 0.36, 0.42, 0.08, 'W'), ('e', 0.62, 0.56, 0.14, 0.16, 'D'), ('l', 0.56, 0.68, 0.5, 0.8, 0.04, 'A')]),
    ('forest', 'nature', 34, [('p', [(0.18, 0.78), (0.3, 0.4), (0.42, 0.78)], 'A'), ('p', [(0.38, 0.78), (0.5, 0.3), (0.62, 0.78)], 'B'), ('p', [(0.58, 0.78), (0.7, 0.44), (0.82, 0.78)], 'A'), ('l', 0.3, 0.72, 0.3, 0.82, 0.03, 'C'), ('l', 0.5, 0.72, 0.5, 0.82, 0.03, 'C'), ('l', 0.7, 0.72, 0.7, 0.82, 0.03, 'C')]),
    ('bollard', 'street', 8, [('rr', 0.42, 0.34, 0.58, 0.8, 0.05, 'A'), ('a', 0.5, 0.34, 0.08, 180, 360, 0.03, 'C'), ('l', 0.42, 0.5, 0.58, 0.5, 0.025, 'D')]),
    ('envelope', 'household', 6, [('rr', 0.24, 0.34, 0.76, 0.66, 0.02, 'A'), ('p', [(0.24, 0.34), (0.5, 0.56), (0.76, 0.34)], 'D'), ('l', 0.24, 0.66, 0.42, 0.5, 0.012, 'C'), ('l', 0.76, 0.66, 0.58, 0.5, 0.012, 'C')]),
    ('photo_frame', 'household', 10, [('rr', 0.26, 0.28, 0.74, 0.72, 0.02, 'A'), ('r', 0.33, 0.35, 0.67, 0.65, 'D'), ('p', [(0.33, 0.62), (0.46, 0.48), (0.58, 0.62)], 'B'), ('c', 0.6, 0.44, 0.03, 'W')]),
    # hCaptcha object roster (ids 1000..1002): the illustrated animals hCaptcha
    # serves in grid/drag/point challenges (see the live "Flytte" drag set —
    # raccoon / rooster / red panda / boar). Red panda + boar + warthog are
    # the ones missing from the 1000-class vocabulary.
    ('red_panda', 'animal', 15, [('e', 0.44, 0.55, 0.21, 0.135, 'A'), ('e', 0.72, 0.40, 0.105, 0.095, 'A'), ('e', 0.795, 0.44, 0.055, 0.04, 'B'), ('p', [(0.655, 0.33), (0.665, 0.19), (0.715, 0.30)], 'B'), ('p', [(0.745, 0.30), (0.795, 0.19), (0.81, 0.33)], 'B'), ('l', 0.675, 0.385, 0.725, 0.375, 0.02, 'C'), ('l', 0.755, 0.375, 0.805, 0.385, 0.02, 'C'), ('c', 0.74, 0.415, 0.011, 'Bk'), ('r', 0.27, 0.62, 0.32, 0.80, 'C'), ('r', 0.36, 0.64, 0.41, 0.82, 'C'), ('r', 0.50, 0.64, 0.55, 0.82, 'C'), ('r', 0.59, 0.62, 0.64, 0.80, 'C'), ('e', 0.17, 0.52, 0.085, 0.06, 'A'), ('l', 0.115, 0.495, 0.205, 0.49, 0.016, 'C'), ('l', 0.13, 0.555, 0.22, 0.56, 0.016, 'C')]),
    ('boar', 'animal', 20, [('e', 0.44, 0.56, 0.22, 0.14, 'A'), ('e', 0.72, 0.44, 0.10, 0.085, 'A'), ('e', 0.80, 0.47, 0.05, 0.035, 'B'), ('p', [(0.66, 0.37), (0.68, 0.28), (0.72, 0.36)], 'A'), ('p', [(0.76, 0.36), (0.80, 0.28), (0.825, 0.37)], 'A'), ('c', 0.74, 0.42, 0.011, 'Bk'), ('l', 0.82, 0.455, 0.865, 0.425, 0.014, 'W'), ('l', 0.30, 0.47, 0.60, 0.42, 0.02, 'C'), ('r', 0.28, 0.64, 0.34, 0.82, 'C'), ('r', 0.38, 0.66, 0.44, 0.84, 'C'), ('r', 0.50, 0.66, 0.56, 0.84, 'C'), ('r', 0.60, 0.64, 0.66, 0.82, 'C'), ('c', 0.16, 0.55, 0.012, 'C')]),
    ('warthog', 'animal', 21, [('e', 0.44, 0.56, 0.22, 0.135, 'A'), ('e', 0.72, 0.44, 0.105, 0.09, 'A'), ('e', 0.805, 0.465, 0.055, 0.04, 'B'), ('p', [(0.655, 0.36), (0.66, 0.27), (0.71, 0.35)], 'A'), ('p', [(0.75, 0.35), (0.80, 0.27), (0.82, 0.36)], 'A'), ('c', 0.74, 0.415, 0.011, 'Bk'), ('a', 0.835, 0.465, 0.035, 180, 330, 0.014, 'W'), ('l', 0.79, 0.50, 0.80, 0.54, 0.016, 'C'), ('l', 0.30, 0.46, 0.62, 0.42, 0.024, 'C'), ('r', 0.28, 0.63, 0.34, 0.81, 'C'), ('r', 0.38, 0.65, 0.44, 0.83, 'C'), ('r', 0.50, 0.65, 0.56, 0.83, 'C'), ('r', 0.60, 0.63, 0.66, 0.81, 'C'), ('c', 0.155, 0.54, 0.011, 'C')]),
]


# ═══════════════════════════════════════════════════════════════════════════
#  Derived lookups
# ═══════════════════════════════════════════════════════════════════════════

LONGTAIL_NAMES = [t[0] for t in LONGTAIL]
LONGTAIL_RECIPE = {t[0]: t[3] for t in LONGTAIL}
LONGTAIL_CATEGORY = {t[0]: t[1] for t in LONGTAIL}
LONGTAIL_SIZE = {t[0]: t[2] for t in LONGTAIL}

# ── aliases (merged into hcaptcha_types.SYNONYMS) ─────────────────────────
ALIASES = {
    'red_panda': ["red panda", "redpanda", "little panda", "fire fox"],
    'boar': ["wild pig", "boars"],
    'warthog': ["warthogs", "wild hog"],
    'french_fries': ["fries", "chips"],
    'hotdog': ["hot dog"],
    'double_decker': ["double decker bus", "double decker"],
    'hot_air_balloon': ["hot air balloon"],
    'snowman': ["snow man"],
    'street_lamp': ["street light"],
    'fire_engine': ["fire truck"],
    'garbage_truck': ["garbage"],
    'police_car': ["police car"],
    'ice_cream': ["ice cream"],
    'light_bulb': ["bulb"],
    'tape_measure': ["tape measure", "measuring tape"],
    'snowflake': ["snow flake"],
    'traffic_cone': ["cone"],
    'stop_sign': ["stop"],
    'speed_bump': ["speed bump"],
    'speed_cam': ["speed camera"],
    'speedboat': ["speed boat"],
    'surfboard': ["surf board"],
    'snowboard': ["snow board"],
    'walkie_talkie': ["walkie talkie"],
    'ping_pong_ball': ["ping pong ball", "table tennis ball"],
    'baseball_glove': ["baseball glove"],
    'boxing_glove': ["boxing glove"],
    'solar_panel': ["solar panel", "solar panels"],
    'recycling_bin': ["recycling", "recycle bin"],
    'trash_can': ["trash can", "trash", "bin"],
    'sailboat': ["sail boat"],
    'ambulance': ["ambulance"],
    'taxi': ["taxi", "cab"],
    'van': ["van"],
    'pickup': ["pickup truck", "pickup"],
    'golf_cart': ["golf cart"],
    'lifeboat': ["life boat"],
    'caravan': ["camper", "rv"],
    'snowmobile': ["snowmobile"],
    'wheelbarrow': ["wheel barrow"],
}


def _draw_ops(d, W, H, ops, pal):
    """Draw recipe ops onto an existing Draw, scaling unit coords to W x H."""
    for op in ops:
        kind = op[0]
        try:
            if kind == "r":
                x0, y0, x1, y1, c = op[1:6]
                d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], fill=pal[c] + (255,))
            elif kind == "rr":
                x0, y0, x1, y1, rad, c = op[1:7]
                d.rounded_rectangle([x0 * W, y0 * H, x1 * W, y1 * H],
                                    radius=max(1, rad * min(W, H)),
                                    fill=pal[c] + (255,))
            elif kind == "e":
                cx, cy, rx, ry, c = op[1:6]
                d.ellipse([(cx - rx) * W, (cy - ry) * H,
                           (cx + rx) * W, (cy + ry) * H], fill=pal[c] + (255,))
            elif kind == "c":
                cx, cy, r, c = op[1:5]
                d.ellipse([(cx - r) * W, (cy - r) * H,
                           (cx + r) * W, (cy + r) * H], fill=pal[c] + (255,))
            elif kind == "p":
                pts, c = op[1], op[2]
                d.polygon([(x * W, y * H) for x, y in pts],
                          fill=pal[c] + (255,))
            elif kind == "l":
                x0, y0, x1, y1, w, c = op[1:7]
                d.line([x0 * W, y0 * H, x1 * W, y1 * H], fill=pal[c] + (255,),
                       width=max(1, int(w * min(W, H))))
            elif kind == "a":
                cx, cy, r, a0, a1, w, c = op[1:8]
                d.arc([(cx - r) * W, (cy - r) * H, (cx + r) * W, (cy + r) * H],
                      a0, a1, fill=pal[c] + (255,),
                      width=max(1, int(w * min(W, H))))
            elif kind == "t":
                x, y, s, sz, c = op[1:6]
                try:
                    from PIL import ImageFont
                    font = ImageFont.load_default(size=max(4, int(sz * min(W, H))))
                except Exception:
                    font = None
                d.text((x * W, y * H), s, font=font, fill=pal[c] + (255,))
        except Exception:
            continue


def recipe_painter(name):
    """Return a make_dataset-compatible painter fn(d, w, h, rng, mood)
    that renders `name`'s recipe into the object layer."""
    ops = LONGTAIL_RECIPE[name]

    def paint(d, w, h, rng, mood=None):
        # one palette per image (consistent A/B/C/D); hCaptcha roster
        # animals use their species-true fixed palette (see
        # PALETTE_OVERRIDES), everything else picks a random one
        base = PALETTE_OVERRIDES.get(name) \
            or PALETTES[rng.randrange(len(PALETTES))]
        pal = {"A": base[0], "B": base[1], "C": base[2], "D": base[3],
               **COLOUR_KEYS}
        _draw_ops(d, w, h, ops, pal)

    paint.__name__ = "paint_" + name
    return paint


# ── geometry / ground derived from the size rank ─────────────────────────
# size 3..35 -> fraction of the tile the object covers
def size_geometry(size, ar=1.0):
    base = 0.30 + 0.020 * size
    return (max(0.08, base * 0.78), min(0.98, base * 1.18), ar)


_GEOMETRY_OVERRIDES = {
    "river": 1.5, "lake": 1.5, "field": 1.4, "beach": 1.5,
    "rainbow": 1.4, "bridge": 1.5, "dam": 1.2, "canyon": 1.3,
    "tunnel": 1.2, "cave": 1.2, "sidewalk": 1.3, "hotdog": 1.5,
    "kayak": 1.7, "canoe": 1.7, "surfboard": 0.55, "ski": 1.6,
    "snowboard": 1.5, "baguette": 1.6, "croissant": 1.3,
    "sausage": 1.3, "bacon": 1.4, "javelin": 1.6, "pencil": 1.7,
    "ruler": 1.7, "ladder": 0.7, "fence": 1.3, "gate": 1.2,
    "snowflake": 1.0, "snow": 1.2, "rain": 1.1, "fog": 1.2,
    "moon": 1.0, "star": 1.0, "sun": 1.0, "cloud": 1.3,
    "watermelon": 1.1, "pumpkin": 1.1, "pineapple": 0.8,
    "hockey_puck": 1.6, "napkin": 1.2, "blanket": 1.2, "rug": 1.3,
    "mattress": 1.4, "sofa": 1.3, "bed": 1.4, "desk": 1.3,
    "table": 1.3, "stool": 0.8, "bench": 1.4, "barricade": 1.3,
    "ferry": 1.4, "cargo_ship": 1.5, "battleship": 1.5,
    "submarine": 1.4, "sailboat": 1.2, "yacht": 1.3, "airship": 1.4,
    "rocket": 0.6, "balloon": 0.8, "kite": 1.0, "snowman": 0.9,
    "hot_air_balloon": 0.9, "whale": 1.3, "shark": 1.4, "dolphin": 1.4,
    "snake": 1.4, "cobra": 1.0, "crocodile": 1.4, "alligator": 1.4,
    "elephant": 1.2, "rhino": 1.3, "hippo": 1.3, "whale": 1.3,
    "lighthouse": 0.8, "windmill": 0.9, "dam": 1.2, "tunnel": 1.2,
    "mountain": 1.4, "forest": 1.4, "volcano": 1.2, "waterfall": 0.9,
    "island": 1.2, "desert": 1.4, "beach": 1.5, "river": 1.5,
    "oak_tree": 0.9, "palm_tree": 0.9, "tree": 0.9, "bush": 1.2,
    "cactus": 0.8, "flower": 0.8, "grass": 1.4, "branch": 1.3,
}

_WATER_CLASSES = {
    "ferry", "yacht", "kayak", "canoe", "sailboat", "submarine", "whale",
    "dolphin", "shark", "seal", "otter", "walrus", "seahorse", "jellyfish",
    "octopus", "squid", "salmon", "shrimp", "crab", "lobster", "starfish",
    "turtle", "cargo_ship", "tugboat", "lifeboat", "speedboat", "rowboat",
    "battleship", "river", "lake", "beach",
}
_SKY_CLASSES = {
    "cloud", "sun", "moon", "star", "rainbow", "snowflake", "airship",
    "satellite", "drone", "rocket", "hot_air_balloon", "kite", "balloon",
    "helicopter", "bird", "bat", "hawk", "pigeon", "rain", "fog", "snow",
    "helicopter",
}


def longtail_ground_kind(name):
    if name in _WATER_CLASSES:
        return "water"
    if name in _SKY_CLASSES:
        return "sky"
    cat = LONGTAIL_CATEGORY.get(name)
    if cat in ("animal", "plant", "nature"):
        return "grass"
    return "road"


# ═══════════════════════════════════════════════════════════════════════════
#  Colour compounds — 54 core classes x 9 colours = 486
#
#  hCaptcha serves colour grids ("click each image containing a red car")
#  on the same binary families. A compound tile is the base object layer
#  recoloured to the target colour, keeping its shading.
# ═══════════════════════════════════════════════════════════════════════════

CORE_BASES = [
    "bus", "car", "truck", "train", "bicycle", "motorcycle", "boat",
    "airplane", "fire_hydrant", "parking_meter", "dog", "cat", "rabbit",
    "horse", "elephant", "cow", "bird", "frog", "turtle", "snail",
    "kangaroo", "hammer", "drill", "saw", "paintbrush", "wrench",
    "screwdriver", "nail", "screw", "bolt", "apple", "pizza", "table",
    "chair", "cup", "book", "clock", "umbrella", "tree", "flower", "house",
    "mountain", "boot", "zebra", "giraffe", "lion", "bear", "sheep",
    "duck", "fish", "butterfly", "banana", "guitar", "cactus",
]
assert len(CORE_BASES) == 54

COLOURS = [
    ("red", (208, 52, 44)),
    ("blue", (52, 86, 184)),
    ("green", (44, 140, 80)),
    ("yellow", (235, 190, 40)),
    ("orange", (232, 128, 36)),
    ("purple", (140, 80, 180)),
    ("brown", (122, 80, 48)),
    ("black", (44, 44, 50)),
    ("white", (240, 240, 244)),
]

COMPOUNDS = [( "%s_%s" % (col, base), base, col, rgb)
             for base in CORE_BASES for (col, rgb) in COLOURS]
COMPOUND_BASE = {n: b for n, b, c, rgb in COMPOUNDS}
COMPOUND_COLOUR = {n: c for n, b, c, rgb in COMPOUNDS}
COMPOUND_RGB = {n: rgb for n, b, c, rgb in COMPOUNDS}
COMPOUND_NAMES = [n for n, b, c, rgb in COMPOUNDS]
assert len(COMPOUNDS) == 486


def recolor_object(layer, rgb):
    """Recolour every opaque pixel of an RGBA object layer to `rgb`,
    preserving relative shading (luminance -> brightness factor)."""
    import numpy as np
    a = np.asarray(layer, dtype=np.int16)
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255.0
    f = np.clip(0.55 + 0.90 * lum, 0.28, 1.22)
    out = a.copy()
    for i in range(3):
        out[..., i] = np.clip(rgb[i] * f, 0, 255)
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def compound_painter(name):
    """Painter for a compound: render the base object (core painter if
    available, else its longtail recipe) and recolour it."""
    base = COMPOUND_BASE[name]
    rgb = COMPOUND_RGB[name]

    def paint(d, w, h, rng, mood=None):
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        try:
            import make_dataset as md
            if base in getattr(md, "PAINTERS", {}):
                md.PAINTERS[base](ld, w, h, rng, mood)
            else:  # pragma: no cover - all 54 bases have core painters
                _draw_ops(ld, w, h, LONGTAIL_RECIPE[base],
                          {"A": (150, 150, 156), "B": (200, 200, 205),
                           "C": (90, 90, 96), "D": (245, 245, 248),
                           **COLOUR_KEYS})
        except Exception:
            _draw_ops(ld, w, h, LONGTAIL_RECIPE.get(base, [
                ("c", 0.5, 0.5, 0.2, "A")]),
                {"A": (150, 150, 156), "B": (200, 200, 205),
                 "C": (90, 90, 96), "D": (245, 245, 248), **COLOUR_KEYS})
        layer = recolor_object(layer, rgb)
        d._image.paste(layer, (0, 0), layer)

    paint.__name__ = "paint_" + name
    return paint


# ═══════════════════════════════════════════════════════════════════════════
#  Synonyms for the prompt resolver
# ═══════════════════════════════════════════════════════════════════════════

def build_synonyms():
    """name -> [aliases] for the longtail + compound vocabulary.
    Merged into hcaptcha_types.SYNONYMS by the resolver setup."""
    out = {}
    for name in LONGTAIL_NAMES:
        out.setdefault(name, [])
        for a in ALIASES.get(name, []):
            if a not in out[name]:
                out[name].append(a)
    for name, base, col, rgb in COMPOUNDS:
        spaced = "%s %s" % (col, base.replace("_", " "))
        out[name] = [spaced]
    return out


TOTAL_CLASSES = 60 + len(LONGTAIL) + len(COMPOUNDS)  # = 1000


if __name__ == "__main__":
    print("longtail base classes:", len(LONGTAIL_NAMES))
    print("colour compounds    :", len(COMPOUNDS))
    print("with 60 core        :", TOTAL_CLASSES)
    rng = random.Random(11)
    # contact sheet: 10 longtail samples + 6 compound samples
    picks = LONGTAIL_NAMES[:: len(LONGTAIL_NAMES) // 10][:10]
    picks2 = [n[0] for n in COMPOUNDS if n[0] in
              ("red_car", "blue_dog", "green_truck", "yellow_bus",
               "purple_apple", "white_pizza")][:6]
    cell, n = 96, len(picks) + len(picks2)
    cols = 8
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (18, 18, 22))
    for i, name in enumerate(picks):
        layer = Image.new("RGBA", (cell, cell), (205, 210, 218, 255))
        recipe_painter(name)(ImageDraw.Draw(layer), cell, cell, rng)
        sheet.paste(layer.convert("RGB"), ((i % cols) * cell, (i // cols) * cell))
    for i, name in enumerate(picks2):
        j = len(picks) + i
        base = COMPOUND_BASE[name]
        layer = Image.new("RGBA", (cell, cell), (205, 210, 218, 255))
        try:
            import make_dataset as md
            core_layer = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
            md.PAINTERS[base](ImageDraw.Draw(core_layer), cell, cell, rng, "day")
            core_layer = recolor_object(core_layer, COMPOUND_RGB[name])
            layer.paste(core_layer, (0, 0), core_layer)
        except Exception as exc:
            print("skip", name, exc)
            continue
        sheet.paste(layer.convert("RGB"),
                    ((j % cols) * cell, (j // cols) * cell))
    sheet.save("/tmp/lt_sheet.jpg")
    print("sheet: /tmp/lt_sheet.jpg")

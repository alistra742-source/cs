#!/usr/bin/env python3
"""make_dataset.py — generate a labeled image dataset for hCaptcha image-grid
classes (bus, car, truck, train, bicycle, motorcycle, boat, airplane,
traffic light, red light, crosswalk, fire hydrant, parking meter).

The vision solver (vision_solver.py -> Ollama qwen3-vl) picks which grid tiles
match a prompt like "select all images with a bus". This script produces a
folder-per-class dataset (ImageFolder style) you can fine-tune a small vision
classifier / VLM on — fully synthetic, procedural, deterministic, no network.

Output layout:

    data/
      bus/bus_00000.jpg ... bus_00599.jpg
      car/car_00000.jpg ...
      red_light/red_light_00000.jpg ...
      manifest.jsonl     # {"image": "bus/bus_00000.jpg", "label": "bus", "class_id": 0}
      _preview.jpg       # contact sheet so you can eyeball the classes

Usage:

    python make_dataset.py                       # ~600 images per class
    python make_dataset.py --per_class 3000      # tens of thousands total
    python make_dataset.py --per_class 500 --classes bus,car,red_light
    python make_dataset.py --size 128            # 128x128 tiles (bigger = slower)

Tens of thousands: 13 classes * 3000 = 39,000 images (one command, no GPU).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# ── Classes (folder name == label) ────────────────────────────────────────
CLASSES = [
    "bus",
    "car",
    "truck",
    "train",
    "bicycle",
    "motorcycle",
    "boat",
    "airplane",
    "traffic_light",
    "red_light",
    "crosswalk",
    "fire_hydrant",
    "parking_meter",
]

# hCaptcha prompt phrase per class — handy for VLM-style prompt/answer pairs.
PROMPTS = {
    "bus": "bus",
    "car": "car",
    "truck": "truck",
    "train": "train",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "boat": "boat",
    "airplane": "airplane",
    "traffic_light": "traffic light",
    "red_light": "red traffic light",
    "crosswalk": "crosswalk",
    "fire_hydrant": "fire hydrant",
    "parking_meter": "parking meter",
}

# ── Small helpers ─────────────────────────────────────────────────────────

def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _sky(d, w, h, top, bottom):
    for y in range(h):
        d.line([(0, y), (w, y)], fill=_lerp(top, bottom, y / max(1, h - 1)))


def _wheel(d, cx, cy, r, tire=(28, 28, 32), hub=(210, 210, 214), accent=None):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tire)
    d.ellipse([cx - r * 0.62, cy - r * 0.62, cx + r * 0.62, cy + r * 0.62],
              fill=accent or hub)
    d.ellipse([cx - r * 0.22, cy - r * 0.22, cx + r * 0.22, cy + r * 0.22],
              fill=tire)


def _rounded(d, box, radius, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline,
                        width=width)


# ── Per-class object painters (design-space pixel coords) ─────────────────

def paint_bus(d, rng, pal):
    _rounded(d, (4, 8, 116, 58), 10, fill=pal["body"], outline=(20, 20, 24), width=2)
    _rounded(d, (8, 10, 112, 20), 4, fill=pal["roof"])
    # windows
    for i, x in enumerate((16, 40, 64, 88)):
        _rounded(d, (x, 22, x + 20, 40), 3, fill=pal["glass"])
    # door
    _rounded(d, (10, 30, 30, 56), 3, fill=pal["door"])
    _wheel(d, 30, 56, 11, accent=pal["hub"])
    _wheel(d, 90, 56, 11, accent=pal["hub"])
    d.ellipse([100, 44, 112, 52], fill=pal["light"])
    d.ellipse([6, 44, 18, 52], fill=pal["light"])


def paint_car(d, rng, pal):
    _rounded(d, (6, 22, 94, 52), 8, fill=pal["body"], outline=(20, 20, 24), width=2)
    d.polygon([(28, 24), (36, 10), (62, 10), (70, 24)], fill=pal["body"])
    _rounded(d, (32, 12, 58, 24), 3, fill=pal["glass"])
    _wheel(d, 28, 50, 10, accent=pal["hub"])
    _wheel(d, 72, 50, 10, accent=pal["hub"])
    d.ellipse([80, 40, 90, 47], fill=pal["light"])
    d.ellipse([8, 40, 18, 47], fill=pal["light"])


def paint_truck(d, rng, pal):
    _rounded(d, (6, 26, 44, 58), 7, fill=pal["cab"], outline=(20, 20, 24), width=2)
    _rounded(d, (10, 30, 40, 44), 3, fill=pal["glass"])
    d.rectangle((44, 14, 122, 60), fill=pal["box"], outline=(20, 20, 24), width=2)
    for x in (52, 76, 100):
        d.line([(x, 18), (x, 56)], fill=pal["panel"], width=2)
    _wheel(d, 22, 58, 11, accent=pal["hub"])
    _wheel(d, 44, 58, 11, accent=pal["hub"])
    _wheel(d, 82, 58, 11, accent=pal["hub"])
    _wheel(d, 104, 58, 11, accent=pal["hub"])
    d.ellipse([110, 40, 120, 48], fill=pal["light"])


def paint_train(d, rng, pal):
    d.polygon([(6, 30), (30, 30), (40, 20), (144, 20), (144, 56), (6, 56)],
              fill=pal["body"], outline=(20, 20, 24))
    for x in (48, 66, 84, 102, 120, 134):
        _rounded(d, (x, 26, x + 12, 42), 2, fill=pal["glass"])
    d.rectangle((10, 34, 40, 46), fill=pal["glass"], outline=(20, 20, 24))
    for cx in (30, 66, 102, 132):
        _wheel(d, cx, 56, 8, accent=pal["hub"])
    d.ellipse([136, 22, 144, 30], fill=pal["light"])
    d.line([(20, 20), (20, 10)], fill=(40, 40, 44), width=3)
    d.line([(12, 12), (28, 12)], fill=(40, 40, 44), width=3)


def paint_bicycle(d, rng, pal):
    _wheel(d, 28, 58, 18, accent=pal["frame"])
    _wheel(d, 70, 58, 18, accent=pal["frame"])
    d.line([(28, 58), (50, 34), (70, 58), (28, 58)], fill=pal["frame"], width=4)
    d.line([(50, 34), (40, 18)], fill=pal["frame"], width=4)
    d.line([(40, 18), (66, 20)], fill=pal["frame"], width=3)
    d.line([(50, 34), (52, 20)], fill=pal["frame"], width=3)
    d.line([(40, 18), (44, 12)], fill=pal["seat"], width=5)
    d.ellipse([60, 16, 70, 24], fill=pal["frame"])


def paint_motorcycle(d, rng, pal):
    _wheel(d, 28, 46, 16, accent=pal["hub"])
    _wheel(d, 66, 46, 16, accent=pal["hub"])
    d.polygon([(30, 46), (46, 30), (62, 30), (64, 46)], fill=pal["body"])
    d.line([(50, 30), (36, 16)], fill=pal["frame"], width=4)
    d.line([(36, 16), (52, 14)], fill=pal["frame"], width=4)
    d.line([(60, 30), (66, 22)], fill=pal["frame"], width=3)
    d.line([(30, 46), (28, 32)], fill=pal["frame"], width=3)


def paint_boat(d, rng, pal):
    variant = rng.randrange(2)
    d.polygon([(10, 40), (100, 40), (86, 62), (24, 62)], fill=pal["hull"],
              outline=(20, 20, 24))
    d.line([(24, 62), (10, 40)], fill=pal["hull"], width=2)
    if variant == 0:  # sailboat
        d.line([(55, 40), (55, 14)], fill=(40, 40, 44), width=3)
        d.polygon([(57, 16), (82, 38), (57, 38)], fill=pal["sail"],
                  outline=(20, 20, 24))
        d.polygon([(53, 18), (30, 38), (53, 38)], fill=pal["sail2"],
                  outline=(20, 20, 24))
    else:  # motorboat
        d.rectangle((36, 20, 64, 40), fill=pal["cab"], outline=(20, 20, 24))
        d.rectangle((40, 24, 60, 34), fill=pal["glass"], outline=(20, 20, 24))
        d.rectangle((88, 44, 102, 52), fill=pal["motor"])


def paint_airplane(d, rng, pal):
    d.ellipse((14, 24, 104, 48), fill=pal["body"], outline=(20, 20, 24), width=2)
    d.polygon([(14, 36), (6, 30), (6, 40)], fill=pal["body"])
    d.polygon([(96, 40), (110, 20), (112, 24), (96, 44)], fill=pal["wing"],
              outline=(20, 20, 24))
    d.polygon([(60, 44), (70, 60), (82, 46)], fill=pal["wing"],
              outline=(20, 20, 24))
    for x in (30, 44, 58, 72):
        d.ellipse([x, 32, x + 5, 37], fill=pal["glass"])


def _lamp(d, cx, cy, r, color, state, glow):
    # state: "lit" (bright + halo), "dim" (muted colour), "off" (dark)
    if state == "lit":
        d.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill=glow)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                  outline=(250, 250, 250))
    elif state == "dim":
        muted = tuple(int(c * 0.55) for c in color)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=muted,
                  outline=(10, 10, 12))
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 30, 34),
                  outline=(10, 10, 12))


def paint_traffic_light(d, rng, pal, force_red=False):
    _rounded(d, (22, 6, 50, 80), 8, fill=pal["housing"], outline=(18, 18, 22), width=2)
    if force_red:
        mode = "red"
    else:
        # "traffic light" = any 3-lamp signal; "red light" (separate class) is
        # the only one with the red lamp lit, so the two labels never overlap.
        mode = rng.choice(["all", "yellow", "green"])
    _lamp(d, 36, 20, 7, (235, 60, 55),
          "lit" if mode == "red" else ("dim" if mode == "all" else "off"),
          pal["redglow"])
    _lamp(d, 36, 40, 7, (235, 190, 60),
          "lit" if mode == "yellow" else ("dim" if mode == "all" else "off"),
          pal["yelglow"])
    _lamp(d, 36, 60, 7, (80, 205, 120),
          "lit" if mode == "green" else ("dim" if mode == "all" else "off"),
          pal["grnglow"])
    d.rectangle((34, 80, 38, 108), fill=pal["pole"], outline=(18, 18, 22))


def paint_red_light(d, rng, pal):
    paint_traffic_light(d, rng, pal, force_red=True)


def paint_fire_hydrant(d, rng, pal):
    _rounded(d, (18, 26, 42, 64), 6, fill=pal["body"], outline=(20, 20, 24), width=2)
    _rounded(d, (14, 16, 46, 30), 6, fill=pal["body"], outline=(20, 20, 24), width=2)
    d.ellipse([24, 10, 36, 20], fill=pal["body"], outline=(20, 20, 24))
    d.rectangle((6, 34, 18, 42), fill=pal["body"], outline=(20, 20, 24))
    d.rectangle((42, 34, 54, 42), fill=pal["body"], outline=(20, 20, 24))
    _rounded(d, (12, 64, 48, 72), 4, fill=pal["body"], outline=(20, 20, 24), width=2)


def paint_parking_meter(d, rng, pal):
    d.rectangle((26, 52, 32, 92), fill=pal["pole"], outline=(18, 18, 22))
    _rounded(d, (12, 6, 46, 52), 8, fill=pal["head"], outline=(20, 20, 24), width=2)
    d.rectangle((16, 10, 42, 28), fill=pal["screen"], outline=(20, 20, 24))
    d.rectangle((18, 14, 40, 22), fill=pal["display"])
    d.ellipse([26, 38, 32, 44], fill=(20, 20, 24))
    d.rectangle((20, 92, 38, 100), fill=pal["pole"], outline=(18, 18, 22))


# ── Scene + compositing ───────────────────────────────────────────────────

_LIGHTING = {
    "day": (1.0, (150, 190, 235), (205, 220, 240)),
    "dusk": (0.72, (235, 150, 90), (250, 205, 150)),
    "night": (0.45, (25, 30, 55), (45, 55, 80)),
}


def _ground_palette(rng):
    kind = rng.choice(["road", "street", "grass"])
    if kind == "road":
        return (70, 70, 76), (95, 95, 102), True
    if kind == "street":
        return (120, 120, 126), (150, 150, 156), False
    return (95, 150, 95), (120, 175, 120), False


def _render_scene(size, rng, obj_class, obj_painter, pal):
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    light_name, sky_top, sky_bot = rng.choice(
        [("day", *_LIGHTING["day"][1:]), ("dusk", *_LIGHTING["dusk"][1:]),
         ("night", *_LIGHTING["night"][1:])]) if obj_class != "crosswalk" else \
        (("day", *_LIGHTING["day"][1:]))
    tall = obj_class in ("traffic_light", "red_light", "fire_hydrant",
                         "parking_meter")
    horizon = int(size * (rng.uniform(0.30, 0.45) if tall
                          else rng.uniform(0.55, 0.72)))
    _sky(d, size, horizon, sky_top, sky_bot)
    g_top, g_bot, road = _ground_palette(rng)
    if obj_class == "crosswalk":
        g_top, g_bot, road = (70, 70, 76), (95, 95, 102), True
    d.rectangle((0, horizon, size, size), fill=g_top)
    if road:
        d.rectangle((0, horizon, size, size), fill=g_top)
        for i in range(horizon, size, 4):
            t = (i - horizon) / max(1, size - horizon)
            d.line([(0, i), (size, i)], fill=_lerp(g_top, g_bot, t))
        lane = size // 2
        for yy in range(horizon + 4, size - 4, 10):
            d.rectangle((lane - 2, yy, lane + 2, yy + 6), fill=(215, 215, 205))

    # ── The object on a transparent layer ──
    design = {"bus": (120, 64), "car": (100, 54), "truck": (128, 76),
              "train": (150, 66), "bicycle": (96, 76), "motorcycle": (96, 62),
              "boat": (110, 76), "airplane": (128, 64), "traffic_light": (64, 112),
              "red_light": (64, 112), "fire_hydrant": (56, 76),
              "parking_meter": (56, 96)}.get(obj_class, (120, 64))
    dw, dh = design
    layer = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    obj_painter(ld, rng, pal)

    if obj_class == "crosswalk":
        # stripes painted directly onto the road band
        for i in range(6):
            x = int(size * (0.15 + i * 0.14))
            d.polygon([(x, size), (x + int(size * 0.05), size),
                       (x + int(size * 0.05) + 4, horizon),
                       (x + 4, horizon)], fill=(235, 235, 230))
        return img

    # ambient darkening for dusk/night
    amb = {"day": 1.0, "dusk": 0.72, "night": 0.45}[light_name]
    layer = ImageEnhance.Brightness(layer.convert("RGB")).enhance(amb).convert("RGBA")

    scale = rng.uniform(0.55, 0.9)
    sw = max(8, int(dw * scale))
    sh = max(8, int(dh * scale))
    layer = layer.resize((sw, sh), Image.LANCZOS)
    angle = rng.uniform(-7, 7)
    if angle:
        layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)

    # keep the (possibly rotated) sprite inside the canvas
    if layer.width > size - 6 or layer.height > size - 6:
        s = min((size - 6) / layer.width, (size - 6) / layer.height)
        layer = layer.resize((max(4, int(layer.width * s)),
                              max(4, int(layer.height * s))), Image.LANCZOS)

    if obj_class == "airplane":
        lo = layer.height
        cy = rng.randint(lo, max(lo, int(size * 0.75)))
    else:
        cy = rng.randint(max(horizon, layer.height), size)
    max_cx = max(0, size - layer.width)
    cx = rng.randint(0, max_cx) if max_cx > 0 else 0
    img.paste(layer, (cx, cy - layer.height), layer)
    return img


# ── Palette pickers ───────────────────────────────────────────────────────

def _vehicle_pal(rng, base_colors):
    body = rng.choice(base_colors)
    return {
        "body": body,
        "roof": _lerp(body, (255, 255, 255), 0.35),
        "glass": rng.choice([(35, 45, 60), (60, 80, 95), (25, 35, 48)]),
        "door": _lerp(body, (255, 255, 255), 0.18),
        "hub": rng.choice([(205, 205, 210), (180, 180, 185), (220, 220, 224)]),
        "light": rng.choice([(250, 240, 160), (255, 200, 90)]),
    }


def _bus_pal(rng):
    return _vehicle_pal(rng, [(240, 190, 40), (210, 70, 60), (70, 130, 210),
                              (90, 180, 90), (230, 120, 40)])


def _car_pal(rng):
    return _vehicle_pal(rng, [(220, 60, 60), (60, 120, 220), (70, 70, 80),
                              (240, 240, 240), (30, 40, 60), (180, 60, 180),
                              (70, 170, 150)])


def _truck_pal(rng):
    pal = _vehicle_pal(rng, [(240, 190, 40), (220, 80, 60), (70, 130, 210),
                             (220, 220, 220)])
    pal["cab"] = rng.choice([(200, 60, 60), (60, 110, 200), (230, 160, 40)])
    pal["box"] = rng.choice([(230, 230, 225), (210, 210, 205), (240, 240, 235),
                             (180, 190, 200)])
    pal["panel"] = (150, 150, 155)
    return pal


def _train_pal(rng):
    pal = _vehicle_pal(rng, [(220, 60, 60), (70, 120, 210), (90, 180, 90),
                             (220, 220, 220), (240, 190, 40)])
    pal["body"] = pal["body"]
    return pal


def _bike_pal(rng):
    return {"frame": rng.choice([(200, 60, 60), (60, 110, 200), (240, 170, 40),
                                 (60, 60, 66), (90, 180, 90)]),
            "seat": (30, 30, 34),
            "hub": (205, 205, 210)}


def _motor_pal(rng):
    return {"body": rng.choice([(200, 60, 60), (60, 110, 200), (240, 170, 40),
                                (90, 180, 90), (150, 60, 160)]),
            "frame": (40, 40, 46), "hub": (205, 205, 210)}


def _boat_pal(rng):
    return {"hull": rng.choice([(170, 90, 60), (90, 60, 40), (60, 110, 170),
                                (220, 220, 220)]),
            "sail": (240, 240, 240),
            "sail2": rng.choice([(220, 70, 60), (70, 130, 210), (240, 190, 40)]),
            "cab": (230, 230, 230),
            "glass": (60, 80, 95),
            "motor": (40, 40, 46)}


def _plane_pal(rng):
    return {"body": rng.choice([(235, 235, 240), (200, 210, 220), (240, 240, 240)]),
            "wing": rng.choice([(200, 60, 60), (60, 110, 200), (240, 170, 40),
                                (220, 220, 220)]),
            "glass": (60, 80, 95)}


def _light_pal(rng):
    return {"housing": rng.choice([(40, 40, 46), (50, 50, 56), (30, 34, 40),
                                   (60, 55, 40)]),
            "pole": rng.choice([(60, 60, 66), (90, 90, 96), (45, 45, 50)]),
            "redglow": (120, 25, 20), "yelglow": (120, 95, 25),
            "grnglow": (30, 95, 45)}


def _hydrant_pal(rng):
    body = rng.choice([(200, 55, 50), (190, 45, 40), (220, 80, 60)])
    return {"body": body}


def _meter_pal(rng):
    return {"head": rng.choice([(50, 90, 180), (70, 70, 78), (40, 40, 48),
                                (60, 60, 68)]),
            "screen": (20, 24, 30),
            "display": rng.choice([(120, 230, 140), (240, 200, 90), (230, 230, 235)]),
            "pole": (60, 60, 66)}


PAINTERS = {
    "bus": (paint_bus, _bus_pal),
    "car": (paint_car, _car_pal),
    "truck": (paint_truck, _truck_pal),
    "train": (paint_train, _train_pal),
    "bicycle": (paint_bicycle, _bike_pal),
    "motorcycle": (paint_motorcycle, _motor_pal),
    "boat": (paint_boat, _boat_pal),
    "airplane": (paint_airplane, _plane_pal),
    "traffic_light": (paint_traffic_light, _light_pal),
    "red_light": (paint_red_light, _light_pal),
    "fire_hydrant": (paint_fire_hydrant, _hydrant_pal),
    "parking_meter": (paint_parking_meter, _meter_pal),
}


def _crosswalk_pal(rng):
    return {}


# ── Main ──────────────────────────────────────────────────────────────────

def render_one(cls, index, size, seed):
    cid = CLASSES.index(cls)
    rng = random.Random(seed + index * 100_003 + cid * 7_919)
    if cls == "crosswalk":
        painter, pal_fn = (lambda d, r, p: None), _crosswalk_pal
    else:
        painter, pal_fn = PAINTERS[cls]
    pal = pal_fn(rng)
    img = _render_scene(size, rng, cls, painter, pal)
    # augmentation: exposure + colour + slight softness (mimics real tiles)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.86, 1.14))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.86, 1.16))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.7)))
    return img


def make_preview(out_dir: Path, classes, size, seed, per=3):
    cols = per
    rows = len(classes)
    tile = 72
    sheet = Image.new("RGB", (cols * tile, rows * tile), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    for ri, cls in enumerate(classes):
        for ci in range(per):
            im = render_one(cls, ci, size, seed + 777)
            im = im.resize((tile, tile), Image.LANCZOS)
            sheet.paste(im, (ci * tile, ri * tile))
        d.text((4, ri * tile + 2), cls, fill=(120, 230, 160))
    sheet.save(out_dir / "_preview.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser(description="Generate a labeled hCaptcha image dataset")
    ap.add_argument("--out", default="data")
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--per_class", type=int, default=600)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no_preview", action="store_true")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    unknown = [c for c in classes if c not in CLASSES]
    if unknown:
        ap.error(f"unknown classes: {unknown} (choose from {CLASSES})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total = 0
    for cid, cls in enumerate(classes):
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            name = f"{cls}_{i:05d}.jpg"
            img = render_one(cls, i, args.size, args.seed)
            img.save(cls_dir / name, "JPEG", quality=args.quality)
            manifest.append({"image": f"{cls}/{name}", "label": cls,
                             "class_id": cid, "prompt": PROMPTS.get(cls, cls)})
            total += 1
        print(f"  {cls:<15} {args.per_class} images")

    with open(out_dir / "manifest.jsonl", "w") as f:
        for rec in manifest:
            f.write(json.dumps(rec) + "\n")

    if not args.no_preview:
        make_preview(out_dir, classes, args.size, args.seed, per=4)

    print(f"\nDone: {total} images in {out_dir}/ ({len(classes)} classes)")
    print(f"Manifest: {out_dir}/manifest.jsonl")
    print(f"Preview:  {out_dir}/_preview.jpg")


if __name__ == "__main__":
    main()

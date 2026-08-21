#!/usr/bin/env python3
"""
synth_shapes.py — extra procedural painters for the hCaptcha offline solver.

Adds 36 object families on top of the 13 traffic classes drawn by
make_dataset.py (49 classes total): animals (parametric quadrupeds + bird,
frog, turtle, snail, kangaroo), tools, materials and household/terrain
objects. These families are what the *other* hCaptcha challenge types ask
about:

  * area_select  ("click on the animal who jumps the highest")
  * image_label_binary with a reference item
    ("pick all the things you can work on with the item shown" — a drill
    reference plus wood/wall/table tiles)
  * image_drag_drop piece/slot scenes are composed in make_challenges.py

Every painter has the same signature as make_dataset's:

    fn(draw, w, h, rng, mood)   — draw into an RGBA layer, box (0,0,w,h)

Exports
-------
EXTRA_PAINTERS   {class_name: painter}
EXTRA_GEOMETRY   {class_name: (scale_min, scale_max, aspect w/h)}
EXTRA_GROUND     {class_name: ground kind}   (defaults to "road")

Everything is deterministic given ``rng`` — make_dataset derives a stable
per-(seed, class_id, index) PRNG, so adding classes never disturbs the
renders of existing ones.
"""

import math

try:
    from PIL import ImageDraw  # noqa: F401  (painters receive a Draw already)
except ImportError:  # pragma: no cover
    import sys
    sys.stderr.write("Pillow is required:  pip install Pillow\n")
    raise

# ── local colour helpers (kept self-contained to avoid import cycles) ─────


def _c(v, lo=0, hi=255):
    return int(max(lo, min(hi, v)))


def _mix(a, b, t):
    return tuple(_c(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _shade(c, f):
    return tuple(_c(v * f) for v in c)


def _jit(rng, c, amt=14):
    return tuple(_c(v + rng.randint(-amt, amt)) for v in c)


def _i(v):
    return int(round(v))


_FUR = [(166, 122, 74), (96, 66, 44), (210, 190, 160), (120, 120, 126),
        (228, 222, 210), (70, 60, 52), (150, 100, 60), (188, 142, 96)]
_GREEN = [(88, 148, 70), (66, 128, 84), (104, 160, 70)]
_METAL = [(168, 172, 180), (128, 132, 140), (190, 194, 202)]
_WOOD = [(150, 102, 58), (128, 86, 48), (172, 122, 72)]
_RED = [(206, 54, 48), (176, 44, 40), (222, 72, 60)]
_BLUE = [(52, 96, 170), (40, 74, 140), (70, 120, 200)]
_YELLOW = [(232, 186, 52), (240, 204, 84)]
_TOOL = [(52, 96, 170), (206, 54, 48), (232, 186, 52), (58, 132, 96),
         (120, 66, 156), (222, 120, 48)]


# ── parametric quadruped ──────────────────────────────────────────────────
#
# One body plan -> dog / cat / rabbit / horse / elephant / cow, switched by
# the feature flags. Head faces right. `flags`: ears ("pointy"|"floppy"|
# "long"|"biground"|"small"), tail ("up"|"long"|"tuft"|"ball"|"thin"),
# stripes, trunk, mane, patches.


def _quadruped(d, w, h, rng, mood, coat=None, *, ears="pointy",
               tail="up", stripes=False, trunk=False, mane=False,
               patches=False, stocky=0.0):
    coat = _jit(rng, coat or rng.choice(_FUR), 16)
    dark = _shade(coat, 0.62)
    light = _mix(coat, (255, 255, 255), 0.30)

    # --- legs (behind the body) ---
    leg_h = h * (0.34 - 0.10 * stocky)
    leg_w = max(2, w * (0.075 + 0.03 * stocky))
    leg_y0 = h * (0.58 - 0.04 * stocky)
    for i, fx in enumerate((0.22, 0.36, 0.62, 0.76)):
        x = w * fx
        col = _shade(coat, 0.85) if i % 2 else coat
        d.rectangle([x - leg_w / 2, leg_y0, x + leg_w / 2, leg_y0 + leg_h],
                    fill=col)
        d.ellipse([x - leg_w / 2, leg_y0 + leg_h - leg_w * 0.7,
                   x + leg_w / 2, leg_y0 + leg_h + leg_w * 0.35], fill=dark)

    # --- body ---
    by0, by1 = h * (0.30 - 0.06 * stocky), h * (0.66 - 0.05 * stocky)
    bx0, bx1 = w * 0.16, w * 0.80
    d.ellipse([bx0, by0, bx1, by1], fill=coat, outline=_shade(coat, 0.7))
    if patches:
        for _ in range(rng.randint(2, 4)):
            px = rng.uniform(bx0 + 4, bx1 - w * 0.13)
            py = rng.uniform(by0 + 2, by1 - h * 0.10)
            pr = rng.uniform(w * 0.05, w * 0.12)
            d.ellipse([px - pr, py - pr * 0.7, px + pr, py + pr * 0.7],
                      fill=_jit(rng, (52, 44, 40), 10))
    if stripes:
        for i in range(4):
            sx = bx0 + (bx1 - bx0) * (0.22 + 0.16 * i)
            d.arc([sx - w * 0.05, by0 + 2, sx + w * 0.05, by1 - 2],
                  start=250, end=110, fill=dark, width=max(1, _i(w * 0.02)))

    # --- tail (drawn behind head end matters less; keep left) ---
    tx, ty = bx0, (by0 + by1) / 2
    if tail == "up":
        d.line([(tx, ty), (tx - w * 0.10, ty - h * 0.22)], fill=coat,
               width=max(2, _i(w * 0.045)))
    elif tail == "long":  # horse
        d.line([(tx, ty - h * 0.05), (tx - w * 0.07, ty + h * 0.30)],
               fill=dark, width=max(2, _i(w * 0.06)))
    elif tail == "tuft":
        d.line([(tx, ty), (tx - w * 0.10, ty + h * 0.16)], fill=coat,
               width=max(1, _i(w * 0.03)))
        d.ellipse([tx - w * 0.14, ty + h * 0.13, tx - w * 0.07, ty + h * 0.22],
                  fill=dark)
    elif tail == "ball":  # rabbit
        d.ellipse([tx - w * 0.10, ty - h * 0.02, tx + w * 0.02, ty + h * 0.12],
                  fill=light)
    else:  # thin
        d.line([(tx, ty), (tx - w * 0.11, ty + h * 0.10)], fill=coat,
               width=max(1, _i(w * 0.025)))

    # --- head ---
    hr = w * (0.15 + 0.02 * stocky)
    hx, hy = w * 0.82, h * (0.30 - 0.02 * stocky)
    if mane:  # horse: neck + dark mane ridge
        d.polygon([(hx - hr * 1.4, hy + hr * 0.4), (hx - hr * 0.2, hy - hr),
                   (bx1 - 4, by0 + 4)], fill=coat)
        d.line([(hx - hr * 0.9, hy - hr * 0.5), (bx1 - 4, by0 + 4)],
               fill=dark, width=max(2, _i(w * 0.05)))
    d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=coat,
              outline=_shade(coat, 0.7))
    # muzzle
    d.ellipse([hx + hr * 0.30, hy + hr * 0.10, hx + hr * 1.15, hy + hr * 0.75],
              fill=light)
    d.ellipse([hx + hr * 0.92, hy + hr * 0.28, hx + hr * 1.12, hy + hr * 0.50],
              fill=(28, 24, 22))  # nose
    d.ellipse([hx - hr * 0.15, hy - hr * 0.25, hx + hr * 0.30, hy + hr * 0.10],
              fill=(24, 22, 20))  # eye

    # --- ears ---
    ex = hx - hr * 0.30
    ey = hy - hr
    if ears == "pointy":
        for k in (0, 1):
            bx = ex + k * hr * 0.75
            d.polygon([(bx - hr * 0.34, ey + hr * 0.25),
                       (bx + hr * 0.34, ey + hr * 0.25),
                       (bx + hr * 0.05, ey - hr * 0.75)], fill=coat)
    elif ears == "floppy":
        for k in (0, 1):
            bx = ex + k * hr * 0.8
            d.ellipse([bx - hr * 0.32, ey - hr * 0.10,
                       bx + hr * 0.32, ey + hr * 1.05], fill=dark)
    elif ears == "long":  # rabbit
        for k in (0, 1):
            bx = ex + k * hr * 0.62 - hr * 0.2
            d.ellipse([bx - hr * 0.20, ey - hr * 1.9,
                       bx + hr * 0.20, ey + hr * 0.30], fill=coat)
            d.ellipse([bx - hr * 0.09, ey - hr * 1.7,
                       bx + hr * 0.09, ey + hr * 0.1], fill=(236, 170, 170))
    elif ears == "biground":  # elephant
        d.ellipse([hx - hr * 1.35, hy - hr * 0.9, hx + hr * 0.1, hy + hr * 0.9],
                  fill=_shade(coat, 0.88), outline=dark)
    else:  # small
        for k in (0, 1):
            bx = ex + k * hr * 0.7
            d.ellipse([bx - hr * 0.25, ey - hr * 0.25,
                       bx + hr * 0.25, ey + hr * 0.30], fill=coat)

    # --- trunk (elephant) ---
    if trunk:
        x0 = hx + hr * 0.55
        d.line([(x0, hy + hr * 0.35), (x0 + hr * 0.55, hy + hr * 0.9),
                (x0 + hr * 0.28, hy + hr * 1.9)], fill=coat,
               width=max(2, _i(w * 0.06)))


def paint_dog(d, w, h, rng, mood):
    _quadruped(d, w, h, rng, mood, ears="floppy", tail="up")


def paint_cat(d, w, h, rng, mood):
    _quadruped(d, w, h, rng, mood, coat=rng.choice(
        [(206, 140, 70), (96, 96, 102), (224, 216, 200), (60, 52, 48)]),
        ears="pointy", tail="up", stripes=True)


def paint_rabbit(d, w, h, rng, mood):
    _quadruped(d, w, h, rng, mood, coat=rng.choice(
        [(222, 218, 208), (150, 122, 96), (110, 110, 118)]),
        ears="long", tail="ball", stocky=0.25)


def paint_horse(d, w, h, rng, mood):
    _quadruped(d, w, h, rng, mood, coat=rng.choice(
        [(128, 82, 46), (70, 54, 42), (198, 178, 150)]),
        ears="pointy", tail="long", mane=True)


def paint_elephant(d, w, h, rng, mood):
    _quadruped(d, w, h, rng, mood, coat=rng.choice(
        [(146, 148, 156), (128, 130, 140)]),
        ears="biground", tail="thin", trunk=True, stocky=0.55)


def paint_cow(d, w, h, rng, mood):
    _quadruped(d, w, h, rng, mood, coat=rng.choice(
        [(226, 222, 212), (232, 228, 220)]),
        ears="small", tail="tuft", patches=True, stocky=0.4)


# ── other animals ─────────────────────────────────────────────────────────


def paint_bird(d, w, h, rng, mood):
    body = _jit(rng, rng.choice(_BLUE + _RED + [(232, 174, 44)]), 12)
    dark = _shade(body, 0.6)
    # body + head
    d.ellipse([w * 0.22, h * 0.42, w * 0.72, h * 0.74], fill=body, outline=dark)
    d.ellipse([w * 0.58, h * 0.26, w * 0.88, h * 0.56], fill=body, outline=dark)
    # belly
    d.ellipse([w * 0.32, h * 0.52, w * 0.62, h * 0.72],
              fill=_mix(body, (255, 255, 255), 0.45))
    # wing
    d.pieslice([w * 0.30, h * 0.40, w * 0.66, h * 0.80], start=200, end=340,
               fill=dark)
    # beak + eye
    d.polygon([(w * 0.86, h * 0.36), (w * 0.99, h * 0.42), (w * 0.86, h * 0.47)],
              fill=(236, 168, 52))
    d.ellipse([w * 0.70, h * 0.34, w * 0.78, h * 0.42], fill=(20, 20, 24))
    # tail feathers
    for k in range(3):
        d.line([(w * 0.26, h * 0.55), (w * (0.06 - 0.01 * k), h * (0.44 + 0.09 * k))],
               fill=dark, width=max(1, _i(h * 0.035)))
    # legs
    for fx in (0.42, 0.54):
        d.line([(w * fx, h * 0.74), (w * fx, h * 0.92)], fill=(90, 70, 40),
               width=max(1, _i(w * 0.02)))


def paint_frog(d, w, h, rng, mood):
    body = _jit(rng, rng.choice(_GREEN), 12)
    dark = _shade(body, 0.6)
    d.ellipse([w * 0.16, h * 0.42, w * 0.84, h * 0.78], fill=body, outline=dark)
    # eye bumps
    for fx in (0.34, 0.66):
        x = w * fx
        d.ellipse([x - w * 0.11, h * 0.24, x + w * 0.11, h * 0.48],
                  fill=body, outline=dark)
        d.ellipse([x - w * 0.06, h * 0.30, x + w * 0.06, h * 0.43],
                  fill=(240, 240, 236))
        d.ellipse([x - w * 0.028, h * 0.33, x + w * 0.028, h * 0.40],
                  fill=(20, 20, 20))
    # mouth
    d.arc([w * 0.30, h * 0.44, w * 0.70, h * 0.72], start=20, end=160,
          fill=dark, width=max(1, _i(h * 0.02)))
    # hind legs
    for sx in (-1, 1):
        x0 = w * (0.5 + sx * 0.30)
        d.pieslice([x0 - w * 0.14, h * 0.60, x0 + w * 0.14, h * 0.95],
                   start=140 if sx < 0 else 260, end=40 if sx < 0 else 140,
                   fill=_shade(body, 0.85))
    # front legs
    for fx in (0.32, 0.68):
        d.line([(w * fx, h * 0.74), (w * fx, h * 0.93)], fill=dark,
               width=max(1, _i(w * 0.03)))


def paint_turtle(d, w, h, rng, mood):
    shell = _jit(rng, (96, 110, 62), 12)
    skin = _jit(rng, (110, 140, 80), 12)
    dark = _shade(shell, 0.55)
    # legs + tail
    for fx, fy in ((0.24, 0.72), (0.42, 0.76), (0.62, 0.76), (0.78, 0.72)):
        d.ellipse([w * fx - w * 0.06, h * fy - h * 0.05,
                   w * fx + w * 0.06, h * fy + h * 0.10], fill=skin)
    d.polygon([(w * 0.10, h * 0.62), (w * 0.0, h * 0.68), (w * 0.12, h * 0.72)],
              fill=skin)
    # head
    d.ellipse([w * 0.82, h * 0.50, w * 1.00, h * 0.70], fill=skin)
    d.ellipse([w * 0.90, h * 0.54, w * 0.945, h * 0.60], fill=(20, 20, 20))
    # dome shell
    d.chord([w * 0.14, h * 0.22, w * 0.86, h * 1.05], start=180, end=360,
            fill=shell, outline=dark, width=2)
    d.rectangle([w * 0.14, h * 0.615, w * 0.86, h * 0.68], fill=_shade(shell, 0.8))
    # shell pattern
    for i in range(3):
        x = w * (0.30 + 0.18 * i)
        d.arc([x - w * 0.10, h * 0.30, x + w * 0.10, h * 0.95],
              start=180, end=360, fill=dark, width=max(1, _i(w * 0.015)))


def paint_snail(d, w, h, rng, mood):
    shell = _jit(rng, rng.choice([(172, 120, 70), (196, 148, 92), (140, 96, 60)]), 10)
    skin = _jit(rng, (176, 166, 140), 8)
    # body
    d.rounded_rectangle([w * 0.08, h * 0.62, w * 0.95, h * 0.80],
                        radius=max(2, _i(h * 0.08)), fill=skin)
    # head + eyestalks
    d.ellipse([w * 0.84, h * 0.52, w * 1.00, h * 0.72], fill=skin)
    for fx in (0.88, 0.96):
        d.line([(w * fx, h * 0.55), (w * (fx + 0.02), h * 0.30)], fill=skin,
               width=max(1, _i(w * 0.018)))
        d.ellipse([w * (fx + 0.005), h * 0.27, w * (fx + 0.04), h * 0.33],
                  fill=(30, 28, 26))
    # spiral shell
    cx, cy, r = w * 0.45, h * 0.47, w * 0.26
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=shell,
              outline=_shade(shell, 0.55), width=2)
    rr = r * 0.9
    while rr > r * 0.15:
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], start=0, end=300,
              fill=_shade(shell, 0.6), width=max(1, _i(r * 0.09)))
        rr *= 0.72


def paint_kangaroo(d, w, h, rng, mood):
    coat = _jit(rng, rng.choice([(172, 116, 66), (146, 96, 54), (190, 140, 88)]), 10)
    dark = _shade(coat, 0.6)
    # tail (behind)
    d.polygon([(w * 0.30, h * 0.66), (w * 0.02, h * 0.92), (w * 0.16, h * 0.94),
               (w * 0.42, h * 0.72)], fill=_shade(coat, 0.9))
    # big hind leg + foot
    d.ellipse([w * 0.30, h * 0.52, w * 0.62, h * 0.82], fill=coat, outline=dark)
    d.rounded_rectangle([w * 0.42, h * 0.82, w * 0.78, h * 0.92],
                        radius=max(2, _i(h * 0.04)), fill=dark)
    # body (leaning back)
    d.ellipse([w * 0.38, h * 0.30, w * 0.70, h * 0.68], fill=coat, outline=dark)
    d.ellipse([w * 0.46, h * 0.42, w * 0.66, h * 0.66],
              fill=_mix(coat, (255, 255, 255), 0.35))
    # forearms
    d.line([(w * 0.62, h * 0.48), (w * 0.74, h * 0.60)], fill=coat,
           width=max(2, _i(w * 0.035)))
    # head
    d.ellipse([w * 0.62, h * 0.14, w * 0.86, h * 0.36], fill=coat, outline=dark)
    d.ellipse([w * 0.78, h * 0.24, w * 0.90, h * 0.34], fill=(28, 24, 22))  # nose
    d.ellipse([w * 0.70, h * 0.20, w * 0.76, h * 0.26], fill=(20, 20, 20))  # eye
    # ears
    for k in (0, 1):
        bx = w * (0.66 + 0.09 * k)
        d.ellipse([bx, h * 0.0, bx + w * 0.07, h * 0.17], fill=coat)


# ── tools ─────────────────────────────────────────────────────────────────


def paint_hammer(d, w, h, rng, mood):
    wood = _jit(rng, rng.choice(_WOOD), 10)
    head = _jit(rng, rng.choice(_METAL), 8)
    # handle
    d.rounded_rectangle([w * 0.42, h * 0.28, w * 0.58, h * 0.98],
                        radius=max(1, _i(w * 0.05)), fill=wood,
                        outline=_shade(wood, 0.6))
    # head across the top
    d.rounded_rectangle([w * 0.10, h * 0.06, w * 0.90, h * 0.30],
                        radius=max(1, _i(h * 0.05)), fill=head,
                        outline=_shade(head, 0.55))
    # claw split
    d.polygon([(w * 0.10, h * 0.06), (w * 0.04, h * 0.02), (w * 0.04, h * 0.34),
               (w * 0.10, h * 0.30)], fill=_shade(head, 0.8))
    d.rectangle([w * 0.04, h * 0.15, w * 0.12, h * 0.21],
                fill=_shade(head, 0.35))


def paint_drill(d, w, h, rng, mood):
    body = _jit(rng, rng.choice(_TOOL), 10)
    dark = _shade(body, 0.55)
    # motor housing
    d.rounded_rectangle([w * 0.14, h * 0.16, w * 0.74, h * 0.44],
                        radius=max(2, _i(h * 0.06)), fill=body, outline=dark)
    # vents
    for i in range(3):
        x = w * (0.24 + 0.09 * i)
        d.line([(x, h * 0.22), (x, h * 0.38)], fill=dark, width=max(1, _i(w * 0.02)))
    # chuck + bit
    d.rounded_rectangle([w * 0.72, h * 0.22, w * 0.84, h * 0.38],
                        radius=max(1, _i(h * 0.03)), fill=(120, 124, 132),
                        outline=(70, 72, 80))
    d.rectangle([w * 0.84, h * 0.28, w * 0.99, h * 0.32], fill=(60, 62, 70))
    # handle + battery
    d.rounded_rectangle([w * 0.30, h * 0.42, w * 0.46, h * 0.80],
                        radius=max(1, _i(w * 0.04)), fill=_shade(body, 0.8),
                        outline=dark)
    d.rounded_rectangle([w * 0.24, h * 0.78, w * 0.54, h * 0.94],
                        radius=max(1, _i(h * 0.03)), fill=dark)
    # trigger
    d.rectangle([w * 0.47, h * 0.46, w * 0.53, h * 0.58], fill=(30, 30, 34))


def paint_saw(d, w, h, rng, mood):
    steel = _jit(rng, rng.choice(_METAL), 6)
    wood = _jit(rng, rng.choice(_WOOD), 10)
    # blade with teeth along the bottom
    top, tip = h * 0.30, h * 0.52
    teeth = []
    n = 14
    for i in range(n + 1):
        x = w * 0.30 + (w * 0.62) * i / n
        teeth.append((x, tip if i % 2 else tip - h * 0.06))
    blade = [(w * 0.30, top), (w * 0.92, top)] + list(reversed(teeth))
    d.polygon(blade, fill=steel, outline=_shade(steel, 0.5))
    # handle
    d.rounded_rectangle([w * 0.04, h * 0.20, w * 0.34, h * 0.62],
                        radius=max(2, _i(w * 0.06)), fill=wood,
                        outline=_shade(wood, 0.55))
    d.rounded_rectangle([w * 0.12, h * 0.30, w * 0.26, h * 0.52],
                        radius=max(1, _i(w * 0.03)), fill=_shade(wood, 0.4))


def paint_paintbrush(d, w, h, rng, mood):
    wood = _jit(rng, rng.choice(_WOOD), 10)
    paint = _jit(rng, rng.choice(_TOOL), 12)
    # handle
    d.rounded_rectangle([w * 0.42, h * 0.02, w * 0.58, h * 0.52],
                        radius=max(1, _i(w * 0.05)), fill=wood,
                        outline=_shade(wood, 0.6))
    # ferrule
    d.rectangle([w * 0.38, h * 0.50, w * 0.62, h * 0.64],
                fill=(176, 178, 186), outline=(110, 112, 120))
    # bristles with paint
    d.polygon([(w * 0.38, h * 0.64), (w * 0.62, h * 0.64),
               (w * 0.60, h * 0.94), (w * 0.40, h * 0.94)], fill=(222, 206, 172))
    d.polygon([(w * 0.40, h * 0.80), (w * 0.60, h * 0.80),
               (w * 0.60, h * 0.94), (w * 0.40, h * 0.94)], fill=paint)
    d.ellipse([w * 0.44, h * 0.90, w * 0.56, h * 0.985], fill=paint)


def paint_wrench(d, w, h, rng, mood):
    steel = _jit(rng, rng.choice(_METAL), 6)
    dark = _shade(steel, 0.55)
    # shaft
    d.rounded_rectangle([w * 0.42, h * 0.20, w * 0.58, h * 0.96],
                        radius=max(1, _i(w * 0.05)), fill=steel, outline=dark)
    # open jaw: ring with a wedge cut out
    cx, cy, r = w * 0.5, h * 0.16, w * 0.26
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=steel, outline=dark)
    d.polygon([(cx, cy - r), (cx + r, cy - r * 0.4), (cx + r, cy + r * 0.5),
               (cx, cy + r * 0.15)], fill=(0, 0, 0, 0))
    d.ellipse([cx - r * 0.6, cy - r * 0.26, cx + r * 0.6, cy + r * 0.30],
              fill=(0, 0, 0, 0))
    # re-draw jaw as thick arc so the alpha punch stays clean
    d.arc([cx - r, cy - r, cx + r, cy + r], start=40, end=320,
          fill=steel, width=max(3, _i(r * 0.5)))
    # bottom hole
    d.ellipse([cx - r * 0.28, h * 0.80 - r * 0.28, cx + r * 0.28, h * 0.80 + r * 0.28],
              outline=dark, width=max(1, _i(r * 0.16)))


def paint_screwdriver(d, w, h, rng, mood):
    grip = _jit(rng, rng.choice(_RED + _BLUE + _YELLOW), 10)
    steel = _jit(rng, rng.choice(_METAL), 6)
    dark = _shade(grip, 0.55)
    # shaft + tip
    d.rectangle([w * 0.465, h * 0.16, w * 0.535, h * 0.60], fill=steel,
                outline=_shade(steel, 0.5))
    d.polygon([(w * 0.465, h * 0.16), (w * 0.535, h * 0.16),
               (w * 0.52, h * 0.05), (w * 0.48, h * 0.05)], fill=_shade(steel, 0.75))
    # handle with flutes
    d.rounded_rectangle([w * 0.33, h * 0.58, w * 0.67, h * 0.97],
                        radius=max(2, _i(w * 0.09)), fill=grip, outline=dark)
    for fx in (0.42, 0.5, 0.58):
        d.line([(w * fx, h * 0.62), (w * fx, h * 0.93)], fill=dark,
               width=max(1, _i(w * 0.018)))


# ── materials ─────────────────────────────────────────────────────────────


def paint_wood(d, w, h, rng, mood):
    """Stacked planks."""
    n = rng.randint(2, 3)
    for i in range(n):
        y0 = h * (0.14 + 0.27 * i)
        wood = _jit(rng, rng.choice(_WOOD), 12)
        d.rounded_rectangle([w * 0.06, y0, w * 0.94, y0 + h * 0.24],
                            radius=max(1, _i(h * 0.03)), fill=wood,
                            outline=_shade(wood, 0.55))
        for _ in range(5):
            gy = y0 + rng.uniform(0.03, 0.21) * h
            d.line([(w * 0.10, gy), (w * 0.90, gy + rng.uniform(-2, 2))],
                   fill=_shade(wood, 0.72), width=1)
        # knot
        kx = rng.uniform(w * 0.25, w * 0.8)
        ky = y0 + h * 0.12
        d.ellipse([kx - w * 0.03, ky - h * 0.025, kx + w * 0.03, ky + h * 0.025],
                  outline=_shade(wood, 0.5))


def paint_nail(d, w, h, rng, mood):
    steel = _jit(rng, rng.choice(_METAL), 6)
    dark = _shade(steel, 0.5)
    d.ellipse([w * 0.24, h * 0.04, w * 0.76, h * 0.18], fill=steel, outline=dark)
    d.rectangle([w * 0.44, h * 0.16, w * 0.56, h * 0.82], fill=steel,
                outline=dark)
    d.polygon([(w * 0.44, h * 0.82), (w * 0.56, h * 0.82),
               (w * 0.50, h * 0.97)], fill=_shade(steel, 0.8))
    # sheen
    d.line([(w * 0.47, h * 0.2), (w * 0.47, h * 0.8)], fill=(226, 228, 234))


def paint_screw(d, w, h, rng, mood):
    steel = _jit(rng, rng.choice([(150, 154, 162), (120, 124, 132)]), 6)
    dark = _shade(steel, 0.5)
    # head with slot
    d.ellipse([w * 0.26, h * 0.04, w * 0.74, h * 0.20], fill=steel, outline=dark)
    d.line([(w * 0.35, h * 0.12), (w * 0.65, h * 0.12)], fill=dark,
           width=max(1, _i(w * 0.035)))
    # threaded body
    d.polygon([(w * 0.42, h * 0.18), (w * 0.58, h * 0.18),
               (w * 0.52, h * 0.95), (w * 0.48, h * 0.95)], fill=steel,
              outline=dark)
    for i in range(6):
        y = h * (0.26 + 0.11 * i)
        d.line([(w * 0.40, y), (w * 0.60, y + h * 0.05)], fill=dark,
               width=max(1, _i(w * 0.02)))


def paint_bolt(d, w, h, rng, mood):
    steel = _jit(rng, rng.choice(_METAL), 6)
    dark = _shade(steel, 0.5)
    # hex head
    cx, cy, r = w * 0.5, h * 0.22, w * 0.30
    pts = [(cx + r * math.cos(math.radians(60 * i - 30)),
            cy + r * math.sin(math.radians(60 * i - 30))) for i in range(6)]
    d.polygon(pts, fill=steel, outline=dark)
    d.ellipse([cx - r * 0.4, cy - r * 0.36, cx + r * 0.4, cy + r * 0.36],
              outline=dark, width=max(1, _i(w * 0.02)))
    # threaded shaft
    d.rectangle([w * 0.40, h * 0.34, w * 0.60, h * 0.95], fill=steel,
                outline=dark)
    for i in range(5):
        y = h * (0.42 + 0.11 * i)
        d.line([(w * 0.38, y), (w * 0.62, y + h * 0.045)], fill=dark,
               width=max(1, _i(w * 0.02)))


def paint_wall(d, w, h, rng, mood):
    """Brick wall (a material surface you can drill / paint)."""
    mortar = _jit(rng, (196, 188, 178), 8)
    d.rectangle([0, 0, w - 1, h - 1], fill=mortar)
    brick_h = max(4, h // 5)
    rows = int(h / brick_h) + 1
    for r_i in range(rows):
        y0 = r_i * brick_h
        off = (w // 4) if r_i % 2 else 0
        for c_x in range(-1, 3):
            x0 = c_x * (w // 2) + off
            brick = _jit(rng, rng.choice(
                [(168, 84, 62), (188, 104, 78), (146, 74, 58), (178, 118, 96)]), 10)
            d.rectangle([x0 + 2, y0 + 2, x0 + w // 2 - 2, y0 + brick_h - 2],
                        fill=brick)


def paint_canvas(d, w, h, rng, mood):
    """Artist canvas on a wooden frame."""
    wood = _jit(rng, rng.choice(_WOOD), 8)
    d.rounded_rectangle([w * 0.04, h * 0.06, w * 0.96, h * 0.94],
                        radius=max(1, _i(w * 0.03)), fill=wood,
                        outline=_shade(wood, 0.5))
    d.rounded_rectangle([w * 0.10, h * 0.13, w * 0.90, h * 0.87],
                        radius=max(1, _i(w * 0.02)), fill=(240, 238, 230),
                        outline=(210, 206, 196))
    # a few paint strokes so it reads as a canvas, not a card
    for _ in range(rng.randint(2, 4)):
        col = _jit(rng, rng.choice(_TOOL), 16)
        x0 = rng.uniform(w * 0.2, w * 0.7)
        y0 = rng.uniform(h * 0.25, h * 0.7)
        d.arc([x0, y0, x0 + w * 0.25, y0 + h * 0.22],
              start=rng.randint(0, 180), end=rng.randint(190, 350),
              fill=col, width=max(2, _i(w * 0.03)))


# ── household / terrain ───────────────────────────────────────────────────


def paint_apple(d, w, h, rng, mood):
    red = _jit(rng, rng.choice(_RED + [(112, 158, 64)]), 10)
    dark = _shade(red, 0.6)
    d.ellipse([w * 0.16, h * 0.24, w * 0.56, h * 0.88], fill=red)
    d.ellipse([w * 0.44, h * 0.24, w * 0.84, h * 0.88], fill=red)
    d.ellipse([w * 0.20, h * 0.40, w * 0.80, h * 0.90], fill=red, outline=dark)
    # stem + leaf
    d.line([(w * 0.5, h * 0.28), (w * 0.54, h * 0.10)], fill=(96, 66, 40),
           width=max(1, _i(w * 0.035)))
    d.ellipse([w * 0.54, h * 0.06, w * 0.82, h * 0.22], fill=(72, 128, 60),
              outline=(48, 92, 42))
    # highlight
    d.ellipse([w * 0.30, h * 0.36, w * 0.42, h * 0.54],
              fill=_mix(red, (255, 255, 255), 0.5))


def paint_pizza(d, w, h, rng, mood):
    crust = _jit(rng, (208, 158, 92), 10)
    cheese = _jit(rng, (240, 206, 110), 8)
    cx, cy, r = w * 0.5, h * 0.5, w * 0.42
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=crust,
              outline=_shade(crust, 0.6))
    d.ellipse([cx - r * 0.82, cy - r * 0.82, cx + r * 0.82, cy + r * 0.82],
              fill=cheese, outline=_shade(cheese, 0.7))
    # pepperoni
    for _ in range(rng.randint(4, 6)):
        a = rng.uniform(0, math.tau)
        rr = rng.uniform(0.1, 0.6) * r
        px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
        pr = r * 0.13
        d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(172, 54, 44),
                  outline=(130, 38, 32))
    # a single slice cut line
    d.line([(cx, cy), (cx + r * 0.8, cy)], fill=_shade(cheese, 0.6),
           width=max(1, _i(w * 0.015)))


def paint_table(d, w, h, rng, mood):
    wood = _jit(rng, rng.choice(_WOOD + [(200, 200, 204), (90, 94, 102)]), 10)
    dark = _shade(wood, 0.55)
    d.rounded_rectangle([w * 0.04, h * 0.30, w * 0.96, h * 0.42],
                        radius=max(1, _i(h * 0.03)), fill=wood, outline=dark)
    for fx in (0.12, 0.84):
        d.rectangle([w * fx, h * 0.42, w * (fx + 0.08), h * 0.96], fill=_shade(wood, 0.85),
                    outline=dark)
    d.line([(w * 0.06, h * 0.36), (w * 0.94, h * 0.36)], fill=dark)


def paint_chair(d, w, h, rng, mood):
    col = _jit(rng, rng.choice(_TOOL + _WOOD), 10)
    dark = _shade(col, 0.55)
    # side profile
    d.rectangle([w * 0.16, h * 0.04, w * 0.26, h * 0.52], fill=col, outline=dark)  # back
    d.rectangle([w * 0.16, h * 0.50, w * 0.80, h * 0.62], fill=col, outline=dark)  # seat
    for fx in (0.20, 0.72):
        d.rectangle([w * fx, h * 0.62, w * (fx + 0.07), h * 0.96],
                    fill=_shade(col, 0.8), outline=dark)
    d.rectangle([w * 0.18, h * 0.78, w * 0.78, h * 0.84], fill=_shade(col, 0.8))


def paint_cup(d, w, h, rng, mood):
    col = _jit(rng, rng.choice(_TOOL + [(232, 232, 236)]), 10)
    dark = _shade(col, 0.6)
    d.polygon([(w * 0.22, h * 0.26), (w * 0.70, h * 0.26),
               (w * 0.64, h * 0.88), (w * 0.28, h * 0.88)], fill=col, outline=dark)
    d.ellipse([w * 0.22, h * 0.18, w * 0.70, h * 0.32], fill=_shade(col, 0.85),
              outline=dark)
    d.ellipse([w * 0.26, h * 0.21, w * 0.66, h * 0.29], fill=(88, 54, 34))  # coffee
    # handle
    d.arc([w * 0.62, h * 0.34, w * 0.95, h * 0.64], start=-80, end=100,
          fill=dark, width=max(2, _i(w * 0.05)))
    # steam
    for fx in (0.36, 0.52):
        d.arc([w * fx, h * 0.0, w * (fx + 0.12), h * 0.16], start=90, end=270,
              fill=(220, 220, 226))


def paint_book(d, w, h, rng, mood):
    """Small stack of books."""
    for i, frac in enumerate((0.62, 0.38)):
        y0 = h * (0.74 - 0.30 * i)
        cover = _jit(rng, rng.choice(_TOOL), 12)
        d.rounded_rectangle([w * frac / 2, y0, w * (1 - frac / 2), y0 + h * 0.22],
                            radius=max(1, _i(h * 0.03)), fill=cover,
                            outline=_shade(cover, 0.5))
        d.rectangle([w * frac / 2 + 4, y0 + h * 0.16, w * (1 - frac / 2) - 4,
                     y0 + h * 0.22], fill=(238, 234, 222))
        d.line([(w * frac / 2 + 8, y0 + h * 0.05), (w * (1 - frac / 2) - 8,
                y0 + h * 0.05)], fill=_shade(cover, 0.6),
               width=max(1, _i(h * 0.015)))


def paint_clock(d, w, h, rng, mood):
    rim = _jit(rng, rng.choice([(60, 62, 70)] + _WOOD), 8)
    cx, cy, r = w * 0.5, h * 0.5, w * 0.40
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rim, outline=_shade(rim, 0.5))
    d.ellipse([cx - r * 0.84, cy - r * 0.84, cx + r * 0.84, cy + r * 0.84],
              fill=(244, 242, 236), outline=(200, 196, 186))
    for k in range(12):
        a = math.radians(k * 30)
        x0 = cx + math.cos(a) * r * 0.72
        y0 = cy + math.sin(a) * r * 0.72
        d.ellipse([x0 - 1.5, y0 - 1.5, x0 + 1.5, y0 + 1.5], fill=(60, 60, 66))
    ha = rng.uniform(0, math.tau)
    ma = rng.uniform(0, math.tau)
    d.line([(cx, cy), (cx + math.cos(ha) * r * 0.45, cy + math.sin(ha) * r * 0.45)],
           fill=(30, 30, 34), width=max(2, _i(r * 0.10)))
    d.line([(cx, cy), (cx + math.cos(ma) * r * 0.68, cy + math.sin(ma) * r * 0.68)],
           fill=(30, 30, 34), width=max(1, _i(r * 0.06)))


def paint_umbrella(d, w, h, rng, mood):
    col = _jit(rng, rng.choice(_TOOL), 10)
    dark = _shade(col, 0.55)
    # canopy with scalloped lower edge
    d.pieslice([w * 0.06, h * 0.06, w * 0.94, h * 0.94], start=180, end=360,
               fill=col, outline=dark)
    for k in range(4):
        x = w * (0.17 + 0.19 * k)
        d.pieslice([x - w * 0.06, h * 0.44, x + w * 0.06, h * 0.55],
                   start=0, end=180, fill=_shade(col, 0.85), outline=dark)
    # tip + shaft + hook
    d.line([(w * 0.5, h * 0.02), (w * 0.5, h * 0.10)], fill=dark,
           width=max(1, _i(w * 0.03)))
    d.line([(w * 0.5, h * 0.48), (w * 0.5, h * 0.90)], fill=(90, 92, 100),
           width=max(2, _i(w * 0.03)))
    d.arc([w * 0.42, h * 0.82, w * 0.60, h * 0.99], start=0, end=270,
          fill=(90, 92, 100), width=max(2, _i(w * 0.03)))


def paint_tree(d, w, h, rng, mood):
    trunk = _jit(rng, rng.choice(_WOOD), 8)
    leaf = _jit(rng, rng.choice(_GREEN), 14)
    d.rectangle([w * 0.44, h * 0.55, w * 0.58, h * 0.99], fill=trunk,
                outline=_shade(trunk, 0.55))
    for _ in range(rng.randint(4, 6)):
        cx = rng.uniform(w * 0.25, w * 0.75)
        cy = rng.uniform(h * 0.12, h * 0.55)
        r = rng.uniform(w * 0.13, w * 0.22)
        d.ellipse([cx - r, cy - r * 0.85, cx + r, cy + r * 0.85],
                  fill=_jit(rng, leaf, 10), outline=_shade(leaf, 0.6))


def paint_flower(d, w, h, rng, mood):
    petal = _jit(rng, rng.choice(_RED + _YELLOW + _BLUE + [(226, 120, 170)]), 12)
    cx, cy = w * 0.5, h * 0.32
    # stem + leaves
    d.line([(cx, h * 0.42), (cx, h * 0.98)], fill=(66, 118, 58),
           width=max(2, _i(w * 0.035)))
    for sx in (-1, 1):
        d.ellipse([cx + sx * w * 0.16 - w * 0.10, h * 0.62,
                   cx + sx * w * 0.16 + w * 0.10, h * 0.74],
                  fill=_jit(rng, rng.choice(_GREEN), 8))
    # petals
    pr = w * 0.13
    for k in range(6):
        a = math.radians(k * 60)
        px, py = cx + math.cos(a) * w * 0.14, cy + math.sin(a) * w * 0.14
        d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=_jit(rng, petal, 8),
                  outline=_shade(petal, 0.6))
    d.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=(240, 200, 70),
              outline=(190, 150, 40))


def paint_house(d, w, h, rng, mood):
    wallc = _jit(rng, rng.choice([(226, 214, 196), (200, 208, 220),
                                  (222, 186, 160), (188, 200, 176)]), 10)
    roof = _jit(rng, rng.choice([(140, 62, 52), (96, 74, 60), (110, 96, 110)]), 8)
    dark = _shade(wallc, 0.6)
    d.rectangle([w * 0.14, h * 0.44, w * 0.86, h * 0.97], fill=wallc, outline=dark)
    d.polygon([(w * 0.06, h * 0.46), (w * 0.94, h * 0.46), (w * 0.5, h * 0.06)],
              fill=roof, outline=_shade(roof, 0.55))
    # chimney
    d.rectangle([w * 0.70, h * 0.12, w * 0.82, h * 0.40], fill=_shade(roof, 0.9),
                outline=_shade(roof, 0.5))
    # door + windows
    d.rectangle([w * 0.44, h * 0.66, w * 0.58, h * 0.97], fill=(110, 78, 52),
                outline=(80, 56, 38))
    for fx in (0.20, 0.66):
        d.rectangle([w * fx, h * 0.52, w * (fx + 0.14), h * 0.62],
                    fill=(170, 206, 226), outline=dark)
        d.line([(w * (fx + 0.07), h * 0.52), (w * (fx + 0.07), h * 0.62)],
               fill=dark)


def paint_mountain(d, w, h, rng, mood):
    rock = _jit(rng, (116, 106, 104), 10)
    snow = (238, 240, 246)
    # back peak
    d.polygon([(w * 0.02, h * 0.95), (w * 0.46, h * 0.22), (w * 0.82, h * 0.95)],
              fill=_shade(rock, 0.85))
    d.polygon([(w * 0.375, h * 0.38), (w * 0.46, h * 0.22), (w * 0.545, h * 0.38),
               (w * 0.46, h * 0.45)], fill=snow)
    # front peak
    d.polygon([(w * 0.28, h * 0.95), (w * 0.68, h * 0.10), (w * 1.0, h * 0.95)],
              fill=rock, outline=_shade(rock, 0.6))
    d.polygon([(w * 0.60, h * 0.28), (w * 0.68, h * 0.10), (w * 0.76, h * 0.28),
               (w * 0.68, h * 0.35)], fill=snow)


def paint_boot(d, w, h, rng, mood):
    col = _jit(rng, rng.choice([(96, 66, 44), (60, 58, 64), (150, 100, 60),
                                (140, 40, 36)]), 10)
    dark = _shade(col, 0.55)
    leg = [(w * 0.22, h * 0.04), (w * 0.52, h * 0.04), (w * 0.52, h * 0.62),
           (w * 0.80, h * 0.68), (w * 0.88, h * 0.80), (w * 0.86, h * 0.94),
           (w * 0.22, h * 0.94)]
    d.polygon(leg, fill=col, outline=dark)
    # sole
    d.rectangle([w * 0.20, h * 0.86, w * 0.88, h * 0.96], fill=(40, 38, 40))
    # laces
    for i in range(4):
        y = h * (0.52 + 0.08 * i)
        d.line([(w * 0.52, y), (w * (0.60 + 0.02 * i), y + h * 0.02)],
               fill=(230, 228, 222), width=max(1, _i(w * 0.015)))
    # pull loop
    d.line([(w * 0.24, h * 0.04), (w * 0.24, h * 0.30)], fill=dark,
           width=max(1, _i(w * 0.03)))


# ── NEW CLASSES: More animals, foods, vehicles, objects ───────────────────


def paint_zebra(d, w, h, rng, mood):
    """Zebra with distinctive stripes."""
    coat = (240, 240, 240)
    d.ellipse([w * 0.20, h * 0.38, w * 0.80, h * 0.70], fill=coat, outline=(40, 40, 40))
    # stripes
    for i in range(6):
        x = w * (0.28 + 0.08 * i)
        d.line([(x, h * 0.40), (x + w * 0.04, h * 0.68)], fill=(30, 30, 30), width=2)
    # legs
    for fx in (0.26, 0.38, 0.60, 0.72):
        d.rectangle([w * fx - w * 0.03, h * 0.68, w * fx + w * 0.03, h * 0.94], fill=coat)
    # head
    d.ellipse([w * 0.72, h * 0.28, w * 0.95, h * 0.48], fill=coat)
    d.line([(w * 0.92, h * 0.32), (w * 1.02, h * 0.20)], fill=(30, 30, 30), width=1)
    d.ellipse([w * 0.82, h * 0.36, w * 0.88, h * 0.42], fill=(20, 20, 20))


def paint_giraffe(d, w, h, rng, mood):
    """Tall giraffe with spots."""
    coat = (232, 186, 82)
    d.ellipse([w * 0.25, h * 0.30, w * 0.75, h * 0.65], fill=coat)
    # spots
    for _ in range(5):
        sx = rng.uniform(w * 0.30, w * 0.70)
        sy = rng.uniform(h * 0.35, h * 0.60)
        sr = rng.uniform(w * 0.04, w * 0.08)
        d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(160, 110, 60))
    # long neck
    d.rectangle([w * 0.42, h * 0.10, w * 0.58, h * 0.35], fill=coat)
    # legs
    for fx in (0.30, 0.42, 0.58, 0.70):
        d.rectangle([w * fx - w * 0.03, h * 0.63, w * fx + w * 0.03, h * 0.94], fill=coat)
    # head
    d.ellipse([w * 0.38, h * 0.04, w * 0.62, h * 0.16], fill=coat)
    d.line([(w * 0.42, h * 0.08), (w * 0.38, h * 0.02)], fill=(160, 110, 60), width=2)
    d.line([(w * 0.58, h * 0.08), (w * 0.62, h * 0.02)], fill=(160, 110, 60), width=2)


def paint_lion(d, w, h, rng, mood):
    """Lion with mane."""
    coat = (210, 160, 80)
    d.ellipse([w * 0.22, h * 0.36, w * 0.78, h * 0.72], fill=coat)
    # mane
    for i in range(10):
        a = math.radians(180 + 30 * i)
        mx = w * 0.50 + math.cos(a) * w * 0.22
        my = h * 0.42 + math.sin(a) * h * 0.14
        d.ellipse([mx - 4, my - 4, mx + 4, my + 4], fill=(140, 90, 50))
    # legs
    for fx in (0.28, 0.40, 0.58, 0.72):
        d.rectangle([w * fx - w * 0.04, h * 0.70, w * fx + w * 0.04, h * 0.94], fill=coat)
    # head
    d.ellipse([w * 0.64, h * 0.26, w * 0.90, h * 0.48], fill=coat)
    d.ellipse([w * 0.78, h * 0.36, w * 0.84, h * 0.42], fill=(20, 20, 20))


def paint_bear(d, w, h, rng, mood):
    """Brown bear."""
    coat = (100, 60, 30)
    d.ellipse([w * 0.20, h * 0.34, w * 0.80, h * 0.72], fill=coat)
    # legs
    for fx in (0.26, 0.40, 0.58, 0.74):
        d.ellipse([w * fx - w * 0.05, h * 0.70, w * fx + w * 0.05, h * 0.92], fill=coat)
    # head
    d.ellipse([w * 0.68, h * 0.20, w * 0.95, h * 0.46], fill=coat)
    # ears
    for ex in (0.70, 0.88):
        d.ellipse([w * ex - w * 0.05, h * 0.16, w * ex + w * 0.05, h * 0.28], fill=coat)
    d.ellipse([w * 0.80, h * 0.32, w * 0.86, h * 0.40], fill=(40, 20, 10))


def paint_monkey(d, w, h, rng, mood):
    """Monkey with long tail."""
    coat = (130, 80, 50)
    d.ellipse([w * 0.28, h * 0.38, w * 0.72, h * 0.66], fill=coat)
    # tail
    d.line([(w * 0.28, h * 0.52), (w * 0.08, h * 0.80)], fill=coat, width=4)
    # legs
    for fx in (0.38, 0.62):
        d.rectangle([w * fx - w * 0.04, h * 0.64, w * fx + w * 0.04, h * 0.92], fill=coat)
    # arms
    d.line([(w * 0.72, h * 0.46), (w * 0.92, h * 0.60)], fill=coat, width=4)
    # head
    d.ellipse([w * 0.62, h * 0.24, w * 0.88, h * 0.46], fill=coat)
    d.ellipse([w * 0.74, h * 0.34, w * 0.80, h * 0.40], fill=(40, 20, 10))


def paint_pig(d, w, h, rng, mood):
    """Pink pig."""
    coat = (255, 180, 200)
    d.ellipse([w * 0.22, h * 0.36, w * 0.78, h * 0.70], fill=coat)
    # legs
    for fx in (0.28, 0.40, 0.58, 0.72):
        d.rectangle([w * fx - w * 0.04, h * 0.68, w * fx + w * 0.04, h * 0.92], fill=coat)
    # head
    d.ellipse([w * 0.70, h * 0.26, w * 0.94, h * 0.50], fill=coat)
    # snout
    d.ellipse([w * 0.86, h * 0.36, w * 0.96, h * 0.46], fill=(255, 140, 160))
    d.ellipse([w * 0.82, h * 0.40, w * 0.85, h * 0.43], fill=(200, 100, 120))
    d.ellipse([w * 0.87, h * 0.40, w * 0.90, h * 0.43], fill=(200, 100, 120))


def paint_chicken(d, w, h, rng, mood):
    """Chicken/bird."""
    body = (255, 240, 220)
    comb = (220, 50, 50)
    # body
    d.ellipse([w * 0.24, h * 0.42, w * 0.76, h * 0.76], fill=body)
    # head
    d.ellipse([w * 0.62, h * 0.28, w * 0.90, h * 0.52], fill=body)
    # comb
    for i in range(3):
        d.ellipse([w * (0.66 + 0.05 * i), h * 0.22, w * (0.70 + 0.05 * i), h * 0.32], fill=comb)
    # beak
    d.polygon([(w * 0.88, h * 0.38), (w * 1.00, h * 0.42), (w * 0.88, h * 0.46)], fill=(240, 180, 50))
    # legs
    for fx in (0.42, 0.58):
        d.line([(w * fx, h * 0.76), (w * fx, h * 0.92)], fill=(220, 150, 50), width=2)
        d.line([(w * fx - 4, h * 0.92), (w * fx, h * 0.96), (w * fx + 4, h * 0.92)], fill=(220, 150, 50), width=2)


def paint_fish(d, w, h, rng, mood):
    """Colorful fish."""
    col = _jit(rng, rng.choice([(60, 120, 200), (200, 100, 60), (100, 180, 100), (200, 180, 80)]), 12)
    # body
    d.ellipse([w * 0.20, h * 0.30, w * 0.80, h * 0.70], fill=col)
    # tail
    d.polygon([(w * 0.20, h * 0.50), (w * 0.02, h * 0.30), (w * 0.02, h * 0.70)], fill=_shade(col, 0.8))
    # fins
    d.polygon([(w * 0.50, h * 0.30), (w * 0.60, h * 0.10), (w * 0.70, h * 0.30)], fill=_shade(col, 0.9))
    d.polygon([(w * 0.50, h * 0.70), (w * 0.60, h * 0.90), (w * 0.70, h * 0.70)], fill=_shade(col, 0.9))
    # eye
    d.ellipse([w * 0.72, h * 0.42, w * 0.78, h * 0.50], fill=(20, 20, 20))


def paint_spider(d, w, h, rng, mood):
    """Black spider."""
    col = (30, 30, 30)
    # body
    d.ellipse([w * 0.40, h * 0.40, w * 0.60, h * 0.65], fill=col)
    d.ellipse([w * 0.44, h * 0.30, w * 0.56, h * 0.42], fill=col)
    # legs
    for side in (-1, 1):
        for i in range(4):
            a = math.radians(180 + 30 * i if side < 0 else 0 - 30 * i)
            lx = w * 0.50 + math.cos(a) * w * 0.28
            ly = h * 0.48 + math.sin(a) * h * 0.20
            d.line([(w * 0.50, h * 0.48), (lx, ly)], fill=col, width=2)


def paint_snake(d, w, h, rng, mood):
    """Coiled snake."""
    col = _jit(rng, rng.choice([(60, 140, 60), (180, 140, 60), (120, 80, 40)]), 10)
    # coiled body
    for i in range(5):
        r = w * (0.35 - 0.06 * i)
        cx, cy = w * 0.50, h * (0.50 - 0.04 * i)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=3)
    # head
    d.ellipse([w * 0.70, h * 0.28, w * 0.86, h * 0.42], fill=col)
    d.ellipse([w * 0.76, h * 0.33, w * 0.80, h * 0.37], fill=(200, 50, 50))


# ── More foods ─────────────────────────────────────────────────────────────


def paint_banana(d, w, h, rng, mood):
    """Yellow banana."""
    col = (240, 220, 80)
    d.arc([w * 0.20, h * 0.20, w * 0.85, h * 0.85], start=30, end=150, fill=col, width=max(3, w // 10))
    d.arc([w * 0.28, h * 0.28, w * 0.72, h * 0.72], start=40, end=140, fill=(240, 220, 160), width=max(2, w // 14))


def paint_orange(d, w, h, rng, mood):
    """Orange fruit."""
    col = (230, 130, 40)
    d.ellipse([w * 0.20, h * 0.20, w * 0.80, h * 0.80], fill=col)
    d.ellipse([w * 0.38, h * 0.10, w * 0.58, h * 0.22], fill=(100, 160, 60))
    d.ellipse([w * 0.28, h * 0.30, w * 0.40, h * 0.44], fill=_shade(col, 1.2))


def paint_watermelon(d, w, h, rng, mood):
    """Watermelon slice."""
    # rind
    d.chord([w * 0.08, h * 0.12, w * 0.92, h * 0.92], start=0, end=180, fill=(60, 140, 60))
    # red flesh
    d.chord([w * 0.12, h * 0.16, w * 0.88, h * 0.88], start=0, end=180, fill=(200, 60, 60))
    # seeds
    for i in range(5):
        sx = w * (0.28 + 0.11 * i)
        sy = h * 0.52
        d.ellipse([sx - 2, sy - 3, sx + 2, sy + 3], fill=(30, 20, 10))


def paint_strawberry(d, w, h, rng, mood):
    """Red strawberry."""
    col = (200, 50, 50)
    d.ellipse([w * 0.25, h * 0.30, w * 0.75, h * 0.85], fill=col)
    # seeds
    for i in range(4):
        for j in range(3):
            sx = w * (0.32 + 0.10 * i)
            sy = h * (0.40 + 0.12 * j)
            d.ellipse([sx - 1, sy - 1, sx + 1, sy + 1], fill=(240, 220, 80))
    # leaves
    d.polygon([(w * 0.50, h * 0.30), (w * 0.30, h * 0.18), (w * 0.40, h * 0.32)], fill=(60, 140, 60))
    d.polygon([(w * 0.50, h * 0.30), (w * 0.70, h * 0.18), (w * 0.60, h * 0.32)], fill=(60, 140, 60))
    d.polygon([(w * 0.50, h * 0.30), (w * 0.50, h * 0.14), (w * 0.58, h * 0.28)], fill=(60, 140, 60))


def paint_grapes(d, w, h, rng, mood):
    """Bunch of grapes."""
    col = _jit(rng, rng.choice([(100, 60, 140), (80, 140, 80), (180, 180, 100)]), 8)
    # grapes
    positions = [(0.50, 0.70), (0.40, 0.60), (0.60, 0.60), (0.35, 0.48), (0.50, 0.48), (0.65, 0.48), (0.42, 0.36), (0.58, 0.36), (0.50, 0.24)]
    for px, py in positions:
        r = w * 0.08
        d.ellipse([w * px - r, h * py - r, w * px + r, h * py + r], fill=col)
    # stem
    d.line([(w * 0.50, h * 0.24), (w * 0.50, h * 0.12)], fill=(80, 120, 60), width=3)


def paint_lemon(d, w, h, rng, mood):
    """Yellow lemon."""
    col = (240, 220, 80)
    d.ellipse([w * 0.18, h * 0.30, w * 0.82, h * 0.70], fill=col)
    # tips
    d.polygon([(w * 0.18, h * 0.50), (w * 0.04, h * 0.50), (w * 0.18, h * 0.55)], fill=(180, 200, 80))
    d.polygon([(w * 0.82, h * 0.50), (w * 0.96, h * 0.50), (w * 0.82, h * 0.55)], fill=(180, 200, 80))


def paint_cherry(d, w, h, rng, mood):
    """Two cherries."""
    col = (180, 40, 40)
    # cherries
    d.ellipse([w * 0.30, h * 0.48, w * 0.48, h * 0.72], fill=col)
    d.ellipse([w * 0.52, h * 0.48, w * 0.70, h * 0.72], fill=col)
    # stems
    d.line([(w * 0.39, h * 0.48), (w * 0.50, h * 0.22)], fill=(60, 120, 40), width=2)
    d.line([(w * 0.61, h * 0.48), (w * 0.50, h * 0.22)], fill=(60, 120, 40), width=2)
    # leaf
    d.ellipse([w * 0.44, h * 0.16, w * 0.58, h * 0.28], fill=(60, 140, 60))


def paint_burger(d, w, h, rng, mood):
    """Hamburger."""
    # bottom bun
    d.chord([w * 0.10, h * 0.60, w * 0.90, h * 0.92], start=0, end=180, fill=(210, 150, 70))
    # patty
    d.rectangle([w * 0.12, h * 0.50, w * 0.88, h * 0.62], fill=(120, 70, 40))
    # cheese
    d.polygon([(w * 0.10, h * 0.50), (w * 0.30, h * 0.58), (w * 0.50, h * 0.50), (w * 0.70, h * 0.58), (w * 0.90, h * 0.50), (w * 0.90, h * 0.54), (w * 0.10, h * 0.54)], fill=(240, 200, 60))
    # lettuce
    for i in range(5):
        x = w * (0.15 + 0.15 * i)
        d.arc([x, h * 0.44, x + w * 0.12, h * 0.54], start=0, end=180, fill=(80, 160, 60))
    # top bun
    d.chord([w * 0.10, h * 0.20, w * 0.90, h * 0.52], start=0, end=180, fill=(220, 160, 80))
    # sesame seeds
    for i in range(4):
        sx = w * (0.28 + 0.12 * i)
        sy = h * 0.34
        d.ellipse([sx - 2, sy - 1, sx + 2, sy + 1], fill=(240, 220, 180))


def paint_hotdog(d, w, h, rng, mood):
    """Hot dog."""
    # bun
    d.rounded_rectangle([w * 0.08, h * 0.32, w * 0.92, h * 0.68], radius=w * 0.08, fill=(210, 150, 70))
    # sausage
    d.rounded_rectangle([w * 0.12, h * 0.38, w * 0.88, h * 0.62], radius=w * 0.04, fill=(180, 80, 60))
    # mustard
    d.line([(w * 0.20, h * 0.48), (w * 0.50, h * 0.42), (w * 0.80, h * 0.50)], fill=(240, 220, 60), width=3)


def paint_pancakes(d, w, h, rng, mood):
    """Stack of pancakes."""
    for i in range(3):
        y = h * (0.72 - 0.16 * i)
        d.ellipse([w * 0.18, y, w * 0.82, y + h * 0.18], fill=(200, 150, 90))
        if i < 2:
            d.rectangle([w * 0.20, y + h * 0.14, w * 0.80, y + h * 0.18], fill=(180, 130, 70))
    # butter
    d.rectangle([w * 0.42, h * 0.36, w * 0.58, h * 0.50], fill=(240, 230, 180))
    # syrup
    d.rectangle([w * 0.10, h * 0.36, w * 0.90, h * 0.70], fill=(180, 100, 40))


def paint_icecream(d, w, h, rng, mood):
    """Ice cream cone."""
    cone = (210, 170, 110)
    scoop = _jit(rng, rng.choice([(255, 220, 220), (220, 180, 140), (180, 220, 180), (255, 200, 150)]), 10)
    # scoop
    d.ellipse([w * 0.22, h * 0.10, w * 0.78, h * 0.56], fill=scoop)
    # cone
    d.polygon([(w * 0.24, h * 0.52), (w * 0.76, h * 0.52), (w * 0.50, h * 0.96)], fill=cone)
    # cone pattern
    for i in range(4):
        y = h * (0.56 + 0.10 * i)
        d.line([(w * 0.30 + i * 2, y), (w * 0.70 - i * 2, y)], fill=(180, 140, 80), width=1)


def paint_sushi(d, w, h, rng, mood):
    """Sushi roll."""
    # nori wrap
    d.ellipse([w * 0.18, h * 0.30, w * 0.82, h * 0.72], fill=(40, 60, 40))
    # rice
    d.ellipse([w * 0.22, h * 0.34, w * 0.78, h * 0.68], fill=(240, 240, 240))
    # filling
    d.ellipse([w * 0.35, h * 0.42, w * 0.65, h * 0.60], fill=(200, 80, 80))


def paint_donut(d, w, h, rng, mood):
    """Doughnut with frosting."""
    # donut
    d.ellipse([w * 0.10, h * 0.20, w * 0.90, h * 0.80], fill=(200, 140, 80))
    # hole
    d.ellipse([w * 0.40, h * 0.40, w * 0.60, h * 0.60], fill=(240, 240, 240))
    # frosting
    d.chord([w * 0.10, h * 0.20, w * 0.90, h * 0.80], start=200, end=340, fill=(255, 180, 200))
    # sprinkles
    for i in range(6):
        sx = w * (0.30 + 0.08 * i)
        sy = h * 0.36
        col = rng.choice([(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)])
        d.rectangle([sx - 2, sy - 1, sx + 2, sy + 1], fill=col)


# ── More objects ─────────────────────────────────────────────────────────


def paint_laptop(d, w, h, rng, mood):
    """Laptop computer."""
    col = (60, 60, 70)
    # screen
    d.rounded_rectangle([w * 0.12, h * 0.12, w * 0.88, h * 0.72], radius=w * 0.04, fill=col)
    d.rounded_rectangle([w * 0.16, h * 0.16, w * 0.84, h * 0.68], radius=w * 0.02, fill=(30, 40, 60))
    # base
    d.rounded_rectangle([w * 0.06, h * 0.72, w * 0.94, h * 0.86], radius=w * 0.02, fill=(80, 80, 90))


def paint_phone(d, w, h, rng, mood):
    """Smartphone."""
    col = (30, 30, 36)
    d.rounded_rectangle([w * 0.28, h * 0.08, w * 0.72, h * 0.92], radius=w * 0.06, fill=col)
    d.rounded_rectangle([w * 0.32, h * 0.14, w * 0.68, h * 0.80], radius=w * 0.02, fill=(40, 50, 70))
    # camera
    d.ellipse([w * 0.58, h * 0.10, w * 0.64, h * 0.16], fill=(20, 20, 24))


def paint_tv(d, w, h, rng, mood):
    """Television."""
    col = (20, 20, 24)
    d.rectangle([w * 0.06, h * 0.14, w * 0.94, h * 0.78], fill=col)
    d.rectangle([w * 0.10, h * 0.18, w * 0.90, h * 0.74], fill=(30, 50, 80))
    # stand
    d.rectangle([w * 0.44, h * 0.78, w * 0.56, h * 0.88], fill=(40, 40, 44))
    d.rectangle([w * 0.32, h * 0.86, w * 0.68, h * 0.92], fill=(40, 40, 44))


def paint_keyboard(d, w, h, rng, mood):
    """Computer keyboard."""
    col = (80, 80, 90)
    d.rounded_rectangle([w * 0.08, h * 0.20, w * 0.92, h * 0.80], radius=w * 0.04, fill=col)
    # keys
    for row in range(3):
        for col_i in range(9):
            kx = w * (0.14 + 0.08 * col_i)
            ky = h * (0.28 + 0.16 * row)
            d.rectangle([kx, ky, kx + w * 0.06, ky + h * 0.12], fill=(60, 60, 70))


def paint_mouse_animal(d, w, h, rng, mood):
    """Computer mouse."""
    col = (50, 50, 60)
    d.ellipse([w * 0.24, h * 0.28, w * 0.76, h * 0.72], fill=col)
    # button
    d.rectangle([w * 0.44, h * 0.30, w * 0.56, h * 0.48], fill=(40, 40, 50))
    # wheel
    d.line([(w * 0.50, h * 0.36), (w * 0.50, h * 0.44)], fill=(30, 30, 40), width=3)


def paint_headphones(d, w, h, rng, mood):
    """Headphones."""
    col = _jit(rng, rng.choice([(40, 40, 50), (200, 60, 60), (60, 100, 200)]), 8)
    # band
    d.arc([w * 0.18, h * 0.20, w * 0.82, h * 0.70], start=20, end=160, fill=col, width=max(3, w // 14))
    # earcups
    d.ellipse([w * 0.14, h * 0.52, w * 0.38, h * 0.78], fill=col)
    d.ellipse([w * 0.62, h * 0.52, w * 0.86, h * 0.78], fill=col)


def paint_trophy(d, w, h, rng, mood):
    """Gold trophy."""
    gold = (220, 180, 60)
    dark = (180, 140, 40)
    # cup
    d.chord([w * 0.28, h * 0.14, w * 0.72, h * 0.62], start=0, end=180, fill=gold)
    d.rectangle([w * 0.36, h * 0.50, w * 0.64, h * 0.58], fill=gold)
    # handles
    d.arc([w * 0.14, h * 0.28, w * 0.36, h * 0.54], start=270, end=90, fill=gold, width=3)
    d.arc([w * 0.64, h * 0.28, w * 0.86, h * 0.54], start=90, end=270, fill=gold, width=3)
    # stem
    d.rectangle([w * 0.46, h * 0.58, w * 0.54, h * 0.78], fill=gold)
    # base
    d.rectangle([w * 0.32, h * 0.78, w * 0.68, h * 0.86], fill=dark)
    d.rectangle([w * 0.26, h * 0.84, w * 0.74, h * 0.92], fill=gold)


def paint_medal(d, w, h, rng, mood):
    """Medal with ribbon."""
    # ribbon
    d.polygon([(w * 0.40, h * 0.08), (w * 0.60, h * 0.08), (w * 0.55, h * 0.40), (w * 0.45, h * 0.40)], fill=(200, 50, 50))
    # medal
    d.ellipse([w * 0.30, h * 0.38, w * 0.70, h * 0.80], fill=(220, 180, 60))
    d.ellipse([w * 0.38, h * 0.46, w * 0.62, h * 0.72], fill=(240, 200, 80))


def paint_balloon(d, w, h, rng, mood):
    """Balloon."""
    col = _jit(rng, rng.choice([(200, 60, 60), (60, 120, 200), (200, 180, 60), (60, 180, 120), (180, 60, 180)]), 12)
    d.ellipse([w * 0.26, h * 0.08, w * 0.74, h * 0.68], fill=col)
    # knot
    d.polygon([(w * 0.46, h * 0.68), (w * 0.54, h * 0.68), (w * 0.50, h * 0.78)], fill=_shade(col, 0.8))
    # string
    d.line([(w * 0.50, h * 0.78), (w * 0.48, h * 0.96)], fill=(180, 180, 180), width=1)


def paint_candle(d, w, h, rng, mood):
    """Candle."""
    col = (240, 230, 200)
    d.rounded_rectangle([w * 0.38, h * 0.30, w * 0.62, h * 0.90], radius=w * 0.04, fill=col)
    # flame
    d.ellipse([w * 0.44, h * 0.12, w * 0.56, h * 0.32], fill=(240, 180, 60))
    d.ellipse([w * 0.47, h * 0.18, w * 0.53, h * 0.26], fill=(255, 240, 200))


def paint_lamp(d, w, h, rng, mood):
    """Table lamp."""
    shade = _jit(rng, rng.choice([(200, 180, 140), (140, 180, 200), (200, 160, 180)]), 10)
    # shade
    d.polygon([(w * 0.20, h * 0.30), (w * 0.80, h * 0.30), (w * 0.70, h * 0.60), (w * 0.30, h * 0.60)], fill=shade)
    # base
    d.rectangle([w * 0.44, h * 0.60, w * 0.56, h * 0.90], fill=(80, 80, 90))
    d.ellipse([w * 0.38, h * 0.88, w * 0.62, h * 0.98], fill=(60, 60, 70))


def paint_bottle(d, w, h, rng, mood):
    """Bottle."""
    col = _jit(rng, rng.choice([(60, 140, 80), (180, 60, 60), (60, 100, 160), (200, 180, 80)]), 8)
    # body
    d.ellipse([w * 0.28, h * 0.38, w * 0.72, h * 0.88], fill=col)
    # neck
    d.rectangle([w * 0.42, h * 0.18, w * 0.58, h * 0.40], fill=col)
    # cap
    d.rectangle([w * 0.40, h * 0.12, w * 0.60, h * 0.20], fill=_shade(col, 0.7))


def paint_glass(d, w, h, rng, mood):
    """Drinking glass."""
    col = (180, 220, 240)
    d.polygon([(w * 0.30, h * 0.18), (w * 0.70, h * 0.18), (w * 0.65, h * 0.88), (w * 0.35, h * 0.88)], fill=col, outline=(160, 200, 220))
    # liquid
    d.polygon([(w * 0.32, h * 0.50), (w * 0.68, h * 0.50), (w * 0.64, h * 0.86), (w * 0.36, h * 0.86)], fill=_jit(rng, rng.choice([(200, 100, 50), (240, 200, 80), (150, 80, 200)]), 10))


def paint_guitar(d, w, h, rng, mood):
    """Guitar."""
    wood = (140, 90, 50)
    dark = (100, 60, 30)
    # body
    d.ellipse([w * 0.28, h * 0.50, w * 0.72, h * 0.88], fill=wood)
    # waist curves
    d.ellipse([w * 0.30, h * 0.42, w * 0.70, h * 0.70], fill=wood)
    # neck
    d.rectangle([w * 0.46, h * 0.06, w * 0.54, h * 0.55], fill=dark)
    # headstock
    d.rectangle([w * 0.44, h * 0.02, w * 0.56, h * 0.10], fill=dark)
    # sound hole
    d.ellipse([w * 0.44, h * 0.64, w * 0.56, h * 0.78], fill=(30, 20, 10))


def paint_violin(d, w, h, rng, mood):
    """Violin."""
    wood = (180, 120, 70)
    dark = (120, 70, 40)
    # body
    d.ellipse([w * 0.28, h * 0.48, w * 0.72, h * 0.88], fill=wood)
    # upper bout
    d.ellipse([w * 0.34, h * 0.36, w * 0.66, h * 0.58], fill=wood)
    # neck
    d.rectangle([w * 0.46, h * 0.06, w * 0.54, h * 0.42], fill=dark)
    # scroll
    d.ellipse([w * 0.44, h * 0.02, w * 0.56, h * 0.12], fill=dark)


def paint_drum(d, w, h, rng, mood):
    """Drum."""
    col = _jit(rng, rng.choice([(180, 60, 60), (60, 100, 180), (200, 180, 60)]), 8)
    d.ellipse([w * 0.18, h * 0.50, w * 0.82, h * 0.82], fill=col)
    d.ellipse([w * 0.22, h * 0.54, w * 0.78, h * 0.78], fill=(240, 230, 200))
    # shell
    d.rectangle([w * 0.18, h * 0.50, w * 0.82, h * 0.62], fill=col)
    # tension rods
    for i in range(4):
        x = w * (0.28 + 0.14 * i)
        d.line([(x, h * 0.50), (x, h * 0.82)], fill=(100, 100, 110), width=1)


def paint_piano(d, w, h, rng, mood):
    """Piano keys."""
    # white keys
    for i in range(5):
        x = w * (0.12 + 0.14 * i)
        d.rectangle([x, h * 0.40, x + w * 0.10, h * 0.80], fill=(240, 240, 230))
    # black keys
    for i in range(4):
        x = w * (0.18 + 0.14 * i)
        if i != 2:
            d.rectangle([x, h * 0.40, x + w * 0.07, h * 0.60], fill=(30, 30, 36))


def paint_sword(d, w, h, rng, mood):
    """Sword."""
    blade = (180, 190, 210)
    guard = (200, 180, 80)
    # blade
    d.polygon([(w * 0.50, h * 0.02), (w * 0.54, h * 0.60), (w * 0.46, h * 0.60)], fill=blade)
    # guard
    d.rectangle([w * 0.36, h * 0.58, w * 0.64, h * 0.66], fill=guard)
    # handle
    d.rectangle([w * 0.46, h * 0.66, w * 0.54, h * 0.90], fill=(80, 50, 30))
    # pommel
    d.ellipse([w * 0.44, h * 0.88, w * 0.56, h * 0.96], fill=guard)


def paint_shield(d, w, h, rng, mood):
    """Shield."""
    col = _jit(rng, rng.choice([(60, 100, 180), (180, 60, 60), (60, 180, 100), (180, 180, 60)]), 10)
    d.polygon([(w * 0.50, h * 0.06), (w * 0.88, h * 0.28), (w * 0.88, h * 0.70), (w * 0.50, h * 0.96), (w * 0.12, h * 0.70), (w * 0.12, h * 0.28)], fill=col, outline=_shade(col, 0.6))
    # cross
    d.rectangle([w * 0.46, h * 0.30, w * 0.54, h * 0.72], fill=(240, 220, 180))
    d.rectangle([w * 0.32, h * 0.48, w * 0.68, h * 0.54], fill=(240, 220, 180))


def paint_crown(d, w, h, rng, mood):
    """Golden crown."""
    gold = (220, 180, 60)
    dark = (180, 140, 40)
    # base
    d.polygon([(w * 0.14, h * 0.50), (w * 0.86, h * 0.50), (w * 0.80, h * 0.80), (w * 0.20, h * 0.80)], fill=gold)
    # points
    for i in range(5):
        x = w * (0.18 + 0.16 * i)
        d.polygon([(x, h * 0.50), (x + w * 0.08, h * 0.16), (x + w * 0.16, h * 0.50)], fill=gold)
        d.ellipse([x + w * 0.04, h * 0.12, x + w * 0.12, h * 0.24], fill=(200, 60, 60))


def paint_rocket(d, w, h, rng, mood):
    """Rocket."""
    col = (220, 220, 230)
    # body
    d.polygon([(w * 0.50, h * 0.02), (w * 0.68, h * 0.60), (w * 0.32, h * 0.60)], fill=col)
    d.rectangle([w * 0.32, h * 0.56, w * 0.68, h * 0.88], fill=col)
    # fins
    d.polygon([(w * 0.32, h * 0.70), (w * 0.12, h * 0.92), (w * 0.32, h * 0.88)], fill=(200, 60, 60))
    d.polygon([(w * 0.68, h * 0.70), (w * 0.88, h * 0.92), (w * 0.68, h * 0.88)], fill=(200, 60, 60))
    # window
    d.ellipse([w * 0.42, h * 0.30, w * 0.58, h * 0.46], fill=(100, 160, 220))
    # flame
    d.polygon([(w * 0.38, h * 0.88), (w * 0.50, h * 0.98), (w * 0.62, h * 0.88), (w * 0.56, h * 0.88), (w * 0.50, h * 0.94), (w * 0.44, h * 0.88)], fill=(240, 180, 60))


def paint_ufo(d, w, h, rng, mood):
    """UFO flying saucer."""
    # dome
    d.ellipse([w * 0.36, h * 0.28, w * 0.64, h * 0.54], fill=(180, 200, 240))
    # saucer
    d.ellipse([w * 0.16, h * 0.48, w * 0.84, h * 0.72], fill=(160, 160, 180))
    # lights
    for i in range(5):
        x = w * (0.24 + 0.10 * i)
        d.ellipse([x - 2, h * 0.58, x + 2, h * 0.64], fill=(240, 240, 100))


def paint_tent(d, w, h, rng, mood):
    """Camping tent."""
    col = _jit(rng, rng.choice([(60, 120, 80), (180, 100, 60), (80, 80, 140)]), 10)
    d.polygon([(w * 0.50, h * 0.12), (w * 0.88, h * 0.84), (w * 0.12, h * 0.84)], fill=col)
    # entrance
    d.polygon([(w * 0.42, h * 0.84), (w * 0.58, h * 0.84), (w * 0.52, h * 0.60)], fill=(40, 30, 20))


def paint_flag(d, w, h, rng, mood):
    """Flag on pole."""
    col = _jit(rng, rng.choice([(200, 60, 60), (60, 120, 200), (60, 180, 100), (200, 180, 60)]), 10)
    # pole
    d.rectangle([w * 0.18, h * 0.08, w * 0.22, h * 0.96], fill=(140, 140, 150))
    # flag
    d.polygon([(w * 0.22, h * 0.12), (w * 0.85, h * 0.28), (w * 0.85, h * 0.52), (w * 0.22, h * 0.42)], fill=col)
    # ball top
    d.ellipse([w * 0.16, h * 0.04, w * 0.24, h * 0.12], fill=(220, 180, 60))


def paint_binoculars(d, w, h, rng, mood):
    """Binoculars."""
    col = (40, 40, 50)
    d.ellipse([w * 0.20, h * 0.30, w * 0.46, h * 0.62], fill=col)
    d.ellipse([w * 0.54, h * 0.30, w * 0.80, h * 0.62], fill=col)
    d.rectangle([w * 0.44, h * 0.40, w * 0.56, h * 0.52], fill=col)
    d.ellipse([w * 0.24, h * 0.34, w * 0.42, h * 0.50], fill=(60, 80, 120))
    d.ellipse([w * 0.58, h * 0.34, w * 0.76, h * 0.50], fill=(60, 80, 120))


def paint_telescope(d, w, h, rng, mood):
    """Telescope."""
    col = (60, 60, 70)
    # tube
    d.rectangle([w * 0.20, h * 0.38, w * 0.80, h * 0.52], fill=col)
    d.ellipse([w * 0.18, h * 0.34, w * 0.34, h * 0.56], fill=col)
    # tripod
    for side in (-1, 1):
        d.line([(w * 0.50, h * 0.52), (w * 0.50 + side * w * 0.20, h * 0.92)], fill=(80, 80, 90), width=3)


def paint_compass(d, w, h, rng, mood):
    """Compass."""
    d.ellipse([w * 0.20, h * 0.20, w * 0.80, h * 0.80], fill=(220, 200, 160), outline=(140, 120, 80))
    # needle
    d.polygon([(w * 0.50, h * 0.28), (w * 0.54, h * 0.50), (w * 0.46, h * 0.50)], fill=(200, 50, 50))
    d.polygon([(w * 0.50, h * 0.72), (w * 0.54, h * 0.50), (w * 0.46, h * 0.50)], fill=(220, 220, 220))
    # markings
    for i in range(8):
        a = math.radians(45 * i)
        x = w * 0.50 + math.cos(a) * w * 0.28
        y = h * 0.50 + math.sin(a) * h * 0.28
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(80, 60, 40))


def paint_watch(d, w, h, rng, mood):
    """Wrist watch."""
    col = _jit(rng, rng.choice([(40, 40, 50), (200, 60, 60), (60, 100, 180), (80, 160, 80)]), 8)
    # band
    d.rectangle([w * 0.38, h * 0.04, w * 0.62, h * 0.22], fill=(60, 60, 70))
    d.rectangle([w * 0.38, h * 0.78, w * 0.62, h * 0.96], fill=(60, 60, 70))
    # face
    d.ellipse([w * 0.20, h * 0.22, w * 0.80, h * 0.78], fill=col)
    d.ellipse([w * 0.28, h * 0.30, w * 0.72, h * 0.70], fill=(240, 240, 240))
    # hands
    d.line([(w * 0.50, h * 0.50), (w * 0.50, h * 0.38)], fill=(30, 30, 36), width=2)
    d.line([(w * 0.50, h * 0.50), (w * 0.62, h * 0.56)], fill=(30, 30, 36), width=1)


def paint_radio(d, w, h, rng, mood):
    """Retro radio."""
    col = (180, 120, 80)
    d.rounded_rectangle([w * 0.10, h * 0.28, w * 0.90, h * 0.80], radius=w * 0.04, fill=col)
    # speaker grille
    for i in range(6):
        d.line([(w * 0.16, h * (0.36 + 0.06 * i)), (w * 0.50, h * (0.36 + 0.06 * i))], fill=(100, 60, 40), width=2)
    # dial
    d.ellipse([w * 0.64, h * 0.40, w * 0.82, h * 0.58], fill=(240, 240, 220))
    # antenna
    d.line([(w * 0.80, h * 0.28), (w * 0.88, h * 0.08)], fill=(60, 60, 70), width=2)


def paint_camera(d, w, h, rng, mood):
    """Camera."""
    col = (40, 40, 50)
    d.rounded_rectangle([w * 0.16, h * 0.28, w * 0.84, h * 0.74], radius=w * 0.04, fill=col)
    # lens
    d.ellipse([w * 0.36, h * 0.40, w * 0.64, h * 0.64], fill=(30, 30, 36))
    d.ellipse([w * 0.42, h * 0.46, w * 0.58, h * 0.58], fill=(60, 80, 120))
    # flash
    d.rectangle([w * 0.70, h * 0.32, w * 0.80, h * 0.42], fill=(220, 220, 200))


def paint_bicycle_wheel(d, w, h, rng, mood):
    """Bicycle wheel."""
    r = w * 0.40
    cx, cy = w * 0.50, h * 0.50
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(60, 60, 70), width=4)
    d.ellipse([cx - r * 0.2, cy - r * 0.2, cx + r * 0.2, cy + r * 0.2], fill=(80, 80, 90))
    # spokes
    for i in range(8):
        a = math.radians(45 * i)
        d.line([(cx, cy), (cx + math.cos(a) * r * 0.9, cy + math.sin(a) * r * 0.9)], fill=(140, 140, 150), width=1)


def paint_skateboard(d, w, h, rng, mood):
    """Skateboard."""
    col = _jit(rng, rng.choice([(60, 180, 100), (200, 60, 60), (60, 100, 200), (200, 180, 60)]), 10)
    d.rounded_rectangle([w * 0.08, h * 0.40, w * 0.92, h * 0.60], radius=w * 0.12, fill=col)
    # wheels
    for x in (w * 0.24, w * 0.76):
        d.ellipse([x - 6, h * 0.60, x + 6, h * 0.72], fill=(40, 40, 50))


def paint_surfboard(d, w, h, rng, mood):
    """Surfboard."""
    col = _jit(rng, rng.choice([(240, 240, 240), (60, 120, 200), (200, 60, 100)]), 10)
    d.ellipse([w * 0.42, h * 0.06, w * 0.58, h * 0.94], fill=col)
    # stripes
    for i in range(3):
        d.line([(w * 0.44, h * (0.20 + 0.20 * i)), (w * 0.56, h * (0.20 + 0.20 * i))], fill=_shade(col, 0.8), width=2)


def paint_snowman(d, w, h, rng, mood):
    """Snowman."""
    snow = (245, 250, 255)
    # bottom ball
    d.ellipse([w * 0.20, h * 0.58, w * 0.80, h * 0.96], fill=snow)
    # middle ball
    d.ellipse([w * 0.28, h * 0.38, w * 0.72, h * 0.72], fill=snow)
    # head
    d.ellipse([w * 0.36, h * 0.20, w * 0.64, h * 0.44], fill=snow)
    # eyes
    d.ellipse([w * 0.44, h * 0.30, w * 0.48, h * 0.34], fill=(30, 30, 36))
    d.ellipse([w * 0.52, h * 0.30, w * 0.56, h * 0.34], fill=(30, 30, 36))
    # nose
    d.polygon([(w * 0.50, h * 0.36), (w * 0.58, h * 0.40), (w * 0.50, h * 0.42)], fill=(220, 100, 50))
    # buttons
    for i in range(3):
        y = h * (0.48 + 0.08 * i)
        d.ellipse([w * 0.48, y, w * 0.52, y + 4], fill=(30, 30, 36))


def paint_ghost(d, w, h, rng, mood):
    """Ghost."""
    col = (240, 240, 250)
    d.ellipse([w * 0.24, h * 0.18, w * 0.76, h * 0.72], fill=col)
    # wavy bottom
    for i in range(4):
        x = w * (0.26 + 0.12 * i)
        d.arc([x - 4, h * 0.66, x + 8, h * 0.86], start=0, end=180, fill=col)
    # eyes
    d.ellipse([w * 0.36, h * 0.36, w * 0.46, h * 0.48], fill=(30, 30, 40))
    d.ellipse([w * 0.54, h * 0.36, w * 0.64, h * 0.48], fill=(30, 30, 40))


def paint_skull(d, w, h, rng, mood):
    """Skull."""
    col = (240, 235, 220)
    d.ellipse([w * 0.22, h * 0.20, w * 0.78, h * 0.72], fill=col)
    # eye sockets
    d.ellipse([w * 0.32, h * 0.38, w * 0.46, h * 0.52], fill=(40, 30, 30))
    d.ellipse([w * 0.54, h * 0.38, w * 0.68, h * 0.52], fill=(40, 30, 30))
    # nose
    d.polygon([(w * 0.47, h * 0.56), (w * 0.53, h * 0.56), (w * 0.50, h * 0.64)], fill=(60, 50, 50))
    # jaw
    d.rectangle([w * 0.30, h * 0.62, w * 0.70, h * 0.78], fill=col)
    # teeth
    for i in range(5):
        x = w * (0.34 + 0.07 * i)
        d.rectangle([x, h * 0.70, x + 3, h * 0.76], fill=(220, 215, 200))


def paint_pumpkin(d, w, h, rng, mood):
    """Pumpkin."""
    col = (220, 120, 40)
    # segments
    for i in range(5):
        x = w * (0.22 + 0.14 * i)
        d.ellipse([x - w * 0.08, h * 0.24, x + w * 0.08, h * 0.80], fill=col)
    # stem
    d.rectangle([w * 0.46, h * 0.14, w * 0.54, h * 0.28], fill=(80, 120, 60))


def paint_broom(d, w, h, rng, mood):
    """Broom."""
    wood = (160, 110, 60)
    straw = (200, 180, 100)
    d.rectangle([w * 0.46, h * 0.02, w * 0.54, h * 0.70], fill=wood)
    d.rectangle([w * 0.28, h * 0.68, w * 0.72, h * 0.96], fill=straw)
    for i in range(8):
        x = w * (0.32 + 0.05 * i)
        d.line([(x, h * 0.70), (x, h * 0.96)], fill=(180, 160, 80), width=2)


def paint_mop(d, w, h, rng, mood):
    """Mop."""
    wood = (160, 110, 60)
    strands = (200, 200, 180)
    d.rectangle([w * 0.46, h * 0.02, w * 0.54, h * 0.55], fill=wood)
    # mop head
    d.rounded_rectangle([w * 0.30, h * 0.52, w * 0.70, h * 0.96], radius=w * 0.08, fill=strands)
    for i in range(5):
        x = w * (0.36 + 0.06 * i)
        d.line([(x, h * 0.60), (x, h * 0.94)], fill=(180, 180, 160), width=2)


def paint_bucket(d, w, h, rng, mood):
    """Bucket."""
    col = _jit(rng, rng.choice([(60, 120, 200), (200, 60, 120), (200, 180, 60)]), 10)
    # tapered body
    d.polygon([(w * 0.24, h * 0.24), (w * 0.76, h * 0.24), (w * 0.68, h * 0.86), (w * 0.32, h * 0.86)], fill=col)
    # rim
    d.ellipse([w * 0.22, h * 0.20, w * 0.78, h * 0.36], fill=_shade(col, 0.8), outline=_shade(col, 0.6))
    # handle
    d.arc([w * 0.28, h * 0.12, w * 0.72, h * 0.42], start=0, end=180, fill=(100, 100, 110), width=3)


def paint_wheelbarrow(d, w, h, rng, mood):
    """Wheelbarrow."""
    tray = _jit(rng, rng.choice([(80, 140, 80), (180, 100, 60), (140, 80, 180)]), 10)
    # tray
    d.polygon([(w * 0.20, h * 0.36), (w * 0.72, h * 0.36), (w * 0.65, h * 0.70), (w * 0.28, h * 0.70)], fill=tray)
    # wheel
    d.ellipse([w * 0.72, h * 0.62, w * 0.88, h * 0.80], fill=(50, 50, 60))
    # handles
    d.line([(w * 0.20, h * 0.42), (w * 0.08, h * 0.28)], fill=(140, 110, 70), width=4)
    d.line([(w * 0.65, h * 0.70), (w * 0.68, h * 0.52)], fill=(140, 110, 70), width=4)


def paint_ladder(d, w, h, rng, mood):
    """Ladder."""
    wood = (160, 110, 60)
    # rails
    d.rectangle([w * 0.24, h * 0.06, w * 0.32, h * 0.94], fill=wood)
    d.rectangle([w * 0.68, h * 0.06, w * 0.76, h * 0.94], fill=wood)
    # rungs
    for i in range(6):
        y = h * (0.14 + 0.13 * i)
        d.rectangle([w * 0.24, y, w * 0.76, y + h * 0.06], fill=wood)


def paint_fence(d, w, h, rng, mood):
    """Wooden fence section."""
    wood = (180, 140, 90)
    # posts
    for i in range(4):
        x = w * (0.10 + 0.24 * i)
        d.rectangle([x, h * 0.16, x + w * 0.12, h * 0.90], fill=wood)
        # pointed top
        d.polygon([(x, h * 0.16), (x + w * 0.06, h * 0.06), (x + w * 0.12, h * 0.16)], fill=wood)


def paint_mailbox(d, w, h, rng, mood):
    """Mailbox."""
    col = (60, 120, 200)
    # post
    d.rectangle([w * 0.46, h * 0.50, w * 0.54, h * 0.96], fill=(100, 70, 40))
    # box
    d.rounded_rectangle([w * 0.20, h * 0.28, w * 0.80, h * 0.58], radius=w * 0.06, fill=col)
    # door
    d.arc([w * 0.50, h * 0.28, w * 0.82, h * 0.60], start=90, end=180, fill=_shade(col, 0.8))


def paint_fireplace(d, w, h, rng, mood):
    """Fireplace."""
    brick = (180, 80, 60)
    dark = (60, 30, 20)
    # frame
    d.rectangle([w * 0.10, h * 0.24, w * 0.90, h * 0.92], fill=brick)
    # opening
    d.ellipse([w * 0.20, h * 0.40, w * 0.80, h * 0.88], fill=dark)
    # fire
    d.polygon([(w * 0.30, h * 0.80), (w * 0.40, h * 0.50), (w * 0.50, h * 0.75), (w * 0.60, h * 0.45), (w * 0.70, h * 0.80)], fill=(240, 150, 40))


def paint_chimney(d, w, h, rng, mood):
    """Chimney."""
    brick = (160, 70, 50)
    d.rectangle([w * 0.30, h * 0.20, w * 0.70, h * 0.90], fill=brick)
    # bricks
    for row in range(4):
        for col_i in range(2):
            x = w * (0.32 + 0.18 * col_i)
            y = h * (0.24 + 0.16 * row)
            if col_i == 0:
                d.rectangle([x, y, x + w * 0.16, y + h * 0.12], outline=(120, 50, 40))
            else:
                d.rectangle([x + w * 0.02, y, x + w * 0.18, y + h * 0.12], outline=(120, 50, 40))


# ── EVEN MORE CLASSES: Ocean, Nature, Sports ──────────────────────────────


def paint_dolphin(d, w, h, rng, mood):
    """Dolphin."""
    col = (100, 150, 180)
    # body
    d.ellipse([w * 0.20, h * 0.40, w * 0.80, h * 0.62], fill=col)
    # head
    d.ellipse([w * 0.72, h * 0.34, w * 0.92, h * 0.56], fill=col)
    # fin
    d.polygon([(w * 0.50, h * 0.40), (w * 0.56, h * 0.22), (w * 0.64, h * 0.40)], fill=_shade(col, 0.8))
    # tail
    d.polygon([(w * 0.20, h * 0.50), (w * 0.04, h * 0.36), (w * 0.12, h * 0.50), (w * 0.04, h * 0.64)], fill=col)


def paint_whale(d, w, h, rng, mood):
    """Blue whale."""
    col = (60, 90, 140)
    d.ellipse([w * 0.12, h * 0.36, w * 0.88, h * 0.66], fill=col)
    # head
    d.ellipse([w * 0.78, h * 0.32, w * 0.94, h * 0.60], fill=col)
    # tail
    d.polygon([(w * 0.12, h * 0.50), (w * 0.02, h * 0.36), (w * 0.06, h * 0.50), (w * 0.02, h * 0.64)], fill=col)
    # water spout
    d.line([(w * 0.80, h * 0.28), (w * 0.82, h * 0.16)], fill=(200, 220, 240), width=3)


def paint_shark(d, w, h, rng, mood):
    """Shark."""
    col = (100, 110, 120)
    d.ellipse([w * 0.18, h * 0.40, w * 0.85, h * 0.62], fill=col)
    # dorsal fin
    d.polygon([(w * 0.50, h * 0.40), (w * 0.56, h * 0.18), (w * 0.62, h * 0.40)], fill=col)
    # tail
    d.polygon([(w * 0.18, h * 0.50), (w * 0.04, h * 0.30), (w * 0.10, h * 0.50)], fill=col)
    # teeth
    d.polygon([(w * 0.82, h * 0.50), (w * 0.88, h * 0.56), (w * 0.84, h * 0.50)], fill=(240, 240, 240))


def paint_crab(d, w, h, rng, mood):
    """Crab."""
    col = (200, 80, 60)
    # body
    d.ellipse([w * 0.28, h * 0.44, w * 0.72, h * 0.72], fill=col)
    # eyes
    for x in (w * 0.38, w * 0.62):
        d.line([(x, h * 0.44), (x, h * 0.34)], fill=col, width=3)
        d.ellipse([x - 3, h * 0.30, x + 3, h * 0.38], fill=(40, 30, 30))
    # claws
    d.ellipse([w * 0.14, h * 0.48, w * 0.30, h * 0.62], fill=col)
    d.ellipse([w * 0.70, h * 0.48, w * 0.86, h * 0.62], fill=col)
    # legs
    for i in range(3):
        for side in (-1, 1):
            x = w * (0.32 + 0.08 * i)
            d.line([(x, h * 0.72), (x + side * w * 0.06, h * 0.86)], fill=col, width=2)


def paint_octopus(d, w, h, rng, mood):
    """Octopus."""
    col = _jit(rng, rng.choice([(180, 80, 140), (80, 140, 180), (200, 140, 80)]), 10)
    # head
    d.ellipse([w * 0.30, h * 0.18, w * 0.70, h * 0.56], fill=col)
    # eyes
    d.ellipse([w * 0.40, h * 0.34, w * 0.48, h * 0.42], fill=(240, 240, 240))
    d.ellipse([w * 0.52, h * 0.34, w * 0.60, h * 0.42], fill=(240, 240, 240))
    d.ellipse([w * 0.43, h * 0.36, w * 0.47, h * 0.40], fill=(20, 20, 30))
    d.ellipse([w * 0.53, h * 0.36, w * 0.57, h * 0.40], fill=(20, 20, 30))
    # tentacles
    for i in range(6):
        a = math.radians(200 + 40 * i)
        tx = w * 0.50 + math.cos(a) * w * 0.30
        ty = h * 0.56 + math.sin(a) * h * 0.28
        d.line([(w * 0.50, h * 0.56), (tx, ty)], fill=col, width=4)


def paint_jellyfish(d, w, h, rng, mood):
    """Jellyfish."""
    col = (220, 180, 240)
    # bell
    d.ellipse([w * 0.28, h * 0.16, w * 0.72, h * 0.50], fill=col)
    # tentacles
    for i in range(5):
        x = w * (0.34 + 0.07 * i)
        d.arc([x - 4, h * 0.50, x + 4, h * 0.90], start=0, end=180, fill=(200, 160, 220), width=2)


def paint_seahorse(d, w, h, rng, mood):
    """Seahorse."""
    col = (240, 180, 80)
    # body
    d.ellipse([w * 0.40, h * 0.30, w * 0.68, h * 0.68], fill=col)
    # head
    d.ellipse([w * 0.56, h * 0.18, w * 0.78, h * 0.38], fill=col)
    # snout
    d.rectangle([w * 0.74, h * 0.28, w * 0.88, h * 0.34], fill=col)
    # tail curled
    d.arc([w * 0.34, h * 0.62, w * 0.52, h * 0.86], start=90, end=270, fill=col, width=4)
    # fin
    d.polygon([(w * 0.50, h * 0.42), (w * 0.62, h * 0.54), (w * 0.50, h * 0.54)], fill=_shade(col, 0.8))


def paint_tornado(d, w, h, rng, mood):
    """Tornado."""
    col = (180, 180, 200)
    for i in range(6):
        r = w * (0.30 - 0.04 * i)
        cy = h * (0.20 + 0.12 * i)
        d.arc([w * 0.50 - r, cy - r * 0.4, w * 0.50 + r, cy + r * 0.4], start=0, end=180, fill=col, width=3)


def paint_cloud(d, w, h, rng, mood):
    """Cloud."""
    col = (240, 245, 250)
    for i in range(4):
        x = w * (0.18 + 0.18 * i)
        y = h * (0.40 - 0.06 * (i % 2))
        r = w * (0.16 - 0.02 * abs(i - 1.5))
        d.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], fill=col)


def paint_rainbow(d, w, h, rng, mood):
    """Rainbow."""
    colors = [(220, 60, 60), (240, 180, 60), (240, 240, 80), (80, 200, 80), (60, 120, 220), (140, 80, 200)]
    for i, col in enumerate(colors):
        r_outer = w * (0.42 + 0.06 * i)
        r_inner = w * (0.36 + 0.06 * i)
        d.arc([w * 0.50 - r_outer, h * 0.10 - r_outer * 0.4, w * 0.50 + r_outer, h * 0.10 + r_outer * 0.4], start=180, end=360, fill=col, width=3)


def paint_sun(d, w, h, rng, mood):
    """Sun."""
    col = (255, 220, 80)
    d.ellipse([w * 0.24, h * 0.24, w * 0.76, h * 0.76], fill=col)
    # rays
    for i in range(8):
        a = math.radians(45 * i)
        x1 = w * 0.50 + math.cos(a) * w * 0.36
        y1 = h * 0.50 + math.sin(a) * h * 0.36
        x2 = w * 0.50 + math.cos(a) * w * 0.46
        y2 = h * 0.50 + math.sin(a) * h * 0.46
        d.line([(x1, y1), (x2, y2)], fill=(255, 200, 60), width=3)


def paint_moon(d, w, h, rng, mood):
    """Moon."""
    col = (240, 240, 220)
    d.ellipse([w * 0.20, h * 0.20, w * 0.80, h * 0.80], fill=col)
    # crater
    d.ellipse([w * 0.36, h * 0.40, w * 0.50, h * 0.54], fill=(220, 220, 200))
    d.ellipse([w * 0.52, h * 0.60, w * 0.60, h * 0.68], fill=(220, 220, 200))


def paint_star(d, w, h, rng, mood):
    """Star."""
    col = (255, 220, 80)
    points = []
    for i in range(10):
        a = math.radians(90 + 36 * i)
        r = w * 0.40 if i % 2 == 0 else w * 0.20
        points.append((w * 0.50 + math.cos(a) * r, h * 0.50 + math.sin(a) * r))
    d.polygon(points, fill=col)


def paint_heart(d, w, h, rng, mood):
    """Heart."""
    col = (200, 50, 80)
    d.polygon([(w * 0.50, h * 0.30), (w * 0.20, h * 0.60), (w * 0.50, h * 0.88), (w * 0.80, h * 0.60)], fill=col)
    d.ellipse([w * 0.28, h * 0.30, w * 0.46, h * 0.50], fill=col)
    d.ellipse([w * 0.54, h * 0.30, w * 0.72, h * 0.50], fill=col)


def paint_umbrella_beach(d, w, h, rng, mood):
    """Beach umbrella."""
    pole = (160, 140, 100)
    col = _jit(rng, rng.choice([(200, 60, 60), (60, 140, 200), (240, 180, 80)]), 10)
    d.rectangle([w * 0.48, h * 0.40, w * 0.52, h * 0.96], fill=pole)
    d.ellipse([w * 0.10, h * 0.12, w * 0.90, h * 0.50], fill=col)
    # stripes
    for i in range(3):
        x = w * (0.20 + 0.20 * i)
        d.arc([x, h * 0.12, x + w * 0.12, h * 0.50], start=0, end=180, fill=(255, 255, 255))


def paint_sailboat(d, w, h, rng, mood):
    """Sailboat."""
    col = (240, 240, 240)
    # hull
    d.polygon([(w * 0.18, h * 0.72), (w * 0.82, h * 0.72), (w * 0.70, h * 0.88), (w * 0.30, h * 0.88)], fill=col)
    # mast
    d.rectangle([w * 0.48, h * 0.22, w * 0.52, h * 0.74], fill=(100, 80, 60))
    # sail
    d.polygon([(w * 0.50, h * 0.26), (w * 0.82, h * 0.70), (w * 0.50, h * 0.70)], fill=col)
    d.polygon([(w * 0.50, h * 0.34), (w * 0.20, h * 0.68), (w * 0.50, h * 0.68)], fill=(220, 220, 230))


def paint_kayak(d, w, h, rng, mood):
    """Kayak."""
    col = _jit(rng, rng.choice([(60, 140, 200), (200, 60, 100), (60, 180, 100)]), 10)
    d.ellipse([w * 0.30, h * 0.36, w * 0.70, h * 0.66], fill=col)
    # cockpit
    d.ellipse([w * 0.42, h * 0.44, w * 0.58, h * 0.58], fill=(40, 40, 50))
    # paddles
    d.line([(w * 0.20, h * 0.50), (w * 0.80, h * 0.50)], fill=(80, 80, 90), width=2)


def paint_parachute(d, w, h, rng, mood):
    """Parachute."""
    col = _jit(rng, rng.choice([(200, 80, 80), (80, 140, 200), (200, 180, 80), (180, 80, 180)]), 12)
    d.ellipse([w * 0.18, h * 0.10, w * 0.82, h * 0.50], fill=col)
    # lines
    for i in range(5):
        x = w * (0.26 + 0.12 * i)
        d.line([(x, h * 0.50), (w * 0.50, h * 0.80)], fill=(100, 100, 110), width=1)


def paint_golfball(d, w, h, rng, mood):
    """Golf ball."""
    col = (240, 245, 250)
    d.ellipse([w * 0.20, h * 0.20, w * 0.80, h * 0.80], fill=col)
    # dimples pattern
    for i in range(4):
        for j in range(4):
            dx = w * (0.30 + 0.14 * i)
            dy = h * (0.30 + 0.14 * j)
            d.ellipse([dx - 2, dy - 2, dx + 2, dy + 2], fill=(230, 235, 240))


def paint_baseball(d, w, h, rng, mood):
    """Baseball."""
    col = (240, 240, 240)
    d.ellipse([w * 0.18, h * 0.18, w * 0.82, h * 0.82], fill=col)
    # stitching
    d.arc([w * 0.40, h * 0.18, w * 0.60, h * 0.82], start=0, end=180, fill=(200, 50, 50), width=2)
    d.arc([w * 0.40, h * 0.18, w * 0.60, h * 0.82], start=180, end=360, fill=(200, 50, 50), width=2)


def paint_football(d, w, h, rng, mood):
    """Football."""
    col = (100, 60, 30)
    d.ellipse([w * 0.18, h * 0.24, w * 0.82, h * 0.76], fill=col)
    # laces
    d.line([(w * 0.42, h * 0.50), (w * 0.58, h * 0.50)], fill=(240, 240, 220), width=2)
    for i in range(3):
        x = w * (0.46 + 0.04 * i)
        d.line([(x, h * 0.44), (x, h * 0.56)], fill=(240, 240, 220), width=1)


# ── REGISTRIES ────────────────────────────────────────────────────────────

EXTRA_PAINTERS = {
    # animals (original)
    "dog": paint_dog,
    "cat": paint_cat,
    "rabbit": paint_rabbit,
    "horse": paint_horse,
    "elephant": paint_elephant,
    "cow": paint_cow,
    "bird": paint_bird,
    "frog": paint_frog,
    "turtle": paint_turtle,
    "snail": paint_snail,
    "kangaroo": paint_kangaroo,
    # NEW animals
    "zebra": paint_zebra,
    "giraffe": paint_giraffe,
    "lion": paint_lion,
    "bear": paint_bear,
    "monkey": paint_monkey,
    "pig": paint_pig,
    "chicken": paint_chicken,
    "fish": paint_fish,
    "spider": paint_spider,
    "snake": paint_snake,
    # tools (original)
    "hammer": paint_hammer,
    "drill": paint_drill,
    "saw": paint_saw,
    "paintbrush": paint_paintbrush,
    "wrench": paint_wrench,
    "screwdriver": paint_screwdriver,
    # materials (original)
    "wood": paint_wood,
    "nail": paint_nail,
    "screw": paint_screw,
    "bolt": paint_bolt,
    "wall": paint_wall,
    "canvas": paint_canvas,
    # household (original)
    "apple": paint_apple,
    "pizza": paint_pizza,
    "table": paint_table,
    "chair": paint_chair,
    "cup": paint_cup,
    "book": paint_book,
    "clock": paint_clock,
    "umbrella": paint_umbrella,
    "tree": paint_tree,
    "flower": paint_flower,
    "house": paint_house,
    "mountain": paint_mountain,
    "boot": paint_boot,
    # NEW foods
    "banana": paint_banana,
    "orange": paint_orange,
    "watermelon": paint_watermelon,
    "strawberry": paint_strawberry,
    "grapes": paint_grapes,
    "lemon": paint_lemon,
    "cherry": paint_cherry,
    "burger": paint_burger,
    "hotdog": paint_hotdog,
    "pancakes": paint_pancakes,
    "icecream": paint_icecream,
    "sushi": paint_sushi,
    "donut": paint_donut,
    # NEW objects
    "laptop": paint_laptop,
    "phone": paint_phone,
    "tv": paint_tv,
    "keyboard": paint_keyboard,
    "mouse_device": paint_mouse_animal,
    "headphones": paint_headphones,
    "trophy": paint_trophy,
    "medal": paint_medal,
    "balloon": paint_balloon,
    "candle": paint_candle,
    "lamp": paint_lamp,
    "bottle": paint_bottle,
    "glass": paint_glass,
    "guitar": paint_guitar,
    "violin": paint_violin,
    "drum": paint_drum,
    "piano": paint_piano,
    "sword": paint_sword,
    "shield": paint_shield,
    "crown": paint_crown,
    "rocket": paint_rocket,
    "ufo": paint_ufo,
    "tent": paint_tent,
    "flag": paint_flag,
    "binoculars": paint_binoculars,
    "telescope": paint_telescope,
    "compass": paint_compass,
    "watch": paint_watch,
    "radio": paint_radio,
    "camera": paint_camera,
    "bicycle_wheel": paint_bicycle_wheel,
    "skateboard": paint_skateboard,
    "surfboard": paint_surfboard,
    "snowman": paint_snowman,
    "ghost": paint_ghost,
    "skull": paint_skull,
    "pumpkin": paint_pumpkin,
    "broom": paint_broom,
    "mop": paint_mop,
    "bucket": paint_bucket,
    "wheelbarrow": paint_wheelbarrow,
    "ladder": paint_ladder,
    "fence": paint_fence,
    "mailbox": paint_mailbox,
    "fireplace": paint_fireplace,
    "chimney": paint_chimney,
    # NEW ocean/nature/sports
    "dolphin": paint_dolphin,
    "whale": paint_whale,
    "shark": paint_shark,
    "crab": paint_crab,
    "octopus": paint_octopus,
    "jellyfish": paint_jellyfish,
    "seahorse": paint_seahorse,
    "tornado": paint_tornado,
    "cloud": paint_cloud,
    "rainbow": paint_rainbow,
    "sun": paint_sun,
    "moon": paint_moon,
    "star": paint_star,
    "heart": paint_heart,
    "beach_umbrella": paint_umbrella_beach,
    "sailboat": paint_sailboat,
    "kayak": paint_kayak,
    "parachute": paint_parachute,
    "golfball": paint_golfball,
    "baseball": paint_baseball,
    "football": paint_football,
}

# (scale_min, scale_max, aspect w/h) — same semantics as make_dataset.GEOMETRY
EXTRA_GEOMETRY = {
    # original animals
    "dog":         (0.55, 0.88, 1.55),
    "cat":         (0.50, 0.84, 1.45),
    "rabbit":      (0.44, 0.78, 1.20),
    "horse":       (0.60, 0.92, 1.45),
    "elephant":    (0.60, 0.94, 1.45),
    "cow":         (0.58, 0.90, 1.50),
    "bird":        (0.44, 0.78, 1.15),
    "frog":        (0.40, 0.74, 1.30),
    "turtle":      (0.48, 0.82, 1.40),
    "snail":       (0.42, 0.78, 1.45),
    "kangaroo":    (0.50, 0.86, 0.95),
    # new animals
    "zebra":       (0.55, 0.88, 1.60),
    "giraffe":     (0.60, 0.94, 0.50),
    "lion":        (0.55, 0.88, 1.45),
    "bear":        (0.55, 0.90, 1.40),
    "monkey":      (0.48, 0.82, 1.20),
    "pig":         (0.52, 0.86, 1.50),
    "chicken":     (0.44, 0.78, 1.10),
    "fish":        (0.44, 0.80, 1.60),
    "spider":      (0.48, 0.80, 1.30),
    "snake":       (0.50, 0.85, 0.40),
    # tools
    "hammer":      (0.48, 0.84, 0.80),
    "drill":       (0.52, 0.88, 1.10),
    "saw":         (0.52, 0.90, 1.55),
    "paintbrush":  (0.40, 0.72, 0.55),
    "wrench":      (0.44, 0.78, 0.55),
    "screwdriver": (0.42, 0.76, 0.50),
    # materials
    "wood":        (0.55, 0.90, 1.25),
    "nail":        (0.34, 0.62, 0.40),
    "screw":       (0.34, 0.62, 0.40),
    "bolt":        (0.34, 0.62, 0.45),
    "wall":        (0.80, 0.98, 1.00),
    "canvas":      (0.60, 0.92, 1.15),
    # original household
    "apple":       (0.40, 0.72, 1.00),
    "pizza":       (0.48, 0.84, 1.00),
    "table":       (0.55, 0.90, 1.35),
    "chair":       (0.48, 0.84, 1.00),
    "cup":         (0.40, 0.72, 1.00),
    "book":        (0.46, 0.80, 1.20),
    "clock":       (0.46, 0.80, 1.00),
    "umbrella":    (0.50, 0.86, 1.10),
    "tree":        (0.55, 0.92, 0.85),
    "flower":      (0.40, 0.74, 0.80),
    "house":       (0.60, 0.94, 1.05),
    "mountain":    (0.85, 1.00, 1.00),
    "boot":        (0.42, 0.76, 1.10),
    # new foods
    "banana":      (0.40, 0.75, 0.45),
    "orange":      (0.42, 0.74, 1.00),
    "watermelon":  (0.48, 0.82, 1.10),
    "strawberry":  (0.38, 0.68, 0.90),
    "grapes":      (0.44, 0.78, 0.85),
    "lemon":       (0.38, 0.68, 1.30),
    "cherry":      (0.36, 0.64, 0.70),
    "burger":      (0.52, 0.86, 1.05),
    "hotdog":      (0.48, 0.82, 1.80),
    "pancakes":    (0.50, 0.84, 1.00),
    "icecream":    (0.40, 0.72, 0.70),
    "sushi":       (0.38, 0.68, 1.00),
    "donut":       (0.42, 0.74, 1.00),
    # new objects
    "laptop":      (0.50, 0.82, 1.40),
    "phone":       (0.40, 0.68, 0.50),
    "tv":          (0.55, 0.90, 1.50),
    "keyboard":    (0.48, 0.80, 1.60),
    "mouse_device": (0.38, 0.66, 1.20),
    "headphones":  (0.44, 0.76, 1.10),
    "trophy":      (0.44, 0.78, 0.60),
    "medal":       (0.38, 0.66, 0.70),
    "balloon":     (0.44, 0.78, 0.75),
    "candle":      (0.38, 0.68, 0.35),
    "lamp":        (0.48, 0.82, 0.90),
    "bottle":      (0.40, 0.70, 0.38),
    "glass":       (0.38, 0.66, 0.65),
    "guitar":      (0.52, 0.86, 0.45),
    "violin":      (0.50, 0.84, 0.40),
    "drum":        (0.48, 0.82, 1.00),
    "piano":       (0.52, 0.88, 1.50),
    "sword":       (0.48, 0.82, 0.18),
    "shield":      (0.50, 0.84, 0.90),
    "crown":       (0.44, 0.76, 1.00),
    "rocket":      (0.50, 0.85, 0.45),
    "ufo":         (0.48, 0.82, 1.30),
    "tent":        (0.55, 0.90, 1.00),
    "flag":        (0.48, 0.80, 0.35),
    "binoculars":  (0.42, 0.74, 1.00),
    "telescope":   (0.50, 0.84, 1.40),
    "compass":     (0.40, 0.70, 1.00),
    "watch":       (0.38, 0.66, 1.00),
    "radio":       (0.48, 0.80, 1.20),
    "camera":      (0.46, 0.78, 1.15),
    "bicycle_wheel": (0.48, 0.82, 1.00),
    "skateboard":  (0.44, 0.76, 2.20),
    "surfboard":   (0.48, 0.82, 0.22),
    "snowman":     (0.55, 0.90, 0.85),
    "ghost":       (0.48, 0.82, 0.85),
    "skull":       (0.44, 0.78, 1.05),
    "pumpkin":     (0.52, 0.86, 1.10),
    "broom":       (0.46, 0.80, 0.22),
    "mop":         (0.46, 0.80, 0.25),
    "bucket":      (0.42, 0.74, 0.95),
    "wheelbarrow": (0.50, 0.84, 1.30),
    "ladder":      (0.55, 0.90, 0.35),
    "fence":       (0.52, 0.88, 1.50),
    "mailbox":     (0.44, 0.76, 0.70),
    "fireplace":   (0.55, 0.90, 1.40),
    "chimney":     (0.44, 0.78, 0.45),
    # ocean/nature/sports
    "dolphin":     (0.52, 0.86, 1.80),
    "whale":       (0.60, 0.95, 2.20),
    "shark":       (0.55, 0.90, 1.90),
    "crab":        (0.44, 0.78, 1.20),
    "octopus":     (0.50, 0.84, 1.00),
    "jellyfish":  (0.46, 0.80, 0.80),
    "seahorse":    (0.40, 0.72, 0.60),
    "tornado":     (0.52, 0.88, 0.60),
    "cloud":       (0.55, 0.92, 1.40),
    "rainbow":     (0.60, 0.95, 1.60),
    "sun":         (0.48, 0.80, 1.00),
    "moon":        (0.44, 0.76, 1.00),
    "star":        (0.44, 0.78, 1.00),
    "heart":       (0.40, 0.72, 1.00),
    "beach_umbrella": (0.50, 0.84, 1.00),
    "sailboat":    (0.52, 0.86, 1.20),
    "kayak":       (0.48, 0.82, 1.80),
    "parachute":   (0.50, 0.84, 1.10),
    "golfball":    (0.32, 0.58, 1.00),
    "baseball":    (0.34, 0.62, 1.00),
    "football":    (0.40, 0.70, 1.50),
}

EXTRA_GROUND = {
    # original
    "bird": "sky",
    "tree": "grass",
    "flower": "grass",
    "mountain": "grass",
    "house": "grass",
    "dog": "grass",
    "cat": "grass",
    "rabbit": "grass",
    "horse": "grass",
    "elephant": "grass",
    "cow": "grass",
    "frog": "grass",
    "turtle": "grass",
    "snail": "grass",
    "kangaroo": "grass",
    "apple": "grass",
    "pizza": "grass",
    "table": "grass",
    "chair": "grass",
    "cup": "grass",
    "book": "grass",
    "clock": "grass",
    "umbrella": "grass",
    "boot": "grass",
    # new animals
    "zebra": "grass",
    "giraffe": "grass",
    "lion": "grass",
    "bear": "grass",
    "monkey": "grass",
    "pig": "grass",
    "chicken": "grass",
    "fish": "water",
    "spider": "grass",
    "snake": "grass",
    # new foods (contextual)
    "banana": "grass",
    "orange": "grass",
    "watermelon": "grass",
    "strawberry": "grass",
    "grapes": "grass",
    "lemon": "grass",
    "cherry": "grass",
    "burger": "road",
    "hotdog": "road",
    "pancakes": "road",
    "icecream": "road",
    "sushi": "road",
    "donut": "road",
    # new objects
    "laptop": "road",
    "phone": "road",
    "tv": "road",
    "keyboard": "road",
    "mouse_device": "road",
    "headphones": "grass",
    "trophy": "grass",
    "medal": "grass",
    "balloon": "sky",
    "candle": "road",
    "lamp": "road",
    "bottle": "water",
    "glass": "road",
    "guitar": "grass",
    "violin": "grass",
    "drum": "grass",
    "piano": "road",
    "sword": "grass",
    "shield": "grass",
    "crown": "grass",
    "rocket": "sky",
    "ufo": "sky",
    "tent": "grass",
    "flag": "grass",
    "binoculars": "grass",
    "telescope": "grass",
    "compass": "grass",
    "watch": "grass",
    "radio": "road",
    "camera": "grass",
    "bicycle_wheel": "road",
    "skateboard": "road",
    "surfboard": "water",
    "snowman": "grass",
    "ghost": "sky",
    "skull": "grass",
    "pumpkin": "grass",
    "broom": "road",
    "mop": "road",
    "bucket": "road",
    "wheelbarrow": "road",
    "ladder": "road",
    "fence": "grass",
    "mailbox": "grass",
    "fireplace": "road",
    "chimney": "grass",
    # ocean/nature/sports
    "dolphin": "water",
    "whale": "water",
    "shark": "water",
    "crab": "water",
    "octopus": "water",
    "jellyfish": "water",
    "seahorse": "water",
    "tornado": "sky",
    "cloud": "sky",
    "rainbow": "sky",
    "sun": "sky",
    "moon": "sky",
    "star": "sky",
    "heart": "grass",
    "beach_umbrella": "grass",
    "sailboat": "water",
    "kayak": "water",
    "parachute": "sky",
    "golfball": "grass",
    "baseball": "grass",
    "football": "grass",
}

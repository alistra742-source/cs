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


# ── registries ────────────────────────────────────────────────────────────

EXTRA_PAINTERS = {
    # animals
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
    # tools
    "hammer": paint_hammer,
    "drill": paint_drill,
    "saw": paint_saw,
    "paintbrush": paint_paintbrush,
    "wrench": paint_wrench,
    "screwdriver": paint_screwdriver,
    # materials
    "wood": paint_wood,
    "nail": paint_nail,
    "screw": paint_screw,
    "bolt": paint_bolt,
    "wall": paint_wall,
    "canvas": paint_canvas,
    # household / terrain
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
}

# (scale_min, scale_max, aspect w/h) — same semantics as make_dataset.GEOMETRY
EXTRA_GEOMETRY = {
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
    "hammer":      (0.48, 0.84, 0.80),
    "drill":       (0.52, 0.88, 1.10),
    "saw":         (0.52, 0.90, 1.55),
    "paintbrush":  (0.40, 0.72, 0.55),
    "wrench":      (0.44, 0.78, 0.55),
    "screwdriver": (0.42, 0.76, 0.50),
    "wood":        (0.55, 0.90, 1.25),
    "nail":        (0.34, 0.62, 0.40),
    "screw":       (0.34, 0.62, 0.40),
    "bolt":        (0.34, 0.62, 0.45),
    "wall":        (0.80, 0.98, 1.00),
    "canvas":      (0.60, 0.92, 1.15),
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
}

EXTRA_GROUND = {
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
}

#!/usr/bin/env python3
"""
gen_variants.py — the "make the drawn ones indistinguishable" tool.

WHY
---
For the hCaptcha illustrated roster the training images must be REAL-LOOKING —
so real that you cannot tell a "drawn" one from a "real" one. Flat vector
illustrations fail that (they are obviously drawn). The fix is REFERENCE-BASED
photoreal generation: each "drawn" image is a photorealistic *variant* of a
real reference photo of the SAME animal in the SAME setting — same subject,
pose, lighting and background, with only subtle natural variation. The result
is a near-twin of the reference: indistinguishable from a real photograph.

LAYOUT  (hcap_gen/<class>/)
---------------------------
    photo_1.jpg, photo_2.jpg         SEEDS  — the "real" reference photos
    photo_3.jpg, photo_4.jpg, ...    VARIANTS — photoreal twins generated FROM
                                         the seeds (indistinguishable from real)
    photo_5.jpg, photo_6.jpg         FLAT — the ~30% "drawn but recognisable"
                                         layer: hCaptcha-style flat
                                         illustrations (clearly drawn, same
                                         animal, similar look). Named photo_*
                                         so the KAGGLE split routes them through
                                         the same centre-crop path.

    Mix per class = ~70% photoreal (photo_1-4) + ~30% flat (photo_5-6).

    _archive/...                     extra generated images, kept on disk but
                                         NOT ingested.

The pipeline ingests every photo_*.jpg in a class folder as real photos
(centre-biased crop, 16 augmented + degraded views each). So a class with 2
seeds + 2 variants trains on 4 indistinguishable real-looking photos.

USING IT
--------
    python gen_variants.py status    # per class: seeds present, variants pending
    python gen_variants.py sheet     # labelled real-vs-drawn preview (QA)
    python gen_variants.py blind     # UNlabelled shuffled mix — the eyeball test

The `SPEC` below is the exact recipe the image engine executes to produce each
variant: pass the seed as the reference image, generate a photoreal near-twin
with only subtle natural variation. Generation itself runs in the authoring
environment (Kaggle/the repo has no image API); the finished images are
committed into hcap_gen/ so the training pipeline just consumes them.
"""
from __future__ import annotations

import os
import random
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(ROOT, "hcap_gen")

# seeds per class = the "real" reference photos that already exist. Variants
# (photo_3..) are generated 1:1 from each seed. Classes with no seeds yet get
# a text-to-image seed first, then reference-based variants.
CLASSES = {
    "raccoon":    ["photo_1.jpg", "photo_2.jpg"],
    "rooster":    ["photo_1.jpg", "photo_2.jpg"],
    "red_panda":  ["photo_1.jpg", "photo_2.jpg"],
    "boar":       ["photo_1.jpg", "photo_2.jpg"],
    "bear":       ["photo_1.jpg", "photo_2.jpg"],
    "lion":       ["photo_1.jpg", "photo_2.jpg"],
    "duck":       ["photo_1.jpg", "photo_2.jpg"],
    "squirrel":   ["photo_1.jpg", "photo_2.jpg"],
    "parrot":     ["photo_1.jpg", "photo_2.jpg"],
    "guitar":     ["photo_1.jpg", "photo_2.jpg"],
    "bat":        ["photo_1.jpg", "photo_2.jpg"],
    "lighthouse": ["photo_1.jpg", "photo_2.jpg"],
    "warthog":    ["photo_1.jpg", "photo_2.jpg"],
}

# scene description baked into every variant prompt (matches the real photo).
SCENE = {
    "raccoon":    "a gray raccoon foraging on a forest floor covered in brown autumn leaves, head lowered, overcast soft light, blurred green woodland background, crisp realistic fur, natural unposed stance",
    "rooster":    "a rooster with red comb and wattle standing in a farmyard, straw and a weathered wooden fence behind, soft morning light, pebbles and dry grass, realistic feather detail",
    "red_panda":  "a red panda sitting on a mossy branch in a misty bamboo forest, rust-orange fur with cream face markings, soft diffused light, green bokeh, natural unposed pose",
    "boar":       "a wild boar grazing in a beech forest clearing, coarse brown fur, dappled sunlight on leaf litter, earthy tones, candid natural pose",
    "bear":       "a brown bear standing in a grassy river meadow with a pine forest behind it, soft morning light, natural stance, thick detailed fur",
    "lion":       "a male lion on golden savanna grass at golden hour, dry amber grassland, blurred acacia trees, warm low sunlight, full dark mane, candid",
    "duck":       "a mallard duck swimming in a pond, glossy green head and orange bill, gentle ripples, blurred reeds and grass at the bank, soft daylight",
    "squirrel":   "a red squirrel perched on a tree branch, reddish-brown fur and a large bushy tail, blurred forest bokeh, dappled light through leaves, candid",
    "parrot":     "a parrot perched on a wooden branch, colorful red green and yellow plumage, blurred tropical foliage background, soft natural light, crisp feather detail",
    "guitar":     "an acoustic guitar standing upright on a wooden floor, warm indoor lighting, blurred living room background, realistic wood grain and strings",
    "bat":        "a bat in natural flight against a dusk sky, wings spread, soft muted light, realistic fur and wing membrane, candid wildlife shot",
    "lighthouse": "a lighthouse on a rocky coastal headland, waves and overcast sky, realistic paintwork and weathering, natural daylight",
    "warthog":    "a warthog standing in dry savanna grass, coarse grey-brown fur and curved tusks, warm daylight, blurred golden grass background, candid",
}

# text-to-image seed prompt (only for classes that still have no seed photo).
SEED_PROMPT = {
    "bat":        "Wildlife photograph of a bat gliding in flight against a dusky sky, wings fully spread, soft muted evening light, realistic fur and delicate wing membrane, candid, no text, no watermark, no border",
    "lighthouse": "Photograph of a lighthouse on a rocky coastal headland, waves breaking below, overcast sky, realistic weathered paintwork, natural daylight, no text, no watermark, no border",
    "warthog":    "Wildlife photograph of a warthog standing in dry savanna grass, coarse grey-brown fur, curved tusks and dewlap, warm daylight, blurred golden grass background, candid, no text, no watermark, no border",
}


def cls_dir(cls):
    return os.path.join(GEN, cls)


def path(cls, fn):
    return os.path.join(cls_dir(cls), fn)


def variant_files(cls):
    """photo_3..photo_(2+n_seeds) — one variant per seed."""
    n = len(CLASSES[cls])
    return ["photo_%d.jpg" % i for i in range(3, 3 + n)]


def flat_files(cls):
    """The ~30% 'drawn but recognisable' layer (hCaptcha-style flats)."""
    return ["photo_5.jpg", "photo_6.jpg"]


def variant_prompt(cls):
    return ("A photorealistic photograph of the same subject in the exact same "
            "scene: %s. Looks exactly like a real camera photograph with only "
            "subtle natural variation in pose and angle, no text, no watermark, "
            "no border" % SCENE[cls])


def status():
    print("%-12s  %10s  %10s   %s" % ("class", "photoreal", "flat ~30%", "mix"))
    for cls in CLASSES:
        seeds = [f for f in CLASSES[cls] if os.path.isfile(path(cls, f))]
        vars_ = [f for f in variant_files(cls) if os.path.isfile(path(cls, f))]
        flats = [f for f in flat_files(cls) if os.path.isfile(path(cls, f))]
        photo = len(seeds) + len(vars_)
        tot = photo + len(flats)
        flatpct = (100 * len(flats) // tot) if tot else 0
        mark = "OK" if (photo >= 4 and len(flats) == 2) else (".." if photo else "NO SEED")
        print("%-12s  %4d/4     %4d/2     %2d%% flat   %s"
              % (cls, min(photo, 4), len(flats), flatpct, mark))


def _panel(im, lab=None, col=None, C=300):
    d = ImageDraw.Draw(im)
    if lab:
        d.rectangle([0, C, im.size[0], C + 24], fill=col)
        d.text((6, C + 6), lab, fill=(255, 255, 255))
    return im


def sheet(labelled=True, out=None):
    C = 300
    rows = []
    for cls in CLASSES:
        items = []
        for f in CLASSES[cls] + variant_files(cls) + flat_files(cls):
            p = path(cls, f)
            if os.path.isfile(p):
                items.append((f, Image.open(p).convert("RGB").resize((C, C))))
        if items:
            rows.append((cls, items))
    maxn = max(len(it) for _, it in rows)
    NW = 110 + maxn * (C + 8)
    NH = 40 + len(rows) * (C + 40)
    s = Image.new("RGB", (NW, NH), (255, 255, 255))
    d = ImageDraw.Draw(s)
    d.text((10, 12),
           "70% photoreal / 30% flat-drawn QA" if labelled
           else "BLIND TEST - can you tell which are drawn?",
           fill=(80, 80, 80))
    y = 40
    for cls, items in rows:
        d.text((10, y + C // 2 - 6), cls, fill=(0, 0, 0))
        for i, (f, im) in enumerate(items):
            x = 110 + i * (C + 8)
            s.paste(im, (x, y))
            if labelled:
                is_photo = f in CLASSES[cls] or f in variant_files(cls)
                col = (0, 130, 0) if is_photo else (200, 110, 0)
                _panel(im, "PHOTOREAL" if is_photo else "DRAWN-FLAT", col, C)
                s.paste(im, (x, y))
    out = out or os.path.join(ROOT, "preview_indistinguishable.jpg")
    s.save(out, quality=92)
    print("wrote", out)
    return out


def blind(out=None):
    """Shuffle all real+drawn together with NO labels — the eyeball test."""
    C = 300
    pool = []
    for cls in CLASSES:
        for f in CLASSES[cls] + variant_files(cls) + flat_files(cls):
            p = path(cls, f)
            if os.path.isfile(p):
                pool.append(Image.open(p).convert("RGB").resize((C, C)))
    random.Random(1234).shuffle(pool)
    cols = 5
    rows = (len(pool) + cols - 1) // cols
    s = Image.new("RGB", (cols * (C + 8), rows * (C + 8)), (240, 240, 240))
    for k, im in enumerate(pool):
        s.paste(im, ((k % cols) * (C + 8), (k // cols) * (C + 8)))
    out = out or os.path.join(ROOT, "preview_blind.jpg")
    s.save(out, quality=92)
    print("wrote", out, "(%d images, no labels)" % len(pool))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    elif cmd == "sheet":
        sheet(labelled=True)
    elif cmd == "blind":
        blind()
    else:
        print(__doc__)

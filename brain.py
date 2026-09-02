#!/usr/bin/env python3
"""
brain.py — the Brain v2: ONE unified, much bigger neural network that solves
EVERY hCaptcha challenge family.

v2 is the heavily upgraded Brain. Everything from v1 is kept (same public API:
Brain, BrainSolver, build_brain_corpus, train_brain, eval_brain, stress_test,
the `python brain.py train|eval|smoke` CLI, the models/brain.pt + brain.json
sidecar, the brain_part_NN split-file distribution) and the following are new:

  BIGGER, MULTI-SCALE VISION
    - preactivation ResNet backbone (width up to 120 -> 960 channels)
    - DUAL feature maps: S/8 (identity) + S/4 (localisation). The heatmap,
      drag and bbox heads fuse both — heatmap resolution doubles (24x24 on a
      96 px scene instead of 12x12), which is the difference between "in
      that third of the image" and "on that object".
    - architecture presets: small / medium / large / mega / giga. `giga`
      (width 160, 13-layer 1024-d prompt transformer, 512-d concepts, 282M
      params) is sized to the ~1.1 GB checkpoint budget and is the 1000-
      class flagship; the smaller presets train on CPU in minutes.

  THE LANGUAGE BRAIN, KNOWING (ALMOST) EVERYTHING
    - PromptEncoder: char-level transformer, dim up to 640, 12 layers,
      160-char prompts.
    - build_router_bank(): a ~30k-pair prompt->family bank generated from
      every class, every synonym in hcaptcha_types.SYNONYMS, article/plural
      surface forms, superlative tables (SIZE/JUMP/SPEED/TEMP), tool
      affordances, materials, set-down wording, drag/pattern/tower/choice/
      text wording. The prompt encoder is trained on all of it, so the Brain
      reads essentially every noun and every phrasing hCaptcha serves.

  WORLD KNOWLEDGE, SEEDED
    - KnowledgeBank (class concept embeddings + relation matrix) is WARM
      STARTED from a structured ontology built out of the repo's hand-coded
      knowledge (categories, SIZE/JUMP/SPEED/TEMP ranks, affordances,
      materials, surfaces) and gently regularised back toward it while
      training (kreg). The pattern reasoner and router therefore start from
      a meaningful world model instead of random vectors.

  REAL HCAPTCHA PIXELS
    - load_hcap_tiles(): ingests a real hCaptcha challenge-image dataset
      (the "hcap" datasets on GitHub — orlov-ai, drandule ~100k, xtekky:
      folder per vehicle class, 128 px tiles) and aliases the folders into
      the 1000-class vocabulary (motorbus -> bus, seaplane -> airplane, ...
      plus the longtail recipes and the colour compounds).
      Every real tile yields several random-crop views with full
      degradation — the tile head finally trains on the actual GAN/photo
      tiles hCaptcha serves, which is what the live vehicle grids are.

  HARD CONDITIONS (blur & friends)
    - _degrade_hard(): motion blur, gaussian blur, gaussian + salt&pepper
      noise, per-channel colour cast, brightness/contrast/saturation
      extremes, hCaptcha dark-mode tint, gamma, vignette, scanlines,
      downscale-then-up, single or DOUBLE JPEG q15-85.
    - every training round rolls clean/soft/hard degradation; a phase-2
      "hardening" pass trains with extra tensor-space hard photometrics and
      confusion-mined class focus (the top-12 most-confused classes get a
      third of the tile steps).
    - _add_clutter(): pastes extra distractor objects into point/count
      scenes (labels stay valid: relational rounds only get distractors
      that lose the superlative, count rounds never get the counted class)
      — "find the dog in a crowded scene".

  FASTER INFERENCE
    - inference-mode + fp16 on GPU, an LRU prompt-vector cache (hCaptcha
      repeats the same prompt wording across rounds — the language brain
      then runs once per unique prompt, not once per tile), single-pass
      scans, batched tile heads.

It is Kaggle-ready: the corpus is generated IN MEMORY from the repo's own
deterministic generators (+ the optional real hcap dataset on disk), so a
notebook needs nothing but `pip install torch numpy Pillow` and a GPU.
See KAGGLE.md for the exact runbook (300k+ images, 100k+ challenge rounds,
~1.1 GB giga brain on a T4/P100).

Drop-in for the shipped solver: BrainSolver exposes the exact method names
TileClassifier / PointLocator / DragLocator use (classify_many, probabilities,
scan, locate, count, locate_relational, locate_drag), plus a single
`solve(...)` entry point that routes a round and returns the answer for ANY
family, confidence-gated so the server can still fall back to the vision
model below the threshold.

CLI
---
    # Kaggle (GPU): the ~1.1 GB 1000-class giga brain on 100k+ rounds:
    python brain.py train --preset giga --device cuda --epochs 14 --phase2 4
        --per_class 310 --n_point 18000 --n_count 12000 --n_drag 14000
        --n_grid 9000 --n_pattern 12000 --n_bbox 10000 --n_pipe 7000
        --n_tower 7000 --n_shape 7000 --n_text 6000
        --hcap_dir /path/to/hcap-dataset --hcap_views 16 \
        --photos_dir /path/to/real_photos --photo_views 16 --split_parts
    # held-out, per-family self-test + the round-solve rate:
    python brain.py eval          # clean held-out
    python brain.py eval --stress # fresh rounds, EVERY image degraded
    # quick smoke (tiny corpus, 1 epoch, CPU):
    python brain.py smoke

Weights land in models/brain.pt (+ models/brain.json sidecar with class
list, family list, sizes, widths and held-out metrics). With --split_parts
the checkpoint is ALSO split into brain_part_NN files (<=96 MB each,
GitHub-friendly) at the repo root — that is how the Test tab
(brain_test.py) reassembles the Brain on any machine.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import hashlib
import pickle
import math
import os
import random
import re
import time

# Works both as a script (ROOT = brain.py's dir) AND when pasted into a
# Kaggle/Jupyter cell (where __file__ is undefined -> ROOT = the notebook's
# working directory, so models/ and data land next to the notebook).
ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() \
    else os.getcwd()
MODELS_DIR = os.environ.get("SOLVER_MODELS_DIR", os.path.join(ROOT, "models"))

# torch is optional at import time: the inference wrapper degrades to
# `available = False` (server.py then falls back to the vision model), exactly
# like tile_classifier.py. Training obviously needs it.
try:  # pragma: no cover
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False
    torch = None
    nn = None
    F = None

# Reduce CUDA allocator fragmentation (matters on 16 GB T4s where a few
# hundred MB of reserved-but-unallocated blocks can turn into OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Decorator that works with AND without torch: the @_no_grad
# decorators below are evaluated at import time, and with torch absent
# (torch = None) they would raise "AttributeError on None" and make the whole
# module unimportable - exactly the failure seen on app hosts that do not
# install torch. With torch missing this is a no-op and BrainSolver degrades
# to available=False (the app's vision fallback path).
if _TORCH:
    _no_grad = torch.no_grad()
else:  # pragma: no cover
    def _no_grad(fn):
        return fn


def _bootstrap_siblings():
    """Make brain.py self-sufficient when run outside the repo.

    brain.py normally lives inside the repo next to hcaptcha_types.py,
    make_dataset.py, make_challenges.py and realdata.py. When it is pasted
    into a Kaggle/Jupyter cell (or copied elsewhere) those siblings are
    missing and the imports below raise ModuleNotFoundError. Here we detect
    that and clone the repo next to the notebook, then put it on sys.path —
    so paste-and-run just works (Kaggle has internet on by default). All four
    siblings are self-contained (stdlib + Pillow), so the clone is all that's
    needed.
    """
    import sys
    try:
        import hcaptcha_types  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    import subprocess
    repo = "https://github.com/alistra742-source/cs.git"
    branch = "arena/01a033e0-cs"
    dst = os.path.join(ROOT, "_cs_repo")
    if not os.path.isfile(os.path.join(dst, "hcaptcha_types.py")):
        os.makedirs(ROOT, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, repo, dst],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "brain.py needs hcaptcha_types.py / make_dataset.py / "
                "make_challenges.py / realdata.py from the repo and they "
                "weren't importable. Auto-clone failed (%r).\n"
                "Make sure internet is ON, or clone the repo first:\n"
                "    !git clone -b %s %s\n"
                "    import sys; sys.path.insert(0, 'cs')" % (e, branch, repo))
    if dst not in sys.path:
        sys.path.insert(0, dst)


_bootstrap_siblings()

import hcaptcha_types as hct
import make_dataset as md

CLASSES = md.CLASSES
N_CLASSES = len(CLASSES)
CID = {n: i for i, n in enumerate(CLASSES)}

# Challenge families the Brain routes between (mirrors hcaptcha_types).
BINARY = hct.BINARY
AREA_POINT = hct.AREA_POINT
AREA_BBOX = hct.AREA_BBOX
DRAG_DROP = hct.DRAG_DROP
MULTIPLE_CHOICE = hct.MULTIPLE_CHOICE
TEXT_ENTRY = hct.TEXT_ENTRY
COUNT = hct.COUNT
PATTERN = "pattern"          # pattern-completion drag (a DRAG_DROP sub-family)
TOWER = "tower"              # wooden-block-tower drag (a DRAG_DROP sub-family)
FAMILIES = [BINARY, AREA_POINT, AREA_BBOX, DRAG_DROP,
            MULTIPLE_CHOICE, TEXT_ENTRY, COUNT, PATTERN, TOWER]
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}

DEFAULT_TILE_SIZE = 64
DEFAULT_SCENE_SIZE = 96

# ── architecture presets ───────────────────────────────────────────────────
# Sized against the ~1.1 GB checkpoint budget the Brain may now use (fp32):
#   small ~70 MB   medium ~140 MB   large ~270 MB   mega ~460 MB
#   giga  ~1.1 GB  (the 1000-class flagship: width 160 backbone,
#                   13-layer 1024-d prompt transformer, 512-d concepts,
#                   8-layer 640-d pattern reasoner — 282M parameters)
# The bulk of the parameters is genuine capacity: the language brain
# (prompt encoder) and the class ontology (knowledge bank), not a padded
# conv backbone.
PRESETS = {
    "small":  dict(width=48,  prompt_dim=320, prompt_layers=6,
                   d_concept=192, pattern_d=192, pattern_layers=3),
    "medium": dict(width=72,  prompt_dim=384, prompt_layers=8,
                   d_concept=256, pattern_d=256, pattern_layers=4),
    "large":  dict(width=96,  prompt_dim=512, prompt_layers=10,
                   d_concept=320, pattern_d=320, pattern_layers=5),
    "mega":   dict(width=120, prompt_dim=640, prompt_layers=12,
                   d_concept=384, pattern_d=384, pattern_layers=6),
    "giga":   dict(width=160, prompt_dim=1024, prompt_layers=13,
                   d_concept=512, pattern_d=640, pattern_layers=8),
}

# Real hCaptcha dataset folder names -> Brain class (the hcap datasets
# label vehicles as motorbus/seaplane/... — the 1000-class vocabulary uses
# bus/airplane; the rest fall through to hct.canonical()).
HCAP_FOLDER_ALIAS = {
    "motorbus": "bus", "seaplane": "airplane", "aeroplane": "airplane",
    "jet": "airplane", "lorry": "truck", "motorbike": "motorcycle",
    "bike": "bicycle", "ship": "boat",
}

# Real-world mix of the offline-capable families, used by the headline
# round-solve metric (multiple_choice is vision-only, so it is weighted out).
FAMILY_WEIGHTS = {
    BINARY: 0.34, AREA_POINT: 0.20, DRAG_DROP: 0.14, COUNT: 0.10,
    PATTERN: 0.08, AREA_BBOX: 0.05, TOWER: 0.04, TEXT_ENTRY: 0.05,
}

# ═══════════════════════════════════════════════════════════════════════════
#  Model (v2: multi-scale backbone, hr localisation heads, seeded ontology)
# ═══════════════════════════════════════════════════════════════════════════

if _TORCH:

    def soft_argmax2d(hm):
        """(B, H, W) heatmap -> (B, 2) normalised (x, y) in 0..1 (top-left).

        Same decode the production train_models.soft_argmax uses: a spatial
        softmax then the expected (x, y). Sub-cell precision, which is what
        makes the heatmap heads accurate (centre-cells plateau, see SOLVER.md).
        """
        B, H, W = hm.shape
        p = F.softmax(hm.reshape(B, H * W), dim=1).reshape(B, H, W)
        ys = (torch.arange(H, dtype=hm.dtype, device=hm.device) + 0.5) / H
        xs = (torch.arange(W, dtype=hm.dtype, device=hm.device) + 0.5) / W
        px = (p.sum(dim=1) * xs).sum(dim=1)
        py = (p.sum(dim=2) * ys).sum(dim=1)
        return torch.stack([px, py], dim=1)

    class ResBlock(nn.Module):
        """Preactivation residual block: x + conv(relu(bn(conv(relu(bn(x))))))."""

        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(cin)
            self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
            self.skip = (nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
                         if (cin != cout or stride != 1) else nn.Identity())

        def forward(self, x):
            h = self.conv1(F.relu(self.bn1(x), inplace=True))
            h = self.conv2(F.relu(self.bn2(h), inplace=True))
            return self.skip(x) + h

    class BrainBackbone(nn.Module):
        """Multi-scale preactivation ResNet: stem + 4 stages.

        Returns BOTH the deep S/8 feature map (identity / class / recognition
        signal) and the shallow S/4 feature map (localisation signal). The
        localisation heads (heatmap / drag / bbox) fuse both — the S/4 map
        doubles the heatmap resolution (24x24 on a 96 px scene instead of
        12x12), which is what separates "in that third of the image" from
        "on that object", and lets counting peaks separate in cluttered
        scenes. Residual shortcuts carry gradient to every head.
        """

        def __init__(self, width: int = 96):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True))
            self.s1 = nn.Sequential(ResBlock(width, width), ResBlock(width, width))
            self.s2 = nn.Sequential(ResBlock(width, 2 * width, stride=2),
                                    ResBlock(2 * width, 2 * width))
            self.s3 = nn.Sequential(ResBlock(2 * width, 4 * width, stride=2),
                                    ResBlock(4 * width, 4 * width))
            self.s4 = nn.Sequential(ResBlock(4 * width, 8 * width, stride=2),
                                    ResBlock(8 * width, 8 * width))
            self.out_channels = 8 * width      # deep map (S/8)
            self.mid_channels = 4 * width      # shallow map (S/4)
            self.final_bn = nn.BatchNorm2d(self.out_channels)

        def forward(self, x):
            x = self.stem(x)
            x = self.s1(x)     # S
            x = self.s2(x)     # S/2
            x = self.s3(x)     # S/4  <- localisation signal
            f4 = x
            x = self.s4(x)     # S/8  <- identity signal
            f8 = F.relu(self.final_bn(x), inplace=True)
            return f8, f4

        def features(self, x):
            return self.forward(x)[0]

    class PromptEncoder(nn.Module):
        """The LANGUAGE BRAIN: prompt string -> dense vector via a character
        transformer.

        This is deliberately where a big share of the Brain's parameters
        live, because reading the question is the core 'smartness' — every
        family's wording, every object noun, every superlative ("jumps the
        highest"), every affordance ("things you can work on with the item
        shown") has to be understood. Character-level tokenisation means
        there is NO fixed vocabulary, so it generalises to words it has
        never seen (helicopter, skyscraper, …) the way the vision fallback
        has to — it learns morphology, not a lookup table.

        v2: 160-char context (full hCaptcha prompts incl. the "…in the
        reference" tail) and dim up to 640 under the mega preset. No
        padding filler: every parameter here is exercised on every forward
        pass, so the size is genuine language capacity, not bloat.
        """

        def __init__(self, dim: int = 512, n_layers: int = 10, nhead: int = 8,
                     max_len: int = 160, ff_mult: int = 4):
            super().__init__()
            self.dim = dim
            self.max_len = max_len
            # printable ASCII vocabulary (~95 chars); id 0 is reserved for pad.
            # Every one of these ids is hit by real prompts, so nothing is
            # wasted capacity.
            self.chars = [chr(c) for c in range(32, 127)]
            self.char2id = {ch: i + 1 for i, ch in enumerate(self.chars)}
            self.pad_id = 0
            vocab = len(self.chars) + 1
            self.emb = nn.Embedding(vocab, dim, padding_idx=0)
            self.pos = nn.Embedding(max_len, dim)
            self.emb_norm = nn.LayerNorm(dim)
            layer = nn.TransformerEncoderLayer(
                dim, nhead, ff_mult * dim, batch_first=True,
                norm_first=True, activation="gelu", dropout=0.0)
            self.enc = nn.TransformerEncoder(layer, n_layers)
            self.out_norm = nn.LayerNorm(dim)

        def _tok(self, text):
            ids = [self.char2id.get(ch, 1)
                   for ch in (text or "").lower()[:self.max_len]]
            if len(ids) < self.max_len:
                ids = ids + [self.pad_id] * (self.max_len - len(ids))
            return ids

        def _batch(self, prompts):
            if isinstance(prompts, str):
                prompts = [prompts]
            t = torch.tensor([self._tok(p) for p in prompts], dtype=torch.long,
                             device=self.emb.weight.device)
            mask = (t == self.pad_id)            # True at pad positions
            return t, mask

        def forward(self, prompts):
            """prompts: str | list[str] (len B) -> (B, dim)."""
            t, mask = self._batch(prompts)
            B, L = t.shape
            x = self.emb_norm(self.emb(t) +
                              self.pos(torch.arange(L, device=t.device)).unsqueeze(0))
            x = self.enc(x, src_key_padding_mask=mask)
            keep = (~mask).float().unsqueeze(-1)           # mean-pool over text
            return self.out_norm((x * keep).sum(1) / keep.sum(1).clamp(min=1))

    # ── world knowledge: structured ontology for the KnowledgeBank ─────────

    CATEGORY_NAMES = ("vehicle", "watercraft", "tool", "animal", "plant",
                      "food", "furniture", "street", "terrain", "material",
                      "household", "structure", "electronics", "clothing",
                      "sports")

    CATEGORY_OF = {
        "bus": "vehicle", "car": "vehicle", "truck": "vehicle",
        "train": "vehicle", "bicycle": "vehicle", "motorcycle": "vehicle",
        "airplane": "vehicle", "boat": "watercraft",
        "traffic_light": "street", "red_light": "street",
        "crosswalk": "street", "fire_hydrant": "street",
        "parking_meter": "street",
        "dog": "animal", "cat": "animal", "rabbit": "animal",
        "horse": "animal", "elephant": "animal", "cow": "animal",
        "bird": "animal", "frog": "animal", "turtle": "animal",
        "snail": "animal", "kangaroo": "animal", "zebra": "animal",
        "giraffe": "animal", "lion": "animal", "bear": "animal",
        "sheep": "animal", "duck": "animal", "fish": "animal",
        "butterfly": "animal",
        "hammer": "tool", "drill": "tool", "saw": "tool",
        "paintbrush": "tool", "wrench": "tool", "screwdriver": "tool",
        "tree": "plant", "flower": "plant", "cactus": "plant",
        "apple": "food", "pizza": "food", "banana": "food",
        "table": "furniture", "chair": "furniture",
        "mountain": "terrain",
        "wood": "material", "nail": "material", "screw": "material",
        "bolt": "material", "wall": "material", "canvas": "material",
        "cup": "household", "book": "household", "clock": "household",
        "umbrella": "household", "boot": "household", "guitar": "household",
        "house": "structure",
    }

    # 1000-class extension: the longtail categories (make_longtail) + the
    # colour compounds (each inherits its base object's category).
    try:
        import make_longtail as _mlt_onto
        for _n in _mlt_onto.LONGTAIL_NAMES:
            _cat = _mlt_onto.LONGTAIL_CATEGORY[_n]
            if _cat == "nature":
                _cat = "terrain"
            if (_cat == "vehicle"
                    and _mlt_onto.longtail_ground_kind(_n) == "water"):
                _cat = "watercraft"
            CATEGORY_OF.setdefault(_n, _cat)
        for _n, _b, _c, _rgb in _mlt_onto.COMPOUNDS:
            CATEGORY_OF.setdefault(_n, CATEGORY_OF.get(_b, "household"))
    except Exception:  # pragma: no cover
        pass

    # Things you can set items down on / paint / work on.
    _SURFACES = ("table", "chair", "wood", "wall", "canvas", "house")

    ONTOLOGY_DIM = 31  # 15 category bits + 16 attribute dims

    def ontology_targets(n_classes: int, classes):
        """(n_classes, ONTOLOGY_DIM) structured WORLD-KNOWLEDGE vectors.

        This is the hand-coded prompt catalog (SIZE/JUMP/SPEED/TEMP ranks,
        tool affordances, material sets, category membership) turned into a
        dense per-class vector. The KnowledgeBank is warm-started from it,
        so the pattern reasoner and the router start from a meaningful
        ontology instead of random vectors — and a small MSE regulariser
        keeps it anchored there while training.
        """
        import numpy as _np
        T = _np.zeros((n_classes, ONTOLOGY_DIM), dtype=_np.float32)
        cat_id = {c: i for i, c in enumerate(CATEGORY_NAMES)}
        for i, name in enumerate(classes[:n_classes]):
            cat = CATEGORY_OF.get(name)
            if cat in cat_id:
                T[i, cat_id[cat]] = 1.0
            a = len(CATEGORY_NAMES)
            T[i, a] = min(1.0, hct.SIZE_RANK.get(name, 15) / 35.0)
            T[i, a + 1] = hct.JUMP_RANK.get(name, 0) / 11.0
            T[i, a + 2] = hct.SPEED_RANK.get(name, 0) / 18.0
            T[i, a + 3] = hct.TEMP_RANK.get(name, 0) / 10.0
            T[i, a + 4] = 1.0 if cat in ("animal", "plant") else 0.0
            T[i, a + 5] = 1.0 if name in hct.WHEELED else 0.0
            T[i, a + 6] = 1.0 if name in hct.MOTORISED else 0.0
            T[i, a + 7] = 1.0 if name in hct.EDIBLE else 0.0
            T[i, a + 8] = 1.0 if name in hct.FURRY else 0.0
            T[i, a + 9] = 1.0 if name in hct.METAL else 0.0
            T[i, a + 10] = 1.0 if name in hct.WOODEN else 0.0
            T[i, a + 11] = 1.0 if name in hct.PLANTS else 0.0
            T[i, a + 12] = 1.0 if name in hct.TOOLS else 0.0
            T[i, a + 13] = (sum(1 for t in hct.TOOL_AFFORDANCE.values()
                                if name in t) / 6.0)
            T[i, a + 14] = len(hct.TOOL_AFFORDANCE.get(name, ())) / 6.0
            T[i, a + 15] = 1.0 if name in _SURFACES else 0.0
        return T

    class KnowledgeBank(nn.Module):
        """The Brain's WORLD KNOWLEDGE: a learned ontology of 1000 classes.

        Each class owns a rich CONCEPT embedding (what it *means*: a bus and
        a truck share vehicle-ness; a drill and a hammer share tool-ness; a
        cat and a dog share animal-ness) plus a class->class RELATION matrix
        that captures affordances and category ties (drill -> wood/wall,
        animals vs vehicles, same size tier).

        v2: WARM-STARTED from the structured ontology above (concept rows
        initialised with the category/attribute vector; the relation matrix
        with semantic similarity + affordances, minus the deliberate
        red_light/traffic_light exclusion), then refined by training with a
        small MSE anchor (kreg) so the ontology stays meaningful. The
        pattern reasoner reasons over these concept tokens, so it solves
        "put one of the animals into the empty spot to complete the pattern"
        by THINKING about what each cell IS (its concept), not just by
        pixels — a learned Latin-square solver that can also do analogies
        the hand-coded resolver refuses.
        """

        def __init__(self, n_classes=N_CLASSES, d_concept=320, warm_start=True):
            super().__init__()
            self.n_classes = n_classes
            self.concept = nn.Embedding(n_classes, d_concept)
            self.rel = nn.Parameter(torch.zeros(n_classes, n_classes))
            self.norm = nn.LayerNorm(d_concept)
            if warm_start:
                self._warm_start()

        def _warm_start(self):
            T = ontology_targets(self.n_classes, CLASSES)   # (C, 28) f32
            w = self.concept.weight
            with torch.no_grad():
                for i in range(self.n_classes):
                    row = torch.from_numpy(T[i]).float()
                    nrm = row.norm().clamp(min=1e-8)
                    row = row / nrm * math.sqrt(float(row.numel()))
                    w[i, :row.numel()] = row
                    if w.shape[1] > row.numel():
                        w[i, row.numel():] = torch.randn(w.shape[1] - row.numel()) * 0.02
                # relation matrix: semantic similarity + affordances
                Tv = torch.from_numpy(T).float()
                nrm = Tv.norm(dim=1).clamp(min=1e-8)
                sim = (Tv @ Tv.t()) / (nrm.unsqueeze(1) * nrm.unsqueeze(0))
                rel = 0.5 * sim.clone()
                for t_name, surfaces in hct.TOOL_AFFORDANCE.items():
                    if t_name not in CID:
                        continue
                    ti = CID[t_name]
                    for s_name in surfaces:
                        if s_name in CID:
                            rel[ti, CID[s_name]] += 1.5
                            rel[CID[s_name], ti] += 0.75
                # deliberate exclusion: red_light vs traffic_light are
                # OPPOSITE labels (only one may be lit red)
                for a, b in (("red_light", "traffic_light"),):
                    if a in CID and b in CID:
                        rel[CID[a], CID[b]] -= 2.0
                        rel[CID[b], CID[a]] -= 2.0
                self.rel.data = rel

        def from_probs(self, probs):
            """(..., C) class distribution -> (..., d_concept) expected concept.

            How a cell/candidate of uncertain identity projects into concept
            space: the weighted concept of every class it might be. A confident
            cell maps to one concept; the empty/hole cell (flat distribution)
            maps near the concept centroid, which the reasoner learns to treat
            as 'unknown slot'."""
            return self.norm(probs @ self.concept.weight)

        def lookup(self, ids):
            return self.norm(self.concept(ids))

    class TileHead(nn.Module):
        """60-way tile classifier (global avg-pool -> FC)."""

        def __init__(self, c_in, n_classes):
            super().__init__()
            self.fc = nn.Linear(c_in, n_classes)

        def forward(self, feat):
            return self.fc(F.adaptive_avg_pool2d(feat, 1).flatten(1))

        def from_pooled(self, pooled):
            """Class logits from an already-pooled (B, c) feature vector — used
            by the pattern reasoner, which runs the backbone under no_grad and
            then labels each crop so it can think in concepts."""
            return self.fc(pooled)

    class HeatmapHead(nn.Module):
        """Dual-resolution (n_classes + 1 background) spatial channels.

        A 1x1 conv on the deep S/8 map (identity-conditioned location) is
        upsampled and ADDED to a 1x1 conv on the shallow S/4 map (fine
        spatial detail) — the fused map is 2x the v1 resolution (24x24 on a
        96 px scene). One channel per class so a single forward pass
        localises EVERY class (point / scan / count) and the per-cell
        background channel suppresses phantom presences.
        """

        def __init__(self, c_lo, c_hi, n_classes):
            super().__init__()
            self.lo = nn.Conv2d(c_lo, n_classes + 1, 1)
            self.hi = nn.Conv2d(c_hi, n_classes + 1, 1)

        def forward(self, f_lo, f_hi=None):
            if f_hi is None:      # degraded single-resolution fallback
                return F.interpolate(self.lo(f_lo), scale_factor=2,
                                     mode="nearest")
            out = self.hi(f_hi)
            low = F.interpolate(self.lo(f_lo), size=out.shape[-2:],
                                mode="nearest")
            return out + low

    class DragHead(nn.Module):
        """Dual-resolution 2 channels: piece (drag-from) and slot (drag-to)."""

        def __init__(self, c_lo, c_hi):
            super().__init__()
            self.lo = nn.Conv2d(c_lo, 2, 1)
            self.hi = nn.Conv2d(c_hi, 2, 1)

        def forward(self, f_lo, f_hi=None):
            if f_hi is None:
                return F.interpolate(self.lo(f_lo), scale_factor=2,
                                     mode="nearest")
            out = self.hi(f_hi)
            low = F.interpolate(self.lo(f_lo), size=out.shape[-2:],
                                mode="nearest")
            return out + low

    class BBoxHead(nn.Module):
        """area_select_bbox: dual-resolution centre heatmap + global (w, h).

        Bbox rounds have exactly one target object, so a centre heatmap
        (decoded with soft-argmax at 2x resolution) plus a single global
        width/height regression is the right shape and trains cleanly from
        the in-memory bbox generator.
        """

        def __init__(self, c_lo, c_hi):
            super().__init__()
            self.center_lo = nn.Conv2d(c_lo, 1, 1)
            self.center_hi = nn.Conv2d(c_hi, 1, 1)
            self.size = nn.Sequential(
                nn.Linear(c_lo, c_lo), nn.ReLU(inplace=True), nn.Linear(c_lo, 2))

        def forward(self, f_lo, f_hi=None):
            if f_hi is None:
                ctr = F.interpolate(self.center_lo(f_lo), scale_factor=2,
                                    mode="nearest")[:, 0]
            else:
                ctr_hi = self.center_hi(f_hi)[:, 0]
                ctr_lo = F.interpolate(self.center_lo(f_lo),
                                       size=ctr_hi.shape[-2:], mode="nearest")[:, 0]
                ctr = ctr_hi + ctr_lo
            pooled = F.adaptive_avg_pool2d(f_lo, 1).flatten(1)
            wh = torch.sigmoid(self.size(pooled))         # (B, 2) in 0..1
            return ctr, wh

    class TextHead(nn.Module):
        """text_entry ("Type the text you see"): column-strip pooling -> one
        36-way (A-Z0-9) classification per character position.

        Position-aware by construction: the feature map is pooled into
        text_len vertical STRIPS (one per character column) and each strip is
        classified separately. A global average pool would destroy the
        horizontal position of each character and make per-position
        prediction impossible (that was the v1 bug - the head could not
        learn)."""

        def __init__(self, c_in, text_len=5, n_chars=36):
            super().__init__()
            self.text_len = text_len
            self.n_chars = n_chars
            self.fc = nn.Linear(c_in, n_chars)

        def forward(self, feat):
            # (B, C, H, W) -> (B, C, 1, L) -> (B, L, C) -> (B, L*n_chars)
            strips = F.adaptive_avg_pool2d(feat, (1, self.text_len))
            strips = strips.squeeze(2).permute(0, 2, 1)      # (B, L, C)
            out = self.fc(strips)                            # (B, L, 36)
            return out.flatten(1)

    class RouterHead(nn.Module):
        """Learned (prompt + image) -> family classifier.

        Trained from the manifest `type` of every generated round plus the
        ~30k-pair prompt bank (build_router_bank), which covers all 9
        families with many wordings and every class name + synonym, so the
        router sees each family often. The rule router
        (hcaptcha_types.classify) is the production source of truth; this
        head is a learnable, image-aware alternative / cross-check.
        """

        def __init__(self, c_in, prompt_dim, n_families):
            super().__init__()
            self.img = nn.Sequential(nn.Linear(c_in, 256), nn.ReLU(inplace=True))
            self.txt = nn.Sequential(nn.Linear(prompt_dim, 256), nn.ReLU(inplace=True))
            self.out = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, n_families))

        def forward(self, img_pool, prompt_vec):
            return self.out(torch.cat([self.img(img_pool), self.txt(prompt_vec)], dim=1))

    class PatternReasoner(nn.Module):
        """Set-transformer pattern solver that THINKS IN CONCEPTS.

        Each cell/candidate is labelled by the tile head -> a class
        distribution -> a CONCEPT token (KnowledgeBank.from_probs). The token
        the transformer attends over is therefore "what is this thing?" (its
        learned meaning), not raw pixels — so the net can complete a pattern
        by reasoning "row has cat,dog,? ; the missing one must be the third
        animal", the way the hand-coded Latin-square resolver does, but
        learned and able to generalise. A residual visual token is added so
        appearance still informs identity. Roles: 0-8 cells, 9-11 candidates,
        12 prompt.
        """

        def __init__(self, c_in, prompt_dim, d_concept,
                     d_model=320, nhead=4, layers=5):
            super().__init__()
            self.d_model = d_model
            self.proj_vis = nn.Linear(c_in, d_model)
            self.proj_concept = nn.Linear(d_concept, d_model)
            self.proj_prompt = nn.Linear(prompt_dim, d_model)
            self.role = nn.Embedding(13, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, nhead, 4 * d_model, batch_first=True,
                norm_first=True, activation="gelu", dropout=0.0)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.score = nn.Linear(d_model, 1)

        def forward(self, vis_cells, concept_cells, vis_cands,
                    concept_cands, prompt_vec):
            # vis_cells: (B,9,c_in)  concept_cells: (B,9,d_concept)
            cell_tok = self.proj_vis(vis_cells) + self.proj_concept(concept_cells)
            cand_tok = self.proj_vis(vis_cands) + self.proj_concept(concept_cands)
            prompt_tok = self.proj_prompt(prompt_vec).unsqueeze(1)   # (B,1,d)
            tokens = torch.cat([cell_tok, cand_tok, prompt_tok], dim=1)  # (B,13,d)
            roles = torch.arange(13, device=tokens.device)
            tokens = self.enc(tokens + self.role(roles).unsqueeze(0))
            return self.score(tokens[:, 9:12]).squeeze(-1)          # (B,3)

    class Brain(nn.Module):
        """The whole network: shared multi-scale backbone + knowledge +
        every head.

        Architecture hyper-parameters are passed explicitly and persisted in
        the sidecar, so a trained Brain reloads with the EXACT shape it was
        built with — no load-time shape mismatches, ever. `version` marks
        the v2 (multi-scale) checkpoint format.
        """

        def __init__(self, n_classes=N_CLASSES, width=96,
                     prompt_dim=512, prompt_layers=10, d_concept=320,
                     pattern_d=320, pattern_layers=5,
                     text_len=5,
                     n_families=len(FAMILIES), version=2):
            super().__init__()
            self.n_classes = n_classes
            self.width = width
            self.prompt_dim = prompt_dim
            self.prompt_layers = prompt_layers
            self.d_concept = d_concept
            self.pattern_d = pattern_d
            self.pattern_layers = pattern_layers
            self.text_len = text_len
            self.n_families = n_families
            self.version = version
            self.backbone = BrainBackbone(width)
            c8 = self.backbone.out_channels     # deep, S/8
            c4 = self.backbone.mid_channels     # shallow, S/4
            self.prompt_enc = PromptEncoder(prompt_dim, prompt_layers)
            self.knowledge = KnowledgeBank(n_classes, d_concept)
            self.tile_head = TileHead(c8, n_classes)
            self.heatmap_head = HeatmapHead(c8, c4, n_classes)
            self.drag_head = DragHead(c8, c4)
            self.bbox_head = BBoxHead(c8, c4)
            self.text_head = TextHead(c8, text_len)
            self.router_head = RouterHead(c8, prompt_dim, n_families)
            self.pattern_reasoner = PatternReasoner(
                c8, prompt_dim, d_concept, pattern_d, layers=pattern_layers)

        # ── per-head forward helpers (each reuses the shared backbone) ──
        def features2(self, x):
            """(f8, f4) — deep S/8 and shallow S/4 feature maps."""
            return self.backbone(x)

        def features(self, x):
            return self.backbone.features(x)

        def tile_logits(self, f8):
            return self.tile_head(f8)

        def heatmaps(self, f8, f4=None):
            """(B, n_classes+1, H, W) fused dual-resolution logits."""
            return self.heatmap_head(f8, f4)

        def drag_maps(self, f8, f4=None):
            return self.drag_head(f8, f4)       # (B, 2, H, W)

        def bbox(self, f8, f4=None):
            return self.bbox_head(f8, f4)       # center (B,H,W), wh (B,2)

        def text_logits(self, f8):
            """(B, text_len*36) per-character logits for text rounds."""
            return self.text_head(f8)

        def route(self, f8, prompt_vec):
            pool = F.adaptive_avg_pool2d(f8, 1).flatten(1)
            return self.router_head(pool, prompt_vec)

        def pattern(self, vis_cells, vis_cands, prompt_vec):
            """Concept-aware pattern solve.

            vis_cells: (B,9,c)  vis_cands: (B,3,c) — pooled backbone features
            (the caller runs the backbone under no_grad for speed). Each crop
            is labelled by the tile head, its class distribution is projected
            into concept space by the KnowledgeBank, then the set-transformer
            reasons over (visual + concept) tokens. Gradient still flows into
            the tile head, the knowledge bank, the prompt encoder and the
            reasoner — the four 'thinking' modules — while the heavy backbone
            stays frozen on pattern steps."""
            B = vis_cells.shape[0]
            C = self.n_classes
            pc = F.softmax(
                self.tile_head.from_pooled(vis_cells.reshape(B * 9, -1)),
                dim=-1).reshape(B, 9, C)
            pa = F.softmax(
                self.tile_head.from_pooled(vis_cands.reshape(B * 3, -1)),
                dim=-1).reshape(B, 3, C)
            cc = self.knowledge.from_probs(pc)     # (B,9,d_concept)
            ca = self.knowledge.from_probs(pa)     # (B,3,d_concept)
            return self.pattern_reasoner(vis_cells, cc, vis_cands, ca,
                                         prompt_vec)

        def param_mb(self):
            return sum(p.numel() for p in self.parameters()) * 4 / 1e6

# ═══════════════════════════════════════════════════════════════════════════
#  Data: generate the full multi-task corpus in memory (Kaggle-friendly)
# ═══════════════════════════════════════════════════════════════════════════

def _img_to_u8(im, size):
    im = im.convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1)


def _to_float(x):
    return (x.float() / 255.0 - 0.5) / 0.5


def _degrade_pil(im, rng):
    """SOFT degradation - the kind real screenshots carry: JPEG compression
    (q35-85), gaussian blur, sensor noise, brightness/contrast/colour shift,
    low-res resampling. (v1's degrade; kept as the 'soft' tier.)"""
    import io as _io
    from PIL import Image as _Im, ImageEnhance, ImageFilter
    resample = getattr(_Im, "Resampling", _Im).BILINEAR
    im = im.convert("RGB")
    if rng.random() < 0.8:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.70, 1.30))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.75, 1.25))
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.70, 1.30))
    if rng.random() < 0.6:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4)))
    if rng.random() < 0.6:
        arr = np.asarray(im).astype(np.int16)
        rs = np.random.RandomState(rng.randrange(1 << 30))
        arr = np.clip(arr + (rs.randn(*arr.shape)
                             * rng.uniform(4, 14)).astype(np.int16), 0, 255)
        im = _Im.fromarray(arr.astype(np.uint8))
    if rng.random() < 0.8:
        buf = _io.BytesIO()
        im.save(buf, "JPEG", quality=rng.randint(35, 85))
        buf.seek(0)
        im = _Im.open(buf).convert("RGB")
    if rng.random() < 0.4:
        w, h = im.size
        f = rng.uniform(0.55, 0.80)
        im = im.resize((max(8, int(w * f)), max(8, int(h * f))),
                       resample).resize((w, h), resample)
    return im


def _motion_blur_pil(im, rng):
    """Directional (motion) blur: a box blur of length 5-15 px along a random
    angle (rotate -> numpy cumulative-sum box blur -> rotate back) —
    simulates screen tearing / capture jitter. (Pillow's Kernel filter only
    allows 3x3/5x5, so the blur runs in numpy.)"""
    L = rng.choice([5, 9, 15])
    ang = rng.uniform(0, 360)
    r = im.rotate(-ang, resample=Image.BILINEAR, expand=False)
    a = np.asarray(r, dtype=np.float32)
    for c in range(3):
        ch = a[..., c]
        cp = np.pad(ch, ((0, 0), (0, L)), mode="edge")
        cs = cp.cumsum(axis=1)
        a[..., c] = (cs[:, L:] - cs[:, :-L]) / float(L)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).rotate(
        ang, resample=Image.BILINEAR, expand=False)
    if out.size != im.size:
        out = out.resize(im.size, Image.BILINEAR)
    return out


def _degrade_hard(im, rng):
    """HARD degradation - the nasty end of the screenshot distribution:
    motion blur, gaussian blur, gaussian + salt&pepper noise, per-channel
    colour cast, brightness/contrast/saturation extremes, hCaptcha dark-mode
    tint, gamma, vignette, scanlines, downscale-then-up, and single or
    DOUBLE JPEG compression (q15-85). Every training round rolls
    clean/soft/hard so the Brain learns to see through all of it, and the
    stress test degrades EVERY held-out round with this.
    """
    import io as _io
    from PIL import Image as _Im, ImageEnhance, ImageFilter
    resample = getattr(_Im, "Resampling", _Im).BILINEAR
    im = im.convert("RGB")
    # downscale (small rendered tiles / bad capture)
    if rng.random() < 0.7:
        w, h = im.size
        f = rng.uniform(0.35, 0.8)
        im = im.resize((max(8, int(w * f)), max(8, int(h * f))), resample)
    # blur: motion or gaussian
    if rng.random() < 0.75:
        if rng.random() < 0.45:
            im = _motion_blur_pil(im, rng)
        else:
            im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.4)))
    # gaussian noise
    if rng.random() < 0.7:
        a = np.asarray(im).astype(np.int16)
        rs = np.random.RandomState(rng.randrange(1 << 30))
        a = a + (rs.randn(*a.shape) * rng.uniform(5, 16)).astype(np.int16)
        im = _Im.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    # salt & pepper
    if rng.random() < 0.4:
        a = np.asarray(im).astype(np.int16)
        rs = np.random.RandomState(rng.randrange(1 << 30))
        m = rs.rand(*a.shape[:2])
        a[m < 0.012] = 255
        a[(m >= 0.012) & (m < 0.026)] = 0
        im = _Im.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    # per-channel colour cast
    g = (rng.uniform(0.70, 1.30), rng.uniform(0.70, 1.30),
         rng.uniform(0.70, 1.30))
    r, gr, b = im.split()
    im = _Im.merge("RGB", (
        r.point(lambda v, m=g[0]: min(255, int(v * m))),
        gr.point(lambda v, m=g[1]: min(255, int(v * m))),
        b.point(lambda v, m=g[2]: min(255, int(v * m)))))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.55, 1.45))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.55, 1.50))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.35, 1.60))
    # hCaptcha dark-mode tint (the widget's dark theme tints tiles blue)
    if rng.random() < 0.5:
        a = np.asarray(im).astype(np.float32)
        a *= (0.84, 0.87, 1.00)
        a += (4, 6, 12)
        im = _Im.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    # gamma
    if rng.random() < 0.5:
        g = rng.uniform(0.65, 1.60)
        lut = [int(255 * ((i / 255.0) ** (1.0 / g))) for i in range(256)]
        im = im.point(tuple(lut * 3))   # RGB needs 3x256 entries
    # vignette
    if rng.random() < 0.35:
        w, h = im.size
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        d = np.sqrt(((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2)
        fall = 1.0 - 0.20 * np.clip(d - 0.5, 0, 1.25) / 1.25
        a = np.asarray(im).astype(np.float32) * fall[..., None]
        im = _Im.fromarray(a.astype(np.uint8))
    # scanlines
    if rng.random() < 0.3:
        a = np.asarray(im).astype(np.int16)
        a[::2, :] = np.clip(a[::2, :] * 0.92, 0, 255)
        im = _Im.fromarray(a.astype(np.uint8))
    # JPEG, possibly double-compressed
    passes = (rng.randint(15, 60),)
    if rng.random() < 0.4:
        passes = passes + (rng.randint(15, 45),)
    for q in passes:
        buf = _io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        buf.seek(0)
        im = _Im.open(buf).convert("RGB")
    return im


def _degrade(im, rng, mode):
    """Dispatch: mode in {'clean','soft','hard'}."""
    if mode == "soft":
        return _degrade_pil(im, rng)
    if mode == "hard":
        return _degrade_hard(im, rng)
    return im


def _degrade_roll(rng, hard_frac, soft_frac):
    r = rng.random()
    if r < hard_frac:
        return "hard"
    if r < hard_frac + soft_frac:
        return "soft"
    return "clean"


def _count_peaks(chan, min_peak=0.08, min_sep=0.16, weak_gate=0.20,
                 max_n=9, margin=0.04):
    """Count instances from one class's presence channel (H, W).

    Same self-gating peak counter as the production PointLocator.count: a
    count answer is graded EXACTLY, so border-touching peaks, fragmented maps
    or a weak weakest-peak return None (defer to the vision model). Shared by
    BrainSolver.count and the held-out eval so both measure the same thing."""
    H, W = chan.shape
    peaks = []
    for y in range(H):
        for x in range(W):
            v = float(chan[y, x])
            if v < min_peak:
                continue
            y0, y1 = max(0, y - 1), min(H, y + 2)
            x0, x1 = max(0, x - 1), min(W, x + 2)
            if v < float(chan[y0:y1, x0:x1].max()):
                continue
            peaks.append((v, x, y))
    peaks.sort(reverse=True)
    kept = []
    for v, x, y in peaks:
        if all(max(abs(x - kx), abs(y - ky)) >= min_sep * W
               for _, kx, ky in kept):
            kept.append((v, x, y))
    if not kept:
        return None
    for _, x, y in kept:
        if (x / W) < margin or (x / W) > 1 - margin \
                or (y / H) < margin or (y / H) > 1 - margin:
            return None
    if len(kept) >= max_n or kept[-1][0] < weak_gate:
        return None
    return len(kept)


def _add_clutter(img, meta, rng, max_extra=3):
    """Hard condition: paste EXTRA distractor objects into an existing
    point/count scene, never changing the correct answer.

    Label validity: count rounds only get distractors of a DIFFERENT class
    (the counted instances stay the only target-class objects); relational
    point rounds only get distractors whose superlative table value LOSES to
    the target (no ties), so the argmax target stays unique. Distractors are
    recorded in meta['clutter'] (kept OUT of meta['objects'] so the count
    mask supervision stays exact).
    """
    from make_challenges import _paste_object, POINT_CLASSES
    S = img.size[0]
    tname = meta.get("target")
    objs = list(meta.get("objects", []))
    table, direction = (None, None)
    if meta.get("relational"):
        sup = hct.superlative_table(meta.get("prompt", ""))
        if sup is not None:
            table, direction = sup
    tval = table.get(tname) if (table is not None and tname in table) else None
    clutter = []
    banned = {tname}
    for _ in range(max_extra):
        cands = [c for c in POINT_CLASSES if c not in banned]
        if not cands:
            break
        c = rng.choice(cands)
        if table is not None and c in table and tval is not None:
            if direction == "max" and table[c] >= tval:
                continue
            if direction == "min" and table[c] <= tval:
                continue
        for _try in range(14):
            x = rng.uniform(0.08, 0.92)
            y = rng.uniform(0.08, 0.92)
            size = rng.uniform(0.22, 0.36)
            r = size * 0.5
            ok = True
            for o in objs + clutter:
                if math.hypot(o["x"] - x, o["y"] - y) < (o.get("r", 0.2) + r) * 0.95:
                    ok = False
                    break
            if not ok:
                continue
            px, py, pr = _paste_object(img, c, (x, y, size), rng)
            clutter.append({"name": c, "x": round(px, 4), "y": round(py, 4),
                            "r": round(pr, 4)})
            break
    if clutter:
        meta["clutter"] = clutter
        meta["cluttered"] = True
    return img, meta


def _hcap_view(im, tile_size, rng):
    """One random training view of a real 128 px hCaptcha tile: a random
    cover-crop (zoom 0.72-1.0), optional flip and small rotation, then a
    resize to the Brain's tile size."""
    w, h = im.size
    side = min(w, h)
    z = rng.uniform(0.72, 1.0)
    cs = max(8, int(side * z))
    x0 = rng.randint(0, max(0, w - cs))
    y0 = rng.randint(0, max(0, h - cs))
    im = im.crop((x0, y0, x0 + cs, y0 + cs))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    ang = rng.uniform(-8, 8)
    if abs(ang) > 0.5:
        im = im.rotate(ang, resample=Image.BILINEAR, expand=True)
    return im.resize((tile_size, tile_size), Image.LANCZOS)


def load_hcap_tiles(hcap_dir, tile_size=DEFAULT_TILE_SIZE, views=16,
                    hard_frac=0.5, seed=11, verbose=True):
    """Ingest a REAL hCaptcha challenge-image dataset — the "hcap" datasets
    (GitHub: orlov-ai/hcaptcha-dataset, drandule/hcaptcha_dataset ~100k,
    xtekky/hcaptcha-dataset): one folder per vehicle class, 128 px tiles.

    Folders are aliased into the Brain's 1000-class vocabulary (motorbus ->
    bus, seaplane -> airplane, lorry -> truck, ...; anything else goes
    through hct.canonical). Each real tile yields `views` random views with
    a clean/soft/hard degradation roll, so a few thousand real tiles become
    tens of thousands of robust training images.

    Returns (x: (N,3,S,S) uint8, y: (N,) long class ids, n_files).
    """
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    if not hcap_dir or not os.path.isdir(hcap_dir):
        log("    hcap: %r is not a directory - skipping real tiles" % hcap_dir)
        return torch.empty((0, 3, tile_size, tile_size), dtype=torch.uint8), \
            torch.empty((0,), dtype=torch.long), 0
    dirs = {}
    for sub in sorted(os.listdir(hcap_dir)):
        p = os.path.join(hcap_dir, sub)
        if not os.path.isdir(p):
            continue
        folder = sub.lower().replace("-", "_")
        cls = HCAP_FOLDER_ALIAS.get(folder) or hct.canonical(folder)
        if cls not in CID:
            log("    hcap: folder %r -> %r (not in the classes, skipped)"
                % (sub, cls))
            continue
        files = sorted(f for f in os.listdir(p)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if files:
            dirs[sub] = (cls, files)
            log("    hcap: %-12s -> %-10s %d real tiles" % (sub, cls, len(files)))
    if not dirs:
        log("    hcap: no usable class folders in %s" % hcap_dir)
        return torch.empty((0, 3, tile_size, tile_size), dtype=torch.uint8), \
            torch.empty((0,), dtype=torch.long), 0
    xs, ys = [], []
    t0 = time.time()
    n_files = 0
    for sub, (cls, files) in dirs.items():
        cid = CID[cls]
        for fi, fn in enumerate(files):
            n_files += 1
            try:
                im = Image.open(os.path.join(hcap_dir, sub, fn)).convert("RGB")
            except Exception:
                continue
            for v in range(views):
                rng = random.Random("hcap|%d|%s|%d|%d" % (seed, sub, fi, v))
                view = _hcap_view(im, tile_size, rng)
                view = _degrade(view, rng, _degrade_roll(rng, hard_frac, 0.45))
                xs.append(_img_to_u8(view, tile_size))
                ys.append(cid)
            if n_files % 500 == 0:
                log("    hcap: %d/%d files (%.0fs)" % (n_files,
                                                        sum(len(f) for _, f in dirs.values()),
                                                        time.time() - t0))
    x = torch.stack(xs) if xs else torch.empty(
        (0, 3, tile_size, tile_size), dtype=torch.uint8)
    y = torch.tensor(ys, dtype=torch.long)
    log("    hcap: %d real files -> %d training views in %.0fs"
        % (n_files, len(ys), time.time() - t0))
    return x, y, n_files

def _photo_view(im, tile_size, rng):
    """One random training view of a REAL PHOTO (full frame, subject usually
    centred): a centre-biased cover-crop (zoom 0.5-1.0), optional flip and
    small rotation, then resize to the Brain's tile size. The centre bias is
    the difference from _hcap_view — a photo's subject is rarely in a random
    corner, and cropping it out would teach the wrong thing."""
    w, h = im.size
    side = min(w, h)
    z = rng.uniform(0.50, 1.0)
    cs = max(8, int(side * z))
    # crop window centre jittered around the image centre (±28%)
    cx = w / 2 + rng.uniform(-0.28, 0.28) * (w - cs) * 0.5
    cy = h / 2 + rng.uniform(-0.28, 0.28) * (h - cs) * 0.5
    x0 = int(min(max(0, cx - cs / 2), max(0, w - cs)))
    y0 = int(min(max(0, cy - cs / 2), max(0, h - cs)))
    im = im.crop((x0, y0, x0 + cs, y0 + cs))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    ang = rng.uniform(-8, 8)
    if abs(ang) > 0.5:
        im = im.rotate(ang, resample=Image.BILINEAR, expand=True)
    return im.resize((tile_size, tile_size), Image.LANCZOS)


def load_photo_tiles(photos_dir, tile_size=DEFAULT_TILE_SIZE, views=16,
                     hard_frac=0.5, seed=21, verbose=True):
    """Ingest a REAL-PHOTO corpus (fetch_photos.py output): one folder per
    class name, 640 px JPEGs from Wikimedia Commons.

    Each photo yields `views` random centre-cropped views with a
    clean/soft/hard degradation roll — the exact same treatment the real
    hCaptcha tiles get, so the tile head learns photo texture (grain,
    lighting, backgrounds) instead of only drawings. Returns
    (x: (N,3,S,S) uint8, y: (N,) long class ids, n_files).
    """
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    if not photos_dir or not os.path.isdir(photos_dir):
        log("    photos: %r is not a directory - skipping" % photos_dir)
        return torch.empty((0, 3, tile_size, tile_size), dtype=torch.uint8), \
            torch.empty((0,), dtype=torch.long), 0
    dirs = {}
    for sub in sorted(os.listdir(photos_dir)):
        p = os.path.join(photos_dir, sub)
        if not os.path.isdir(p):
            continue
        folder = sub.lower().replace("-", "_")
        cls = HCAP_FOLDER_ALIAS.get(folder) or hct.canonical(folder)
        if cls not in CID:
            continue
        files = sorted(f for f in os.listdir(p)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if files:
            dirs[sub] = (cls, files)
    if not dirs:
        log("    photos: no usable class folders in %s" % photos_dir)
        return torch.empty((0, 3, tile_size, tile_size), dtype=torch.uint8), \
            torch.empty((0,), dtype=torch.long), 0
    n_classes = len(set(c for c, _ in dirs.values()))
    n_files = sum(len(f) for _, f in dirs.values())
    log("    photos: %d classes, %d real photos (%d views each)" %
        (n_classes, n_files, views))
    xs, ys = [], []
    t0 = time.time()
    done = 0
    for sub, (cls, files) in dirs.items():
        cid = CID[cls]
        for fi, fn in enumerate(files):
            done += 1
            try:
                im = Image.open(os.path.join(photos_dir, sub, fn)) \
                         .convert("RGB")
            except Exception:
                continue
            for v in range(views):
                rng = random.Random("photo|%d|%s|%d|%d" % (seed, sub, fi, v))
                view = _photo_view(im, tile_size, rng)
                view = _degrade(view, rng, _degrade_roll(rng, hard_frac, 0.45))
                xs.append(_img_to_u8(view, tile_size))
                ys.append(cid)
            if done % 200 == 0:
                log("    photos: %d/%d files (%.0fs)" % (done, n_files,
                                                         time.time() - t0))
    x = torch.stack(xs) if xs else torch.empty(
        (0, 3, tile_size, tile_size), dtype=torch.uint8)
    y = torch.tensor(ys, dtype=torch.long)
    log("    photos: %d real files -> %d training views in %.0fs"
        % (done, len(ys), time.time() - t0))
    return x, y, done


def make_bbox_round(rng: random.Random, size: int = DEFAULT_SCENE_SIZE):
    """A single-object 'draw a box around the X' round (no manifest equivalent
    in make_challenges). Renders one object on a scene at a random place/scale
    and returns the tight bbox (cx, cy, w, h), all normalised 0..1."""
    from make_challenges import _scene_bg, POINT_CLASSES
    name = rng.choice(POINT_CLASSES)
    img = _scene_bg(size, rng)
    s = int(size * rng.uniform(0.26, 0.46))
    tile = md.render(name, s, rng)
    cx = rng.uniform(0.30, 0.70)
    cy = rng.uniform(0.30, 0.70)
    x = int(cx * size - s / 2)
    y = int(cy * size - s / 2)
    img.paste(tile, (x, y))
    meta = {"type": "bbox", "target": name, "cx": round(cx, 4),
            "cy": round(cy, 4), "w": round(s / size, 4), "h": round(s / size, 4)}
    return img, meta


def make_pipe_round(rng: random.Random, size: int = DEFAULT_SCENE_SIZE):
    """"Drag the pipe to where it fits" — a pipe run (horizontal or vertical)
    with a missing section (the slot), plus a loose pipe segment carrying a
    Move badge. Same piece->slot from/to supervision as the drag rounds, on
    plumbing imagery, so the drag head learns pipes explicitly."""
    from make_challenges import _scene_bg
    from PIL import ImageDraw, ImageFont
    S = size
    img = _scene_bg(S, rng)
    d = ImageDraw.Draw(img)
    metals = [(146, 150, 156), (170, 174, 180), (122, 126, 134),
              (158, 150, 138), (132, 140, 148)]
    col = metals[rng.randrange(len(metals))]
    dark = tuple(int(v * 0.62) for v in col)
    light = tuple(min(255, int(v * 1.25)) for v in col)

    pd = S * rng.uniform(0.09, 0.13)        # pipe diameter
    seg = S * rng.uniform(0.17, 0.25)       # gap length == piece length
    horiz = rng.random() < 0.5

    def draw_pipe(x0, y0, x1, y1):
        d.rectangle([x0, y0, x1, y1], fill=col, outline=dark, width=2)
        if y1 - y0 > 0:                       # cylinder highlight
            hy = y0 + (y1 - y0) * 0.28
            d.line([x0 + 2, hy, x1 - 2, hy], fill=light, width=2)

    def flange(x, y):
        fw = pd * 0.28
        d.rectangle([x - fw / 2, y - pd * 0.12, x + fw / 2, y + pd * 1.12],
                    fill=dark, outline=tuple(int(v * 0.8) for v in dark),
                    width=1)

    if horiz:
        ry = S * rng.uniform(0.30, 0.68)     # run centre y
        gx = S * rng.uniform(0.36, 0.62)     # gap centre x
        draw_pipe(S * 0.06, ry - pd / 2, gx - seg / 2, ry + pd / 2)
        draw_pipe(gx + seg / 2, ry - pd / 2, S * 0.94, ry + pd / 2)
        flange(gx - seg / 2, ry - pd / 2)
        flange(gx + seg / 2, ry - pd / 2)
        tx, ty = gx / S, ry / S
        px = S * rng.uniform(0.18, 0.80)
        py = S * rng.uniform(0.08, 0.22) if ry > S * 0.45 else S * rng.uniform(0.78, 0.92)
        draw_pipe(px - seg / 2, py - pd / 2, px + seg / 2, py + pd / 2)
        ptop = py - pd / 2
    else:
        rx = S * rng.uniform(0.30, 0.68)
        gy = S * rng.uniform(0.36, 0.62)
        draw_pipe(rx - pd / 2, S * 0.06, rx + pd / 2, gy - seg / 2)
        draw_pipe(rx - pd / 2, gy + seg / 2, rx + pd / 2, S * 0.94)
        flange(rx - pd / 2, gy - seg / 2)
        flange(rx + pd / 2, gy + seg / 2)
        tx, ty = rx / S, gy / S
        py = S * rng.uniform(0.18, 0.80)
        px = S * rng.uniform(0.08, 0.22) if rx > S * 0.45 else S * rng.uniform(0.78, 0.92)
        draw_pipe(px - seg / 2, py - pd / 2, px + seg / 2, py + pd / 2)
        ptop = py - pd / 2

    # "Move" badge above the loose piece
    try:
        font = ImageFont.load_default(size=max(9, S // 9))
    except TypeError:
        font = ImageFont.load_default()
    tb = d.textbbox((0, 0), "Move", font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    bx = int(px - tw / 2)
    by = max(2, int(ptop - th - 8))
    d.rounded_rectangle([bx - 5, by - 3, bx + tw + 5, by + th + 4],
                        radius=4, fill=(250, 250, 252),
                        outline=(60, 60, 66))
    d.text((bx, by), "Move", font=font, fill=(40, 40, 46))

    prompt = rng.choice([
        "Drag the pipe to where it fits",
        "Drag the pipe segment to the place where it fits",
        "Move the pipe to where it fits best",
    ])
    meta = {"type": "pipe", "prompt": prompt,
            "fx": round(px / S, 4), "fy": round(py / S, 4),
            "tx": round(tx, 4), "ty": round(ty, 4)}
    return img, meta


def make_tower_round(rng: random.Random, size: int = DEFAULT_SCENE_SIZE):
    """Wooden-block tower round ('Move the correct missing block segment onto
    the incomplete tower'): three wood stacks, one clearly the shortest (the
    drop target), plus a loose 1-2 block piece on the right with a Move badge.
    Label = normalised (fx, fy) piece centre and (tx, ty) where the piece sits
    on top of the short stack - the from/to the drag head predicts."""
    from make_challenges import _scene_bg
    from PIL import ImageDraw, ImageFont
    S = size
    img = _scene_bg(S, rng)
    d = ImageDraw.Draw(img)
    woods = [(150, 106, 60), (168, 124, 72), (134, 94, 52),
             (180, 136, 86), (144, 100, 58)]

    bw = S * rng.uniform(0.11, 0.15)         # block width
    bh = S * rng.uniform(0.075, 0.105)       # block height
    bottom = S * rng.uniform(0.78, 0.88)
    short_h = rng.randint(1, 3)
    short_i = rng.randrange(3)
    heights = [rng.randint(short_h + 1, short_h + 3) for _ in range(3)]
    heights[short_i] = short_h
    xs = [S * rng.uniform(0.08, 0.14), S * rng.uniform(0.28, 0.34),
          S * rng.uniform(0.48, 0.54)]
    tops = []
    for x, h in zip(xs, heights):
        col = woods[rng.randrange(len(woods))]
        for k in range(h):
            y0 = bottom - (k + 1) * bh
            y1 = bottom - k * bh
            c = tuple(min(255, v + rng.randint(-10, 10)) for v in col)
            oc = tuple(int(v * 0.62) for v in c)
            d.rectangle([x, y0, x + bw, y1], fill=c, outline=oc, width=2)
            d.line([x + 2, y0 + bh / 2, x + bw - 2, y0 + bh / 2],
                   fill=tuple(int(v * 0.78) for v in c), width=1)
        tops.append(bottom - h * bh)

    ph = rng.choice([1, 1, 2])               # piece height in blocks
    px = S * rng.uniform(0.76, 0.86)
    ptop = S * rng.uniform(0.28, 0.55)
    pcol = woods[rng.randrange(len(woods))]
    for k in range(ph):
        y0 = ptop + k * bh
        c = tuple(min(255, v + rng.randint(-10, 10)) for v in pcol)
        d.rectangle([px, y0, px + bw, y0 + bh], fill=c,
                    outline=tuple(int(v * 0.62) for v in c), width=2)
        d.line([px + 2, y0 + bh / 2, px + bw - 2, y0 + bh / 2],
               fill=tuple(int(v * 0.78) for v in c), width=1)
    try:
        font = ImageFont.load_default(size=max(9, S // 9))
    except TypeError:
        font = ImageFont.load_default()
    tb = d.textbbox((0, 0), "Move", font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    bx = int(px + bw / 2 - tw / 2)
    by = max(2, int(ptop - th - 8))
    d.rounded_rectangle([bx - 5, by - 3, bx + tw + 5, by + th + 4],
                        radius=4, fill=(250, 250, 252),
                        outline=(60, 60, 66))
    d.text((bx, by), "Move", font=font, fill=(40, 40, 46))

    fx, fy = (px + bw / 2) / S, (ptop + ph * bh / 2) / S
    tx, ty = (xs[short_i] + bw / 2) / S, (tops[short_i] - ph * bh / 2) / S
    prompt = rng.choice([
        "Move the correct missing block segment onto the incomplete tower",
        "Move the missing block segment onto the incomplete tower",
        "Put the missing block segment onto the incomplete tower",
    ])
    meta = {"type": "tower", "prompt": prompt,
            "fx": round(fx, 4), "fy": round(fy, 4),
            "tx": round(tx, 4), "ty": round(ty, 4)}
    return img, meta


SHAPE_KINDS = ["star", "hexagon", "diamond", "cross", "arrow", "semicircle"]


def _shape_points(kind, w, h):
    """Polygon points for a drag-piece shape inside a w x h box."""
    cx, cy = w / 2, h / 2
    if kind == "star":
        R, r = w * 0.48, w * 0.20
        return [(cx + (R if i % 2 == 0 else r) * math.cos(-math.pi / 2 + i * math.pi / 5),
                 cy + (R if i % 2 == 0 else r) * math.sin(-math.pi / 2 + i * math.pi / 5))
                for i in range(10)]
    if kind == "hexagon":
        R = w * 0.46
        return [(cx + R * math.cos(-math.pi / 2 + i * math.pi / 3),
                 cy + R * math.sin(-math.pi / 2 + i * math.pi / 3))
                for i in range(6)]
    if kind == "diamond":
        return [(cx, cy - h * 0.48), (cx + w * 0.48, cy),
                (cx, cy + h * 0.48), (cx - w * 0.48, cy)]
    if kind == "cross":
        a, b = w * 0.30, w * 0.48
        return [(cx - a, cy - b), (cx + a, cy - b), (cx + a, cy - a),
                (cx + b, cy - a), (cx + b, cy + a), (cx + a, cy + a),
                (cx + a, cy + b), (cx - a, cy + b), (cx - a, cy + a),
                (cx - b, cy + a), (cx - b, cy - a), (cx - a, cy - a)]
    if kind == "arrow":
        return [(cx - w * 0.48, cy - h * 0.14), (cx, cy - h * 0.14),
                (cx, cy - h * 0.48), (cx + w * 0.48, cy),
                (cx, cy + h * 0.48), (cx - w * 0.48, cy + h * 0.14)]
    # semicircle: top-half arc + flat base
    pts = [(cx + w * 0.46 * math.cos(math.pi + i * math.pi / 12),
            cy + h * 0.46 * math.sin(math.pi + i * math.pi / 12))
           for i in range(13)]
    pts += [(cx + w * 0.46, cy + h * 0.34), (cx - w * 0.46, cy + h * 0.34)]
    return pts


def make_shape_round(rng: random.Random, size: int = DEFAULT_SCENE_SIZE):
    """Extended-shape drag round: a punched slot of a less common shape (star,
    hexagon, diamond, cross, arrow, semicircle) plus a matching loose piece
    with a Move badge. Same piece->slot from/to supervision as the classic
    drag rounds - this widens the drag head's shape vocabulary beyond
    circle/square/triangle/puzzle."""
    from make_challenges import _scene_bg
    from PIL import ImageDraw, ImageFont
    S = size
    img = _scene_bg(S, rng)
    d = ImageDraw.Draw(img)
    kind = rng.choice(SHAPE_KINDS)
    pw = S * rng.uniform(0.18, 0.28)
    ph = pw * rng.uniform(0.9, 1.2)
    tx, ty = rng.uniform(0.38, 0.72), rng.uniform(0.28, 0.68)
    while True:
        fx, fy = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.75)
        if math.hypot(fx - tx, fy - ty) >= 0.34:
            break
    # punched slot: dark shape with a light outline ring
    base = _shape_points(kind, pw, ph)
    slot = [(x + tx * S - pw / 2, y + ty * S - ph / 2) for x, y in base]
    d.polygon(slot, fill=(28, 28, 34), outline=(250, 250, 252))
    # loose piece: vivid fill, same silhouette
    col = tuple(rng.randint(60, 235) for _ in range(3))
    piece = [(x + fx * S - pw / 2, y + fy * S - ph / 2) for x, y in base]
    d.polygon(piece, fill=col, outline=tuple(int(v * 0.6) for v in col))
    # "Move" badge above the piece
    try:
        font = ImageFont.load_default(size=max(9, S // 9))
    except TypeError:
        font = ImageFont.load_default()
    tb = d.textbbox((0, 0), "Move", font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    bx = int(fx * S - tw / 2)
    by = max(2, int(fy * S - ph / 2 - th - 8))
    d.rounded_rectangle([bx - 5, by - 3, bx + tw + 5, by + th + 4],
                        radius=4, fill=(250, 250, 252),
                        outline=(60, 60, 66))
    d.text((bx, by), "Move", font=font, fill=(40, 40, 46))
    prompt = rng.choice([
        "Drag the element to the place where it fits best",
        "Move the shape to the matching hole",
        "Drag the piece to where it fits",
    ])
    meta = {"type": "shape", "prompt": prompt, "shape": kind,
            "fx": round(fx, 4), "fy": round(fy, 4),
            "tx": round(tx, 4), "ty": round(ty, 4)}
    return img, meta


TEXT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TEXT_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # no I O 0 1 (ambiguous)


def make_text_round(rng: random.Random, size: int = DEFAULT_SCENE_SIZE):
    """text_entry round ("Type the text you see"): a 5-char code on a light
    noisy background. Label = the code, one class per character position."""
    from PIL import ImageDraw, ImageFont, ImageFilter
    S = size
    img = Image.new("RGB", (S, S), (rng.randint(232, 246),) * 3)
    d = ImageDraw.Draw(img)
    # speckle noise
    for _ in range(rng.randint(60, 140)):
        x, y = rng.randrange(S), rng.randrange(S)
        v = rng.randint(190, 225)
        d.point((x, y), fill=(v, v, v))
    code = "".join(rng.choice(TEXT_CHARS) for _ in range(5))
    try:
        font = ImageFont.load_default(size=int(S * 0.40))
    except TypeError:
        font = ImageFont.load_default()
    x = S * 0.10
    for ch in code:
        d.text((x, S * 0.26 + rng.uniform(-0.04, 0.04) * S), ch,
               font=font, fill=(28, 30, 40))
        x += S * 0.16
    # a couple of distractor lines
    for _ in range(2):
        x0, y0 = rng.randrange(S), rng.randrange(S)
        x1, y1 = rng.randrange(S), rng.randrange(S)
        d.line([x0, y0, x1, y1],
               fill=(rng.randint(150, 190),) * 3, width=1)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    meta = {"type": "text", "text": code,
            "prompt": rng.choice(["Type the text you see",
                                  "Enter the code below",
                                  "Type the characters you see"])}
    return img, meta


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt bank: the Brain's "knows every possible thing" language corpus
# ═══════════════════════════════════════════════════════════════════════════

_BANK_BIN = [
    "Please click each image containing {n}",
    "Click each image containing {n}",
    "Select all images containing {n}",
    "Select all the images with {n}",
    "Click on all the pictures of {n}",
    "Tap every tile that shows {n}",
    "Pick every image that contains {n}",
    "Choose all images showing {n}",
    "Mark all tiles with {n}",
    "Click the tiles that contain {n}",
    "Which images contain {n}?",
    "Click every picture with {n} in it",
]
_BANK_CNT = [
    "How many {ns} are in this image?",
    "How many {n} are in this image?",
    "Count the {ns} in the image",
    "What number of {ns} do you see?",
    "How many {n} can you find?",
    "Count every {n} you see",
]
_BANK_PT = [
    "Please click on the {n}",
    "Click the {n}",
    "Please click on the {n} in the image",
    "Tap the {n}",
    "Click on {n}",
    "Find the {n} and click it",
    "Click on the {n} in this picture",
    "Where is the {n}? Click it",
]
_BANK_BB = [
    "Draw a box around the {n}",
    "Please draw a box around the {n}",
    "Draw a rectangle around the {n}",
    "Box the {n}",
    "Draw a box around {n}",
    "Outline the {n} with a box",
]
_BANK_REL = {
    "SIZE": ["Click the largest {n}", "Click the smallest {n}",
             "Which {ns} is the biggest?", "Which {ns} is the smallest?",
             "Tap the largest {n}", "Select the smallest {n}",
             "Find the biggest {n}", "Find the tiniest {n}"],
    "JUMP": ["Click the animal who jumps the highest",
             "Click the animal who jumps the lowest",
             "Which animal jumps the highest?",
             "Which animal jumps the lowest?",
             "Click the {ns} that jumps the highest",
             "Click the {ns} that jumps the lowest"],
    "SPEED": ["Click the fastest {n}", "Click the slowest {n}",
              "Which {ns} is the fastest?", "Which {ns} is the slowest?",
              "Tap the quickest {n}", "Tap the slowest {n}"],
    "TEMP": ["Click the coldest place", "Click the hottest place",
             "Where is it coldest?", "Where is it hottest?",
             "Click the animal in the coldest place",
             "Click the animal in the hottest place",
             "Which {ns} lives in the coldest place?",
             "Which {ns} lives in the hottest place?"],
}
_BANK_AFF = ["Pick all things you can work on with the {t}",
             "Click the surfaces the {t} works on",
             "Select everything the {t} can work on",
             "Which items can the {t} be used on?"]
_BANK_MAT = ["Select items that are primarily metal",
             "Select all the metal objects",
             "Click things made of metal",
             "Select items that are made of wood",
             "Select all the wooden things",
             "Click the wood items",
             "Select items that have fur",
             "Pick the furry animals",
             "Click the things with fur",
             "Select the plants",
             "Click every plant you see",
             "Pick all the green plants"]
_BANK_SET = ["Find places safe for setting down the item in the reference",
             "Click the surfaces you can set the item down on",
             "Pick safe places to put the item shown",
             "Which tiles can the item be placed on?"]
_BANK_DRAG = ["Drag the element to the place where it fits best",
              "Drag the piece to where it fits",
              "Move the shape into its matching hole",
              "Drag the shape to the empty space",
              "Complete the puzzle",
              "Find the missing piece and move it",
              "Drag the missing piece into place",
              "Move the piece to where it belongs",
              "Drag the pipe to where it fits",
              "Drag the pipe segment to the place where it fits",
              "Move the pipe to where it fits best"]
_BANK_PATTERN = ["Put one of the animals into the empty spot to complete the pattern",
                 "Complete the pattern by dragging the right tile",
                 "Which tile completes the pattern?",
                 "Put one of the items into the empty spot",
                 "Select the tile that finishes the pattern"]
_BANK_TOWER = ["Move the correct missing block segment onto the incomplete tower",
               "Move the missing block segment onto the incomplete tower",
               "Put the missing block segment on the tower",
               "Stack the block onto the shortest tower"]
_BANK_CHOICE = ["Select the most accurate description",
                "Which of these is correct?",
                "Choose the right answer",
                "Select the correct statement",
                "Pick the best description of the image"]
_BANK_TEXT = ["Type the text you see",
              "Enter the code below",
              "Type the characters you see",
              "Type the letters you see in the image",
              "Enter the code you see"]


def _plural(name):
    if name.endswith("s"):
        return name
    if name.endswith("y") and name[-2:] not in ("ay", "ey", "oy", "uy"):
        return name[:-1] + "ies"
    if name.endswith(("x", "ch", "sh", "ss")):
        return name + "es"
    return name + "s"


def build_router_bank(verbose=True):
    """The prompt->family TRAINING bank — the Brain's language corpus.

    Every class (60) x every synonym from hcaptcha_types.SYNONYMS x article/
    plural surface forms x the real hCaptcha wording templates for binary /
    count / point / bbox, plus the superlative tables (SIZE/JUMP/SPEED/TEMP),
    tool affordances, material & set-down grids, and the drag/pattern/tower/
    choice/text wordings. ~30k pairs: the prompt encoder is trained on all
    of it, so the learned router reads essentially every noun and phrasing
    hCaptcha serves. Returns a deduped list of (prompt, family) tuples.
    """
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    pairs = []
    seen = set()

    def add(p, fam):
        p = " ".join(p.split()).strip()
        if p and (p, fam) not in seen:
            seen.add((p, fam))
            pairs.append((p, fam))

    # nouns per class: canonical + every synonym that canonicalises to it
    by_class = {c: set() for c in CLASSES}
    for k, v in hct.SYNONYMS.items():
        if v in by_class:
            by_class[v].add(k.replace("_", " "))
    for c in CLASSES:
        by_class[c].add(c.replace("_", " "))

    def forms(name):
        base = name
        out = [base, _plural(base)]
        art = "an " if base[0] in "aeiou" else "a "
        out += [art + base, "the " + base, "each " + base,
                "one " + base, "any " + _plural(base)]
        return out

    for c in CLASSES:
        for noun in sorted(by_class[c]):
            n = noun.replace("_", " ")
            ns = _plural(n)
            for f in forms(n):
                for t in _BANK_BIN:
                    add(t.format(n=f), BINARY)
            for t in _BANK_CNT:
                add(t.format(n=n, ns=ns), COUNT)
            for t in _BANK_PT:
                add(t.format(n=n, ns=ns), AREA_POINT)
            for t in _BANK_BB:
                add(t.format(n=n), AREA_BBOX)
    # superlative / relational wordings
    for table_name, table in (("SIZE", hct.SIZE_RANK),
                              ("JUMP", hct.JUMP_RANK),
                              ("SPEED", hct.SPEED_RANK),
                              ("TEMP", hct.TEMP_RANK)):
        for c, _v in sorted(table.items()):
            if c not in CID:
                continue
            n = c.replace("_", " ")
            ns = _plural(n)
            if table_name == "TEMP":
                for t in _BANK_REL["TEMP"]:
                    add(t.format(n=n, ns=ns), AREA_POINT)
                continue
            for t in _BANK_REL[table_name]:
                if table_name in ("JUMP", "TEMP") and n not in (
                        c.replace("_", " ") for c in hct.ANIMALS):
                    # "the animal who jumps..." is animal-specific wording
                    if "animal" in t and c not in hct.ANIMALS:
                        continue
                add(t.format(n=n, ns=ns), AREA_POINT)
    # affordance grids (reference-image binary family)
    for t_name, _surfaces in sorted(hct.TOOL_AFFORDANCE.items()):
        t = t_name.replace("_", " ")
        for tmpl in _BANK_AFF:
            add(tmpl.format(t=t), BINARY)
    for p in _BANK_MAT:
        add(p, BINARY)
    for p in _BANK_SET:
        add(p, BINARY)
    for p in _BANK_DRAG:
        add(p, DRAG_DROP)
    for p in _BANK_PATTERN:
        add(p, PATTERN)
    for p in _BANK_TOWER:
        add(p, TOWER)
    for p in _BANK_CHOICE:
        add(p, MULTIPLE_CHOICE)
    for p in _BANK_TEXT:
        add(p, TEXT_ENTRY)
    log("  router bank: %d (prompt, family) pairs" % len(pairs))
    return pairs

def _corpus_cache_paths(**params):
    """Deterministic cache file paths for a build_brain_corpus config.

    The rendered tile/scene arrays are the slow part (~50 min for the
    310k-tile giga corpus); caching them on disk means a Kaggle session
    restart or a hyperparameter re-run never pays that cost twice.
    """
    src = json.dumps({k: (str(v) if v is not None else None)
                      for k, v in sorted(params.items())}, sort_keys=True)
    key = hashlib.sha1(src.encode()).hexdigest()[:16]
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "corpus_cache")
    return (os.path.join(d, "corpus_%s.npz" % key),
            os.path.join(d, "corpus_%s.meta.pkl" % key), d)


def _corpus_cache_save(corpus, npz, pkl, log):
    # pattern/grid PILs are kept at their ORIGINAL size: hard-degradation
    # randomly downscales them (bad-capture simulation), so the stack must
    # store each image as its own npz entry.
    a = dict(
        tile_x=np.ascontiguousarray(corpus["tile_x"].numpy()),
        tile_y=np.ascontiguousarray(corpus["tile_y"].numpy()),
        point_x=np.ascontiguousarray(corpus["point_x"].numpy()),
        count_x=np.ascontiguousarray(corpus["count_x"].numpy()),
        drag_x=np.ascontiguousarray(corpus["drag_x"].numpy()),
        bbox_x=np.ascontiguousarray(corpus["bbox_x"].numpy()),
        text_x=np.ascontiguousarray(corpus["text_x"].numpy()),
    )
    for prefix, imgs in (("pat", corpus["pat_imgs"]),
                         ("grid", corpus["grid_imgs"])):
        for i, im in enumerate(imgs):
            a["%s_%06d" % (prefix, i)] = np.asarray(im.convert("RGB"))
    np.savez(npz, **a)
    meta = {k: corpus[k] for k in
            ("point_m", "count_m", "drag_m", "bbox_m", "text_m", "pat_m",
             "grid_m", "router", "tile_size", "scene_size", "hcap_n",
             "photo_n")}
    meta["pat_n"] = len(corpus["pat_imgs"])
    meta["grid_n"] = len(corpus["grid_imgs"])
    with open(pkl, "wb") as f:
        pickle.dump(meta, f)
    log("  corpus cache: saved %.1f GB -> %s" %
        (os.path.getsize(npz) / 1e9, npz))


def _corpus_cache_load(npz, pkl, log):
    with open(pkl, "rb") as f:
        meta = pickle.load(f)
    a = np.load(npz)

    def T(k):
        return torch.from_numpy(np.ascontiguousarray(a[k]))

    def imgs(prefix, n):
        return [Image.fromarray(a["%s_%06d" % (prefix, i)])
                for i in range(n)]

    return {
        "tile_x": T("tile_x"), "tile_y": T("tile_y"),
        "point_x": T("point_x"), "point_m": meta["point_m"],
        "count_x": T("count_x"), "count_m": meta["count_m"],
        "drag_x": T("drag_x"), "drag_m": meta["drag_m"],
        "bbox_x": T("bbox_x"), "bbox_m": meta["bbox_m"],
        "text_x": T("text_x"), "text_m": meta["text_m"],
        "pat_imgs": imgs("pat", meta["pat_n"]), "pat_m": meta["pat_m"],
        "grid_imgs": imgs("grid", meta["grid_n"]), "grid_m": meta["grid_m"],
        "router": meta["router"],
        "tile_size": meta["tile_size"], "scene_size": meta["scene_size"],
        "hcap_n": meta["hcap_n"], "photo_n": meta.get("photo_n", 0),
    }


def build_brain_corpus(per_class=1200, n_point=14000, n_count=8000,
                       n_drag=9000, n_grid=6000, n_pattern=6000, n_bbox=7000,
                       n_pipe=3000, n_tower=3000, n_shape=3000,
                       n_odrag=4000, n_text=4000,
                       degrade_frac=0.40, hard_frac=0.30, clutter_frac=0.5,
                       router_bank=True, hcap_dir=None, hcap_views=16,
                       photos_dir=None, photo_views=16, real_only=False,
                       tile_size=DEFAULT_TILE_SIZE,
                       scene_size=DEFAULT_SCENE_SIZE,
                       seed=7, verbose=True):
    """Build the full multi-task corpus in memory.

    Returns a dict of per-family tensors + metadata, ready for the joint
    training loop. Everything is generated by the repo's deterministic
    generators, so two runs with the same seed are identical and a Kaggle
    notebook needs no pre-built data.

    v2 additions:
      - every round rolls a clean/soft/hard degradation (`hard_frac` hard,
        `degrade_frac` soft, the rest clean) instead of one soft tier;
      - point/count rounds get extra distractor objects pasted in with
        probability `clutter_frac` (labels stay valid — see _add_clutter);
      - `hcap_dir`: a real hCaptcha challenge-image dataset (the "hcap"
        GitHub datasets) is ingested as extra tile training views;
      - `router_bank=True` builds the ~30k-pair prompt->family bank instead
        of the small v1 pair list.
    """
    from make_challenges import (make_point_round, make_count_round,
                                 make_drag_round, make_pattern_round,
                                 make_grid_round, make_object_drag_round)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    t0 = time.time()

    # ── disk cache: skip the ~50 min of re-rendering when identical ──────
    ck_npz, ck_pkl, ck_dir = _corpus_cache_paths(
        per_class=per_class, n_point=n_point, n_count=n_count,
        n_drag=n_drag, n_grid=n_grid, n_pattern=n_pattern, n_bbox=n_bbox,
        n_pipe=n_pipe, n_tower=n_tower, n_shape=n_shape, n_odrag=n_odrag,
        n_text=n_text,
        hard_frac=hard_frac, degrade_frac=degrade_frac,
        clutter_frac=clutter_frac, router_bank=router_bank,
        hcap_dir=hcap_dir, hcap_views=hcap_views,
        photos_dir=photos_dir, photo_views=photo_views,
        real_only=real_only,
        tile_size=tile_size, scene_size=scene_size, seed=seed,
        n_classes=N_CLASSES)
    if os.path.exists(ck_npz) and os.path.exists(ck_pkl):
        log("  corpus cache HIT - loading rendered corpus from disk...")
        try:
            corpus = _corpus_cache_load(ck_npz, ck_pkl, log)
        except Exception as e:
            log("  corpus cache: read failed (%s) - rebuilding fresh" % e)
            for p in (ck_npz, ck_pkl):
                try:
                    os.remove(p)
                except OSError:
                    pass
        else:
            log("  corpus loaded in %.0fs" % (time.time() - t0))
            return corpus
    os.makedirs(ck_dir, exist_ok=True)

    # ── tile classification: painted tiles + REAL hCaptcha tiles ──────────
    # REAL-ONLY mode: any class that has real photos (hcap or photos_dir)
    # is trained EXCLUSIVELY on those real images - zero renders for it.
    # Only classes with no real photo anywhere fall back to renders (and
    # are listed, so the gap is visible, not hidden).
    real_counts = {}
    if real_only:
        for d in (hcap_dir, photos_dir):
            if not d or not os.path.isdir(d):
                continue
            for sub in sorted(os.listdir(d)):
                p = os.path.join(d, sub)
                if not os.path.isdir(p):
                    continue
                folder = sub.lower().replace("-", "_")
                cls = HCAP_FOLDER_ALIAS.get(folder) or hct.canonical(folder)
                if cls in CID:
                    n = len([f for f in os.listdir(p)
                             if f.lower().endswith((".jpg", ".jpeg",
                                                   ".png"))])
                    if n:
                        real_counts[CID[cls]] = \
                            real_counts.get(CID[cls], 0) + n
        n_render_total = sum(0 if real_counts.get(cid, 0) > 0 else per_class
                             for cid in range(N_CLASSES))
        log("  tiles: REAL-ONLY mode - %d/%d classes have real photos and "
            "train render-free; %d class(es) fall back to renders: %s" % (
                len(real_counts), N_CLASSES, N_CLASSES - len(real_counts),
                ", ".join(CLASSES[c] for c in range(N_CLASSES)
                          if real_counts.get(c, 0) == 0)[:600]))
    else:
        n_render_total = per_class * N_CLASSES
    log("  tiles: rendering %d painted images (%d/class where needed)..."
        % (n_render_total, per_class))
    tx = torch.empty((n_render_total, 3, tile_size, tile_size),
                     dtype=torch.uint8)
    ty = torch.empty(n_render_total, dtype=torch.long)
    i = 0
    for cid, name in enumerate(CLASSES):
        n = 0 if (real_only and real_counts.get(cid, 0) > 0) else per_class
        for k in range(n):
            rng = random.Random("tile|%s|%d|%d" % (name, seed, k))
            im = md.render(name, tile_size, rng)
            im = _degrade(im, rng, _degrade_roll(rng, hard_frac, degrade_frac))
            tx[i] = _img_to_u8(im, tile_size)
            ty[i] = cid
            i += 1
        if n and ((cid + 1) % 5 == 0 or cid == N_CLASSES - 1):
            log("    tiles: %d/%d classes done (%d images)" % (
                cid + 1, N_CLASSES, i))

    hcap_n = 0
    if hcap_dir:
        hx, hy, hcap_n = load_hcap_tiles(
            hcap_dir, tile_size=tile_size, views=hcap_views,
            hard_frac=hard_frac, seed=seed + 1, verbose=verbose)
        if len(hx):
            tx = torch.cat([tx, hx])
            ty = torch.cat([ty, hy])
            log("    tiles: + %d real hcap views (total %d)" % (
                len(hx), len(tx)))

    photo_n = 0
    if photos_dir:
        px, py, photo_n = load_photo_tiles(
            photos_dir, tile_size=tile_size, views=photo_views,
            hard_frac=hard_frac, seed=seed + 2, verbose=verbose)
        if len(px):
            tx = torch.cat([tx, px])
            ty = torch.cat([ty, py])
            log("    tiles: + %d real-photo views (total %d)" % (
                len(px), len(tx)))

    # ── grid rounds: feed their 9 tiles through the SAME tile head ─────────
    grid_tiles, grid_labels = [], []
    grid_imgs, grid_m = [], []
    keep_grid_imgs = n_grid <= 2000     # big corpora drop the grid PILs (RAM)
    log("  grids: generating %d (9 tiles each)..." % n_grid)
    for k in range(n_grid):
        rng = random.Random("grid|%d|%d" % (seed, k))
        img, meta = make_grid_round(rng, scene_size)
        mode = _degrade_roll(rng, hard_frac, degrade_frac)
        if mode != "clean":
            img = _degrade(img, rng, mode)
        names = meta["tiles"]
        boxes = meta["tile_boxes"]
        W, H = img.size
        for name, box in zip(names, boxes):
            x0, y0, s = box[0], box[1], box[2]
            crop = img.crop((x0, y0, x0 + s, y0 + s))
            grid_tiles.append(_img_to_u8(crop, tile_size))
            grid_labels.append(CID[name])
        if keep_grid_imgs:
            m = {kk: vv for kk, vv in meta.items() if kk != "reference_image"}
            grid_imgs.append(img)
            grid_m.append(m)
        if n_grid >= 1000 and (k + 1) % 500 == 0:
            log("    grids: %d/%d" % (k + 1, n_grid))
    log("    grids: done (%d)" % n_grid)
    if grid_tiles:
        gx = torch.stack(grid_tiles)
        gy = torch.tensor(grid_labels, dtype=torch.long)
        tx = torch.cat([tx, gx])
        ty = torch.cat([ty, gy])

    # ── point + count rounds (heatmap head) ────────────────────────────────
    def _load_scenes(fn, n, kind, clutter=False):
        log("  %s: generating %d..." % (kind, n))
        xs = torch.empty((n, 3, scene_size, scene_size), dtype=torch.uint8)
        metas = []
        for k in range(n):
            rng = random.Random("%s|%d|%d" % (kind, seed, k))
            img, meta = fn(rng, scene_size)
            if clutter and rng.random() < clutter_frac:
                img, meta = _add_clutter(img, meta, rng)
            img = _degrade(img, rng, _degrade_roll(rng, hard_frac, degrade_frac))
            xs[k] = _img_to_u8(img, scene_size)
            metas.append(meta)
            if n >= 4000 and (k + 1) % 2000 == 0:
                log("    %s: %d/%d" % (kind, k + 1, n))
        log("    %s: done (%d)" % (kind, n))
        return xs, metas

    point_x, point_m = _load_scenes(make_point_round, n_point, "point",
                                    clutter=True)
    count_x, count_m = _load_scenes(make_count_round, n_count, "count",
                                    clutter=True)

    # ── drag rounds (drag head) ────────────────────────────────────────────
    drag_x, drag_m = _load_scenes(make_drag_round, n_drag, "drag")

    # ── pipe + tower + extended-shape rounds: same piece->slot from/to
    # supervision, different imagery ("drag the pipe to where it fits" /
    # wood towers / star-hexagon-diamond-cross-arrow pieces). They join the
    # drag training pool so the drag head learns all the looks. ───────────
    for kind, fn, n in (("pipe", make_pipe_round, n_pipe),
                        ("tower", make_tower_round, n_tower),
                        ("shape", make_shape_round, n_shape),
                        ("odrag", make_object_drag_round, n_odrag)):
        if n <= 0:
            continue
        log("  %s: generating %d..." % (kind, n))
        kx = torch.empty((n, 3, scene_size, scene_size), dtype=torch.uint8)
        km = []
        for k in range(n):
            rng = random.Random("%s|%d|%d" % (kind, seed, k))
            img, meta = fn(rng, scene_size)
            img = _degrade(img, rng, _degrade_roll(rng, hard_frac, degrade_frac))
            kx[k] = _img_to_u8(img, scene_size)
            km.append(meta)
            if n >= 2000 and (k + 1) % 1000 == 0:
                log("    %s: %d/%d" % (kind, k + 1, n))
        log("    %s: done (%d)" % (kind, n))
        drag_x = torch.cat([drag_x, kx])
        drag_m = drag_m + km

    # ── text rounds (text head): 5-char codes ─────────────────────────────
    if n_text > 0:
        log("  text: generating %d..." % n_text)
        text_x = torch.empty((n_text, 3, scene_size, scene_size),
                             dtype=torch.uint8)
        text_m = []
        for k in range(n_text):
            rng = random.Random("text|%d|%d" % (seed, k))
            img, meta = make_text_round(rng, scene_size)
            img = _degrade(img, rng, _degrade_roll(rng, hard_frac, degrade_frac))
            text_x[k] = _img_to_u8(img, scene_size)
            text_m.append(meta)
            if n_text >= 2000 and (k + 1) % 1000 == 0:
                log("    text: %d/%d" % (k + 1, n_text))
        log("    text: done (%d)" % n_text)
    else:
        text_x = torch.empty((0, 3, scene_size, scene_size),
                             dtype=torch.uint8)
        text_m = []

    # ── bbox rounds (bbox head) ────────────────────────────────────────────
    log("  bbox: generating %d..." % n_bbox)
    bbox_x = torch.empty((n_bbox, 3, scene_size, scene_size), dtype=torch.uint8)
    bbox_m = []
    for k in range(n_bbox):
        rng = random.Random("bbox|%d|%d" % (seed, k))
        img, meta = make_bbox_round(rng, scene_size)
        img = _degrade(img, rng, _degrade_roll(rng, hard_frac, degrade_frac))
        bbox_x[k] = _img_to_u8(img, scene_size)
        bbox_m.append(meta)
        if n_bbox >= 4000 and (k + 1) % 2000 == 0:
            log("    bbox: %d/%d" % (k + 1, n_bbox))
    log("    bbox: done (%d)" % n_bbox)

    # ── pattern rounds (pattern reasoner): keep the big image + boxes ──────
    log("  pattern: generating %d..." % n_pattern)
    pat_imgs, pat_m = [], []
    for k in range(n_pattern):
        rng = random.Random("pattern|%d|%d" % (seed, k))
        img, meta = make_pattern_round(rng, scene_size)
        img = _degrade(img, rng, _degrade_roll(rng, hard_frac, degrade_frac))
        pat_imgs.append(img)
        pat_m.append(meta)
        if n_pattern >= 400 and (k + 1) % 200 == 0:
            log("    pattern: %d/%d" % (k + 1, n_pattern))
    log("    pattern: done (%d)" % n_pattern)

    # ── router training pairs (prompt -> family) ──────────────────────────
    if router_bank:
        router = build_router_bank(verbose=verbose)
    else:
        router = []
        for m in point_m:
            router.append((m.get("prompt", ""), AREA_POINT))
        for m in count_m:
            router.append((m.get("prompt", ""), COUNT))
        for m in drag_m:
            fam = TOWER if m.get("type") == "tower" else DRAG_DROP
            router.append((m.get("prompt", ""), fam))
        for m in pat_m:
            router.append((m.get("prompt", ""), PATTERN))
        for m in bbox_m:
            router.append((m.get("prompt", ""), AREA_BBOX))
        for m in text_m:
            router.append((m.get("prompt", ""), TEXT_ENTRY))
        log("  router prompt pairs: %d (small v1 set)" % len(router))

    log("  corpus built in %.0fs (%.0f MB uint8 tiles, %d tile images)" % (
        time.time() - t0,
        sum(t.element_size() * t.numel() for t in
            [tx, point_x, count_x, drag_x, bbox_x]) / 1e6,
        len(tx)))
    corpus = {
        "tile_x": tx, "tile_y": ty,
        "point_x": point_x, "point_m": point_m,
        "count_x": count_x, "count_m": count_m,
        "drag_x": drag_x, "drag_m": drag_m,
        "bbox_x": bbox_x, "bbox_m": bbox_m,
        "text_x": text_x, "text_m": text_m,
        "pat_imgs": pat_imgs, "pat_m": pat_m,
        "grid_imgs": grid_imgs, "grid_m": grid_m,
        "router": router,
        "tile_size": tile_size, "scene_size": scene_size,
        "hcap_n": hcap_n, "photo_n": photo_n,
    }
    try:
        _corpus_cache_save(corpus, ck_npz, ck_pkl, log)
    except Exception as e:  # full disk etc. — never kill a run over the cache
        log("  corpus cache: NOT saved (%s)" % e)
    return corpus


# ═══════════════════════════════════════════════════════════════════════════
#  Augmentation: coordinate-mapped geometric aug (ported from train_models)
# ═══════════════════════════════════════════════════════════════════════════

def _prep_geom(batch_u8, targets, rng, flip=True):
    """Apply the SAME affine (rotate/scale/translate/flip) to images AND
    coordinate targets, so one round teaches a continuum of poses. Returns
    (float batch in [-1,1], augmented targets clamped 0..1) and per-sample
    `scale` (so bbox sizes can be scaled consistently).

    Device-safe: all auxiliary tensors are built on the batch's device and the
    targets are moved there too, so it works on CUDA (the CPU-default
    torch.zeros/eye/full/rand would otherwise mismatch a CUDA batch)."""
    B = batch_u8.shape[0]
    dev = batch_u8.device
    targets = targets.to(dev)
    theta = torch.zeros(B, device=dev)
    scale = torch.zeros(B, device=dev)
    txy = torch.zeros(B, 2, device=dev)
    flipm = torch.zeros(B, dtype=torch.bool, device=dev)
    for i in range(B):
        theta[i] = rng.uniform(-0.26, 0.26)        # +-15 deg
        scale[i] = rng.uniform(0.78, 1.28)
        txy[i, 0] = rng.uniform(-0.10, 0.10)
        txy[i, 1] = rng.uniform(-0.10, 0.10)
        flipm[i] = flip and rng.random() < 0.5
    cos, sin = torch.cos(theta), torch.sin(theta)
    R = torch.stack([cos, -sin, sin, cos], dim=1).view(B, 2, 2)
    Fm = torch.eye(2, device=dev).repeat(B, 1, 1)
    Fm[flipm, 0, 0] = -1.0
    f = torch.zeros(B, 2, device=dev)
    f[flipm, 0] = 1.0
    c = torch.full((B, 2), 0.5, device=dev)
    A = scale.view(B, 1, 1) * (R @ Fm)
    b = (scale.view(B, 1, 1) * (R @ (f - c).unsqueeze(-1))).squeeze(-1) + c + txy
    tout = torch.einsum("bij,bkj->bki", A, targets) + b.unsqueeze(1)
    Ainv = torch.inverse(A)
    M = torch.zeros(B, 2, 3, device=dev)
    M[:, :, :2] = Ainv
    M[:, :, 2] = (Ainv @ (1.0 - 2.0 * b).unsqueeze(-1)).squeeze(-1) - 1.0
    grid = F.affine_grid(M, batch_u8.shape, align_corners=False)
    xf = F.grid_sample(batch_u8.float() / 255.0, grid,
                       align_corners=False, padding_mode="border")
    gain = 0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=dev)
    xf = (xf * gain).clamp(0, 1)
    return (xf - 0.5) / 0.5, tout.clamp(0.0, 1.0), scale


def _jitter(batch_u8, flip=True):
    """Photometric jitter + optional flip for the tile head (no coordinates).
    Device-safe: every auxiliary tensor is created on the input's device, so it
    works on CUDA (the default torch.rand/zeros land on CPU and would otherwise
    mismatch a CUDA batch)."""
    dev = batch_u8.device
    x = batch_u8.float() / 255.0
    if flip:
        m = torch.rand(x.shape[0], device=dev) < 0.5
        x[m] = torch.flip(x[m], dims=[3])
    gain = 0.85 + 0.30 * torch.rand(x.shape[0], 1, 1, 1, device=dev)
    x = (x * gain).clamp(0, 1)
    return (x - 0.5) / 0.5


def _hard_photometric(x, rng):
    """Phase-2 'hardening' photometrics, applied in tensor space (no PIL
    round-trip): per-channel colour cast, brightness, contrast and sensor
    noise. x is a float batch in [-1, 1]."""
    B = x.shape[0]
    dev = x.device
    ch = 0.72 + 0.56 * torch.rand(B, 3, 1, 1, device=dev)
    bri = 0.67 + 0.65 * torch.rand(B, 1, 1, 1, device=dev)
    x = x * ch * bri
    if rng.random() < 0.8:
        sigma = rng.uniform(0.02, 0.07)
        x = x + torch.randn_like(x) * sigma
    if rng.random() < 0.4:
        con = 0.80 + 0.55 * torch.rand(B, 1, 1, 1, device=dev)
        x = x * con
    return x.clamp(-1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Losses (heatmap spatial-CE + soft-argmax L1, same recipe as train_models)
# ═══════════════════════════════════════════════════════════════════════════

def _spatial_ce(hm_all, target_cls, masks, l1_targets, single_rows):
    """Per-cell classification with a background channel, multi-instance.

    Identical formulation to train_models._point_loss_bg: instance cells are
    supervised to their class with 5x weight, every other cell to background,
    plus 4x soft-argmax L1 on single-instance rows.
    """
    B, C1, H, W = hm_all.shape
    device = hm_all.device
    cells = masks.reshape(B, H * W)
    bg = torch.full((B, H * W), C1 - 1, dtype=torch.long, device=device)
    labels = torch.where(cells > 0.5,
                         target_cls.view(B, 1).expand(B, H * W), bg)
    logits = hm_all.reshape(B, C1, H * W).permute(0, 2, 1).reshape(-1, C1)
    nll = F.cross_entropy(logits, labels.reshape(-1),
                          reduction="none").reshape(B, H * W)
    w = 1.0 + 4.0 * cells
    ce = (nll * w).sum() / w.sum()
    sel = hm_all.gather(1, target_cls.view(-1, 1, 1, 1).expand(-1, 1, H, W)
                        ).squeeze(1)
    l1 = torch.zeros((), device=device)
    if single_rows.any():
        l1 = F.l1_loss(soft_argmax2d(sel)[single_rows], l1_targets[single_rows])
    return ce + 4.0 * l1


def _channel_ce_l1(hm_sel, target_xy):
    """Spatial CE on the target cell + 4x soft-argmax L1 (drag channels)."""
    B, H, W = hm_sel.shape
    cx = (target_xy[:, 0] * W).long().clamp(0, W - 1)
    cy = (target_xy[:, 1] * H).long().clamp(0, H - 1)
    cell = (cy * W + cx).view(-1)
    ce = F.cross_entropy(hm_sel.reshape(B, H * W), cell)
    l1 = F.l1_loss(soft_argmax2d(hm_sel), target_xy)
    return ce + 4.0 * l1

# ═══════════════════════════════════════════════════════════════════════════
#  Training: joint multi-task loop over every family (v2)
# ═══════════════════════════════════════════════════════════════════════════

def _split(n):
    idx = list(range(n))
    return [i for i in idx if i % 20], [i for i in idx if not i % 20]


def _instances(meta):
    if meta.get("type") == "count":
        return [(o["x"], o["y"]) for o in meta["objects"]]
    return [(meta["x"], meta["y"])]


def _move(t, device):
    return t.to(device, non_blocking=True)


def _checkpoint(model, corpus, models_dir, epoch, verbose=True,
                opt=None, scaler=None, step_ctr=None):
    """Save brain.pt + brain.json after every epoch so an interrupt (Ctrl+C)
    or a session timeout never loses progress. Writes weights + the full arch
    sidecar (no held-out metrics - those are added by the final _save_brain).
    BrainSolver can load this checkpoint as-is at any time.

    With ``opt`` given it also writes resume.pt — the FULL training state
    (weights + optimizer + AMP scaler + epoch + scheduler position) — so a
    killed 12 h Kaggle session can be continued with --resume from the next
    session instead of restarting at epoch 1. Also writes one loadable
    brain file per epoch (brain_epNN.pt) so every epoch is a usable brain
    and a valid --resume warm-start point."""
    os.makedirs(models_dir, exist_ok=True)
    pt = os.path.join(models_dir, "brain.pt")
    torch.save(model.state_dict(), pt)
    try:
        torch.save(model.state_dict(),
                   os.path.join(models_dir, "brain_ep%02d.pt" % epoch))
    except OSError:
        pass
    sidecar = {
        "kind": "brain", "version": 2, "hr": True,
        "classes": CLASSES, "families": FAMILIES,
        "n_classes": N_CLASSES,
        "arch": {
            "width": model.width,
            "prompt_dim": model.prompt_dim,
            "prompt_layers": model.prompt_layers,
            "d_concept": model.d_concept,
            "pattern_d": model.pattern_d,
            "pattern_layers": model.pattern_layers,
            "text_len": model.text_len,
        },
        "tile_size": corpus["tile_size"], "scene_size": corpus["scene_size"],
        "epoch": epoch, "metrics": {}, "size_mb": os.path.getsize(pt) / 1e6,
    }
    with open(os.path.join(models_dir, "brain.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    if opt is not None:
        # atomic: write .tmp then rename, so a mid-write kill can never
        # corrupt the previous good resume point
        tmp = os.path.join(models_dir, "resume.pt.tmp")
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step_ctr": step_ctr[0] if step_ctr is not None else 0,
            "n_classes": N_CLASSES,
            "arch": {k: getattr(model, k) for k in
                     ("width", "prompt_dim", "prompt_layers", "d_concept",
                      "pattern_d", "pattern_layers")},
        }, tmp)
        os.replace(tmp, os.path.join(models_dir, "resume.pt"))
    if verbose:
        print("    [checkpoint] models/brain.pt saved (after epoch %d)"
              % epoch)


def _find_resume(explicit, models_dir):
    """Resolve --resume: an explicit path, or (bare --resume / 'auto') the
    newest checkpoint in the models dir, else in any Kaggle input upload.
    Preference order: resume.pt (full state) > brain_epNN.pt (newest) >
    brain.pt + brain.json (the latest per-epoch weights)."""
    if not explicit:
        return None
    if explicit != "auto":
        return explicit if os.path.isfile(explicit) else None
    import glob
    cand = os.path.join(models_dir, "resume.pt")
    if os.path.isfile(cand):
        return cand
    epcs = glob.glob(os.path.join(models_dir, "brain_ep*.pt"))
    if epcs:
        return max(epcs, key=os.path.getmtime)
    if os.path.isdir("/kaggle/input"):
        for pat in ("resume.pt", "brain_ep*.pt"):
            hits = glob.glob("/kaggle/input/**/%s" % pat, recursive=True)
            if hits:
                return max(hits, key=os.path.getmtime)
        plain = glob.glob("/kaggle/input/**/brain.pt", recursive=True)
        if plain:
            return max(plain, key=os.path.getmtime)
    return None


def _save_brain(model, metrics, corpus, models_dir):
    """Final save: weights + sidecar with the full held-out metric table."""
    os.makedirs(models_dir, exist_ok=True)
    pt = os.path.join(models_dir, "brain.pt")
    torch.save(model.state_dict(), pt)
    sidecar = {
        "kind": "brain", "version": 2, "hr": True,
        "classes": CLASSES, "families": FAMILIES,
        "n_classes": N_CLASSES,
        "arch": {
            "width": model.width,
            "prompt_dim": model.prompt_dim,
            "prompt_layers": model.prompt_layers,
            "d_concept": model.d_concept,
            "pattern_d": model.pattern_d,
            "pattern_layers": model.pattern_layers,
            "text_len": model.text_len,
        },
        "tile_size": corpus["tile_size"], "scene_size": corpus["scene_size"],
        "metrics": metrics, "size_mb": os.path.getsize(pt) / 1e6,
    }
    with open(os.path.join(models_dir, "brain.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    return sidecar


def split_brain_parts(pt_path, part_dir=None, max_mb=96, prefix="brain_part"):
    """Split models/brain.pt into `brain_part_NN` files (<= max_mb each) at
    the repo root — the Test tab (brain_test.py) reassembles them on any
    machine (loose files, git, or GitHub raw). Keeps GitHub under its 100 MB
    per-file limit, so a 1.1 GB Brain ships as ~12 parts. Returns part count."""
    part_dir = part_dir or ROOT
    size = os.path.getsize(pt_path)
    n = max(1, math.ceil(size / (max_mb * 1024 * 1024)))
    chunk = math.ceil(size / n)
    for f in os.listdir(part_dir):
        if f.startswith(prefix + "_"):
            try:
                os.remove(os.path.join(part_dir, f))
            except OSError:
                pass
    with open(pt_path, "rb") as src:
        i = 0
        while True:
            data = src.read(chunk)
            if not data:
                break
            with open(os.path.join(part_dir, "%s_%02d" % (prefix, i)), "wb") as out:
                out.write(data)
            i += 1
    return i


def write_arch_sidecar(corpus, arch, out_path=None):
    """brain_arch.json at the repo root: the arch a committed set of
    brain_part_NN was trained with, so brain_test.py can rebuild models/
    brain.json with the RIGHT shape when it reassembles loose parts."""
    out_path = out_path or os.path.join(ROOT, "brain_arch.json")
    d = {
        "kind": "brain", "version": 2, "hr": True,
        "classes": CLASSES, "families": FAMILIES,
        "n_classes": N_CLASSES,
        "arch": arch,
        "tile_size": corpus["tile_size"], "scene_size": corpus["scene_size"],
    }
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    return out_path


def _focus_and_acc(model, corpus, tile_tr, device, ep, sample=2048,
                   verbose=True):
    """Confusion mining: sample tile train rows, count misclassified classes,
    return (indices of the top-12 most-confused classes, sampled accuracy).
    One third of the NEXT epoch's tile steps train on that focus list — the
    brain keeps hammering exactly the confusions it still makes."""
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    model.eval()
    rng = random.Random(999 + ep)
    idx = rng.sample(tile_tr, min(sample, len(tile_tr)))
    wrong = collections.Counter()
    ok = 0
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            x = _to_float(_move(corpus["tile_x"][b], device))
            pred = model.tile_logits(model.features(x)).argmax(1)
            tgt = _move(corpus["tile_y"][b], device)
            ok += int((pred == tgt).sum())
            mis = (pred != tgt)
            for p, t in zip(pred[mis].cpu().tolist(), tgt[mis].cpu().tolist()):
                wrong[t] += 1
    model.train()
    acc = ok / max(1, len(idx))
    if not wrong:
        log("    epoch %d  focus: no confusions in the sample (acc %.3f)"
            % (ep + 1, acc))
        return None, acc
    top = [c for c, _ in wrong.most_common(12)]
    y = corpus["tile_y"]
    focus = [i for i in tile_tr if int(y[i]) in top]
    log("    epoch %d  tile acc %.3f | focus classes: %s" % (
        ep + 1, acc,
        ", ".join(CLASSES[c] for c in top[:6]) +
        (" ..." if len(top) > 6 else "")))
    return focus, acc


def _class_tolerant_load(model, ck, log):
    """Warm start when a checkpoint has FEWER classes than the current
    vocabulary.

    Class ids are stable (new classes are APPENDED at the end), so a
    1000-class giga brain carries over perfectly into a 1003-class one:
    the shared backbone is copied verbatim; for the class-dependent
    tensors (tile head, knowledge bank, heatmap channels) the overlapping
    class rows are copied and the new class rows keep their init values
    (the heatmap background channel is relocated to its new slot).

    Returns True if the model was updated in place, False otherwise.
    """
    try:
        n_old = int(ck["tile_head.fc.bias"].shape[0])
    except Exception:  # noqa: BLE001
        return False
    n_new = int(model.n_classes)
    if not (0 < n_old < n_new):
        return False
    want = model.state_dict()
    filt = {}
    for k, v in want.items():
        src = ck.get(k)
        if src is None:
            continue
        if tuple(src.shape) == tuple(v.shape):
            filt[k] = src
            continue
        if k == "knowledge.concept.weight":            # (C, d)
            t = v.clone()
            t[:n_old].copy_(src)
            filt[k] = t
        elif k == "knowledge.rel":                     # (C, C)
            t = v.clone()
            t[:n_old, :n_old].copy_(src)
            filt[k] = t
        elif k == "tile_head.fc.weight":               # (C, c_in)
            t = v.clone()
            t[:n_old].copy_(src)
            filt[k] = t
        elif k == "tile_head.fc.bias":                 # (C,)
            t = v.clone()
            t[:n_old].copy_(src)
            filt[k] = t
        elif k in ("heatmap_head.lo.weight", "heatmap_head.hi.weight"):
            # (C+1, c, 1, 1): channel i = class i, channel C = background
            t = v.clone()
            t[:n_old].copy_(src[:n_old])
            t[n_new].copy_(src[n_old])                 # bg -> new slot
            filt[k] = t
        elif k in ("heatmap_head.lo.bias", "heatmap_head.hi.bias"):
            t = v.clone()
            t[:n_old].copy_(src[:n_old])
            t[n_new].copy_(src[n_old])
            filt[k] = t
        # any other shape mismatch: keep init (not in filt)
    if len(filt) < len(want) // 2:
        return False  # too much missing (foreign/v1 format) - don't risk it
    model.load_state_dict(filt, strict=False)
    log("   resume: class-tolerant warm start - %d -> %d classes "
        "(%d shared classes carried over, %d new classes initialised)"
        % (n_old, n_new, n_old, n_new - n_old))
    return True


def train_brain(epochs=12, batch=64, lr=1e-3, seed=0,
                device=None, corpus=None, corpus_kwargs=None,
                models_dir=MODELS_DIR, resume=None, verbose=True,
                preset="large", width=None,
                prompt_dim=None, prompt_layers=None, d_concept=None,
                pattern_d=None, pattern_layers=None,
                phase2=0, kreg=0.02, focus=True, amp=None):
    """Train every head of the Brain jointly (v2).

    Phase 1 (``epochs``): every family, clean/soft/hard degradation mix as
    built into the corpus, warmup + cosine schedule, optional confusion-focus
    tile steps, ontology anchor (kreg).
    Phase 2 (``phase2`` extra epochs): the HARDENING pass — lr x 0.3, extra
    tensor-space hard photometrics on every image batch, focus mining on.

    Each epoch cycles through the families (tile, point, count, drag, bbox,
    text, pattern, router); each step trains ONE head with its own loss.
    Cycling families (rather than per-sample masking) is the standard,
    stable way to train a multi-task net and keeps every batch loss clean.
    """
    assert _TORCH, "torch is required to train the Brain"
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    # Resolve the device gracefully: if the user asked for cuda but this torch
    # build has no CUDA (e.g. the CPU wheel got installed on a GPU machine),
    # warn clearly and fall back to CPU instead of crashing on .to('cuda').
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        log("  [brain] WARNING: --device cuda requested, but this torch build has")
        log("  [brain] no CUDA (it's the CPU wheel: %s)." % torch.__version__)
        log("  [brain] On Kaggle: do NOT 'pip install torch' (CUDA torch is")
        log("  [brain] preinstalled on GPU images) - restart the session to restore")
        log("  [brain] it, or run:  !pip install -q torch --index-url "
            "https://download.pytorch.org/whl/cu121")
        log("  [brain] Falling back to CPU for now (will be slow).")
        device = "cpu"
    n_gpu = torch.cuda.device_count() if device.startswith("cuda") else 0
    use_amp = bool(device.startswith("cuda") and
                   (amp if amp is not None else True))

    # architecture: preset, with explicit overrides
    P = dict(PRESETS[preset])
    for k, v in (("width", width), ("prompt_dim", prompt_dim),
                 ("prompt_layers", prompt_layers), ("d_concept", d_concept),
                 ("pattern_d", pattern_d), ("pattern_layers", pattern_layers)):
        if v is not None:
            P[k] = v
    if corpus is None:
        corpus = build_brain_corpus(seed=seed, **(corpus_kwargs or {}))
    tile_size = corpus["tile_size"]
    scene_size = corpus["scene_size"]
    Sfeat = scene_size // 4      # v2: fused dual-resolution heatmap size
    model = Brain(N_CLASSES, **P).to(device)
    if n_gpu:
        torch.backends.cudnn.benchmark = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # held-out splits per family (every 20th index)
    tile_tr, tile_va = _split(len(corpus["tile_y"]))
    point_tr, point_va = _split(len(corpus["point_m"]))
    point_va_single = [i for i in point_va if corpus["point_m"][i].get("type") != "count"]
    count_tr, count_va = _split(len(corpus["count_m"]))   # count joins point head
    drag_tr, drag_va = _split(len(corpus["drag_m"]))
    bbox_tr, bbox_va = _split(len(corpus["bbox_m"]))
    text_tr, text_va = (_split(len(corpus["text_m"]))
                        if len(corpus["text_m"]) else ([], []))
    pat_tr, pat_va = _split(len(corpus["pat_m"]))
    router = corpus["router"]
    router_tr, router_va = _split(len(router))
    # router: balance families (the bank is binary-heavy) via round-robin
    router_by_fam = collections.defaultdict(list)
    for i, (_p, fam) in enumerate(router):
        if i % 20 == 0:
            continue
        router_by_fam[fam].append(i)

    total_epochs = epochs + phase2
    pat_batch = max(4, batch // 4)
    steps_per_epoch = (math.ceil(len(tile_tr) / batch) +
                       math.ceil(len(point_tr) / batch) +
                       math.ceil(len(count_tr) / batch) +
                       math.ceil(len(drag_tr) / batch) +
                       math.ceil(len(bbox_tr) / batch) +
                       (math.ceil(len(text_tr) / batch) if text_tr else 0) +
                       math.ceil(len(pat_tr) / pat_batch) +
                       sum(math.ceil(len(v) / 64) for v in router_by_fam.values()))
    total_steps = max(1, steps_per_epoch * total_epochs)
    warmup = max(1, int(0.05 * total_steps))
    step_ctr = [0]

    def _lr_fn(_s):
        s = step_ctr[0]
        if s < warmup:
            return (s + 1) / float(warmup)
        p = min(1.0, (s - warmup) / float(max(1, total_steps - warmup)))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_fn)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _place_schedule():
        """Put the LR schedule exactly at the current step_ctr position in
        one shot (no N-step loop). Works because _lr_fn reads step_ctr
        directly rather than the scheduler's internal epoch counter."""
        sched.last_epoch = step_ctr[0]
        for pg, base_lr in zip(sched.optimizer.param_groups,
                               sched.base_lrs):
            pg["lr"] = base_lr * _lr_fn(step_ctr[0])

    log("== Brain v2 [%s]: %.1fM params (%.1f MB fp32) | device %s%s%s ==" % (
        preset, sum(p.numel() for p in model.parameters()) / 1e6,
        model.param_mb(), device,
        (" (%d GPU)" % n_gpu) if n_gpu else "",
        " | AMP" if use_amp else ""))
    log("   corpus: %d tile images (incl. %d real hcap + %d real-photo "
        "views), %d point, %d count, %d drag, %d bbox, %d text, %d pattern, "
        "%d grids, %d router pairs" % (
            len(corpus["tile_y"]), corpus.get("hcap_n", 0),
            corpus.get("photo_n", 0),
            len(corpus["point_m"]), len(corpus["count_m"]),
            len(corpus["drag_m"]), len(corpus["bbox_m"]),
            len(corpus["text_m"]), len(corpus["pat_m"]),
            len(corpus["grid_m"]), len(router)))
    log("   schedule: %d epochs (%d phase-2 hardening) x ~%d steps, "
        "warmup %d, kreg %.3f, focus %s" % (
            total_epochs, phase2, steps_per_epoch, warmup, kreg, focus))

    # ── resume: continue a killed session (Kaggle 12h cap) ────────────────
    # Two kinds of checkpoint are accepted:
    #   resume.pt            full state (weights+optimizer+scaler+epoch)
    #   brain.pt/brain_epNN  weights only -> "warm start" (fresh optimizer,
    #                        continue from the epoch the file represents)
    start_ep = 0
    if resume:
        if not os.path.isfile(resume):
            log("   resume: %r not found - starting fresh" % resume)
        else:
            try:
                ck = torch.load(resume, map_location=device,
                                weights_only=False)
            except Exception as e:  # noqa: BLE001 - corrupt/foreign file
                log("   resume: could not load %s (%s) - starting fresh"
                    % (resume, e))
                ck = None
            if ck is not None:
                if isinstance(ck, dict) and ("model" in ck or "opt" in ck):
                    # ---- full training state (resume.pt) ----
                    ok = (ck.get("n_classes") == N_CLASSES
                          and all(ck.get("arch", {}).get(k)
                                  == getattr(model, k)
                                  for k in ("width", "prompt_dim",
                                            "prompt_layers", "d_concept",
                                            "pattern_d", "pattern_layers")))
                    if not ok:
                        if ck.get("n_classes") not in (None, N_CLASSES):
                            log("   resume: class count changed %s -> %d "
                                "(%s): the optimizer state can't transfer, "
                                "but the WEIGHTS can - trying to carry the "
                                "shared classes over"
                                % (ck.get("n_classes"), N_CLASSES, resume))
                        else:
                            log("   resume: arch mismatch (%s)" % resume)
                        # salvage: transfer the weights class-tolerantly,
                        # keep the checkpoint's epoch counter, discard the
                        # optimizer (fresh AdamW moments are fine)
                        if _class_tolerant_load(model, ck.get("model", {}),
                                                log):
                            ep = int(ck.get("epoch", 0))
                            start_ep = min(ep, total_epochs - 1)
                            step_ctr[0] = start_ep * steps_per_epoch
                            _place_schedule()
                            log("   resume: weights carried over from %s "
                                "(epoch %d) - fresh optimizer, continuing "
                                "from epoch %d/%d"
                                % (resume, ep, start_ep, total_epochs))
                        else:
                            log("   resume: weights don't fit either - "
                                "starting fresh")
                    else:
                        # optimizer state first: if it doesn't fit, NOTHING
                        # is applied and we start fresh; if it fits, the
                        # weights must too (same param structure), so a
                        # failure there crashes loudly instead of silently
                        # continuing with a corrupted mix
                        try:
                            opt.load_state_dict(ck["opt"])
                        except Exception as e:  # noqa: BLE001
                            log("   resume: optimizer state doesn't match "
                                "(%s) - starting fresh" % e)
                        else:
                            model.load_state_dict(ck["model"])
                            if ck.get("scaler") is not None and \
                                    scaler is not None:
                                scaler.load_state_dict(ck["scaler"])
                            step_ctr[0] = int(ck.get("step_ctr", 0))
                            _place_schedule()
                            start_ep = min(int(ck.get("epoch", 0)),
                                           total_epochs - 1)
                            log("   resume: loaded %s - continuing from "
                                "epoch %d/%d (%d steps done)"
                                % (resume, start_ep, total_epochs,
                                   step_ctr[0]))
                else:
                    # ---- weights only (brain.pt / brain_epNN.pt) ----
                    base = os.path.basename(resume)
                    m = re.match(r"brain_ep(\d+)\.pt$", base)
                    ep = int(m.group(1)) if m else 0
                    if ep == 0:
                        # sidecar next to the file, or (separate-dataset
                        # uploads) the newest brain.json anywhere in inputs
                        import glob
                        sjs = [os.path.join(os.path.dirname(resume) or ".",
                                            "brain.json")]
                        if os.path.isdir("/kaggle/input"):
                            sjs += sorted(
                                glob.glob("/kaggle/input/**/brain.json",
                                          recursive=True),
                                key=os.path.getmtime)
                        for sj in sjs:
                            if os.path.isfile(sj):
                                try:
                                    ep = int(json.load(open(sj)).get("epoch", 0))
                                except Exception:  # noqa: BLE001
                                    ep = 0
                            if ep:
                                break
                    loaded = True
                    try:
                        model.load_state_dict(ck, strict=True)
                    except Exception as e:  # noqa: BLE001
                        loaded = _class_tolerant_load(model, ck, log)
                        if not loaded:
                            log("   resume: weights don't match this preset "
                                "(%s) - starting fresh" % e)
                    if loaded:
                        start_ep = min(ep, total_epochs - 1)
                        step_ctr[0] = start_ep * steps_per_epoch
                        _place_schedule()
                        log("   resume: warm-started weights from %s "
                            "(epoch %d) - fresh optimizer, continuing from "
                            "epoch %d/%d"
                            % (resume, ep, start_ep, total_epochs))

    ont_target = torch.from_numpy(
        ontology_targets(N_CLASSES, CLASSES)).float().to(device)
    focus_idx = None

    def _step_count_metas(metas, ids, device):
        """Build (target_cls, masks-ready pts, valid, K) for a point/count
        batch, on ``device``. pts carries -1 padding for unused instance
        slots; valid marks the real ones (used to build the heatmap masks)."""
        K = 1
        for i in ids:
            K = max(K, len(_instances(metas[i])))
        pts = torch.full((len(ids), K, 2), -1.0, device=device)
        tc = torch.zeros(len(ids), dtype=torch.long, device=device)
        for row, i in enumerate(ids):
            inst = _instances(metas[i])
            pts[row, :len(inst)] = torch.tensor(inst, device=device)
            tc[row] = metas[i]["target_id"]
        valid = (pts >= 0).all(dim=2)
        return pts, tc, valid, K

    for ep in range(start_ep, total_epochs):
        p2 = ep >= epochs
        model.train()
        random.Random(seed * 100 + ep).shuffle(tile_tr)
        random.Random(seed * 100 + ep).shuffle(point_tr)
        random.Random(seed * 100 + ep).shuffle(drag_tr)
        random.Random(seed * 100 + ep).shuffle(bbox_tr)
        random.Random(seed * 100 + ep).shuffle(pat_tr)
        for v in router_by_fam.values():
            random.Random(seed * 100 + ep).shuffle(v)
        ep_loss = 0.0
        n_steps = 0
        t0 = time.time()
        log("  epoch %d/%d%s  training on %s ..." % (
            ep + 1, total_epochs,
            "  [PHASE 2: hardening]" if p2 else "", device))

        # 1) TILE head — painted tiles + real hcap tiles + grid tiles. The
        # backbone is trained here too (NO no_grad): this is the main
        # class-recognition signal, and freezing it was why tile accuracy
        # plateaued. Every 3rd step trains on the confusion-focus list.
        n_tile_steps = math.ceil(len(tile_tr) / batch)
        for s in range(n_tile_steps):
            if (focus_idx and len(focus_idx) >= batch and s % 3 == 0):
                rr = random.Random(seed * 7777 + ep * 5000 + s)
                b = rr.sample(focus_idx, min(batch, len(focus_idx)))
            else:
                b = tile_tr[s * batch:(s + 1) * batch]
            x = _jitter(_move(corpus["tile_x"][b], device))
            if p2:
                x = _hard_photometric(x, random.Random(seed + s))
            with torch.amp.autocast("cuda", enabled=use_amp):
                feat = model.features(x)
                logits = model.tile_logits(feat)
                loss = F.cross_entropy(
                    logits, _move(corpus["tile_y"][b], device),
                    label_smoothing=0.05)
                if kreg > 0:
                    loss = loss + kreg * F.mse_loss(
                        model.knowledge.concept.weight[:, :ont_target.shape[1]],
                        ont_target)
            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            step_ctr[0] += 1
            sched.step()
            ep_loss += loss.item(); n_steps += 1
            if n_tile_steps >= 300 and (s + 1) % 300 == 0:
                log("    epoch %d  tiles %d/%d  loss %.3f" % (
                    ep + 1, s + 1, n_tile_steps, loss.item()))

        # 2) POINT head — single + relational point rounds
        n_pt_steps = math.ceil(len(point_tr) / batch)
        for s in range(n_pt_steps):
            b = point_tr[s * batch:(s + 1) * batch]
            pts, tc, valid, K = _step_count_metas(corpus["point_m"], b, device)
            x, tpts, _ = _prep_geom(_move(corpus["point_x"][b], device), pts,
                                    random.Random(seed * 100000 + ep * 1000 + s))
            if p2:
                x = _hard_photometric(x, random.Random(seed * 3 + s))
            with torch.amp.autocast("cuda", enabled=use_amp):
                f8, f4 = model.features2(x)
                hm = model.heatmaps(f8, f4)
                masks = torch.zeros(len(b), Sfeat, Sfeat, device=device)
                cx = (tpts[:, :, 0] * Sfeat).long().clamp(0, Sfeat - 1)
                cy = (tpts[:, :, 1] * Sfeat).long().clamp(0, Sfeat - 1)
                rows = torch.arange(len(b), device=device).view(-1, 1).expand(len(b), K)
                keep = valid.reshape(len(b), K)
                masks[rows[keep], cy[keep], cx[keep]] = 1.0
                single = masks.sum(dim=(1, 2)) == 1
                l1_xy = tpts.sum(dim=1) / keep.sum(dim=1, keepdim=True).clamp(min=1)
                loss = _spatial_ce(hm, tc, masks, l1_xy, single)
            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            step_ctr[0] += 1; sched.step()
            ep_loss += loss.item(); n_steps += 1
            if n_pt_steps >= 200 and (s + 1) % 200 == 0:
                log("    epoch %d  point %d/%d  loss %.3f" % (
                    ep + 1, s + 1, n_pt_steps, loss.item()))

        # 3) COUNT head — multi-instance count rounds through the same heatmap head
        if count_tr:
            for s in range(math.ceil(len(count_tr) / batch)):
                b = count_tr[s * batch:(s + 1) * batch]
                pts, tc, valid, K = _step_count_metas(corpus["count_m"], b, device)
                x, tpts, _ = _prep_geom(_move(corpus["count_x"][b], device), pts,
                                        random.Random(seed * 100000 + ep * 2000 + s))
                if p2:
                    x = _hard_photometric(x, random.Random(seed * 5 + s))
                with torch.amp.autocast("cuda", enabled=use_amp):
                    f8, f4 = model.features2(x)
                    hm = model.heatmaps(f8, f4)
                    masks = torch.zeros(len(b), Sfeat, Sfeat, device=device)
                    cx = (tpts[:, :, 0] * Sfeat).long().clamp(0, Sfeat - 1)
                    cy = (tpts[:, :, 1] * Sfeat).long().clamp(0, Sfeat - 1)
                    rows = torch.arange(len(b), device=device).view(-1, 1).expand(len(b), K)
                    keep = valid.reshape(len(b), K)
                    masks[rows[keep], cy[keep], cx[keep]] = 1.0
                    single = masks.sum(dim=(1, 2)) == 1
                    l1_xy = tpts.sum(dim=1) / keep.sum(dim=1, keepdim=True).clamp(min=1)
                    loss = _spatial_ce(hm, tc, masks, l1_xy, single)
                opt.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                step_ctr[0] += 1; sched.step()
                ep_loss += loss.item(); n_steps += 1

        # 4) DRAG head — piece + slot heatmaps (drag + pipe + tower + shape)
        for s in range(math.ceil(len(drag_tr) / batch)):
            b = drag_tr[s * batch:(s + 1) * batch]
            tf = torch.tensor([[corpus["drag_m"][i]["fx"],
                                corpus["drag_m"][i]["fy"]] for i in b])
            tt = torch.tensor([[corpus["drag_m"][i]["tx"],
                                corpus["drag_m"][i]["ty"]] for i in b])
            tgt = torch.cat([tf.unsqueeze(1), tt.unsqueeze(1)], dim=1)
            x, txy, _ = _prep_geom(_move(corpus["drag_x"][b], device), tgt,
                                   random.Random(seed * 100000 + ep * 3000 + s))
            if p2:
                x = _hard_photometric(x, random.Random(seed * 7 + s))
            with torch.amp.autocast("cuda", enabled=use_amp):
                f8, f4 = model.features2(x)
                hms = model.drag_maps(f8, f4)
                loss = (_channel_ce_l1(hms[:, 0], txy[:, 0]) +
                        _channel_ce_l1(hms[:, 1], txy[:, 1]))
            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            step_ctr[0] += 1; sched.step()
            ep_loss += loss.item(); n_steps += 1

        # 5) BBOX head — centre heatmap + size regression
        for s in range(math.ceil(len(bbox_tr) / batch)):
            b = bbox_tr[s * batch:(s + 1) * batch]
            ctr = torch.tensor([[corpus["bbox_m"][i]["cx"],
                                 corpus["bbox_m"][i]["cy"]] for i in b])
            wh = torch.tensor([[corpus["bbox_m"][i]["w"],
                                corpus["bbox_m"][i]["h"]] for i in b], device=device)
            x, tctr, scl = _prep_geom(_move(corpus["bbox_x"][b], device),
                                      ctr.unsqueeze(1),
                                      random.Random(seed * 100000 + ep * 4000 + s))
            if p2:
                x = _hard_photometric(x, random.Random(seed * 11 + s))
            tctr = tctr[:, 0]
            twh = (wh * scl.unsqueeze(1)).clamp(0.02, 1.0)
            with torch.amp.autocast("cuda", enabled=use_amp):
                f8, f4 = model.features2(x)
                cm, pw = model.bbox(f8, f4)
                loss = (_channel_ce_l1(cm, tctr) + 2.0 * F.l1_loss(pw, twh))
            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            step_ctr[0] += 1; sched.step()
            ep_loss += loss.item(); n_steps += 1

        # 5.5) TEXT head - "type the text you see" codes (per-char CE)
        if text_tr:
            for s in range(math.ceil(len(text_tr) / batch)):
                b = text_tr[s * batch:(s + 1) * batch]
                x = _jitter(_move(corpus["text_x"][b], device), flip=False)
                if p2:
                    x = _hard_photometric(x, random.Random(seed * 13 + s))
                with torch.amp.autocast("cuda", enabled=use_amp):
                    feat = model.features(x)
                    logits = model.text_logits(feat).view(
                        len(b), model.text_head.text_len, model.text_head.n_chars)
                    tgt = torch.tensor(
                        [[TEXT_ALPHABET.index(ch) for ch in corpus["text_m"][i]["text"]]
                         for i in b], device=device)
                    loss = F.cross_entropy(
                        logits.reshape(len(b) * model.text_head.text_len,
                                       model.text_head.n_chars),
                        tgt.reshape(-1))
                opt.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                step_ctr[0] += 1; sched.step()
                ep_loss += loss.item(); n_steps += 1

        # 6) PATTERN reasoner — set-transformer over cells + candidates
        for s in range(math.ceil(len(pat_tr) / pat_batch)):
            b = pat_tr[s * pat_batch:(s + 1) * pat_batch]
            cell_feats, cand_feats, prompts, correct = [], [], [], []
            for i in b:
                m = corpus["pat_m"][i]
                im = corpus["pat_imgs"][i]
                W, H = im.size
                cells, cands = [], []
                for box in m["cell_boxes"]:
                    x0, y0 = box["x"] * W, box["y"] * H
                    w0, h0 = box["w"] * W, box["h"] * H
                    crop = im.crop((x0, y0, x0 + w0, y0 + h0))
                    cells.append(_img_to_u8(crop, tile_size))
                for box in m["candidate_boxes"]:
                    x0, y0 = box["x"] * W, box["y"] * H
                    w0, h0 = box["w"] * W, box["h"] * H
                    crop = im.crop((x0, y0, x0 + w0, y0 + h0))
                    cands.append(_img_to_u8(crop, tile_size))
                cell_feats.append(torch.stack(cells))   # (9,3,s,s)
                cand_feats.append(torch.stack(cands))   # (3,3,s,s)
                prompts.append(m.get("prompt", ""))
                correct.append(m["correct"])
            # backbone pass on all 12 crops per sample, batched
            cells = _move(torch.stack(cell_feats).view(-1, 3, tile_size, tile_size), device)
            cands = _move(torch.stack(cand_feats).view(-1, 3, tile_size, tile_size), device)
            cells = _jitter(cells, flip=False)
            cands = _jitter(cands, flip=False)
            with torch.no_grad():
                cf = F.adaptive_avg_pool2d(model.features(cells), 1).flatten(1)
                xf = F.adaptive_avg_pool2d(model.features(cands), 1).flatten(1)
            cf = cf.view(len(b), 9, -1)
            xf = xf.view(len(b), 3, -1)
            pv = model.prompt_enc(prompts)
            logits = model.pattern(cf, xf, pv)
            loss = F.cross_entropy(logits, torch.tensor(correct, device=device))
            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            step_ctr[0] += 1; sched.step()
            ep_loss += loss.item(); n_steps += 1

        # 7) ROUTER head — prompt -> family, family-balanced round-robin over
        # the ~30k-pair bank (no image gradient needed through the backbone;
        # a detached pooled feature of a random scene is not even needed:
        # the router learns primarily from text, as in v1).
        for fam in FAMILIES:
            ids = router_by_fam.get(fam)
            if not ids:
                continue
            # 64 prompts x 160 chars per step: the giga prompt transformer
            # (1024-d x 13 layers) keeps ~13 GB of backward intermediates for
            # a 256-chunk, which OOMs a 16 GB T4 on top of the 282 M-param
            # backbone + Adam states. 64-chunks cost ~3 GB and still see
            # every pair.
            for s in range(math.ceil(len(ids) / 64)):
                b = ids[s * 64:(s + 1) * 64]
                prompts = [router[i][0] for i in b]
                labels = torch.tensor([FAM_ID[router[i][1]] for i in b],
                                      device=device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pv = model.prompt_enc(prompts)
                    zero_img = torch.zeros(len(b), model.backbone.out_channels,
                                           device=device)
                    logits = model.router_head(zero_img, pv)
                    loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                step_ctr[0] += 1; sched.step()
                ep_loss += loss.item(); n_steps += 1

        log("  epoch %d/%d  mean_loss %.4f  (%d steps, %.0fs)" % (
            ep + 1, total_epochs, ep_loss / max(1, n_steps), n_steps,
            time.time() - t0))
        # confusion mining for the next epoch's focus
        if focus and ep < total_epochs - 1:
            focus_idx, _ = _focus_and_acc(model, corpus, tile_tr, device, ep)
        # checkpoint after every epoch: an interrupt or session timeout then
        # only costs the in-progress epoch, never the whole run.
        _checkpoint(model, corpus, models_dir, ep + 1, verbose=verbose,
                    opt=opt, scaler=scaler, step_ctr=step_ctr)

    metrics = eval_brain(model, corpus, device=device,
                         tile_va=tile_va, point_va_single=point_va_single,
                         drag_va=drag_va, bbox_va=bbox_va, pat_va=pat_va,
                         count_va=count_va, text_va=text_va,
                         router_va=router_va, verbose=verbose)
    log("== STRESS eval: fresh rounds, EVERY image degraded + cluttered "
        "(the honest numbers) ==")
    sm = stress_test(model, device=device, verbose=verbose)
    for k, v in sm.items():
        metrics["stress_" + k] = v
    log("== ROUND-SOLVE: end-to-end, confidence-gated, EVERY image degraded ==")
    rs = _round_solve_metrics(model, device=device, verbose=verbose)
    metrics["round_solve"] = rs["round_solve"]
    metrics["round_solve_answered"] = rs["round_solve_answered"]
    for k, v in rs.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                metrics["rs_%s_%s" % (k, k2)] = v2
    _save_brain(model, metrics, corpus, models_dir)
    log("== Brain v2 saved: models/brain.pt (%.1f MB) ==" % (
        os.path.getsize(os.path.join(models_dir, "brain.pt")) / 1e6))
    return model, metrics


def stress_test(model, device="cpu", n_rounds=12, n_grids=8, seed=999,
                verbose=True):
    """The HONEST accuracy number.

    Fresh held-out rounds (a seed namespace disjoint from any training run)
    with EVERY image HARD-degraded (motion/gaussian blur, noise, double
    JPEG, dark-mode tint, ...) AND cluttered (extra distractor objects in
    point/count scenes). The clean-synthetic eval flatters because test
    matches training perfectly (the repo docs say so themselves); this is
    the number to trust. After training with hard_frac>0 these numbers
    climb toward the clean ones.
    """
    corpus = build_brain_corpus(
        per_class=20, n_point=n_rounds, n_count=max(4, n_rounds // 2),
        n_drag=n_rounds, n_grid=n_grids, n_pattern=max(4, n_rounds // 3),
        n_bbox=n_rounds, n_pipe=max(4, n_rounds // 2),
        n_tower=max(4, n_rounds // 2), n_shape=max(4, n_rounds // 2),
        n_text=n_rounds, hard_frac=1.0, degrade_frac=0.0,
        clutter_frac=1.0, router_bank=False,
        seed=seed, verbose=verbose)
    va_tile = list(range(len(corpus["tile_y"])))
    va_point = [i for i in range(len(corpus["point_m"]))
                if corpus["point_m"][i].get("type") != "count"]
    m = eval_brain(model, corpus, device=device, tile_va=va_tile,
                  point_va_single=va_point,
                  drag_va=list(range(len(corpus["drag_m"]))),
                  bbox_va=list(range(len(corpus["bbox_m"]))),
                  pat_va=list(range(len(corpus["pat_m"]))),
                  count_va=list(range(len(corpus["count_m"]))),
                  text_va=list(range(len(corpus["text_m"]))),
                  router_va=list(range(len(corpus["router"]))),
                  verbose=verbose)
    rs = _round_solve_metrics(model, device=device, corpus=corpus,
                              verbose=verbose)
    m["stress_round_solve"] = rs["round_solve"]
    m["stress_round_solve_answered"] = rs["round_solve_answered"]
    return m

def _round_solve_metrics(model, device="cpu", corpus=None, solver=None,
                         verbose=True, seed=1234, n=24):
    """THE headline number: the ROUND SOLVE RATE.

    For every offline-capable family, a fresh held-out round (every image
    HARD-degraded + cluttered, seed namespace disjoint from training) is
    routed and answered END-TO-END by BrainSolver — the same confidence
    gates the live solver uses. A round counts as solved only when the
    answer is exactly right (indices, point within 10%, exact count, both
    drag ends within 10%, right pattern candidate, box within tolerance,
    exact code). Deferred (self-gated) rounds count as unsolved offline —
    in production they hand over to the vision model, so the live rate is
    this number plus whatever the vision model adds.

    Composite = real-world family mix (FAMILY_WEIGHTS).
    """
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    model.eval()
    if corpus is None:
        corpus = build_brain_corpus(
            per_class=0, n_point=n, n_count=n, n_drag=n, n_grid=n,
            n_pattern=max(6, n // 2), n_bbox=n, n_pipe=max(6, n // 2),
            n_tower=max(6, n // 2), n_shape=max(6, n // 2), n_text=n,
            hard_frac=1.0, degrade_frac=0.0, clutter_frac=1.0,
            router_bank=False, seed=seed, verbose=False)
    if solver is None:
        solver = BrainSolver(model=model, device=device)
    per = collections.OrderedDict()

    def rec(fam, total, answered, exact):
        per[fam] = {"total": total, "answered": answered, "exact": exact,
                    "solve_rate": exact / max(1, total),
                    "answer_rate": answered / max(1, total)}

    # ── binary grids ───────────────────────────────────────────────────────
    tot = ans = ok = 0
    for i, m in enumerate(corpus["grid_m"]):
        im = corpus["grid_imgs"][i]
        tiles = []
        for (x0, y0, s, _g) in m["tile_boxes"]:
            tiles.append(im.crop((x0, y0, x0 + s, y0 + s)))
        res = solver.solve(image=im, prompt=m["prompt"], tiles=tiles)
        tot += 1
        if res is None:
            continue
        ans += 1
        if sorted(res["answer"]) == sorted(m["correct"]):
            ok += 1
    rec(BINARY, tot, ans, ok)

    # ── point ──────────────────────────────────────────────────────────────
    tot = ans = ok = 0
    for i, m in enumerate(corpus["point_m"]):
        from PIL import Image as _Im
        im = _Im.fromarray(corpus["point_x"][i].permute(1, 2, 0).cpu().numpy())
        res = solver.solve(image=im, prompt=m["prompt"])
        tot += 1
        if res is None:
            continue
        ans += 1
        ax, ay = res["answer"]
        if math.hypot(ax - m["x"], ay - m["y"]) <= 0.10:
            ok += 1
    rec(AREA_POINT, tot, ans, ok)

    # ── count ─────────────────────────────────────────────────────────────
    tot = ans = ok = 0
    for i, m in enumerate(corpus["count_m"]):
        from PIL import Image as _Im
        im = _Im.fromarray(corpus["count_x"][i].permute(1, 2, 0).cpu().numpy())
        res = solver.solve(image=im, prompt=m["prompt"])
        tot += 1
        if res is None:
            continue
        ans += 1
        if res["answer"] == m["count"]:
            ok += 1
    rec(COUNT, tot, ans, ok)

    # ── drag / pipe / shape / tower ────────────────────────────────────────
    tot = ans = ok = 0
    for i, m in enumerate(corpus["drag_m"]):
        from PIL import Image as _Im
        im = _Im.fromarray(corpus["drag_x"][i].permute(1, 2, 0).cpu().numpy())
        res = solver.solve(image=im, prompt=m["prompt"])
        tot += 1
        if res is None:
            continue
        ans += 1
        a = res["answer"]
        d1 = math.hypot(a["from"][0] - m["fx"], a["from"][1] - m["fy"])
        d2 = math.hypot(a["to"][0] - m["tx"], a["to"][1] - m["ty"])
        if d1 <= 0.10 and d2 <= 0.10:
            ok += 1
    fam_drag = DRAG_DROP
    rec(fam_drag, tot, ans, ok)

    # ── pattern ────────────────────────────────────────────────────────────
    tot = ans = ok = 0
    for i, m in enumerate(corpus["pat_m"]):
        res = solver.solve(image=corpus["pat_imgs"][i], prompt=m["prompt"],
                           cell_boxes=m["cell_boxes"],
                           cand_boxes=m["candidate_boxes"])
        tot += 1
        if res is None:
            continue
        ans += 1
        if res["answer"] == m["correct"]:
            ok += 1
    rec(PATTERN, tot, ans, ok)

    # ── bbox ───────────────────────────────────────────────────────────────
    tot = ans = ok = 0
    for i, m in enumerate(corpus["bbox_m"]):
        from PIL import Image as _Im
        im = _Im.fromarray(corpus["bbox_x"][i].permute(1, 2, 0).cpu().numpy())
        prompt = "Draw a box around the %s" % m["target"].replace("_", " ")
        res = solver.solve(image=im, prompt=prompt)
        tot += 1
        if res is None:
            continue
        ans += 1
        a = res["answer"]
        d = math.hypot(a["x"] + a["w"] / 2 - m["cx"],
                       a["y"] + a["h"] / 2 - m["cy"])
        sd = max(abs(a["w"] - m["w"]), abs(a["h"] - m["h"]))
        if d <= 0.10 and sd <= 0.25:
            ok += 1
    rec(AREA_BBOX, tot, ans, ok)

    # ── text ───────────────────────────────────────────────────────────────
    tot = ans = ok = 0
    for i, m in enumerate(corpus["text_m"]):
        from PIL import Image as _Im
        im = _Im.fromarray(corpus["text_x"][i].permute(1, 2, 0).cpu().numpy())
        res = solver.solve(image=im, prompt=m["prompt"])
        tot += 1
        if res is None:
            continue
        ans += 1
        if res["answer"] == m["text"]:
            ok += 1
    rec(TEXT_ENTRY, tot, ans, ok)

    wsum = sum(FAMILY_WEIGHTS.get(f, 0.0) for f in per)
    comp = sum(FAMILY_WEIGHTS.get(f, 0.0) * per[f]["solve_rate"]
               for f in per) / max(1e-9, wsum)
    comp_ans = sum(FAMILY_WEIGHTS.get(f, 0.0) * per[f]["answer_rate"]
                   for f in per) / max(1e-9, wsum)
    out = dict(per)
    out["round_solve"] = comp
    out["round_solve_answered"] = comp_ans
    if verbose:
        log("  family                    total  answered  exact  solve%")
        for f, r in per.items():
            log("  %-23s %6d  %8d  %6d  %5.1f%%" % (
                f, r["total"], r["answered"], r["exact"],
                100 * r["solve_rate"]))
        log("  ROUND SOLVE (offline, gated, degraded): %.1f%%  "
            "(answered %.1f%%)" % (100 * comp, 100 * comp_ans))
    return out


def eval_brain(model, corpus, device="cpu", tile_va=None, point_va_single=None,
               drag_va=None, bbox_va=None, pat_va=None, count_va=None,
               text_va=None, router_va=None, verbose=True):
    """Inference-only wrapper: no autograd graph → saves several GB of VRAM
    (and is faster) versus the graph-building forwards of the impl below."""
    _prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        return _eval_brain_impl(model, corpus, device, tile_va, point_va_single,
                                drag_va, bbox_va, pat_va, count_va, text_va,
                                router_va, verbose)
    finally:
        torch.set_grad_enabled(_prev)

def _eval_brain_impl(model, corpus, device, tile_va=None,
                     point_va_single=None, drag_va=None, bbox_va=None,
                     pat_va=None, count_va=None, text_va=None, router_va=None,
                     verbose=True):
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    model.eval()
    tile_size = corpus["tile_size"]
    metrics = {}

    # tile accuracy
    if tile_va:
        ok = 0
        for s in range(0, len(tile_va), 256):
            b = tile_va[s:s + 256]
            x = _to_float(_move(corpus["tile_x"][b], device))
            pred = model.tile_logits(model.features(x)).argmax(1)
            ok += int((pred == _move(corpus["tile_y"][b], device)).sum())
        metrics["tile_accuracy"] = ok / len(tile_va)
        log("  tile acc:       %.3f (%d)" % (metrics["tile_accuracy"], len(tile_va)))

    # point hit@10% (single-instance)
    if point_va_single:
        errs = []
        tc = torch.tensor([corpus["point_m"][i]["target_id"] for i in point_va_single])
        ty = torch.tensor([[corpus["point_m"][i]["x"],
                            corpus["point_m"][i]["y"]] for i in point_va_single])
        for s in range(0, len(point_va_single), 256):
            b = range(s, min(s + 256, len(point_va_single)))
            ids = [point_va_single[i] for i in b]
            x = _to_float(_move(corpus["point_x"][ids], device))
            f8, f4 = model.features2(x)
            hm = model.heatmaps(f8, f4)
            sel = hm.gather(1, tc[list(b)].to(device).view(-1, 1, 1, 1).expand(
                -1, 1, *hm.shape[2:])).squeeze(1)
            pred = soft_argmax2d(sel)
            errs.extend(torch.linalg.norm(pred - ty[list(b)].to(device), dim=1).tolist())
        errs = sorted(errs)
        metrics["point_hit_at_10"] = sum(e <= 0.10 for e in errs) / len(errs)
        metrics["point_median_err"] = errs[len(errs) // 2]
        log("  point hit@10%%:  %.3f  med-err %.3f" % (
            metrics["point_hit_at_10"], metrics["point_median_err"]))

    # drag both-endpoints hit@10%
    if drag_va:
        ef, et = [], []
        tf = torch.tensor([[corpus["drag_m"][i]["fx"], corpus["drag_m"][i]["fy"]] for i in drag_va])
        tt = torch.tensor([[corpus["drag_m"][i]["tx"], corpus["drag_m"][i]["ty"]] for i in drag_va])
        for s in range(0, len(drag_va), 256):
            b = drag_va[s:s + 256]
            pos = [drag_va.index(i) for i in b]
            x = _to_float(_move(corpus["drag_x"][b], device))
            f8, f4 = model.features2(x)
            hms = model.drag_maps(f8, f4)
            ef.extend(torch.linalg.norm(soft_argmax2d(hms[:, 0]) - tf[pos].to(device), dim=1).tolist())
            et.extend(torch.linalg.norm(soft_argmax2d(hms[:, 1]) - tt[pos].to(device), dim=1).tolist())
        both = sum(a <= 0.10 and b <= 0.10 for a, b in zip(ef, et)) / len(ef)
        metrics["drag_hit_both"] = both
        log("  drag both@10%%:  %.3f" % both)

    # bbox (centre within 10% AND size within 25%)
    if bbox_va:
        good = 0
        for i in bbox_va:
            m = corpus["bbox_m"][i]
            x = _to_float(_move(corpus["bbox_x"][i:i + 1], device))
            f8, f4 = model.features2(x)
            cm, wh = model.bbox(f8, f4)
            pred = soft_argmax2d(cm)[0]
            pwh = wh[0]
            d = math.hypot(pred[0].item() - m["cx"], pred[1].item() - m["cy"])
            sd = max(abs(pwh[0].item() - m["w"]), abs(pwh[1].item() - m["h"]))
            good += int(d <= 0.10 and sd <= 0.25)
        metrics["bbox_acc"] = good / len(bbox_va)
        log("  bbox acc:       %.3f" % metrics["bbox_acc"])

    # count exact accuracy (self-gated peak counter, same as inference)
    if count_va:
        answered = exact = 0
        for i in count_va:
            m = corpus["count_m"][i]
            x = _to_float(_move(corpus["count_x"][i:i + 1], device))
            f8, f4 = model.features2(x)
            hm = model.heatmaps(f8, f4)                  # (1, C+1, H, W)
            presence = F.softmax(hm, dim=1)[:, :-1]      # drop background
            n = _count_peaks(presence[0, m["target_id"]])
            if n is None:
                continue                                 # self-gated
            answered += 1
            exact += int(n == m["count"])
        metrics["count_exact"] = exact / max(1, answered)
        metrics["count_answer_rate"] = answered / len(count_va)
        log("  count exact:   %.3f (answers %.0f%% of rounds)" % (
            metrics["count_exact"], 100 * metrics["count_answer_rate"]))

    # text codes: exact 5-char match + per-character accuracy
    if text_va:
        exact = chars_ok = chars_n = 0
        for i in text_va:
            m = corpus["text_m"][i]
            x = _to_float(_move(corpus["text_x"][i:i + 1], device))
            logits = model.text_logits(model.features(x)).view(
                1, model.text_head.text_len, model.text_head.n_chars)
            pred = logits.argmax(dim=2)[0].tolist()
            got = "".join(TEXT_ALPHABET[p] for p in pred)
            exact += int(got == m["text"])
            chars_ok += sum(a == b for a, b in zip(got, m["text"]))
            chars_n += len(m["text"])
        metrics["text_exact"] = exact / len(text_va)
        metrics["text_char_acc"] = chars_ok / max(1, chars_n)
        log("  text exact:    %.3f  per-char %.3f" % (
            metrics["text_exact"], metrics["text_char_acc"]))

    # pattern candidate accuracy (the learned reasoner)
    if pat_va:
        ok = 0
        for i in pat_va:
            m = corpus["pat_m"][i]
            im = corpus["pat_imgs"][i]
            W, Hh = im.size
            cells = torch.stack([_img_to_u8(im.crop(
                (b["x"] * W, b["y"] * Hh, (b["x"] + b["w"]) * W,
                 (b["y"] + b["h"]) * Hh)), tile_size) for b in m["cell_boxes"]])
            cands = torch.stack([_img_to_u8(im.crop(
                (b["x"] * W, b["y"] * Hh, (b["x"] + b["w"]) * W,
                 (b["y"] + b["h"]) * Hh)), tile_size) for b in m["candidate_boxes"]])
            cells = _to_float(_move(cells, device))
            cands = _to_float(_move(cands, device))
            cf = F.adaptive_avg_pool2d(model.features(cells), 1).flatten(1).unsqueeze(0)
            xf = F.adaptive_avg_pool2d(model.features(cands), 1).flatten(1).unsqueeze(0)
            pv = model.prompt_enc([m.get("prompt", "")])
            pred = model.pattern(cf, xf, pv).argmax(1).item()
            ok += int(pred == m["correct"])
        metrics["pattern_acc"] = ok / len(pat_va)
        log("  pattern cand:   %.3f" % metrics["pattern_acc"])

    # router accuracy (prompt -> family)
    if router_va:
        ok = 0
        for s in range(0, len(router_va), 128):
            b = router_va[s:s + 128]
            prompts = [corpus["router"][i][0] for i in b]
            labels = torch.tensor([FAM_ID[corpus["router"][i][1]] for i in b],
                                  device=device)
            pv = model.prompt_enc(prompts)
            zero_img = torch.zeros(len(b), model.backbone.out_channels, device=device)
            pred = model.router_head(zero_img, pv).argmax(1)
            ok += int((pred == labels).sum())
        metrics["router_acc"] = ok / len(router_va)
        log("  router acc:     %.3f" % metrics["router_acc"])
    return metrics

# ═══════════════════════════════════════════════════════════════════════════
#  Inference: BrainSolver — unified, drop-in for every family
# ═══════════════════════════════════════════════════════════════════════════

def _to_pil(image):
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return image.convert("RGB")


def _adapt_v1_state(state):
    """v1 checkpoint state_dict -> v2 (dual-resolution head) layout.

    The shipped v1 split (brain_part_00..06, 60 classes) has single-
    resolution heads: heatmap_head.head / drag_head.head / bbox_head.
    center. The v2 model expects .lo/.hi pairs, so the v1 convs are copied
    to .lo and the NEW S/4 (.hi) branch is zeroed by the caller — at load
    time the fused output then equals the v1 output exactly. Returns
    (state, adapted)."""
    if "heatmap_head.lo.weight" in state:
        return state, False          # already v2
    rename = {
        "heatmap_head.head.weight": "heatmap_head.lo.weight",
        "heatmap_head.head.bias": "heatmap_head.lo.bias",
        "drag_head.head.weight": "drag_head.lo.weight",
        "drag_head.head.bias": "drag_head.lo.bias",
        "bbox_head.center.weight": "bbox_head.center_lo.weight",
        "bbox_head.center.bias": "bbox_head.center_lo.bias",
    }
    return ({rename.get(k, k): v for k, v in state.items()}, True)


class BrainSolver:
    """The Brain at inference time.

    Loads models/brain.pt (+ .json). Exposes the method names the shipped
    TileClassifier / PointLocator / DragLocator use, so server.py can swap
    them in, PLUS a single ``solve(...)`` that routes any round and returns
    the answer. Every answer is confidence-gated: below threshold the method
    returns None so the caller falls back to the vision model (same safety
    the production offline path uses).

    v2 speed: inference-mode, fp16 on GPU, an LRU prompt-vector cache (hCapa
    repeats the same prompt wording across rounds/challenges — the language
    brain then runs once per unique prompt instead of once per tile), and a
    single fused dual-resolution forward for every localisation head.
    """

    def __init__(self, models_dir=MODELS_DIR, device=None,
                 min_conf=float(os.environ.get("SOLVER_CNN_MIN_CONF", "0.62")),
                 model=None, half=None):
        self.available = False
        self.text_ok = True     # False when an old-format brain has no text head
        self.model = None
        self.classes = list(CLASSES)
        self.families = list(FAMILIES)
        self.size = DEFAULT_SCENE_SIZE
        self.tile_size = DEFAULT_TILE_SIZE
        self.width = 48
        self.min_conf = min_conf
        # Robustness policy for deployed inference.  Flip TTA is checkpoint
        # compatible and helps crops with off-centre/occluded subjects.
        self.tta = os.environ.get("SOLVER_TTA", "1").lower() not in (
            "0", "false", "no", "off")
        self.target_min = float(os.environ.get("SOLVER_TARGET_MIN", "0.30"))
        if device is None:
            self.device = ("cuda" if (_TORCH and torch.cuda.is_available())
                           else "cpu")
        else:
            self.device = device
        self._pv_cache = collections.OrderedDict()
        self.dtype = torch.float32
        if not _TORCH:
            return
        arch = {}
        if model is not None:
            # test/eval path: an already-built Brain (same shapes as the
            # checkpoint would load)
            self.model = model
            meta = {}
        else:
            pt = os.path.join(models_dir, "brain.pt")
            js = os.path.join(models_dir, "brain.json")
            if not (os.path.exists(pt) and os.path.exists(js)):
                return
            try:
                with open(js, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.classes = meta.get("classes", CLASSES)
                self.families = meta.get("families", FAMILIES)
                self.size = int(meta.get("scene_size", DEFAULT_SCENE_SIZE))
                self.tile_size = int(meta.get("tile_size", DEFAULT_TILE_SIZE))
                arch = meta.get("arch", {})
                self.width = int(arch.get("width", 48))
                prompt_dim = int(arch.get("prompt_dim", 512))
                prompt_layers = int(arch.get("prompt_layers", 6))
                d_concept = int(arch.get("d_concept", 256))
                pattern_d = int(arch.get("pattern_d", 256))
                pattern_layers = int(arch.get("pattern_layers", 3))
                text_len = int(arch.get("text_len", 5))
                n = len(self.classes)
                # Rebuild with the EXACT architecture recorded at train time,
                # so the state_dict always fits — no shape mismatches at load.
                self.model = Brain(n, width=self.width, prompt_dim=prompt_dim,
                                   prompt_layers=prompt_layers,
                                   d_concept=d_concept, pattern_d=pattern_d,
                                   pattern_layers=pattern_layers,
                                   text_len=text_len,
                                   n_families=len(self.families))
                state = torch.load(pt, map_location=self.device)
                state, is_v1 = _adapt_v1_state(state)
                if is_v1:
                    # v1 -> v2: copy the single-resolution heads into .lo,
                    # drop keys whose shape changed (v1 router hidden dim),
                    # zero the new S/4 branch so the fused output equals
                    # the v1 output exactly at load time.
                    want = self.model.state_dict()
                    filt = {k: v for k, v in state.items()
                            if k in want and want[k].shape == v.shape}
                    self.model.load_state_dict(filt, strict=False)
                    with torch.no_grad():
                        for _pth in ("heatmap_head.hi", "drag_head.hi",
                                     "bbox_head.center_hi"):
                            _sub, _name = _pth.split(".")
                            _conv = getattr(getattr(self.model, _sub), _name)
                            _conv.weight.zero_()
                            _conv.bias.zero_()
                    # v1 pos-embedding is shorter (96 vs 160 tokens) — copy
                    # the shared prefix instead of dropping it.
                    _pw = state.get("prompt_enc.pos.weight")
                    if _pw is not None and _pw.dim() == 2 and \
                            _pw.shape[1] == want["prompt_enc.pos.weight"].shape[1] \
                            and _pw.shape[0] < want["prompt_enc.pos.weight"].shape[0]:
                        self.model.prompt_enc.pos.weight.data[
                            :_pw.shape[0]].copy_(_pw)
                    # the committed v1 split predates the position-aware
                    # text head — its fc shape does not fit
                    if "text_head.fc.weight" not in filt:
                        self.text_ok = False
                    print("    [brain] loaded a v1 brain (single-resolution "
                          "heads adapted to the dual-resolution layout)")
                else:
                    try:
                        self.model.load_state_dict(state)
                    except Exception:
                        # older v2 format: may lack the text head
                        filt = {k: v for k, v in state.items()
                                if not k.startswith("text_head.")}
                        missing = (set(self.model.state_dict().keys())
                                   - set(filt.keys()))
                        self.model.load_state_dict(filt, strict=False)
                        if missing and not all(k.startswith("text_head.")
                                               for k in missing):
                            raise RuntimeError(
                                "unexpected missing keys: %s" % missing)
                        self.text_ok = False
                self.model.to(self.device).eval()
                self.available = True
            except Exception:  # pragma: no cover
                self.model = None
                self.available = False
                return
        if model is not None:
            self.available = True
            self.size = getattr(model, "size", self.size) or self.size
        # fp16 on GPU (big speed + memory win; CPU stays fp32)
        want_half = half if half is not None else self.device.startswith("cuda")
        if want_half and self.device.startswith("cuda") and self.model is not None:
            self.model.half()
            self.dtype = torch.float16

    # ── prompt cache: hCaptcha repeats prompt wording across rounds ───────
    def prompt_vec(self, prompts):
        """PromptEncoder with an LRU cache keyed on the exact prompt string —
        the same wording answered twice runs the 12-layer transformer once."""
        if isinstance(prompts, str):
            prompts = [prompts]
        miss = []
        for p in prompts:
            if p not in self._pv_cache:
                miss.append(p)
        if miss:
            uniq = list(dict.fromkeys(miss))
            with torch.no_grad():
                v = self.model.prompt_enc(uniq)
            for p, t in zip(uniq, v):
                self._pv_cache[p] = t
                self._pv_cache.move_to_end(p)
            while len(self._pv_cache) > 4096:
                self._pv_cache.popitem(last=False)
        return torch.stack([self._pv_cache[p] for p in prompts])

    # ── low-level image prep ───────────────────────────────────────────────
    def _prep_tile(self, im, size=None):
        size = size or self.tile_size
        im = _to_pil(im)
        if im.size != (size, size):
            im = im.resize((size, size), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
        x = (x.permute(2, 0, 1).unsqueeze(0) - 0.5) / 0.5
        return x.to(self.device, dtype=self.dtype)

    def _feat2(self, im, size=None):
        with torch.no_grad():
            return self.model.features2(self._prep_tile(im, size))

    # ── TileClassifier drop-in ─────────────────────────────────────────────
    @_no_grad
    def probabilities(self, images):
        """List of {label: prob} dicts, one per image."""
        if not self.available:
            return []
        ims = images if isinstance(images, (list, tuple)) else [images]
        xs = torch.cat([self._prep_tile(im) for im in ims], dim=0)
        logits = self.model.tile_logits(self.model.features(xs))
        if self.tta:
            # Average logits, not probabilities, to avoid making uncertain
            # predictions artificially overconfident.
            fx = torch.flip(xs, dims=[3])
            flogits = self.model.tile_logits(self.model.features(fx))
            logits = 0.70 * logits + 0.30 * flogits
        probs = F.softmax(logits.float(), dim=1)
        return [{self.classes[i]: float(p) for i, p in enumerate(row)}
                for row in probs.tolist()]

    def classify_many(self, images, with_conf=True):
        out = []
        for probs in self.probabilities(images):
            if not probs:
                break
            lab = max(probs, key=probs.get)
            out.append((lab, probs[lab]) if with_conf else lab)
        return out

    # ── PointLocator drop-in ──────────────────────────────────────────────
    @_no_grad
    def _scores(self, image):
        """(presence_map (C,H,W), location (C,2)) in one forward pass."""
        if not self.available:
            return None, None
        x = self._prep_tile(image, self.size)
        f8, f4 = self.model.features2(x)
        hm = self.model.heatmaps(f8, f4)
        if self.tta:
            fx = torch.flip(x, dims=[3])
            ff8, ff4 = self.model.features2(fx)
            # Map the flipped heatmap back before fusing coordinates.
            fhm = torch.flip(self.model.heatmaps(ff8, ff4), dims=[3])
            hm = 0.70 * hm + 0.30 * fhm
        hm = hm.squeeze(0)
        presence = F.softmax(hm, dim=0)
        if hm.shape[0] > len(self.classes):
            presence = presence[:-1]
            hm = hm[:-1]
        loc = soft_argmax2d(hm)
        return presence, loc

    def scan(self, image):
        p, loc = self._scores(image)
        if p is None:
            return []
        C, H, W = p.shape
        presence = p.reshape(C, -1).max(dim=1).values
        out = [{"label": self.classes[c] if c < len(self.classes) else str(c),
                "presence": float(presence[c]),
                "x": float(loc[c][0]), "y": float(loc[c][1])}
               for c in range(C)]
        out.sort(key=lambda r: -r["presence"])
        return out

    def locate(self, image, target):
        name = hct.canonical(target) or target
        for row in self.scan(image):
            if row["label"] == name:
                return (row["x"], row["y"], row["presence"])
        return None

    def count(self, image, target, min_peak=0.08, min_sep=0.16,
              weak_gate=0.20, max_n=9, margin=0.04):
        """Same self-gating peak counter as PointLocator.count — a count
        answer is graded EXACTLY, so border/fragmented/weak maps return
        None."""
        if not self.available:
            return None
        name = hct.canonical(target) or target
        if name not in self.classes:
            return None
        cid = self.classes.index(name)
        presence, _ = self._scores(image)
        if presence is None:
            return None
        chan = presence[cid]
        n = _count_peaks(chan, min_peak=min_peak, min_sep=min_sep,
                         weak_gate=weak_gate, max_n=max_n, margin=margin)
        return n

    def locate_relational(self, image, prompt, verifier=None):
        """Superlative prompts via scan + the shared superlative table."""
        sup = hct.superlative_table(prompt)
        if sup is None:
            tgt = hct.extract_target(prompt)
            hit = self.locate(image, tgt) if tgt else None
            return (hit[0], hit[1], tgt) if hit else None
        table, direction = sup
        rows = self.scan(image)
        pool = [r for r in rows if r["presence"] >= 0.30 and r["label"] in table]
        uniq = {}
        for r in pool:
            have = uniq.get(r["label"])
            if have is None or r["presence"] > have["presence"]:
                uniq[r["label"]] = r
        cands = list(uniq.values())
        if not cands:
            return None
        key = lambda r: table[r["label"]]
        best = max(cands, key=key) if direction == "max" else min(cands, key=key)
        return (best["x"], best["y"], best["label"])

    # ── DragLocator drop-in ───────────────────────────────────────────────
    @_no_grad
    def locate_drag(self, image):
        if not self.available:
            return None
        f8, f4 = self._feat2(image, self.size)
        hm = self.model.drag_maps(f8, f4)                # (1, 2, H, W)
        pf = soft_argmax2d(hm[:, 0])[0]
        pt = soft_argmax2d(hm[:, 1])[0]
        return {"from": (float(pf[0]), float(pf[1])),
                "to": (float(pt[0]), float(pt[1]))}

    # ── BBox ───────────────────────────────────────────────────────────────
    @_no_grad
    def bbox(self, image):
        if not self.available:
            return None
        f8, f4 = self._feat2(image, self.size)
        cm, wh = self.model.bbox(f8, f4)
        c = soft_argmax2d(cm)[0]
        w, h = float(wh[0, 0]), float(wh[0, 1])
        cx, cy = float(c[0]), float(c[1])
        return {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h}

    # ── Text codes ("Type the text you see") ──────────────────────────────
    @_no_grad
    def read_text(self, image):
        """Read a text-entry code image -> the decoded string."""
        if not self.available or not self.text_ok:
            return None
        f8, _f4 = self._feat2(image, self.size)
        logits = self.model.text_logits(f8).view(
            1, self.model.text_head.text_len, self.model.text_head.n_chars)
        pred = logits.argmax(dim=2)[0].tolist()
        return "".join(TEXT_ALPHABET[p] for p in pred)

    # ── Pattern reasoner ──────────────────────────────────────────────────
    @_no_grad
    def solve_pattern(self, image, cell_boxes, cand_boxes, prompt=""):
        if not self.available:
            return None
        im = _to_pil(image)
        W, H = im.size
        cells = torch.stack([self._prep_tile(im.crop(
            (b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W,
             (b["y"] + b["h"]) * H))) for b in cell_boxes]).squeeze(1)
        cands = torch.stack([self._prep_tile(im.crop(
            (b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W,
             (b["y"] + b["h"]) * H))) for b in cand_boxes]).squeeze(1)
        cf = F.adaptive_avg_pool2d(self.model.features(cells), 1).flatten(1).unsqueeze(0)
        xf = F.adaptive_avg_pool2d(self.model.features(cands), 1).flatten(1).unsqueeze(0)
        pv = self.prompt_vec([prompt])
        logits = self.model.pattern(cf, xf, pv)[0]
        prob = F.softmax(logits.float(), dim=0)
        idx = int(prob.argmax())
        return {"candidate": idx, "confidence": float(prob[idx]),
                "box": cand_boxes[idx]}

    # ── Learned router (prompt -> family) ─────────────────────────────────
    @_no_grad
    def router_predict(self, prompt, image=None):
        if not self.available:
            return None
        pv = self.prompt_vec([prompt])
        if image is not None:
            f8, _f4 = self._feat2(image, self.size)
            pool = F.adaptive_avg_pool2d(f8, 1).flatten(1)
        else:
            pool = torch.zeros(1, self.model.backbone.out_channels,
                               device=self.device)
        logits = self.model.router_head(pool, pv)[0]
        prob = F.softmax(logits.float(), dim=0)
        idx = int(prob.argmax())
        return {"family": self.families[idx], "confidence": float(prob[idx])}

    # ── THE unified entry point ───────────────────────────────────────────
    def solve(self, image, prompt="", tiles=None, tile_boxes=None,
              cell_boxes=None, cand_boxes=None, example=None,
              dom=None, payload=None, use_learned_router=False):
        """Route ONE round and return its answer for ANY family.

        Returns a dict {family, answer, confidence} where ``answer`` is
        shaped per family (indices / (x,y) / bbox / drag / count /
        candidate), or None when the Brain is below confidence — so the
        caller falls back to the vision model, exactly like the shipped
        offline path.

        Family routing prefers the production-proven rule router
        (hcaptcha_types.classify); pass use_learned_router=True to use the
        Brain's own learned router head instead.
        """
        if not self.available:
            return None
        if use_learned_router:
            r = self.router_predict(prompt, image)
            family = r["family"] if r else None
        else:
            family = hct.classify(payload=payload, dom=dom, prompt=prompt)

        # DRAG_DROP sub-families: pattern / tower split off by prompt wording
        if family == hct.DRAG_DROP:
            if hct.is_pattern_prompt(prompt) and cell_boxes and cand_boxes:
                family = PATTERN
            elif hct.is_tower_prompt(prompt):
                family = TOWER

        try:
            if family == hct.BINARY and tiles:
                return self._solve_binary(tiles, prompt, example)
            if family == hct.AREA_POINT:
                return self._solve_point(image, prompt)
            if family == hct.AREA_BBOX:
                bb = self.bbox(image)
                return ({"family": AREA_BBOX, "answer": bb, "confidence": 0.7}
                        if bb else None)
            if family == hct.COUNT:
                return self._solve_count(image, prompt)
            if family == hct.TEXT_ENTRY:
                code = self.read_text(image)
                return ({"family": TEXT_ENTRY, "answer": code,
                         "confidence": 0.9} if code else None)  # None if no text head
            if family == PATTERN:
                return self._solve_pattern(image, cell_boxes, cand_boxes, prompt)
            if family in (hct.DRAG_DROP, TOWER):
                d = self.locate_drag(image)
                return ({"family": family, "answer": d, "confidence": 0.7}
                        if d else None)
        except Exception:
            return None
        return None

    def _solve_binary(self, tiles, prompt, example=None):
        probs = self.probabilities(tiles)
        if not probs:
            return None
        labels = [max(p, key=p.get) for p in probs]
        confs = [p[l] for p, l in zip(probs, labels)]
        mean_conf = sum(confs) / len(confs)
        idx = hct.resolve_semantic(prompt, labels, example)
        if idx is None:                       # not understood -> defer
            return None
        # For a direct select-all noun, argmax is unnecessarily brittle: the
        # requested object can be the runner-up in a cluttered tile.  Score
        # the requested class directly while retaining the conservative
        # semantic resolver for relations, materials, and reference prompts.
        target = hct.extract_target(prompt)
        direct = hct.canonical(target) if target else None
        if direct in self.classes and not example:
            positives = []
            for i, p in enumerate(probs):
                pt = p.get(direct, 0.0)
                alt = max((v for k, v in p.items() if k != direct),
                          default=0.0)
                if pt >= self.target_min and pt + 0.05 >= alt:
                    positives.append(i)
            if positives:
                score = sum(probs[i].get(direct, 0.0)
                            for i in positives) / len(positives)
                return {"family": BINARY, "answer": positives,
                        "confidence": score, "labels": labels,
                        "target_scores": [probs[i].get(direct, 0.0)
                                          for i in positives]}
        # confidence gate: a low-confidence labelling is vision territory
        if mean_conf < self.min_conf:
            return None
        return {"family": BINARY, "answer": idx, "confidence": mean_conf,
                "labels": labels}

    def _solve_point(self, image, prompt):
        sup = hct.superlative_table(prompt)
        if sup is not None:
            r = self.locate_relational(image, prompt, verifier=self)
            if r:
                return {"family": AREA_POINT, "answer": (r[0], r[1]),
                        "confidence": 0.6, "label": r[2]}
            return None
        target = hct.extract_target(prompt)
        if not target:
            return None
        hit = self.locate(image, target)
        if hit and hit[2] >= 0.20:
            return {"family": AREA_POINT, "answer": (hit[0], hit[1]),
                    "confidence": hit[2], "label": target}
        return None

    def _solve_count(self, image, prompt):
        target = hct.extract_target(prompt)
        if not target:
            return None
        n = self.count(image, target)
        if n is None:                         # self-gated -> defer
            return None
        return {"family": COUNT, "answer": n, "confidence": 0.6}

    def _solve_pattern(self, image, cell_boxes, cand_boxes, prompt):
        r = self.solve_pattern(image, cell_boxes, cand_boxes, prompt)
        if not r or r["confidence"] < 0.40:
            return None
        return {"family": PATTERN, "answer": r["candidate"],
                "confidence": r["confidence"], "box": r["box"]}


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description="The Brain v2 — one unified "
                                 "model for every hCaptcha challenge family.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="build corpus in-memory + train all heads")
    t.add_argument("--preset", default="large",
                   choices=sorted(PRESETS),
                   help="architecture size (small/medium/large/mega)")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--phase2", type=int, default=3,
                   help="extra hardening epochs (lr x0.3, extra hard photometrics)")
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)
    t.add_argument("--per_class", type=int, default=3200)
    t.add_argument("--n_point", type=int, default=18000)
    t.add_argument("--n_count", type=int, default=12000)
    t.add_argument("--n_drag", type=int, default=14000)
    t.add_argument("--n_grid", type=int, default=9000)
    t.add_argument("--n_pattern", type=int, default=12000)
    t.add_argument("--n_bbox", type=int, default=10000)
    t.add_argument("--n_pipe", type=int, default=7000)
    t.add_argument("--n_tower", type=int, default=7000)
    t.add_argument("--n_shape", type=int, default=7000)
    t.add_argument("--n_odrag", type=int, default=9000,
                   help="object-drag rounds: 'Flytte' format - move a real "
                        "object (hCaptcha roster: bear, raccoon, red panda, "
                        "boar, ...) to a highlighted cell")
    t.add_argument("--n_text", type=int, default=6000)
    # degradation + hard-condition knobs
    t.add_argument("--hard_frac", type=float, default=0.35,
                   help="fraction of rounds with HARD degradation (blur/noise/dark-mode/double-JPEG)")
    t.add_argument("--degrade_frac", type=float, default=0.40,
                   help="fraction of rounds with SOFT degradation (the rest stay clean)")
    t.add_argument("--clutter_frac", type=float, default=0.5,
                   help="fraction of point/count rounds with extra distractor objects")
    t.add_argument("--hcap_dir", default=None,
                   help="real hCaptcha challenge-image dataset (folder per class, e.g. the GitHub 'hcap' datasets)")
    t.add_argument("--hcap_views", type=int, default=16,
                   help="random training views per real hcap tile")
    t.add_argument("--photos_dir", default=None,
                   help="real-photo corpus from fetch_photos.py (folder per "
                        "class) — ingested like the hcap tiles")
    t.add_argument("--photo_views", type=int, default=16,
                   help="random training views per real photo")
    t.add_argument("--real_only", action="store_true",
                   help="tile classes that have real photos (hcap/photos) "
                        "train EXCLUSIVELY on real images - no renders for "
                        "them; only photo-less classes fall back to renders")
    t.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="continue from a training checkpoint: a path to "
                        "resume.pt, or bare --resume to auto-find the newest "
                        "(models dir, then any /kaggle/input upload)")
    t.add_argument("--models_dir", default=None,
                   help="where brain.pt/resume.pt are written (default: the "
                        "repo models/ dir). On Kaggle use /kaggle/output/ckpt "
                        "so checkpoints survive the 12h session kill")
    t.add_argument("--no_router_bank", action="store_true",
                   help="use the small v1 prompt set instead of the ~30k bank")
    t.add_argument("--kreg", type=float, default=0.02,
                   help="ontology anchor weight (0 = off)")
    t.add_argument("--no_focus", action="store_true",
                   help="disable confusion-mined focus steps")
    t.add_argument("--amp", default=None, choices=["1", "0"],
                   help="force AMP on/off (default: on for CUDA)")
    t.add_argument("--split_parts", action="store_true",
                   help="also split brain.pt into brain_part_NN at the repo root")
    t.add_argument("--part_max_mb", type=int, default=96)
    # explicit architecture overrides (win over the preset)
    t.add_argument("--width", type=int, default=None)
    t.add_argument("--prompt_dim", type=int, default=None)
    t.add_argument("--prompt_layers", type=int, default=None)
    t.add_argument("--d_concept", type=int, default=None)
    t.add_argument("--pattern_d", type=int, default=None)
    t.add_argument("--pattern_layers", type=int, default=None)

    e = sub.add_parser("eval", help="load models/brain.pt + held-out self-test")
    e.add_argument("--device", default=None)
    e.add_argument("--seed", type=int, default=999)   # disjoint from training
    e.add_argument("--stress", action="store_true",
                   help="hard-degraded + cluttered rounds only - the honest numbers")

    s = sub.add_parser("smoke", help="tiny corpus, 1 epoch, CPU sanity check")

    a = ap.parse_args(argv)
    assert _TORCH, "torch is required: pip install torch numpy Pillow"

    if a.cmd == "train":
        models_dir = a.models_dir or MODELS_DIR
        resume = _find_resume(a.resume, models_dir)
        model, metrics = train_brain(
            epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed,
            device=a.device, preset=a.preset,
            width=a.width, prompt_dim=a.prompt_dim,
            prompt_layers=a.prompt_layers, d_concept=a.d_concept,
            pattern_d=a.pattern_d, pattern_layers=a.pattern_layers,
            phase2=a.phase2, kreg=a.kreg, focus=not a.no_focus,
            amp=(True if a.amp == "1" else False if a.amp == "0" else None),
            models_dir=models_dir, resume=resume,
            corpus_kwargs=dict(
                per_class=a.per_class, n_point=a.n_point, n_count=a.n_count,
                n_drag=a.n_drag, n_grid=a.n_grid, n_pattern=a.n_pattern,
                n_bbox=a.n_bbox, n_pipe=a.n_pipe, n_tower=a.n_tower,
                n_shape=a.n_shape, n_odrag=a.n_odrag, n_text=a.n_text,
                hard_frac=a.hard_frac, degrade_frac=a.degrade_frac,
                clutter_frac=a.clutter_frac,
                router_bank=not a.no_router_bank,
                hcap_dir=a.hcap_dir, hcap_views=a.hcap_views,
                photos_dir=a.photos_dir, photo_views=a.photo_views,
                real_only=a.real_only))
        if a.split_parts:
            pt = os.path.join(models_dir, "brain.pt")
            n = split_brain_parts(pt, max_mb=a.part_max_mb)
            sidecar = json.load(open(os.path.join(models_dir, "brain.json")))
            write_arch_sidecar(
                {"tile_size": sidecar.get("tile_size", DEFAULT_TILE_SIZE),
                 "scene_size": sidecar.get("scene_size", DEFAULT_SCENE_SIZE)},
                sidecar.get("arch", {}))
            print("== split brain.pt into %d parts (brain_part_NN) at %s =="
                  % (n, ROOT))
            print("   commit brain_part_NN + brain_arch.json so the Test tab")
            print("   can reassemble the new Brain on any machine.")
    elif a.cmd == "eval":
        solver = BrainSolver()
        if not solver.available:
            print("no models/brain.pt - run `python brain.py train` first")
            return
        if a.stress:
            print("== STRESS eval: fresh rounds, EVERY image degraded + cluttered ==")
            stress_test(solver.model, device=solver.device, seed=a.seed)
            return
        corpus = build_brain_corpus(per_class=60, n_point=400, n_count=200,
                                    n_drag=400, n_grid=150, n_pattern=80,
                                    n_bbox=300, n_pipe=150, n_tower=150,
                                    n_shape=150, n_text=300, seed=a.seed)
        print("== Brain eval on held-out rounds (seed %d) ==" % a.seed)
        eval_brain(solver.model, corpus, device=solver.device,
                   tile_va=[i for i in range(len(corpus["tile_y"])) if i % 20 == 0],
                   point_va_single=[i for i in range(len(corpus["point_m"]))
                                    if i % 20 == 0 and corpus["point_m"][i].get("type") != "count"],
                   drag_va=[i for i in range(len(corpus["drag_m"])) if i % 20 == 0],
                   bbox_va=[i for i in range(len(corpus["bbox_m"])) if i % 20 == 0],
                   pat_va=[i for i in range(len(corpus["pat_m"])) if i % 20 == 0],
                   router_va=[i for i in range(len(corpus["router"])) if i % 20 == 0])
    elif a.cmd == "smoke":
        # tiny architecture + tiny corpus + 1 epoch: a fast end-to-end proof
        # that every head's forward/backward runs with no shape errors
        # (including the pipe + tower drag families, hard degradation and
        # the router bank).
        train_brain(epochs=1, batch=16, seed=0, device="cpu",
                    preset="small", phase2=0,
                    corpus_kwargs=dict(per_class=20, n_point=200, n_count=80,
                                       n_drag=200, n_grid=60, n_pattern=40,
                                       n_bbox=120, n_pipe=40, n_tower=40,
                                       n_shape=40, n_text=60,
                                       hard_frac=0.5, degrade_frac=0.5))


def _in_notebook():
    """True inside an IPython/Jupyter kernel, False for a real `python
    brain.py` launch. In a notebook cell __name__ IS "__main__", so the usual
    guard would auto-run main() and argparse would choke on Jupyter's
    kernel-JSON argv; get_ipython() exists only in an interactive kernel."""
    try:
        get_ipython  # noqa: F821  (injected by IPython/Jupyter kernels)
        return True
    except NameError:
        return False


if __name__ == "__main__" and _in_notebook():
    # Pasted into a Kaggle/Jupyter cell: this cell only DEFINES the Brain
    # (it does NOT auto-train — auto-running the CLI here used to crash
    # argparse on the kernel-JSON). Print a clear ready-message so the cell
    # never looks like it silently died, and tell the user exactly what to
    # run next.
    if _TORCH:
        _n_gpu = torch.cuda.device_count()
        if _n_gpu:
            _dev = "cuda (%d GPU ready)" % _n_gpu
        elif "+cpu" in torch.__version__:
            _dev = "CPU-only torch build (%s) - GPU will NOT be used" % torch.__version__
        else:
            _dev = "cpu (no GPU detected - enable the GPU accelerator)"
        print("[brain.py v2] ready. torch %s | device: %s" % (torch.__version__, _dev))
        if _n_gpu == 0 and "+cpu" in torch.__version__:
            print("[brain.py] *** You installed the CPU-only torch wheel. On Kaggle,")
            print("[brain.py]     don't 'pip install torch' - CUDA torch is preinstalled")
            print("[brain.py]     on GPU images. Fix with EITHER:")
            print("[brain.py]       (a) restart the session (reloads preinstalled CUDA torch), or")
            print("[brain.py]       (b) !pip install -q torch --index-url https://download.pytorch.org/whl/cu121")
            print("[brain.py]     then re-run this cell.")
        print("[brain.py] You PASTED this cell, so the functions are GLOBALS - call them")
        print("[brain.py] directly with NO 'brain.' prefix. In the NEXT cell run:")
        print("              main(['smoke'])                                        # quick check")
        print("              main(['train', '--preset', 'giga', '--device', 'cuda',")
        print("                    '--epochs', '14', '--phase2', '4',")
        print("                    '--hcap_dir', '/tmp/hcap', '--split_parts'])     # the ~1.1GB giga")
        print("              train_brain(device='cuda', preset='giga')              # ...or call directly")
    else:
        print("[brain.py] code loaded, but torch is NOT installed. Run this cell:")
        print("              !pip install torch numpy Pillow")
        print("then restart the kernel and re-run the brain.py cell.")
elif __name__ == "__main__":
    main()

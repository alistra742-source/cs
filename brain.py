#!/usr/bin/env python3
"""
brain.py — the Brain: ONE unified neural network that solves EVERY hCaptcha
challenge family.

The shipped solver is three tiny models (TileNet / PointNet / DragNet) plus a
hand-written family router plus a vision-model fallback. That works, but the
"thinking" is spread across a dozen files and every family is solved by a
different path. The Brain collapses all of it into a single network:

    ┌──────────────────────────────────────────────────────────────┐
    │                       shared ResNet backbone                 │
    │  (one feature map feeds every head; trained jointly)         │
    └──────────────────────────────────────────────────────────────┘
       │      │       │       │       │        │          │
       ▼      ▼       ▼       ▼       ▼        ▼          ▼
    Tile   Heatmap  Drag   BBox   Router  Pattern    Prompt
    head   head     head   head   head    reasoner   encoder
    (60)  (60+bg)  (piece/  (ctr+   (family   (set-       (text→vec)
           point/   slot)   size)   router)  transformer)
           count/
           scan)

What each head owns (so the Brain covers every family in SOLVER.md):

  PromptEncoder   the LANGUAGE BRAIN — a character-level transformer that reads
                  ANY prompt (every noun, superlative, affordance wording).
                  Character-level = no fixed vocabulary, so it generalises to
                  words it has never seen (helicopter, skyscraper, ...). This is
                  deliberately where most of the parameters live — reading the
                  question is the core 'smartness', and every parameter is
                  exercised on every pass (no padding filler).
  KnowledgeBank   WORLD KNOWLEDGE — a learned ontology: each of the 60 classes
                  owns a rich CONCEPT embedding (vehicle-ness, tool-ness,
                  animal-ness...) plus a class->class RELATION matrix
                  (affordances, category ties, size tiers). This is the
                  hand-coded prompt catalog made dense and learnable.
  TileHead        binary tile grids ("click each image containing a bus"),
                  affordance / set-down / material grids, AND the per-cell
                  + per-candidate labelling the pattern reasoner needs.
  HeatmapHead     area_select point rounds, relational superlative rounds
                  ("jumps the highest"), and counting (multi-instance peaks).
                  60 class channels + 1 background channel — same decode the
                  production PointLocator uses.
  DragHead        image_drag_drop (piece -> slot) and the pattern / tower drags.
  BBoxHead        area_select_bbox ("draw a box around the cat's head").
  RouterHead      a LEARNED (prompt + image) -> family classifier so the Brain
                  self-routes; the proven rule router (hcaptcha_types.classify)
                  stays available as a strong prior.
  PatternReasoner a set-transformer that THINKS IN CONCEPTS: it labels every
                  cell/candidate, projects each into concept space via the
                  KnowledgeBank, then reasons over (visual + concept) tokens to
                  pick the pattern-completing tile — a learned Latin-square /
                  analogy solver, not the hand-coded resolver.

The whole thing trains END-TO-END on every family at once (joint multi-task
loop). The default architecture is sized to a ~98 MB checkpoint, and the bulk
of those parameters are genuine capacity — the language brain + the class
ontology — not a padded conv backbone. Architecture knobs (--prompt_dim,
--prompt_layers, --d_concept, --pattern_d, --pattern_layers, --width) resize
it; the full arch is persisted in models/brain.json so a trained Brain always
reloads with the EXACT shape it was built with (no load-time mismatches).

It is Kaggle-ready: the corpus is generated IN MEMORY from the repo's own
deterministic generators (make_dataset / make_challenges) plus a small bbox
generator added here, so a notebook needs nothing but `pip install torch
numpy Pillow` and a GPU. If the pre-built data_v2 corpora exist (the repo
workflow) they are reused instead.

Drop-in for the shipped solver: BrainSolver exposes the exact method names
TileClassifier / PointLocator / DragLocator use (classify_many, probabilities,
scan, locate, count, locate_relational, locate_drag), plus a single
`solve(...)` entry point that routes a round and returns the answer for ANY
family, confidence-gated so the server can still fall back to the vision
model below the threshold.

CLI
---
    # build corpus in-memory + train all heads jointly on a GPU (Kaggle):
    python brain.py train --epochs 12 --width 48 --device cuda
    # held-out, per-family self-test (prints tile acc, point hit@10%, count
    # exact, drag both-hit, bbox IoU, pattern cand-acc, router acc):
    python brain.py eval
    # quick smoke (tiny corpus, 1 epoch, CPU):
    python brain.py smoke

Weights land in models/brain.pt (+ models/brain.json sidecar with class list,
family list, sizes, widths and held-out metrics).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
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


# ═══════════════════════════════════════════════════════════════════════════
#  Model
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
        """Small preactivation ResNet: stem + 4 stages, down to S/8.

        Stronger than the shipped 4-conv backbone (residual shortcuts carry
        gradient to every head), but still tiny — width 48 → ~384 final
        channels and only a few MB of weights.
        """

        def __init__(self, width: int = 48):
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
            self.out_channels = 8 * width
            self.final_bn = nn.BatchNorm2d(self.out_channels)

        def forward(self, x):
            x = self.stem(x)
            x = self.s1(x)     # S
            x = self.s2(x)     # S/2
            x = self.s3(x)     # S/4
            x = self.s4(x)     # S/8
            return F.relu(self.final_bn(x), inplace=True)

    class PromptEncoder(nn.Module):
        """The LANGUAGE BRAIN: prompt string -> dense vector via a character
        transformer.

        This is deliberately where most of the Brain's parameters live, because
        reading the question is the core 'smartness' — every family's wording,
        every object noun, every superlative ("jumps the highest"), every
        affordance ("things you can work on with the item shown") has to be
        understood. Character-level tokenisation means there is NO fixed
        vocabulary, so it generalises to words it has never seen (helicopter,
        skyscraper, …) the way the vision fallback has to — it learns
        morphology, not a lookup table.

        No padding filler: every parameter here is exercised on every forward
        pass (the transformer attends across real character tokens), so the
        size is genuine language capacity, not bloat.
        """

        def __init__(self, dim: int = 512, n_layers: int = 6, nhead: int = 8,
                     max_len: int = 96, ff_mult: int = 4):
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

    class KnowledgeBank(nn.Module):
        """The Brain's WORLD KNOWLEDGE: a learned ontology of the 60 classes.

        Each class owns a rich CONCEPT embedding (what it *means*: a bus and a
        truck share vehicle-ness; a drill and a hammer share tool-ness; a cat
        and a dog share animal-ness) plus a class->class RELATION matrix that
        captures affordances and category ties (drill -> wood/wall, animals vs
        vehicles, same size tier). This is the structured knowledge the prompt
        catalog encodes by hand — here it is learned and dense.

        The pattern reasoner reasons over these concept tokens, so it solves
        "put one of the animals into the empty spot to complete the pattern" by
        THINKING about what each cell IS (its concept), not just by pixels — a
        learned Latin-square solver that can also do analogies the hand-coded
        resolver refuses. Every parameter is used on every pattern/route pass.
        """

        def __init__(self, n_classes=N_CLASSES, d_concept=256):
            super().__init__()
            self.n_classes = n_classes
            self.concept = nn.Embedding(n_classes, d_concept)
            self.rel = nn.Parameter(torch.zeros(n_classes, n_classes))
            self.norm = nn.LayerNorm(d_concept)

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
        """1x1 conv -> (n_classes + 1 background) spatial channels.

        One channel per class so a single forward pass localises EVERY class
        (point / scan / count) and the per-cell background channel suppresses
        phantom presences — the same design as the production PointNet.
        """

        def __init__(self, c_in, n_classes):
            super().__init__()
            self.head = nn.Conv2d(c_in, n_classes + 1, 1)

        def forward(self, feat):
            return self.head(feat)

    class DragHead(nn.Module):
        """1x1 conv -> 2 channels: piece (drag-from) and slot (drag-to)."""

        def __init__(self, c_in):
            super().__init__()
            self.head = nn.Conv2d(c_in, 2, 1)

        def forward(self, feat):
            return self.head(feat)

    class BBoxHead(nn.Module):
        """area_select_bbox: a 1-channel centre heatmap + a global (w, h) reg.

        Bbox rounds have exactly one target object, so a centre heatmap (decoded
        with soft-argmax) plus a single global width/height regression is the
        right shape and trains cleanly from the in-memory bbox generator.
        """

        def __init__(self, c_in):
            super().__init__()
            self.center = nn.Conv2d(c_in, 1, 1)
            self.size = nn.Sequential(
                nn.Linear(c_in, c_in), nn.ReLU(inplace=True), nn.Linear(c_in, 2))

        def forward(self, feat):
            ctr = self.center(feat)[:, 0]                 # (B, H, W)
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            wh = torch.sigmoid(self.size(pooled))         # (B, 2) in 0..1
            return ctr, wh

    class RouterHead(nn.Module):
        """Learned (prompt + image) -> family classifier.

        Trained from the manifest `type` of every generated round plus a few
        hand-written (prompt -> family) pairs for choice / text / tower, which
        have no offline image supervision. The rule router
        (hcaptcha_types.classify) is the production source of truth; this head
        is a learnable, image-aware alternative / cross-check.
        """

        def __init__(self, c_in, prompt_dim, n_families):
            super().__init__()
            self.img = nn.Sequential(nn.Linear(c_in, 128), nn.ReLU(inplace=True))
            self.txt = nn.Sequential(nn.Linear(prompt_dim, 128), nn.ReLU(inplace=True))
            self.out = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, n_families))

        def forward(self, img_pool, prompt_vec):
            return self.out(torch.cat([self.img(img_pool), self.txt(prompt_vec)], dim=1))

    class PatternReasoner(nn.Module):
        """Set-transformer pattern solver that THINKS IN CONCEPTS.

        Each cell/candidate is labelled by the tile head -> a class
        distribution -> a CONCEPT token (KnowledgeBank.from_probs). The token
        the transformer attends over is therefore "what is this thing?" (its
        learned meaning), not raw pixels — so the net can complete a pattern by
        reasoning "row has cat,dog,? ; the missing one must be the third
        animal", the way the hand-coded Latin-square resolver does, but learned
        and able to generalise. A residual visual token is added so appearance
        still informs identity. Roles: 0-8 cells, 9-11 candidates, 12 prompt.
        """

        def __init__(self, c_in, prompt_dim, d_concept,
                     d_model=256, nhead=4, layers=3):
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
        """The whole network: shared backbone + knowledge + every head.

        Architecture hyper-parameters are passed explicitly and persisted in
        the sidecar, so a trained Brain reloads with the EXACT shape it was
        built with — no load-time shape mismatches, ever.
        """

        def __init__(self, n_classes=N_CLASSES, width=48,
                     prompt_dim=512, prompt_layers=8, d_concept=320,
                     pattern_d=320, pattern_layers=4,
                     n_families=len(FAMILIES)):
            super().__init__()
            self.n_classes = n_classes
            self.width = width
            self.prompt_dim = prompt_dim
            self.prompt_layers = prompt_layers
            self.d_concept = d_concept
            self.pattern_d = pattern_d
            self.pattern_layers = pattern_layers
            self.n_families = n_families
            self.backbone = BrainBackbone(width)
            c = self.backbone.out_channels
            self.prompt_enc = PromptEncoder(prompt_dim, prompt_layers)
            self.knowledge = KnowledgeBank(n_classes, d_concept)
            self.tile_head = TileHead(c, n_classes)
            self.heatmap_head = HeatmapHead(c, n_classes)
            self.drag_head = DragHead(c)
            self.bbox_head = BBoxHead(c)
            self.router_head = RouterHead(c, prompt_dim, n_families)
            self.pattern_reasoner = PatternReasoner(
                c, prompt_dim, d_concept, pattern_d, layers=pattern_layers)

        # ── per-head forward helpers (each reuses the shared backbone) ──
        def features(self, x):
            return self.backbone(x)

        def tile_logits(self, feat):
            return self.tile_head(feat)

        def heatmaps(self, feat):
            """(B, n_classes+1, H, W) raw per-class + background logits."""
            return self.heatmap_head(feat)

        def drag_maps(self, feat):
            return self.drag_head(feat)            # (B, 2, H, W)

        def bbox(self, feat):
            return self.bbox_head(feat)            # center (B,H,W), wh (B,2)

        def route(self, img_feat, prompt_vec):
            pool = F.adaptive_avg_pool2d(img_feat, 1).flatten(1)
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


def build_brain_corpus(per_class=600, n_point=14000, n_count=5000,
                       n_drag=9000, n_grid=3000, n_pattern=1500, n_bbox=7000,
                       tile_size=DEFAULT_TILE_SIZE, scene_size=DEFAULT_SCENE_SIZE,
                       seed=7, verbose=True):
    """Build the full multi-task corpus in memory.

    Returns a dict of per-family tensors + metadata, ready for the joint
    training loop. Everything is generated by the repo's deterministic
    generators, so two runs with the same seed are identical and a Kaggle
    notebook needs no pre-built data.
    """
    from make_challenges import (make_point_round, make_count_round,
                                 make_drag_round, make_pattern_round,
                                 make_grid_round)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    t0 = time.time()

    # ── tile classification: single painted tiles + real photos (fallback ok) ─
    log("  tiles: %d/class x %d classes  (generating %d images...)" % (
        per_class, N_CLASSES, per_class * N_CLASSES))
    tx = torch.empty((per_class * N_CLASSES, 3, tile_size, tile_size), dtype=torch.uint8)
    ty = torch.empty(per_class * N_CLASSES, dtype=torch.long)
    i = 0
    for cid, name in enumerate(CLASSES):
        for k in range(per_class):
            rng = random.Random("tile|%s|%d|%d" % (name, seed, k))
            tx[i] = _img_to_u8(md.render(name, tile_size, rng), tile_size)
            ty[i] = cid
            i += 1
        if (cid + 1) % 5 == 0 or cid == N_CLASSES - 1:
            log("    tiles: %d/%d classes done (%d images)" % (
                cid + 1, N_CLASSES, i))

    # ── grid rounds: feed their 9 tiles through the SAME tile head ─────────
    grid_tiles, grid_labels = [], []
    log("  grids: generating %d (9 tiles each)..." % n_grid)
    for k in range(n_grid):
        rng = random.Random("grid|%d|%d" % (seed, k))
        img, meta = make_grid_round(rng, scene_size)
        names = meta["tiles"]
        boxes = meta["tile_boxes"]
        W, H = img.size
        for name, box in zip(names, boxes):
            x0, y0, s = box[0], box[1], box[2]
            crop = img.crop((x0, y0, x0 + s, y0 + s))
            grid_tiles.append(_img_to_u8(crop, tile_size))
            grid_labels.append(CID[name])
        if n_grid >= 1000 and (k + 1) % 500 == 0:
            log("    grids: %d/%d" % (k + 1, n_grid))
    log("    grids: done (%d)" % n_grid)
    if grid_tiles:
        gx = torch.stack(grid_tiles)
        gy = torch.tensor(grid_labels, dtype=torch.long)
        tx = torch.cat([tx, gx])
        ty = torch.cat([ty, gy])

    # ── point + count rounds (heatmap head) ────────────────────────────────
    def _load_scenes(fn, n, kind):
        log("  %s: generating %d..." % (kind, n))
        xs = torch.empty((n, 3, scene_size, scene_size), dtype=torch.uint8)
        metas = []
        for k in range(n):
            rng = random.Random("%s|%d|%d" % (kind, seed, k))
            img, meta = fn(rng, scene_size)
            xs[k] = _img_to_u8(img, scene_size)
            metas.append(meta)
            if n >= 4000 and (k + 1) % 2000 == 0:
                log("    %s: %d/%d" % (kind, k + 1, n))
        log("    %s: done (%d)" % (kind, n))
        return xs, metas

    point_x, point_m = _load_scenes(make_point_round, n_point, "point")
    count_x, count_m = _load_scenes(make_count_round, n_count, "count")

    # ── drag rounds (drag head) ────────────────────────────────────────────
    drag_x, drag_m = _load_scenes(make_drag_round, n_drag, "drag")

    # ── bbox rounds (bbox head) ────────────────────────────────────────────
    log("  bbox: generating %d..." % n_bbox)
    bbox_x = torch.empty((n_bbox, 3, scene_size, scene_size), dtype=torch.uint8)
    bbox_m = []
    for k in range(n_bbox):
        rng = random.Random("bbox|%d|%d" % (seed, k))
        img, meta = make_bbox_round(rng, scene_size)
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
        pat_imgs.append(img)
        pat_m.append(meta)
        if n_pattern >= 400 and (k + 1) % 200 == 0:
            log("    pattern: %d/%d" % (k + 1, n_pattern))
    log("    pattern: done (%d)" % n_pattern)

    # ── router training pairs (prompt -> family) from every round + extras ─
    router = []
    for m in point_m:
        router.append((m.get("prompt", ""), AREA_POINT))
    for m in count_m:
        router.append((m.get("prompt", ""), COUNT))
    for m in drag_m:
        router.append((m.get("prompt", ""), DRAG_DROP))
    for m in pat_m:
        router.append((m.get("prompt", ""), PATTERN))
    for m in bbox_m:
        router.append((m.get("prompt", ""), AREA_BBOX))
    # hand pairs for families with no offline image supervision
    for p in ["Please click each image containing a bus",
              "Select all tiles with a cat", "Pick the images showing a tree"]:
        router.append((p, BINARY))
    for p in ["Draw a box around the cat's head", "Box the largest object"]:
        router.append((p, AREA_BBOX))
    for p in ["Select the most accurate description",
              "Which of these is correct?"]:
        router.append((p, MULTIPLE_CHOICE))
    for p in ["Type the text you see", "Enter the code below"]:
        router.append((p, TEXT_ENTRY))
    for p in ["Move the correct missing block segment onto the incomplete tower",
              "Stack the blocks to the same height"]:
        router.append((p, TOWER))

    log("  router prompt pairs: %d" % len(router))
    log("  corpus built in %.0fs (%.0f MB uint8)" % (
        time.time() - t0,
        sum(t.element_size() * t.numel() for t in
            [tx, point_x, count_x, drag_x, bbox_x]) / 1e6))
    return {
        "tile_x": tx, "tile_y": ty,
        "point_x": point_x, "point_m": point_m,
        "count_x": count_x, "count_m": count_m,
        "drag_x": drag_x, "drag_m": drag_m,
        "bbox_x": bbox_x, "bbox_m": bbox_m,
        "pat_imgs": pat_imgs, "pat_m": pat_m,
        "router": router,
        "tile_size": tile_size, "scene_size": scene_size,
    }


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
#  Training: joint multi-task loop over every family
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


def train_brain(epochs=12, width=48, batch=64, lr=1e-3, seed=0,
                device="cpu", corpus=None, corpus_kwargs=None,
                models_dir=MODELS_DIR, verbose=True,
                prompt_dim=512, prompt_layers=8, d_concept=320,
                pattern_d=320, pattern_layers=4):
    """Train every head of the Brain jointly.

    Each epoch cycles through the families (tile, point, count, drag, bbox,
    pattern, router); each step trains ONE head with its own loss. Cycling
    families (rather than per-sample masking) is the standard, stable way to
    train a multi-task net and keeps every batch loss clean.
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
    if corpus is None:
        corpus = build_brain_corpus(seed=seed,
                                     **(corpus_kwargs or {}))
    tile_size = corpus["tile_size"]
    scene_size = corpus["scene_size"]
    Sfeat = scene_size // 8
    model = Brain(N_CLASSES, width=width, prompt_dim=prompt_dim,
                  prompt_layers=prompt_layers, d_concept=d_concept,
                  pattern_d=pattern_d, pattern_layers=pattern_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs), eta_min=lr * 0.05)
    log("== Brain: %.1fM params (%.1f MB fp32) | device %s%s ==" % (
        sum(p.numel() for p in model.parameters()) / 1e6,
        model.param_mb(), device,
        (" (%d GPU)" % n_gpu) if n_gpu else ""))

    # held-out splits per family (every 20th index)
    tile_tr, tile_va = _split(len(corpus["tile_y"]))
    point_tr, point_va = _split(len(corpus["point_m"]))
    point_va_single = [i for i in point_va if corpus["point_m"][i].get("type") != "count"]
    count_tr = [i for i, m in enumerate(corpus["count_m"])]  # count joins point head
    drag_tr, drag_va = _split(len(corpus["drag_m"]))
    bbox_tr, bbox_va = _split(len(corpus["bbox_m"]))
    pat_tr, pat_va = _split(len(corpus["pat_m"]))
    router = corpus["router"]
    router_tr, router_va = _split(len(router))

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

    for ep in range(epochs):
        model.train()
        random.Random(seed * 100 + ep).shuffle(tile_tr)
        random.Random(seed * 100 + ep).shuffle(point_tr)
        random.Random(seed * 100 + ep).shuffle(drag_tr)
        random.Random(seed * 100 + ep).shuffle(bbox_tr)
        random.Random(seed * 100 + ep).shuffle(pat_tr)
        random.Random(seed * 100 + ep).shuffle(router_tr)
        ep_loss = 0.0
        n_steps = 0
        t0 = time.time()

        # 1) TILE head — single tiles + grid tiles
        for s in range(math.ceil(len(tile_tr) / batch)):
            b = tile_tr[s * batch:(s + 1) * batch]
            x = _jitter(_move(corpus["tile_x"][b], device))
            with torch.no_grad():
                feat = model.features(x)
            logits = model.tile_logits(feat)
            loss = F.cross_entropy(logits, _move(corpus["tile_y"][b], device),
                                   label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1

        # 2) POINT head — single + relational point rounds
        for s in range(math.ceil(len(point_tr) / batch)):
            b = point_tr[s * batch:(s + 1) * batch]
            pts, tc, valid, K = _step_count_metas(corpus["point_m"], b, device)
            x, tpts, _ = _prep_geom(_move(corpus["point_x"][b], device), pts,
                                    random.Random(seed * 100000 + ep * 1000 + s))
            feat = model.features(x)
            hm = model.heatmaps(feat)
            masks = torch.zeros(len(b), Sfeat, Sfeat, device=device)
            cx = (tpts[:, :, 0] * Sfeat).long().clamp(0, Sfeat - 1)
            cy = (tpts[:, :, 1] * Sfeat).long().clamp(0, Sfeat - 1)
            rows = torch.arange(len(b), device=device).view(-1, 1).expand(len(b), K)
            keep = valid.reshape(len(b), K)
            masks[rows[keep], cy[keep], cx[keep]] = 1.0
            single = masks.sum(dim=(1, 2)) == 1
            l1_xy = tpts.sum(dim=1) / keep.sum(dim=1, keepdim=True).clamp(min=1)
            loss = _spatial_ce(hm, tc, masks, l1_xy, single)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1

        # 3) COUNT head — multi-instance count rounds through the same heatmap head
        if count_tr:
            for s in range(math.ceil(len(count_tr) / batch)):
                b = count_tr[s * batch:(s + 1) * batch]
                pts, tc, valid, K = _step_count_metas(corpus["count_m"], b, device)
                x, tpts, _ = _prep_geom(_move(corpus["count_x"][b], device), pts,
                                        random.Random(seed * 100000 + ep * 2000 + s))
                feat = model.features(x)
                hm = model.heatmaps(feat)
                masks = torch.zeros(len(b), Sfeat, Sfeat, device=device)
                cx = (tpts[:, :, 0] * Sfeat).long().clamp(0, Sfeat - 1)
                cy = (tpts[:, :, 1] * Sfeat).long().clamp(0, Sfeat - 1)
                rows = torch.arange(len(b), device=device).view(-1, 1).expand(len(b), K)
                keep = valid.reshape(len(b), K)
                masks[rows[keep], cy[keep], cx[keep]] = 1.0
                single = masks.sum(dim=(1, 2)) == 1
                l1_xy = tpts.sum(dim=1) / keep.sum(dim=1, keepdim=True).clamp(min=1)
                loss = _spatial_ce(hm, tc, masks, l1_xy, single)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); n_steps += 1

        # 4) DRAG head — piece + slot heatmaps
        for s in range(math.ceil(len(drag_tr) / batch)):
            b = drag_tr[s * batch:(s + 1) * batch]
            tf = torch.tensor([[corpus["drag_m"][i]["fx"],
                                corpus["drag_m"][i]["fy"]] for i in b])
            tt = torch.tensor([[corpus["drag_m"][i]["tx"],
                                corpus["drag_m"][i]["ty"]] for i in b])
            tgt = torch.cat([tf.unsqueeze(1), tt.unsqueeze(1)], dim=1)
            x, txy, _ = _prep_geom(_move(corpus["drag_x"][b], device), tgt,
                                   random.Random(seed * 100000 + ep * 3000 + s))
            feat = model.features(x)
            hms = model.drag_maps(feat)
            loss = (_channel_ce_l1(hms[:, 0], txy[:, 0]) +
                    _channel_ce_l1(hms[:, 1], txy[:, 1]))
            opt.zero_grad(); loss.backward(); opt.step()
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
            tctr = tctr[:, 0]
            twh = (wh * scl.unsqueeze(1)).clamp(0.02, 1.0)
            feat = model.features(x)
            cm, pw = model.bbox(feat)
            loss = (_channel_ce_l1(cm, tctr) + 2.0 * F.l1_loss(pw, twh))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1

        # 6) PATTERN reasoner — set-transformer over cells + candidates
        pat_batch = max(4, batch // 4)
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
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1

        # 7) ROUTER head — prompt -> family (no image gradient needed through
        #    the backbone; reuse a detached pooled feature of a random scene)
        for s in range(math.ceil(len(router_tr) / 128)):
            b = router_tr[s * 128:(s + 1) * 128]
            prompts = [router[i][0] for i in b]
            labels = torch.tensor([FAM_ID[router[i][1]] for i in b],
                                  device=device)
            pv = model.prompt_enc(prompts)
            # zero image feature so the router learns primarily from text
            zero_img = torch.zeros(len(b), model.backbone.out_channels,
                                   device=device)
            logits = model.router_head(zero_img, pv)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1

        sched.step()
        log("  epoch %d/%d  mean_loss %.4f  (%d steps, %.0fs)" % (
            ep + 1, epochs, ep_loss / max(1, n_steps), n_steps, time.time() - t0))

    metrics = eval_brain(model, corpus, device=device,
                         tile_va=tile_va, point_va_single=point_va_single,
                         drag_va=drag_va, bbox_va=bbox_va, pat_va=pat_va,
                         router_va=router_va, verbose=verbose)
    _save_brain(model, metrics, corpus, models_dir)
    return model, metrics


def _save_brain(model, metrics, corpus, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    pt = os.path.join(models_dir, "brain.pt")
    torch.save(model.state_dict(), pt)
    # Persist the FULL architecture so BrainSolver rebuilds the exact same
    # network shape — no load-time shape mismatches.
    sidecar = {
        "kind": "brain", "classes": CLASSES, "families": FAMILIES,
        "n_classes": N_CLASSES,
        "arch": {
            "width": model.width,
            "prompt_dim": model.prompt_dim,
            "prompt_layers": model.prompt_layers,
            "d_concept": model.d_concept,
            "pattern_d": model.pattern_d,
            "pattern_layers": model.pattern_layers,
        },
        "tile_size": corpus["tile_size"], "scene_size": corpus["scene_size"],
        "metrics": metrics,
        "size_mb": os.path.getsize(pt) / 1e6,
    }
    with open(os.path.join(models_dir, "brain.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    print("  saved %s + sidecar (%.1f MB, %.1fM params)" % (
        pt, os.path.getsize(pt) / 1e6,
        sum(p.numel() for p in model.parameters()) / 1e6))


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation: held-out, per-family self-test (mirrors test_solver.py)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_brain(model, corpus, device="cpu", tile_va=None, point_va_single=None,
               drag_va=None, bbox_va=None, pat_va=None, router_va=None,
               verbose=True):
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    model.eval()
    tile_size = corpus["tile_size"]
    scene_size = corpus["scene_size"]
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
            feat = model.features(x)
            hm = model.heatmaps(feat)
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
            x = _to_float(_move(corpus["drag_x"][b], device))
            hms = model.drag_maps(model.features(x))
            ef.extend(torch.linalg.norm(soft_argmax2d(hms[:, 0]) - tf[
                [drag_va.index(i) for i in b]].to(device), dim=1).tolist())
            et.extend(torch.linalg.norm(soft_argmax2d(hms[:, 1]) - tt[
                [drag_va.index(i) for i in b]].to(device), dim=1).tolist())
        both = sum(a <= 0.10 and b <= 0.10 for a, b in zip(ef, et)) / len(ef)
        metrics["drag_hit_both"] = both
        log("  drag both@10%%:  %.3f" % both)

    # bbox IoU (centre within 10% AND size within 25%)
    if bbox_va:
        good = 0
        for i in bbox_va:
            m = corpus["bbox_m"][i]
            x = _to_float(_move(corpus["bbox_x"][i:i + 1], device))
            cm, wh = model.bbox(model.features(x))
            pred = soft_argmax2d(cm)[0]
            pwh = wh[0]
            d = math.hypot(pred[0].item() - m["cx"], pred[1].item() - m["cy"])
            sd = max(abs(pwh[0].item() - m["w"]), abs(pwh[1].item() - m["h"]))
            good += int(d <= 0.10 and sd <= 0.25)
        metrics["bbox_acc"] = good / len(bbox_va)
        log("  bbox acc:       %.3f" % metrics["bbox_acc"])

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


class BrainSolver:
    """The Brain at inference time.

    Loads models/brain.pt (+ .json). Exposes the method names the shipped
    TileClassifier / PointLocator / DragLocator use, so server.py can swap them
    in, PLUS a single ``solve(...)`` that routes any round and returns the
    answer. Every answer is confidence-gated: below threshold the method
    returns None so the caller falls back to the vision model (same safety the
    production offline path uses).
    """

    def __init__(self, models_dir=MODELS_DIR, device=None,
                 min_conf=float(os.environ.get("SOLVER_CNN_MIN_CONF", "0.62"))):
        self.available = False
        self.model = None
        self.classes = list(CLASSES)
        self.families = list(FAMILIES)
        self.size = DEFAULT_SCENE_SIZE
        self.tile_size = DEFAULT_TILE_SIZE
        self.width = 48
        self.min_conf = min_conf
        self.device = device or ("cuda" if _TORCH and torch.cuda.is_available()
                                 else "cpu")
        if not _TORCH:
            return
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
            n = len(self.classes)
            # Rebuild with the EXACT architecture recorded at train time, so
            # the state_dict always fits — no shape mismatches at load.
            self.model = Brain(n, width=self.width, prompt_dim=prompt_dim,
                               prompt_layers=prompt_layers, d_concept=d_concept,
                               pattern_d=pattern_d, pattern_layers=pattern_layers,
                               n_families=len(self.families))
            self.model.load_state_dict(torch.load(pt, map_location=self.device))
            self.model.to(self.device).eval()
            self.available = True
        except Exception:  # pragma: no cover
            self.model = None
            self.available = False

    # ── low-level image prep ──────────────────────────────────────────────
    def _prep_tile(self, im, size=None):
        size = size or self.tile_size
        im = _to_pil(im)
        if im.size != (size, size):
            im = im.resize((size, size), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
        return ((x.permute(2, 0, 1).unsqueeze(0) - 0.5) / 0.5).to(self.device)

    def _feat(self, im, size=None):
        with torch.no_grad():
            return self.model.features(self._prep_tile(im, size))

    # ── TileClassifier drop-in ────────────────────────────────────────────
    @torch.no_grad()
    def probabilities(self, images):
        """List of {label: prob} dicts, one per image."""
        if not self.available:
            return []
        ims = images if isinstance(images, (list, tuple)) else [images]
        xs = torch.cat([self._prep_tile(im) for im in ims], dim=0)
        probs = F.softmax(self.model.tile_logits(self.model.features(xs)), dim=1)
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
    @torch.no_grad()
    def _scores(self, image):
        """(presence_map (C,H,W), location (C,2)) in one forward pass."""
        if not self.available:
            return None, None
        hm = self.model.heatmaps(self._feat(image, self.size)).squeeze(0)
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
        """Same self-gating peak counter as PointLocator.count — a count answer
        is graded EXACTLY, so border/fragmented/weak maps return None."""
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
    @torch.no_grad()
    def locate_drag(self, image):
        if not self.available:
            return None
        hm = self.model.drag_maps(self._feat(image, self.size)).squeeze(0)
        pf = soft_argmax2d(hm[0])[0]
        pt = soft_argmax2d(hm[1])[0]
        return {"from": (float(pf[0]), float(pf[1])),
                "to": (float(pt[0]), float(pt[1]))}

    # ── BBox ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def bbox(self, image):
        if not self.available:
            return None
        cm, wh = self.model.bbox(self._feat(image, self.size))
        c = soft_argmax2d(cm)[0]
        w, h = float(wh[0, 0]), float(wh[0, 1])
        cx, cy = float(c[0]), float(c[1])
        return {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h}

    # ── Pattern reasoner ──────────────────────────────────────────────────
    @torch.no_grad()
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
        pv = self.model.prompt_enc([prompt])
        logits = self.model.pattern(cf, xf, pv)[0]
        prob = F.softmax(logits, dim=0)
        idx = int(prob.argmax())
        return {"candidate": idx, "confidence": float(prob[idx]),
                "box": cand_boxes[idx]}

    # ── Learned router (prompt -> family) ─────────────────────────────────
    @torch.no_grad()
    def router_predict(self, prompt, image=None):
        if not self.available:
            return None
        pv = self.model.prompt_enc([prompt])
        if image is not None:
            feat = self._feat(image, self.size)
            pool = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        else:
            pool = torch.zeros(1, self.model.backbone.out_channels,
                               device=self.device)
        logits = self.model.router_head(pool, pv)[0]
        prob = F.softmax(logits, dim=0)
        idx = int(prob.argmax())
        return {"family": self.families[idx], "confidence": float(prob[idx])}

    # ── THE unified entry point ───────────────────────────────────────────
    def solve(self, image, prompt="", tiles=None, tile_boxes=None,
              cell_boxes=None, cand_boxes=None, example=None,
              dom=None, payload=None, use_learned_router=False):
        """Route ONE round and return its answer for ANY family.

        Returns a dict {family, answer, confidence} where ``answer`` is shaped
        per family (indices / (x,y) / bbox / drag / count / candidate), or
        None when the Brain is below confidence — so the caller falls back to
        the vision model, exactly like the shipped offline path.

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
    ap = argparse.ArgumentParser(description="The Brain — one unified model "
                                 "for every hCaptcha challenge family.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="build corpus in-memory + train all heads")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--width", type=int, default=48)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)
    t.add_argument("--per_class", type=int, default=600)
    t.add_argument("--n_point", type=int, default=14000)
    t.add_argument("--n_count", type=int, default=5000)
    t.add_argument("--n_drag", type=int, default=9000)
    t.add_argument("--n_grid", type=int, default=3000)
    t.add_argument("--n_pattern", type=int, default=1500)
    t.add_argument("--n_bbox", type=int, default=7000)
    # architecture knobs (the knowledge capacity). Defaults are sized so the
    # saved checkpoint is ~150 MB (under the 200 MB cap), the bulk of it in the
    # language brain + class ontology rather than a padded conv backbone.
    t.add_argument("--prompt_dim", type=int, default=512)
    t.add_argument("--prompt_layers", type=int, default=8)
    t.add_argument("--d_concept", type=int, default=320)
    t.add_argument("--pattern_d", type=int, default=320)
    t.add_argument("--pattern_layers", type=int, default=4)

    e = sub.add_parser("eval", help="load models/brain.pt + held-out self-test")
    e.add_argument("--device", default=None)
    e.add_argument("--seed", type=int, default=999)   # disjoint from training

    s = sub.add_parser("smoke", help="tiny corpus, 1 epoch, CPU sanity check")

    a = ap.parse_args(argv)
    assert _TORCH, "torch is required: pip install torch numpy Pillow"

    if a.cmd == "train":
        train_brain(epochs=a.epochs, width=a.width, batch=a.batch, lr=a.lr,
                    seed=a.seed, device=a.device,
                    prompt_dim=a.prompt_dim, prompt_layers=a.prompt_layers,
                    d_concept=a.d_concept, pattern_d=a.pattern_d,
                    pattern_layers=a.pattern_layers,
                    corpus_kwargs=dict(per_class=a.per_class, n_point=a.n_point,
                                       n_count=a.n_count, n_drag=a.n_drag,
                                       n_grid=a.n_grid, n_pattern=a.n_pattern,
                                       n_bbox=a.n_bbox))
    elif a.cmd == "eval":
        corpus = build_brain_corpus(per_class=60, n_point=400, n_count=200,
                                    n_drag=400, n_grid=150, n_pattern=80,
                                    n_bbox=300, seed=a.seed)
        solver = BrainSolver()
        if not solver.available:
            print("no models/brain.pt — run `python brain.py train` first")
            return
        print("== Brain eval on held-out rounds (seed %d) ==" % a.seed)
        eval_brain(solver.model, corpus, device=solver.device,
                   tile_va=list(range(0, len(corpus["tile_y"]), 1))[::20] or list(range(min(20, len(corpus["tile_y"])))),
                   point_va_single=[i for i in range(len(corpus["point_m"]))
                                    if i % 20 == 0 and corpus["point_m"][i].get("type") != "count"],
                   drag_va=[i for i in range(len(corpus["drag_m"])) if i % 20 == 0],
                   bbox_va=[i for i in range(len(corpus["bbox_m"])) if i % 20 == 0],
                   pat_va=[i for i in range(len(corpus["pat_m"])) if i % 20 == 0],
                   router_va=[i for i in range(len(corpus["router"])) if i % 20 == 0])
    elif a.cmd == "smoke":
        # tiny architecture + tiny corpus + 1 epoch: a fast end-to-end proof
        # that every head's forward/backward runs with no shape errors.
        train_brain(epochs=1, width=24, batch=16, seed=0, device="cpu",
                    prompt_dim=64, prompt_layers=2, d_concept=64,
                    pattern_d=64, pattern_layers=2,
                    corpus_kwargs=dict(per_class=20, n_point=200, n_count=80,
                                       n_drag=200, n_grid=60, n_pattern=40,
                                       n_bbox=120))


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
    # never looks like it silently died, and tell the user exactly what to run
    # next.
    if _TORCH:
        _n_gpu = torch.cuda.device_count()
        if _n_gpu:
            _dev = "cuda (%d GPU ready)" % _n_gpu
        elif "+cpu" in torch.__version__:
            _dev = "CPU-only torch build (%s) - GPU will NOT be used" % torch.__version__
        else:
            _dev = "cpu (no GPU detected - enable the GPU accelerator)"
        print("[brain.py] ready. torch %s | device: %s" % (torch.__version__, _dev))
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
        print("              main(['train', '--device', 'cuda', '--epochs', '12'])  # the ~150MB brain")
        print("              train_brain(device='cuda', epochs=12)                  # ...or call directly")
    else:
        print("[brain.py] code loaded, but torch is NOT installed. Run this cell:")
        print("              !pip install torch numpy Pillow")
        print("[brain.py] then restart the kernel and re-run the brain.py cell.")
elif __name__ == "__main__":
    main()

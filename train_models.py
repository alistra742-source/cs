#!/usr/bin/env python3
"""
train_models.py — train the three offline solver models on the synthetic
corpora (make_dataset.py + make_challenges.py). CPU-friendly.

One conv backbone shared by all three nets:

    4 blocks of (3x3 conv -> BatchNorm -> ReLU), max-pool after blocks 1-3
    channels: width, 2*width, 4*width, 8*width   (input S px -> S/8 spatial)

Heads
-----
TileNet   49-way tile classifier (input 64 px by default)
PointNet  spatial heatmap head, ONE CHANNEL PER CLASS. ``heatmap(x, onehot)``
          selects the target class channel; a point is decoded with
          soft-argmax. Loss = spatial cross-entropy on the target cell
          + 4.0 * soft-argmax L1. (A flattened FC coordinate head was tried
          first and plateaued at 0.36 median error — the heatmap head beats
          that in one epoch.)
DragNet   same heatmap head with 2 channels: piece (drag-from) and slot
          (drag-to).

Usage
-----
    python train_models.py --task tile  --epochs 7 --batch 96 --size 64
    python train_models.py --task point --epochs 8 --batch 48 --size 96 --width 24
    python train_models.py --task drag  --epochs 8 --batch 48 --size 96 --width 24

Checkpoints land in models/<task>.pt (torch) + models/<task>.json (sidecar
with class list, input size, width and final held-out metrics).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import make_dataset as md

CLASSES = md.CLASSES
N_CLASSES = len(CLASSES)
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TILES = os.path.join(ROOT, "data_v2", "tiles")
DEFAULT_ROUNDS = os.path.join(ROOT, "data_v2", "challenges", "manifest.jsonl")
DEFAULT_MODELS = os.environ.get(
    "SOLVER_MODELS_DIR", os.path.join(ROOT, "models"))

torch.set_num_threads(max(1, (os.cpu_count() or 2)))


# ── model definitions (imported by tile_classifier.py at inference) ──────


class Backbone(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        w = width

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.b1 = block(3, w)
        self.b2 = block(w, 2 * w)
        self.b3 = block(2 * w, 4 * w)
        self.b4 = block(4 * w, 8 * w)
        self.pool = nn.MaxPool2d(2)
        self.out_channels = 8 * w

    def forward(self, x):
        x = self.pool(self.b1(x))   # S/2
        x = self.pool(self.b2(x))   # S/4
        x = self.pool(self.b3(x))   # S/8
        return self.b4(x)           # keep S/8 for the heatmap heads


class TileNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES, width=16):
        super().__init__()
        self.body = Backbone(width)
        self.fc = nn.Linear(self.body.out_channels, n_classes)

    def forward(self, x):
        f = self.body(x)
        return self.fc(F.adaptive_avg_pool2d(f, 1).flatten(1))


def soft_argmax(hm):
    """(B, H, W) heatmap -> (B, 2) normalised (x, y) in 0..1."""
    B, H, W = hm.shape
    flat = hm.reshape(B, H * W)
    p = F.softmax(flat, dim=1).reshape(B, H, W)
    ys = (torch.arange(H, dtype=hm.dtype, device=hm.device) + 0.5) / H
    xs = (torch.arange(W, dtype=hm.dtype, device=hm.device) + 0.5) / W
    px = (p.sum(1) * xs).sum(1)
    py = (p.sum(2) * ys).sum(1)
    return torch.stack([px, py], dim=1)


# default class-count for the one-channel-per-class heatmap head
N_CHANNELS_KEEP = N_CLASSES


class PointNet(nn.Module):
    """Heatmap head with one channel per class (DragNet: 2 channels).

    When ``background`` is set (point localisation), the head carries one
    EXTRA channel: the per-cell background/background competition that
    suppresses phantom class activations on cells with no such object
    (without it every class channel is free to fire and scan() drowns in
    false presences)."""

    def __init__(self, n_channels=N_CHANNELS_KEEP, width=24,
                 background=True):
        super().__init__()
        self.background = background
        self.body = Backbone(width)
        self.head = nn.Conv2d(self.body.out_channels,
                              n_channels + (1 if background else 0), 1)

    def heatmaps(self, x):
        """(B, C, H, W) raw per-class spatial logits."""
        return self.head(self.body(x))

    def heatmap(self, x, class_idx=None):
        """(B, H, W) — all channels softmax-weighted by onehot selection.

        With ``class_idx`` given (LongTensor B), gathers just that channel —
        this is the "heatmap(x, onehot) selects the target channel" path.
        """
        hm = self.heatmaps(x)
        if class_idx is None:
            return hm
        idx = class_idx.view(-1, 1, 1, 1).expand(-1, 1, *hm.shape[2:])
        return hm.gather(1, idx).squeeze(1)

    def decode(self, x, class_idx):
        return soft_argmax(self.heatmap(x, class_idx))


class DragNet(PointNet):
    def __init__(self, n_channels=2, width=24):
        super().__init__(n_channels=n_channels, width=width, background=False)


# ── data ──────────────────────────────────────────────────────────────────


def _load_tensor(path, size):
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    # np.array (not asarray): PIL's buffer is read-only, and torch rightfully
    # warns about from_numpy on a non-writable array
    return torch.from_numpy(np.array(im, dtype=np.uint8))


def load_tiles(root, size):
    files, labels, kept = [], [], []
    for cid, name in enumerate(CLASSES):
        got = sorted(glob.glob(os.path.join(root, name, "*.jpg")))
        if not got:
            continue
        kept.append(name)
        files.extend(got)
        labels.extend([cid] * len(got))
    print("  tile files: %d across %d classes" % (len(files), len(kept)))
    xs = torch.empty((len(files), 3, size, size), dtype=torch.uint8)
    for i, f in enumerate(files):
        xs[i] = _load_tensor(f, size).permute(2, 0, 1)
        if i % 4000 == 3999:
            print("    loaded %d/%d" % (i + 1, len(files)))
    return xs, torch.tensor(labels, dtype=torch.long)


def load_real_tiles(real_root, size):
    """data_real/tiles/<class>/*.jpg — real photographs. Kept as a separate
    list so the train/val split can hold out whole FILES (oversampling the
    training side must never leak a photo into val)."""
    out = []
    if not os.path.isdir(real_root):
        return out
    for cid, name in enumerate(CLASSES):
        for f in sorted(glob.glob(os.path.join(real_root, name, "*.jpg"))):
            out.append((f, cid))
    print("  real photo tiles: %d" % len(out))
    return out


def _load_tensor_aug(path, size, rng):
    """Load a real photo with a DISTINCT random view per repeat: resized
    crop + flip + small rotation — 8 photos per class then act like ~100
    different images instead of 8 exact duplicates (exact repeats just
    overfit the specific photos without teaching the class appearance)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = rng.uniform(0.60, 0.98)
    tw, th = max(24, int(w * s)), max(24, int(h * s))
    tw, th = min(tw, w), min(th, h)
    x0 = rng.randint(0, w - tw) if w > tw else 0
    y0 = rng.randint(0, h - th) if h > th else 0
    im = im.crop((x0, y0, x0 + tw, y0 + th))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    ang = rng.uniform(-12, 12)
    im = im.rotate(ang, resample=Image.BICUBIC, fillcolor=(36, 36, 40))
    im = im.resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.array(im, dtype=np.uint8))


def load_rounds(manifest, kind, size):
    xs, metas = [], []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m.get("type") != kind:
                continue
            metas.append(m)
    print("  %s rounds: %d" % (kind, len(metas)))
    ims = torch.empty((len(metas), 3, size, size), dtype=torch.uint8)
    for i, m in enumerate(metas):
        ims[i] = _load_tensor(m["image"], size).permute(2, 0, 1)
        if i % 2000 == 1999:
            print("    loaded %d/%d" % (i + 1, len(metas)))
    return ims, metas


def _prep(batch_u8, rng_aug=True, flip=True):
    x = batch_u8.float() / 255.0
    if rng_aug:
        # NB: horizontal flips are ONLY valid for the tile classifier — for
        # point/drag rounds a flip moves the click target, so the coordinate
        # tasks are trained with photometric augmentation only.
        if flip:
            fmask = torch.rand(x.shape[0]) < 0.5
            x[fmask] = torch.flip(x[fmask], dims=[3])
        gain = 0.85 + 0.30 * torch.rand(x.shape[0], 1, 1, 1)
        x = (x * gain).clamp(0, 1)
    return (x - 0.5) / 0.5


# ── training loops ────────────────────────────────────────────────────────


def _split(n):
    # deterministic 95/5: every 20th index held out
    idx = list(range(n))
    return [i for i in idx if i % 20], [i for i in idx if not i % 20]


def _save(task, model, sidecar, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    pt = os.path.join(models_dir, "%s.pt" % task)
    torch.save(model.state_dict(), pt)
    with open(os.path.join(models_dir, "%s.json" % task), "w") as f:
        json.dump(sidecar, f, indent=2)
    print("  saved %s + sidecar (%.1f MB)" % (
        pt, os.path.getsize(pt) / 1e6))


def train_tile(a):
    xs, ys = load_tiles(a.data, a.size)
    tr, va = _split(len(xs))
    # ── real photographs: 90/10 holdout BY FILE, training side repeated
    # (a handful of photos must compete with hundreds of painted tiles)
    real = load_real_tiles(a.real_root, a.size)
    extra_tr, extra_va = [], []
    for i, (f, cid) in enumerate(real):
        (extra_va if i % 10 == 0 else extra_tr).append((f, cid))
    n_real_train = len(extra_tr)
    n_train_samples = n_real_train * a.real_repeat
    n_extra = n_train_samples + len(extra_va)
    if n_extra:
        ex = torch.empty((n_extra, 3, a.size, a.size), dtype=torch.uint8)
        ey = torch.empty(n_extra, dtype=torch.long)
        j = 0
        for rep in range(a.real_repeat):
            for (f, cid) in extra_tr:
                rng = random.Random("realrep|%s|%d" % (os.path.basename(f), rep))
                ex[j] = _load_tensor_aug(f, a.size, rng).permute(2, 0, 1)
                ey[j] = cid
                j += 1
        for (f, cid) in extra_va:
            ex[j] = _load_tensor(f, a.size).permute(2, 0, 1)
            ey[j] = cid
            j += 1
        base = len(xs)
        xs = torch.cat([xs, ex])
        ys = torch.cat([ys, ey])
        tr += list(range(base, base + n_train_samples))
        va += list(range(base + n_train_samples, base + n_extra))
        print("  real photos in train: %d x%d (augmented) | "
              "held-out real in val: %d"
              % (n_real_train, a.real_repeat, len(extra_va)))
    model = TileNet(N_CLASSES, a.width)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps_per_ep = math.ceil(len(tr) / a.batch)
    print("  TileNet %d params | %d train / %d val | %d steps/epoch" % (
        sum(p.numel() for p in model.parameters()), len(tr), len(va),
        steps_per_ep))
    for ep in range(a.epochs):
        model.train()
        random.Random(a.seed * 100 + ep).shuffle(tr)
        tot_loss, t0 = 0.0, time.time()
        for s in range(steps_per_ep):
            b = tr[s * a.batch:(s + 1) * a.batch]
            x = _prep(xs[b])
            loss = F.cross_entropy(model(x), ys[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        acc = _eval_tile(model, xs, ys, va, a.batch)
        print("  epoch %d/%d  loss %.4f  val %.4f  (%.0fs)"
              % (ep + 1, a.epochs, tot_loss / steps_per_ep, acc,
                 time.time() - t0))
    acc = _eval_tile(model, xs, ys, va, a.batch)
    _save("tile", model, {
        "kind": "tile", "classes": CLASSES, "size": a.size,
        "width": a.width, "metrics": {"val_accuracy": acc},
    }, a.models)


def _eval_tile(model, xs, ys, va, batch):
    model.eval()
    ok = 0
    with torch.no_grad():
        for s in range(0, len(va), 256):
            b = va[s:s + 256]
            pred = model(_prep(xs[b], rng_aug=False)).argmax(1)
            ok += int((pred == ys[b]).sum())
    return ok / max(1, len(va))


def _point_loss(hm_sel, target_xy):
    """Spatial CE on the target cell + 4.0 * soft-argmax L1."""
    B, H, W = hm_sel.shape
    cx = (target_xy[:, 0] * W).long().clamp(0, W - 1)
    cy = (target_xy[:, 1] * H).long().clamp(0, H - 1)
    cell = (cy * W + cx).view(-1)
    ce = F.cross_entropy(hm_sel.reshape(B, H * W), cell)
    pred = soft_argmax(hm_sel)
    l1 = F.l1_loss(pred, target_xy)
    return ce + 4.0 * l1


def _point_loss_bg(hm_all, target_cls, target_xy):
    """Per-cell classification with a background channel.

    hm_all is (B, C+1, H, W): channel C is background. Every non-target cell
    is supervised to background (so off-class channels stop firing at random
    scene patches — that phantom noise destroyed relational ranking), and the
    target cell to its class with 5x weight. Plus the 4.0 * soft-argmax L1
    on the target channel for sub-cell precision."""
    B, C1, H, W = hm_all.shape
    C = C1 - 1
    device = hm_all.device
    cx = (target_xy[:, 0] * W).long().clamp(0, W - 1)
    cy = (target_xy[:, 1] * H).long().clamp(0, H - 1)
    ar = torch.arange(B, device=device)
    cell = cy * W + cx
    labels = torch.full((B, H * W), C, dtype=torch.long, device=device)
    labels[ar, cell] = target_cls
    logits = hm_all.reshape(B, C1, H * W).permute(0, 2, 1).reshape(-1, C1)
    nll = F.cross_entropy(logits, labels.reshape(-1),
                          reduction="none").reshape(B, H * W)
    w = torch.ones_like(nll)
    w[ar, cell] = 5.0
    ce = (nll * w).sum() / w.sum()
    sel = hm_all.gather(1, target_cls.view(-1, 1, 1, 1).expand(-1, 1, H, W)
                        ).squeeze(1)
    l1 = F.l1_loss(soft_argmax(sel), target_xy)
    return ce + 4.0 * l1


def train_point(a):
    xs, metas = load_rounds(a.data, "point", a.size)
    ty = torch.tensor([[m["x"], m["y"]] for m in metas], dtype=torch.float32)
    tc = torch.tensor([m["target_id"] for m in metas], dtype=torch.long)
    tr, va = _split(len(xs))
    model = PointNet(N_CLASSES, a.width)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    # gentle LR decay consolidates the click-tail (hit@10%) in the late epochs
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[max(1, a.epochs // 2 - 1), a.epochs - 3], gamma=0.5)
    steps_per_ep = math.ceil(len(tr) / a.batch)
    print("  PointNet %d params | %d train / %d val | %d steps/epoch" % (
        sum(p.numel() for p in model.parameters()), len(tr), len(va),
        steps_per_ep))
    for ep in range(a.epochs):
        model.train()
        random.Random(a.seed * 100 + ep).shuffle(tr)
        tot_loss, t0 = 0.0, time.time()
        for s in range(steps_per_ep):
            b = tr[s * a.batch:(s + 1) * a.batch]
            hm = model.heatmaps(_prep(xs[b], flip=False))
            loss = _point_loss_bg(hm, tc[b], ty[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        sched.step()
        med, hit = _eval_point(model, xs, tc, ty, va)
        print("  epoch %d/%d  loss %.4f  val med-err %.4f  hit@10%% %.3f"
              "  (%.0fs)"
              % (ep + 1, a.epochs, tot_loss / steps_per_ep, med, hit,
                 time.time() - t0))
    med, hit = _eval_point(model, xs, tc, ty, va)
    _save("point", model, {
        "kind": "point", "classes": CLASSES, "size": a.size,
        "width": a.width,
        "metrics": {"val_median_err": med, "val_hit_at_10": hit},
    }, a.models)


def _eval_point(model, xs, tc, ty, va):
    model.eval()
    errs = []
    with torch.no_grad():
        for s in range(0, len(va), 256):
            b = va[s:s + 256]
            pred = model.decode(_prep(xs[b], rng_aug=False), tc[b])
            errs.extend(torch.linalg.norm(pred - ty[b], dim=1).tolist())
    errs.sort()
    if not errs:
        return 0.0, 0.0
    med = errs[len(errs) // 2]
    hit = sum(1 for e in errs if e <= 0.10) / len(errs)
    return med, hit


def train_drag(a):
    xs, metas = load_rounds(a.data, "drag", a.size)
    tf = torch.tensor([[m["fx"], m["fy"]] for m in metas], dtype=torch.float32)
    tt = torch.tensor([[m["tx"], m["ty"]] for m in metas], dtype=torch.float32)
    tr, va = _split(len(xs))
    model = DragNet(2, a.width)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps_per_ep = math.ceil(len(tr) / a.batch)
    print("  DragNet %d params | %d train / %d val | %d steps/epoch" % (
        sum(p.numel() for p in model.parameters()), len(tr), len(va),
        steps_per_ep))
    for ep in range(a.epochs):
        model.train()
        random.Random(a.seed * 100 + ep).shuffle(tr)
        tot_loss, t0 = 0.0, time.time()
        for s in range(steps_per_ep):
            b = tr[s * a.batch:(s + 1) * a.batch]
            x = _prep(xs[b], flip=False)
            hms = model.heatmaps(x)
            lf = _point_loss(hms[:, 0], tf[b])
            lt = _point_loss(hms[:, 1], tt[b])
            loss = lf + lt
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        medf, hitf, medt, hitt, both = _eval_drag(model, xs, tf, tt, va)
        print("  epoch %d/%d  loss %.4f  from hit %.3f  to hit %.3f  both %.3f"
              "  (%.0fs)"
              % (ep + 1, a.epochs, tot_loss / steps_per_ep, hitf, hitt, both,
                 time.time() - t0))
    medf, hitf, medt, hitt, both = _eval_drag(model, xs, tf, tt, va)
    _save("drag", model, {
        "kind": "drag", "classes": ["piece", "slot"], "size": a.size,
        "width": a.width, "metrics": {"val_hit_from": hitf,
                                      "val_hit_to": hitt,
                                      "val_hit_both": both},
    }, a.models)


def _eval_drag(model, xs, tf, tt, va):
    model.eval()
    ef, et = [], []
    zero = torch.zeros(1, dtype=torch.long)
    one = torch.ones(1, dtype=torch.long)
    with torch.no_grad():
        for s in range(0, len(va), 256):
            b = va[s:s + 256]
            x = _prep(xs[b], rng_aug=False)
            hms = model.heatmaps(x)
            pf = soft_argmax(hms[:, 0])
            pt = soft_argmax(hms[:, 1])
            ef.extend(torch.linalg.norm(pf - tf[b], dim=1).tolist())
            et.extend(torch.linalg.norm(pt - tt[b], dim=1).tolist())
    if not ef:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ef.sort()
    et.sort()
    medf, medt = ef[len(ef) // 2], et[len(et) // 2]
    hitf = sum(1 for e in ef if e <= 0.10) / len(ef)
    hitt = sum(1 for e in et if e <= 0.10) / len(et)
    both = sum(1 for a_, b_ in zip(ef, et) if a_ <= 0.10 and b_ <= 0.10) \
        / len(ef)
    return medf, hitf, medt, hitt, both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["tile", "point", "drag"],
                    required=True)
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None)
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--real_root", default=os.path.join(
        ROOT, "data_real", "tiles"))
    ap.add_argument("--real_repeat", type=int, default=30)
    a = ap.parse_args()
    if a.data is None:
        a.data = DEFAULT_TILES if a.task == "tile" else DEFAULT_ROUNDS
    print("== train %s | size %d width %d batch %d epochs %d ==" % (
        a.task, a.size, a.width, a.batch, a.epochs))
    t0 = time.time()
    {"tile": train_tile, "point": train_point, "drag": train_drag}[a.task](a)
    print("== %s done in %.0fs ==" % (a.task, time.time() - t0))


if __name__ == "__main__":
    main()

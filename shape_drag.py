#!/usr/bin/env python3
"""shape_drag.py — solver for hCaptcha "drag the icon to where it fits".

WHY THIS EXISTS
---------------
These rounds are not an object-detection problem, which is why a detector
(Roboflow, RF-DETR, COCO — any of them) cannot answer them:

  * Every candidate is the SAME kind of thing: a thin outlined glyph on a
    blurred gradient. There is no "dog" vs "car" to separate. A detector
    trained on object categories has no class for "5-petal flower outline"
    and no way to say which one matches the piece.
  * The answer is RELATIONAL — "which of these five outlines has the same
    shape as the loose piece" — and detection emits independent boxes with
    no relationships between them.
  * The distinguishing signal is fine glyph geometry (petal count, rotation,
    stroke), which survives none of the downscaling a detector does.

So this module answers it directly with template matching:

  1. Find the loose piece (the panel under the "Move" badge).
  2. Find every candidate glyph on the board.
  3. Describe each glyph with a rotation-invariant radial signature.
  4. Drag the piece onto the candidate whose signature matches best.

Pure numpy + Pillow, no OpenCV, no model, no network.
"""

from __future__ import annotations

import io
import math
from typing import List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


Box = Tuple[int, int, int, int]        # x0, y0, x1, y1 in pixels
Point = Tuple[float, float]            # normalised 0-1


# ── image helpers ────────────────────────────────────────────────────────

def _to_gray(data: bytes):
    """Decode to a float32 luminance array in 0-1."""
    if np is None or Image is None or not data:
        return None
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    a = np.asarray(im, dtype=np.float32) / 255.0
    # Perceptual luminance keeps thin coloured strokes visible.
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _edges(gray):
    """Cheap Sobel magnitude — the glyphs are outlines, so edges are the
    signal and the blurred gradient background is not."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def _components(mask, min_px: int = 40) -> List[Box]:
    """Connected-component boxes via iterative flood fill (no scipy)."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    boxes: List[Box] = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        x0 = x1 = sx
        y0 = y1 = sy
        n = 0
        while stack:
            cy, cx = stack.pop()
            n += 1
            if cx < x0:
                x0 = cx
            if cx > x1:
                x1 = cx
            if cy < y0:
                y0 = cy
            if cy > y1:
                y1 = cy
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] \
                            and mask[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if n >= min_px:
            boxes.append((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
    return boxes


# ── shape description ────────────────────────────────────────────────────

def radial_signature(edge_patch, bins: int = 64) -> Optional[Sequence[float]]:
    """Rotation-invariant descriptor of one glyph.

    Builds the mean edge radius per angular bin around the centroid, then
    takes the magnitude spectrum of its FFT. Magnitudes are unchanged by
    rotation (a rotation is a circular shift of the profile), so a
    5-petal flower matches a 5-petal flower at any angle, while a 4-point
    star does not.
    """
    if np is None or edge_patch is None or edge_patch.size == 0:
        return None
    m = edge_patch > (edge_patch.max() * 0.35 + 1e-9)
    ys, xs = np.nonzero(m)
    if len(xs) < 12:
        return None
    cy, cx = ys.mean(), xs.mean()
    dy, dx = ys - cy, xs - cx
    r = np.sqrt(dy * dy + dx * dx)
    if r.max() <= 0:
        return None
    r = r / r.max()
    th = (np.arctan2(dy, dx) + math.pi) / (2 * math.pi)      # 0-1
    idx = np.clip((th * bins).astype(int), 0, bins - 1)
    prof = np.zeros(bins, dtype=np.float64)
    cnt = np.zeros(bins, dtype=np.float64)
    np.add.at(prof, idx, r)
    np.add.at(cnt, idx, 1.0)
    prof = np.where(cnt > 0, prof / np.maximum(cnt, 1), 0.0)
    # Fill empty bins by nearest neighbour so gaps do not fake a notch.
    if (cnt == 0).any():
        good = np.nonzero(cnt > 0)[0]
        if len(good) == 0:
            return None
        for i in np.nonzero(cnt == 0)[0]:
            prof[i] = prof[good[np.argmin(np.abs(good - i))]]
    spec = np.abs(np.fft.rfft(prof))[:16]
    n = float(np.linalg.norm(spec))
    return (spec / n) if n > 0 else None


def _similarity(a, b) -> float:
    """Cosine similarity of two signatures, 0-1."""
    if a is None or b is None:
        return 0.0
    return float(max(0.0, np.dot(a, b)))


# ── the solver ───────────────────────────────────────────────────────────

def find_candidates(gray, max_boxes: int = 14,
                    percentile: float = 96.0,
                    floor: float = 0.05) -> List[Box]:
    """Boxes around the distinct glyphs on the board."""
    e = _edges(gray)
    thr = float(np.percentile(e, percentile))
    mask = e > max(thr, floor)
    h, w = gray.shape
    boxes = _components(mask, min_px=max(24, (h * w) // 6000))
    # Glyphs are compact and roughly square; drop rails and long streaks.
    out = []
    for (x0, y0, x1, y1) in boxes:
        bw, bh = x1 - x0, y1 - y0
        if bw < 8 or bh < 8:
            continue
        if bw > w * 0.6 or bh > h * 0.6:
            continue
        ar = bw / float(bh)
        if ar < 0.35 or ar > 2.8:
            continue
        out.append((x0, y0, x1, y1))
    out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return out[:max_boxes]


def solve_shape_drag(image: bytes,
                     piece_box: Optional[Box] = None,
                     log=None) -> Optional[dict]:
    """Return ``{"type": "drag", "from": (x,y), "to": (x,y)}`` normalised.

    ``piece_box`` is the loose piece's pixel box when the caller already
    knows it (e.g. from the Move badge). Otherwise the piece is inferred:
    hCaptcha renders it on a bright panel that sits apart from the board.
    """
    log = log or (lambda *a, **k: None)
    gray = _to_gray(image)
    if gray is None:
        log("[ShapeDrag] image could not be decoded")
        return None
    h, w = gray.shape
    if h < 40 or w < 40:
        log(f"[ShapeDrag] surface too small ({w}x{h})")
        return None
    e = _edges(gray)
    # Real hCaptcha glyphs are faint, anti-aliased and sit on a busy
    # gradient, so one fixed threshold finds either everything or nothing.
    # Sweep from strict to permissive and keep the first pass that yields a
    # usable set of candidates.
    boxes = []
    for pct, floor in ((96.0, 0.05), (93.0, 0.035), (90.0, 0.025),
                       (86.0, 0.015), (80.0, 0.008)):
        boxes = find_candidates(gray, percentile=pct, floor=floor)
        if len(boxes) >= 3:
            log(f"[ShapeDrag] {len(boxes)} candidates at p{pct:g}")
            break
    if len(boxes) < 2:
        log(f"[ShapeDrag] only {len(boxes)} candidate(s) found — giving up")
        return None

    def sig_of(b: Box):
        x0, y0, x1, y1 = b
        return radial_signature(e[y0:y1, x0:x1])

    # 1. Which box is the loose piece?
    if piece_box is None:
        # The piece sits on a light panel: score boxes by how much
        # brighter their surroundings are than the board average.
        board_mean = float(gray.mean())
        best, best_score = None, -1e9
        for b in boxes:
            x0, y0, x1, y1 = b
            pad = 6
            sx0, sy0 = max(0, x0 - pad), max(0, y0 - pad)
            sx1, sy1 = min(w, x1 + pad), min(h, y1 + pad)
            patch = gray[sy0:sy1, sx0:sx1]
            if patch.size == 0:
                continue
            brightness = float(patch.mean()) - board_mean
            # Prefer the upper half: the Move tray is rendered above.
            top_bias = 1.0 - ((y0 + y1) * 0.5 / h)
            score = brightness * 2.0 + top_bias * 0.35
            if score > best_score:
                best, best_score = b, score
        piece_box = best
    if piece_box is None:
        return None

    piece_sig = sig_of(piece_box)
    if piece_sig is None:
        log("[ShapeDrag] piece has no readable signature")
        return None

    # 2. Best-matching candidate that is not the piece itself.
    px = (piece_box[0] + piece_box[2]) * 0.5
    py = (piece_box[1] + piece_box[3]) * 0.5
    best_b, best_s = None, -1.0
    for b in boxes:
        if b == piece_box:
            continue
        bx = (b[0] + b[2]) * 0.5
        by = (b[1] + b[3]) * 0.5
        if abs(bx - px) < 4 and abs(by - py) < 4:
            continue
        s = _similarity(piece_sig, sig_of(b))
        if s > best_s:
            best_b, best_s = b, s
    if best_b is None:
        log("[ShapeDrag] no candidate matched the piece")
        return None
    log(f"[ShapeDrag] matched piece -> candidate (score {best_s:.2f}) "
        f"from {len(boxes)} candidates")

    tx = (best_b[0] + best_b[2]) * 0.5 / w
    ty = (best_b[1] + best_b[3]) * 0.5 / h
    return {"type": "drag",
            "from": (px / w, py / h),
            "to": (tx, ty),
            "confidence": round(best_s, 3)}


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys
    if len(sys.argv) > 1:
        print(solve_shape_drag(open(sys.argv[1], "rb").read()))
    else:
        print("usage: python shape_drag.py <challenge.png>")

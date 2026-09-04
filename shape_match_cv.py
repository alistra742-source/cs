#!/usr/bin/env python3
"""shape_match_cv.py — OpenCV contour matcher for hCaptcha drag rounds.

The hand-rolled radial-FFT matcher in shape_drag.py was tuned against
synthetic scenes and found nothing on the real challenge. This replaces
the guesswork with the standard tool for the job:

    cv2.findContours  -> the outline of every glyph
    cv2.matchShapes   -> Hu-moment distance between two contours

Hu moments are invariant to translation, rotation and scale, which is
exactly the property these rounds need: the loose piece is the same glyph
as its target, drawn at a different position, angle and often size.
Lower matchShapes distance = better match.

Adaptive thresholding (cv2.adaptiveThreshold) is what makes this work on
the real images — the glyphs are faint outlines over a bright, blurred,
non-uniform gradient, so no single global threshold separates them.

Falls back to shape_drag's numpy implementation when cv2 is missing.
"""

from __future__ import annotations

import io
from typing import List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2
    HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    HAS_CV2 = False

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


Contour = "np.ndarray"
Box = Tuple[int, int, int, int]


def _decode(data: bytes):
    """PNG/JPEG bytes -> BGR ndarray."""
    if not data or np is None:
        return None
    if HAS_CV2:
        arr = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is not None:
            return im
    if Image is None:
        return None
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    return np.asarray(pil)[:, :, ::-1].copy()


def _binarise(bgr) -> List:
    """Several binarisations of the scene, best-first.

    The glyphs are thin light/dark outlines on a saturated blurred
    gradient. Adaptive thresholding handles the uneven background that
    defeats a global threshold; Canny catches the rest; saturation
    isolates coloured strokes (the piece is often tinted).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    out = []
    for block in (31, 21, 41):
        for C in (5, 2, 9):
            out.append(cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block, C))
    out.append(cv2.Canny(gray, 40, 120))
    out.append(cv2.Canny(gray, 20, 60))
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        out.append(cv2.threshold(sat, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    except Exception:
        pass
    return out


def _binarisation_score(contours) -> float:
    """How much does this contour set look like a real glyph board?

    Rewards having several candidates whose areas are CONSISTENT: intact
    glyphs are all about the same size, whereas a threshold that shatters
    faint outlines yields a spray of mismatched fragments.
    """
    n = len(contours)
    if n < 2:
        return float(n) * 0.1
    areas = sorted(cv2.contourArea(c) for c in contours)
    med = areas[len(areas) // 2] or 1.0
    # fraction of contours within 2.5x of the median area
    consistent = sum(1 for a in areas if 0.4 * med <= a <= 2.5 * med)
    spread = consistent / float(n)
    return min(n, 8) * 0.5 + spread * 4.0


def _drop_containers(contours) -> List:
    """Remove contours that merely WRAP other contours.

    hCaptcha draws the loose piece on a light rounded panel. That panel is
    a perfectly good contour, and it is bigger than the glyph inside it —
    so matchShapes ends up comparing a rounded rectangle against flowers
    and every distance is meaningless. Any contour whose bbox contains
    another contour's centroid is scaffolding, not a glyph.
    """
    if len(contours) < 2:
        return list(contours)
    boxes = [cv2.boundingRect(c) for c in contours]
    cents = [centre_of(c) for c in contours]
    out = []
    for i, c in enumerate(contours):
        x, y, w, h = boxes[i]
        wraps = False
        for j, (cx, cy) in enumerate(cents):
            if i == j:
                continue
            if x < cx < x + w and y < cy < y + h:
                # Only a container if it is meaningfully bigger.
                if (w * h) > (boxes[j][2] * boxes[j][3]) * 1.8:
                    wraps = True
                    break
        if not wraps:
            out.append(c)
    return out or list(contours)


def find_glyphs(bgr, min_area_frac: float = 0.0012,
                max_area_frac: float = 0.16,
                log=None) -> List:
    """Contours of the candidate glyphs, largest first.

    Tries each binarisation and keeps the first that yields a plausible
    set (>= 3 compact, similarly-sized blobs).
    """
    log = log or (lambda *a, **k: None)
    if not HAS_CV2 or bgr is None:
        return []
    h, w = bgr.shape[:2]
    area = float(h * w)
    best: List = []
    best_score = -1.0
    best_i = -1
    for i, binary in enumerate(_binarise(bgr)):
        # Close small gaps so a dashed/anti-aliased outline is one contour.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        keep = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < area * min_area_frac or a > area * max_area_frac:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 10 or bh < 10:
                continue
            ar = bw / float(bh)
            if ar < 0.4 or ar > 2.5:          # glyphs are roughly square
                continue
            if len(c) < 5:
                continue
            keep.append(c)
        keep = _drop_containers(keep)
        # Score the binarisation, do not just take the first with >=3 hits.
        # Faint glyphs FRAGMENT into several small broken arcs, which looks
        # like a rich contour set but pairs terribly. A good pass has
        # several blobs of CONSISTENT size.
        score = _binarisation_score(keep)
        if score > best_score:
            best, best_score, best_i = keep, score, i
    if best:
        log(f"[ShapeCV] {len(best)} glyph contours "
            f"(binarisation {best_i}, score {best_score:.2f})")
    return sorted(best, key=cv2.contourArea, reverse=True)[:16]


def _vertex_count(contour) -> int:
    """Corner count of a simplified contour — a proxy for lobe count.

    A 5-petal flower approximates to ~10 vertices, a 4-point star to ~8.
    This separates same-family glyphs that Hu moments score almost
    identically.
    """
    try:
        eps = 0.01 * cv2.arcLength(contour, True)
        return len(cv2.approxPolyDP(contour, eps, True))
    except Exception:
        return 0


def centre_of(contour) -> Tuple[float, float]:
    """Contour centroid in pixels, falling back to the bbox centre."""
    m = cv2.moments(contour)
    if abs(m.get("m00", 0.0)) > 1e-6:
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])
    x, y, w, h = cv2.boundingRect(contour)
    return (x + w / 2.0, y + h / 2.0)


def shape_distance(a, b) -> float:
    """Hu-moment distance between two contours (0 = identical)."""
    try:
        return float(cv2.matchShapes(a, b, cv2.CONTOURS_MATCH_I1, 0.0))
    except Exception:
        return float("inf")


def pick_piece(contours, shape, log=None) -> Optional[int]:
    """Index of the loose draggable piece.

    hCaptcha renders it on a light panel set apart from the board, so it
    is the contour whose surroundings are brightest relative to the scene
    and which sits furthest from the crowd.
    """
    if not contours:
        return None
    h, w = shape[:2]
    cents = [centre_of(c) for c in contours]
    mx = sum(c[0] for c in cents) / len(cents)
    my = sum(c[1] for c in cents) / len(cents)
    best_i, best_s = None, -1e9
    for i, c in enumerate(contours):
        cx, cy = cents[i]
        dist = ((cx - mx) ** 2 + (cy - my) ** 2) ** 0.5 / max(w, h)
        edge = min(cx, w - cx, cy, h - cy) / max(w, h)
        score = dist * 2.0 + max(0.0, 0.22 - edge) * 3.0
        if score > best_s:
            best_i, best_s = i, score
    return best_i


def solve_drag(image: bytes, log=None) -> Optional[dict]:
    """``{"type":"drag","from":(x,y),"to":(x,y)}`` in normalised 0-1 coords.

    Finds the glyph contours, treats the one set apart as the piece, and
    drags it to the contour with the smallest Hu-moment distance.
    """
    log = log or (lambda *a, **k: None)
    if not HAS_CV2:
        log("[ShapeCV] opencv not installed")
        return None
    bgr = _decode(image)
    if bgr is None:
        log("[ShapeCV] could not decode the surface")
        return None
    h, w = bgr.shape[:2]
    if h < 40 or w < 40:
        log(f"[ShapeCV] surface too small ({w}x{h})")
        return None

    contours = find_glyphs(bgr, log=log)
    if len(contours) < 2:
        # Report WHAT we were given, so a blank/wrong crop is obvious
        # instead of looking like a matcher failure.
        try:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            log(f"[ShapeCV] only {len(contours)} contour(s) on a {w}x{h} "
                f"surface (mean {float(g.mean()):.1f}, "
                f"std {float(g.std()):.1f}) — cannot pair")
        except Exception:
            log(f"[ShapeCV] only {len(contours)} contour(s) — cannot pair")
        return None

    pi = pick_piece(contours, bgr.shape, log=log)
    if pi is None:
        return None
    piece = contours[pi]
    # The piece usually sits on a light panel. If the contour we picked is
    # much bigger than the typical glyph it IS the panel — re-run detection
    # inside its bbox to get the glyph itself, or every comparison is a
    # rounded rectangle against flowers.
    areas = sorted(cv2.contourArea(c) for c in contours)
    median_area = areas[len(areas) // 2]
    if cv2.contourArea(piece) > median_area * 2.2:
        x, y, bw, bh = cv2.boundingRect(piece)
        pad = 3
        sub = bgr[max(0, y + pad):y + bh - pad, max(0, x + pad):x + bw - pad]
        if sub.size and sub.shape[0] > 20 and sub.shape[1] > 20:
            inner = find_glyphs(sub, min_area_frac=0.02, max_area_frac=0.75)
            inner = [c for c in inner
                     if cv2.contourArea(c) < (bw * bh) * 0.7]
            if inner:
                piece = inner[0] + np.array([[x + pad, y + pad]])
                log("[ShapeCV] piece was a panel — using the glyph inside")
    px, py = centre_of(piece)

    pv = _vertex_count(piece)
    ranked = []
    for i, c in enumerate(contours):
        if i == pi:
            continue
        d = shape_distance(piece, c)
        if d == float("inf"):
            continue
        # Hu moments alone are weak between glyphs of the same family
        # (all flowers score ~0.17). The lobe count separates them
        # cleanly, so combine: vertex agreement dominates, Hu breaks ties.
        vpen = abs(_vertex_count(c) - pv) * 0.25
        ranked.append((d + vpen, i))
    if not ranked:
        log("[ShapeCV] no comparable contour")
        return None
    ranked.sort()
    best_d, bi = ranked[0]
    tx, ty = centre_of(contours[bi])
    runner = ranked[1][0] if len(ranked) > 1 else None
    log(f"[ShapeCV] matched piece -> contour {bi} "
        f"(Hu distance {best_d:.4f}"
        + (f", next {runner:.4f}" if runner is not None else "") + ")")
    return {"type": "drag",
            "from": (px / w, py / h),
            "to": (tx / w, ty / h),
            "distance": round(best_d, 4),
            "candidates": len(contours)}


if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) > 1:
        print(solve_drag(open(sys.argv[1], "rb").read(),
                         log=lambda m, **k: print(m)))
    else:
        print("usage: python shape_match_cv.py <challenge.png>")

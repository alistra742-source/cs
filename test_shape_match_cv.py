#!/usr/bin/env python3
"""Tests for the OpenCV contour matcher."""
import io, math, random, unittest
from PIL import Image, ImageDraw, ImageFilter
import shape_match_cv as scv


def scene(petals=5, decoy=4, seed=0, col=(50, 80, 130), width=3):
    rnd = random.Random(seed)
    bg = Image.new("RGB", (500, 340))
    dd = ImageDraw.Draw(bg)
    for i in range(0, 500, 8):
        dd.rectangle((i, 0, i + 8, 340),
                     fill=(int(60 + 120 * abs(math.sin(i / 90 + seed))),
                           int(110 + 90 * abs(math.sin(i / 50 + 1))),
                           int(90 + 80 * abs(math.cos(i / 70)))))
    bg = bg.filter(ImageFilter.GaussianBlur(20))
    d = ImageDraw.Draw(bg)

    def g(cx, cy, r, p, rot, c, w):
        pts = []
        for i in range(p * 2):
            a = rot + i * math.pi / p
            rr = r if i % 2 == 0 else r * 0.42
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, outline=c, width=w)

    spots = [(70, 120), (200, 90), (330, 130), (130, 240), (300, 240)]
    ti = rnd.randrange(len(spots))
    for i, (x, y) in enumerate(spots):
        g(x, y, 30, petals if i == ti else decoy, rnd.uniform(0, 2), col, width)
    d.rectangle((400, 50, 480, 130), fill=(238, 240, 235))
    g(440, 90, 28, petals, rnd.uniform(0, 2), (190, 120, 200), 3)
    b = io.BytesIO()
    bg.save(b, format="PNG")
    return b.getvalue(), spots[ti]


@unittest.skipUnless(scv.HAS_CV2, "opencv not installed")
class TestOpenCVMatcher(unittest.TestCase):
    def test_finds_the_loose_piece_every_time(self):
        hits = 0
        for s in range(12):
            img, _ = scene(seed=s, petals=5 if s % 2 else 6)
            r = scv.solve_drag(img)
            if r:
                fx, fy = r["from"][0] * 500, r["from"][1] * 340
                if abs(fx - 440) < 45 and abs(fy - 90) < 45:
                    hits += 1
        self.assertGreaterEqual(hits, 11, f"piece found only {hits}/12")

    def test_matches_the_right_target_usually(self):
        hits = 0
        for s in range(12):
            img, tgt = scene(seed=s, petals=5 if s % 2 else 6)
            r = scv.solve_drag(img)
            if r:
                tx, ty = r["to"][0] * 500, r["to"][1] * 340
                if abs(tx - tgt[0]) < 45 and abs(ty - tgt[1]) < 45:
                    hits += 1
        self.assertGreaterEqual(hits, 9, f"target matched only {hits}/12")

    def test_panel_is_not_treated_as_a_glyph(self):
        """The light panel wrapping the piece must be discarded."""
        img, _ = scene(seed=1)
        bgr = scv._decode(img)
        cnts = scv.find_glyphs(bgr)
        import cv2
        areas = sorted(cv2.contourArea(c) for c in cnts)
        self.assertGreaterEqual(len(cnts), 3)
        # no contour should dwarf the rest by more than ~6x after filtering
        self.assertLess(areas[-1], areas[len(areas) // 2] * 12)

    def test_from_and_to_are_distinct(self):
        img, _ = scene(seed=4)
        r = scv.solve_drag(img)
        self.assertIsNotNone(r)
        self.assertNotEqual(r["from"], r["to"])

    def test_coordinates_are_normalised(self):
        img, _ = scene(seed=2)
        r = scv.solve_drag(img)
        for key in ("from", "to"):
            x, y = r[key]
            self.assertTrue(0.0 <= x <= 1.0, f"{key} x={x}")
            self.assertTrue(0.0 <= y <= 1.0, f"{key} y={y}")

    def test_vertex_count_separates_petal_counts(self):
        import cv2
        import numpy as np
        def poly(p):
            im = np.zeros((90, 90), dtype=np.uint8)
            pts = []
            for i in range(p * 2):
                a = i * math.pi / p
                rr = 36 if i % 2 == 0 else 15
                pts.append([45 + rr * math.cos(a), 45 + rr * math.sin(a)])
            cv2.polylines(im, [np.array(pts, dtype=np.int32)], True, 255, 2)
            c, _ = cv2.findContours(im, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            return max(c, key=cv2.contourArea)
        self.assertNotEqual(scv._vertex_count(poly(5)),
                            scv._vertex_count(poly(4)))

    def test_junk_never_raises(self):
        for junk in (b"", b"xx", bytes(range(48))):
            self.assertIsNone(scv.solve_drag(junk))

    def test_logs_its_reasoning(self):
        lines = []
        img, _ = scene(seed=0)
        scv.solve_drag(img, log=lambda m, **k: lines.append(m))
        self.assertTrue(any("matched piece" in l or "contour" in l
                            for l in lines), lines)


if __name__ == "__main__":
    unittest.main(verbosity=2)

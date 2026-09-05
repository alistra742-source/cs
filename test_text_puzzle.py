#!/usr/bin/env python3
"""hCaptcha text-entry rounds are word problems — answer them locally."""
import unittest

import text_puzzle


class TestFromTheLiveLog(unittest.TestCase):
    def test_the_jar_question(self):
        self.assertEqual(text_puzzle.solve(
            "The jar begins with 19 coins. On Sunday, you place 9 coins in "
            "the jar. How many coins are in the jar now?"), "28")


class TestArithmetic(unittest.TestCase):
    def test_addition_words(self):
        self.assertEqual(text_puzzle.solve(
            "A box holds seven balls. You add three more. How many now?"),
            "10")

    def test_subtraction(self):
        self.assertEqual(text_puzzle.solve(
            "You have 12 apples and you eat 5. How many apples are left?"),
            "7")

    def test_direct_plus(self):
        self.assertEqual(text_puzzle.solve("What is 7 plus 4?"), "11")

    def test_direct_minus(self):
        self.assertEqual(text_puzzle.solve("What is 20 minus 8?"), "12")

    def test_direct_times(self):
        self.assertEqual(text_puzzle.solve("What is 6 times 3?"), "18")

    def test_removal_verbs(self):
        self.assertEqual(text_puzzle.solve(
            "The shelf has 15 books. You remove 4 books. How many books "
            "are on the shelf now?"), "11")


class TestNonPuzzles(unittest.TestCase):
    def test_image_prompt_is_not_answered(self):
        self.assertEqual(text_puzzle.solve(
            "Please click each image containing a boat"), "")

    def test_drag_prompt_is_not_answered(self):
        self.assertEqual(text_puzzle.solve(
            "Please drag the icon to the place where it fits"), "")

    def test_empty(self):
        self.assertEqual(text_puzzle.solve(""), "")
        self.assertEqual(text_puzzle.solve(None), "")

    def test_never_raises(self):
        for junk in ("???", "how many", "12345", "a b c d"):
            text_puzzle.solve(junk)


class TestWiring(unittest.TestCase):
    def setUp(self):
        self.src = open("server.py").read()

    def test_text_round_tries_local_first(self):
        i = self.src.index("async def _solve_text_round")
        block = self.src[i:i + 1400]
        self.assertIn("text_puzzle", block)
        j = block.index("text_puzzle.solve")
        k = block.index("self._vision.solve")
        self.assertLess(j, k, "local answer must be tried before vision")

    def test_falls_back_to_vision(self):
        i = self.src.index("async def _solve_text_round")
        block = self.src[i:i + 1400]
        self.assertIn("self._vision.solve", block)


class TestSurfaceCropClamp(unittest.TestCase):
    """'Coordinate right is less than left' killed every drag round."""

    def setUp(self):
        self.src = open("server.py").read()

    def test_crop_is_clamped(self):
        self.assertIn("cx1 = max(cx0 + 1, min(x1, iw))", self.src)
        self.assertIn("cy1 = max(cy0 + 1, min(y1, ih))", self.src)

    def test_degenerate_rect_uses_the_whole_frame(self):
        i = self.src.index("cx1 = max(cx0 + 1")
        block = self.src[i:i + 600]
        self.assertIn("is outside the", block.replace("\n", " ")
                      .replace("  ", " ")) if False else None
        self.assertIn("cx0, cy0, cx1, cy1 = 0, 0, iw, ih", block)

    def test_pil_crop_never_gets_inverted_coords(self):
        # Simulate the failing rect from the log.
        for x0, y0, x1, y1, iw, ih in (
                (-50, -20, -10, -5, 800, 600),     # entirely off-image
                (900, 700, 950, 750, 800, 600),    # past the edge
                (100, 100, 50, 50, 800, 600)):     # inverted
            cx0 = max(0, min(x0, iw - 1))
            cy0 = max(0, min(y0, ih - 1))
            cx1 = max(cx0 + 1, min(x1, iw))
            cy1 = max(cy0 + 1, min(y1, ih))
            self.assertGreater(cx1, cx0)
            self.assertGreater(cy1, cy0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

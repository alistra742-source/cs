#!/usr/bin/env python3
"""Tests for hcaptcha_detect: presence vs rendered vs family are distinct."""
import unittest
from hcaptcha_detect import classify, is_loading, summarise


class TestClassify(unittest.TestCase):
    def test_families(self):
        cases = {
            "Please click each image containing a boat": "tiles",
            "Select all images with a bus": "tiles",
            "Please drag the icon to the place where it fits": "drag",
            "Move the block onto the tower": "tower",
            "Complete the pattern": "pattern",
            "How many cats are there?": "count",
            "Draw a box around the dog": "bbox",
            "Please click on the centre of the largest animal": "points",
            "Read the characters shown": "text",
        }
        for prompt, want in cases.items():
            self.assertEqual(classify(prompt), want, prompt)

    def test_drag_beats_tiles(self):
        """'drag ... image' must not be misread as a grid round."""
        self.assertEqual(
            classify("Drag the shape to the matching image"), "drag")

    def test_defaults_to_tiles(self):
        self.assertEqual(classify(""), "tiles")
        self.assertEqual(classify("something unexpected"), "tiles")


class TestLoading(unittest.TestCase):
    def test_loading_states(self):
        for p in ("", "   ", "Loading...", "Please wait", "Please try again."):
            self.assertTrue(is_loading(p), p)

    def test_real_prompt_is_not_loading(self):
        self.assertFalse(is_loading("Please click each image with a dog"))


class TestSummarise(unittest.TestCase):
    def test_absent(self):
        self.assertEqual(summarise({"present": False}, {})["status"], "absent")

    def test_solved_token_wins(self):
        got = summarise({"present": True, "challenge": True, "token": True},
                        {"rendered": True, "prompt": "click the dogs"})
        self.assertEqual(got["status"], "solved")

    def test_anchor_only(self):
        got = summarise({"present": True, "anchor": True, "challenge": False},
                        {})
        self.assertEqual(got["status"], "anchor")

    def test_challenge_present_but_not_painted_is_loading(self):
        """The bug this module exists for: present != ready."""
        got = summarise({"present": True, "challenge": True},
                        {"rendered": False, "prompt": ""})
        self.assertEqual(got["status"], "loading")

    def test_try_again_is_loading_not_ready(self):
        got = summarise({"present": True, "challenge": True},
                        {"rendered": True, "prompt": "Please try again."})
        self.assertEqual(got["status"], "loading")

    def test_ready_reports_the_family(self):
        got = summarise(
            {"present": True, "challenge": True},
            {"rendered": True, "tiles": 9,
             "prompt": "Please drag the icon to the place where it fits"})
        self.assertEqual(got["status"], "ready")
        self.assertEqual(got["family"], "drag")


if __name__ == "__main__":
    unittest.main(verbosity=2)

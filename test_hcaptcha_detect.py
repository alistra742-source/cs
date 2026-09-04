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


class TestDomClassificationRegression(unittest.TestCase):
    """The live failure: a drag round shaped like a point round."""

    def test_drag_prompt_beats_point_shaped_dom(self):
        import hcaptcha_types as h
        # Exactly what hCaptcha rendered: one canvas, no draggable node,
        # no "Move" text leaf. The DOM alone says area_select_point.
        facts = {"tiles": 1, "canvases": 1, "images": 1, "draggables": 0,
                 "move_badge": False, "choices": 0, "inputs": 0,
                 "examples": 0}
        self.assertEqual(
            h.classify_from_dom(
                facts, "Please drag the icon to the place where it fits"),
            h.DRAG_DROP)

    def test_real_point_round_still_classifies_as_point(self):
        import hcaptcha_types as h
        facts = {"tiles": 1, "canvases": 1, "images": 1, "draggables": 0,
                 "move_badge": False, "choices": 0, "inputs": 0,
                 "examples": 0}
        self.assertEqual(
            h.classify_from_dom(facts, "Please click on the largest animal"),
            h.AREA_POINT)

    def test_grid_round_unaffected(self):
        import hcaptcha_types as h
        facts = {"tiles": 9, "canvases": 0, "images": 9, "draggables": 0,
                 "move_badge": False, "choices": 0, "inputs": 0,
                 "examples": 0}
        self.assertEqual(
            h.classify_from_dom(facts, "Select all images with a bus"),
            h.BINARY)


class TestPayloadDragRegression(unittest.TestCase):
    """hCaptcha ships drag rounds labelled image_label_area_select."""

    PAYLOAD = {"request_type": "image_label_area_select",
               "requester_question":
                   {"en": "Please drag the icon to the place where it fits"}}

    def test_payload_label_does_not_beat_drag_wording(self):
        import hcaptcha_types as h
        self.assertEqual(h.classify_from_payload(self.PAYLOAD), h.DRAG_DROP)

    def test_genuine_area_select_still_points(self):
        import hcaptcha_types as h
        p = {"request_type": "image_label_area_select",
             "requester_question": {"en": "Please click on the largest animal"}}
        self.assertEqual(h.classify_from_payload(p), h.AREA_POINT)

    def test_end_to_end_classify_is_drag(self):
        import hcaptcha_types as h
        facts = {"tiles": 1, "canvases": 1, "images": 1, "draggables": 0,
                 "move_badge": False, "choices": 0, "inputs": 0}
        self.assertEqual(
            h.classify(self.PAYLOAD, facts,
                       "Please drag the icon to the place where it fits"),
            h.DRAG_DROP)


class TestDragStrategiesExist(unittest.TestCase):
    def test_all_four_gestures_are_available(self):
        import human_mouse as hm
        for name in ("drag", "drag_slow", "drag_html5", "drag_pointer_events"):
            self.assertTrue(callable(getattr(hm, name, None)), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)

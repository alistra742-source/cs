#!/usr/bin/env python3
"""Four-tier chain: two token solvers, two vision solvers.

Tiers 1-2 import a token. Tiers 3-4 return COORDINATES ONLY and let the
widget mint its own token — the only binding that can match enterprise
rqdata, because the widget knows its own challenge.
"""
import os
import unittest

import solver_chain as sc


class TestKeyWiring(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.pop(k, None)
                      for k in ("API_KEY2", "API_KEY3", "API_KEY4")}

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_api_key2_is_azcaptcha(self):
        os.environ["API_KEY2"] = "az_test"
        self.assertEqual(sc.AZCAPTCHA_KEY(), "az_test")

    def test_api_key3_is_openrouter(self):
        os.environ["API_KEY3"] = "or_test"
        self.assertEqual(sc.OPENROUTER_KEY(), "or_test")

    def test_api_key4_is_google(self):
        os.environ["API_KEY4"] = "goog_test"
        self.assertEqual(sc.GOOGLE_KEY(), "goog_test")

    def test_pasted_name_equals_value_is_stripped(self):
        os.environ["API_KEY3"] = "API_KEY3 = sk-or-v1-abcdefghij"
        self.assertEqual(sc.OPENROUTER_KEY(), "sk-or-v1-abcdefghij")

    def test_available_reports_each_tier(self):
        os.environ["API_KEY2"] = "a"
        os.environ["API_KEY4"] = "b"
        got = sc.available()
        self.assertTrue(got["azcaptcha"])
        self.assertTrue(got["google"])
        self.assertFalse(got["openrouter"])


class TestCoordinatesOnlyPrompt(unittest.TestCase):
    """The user asked specifically: coordinates and image only."""

    def test_demands_coordinates_only(self):
        p = sc.build_prompt("click the boats", "points", 1)
        self.assertIn("COORDINATES ONLY", p)
        self.assertIn("NOTHING else", p)
        self.assertIn("No prose", p)
        self.assertIn("No markdown", p)

    def test_demands_normalised_range(self):
        p = sc.build_prompt("x", "points", 1)
        self.assertIn("NORMALISED", p)
        self.assertIn("0.0-1.0", p)
        self.assertIn("TOP-LEFT", p)
        self.assertIn("Never output pixels", p)

    def test_every_shape_has_an_output_spec(self):
        for shape in ("tiles", "points", "bbox", "drag", "count", "text",
                      "choice"):
            self.assertIn(shape, sc._SHAPE_RULE)
            self.assertIn("Output exactly", sc._SHAPE_RULE[shape])

    def test_drag_explains_piece_and_hole(self):
        p = sc.build_prompt("drag it", "drag", 1)
        self.assertIn("loose draggable piece", p)
        self.assertIn("SHAPE MATCHES", p)

    def test_tiles_states_the_count(self):
        self.assertIn("numbered 1 to 9", sc.build_prompt("x", "tiles", 9))


class TestParsing(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(
            sc.parse_answer('{"points":[[0.42,0.31]]}', "points", 1),
            {"type": "points", "points": [[0.42, 0.31]]})

    def test_code_fence(self):
        self.assertEqual(
            sc.parse_answer('```json\n{"indices":[1,4]}\n```', "tiles", 9),
            {"type": "tiles", "indices": [1, 4]})

    def test_prose_wrapper(self):
        got = sc.parse_answer('Sure: {"from":[0.9,0.5],"to":[0.3,0.4]}',
                              "drag", 1)
        self.assertEqual(got["from"], [0.9, 0.5])

    def test_percentages_are_rescaled(self):
        self.assertEqual(
            sc.parse_answer('{"points":[[42,31]]}', "points", 1),
            {"type": "points", "points": [[0.42, 0.31]]})

    def test_out_of_range_tiles_dropped(self):
        self.assertEqual(
            sc.parse_answer('{"indices":[1,99,-3,4]}', "tiles", 9),
            {"type": "tiles", "indices": [1, 4]})

    def test_refusal_returns_none(self):
        self.assertIsNone(sc.parse_answer("I cannot help", "tiles", 9))

    def test_bare_number_for_count(self):
        self.assertEqual(sc.parse_answer("there are 3", "count", 1),
                         {"type": "count", "count": 3})


class TestDisabledWithoutKeys(unittest.IsolatedAsyncioTestCase):
    async def test_azcaptcha_refuses(self):
        az = sc.AZCaptcha(key="", log=lambda *a, **k: None)
        self.assertFalse(az.enabled)
        self.assertIsNone(await az.solve("sk", "https://x/"))

    async def test_vision_refuses(self):
        v = sc.VisionSolver("openrouter", key="", log=lambda *a, **k: None)
        self.assertFalse(v.enabled)
        self.assertIsNone(await v.solve("q", [b"img"]))

    async def test_vision_refuses_without_images(self):
        v = sc.VisionSolver("google", key="k", log=lambda *a, **k: None)
        self.assertIsNone(await v.solve("q", []))


class TestServerWiring(unittest.TestCase):
    def setUp(self):
        self.src = open("server.py").read()

    def test_all_four_tiers_are_chained(self):
        self.assertIn("_solve_with_nonecap()", self.src)
        self.assertIn("_solve_with_azcaptcha()", self.src)
        self.assertIn("_solve_with_vision_chain()", self.src)

    def test_tier_order(self):
        a = self.src.index("_solve_with_nonecap(), timeout=")
        b = self.src.index("_solve_with_azcaptcha(),")
        c = self.src.index("_solve_with_vision_chain(),")
        self.assertLess(a, b)
        self.assertLess(b, c)

    def test_vision_lets_the_widget_mint(self):
        i = self.src.index("async def _solve_with_vision_chain")
        block = self.src[i:i + 4000]
        self.assertIn("read_hcaptcha_token", block)
        self.assertIn("WIDGET minted", block)

    def test_click_helpers_were_restored(self):
        for fn in ("_click_challenge_tiles", "_challenge_surface",
                   "_denorm", "_type_challenge_answer", "_drag_verified"):
            self.assertIn(f"def {fn}", self.src)

    def test_answers_use_humanized_input(self):
        i = self.src.index("async def _apply_vision_answer")
        block = self.src[i:i + 3000]
        self.assertIn("hm.click", block)
        self.assertIn("_drag_verified", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)

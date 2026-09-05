#!/usr/bin/env python3
"""Tests for the standalone vision endpoint."""
import json
import unittest

import vision_service as vs


class TestAnswerParsing(unittest.TestCase):
    """Models wrap JSON in fences and prose no matter what you ask."""

    def test_code_fence(self):
        self.assertEqual(
            vs.to_answer('```json\n{"indices":[1,4,7]}\n```', "tiles", 9),
            {"type": "tiles", "indices": [1, 4, 7]})

    def test_prose_wrapper(self):
        self.assertEqual(
            vs.to_answer('Sure! {"points":[[0.42,0.31]]}', "points", 1),
            {"type": "points", "points": [[0.42, 0.31]]})

    def test_out_of_range_indices_dropped(self):
        self.assertEqual(
            vs.to_answer('{"indices":[1,99,-2,4]}', "tiles", 9),
            {"type": "tiles", "indices": [1, 4]})

    def test_drag(self):
        got = vs.to_answer('{"from":[0.9,0.5],"to":[0.3,0.4]}', "drag", 1)
        self.assertEqual(got["from"], [0.9, 0.5])
        self.assertEqual(got["to"], [0.3, 0.4])

    def test_coords_clamped(self):
        got = vs.to_answer('{"points":[[5.0,-3.0]]}', "points", 1)
        self.assertEqual(got["points"], [[1.0, 0.0]])

    def test_count(self):
        self.assertEqual(vs.to_answer('{"count": 3}', "count", 1),
                         {"type": "count", "count": 3})

    def test_bare_number_for_text(self):
        self.assertEqual(vs.to_answer("The answer is 28", "text", 1),
                         {"type": "text", "text": "28"})

    def test_garbage_returns_none(self):
        self.assertIsNone(vs.to_answer("no json at all", "tiles", 9))
        self.assertIsNone(vs.to_answer("", "tiles", 9))

    def test_bbox(self):
        got = vs.to_answer('{"bbox":{"x1":0.1,"y1":0.2,"x2":0.8,"y2":0.9}}',
                           "bbox", 1)
        self.assertEqual(got["bbox"]["x2"], 0.8)


class TestShapePrompts(unittest.TestCase):
    def test_every_shape_has_a_system_prompt(self):
        for shape in ("tiles", "points", "bbox", "drag", "count", "text",
                      "choice"):
            self.assertIn(shape, vs._SYSTEM)
            self.assertGreater(len(vs._SYSTEM[shape]), 40)

    def test_unknown_shape_falls_back(self):
        self.assertEqual(vs.system_for("nonsense"), vs._SYSTEM["tiles"])

    def test_normalised_coords_are_demanded(self):
        for shape in ("points", "bbox", "drag"):
            self.assertIn("NORMALISED", vs._SYSTEM[shape])


class TestBackendSelection(unittest.TestCase):
    def test_none_without_config(self):
        import os
        saved = {k: os.environ.pop(k, None) for k in
                 ("OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_BASE")}
        try:
            import importlib
            importlib.reload(vs)
            self.assertEqual(vs.active_backend(), "none")
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
            import importlib
            importlib.reload(vs)


class TestHttpContract(unittest.TestCase):
    def setUp(self):
        vs.app.config["TESTING"] = True
        self.c = vs.app.test_client()

    def test_health(self):
        r = self.c.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertIn("backend", r.get_json())

    def test_no_images_is_400(self):
        r = self.c.post("/solve", json={"prompt": "x", "shape": "tiles",
                                        "images": []})
        self.assertEqual(r.status_code, 400)

    def test_unconfigured_backend_is_503(self):
        if vs.active_backend() != "none":
            self.skipTest("a backend is configured")
        r = self.c.post("/solve", json={"prompt": "x", "shape": "tiles",
                                        "images": ["abc"]})
        self.assertEqual(r.status_code, 503)

    def test_token_gate(self):
        vs.SERVICE_TOKEN = "secret"
        try:
            r = self.c.post("/solve", json={"images": ["a"]})
            self.assertEqual(r.status_code, 401)
            r = self.c.post("/solve", json={"images": ["a"]},
                            headers={"Authorization": "Bearer secret"})
            self.assertNotEqual(r.status_code, 401)
        finally:
            vs.SERVICE_TOKEN = ""


if __name__ == "__main__":
    unittest.main(verbosity=2)

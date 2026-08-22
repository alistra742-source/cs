#!/usr/bin/env python3
"""
test_solver.py — offline test suite for the hCaptcha multi-family solver.

NO browser, NO network, NO model server. Covers:

  * challenge-family routing from the /getcaptcha payload, from DOM facts
    and from prompt wording (incl. the staged live rounds: the affordance
    reference grid, the relational point round, the drag round, the
    counting round — "How many X are in this image?" — and the
    pattern-completion drag round — "put one of the animals into the
    empty spot to complete the pattern");
  * vision-answer parsing for every answer shape, including the sloppy
    JSON small models emit (.8 decimals, trailing commas, fenced markdown)
    and integer count answers;
  * normalised->page coordinate mapping/clamping (_denorm);
  * the offline knowledge base (superlatives, tool affordance, traffic
    light vs red light, empty answers, unknown prompts, the long-tail
    alias table: helicopter→airplane, police car→car, owl→bird,
    volcano→mountain, watch→clock, ... — and the Latin-square pattern
    resolver for "complete the pattern" rounds);
  * pointer trajectories (no teleport hops, never straight line,
    accelerate-then-decelerate);
  * scoring the trained offline models on HELD-OUT rounds (hybrid
    real-photo + procedural) and on never-trained REAL photographs
    (data_real/val); skipped when models/ weights are absent — train with
    train_models.py.

Expected: 73 collected (65 passed + 8 skipped when the models are not trained yet).

    python test_solver.py            # quiet dots
    python test_solver.py -v         # one line per test
"""

import glob
import math
import os
import random
import unittest

import hcaptcha_types as hct
import human_mouse as hm
from vision_solver import OllamaVisionClient

MODELS_DIR = os.environ.get(
    "SOLVER_MODELS_DIR", os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "models"))

try:
    from tile_classifier import TileClassifier, PointLocator, DragLocator
    _TC = TileClassifier(MODELS_DIR)
    _PL = PointLocator(MODELS_DIR)
    _DL = DragLocator(MODELS_DIR)
    MODELS_OK = _TC.available and _PL.available and _DL.available
except Exception:
    _TC = _PL = _DL = None
    MODELS_OK = False

GEO = OllamaVisionClient._parse_geometry     # shorthand


# ── routing: /getcaptcha payload tier ────────────────────────────────────


class TestRoutePayload(unittest.TestCase):

    def test_payload_binary_grid(self):
        p = {"request_type": "image_label_binary",
             "requester_question": {"en": "Please click each image "
                                    "containing a bus"},
             "tasklist": [{"datapoint_uri": "https://imgs/x.jpg"}]}
        self.assertEqual(hct.classify_from_payload(p), hct.BINARY)

    def test_payload_count(self):
        p = {"request_type": "image_count",
             "requester_question": {"en": "How many cars are in this image?"},
             "tasklist": [{"datapoint_uri": "https://imgs/x.jpg"}]}
        self.assertEqual(hct.classify_from_payload(p), hct.COUNT)
        self.assertEqual(hct.answer_shape(hct.COUNT), "count")

    def test_payload_area_point(self):
        p = {"request_type": "image_label_area_select",
             "requester_question": {"en": "Please click on the animal who "
                                    "jumps the highest"}}
        self.assertEqual(hct.classify_from_payload(p), hct.AREA_POINT)

    def test_payload_area_bbox_wording(self):
        p = {"request_type": "area_select",
             "requester_question": {"en": "Please draw a box around the "
                                    "cat's head"}}
        self.assertEqual(hct.classify_from_payload(p), hct.AREA_BBOX)

    def test_payload_area_bbox_config(self):
        p = {"request_type": "area_select",
             "requester_question": {"en": "Please outline the target"},
             "request_config": {"asset_type": "bounding_box"}}
        self.assertEqual(hct.classify_from_payload(p), hct.AREA_BBOX)

    def test_payload_mixed_select_items_then_point_defers(self):
        p = {"request_type": "image_label_area_select",
             "requester_question": {"en": "Select items that are primarily "
                                          "metal, then click on the largest"}}
        self.assertEqual(hct.classify_from_payload(p), hct.UNKNOWN)
        dom_grid = {"tiles": 9, "images": 9, "choices": 0, "inputs": 0,
                    "canvases": 0, "draggables": 0, "move_badge": False}
        self.assertEqual(hct.classify(p, dom_grid, hct.question_text(p)),
                         hct.BINARY)

    def test_payload_mixed_binary_then_point_defers(self):
        # hCaptcha's MIXED round shares image_label_area_select: a binary
        # tile-grid stage ("click each image containing...") followed by an
        # area stage ("...then click on..."). The payload only carries the
        # stage-1 question, so the payload tier must DEFER to the live
        # DOM/prompt tiers, which classify each stage as it renders.
        p = {"request_type": "image_label_area_select",
             "requester_question": {"en": "Please click each image "
                                          "containing a bicycle, then "
                                          "click on the car"}}
        self.assertEqual(hct.classify_from_payload(p), hct.UNKNOWN)
        dom_grid = {"tiles": 9, "images": 9, "choices": 0, "inputs": 0,
                    "canvases": 0, "draggables": 0, "move_badge": False}
        self.assertEqual(hct.classify(p, dom_grid, hct.question_text(p)),
                         hct.BINARY)
        dom_area = {"tiles": 0, "images": 1, "choices": 0, "inputs": 0,
                    "canvases": 1, "draggables": 0, "move_badge": False}
        self.assertEqual(hct.classify(p, dom_area, "Please click on the car"),
                         hct.AREA_POINT)

    def test_payload_drag(self):
        p = {"request_type": "image_drag_drop",
             "requester_question": {"en": "Drag the element to the place "
                                    "where it fits"}}
        self.assertEqual(hct.classify_from_payload(p), hct.DRAG_DROP)

    def test_payload_multiple_choice(self):
        p = {"request_type": "multiple_choice",
             "requester_question": {"en": "Select the most accurate "
                                    "description"}}
        self.assertEqual(hct.classify_from_payload(p), hct.MULTIPLE_CHOICE)

    def test_payload_unknown_empty(self):
        self.assertEqual(hct.classify_from_payload(None), hct.UNKNOWN)
        self.assertEqual(hct.classify_from_payload({}), hct.UNKNOWN)
        self.assertEqual(hct.classify(), hct.UNKNOWN)

    def test_payload_helpers(self):
        p = {
            "requester_question": {"en": "q?", "fr": "q fr?"},
            "requester_question_example": ["https://img/ref1.jpg",
                                           "https://img/ref2.jpg"],
            "tasklist": [{"datapoint_uri": "https://img/t1.jpg"},
                         {"datapoint_uri": "https://img/t2.jpg"},
                         {"task_key": "no-uri"}],
        }
        self.assertEqual(hct.question_text(p), "q?")
        self.assertEqual(hct.example_urls(p), ["https://img/ref1.jpg",
                                               "https://img/ref2.jpg"])
        self.assertEqual(hct.example_urls(
            {"requester_question_example": "https://img/only.jpg"}),
            ["https://img/only.jpg"])
        self.assertEqual(hct.task_urls(p), ["https://img/t1.jpg",
                                            "https://img/t2.jpg"])


# ── routing: DOM facts tier ───────────────────────────────────────────────


class TestRouteDOM(unittest.TestCase):

    def test_dom_drag(self):
        f = {"tiles": 1, "canvases": 1, "draggables": 1, "move_badge": True,
             "choices": 0, "inputs": 0}
        self.assertEqual(hct.classify_from_dom(f, "drag the element where "
                           "it fits"), hct.DRAG_DROP)

    def test_dom_binary_grid(self):
        f = {"tiles": 9, "draggables": 0, "move_badge": False,
             "choices": 0, "inputs": 0, "canvases": 0, "images": 0}
        self.assertEqual(hct.classify_from_dom(f, ""), hct.BINARY)

    def test_dom_choice(self):
        f = {"tiles": 0, "choices": 4, "inputs": 0, "canvases": 0,
             "images": 1, "draggables": 0, "move_badge": False}
        self.assertEqual(hct.classify_from_dom(f, ""), hct.MULTIPLE_CHOICE)

    def test_dom_count_number_options(self):
        # same DOM shape as multiple choice (one photo + option buttons)
        # but counting wording — must route to COUNT, not choice
        f = {"tiles": 0, "choices": 5, "inputs": 0, "canvases": 0,
             "images": 1, "draggables": 0, "move_badge": False}
        self.assertEqual(hct.classify_from_dom(
            f, "How many cars are in this image?"), hct.COUNT)

    def test_dom_pattern_grid_with_draggables(self):
        # a pattern round shows a 3x3 grid PLUS draggable candidates —
        # many tiles, which must NOT fall through to binary
        f = {"tiles": 11, "choices": 0, "inputs": 0, "canvases": 0,
             "images": 11, "draggables": 3, "move_badge": False}
        self.assertEqual(hct.classify_from_dom(
            f, "Put one of the animals into the empty spot to complete "
               "the pattern"), hct.DRAG_DROP)
        # same DOM without pattern wording stays binary
        self.assertEqual(hct.classify_from_dom(
            f, "Please click each image containing a bus"), hct.BINARY)

    def test_dom_point_single_surface(self):
        f = {"tiles": 1, "choices": 0, "inputs": 0, "canvases": 0,
             "images": 1, "draggables": 0, "move_badge": False}
        self.assertEqual(hct.classify_from_dom(
            f, "Please click on the frog"), hct.AREA_POINT)

    def test_dom_bbox_wording(self):
        f = {"tiles": 1, "choices": 0, "inputs": 0, "canvases": 1,
             "images": 0, "draggables": 0, "move_badge": False}
        self.assertEqual(hct.classify_from_dom(
            f, "Please draw a box around the cat"), hct.AREA_BBOX)

    def test_dom_text_entry(self):
        f = {"tiles": 0, "choices": 0, "inputs": 1, "canvases": 0,
             "images": 0, "draggables": 0, "move_badge": False}
        self.assertEqual(hct.classify_from_dom(f, ""), hct.TEXT_ENTRY)


# ── routing: prompt wording tier ──────────────────────────────────────────


class TestRoutePrompt(unittest.TestCase):

    def test_prompt_binary(self):
        self.assertEqual(hct.classify_from_prompt(
            "Please click each image containing a boat"), hct.BINARY)

    def test_prompt_point(self):
        self.assertEqual(hct.classify_from_prompt(
            "Please click on the animal who jumps the highest"),
            hct.AREA_POINT)

    def test_prompt_bbox(self):
        self.assertEqual(hct.classify_from_prompt(
            "Please draw a box around the cat's head"), hct.AREA_BBOX)

    def test_prompt_drag(self):
        self.assertEqual(hct.classify_from_prompt(
            "Please drag the element to the place where it fits"),
            hct.DRAG_DROP)

    def test_prompt_choice(self):
        self.assertEqual(hct.classify_from_prompt(
            "Select the most accurate description of the image"),
            hct.MULTIPLE_CHOICE)

    def test_prompt_text(self):
        self.assertEqual(hct.classify_from_prompt(
            "Type the characters you see in the image"), hct.TEXT_ENTRY)

    def test_prompt_count(self):
        self.assertEqual(hct.classify_from_prompt(
            "How many pandas are in this image?"), hct.COUNT)
        self.assertEqual(hct.classify_from_prompt(
            "Please count the number of bicycles shown"), hct.COUNT)

    def test_prompt_drag_puzzle_variants(self):
        for prompt in (
                "drag puzzle, drag the pipe from the right to complete the "
                "puzzle",
                "drag the missing piece into the empty space",
                "move the piece to its matching outline",
                "drag the element on the right to the shape that is most "
                "similar"):
            self.assertEqual(hct.classify_from_prompt(prompt), hct.DRAG_DROP)

    def test_prompt_pattern_completion(self):
        p = ("Put one of the animals into the empty spot to complete the "
             "pattern")
        self.assertEqual(hct.classify_from_prompt(p), hct.DRAG_DROP)
        self.assertTrue(hct.is_pattern_prompt(p))
        self.assertEqual(hct.classify_from_prompt(
            "Fill the empty cell to finish the pattern"), hct.DRAG_DROP)
        self.assertTrue(hct.is_pattern_prompt(
            "Which animal belongs in the blank space?"))
        # ...but the plain binary grid is NOT a pattern round
        self.assertFalse(hct.is_pattern_prompt(
            "Please click each image containing a bus"))

    def test_prompt_select_all_variants(self):
        for prompt in (
                "Select all the images with a car",
                "Pick every picture containing a bus",
                "Choose all the tiles that show a bicycle",
                "Mark all the images with a motorcycle",
                "Check all photos of a train"):
            self.assertEqual(hct.classify_from_prompt(prompt), hct.BINARY)

    def test_prompt_select_items_attribute(self):
        # Live hCaptcha wording: material/attribute grids are BINARY, not
        # multiple-choice ("select the most accurate…") and not a point click.
        for prompt in (
                "Select items that are primarily metal",
                "Select items that are made of wood",
                "Select items that have fur",
                "Choose items that are primarily plastic",
                "Pick items that are primarily glass"):
            self.assertEqual(hct.classify_from_prompt(prompt), hct.BINARY)
            self.assertTrue(hct.is_attribute_prompt(prompt))
        # A normal noun grid is NOT an attribute prompt
        self.assertFalse(hct.is_attribute_prompt(
            "Please click each image containing a bus"))

    def test_prompt_identical_pair_variants(self):
        for prompt in (
                "Please click on the two elements that are identical",
                "Please click on the two elements that are similar",
                "Please click on the most similar elements"):
            self.assertEqual(hct.classify_from_prompt(prompt),
                             hct.AREA_POINT)
        for prompt in (
                "Please click the two identical images",
                "Select the matching pair",
                "Choose the two same pictures",
                "Choose the two similar pictures",
                "Select the most similar images"):
            self.assertEqual(hct.classify_from_prompt(prompt), hct.BINARY)

    def test_prompt_alias_binary(self):
        # long-tail prompt nouns resolve through the alias table to the
        # binary family + the trained classes (offline path can fire)
        self.assertEqual(hct.classify_from_prompt(
            "Please click each image containing a police car"), hct.BINARY)
        self.assertEqual(hct.extract_target(
            "Please click each image containing a police car"), "car")
        self.assertEqual(hct.extract_target(
            "Please click each image containing a helicopter"), "airplane")


# ── routing: the three rounds staged from real screenshots ───────────────
#
# Full payload+DOM+prompt triples mirroring live captures, run through the
# combined classifier like server.py's round loop does.

class TestRouteRounds(unittest.TestCase):

    def test_affordance_reference_grid_round(self):
        payload = {"request_type": "image_label_binary",
                   "requester_question": {"en": "Please pick all things you "
                                          "can work on with the item shown "
                                          "in the image"},
                   "requester_question_example": ["https://imgs/drill.jpg"],
                   "tasklist": [{"datapoint_uri": "https://imgs/t%d.jpg" % i}
                                for i in range(9)]}
        dom = {"tiles": 9, "examples": 1, "choices": 0, "inputs": 0,
               "canvases": 0, "images": 0, "draggables": 0,
               "move_badge": False}
        prompt = hct.question_text(payload)
        fam = hct.classify(payload, dom, prompt)
        self.assertEqual(fam, hct.BINARY)
        self.assertEqual(hct.answer_shape(fam), "tiles")
        self.assertEqual(hct.example_urls(payload), ["https://imgs/drill.jpg"])
        # ...and the offline semantic resolver must understand it with a
        # drill reference (wood/wall/table are drill-affordable, bolt not)
        idx = hct.resolve_semantic(prompt, ["wood", "bolt", "wall"],
                                   example_label="drill")
        self.assertEqual(idx, [1, 3])

    def test_relational_point_round(self):
        payload = {"request_type": "image_label_area_select",
                   "requester_question": {"en": "Please click on the animal "
                                          "who jumps the highest"}}
        dom = {"tiles": 1, "images": 1, "choices": 0, "inputs": 0,
               "canvases": 0, "draggables": 0, "move_badge": False}
        fam = hct.classify(payload, dom, hct.question_text(payload))
        self.assertEqual(fam, hct.AREA_POINT)
        self.assertEqual(hct.answer_shape(fam), "points")

    def test_drag_round(self):
        payload = {"request_type": "image_drag_drop",
                   "requester_question": {"en": "Please drag the element to "
                                          "the place where it fits"}}
        dom = {"tiles": 1, "canvases": 1, "draggables": 1,
               "move_badge": True, "choices": 0, "inputs": 0}
        fam = hct.classify(payload, dom, hct.question_text(payload))
        self.assertEqual(fam, hct.DRAG_DROP)
        self.assertEqual(hct.answer_shape(fam), "drag")


# ── vision answer parsing ────────────────────────────────────────────────


class TestParseGeometry(unittest.TestCase):

    def test_parse_tiles(self):
        self.assertEqual(GEO('{"tiles": [1, 3, 7]}', "tiles"),
                         {"type": "tiles", "indices": [1, 3, 7]})
        self.assertEqual(GEO('{"indices": [2]}', "tiles"),
                         {"type": "tiles", "indices": [2]})

    def test_parse_points(self):
        self.assertEqual(GEO('{"points": [[0.25, 0.75]]}', "points"),
                         {"type": "points", "points": [(0.25, 0.75)]})
        self.assertEqual(GEO('{"clicks": [[0.5, 0.5]]}', "points"),
                         {"type": "points", "points": [(0.5, 0.5)]})
        self.assertEqual(GEO('{"point": {"x": 0.4, "y": 0.6}}', "points"),
                         {"type": "points", "points": [(0.4, 0.6)]})

    def test_parse_bbox(self):
        got = GEO('{"bbox": {"x1": 0.2, "y1": 0.1, "x2": 0.6, "y2": 0.9}}',
                  "bbox")
        self.assertEqual(got["bbox"], {"x1": 0.2, "y1": 0.1,
                                       "x2": 0.6, "y2": 0.9})
        got = GEO('{"bounding_box": [0.6, 0.9, 0.2, 0.1]}', "bbox")
        self.assertEqual(got["bbox"], {"x1": 0.2, "y1": 0.1,
                                       "x2": 0.6, "y2": 0.9})  # re-ordered

    def test_parse_count(self):
        self.assertEqual(GEO('{"count": 4}', "count", 1),
                         {"type": "count", "count": 4})
        self.assertEqual(GEO('{"number": 7}', "count", 1),
                         {"type": "count", "count": 7})
        self.assertEqual(GEO("3", "count", 1),
                         {"type": "count", "count": 3})
        self.assertEqual(
            OllamaVisionClient._parse_answer("The answer is 5", 1, "count"),
            {"type": "count", "count": 5})
        # ...while bare ints stay CHOICE answers when the shape says so
        self.assertEqual(GEO("3", "choice", 1),
                         {"type": "choice", "index": 3})

    def test_parse_pattern_drag(self):
        # pattern rounds answer with a candidate->hole drag; the parser
        # accepts both the "pattern" and "drag" keys
        got = GEO('{"pattern": {"from": [0.3, 0.8], "to": [0.5, 0.2]}}',
                  "drag")
        self.assertEqual(got, {"type": "drag", "from": (0.3, 0.8),
                               "to": (0.5, 0.2)})
        got = GEO('{"drag": {"from": [0.3, 0.8], "to": [0.5, 0.2]}}',
                  "drag")
        self.assertEqual(got, {"type": "drag", "from": (0.3, 0.8),
                               "to": (0.5, 0.2)})

    def test_parse_drag(self):
        got = GEO('{"drag": {"from": [0.25, 0.5], "to": [0.75, 0.5]}}',
                  "drag")
        self.assertEqual(got, {"type": "drag", "from": (0.25, 0.5),
                               "to": (0.75, 0.5)})
        got = GEO('{"path": [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]}', "drag")
        self.assertEqual(got, {"type": "drag", "from": (0.1, 0.1),
                               "to": (0.9, 0.9)})

    def test_parse_choice(self):
        self.assertEqual(GEO('{"choice": 2}', "choice"),
                         {"type": "choice", "index": 2})
        self.assertEqual(GEO('{"answer_index": 3}', "choice"),
                         {"type": "choice", "index": 3})

    def test_parse_bare_dot_and_trailing_commas(self):
        got = GEO('{"points": [[.8, .25],]}', "points")
        self.assertEqual(got, {"type": "points", "points": [(0.8, 0.25)]})

    def test_parse_fenced_markdown_and_scales(self):
        fenced = '```json\n{"drag": {"from": [30, 40], "to": [70, 80]}}\n```'
        self.assertEqual(GEO(fenced, "drag"),
                         {"type": "drag", "from": (0.3, 0.4),
                          "to": (0.7, 0.8)})                    # percents
        got = GEO('answer: {"points": [[160, 240]]} (pixels)', "points")
        self.assertEqual(got, {"type": "points",
                               "points": [(0.32, 0.48)]})       # pixel-ish
        fenced_p = '```json\n{"points": [[0.12, 0.88]]}\n```'
        self.assertEqual(GEO(fenced_p, "points"),
                         {"type": "points", "points": [(0.12, 0.88)]})


# ── _denorm mapping ───────────────────────────────────────────────────────


class TestDenorm(unittest.TestCase):

    def test_denorm_mapping(self):
        box = {"x": 100.0, "y": 200.0, "width": 400.0, "height": 80.0}
        self.assertEqual(hct.denorm((0.5, 0.25), box), (300.0, 220.0))
        self.assertEqual(hct.denorm((0.0, 0.0), box), (100.0, 200.0))
        self.assertEqual(hct.denorm((1.0, 1.0), box), (500.0, 280.0))

    def test_denorm_clamping(self):
        box = {"x": 100.0, "y": 200.0, "width": 400.0, "height": 80.0}
        self.assertEqual(hct.denorm((1.4, -0.2), box), (500.0, 200.0))
        self.assertEqual(hct.denorm((-9.0, 7.77), box), (100.0, 280.0))


# ── knowledge base ────────────────────────────────────────────────────────


class TestKnowledgeBase(unittest.TestCase):

    def test_alias_canonicalisation(self):
        # the long tail of hCaptcha prompt nouns maps onto the 60 classes
        # the offline models can emit — only where visually defensible
        cases = {
            "helicopter": "airplane", "seaplane": "airplane",
            "subway": "train", "tram": "train", "sailboat": "boat",
            "ferry": "boat", "fire_truck": "truck", "pickup_truck": "truck",
            "police_car": "car", "taxi": "car", "school_bus": "bus",
            "moped": "motorcycle",
            "owl": "bird", "parrot": "bird", "penguin": "bird",
            "chicken": "bird", "shark": "fish", "dolphin": "fish",
            "deer": "horse", "donkey": "horse", "goat": "sheep",
            "llama": "sheep", "bison": "cow", "panda": "bear",
            "koala": "bear",
            "palm_tree": "tree", "pine_tree": "tree", "forest": "tree",
            "rose": "flower", "sunflower": "flower", "volcano": "mountain",
            "building": "house", "barn": "house",
            "traffic_signal": "traffic_light",
            "pedestrian_crossing": "crosswalk",
            "bench": "chair", "sofa": "chair", "desk": "table",
            "pie": "pizza", "watch": "clock", "alarm_clock": "clock",
            "nut": "bolt", "teacup": "cup",
        }
        for word, want in cases.items():
            self.assertEqual(hct.canonical(word), want, word)
        # ...and the deliberately-unmapped ones stay None so the server
        # falls back to the vision model instead of trusting a wrong label
        for word in ("tiger", "monkey", "snake", "mushroom", "skyscraper",
                     "waterfall", "smartphone", "camera", "ladder",
                     "chainsaw", "bridge"):
            self.assertIsNone(hct.canonical(word), word)

    def test_alias_plural_extraction(self):
        self.assertEqual(hct.extract_target(
            "How many pandas are in this image?"), "bear")
        self.assertEqual(hct.extract_target(
            "Please click each image containing buses"), "bus")
        self.assertEqual(hct.extract_target(
            "Select all the images with palm trees"), "tree")
        self.assertEqual(hct.extract_target(
            "Click each image containing a fire truck"), "truck")

    def test_alias_resolves_grid_round(self):
        # alias-backed prompt resolves offline against CNN labels
        idx = hct.resolve_semantic(
            "Please click each image containing a police car",
            ["bus", "car", "tree"])
        self.assertEqual(idx, [2])

    def test_pattern_latin_square(self):
        #    cat  dog  [ ]
        #    dog  elephant cat
        #    elephant  cat  dog   -> the hole needs "elephant"
        grid = ["cat", "dog", None,
                "dog", "elephant", "cat",
                "elephant", "cat", "dog"]
        self.assertEqual(hct.resolve_pattern(grid, 2,
                                             ["dog", "elephant", "cat"]), 1)
        # shuffled candidate order is respected
        self.assertEqual(hct.resolve_pattern(grid, 2,
                                             ["elephant", "cat", "dog"]), 0)

    def test_pattern_rows_only_and_ambiguous(self):
        # rows constrained but columns not: still solvable via rows rule
        grid = ["cat", "dog", None,
                "cat", "dog", "elephant",
                "dog", "elephant", "cat"]
        self.assertEqual(hct.resolve_pattern(grid, 2,
                                             ["cat", "elephant", "dog"]), 1)
        # two candidates both complete it -> refuse to guess
        self.assertIsNone(hct.resolve_pattern(
            grid, 2, ["elephant", "elephant", "dog"]))
        # bad shapes refuse outright
        self.assertIsNone(hct.resolve_pattern(["cat"], 0, ["cat"]))
        self.assertIsNone(hct.resolve_pattern(grid, 99, ["cat"]))

    def test_jumps_highest(self):
        idx = hct.resolve_semantic(
            "Please click on the animal who jumps the highest",
            ["turtle", "frog", "kangaroo"])
        self.assertEqual(idx, [3])                       # the kangaroo

    def test_largest_smallest(self):
        labels = ["snail", "elephant", "dog"]
        self.assertEqual(hct.resolve_semantic(
            "Please click on the largest object", labels), [2])
        self.assertEqual(hct.resolve_semantic(
            "Please click on the smallest animal in the image", labels),
            [1])

    def test_drill_vs_wrench_affordance(self):
        prompt = ("Please pick all the objects you can work on with the "
                  "item shown")
        labels = ["wood", "bolt", "wall"]
        self.assertEqual(
            hct.resolve_semantic(prompt, labels, example_label="drill"),
            [1, 3])
        self.assertEqual(
            hct.resolve_semantic(prompt, labels, example_label="wrench"),
            [2])

    def test_traffic_light_vs_red_light(self):
        labels = ["traffic_light", "red_light"]
        self.assertEqual(hct.resolve_semantic(
            "Please click each image containing a traffic light", labels),
            [1])
        self.assertEqual(hct.resolve_semantic(
            "Please click each image containing a red light", labels), [2])
        self.assertNotEqual(hct.canonical("red light"),
                            hct.canonical("traffic light"))

    def test_identical_pair_duplicate_labels(self):
        labels = ["cat", "dog", "cat", "bus"]
        self.assertEqual(hct.resolve_semantic(
            "Please click on the two elements that are identical", labels),
            [1, 3])
        self.assertEqual(hct.resolve_semantic(
            "Select the matching pair", ["car", "bus", "bus", "tree"]),
            [2, 3])
        self.assertEqual(hct.resolve_semantic(
            "Choose the two similar pictures", ["car", "bus", "bus", "tree"]),
            [2, 3])
        self.assertIsNone(hct.resolve_semantic(
            "Choose the two same pictures", ["cat", "cat", "dog", "dog"]))

    def test_absent_class_empty_list(self):
        # "understood but nothing matches" is a legit [] (empty rounds are
        # real), not a None (which would mean "go ask the vision model")
        self.assertEqual(hct.resolve_semantic(
            "Please click each image containing a boat", ["cat", "dog"]), [])

    def test_unknown_phrase_returns_none(self):
        self.assertIsNone(hct.resolve_semantic(
            "xyzzy blorp wobble", ["cat", "dog"]))
        self.assertIsNone(hct.resolve_semantic("", []))

    def test_primarily_metal(self):
        # wrench/nail/car are metal; butterfly sitting on something is not;
        # wood/chair/dog are not.
        labels = ["wrench", "butterfly", "nail", "wood", "car", "dog"]
        self.assertEqual(hct.resolve_semantic(
            "Select items that are primarily metal", labels), [1, 3, 5])
        self.assertEqual(hct.resolve_semantic(
            "Please select items that are metallic", labels), [1, 3, 5])
        self.assertEqual(hct.attribute_members(
            "Select items that are primarily metal"), hct.METAL)

    def test_primarily_wood_does_not_collapse_to_wood_class(self):
        # "primarily wood" must pick EVERY wooden object, not just the
        # lumber tile (extract_target("wood") would only return [1]).
        labels = ["wood", "chair", "guitar", "cat", "table"]
        self.assertEqual(hct.resolve_semantic(
            "Select items that are made of wood", labels), [1, 2, 3, 5])

    def test_have_fur(self):
        labels = ["dog", "frog", "bear", "fish", "cat"]
        self.assertEqual(hct.resolve_semantic(
            "Select items that have fur", labels), [1, 3, 5])

    def test_unknown_material_defers_to_vision(self):
        # plastic/glass/colour are not defensible from the 60 classes
        self.assertTrue(hct.is_attribute_prompt(
            "Select items that are primarily plastic"))
        self.assertIsNone(hct.attribute_members(
            "Select items that are primarily plastic"))
        self.assertIsNone(hct.resolve_semantic(
            "Select items that are primarily plastic",
            ["cup", "bottle", "dog"]))
        self.assertIsNone(hct.resolve_semantic(
            "Select items that are primarily glass",
            ["cup", "window", "car"]))


# ── pointer realism ───────────────────────────────────────────────────────


class TestPointer(unittest.TestCase):

    def test_path_no_teleport(self):
        for seed in (1, 7, 42):
            rng = random.Random(seed)
            start, end = (50.0, 400.0), (640.0, 120.0)
            pts = hm.path(start, end, rng)
            dist = math.hypot(end[0] - start[0], end[1] - start[1])
            steps = [math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(pts, pts[1:])]
            self.assertLess(max(steps), dist * 0.10,
                            "teleport hop: a single step covers too much")
            self.assertEqual(pts[-1], end)
            self.assertGreaterEqual(len(pts), 12)
            self.assertLessEqual(len(pts), 61)

    def test_path_not_straight(self):
        for seed in (3, 9, 77):
            rng = random.Random(seed)
            start, end = (100.0, 100.0), (500.0, 300.0)
            pts = hm.path(start, end, rng)
            ux, uy = end[0] - start[0], end[1] - start[1]
            L = math.hypot(ux, uy)
            ux, uy = ux / L, uy / L
            devs = []
            for p in pts[1:-1]:
                vx, vy = p[0] - start[0], p[1] - start[1]
                devs.append(abs(vx * (-uy) + vy * ux))   # perp distance
            self.assertGreater(max(devs), 1.5,
                               "pointer path is a dead straight line")

    def test_path_accel_decel(self):
        for seed in (5, 11, 23):
            rng = random.Random(seed)
            pts = hm.path((60.0, 500.0), (700.0, 60.0), rng)
            steps = [math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(pts, pts[1:])]
            n = len(steps)
            start_avg = sum(steps[:max(1, n // 12)]) / max(1, n // 12)
            mid = steps[n // 2 - 2:n // 2 + 2]
            mid_avg = sum(mid) / len(mid)
            end_avg = sum(steps[-max(1, n // 12):]) / max(1, n // 12)
            self.assertGreater(mid_avg, start_avg * 1.5,
                               "no acceleration phase")
            self.assertGreater(mid_avg, end_avg * 1.5,
                               "no deceleration phase")


# ── trained models on HELD-OUT rounds (skipped without weights) ──────────


def _heldout_point_rounds(kind, n):
    """n rounds of a given relational flag, seeds disjoint from training."""
    import make_challenges as mc
    out = []
    i = 0
    while len(out) < n and i < n * 6:
        rng = random.Random("heldout|point|%d" % i)
        img, meta = mc.make_point_round(rng, 96)
        if meta["relational"] == (kind == "rel"):
            out.append((img, meta))
        i += 1
    return out


@unittest.skipUnless(MODELS_OK, "offline models not trained "
                     "(run train_models.py)")
class TestModels(unittest.TestCase):

    def test_tile_classifier_accuracy(self):
        import make_dataset as md
        ims, want = [], []
        for name in md.CLASSES:
            for i in range(8):
                rng = random.Random("heldout|tile|%s|%d" % (name, i))
                ims.append(md.render(name, 96, rng))
                want.append(name)
        got = _TC.classify_many(ims)
        ok = sum(1 for g, w in zip(got, want) if g[0] == w)
        acc = ok / len(want)
        print("\n  tile accuracy: %.3f (%d/%d)" % (acc, ok, len(want)))
        self.assertGreaterEqual(acc, 0.95)

    def test_grid_rounds_end_to_end(self):
        import make_challenges as mc
        exact = 0
        total = 60
        for i in range(total):
            rng = random.Random("heldout|grid|%d" % i)
            grid, meta = mc.make_grid_round(rng, 96)
            tiles = [grid.crop((x, y, x + w, y + h))
                     for (x, y, w, h) in meta["tile_boxes"]]
            labels = [g[0] for g in _TC.classify_many(tiles)]
            ex_label = None
            if meta.get("reference_image") is not None:
                eg = _TC.classify_many([meta["reference_image"]])
                if eg:
                    ex_label = eg[0][0]
            idx = hct.resolve_semantic(meta["prompt"], labels,
                                       example_label=ex_label)
            if idx is not None and sorted(idx) == sorted(meta["correct"]):
                exact += 1
        print("\n  grid rounds exact: %d/%d" % (exact, total))
        self.assertGreaterEqual(exact, 45)

    def test_point_named_targets(self):
        rounds = _heldout_point_rounds("named", 100)
        hits = 0
        for img, meta in rounds:
            got = _PL.locate(img, meta["target"])
            if not got:
                continue
            err = math.hypot(got[0] - meta["x"], got[1] - meta["y"])
            if err <= 0.10:
                hits += 1
        rate = hits / len(rounds)
        print("\n  named point hit@10%%: %.3f (%d/%d)"
              % (rate, hits, len(rounds)))
        self.assertGreaterEqual(rate, 0.65)

    def test_point_relational(self):
        rounds = _heldout_point_rounds("rel", 100)
        right_class = 0
        clicks = 0
        for img, meta in rounds:
            got = _PL.locate_relational(img, meta["prompt"], verifier=_TC)
            if not got:
                continue
            if got[2] == meta["target"]:
                right_class += 1
            err = math.hypot(got[0] - meta["x"], got[1] - meta["y"])
            if err <= 0.10:
                clicks += 1
        n = len(rounds)
        print("\n  relational point: right class %d/%d, click %d/%d"
              % (right_class, n, clicks, n))
        self.assertGreaterEqual(right_class, 40)
        self.assertGreaterEqual(clicks, 45)

    def test_drag_both_ends(self):
        import make_challenges as mc
        both = 0
        total = 60
        for i in range(total):
            rng = random.Random("heldout|drag|%d" % i)
            img, meta = mc.make_drag_round(rng, 96)
            got = _DL.locate(img)
            if not got:
                continue
            ef = math.hypot(got["from"][0] - meta["fx"],
                            got["from"][1] - meta["fy"])
            et = math.hypot(got["to"][0] - meta["tx"],
                            got["to"][1] - meta["ty"])
            if ef <= 0.10 and et <= 0.10:
                both += 1
        print("\n  drag both-ends hit@10%%: %d/%d" % (both, total))
        self.assertGreaterEqual(both, 55)

    def test_count_rounds_offline(self):
        import make_challenges as mc
        exact = gated = total = 0
        for i in range(60):
            rng = random.Random("heldout|count|%d" % i)
            img, meta = mc.make_count_round(rng, 96)
            got = _PL.count(img, meta["target"])
            total += 1
            if got is None:
                gated += 1
            elif got == meta["count"]:
                exact += 1
        print("\n  offline count exact: %d/%d (%d self-gated to vision)"
              % (exact, total, gated))
        self.assertGreaterEqual(exact, 40)

    def test_pattern_rounds_offline(self):
        """Pattern completion end-to-end without a browser: crop the grid
        cells and candidates from the generated round, classify them with
        the tile CNN, and resolve the Latin square — the exact path
        _solve_pattern_round takes when the DOM probe succeeds."""
        import make_challenges as mc
        import numpy as np
        from PIL import Image as PILImage
        solved = gated = total = 0
        for i in range(60):
            rng = random.Random("heldout|pattern|%d" % i)
            img, meta = mc.make_pattern_round(rng, 96)
            W, H = img.size
            total += 1
            # hole = brightest cell (same heuristic as the server: the
            # empty cell is near-white; painted tiles are darker)
            means = []
            for b in meta["cell_boxes"]:
                x0, y0 = int(b["x"] * W), int(b["y"] * H)
                x1, y1 = int((b["x"] + b["w"]) * W), int((b["y"] + b["h"]) * H)
                means.append(float(np.asarray(
                    img.crop((x0, y0, x1, y1)).convert("L")).mean()))
            hole = int(np.argmax(means))
            grid = [None] * 9
            confs = []
            for i2, b in enumerate(meta["cell_boxes"]):
                if i2 == hole:
                    continue
                x0, y0 = int(b["x"] * W), int(b["y"] * H)
                x1, y1 = int((b["x"] + b["w"]) * W), int((b["y"] + b["h"]) * H)
                g = _TC.classify_many([img.crop((x0, y0, x1, y1))])[0]
                grid[i2] = g[0]
                confs.append(g[1])
            clab, cconf = [], []
            for b in meta["candidate_boxes"]:
                x0, y0 = int(b["x"] * W), int(b["y"] * H)
                x1, y1 = int((b["x"] + b["w"]) * W), int((b["y"] + b["h"]) * H)
                g = _TC.classify_many([img.crop((x0, y0, x1, y1))])[0]
                clab.append(g[0])
                cconf.append(g[1])
            win = hct.resolve_pattern(grid, hole, clab)
            if win is None:
                gated += 1
            elif win == meta["correct"]:
                solved += 1
        print("\n  offline pattern solved: %d/%d (%d gated to vision)"
              % (solved, total, gated))
        self.assertGreaterEqual(solved, 35)

    def test_real_photo_tiles(self):
        """Held-out REAL photographs (data_real/val/) the trainer never saw —
        the honest synthetic->real transfer check. Labels come from the
        image-search query itself, so label noise is expected; the gate is a
        regression floor, the printed number is the honest metric (see
        SOLVER.md for the corpus and the measured transfer)."""
        import realdata
        from PIL import Image
        val = os.path.join(realdata.REAL_DIR, "val")
        if not os.path.isdir(val):
            self.skipTest("no real corpus (run: python realdata.py organize)")
        ims, want = [], []
        for name in sorted(os.listdir(val)):
            for f in sorted(glob.glob(os.path.join(val, name, "*.jpg"))):
                ims.append(Image.open(f).convert("RGB"))
                want.append(name)
        if not ims:
            self.skipTest("empty real val corpus")
        got = _TC.classify_many(ims)
        ok = sum(1 for g, w in zip(got, want) if g and g[0] == w)
        acc = ok / len(want)
        print("\n  REAL-photo tile accuracy: %.3f (%d/%d)"
              % (acc, ok, len(want)))
        self.assertGreaterEqual(acc, 0.45)


if __name__ == "__main__":
    unittest.main(verbosity=1)

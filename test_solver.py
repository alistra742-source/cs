#!/usr/bin/env python3
"""
test_solver.py — offline test suite for the hCaptcha multi-family solver.

NO browser, NO network, NO model server. Covers:

  * challenge-family routing from the /getcaptcha payload, from DOM facts
    and from prompt wording (incl. the staged live rounds: the affordance
    reference grid, the relational point round, the drag round, the
    counting round — "How many X are in this image?" — the
    pattern-completion drag round — "put one of the animals into the
    empty spot to complete the pattern" — and the wooden-block tower
    drag — "move the missing block segment onto the incomplete tower");
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
  * the Roboflow vision client: request shape (image + question), the
    detection->answer mapping, and the per-tile grid path.

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
from vision_solver import (
    RoboflowVisionClient,
    detection_classes,
    parse_yesno,
    shrink_image,
    tile_yes_question,
)

GEO = RoboflowVisionClient._parse_geometry     # shorthand


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

    def test_payload_area_select_tower_is_drag(self):
        # Live: served as image_label_area_select even though the answer
        # is a Move-badge drag. Must NOT commit to a point click.
        p = {"request_type": "image_label_area_select",
             "requester_question": {
                 "en": "Move the correct missing block segment onto "
                       "the incomplete tower"}}
        self.assertEqual(hct.classify_from_payload(p), hct.DRAG_DROP)
        self.assertEqual(hct.classify(p, None, hct.question_text(p)),
                         hct.DRAG_DROP)

    def test_payload_mixed_metal_then_tower_defers(self):
        # Combined payload question mentions the grid AND the tower —
        # do not lock the whole challenge to drag (stage 1 is still tiles).
        p = {"request_type": "image_label_area_select",
             "requester_question": {
                 "en": "Select items that are primarily metal, then "
                       "move the missing block onto the incomplete tower"}}
        self.assertEqual(hct.classify_from_payload(p), hct.UNKNOWN)
        dom_grid = {"tiles": 9, "images": 9, "choices": 0, "inputs": 0,
                    "canvases": 0, "draggables": 0, "move_badge": False}
        self.assertEqual(
            hct.classify(p, dom_grid, "Select items that are primarily metal"),
            hct.BINARY)
        self.assertEqual(
            hct.classify(p, {"tiles": 0, "images": 1, "canvases": 1,
                             "choices": 0, "inputs": 0, "draggables": 0,
                             "move_badge": False},
                         "Move the correct missing block segment onto "
                         "the incomplete tower"),
            hct.DRAG_DROP)

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

    def test_dom_tower_without_move_badge(self):
        # Real badge is "+ Move" / an icon child — the old leaf /^move$/
        # probe missed it and the single canvas fell through to a point click.
        f = {"tiles": 0, "choices": 0, "inputs": 0, "canvases": 1,
             "images": 1, "draggables": 0, "move_badge": False}
        self.assertEqual(hct.classify_from_dom(
            f, "Move the correct missing block segment onto the "
               "incomplete tower"), hct.DRAG_DROP)
        # same DOM without tower wording stays a point round
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

    def test_prompt_tower_drag(self):
        live = "Move the correct missing block segment onto the incomplete tower"
        self.assertTrue(hct.is_tower_prompt(live))
        self.assertEqual(hct.classify_from_prompt(live), hct.DRAG_DROP)
        self.assertFalse(hct.is_pattern_prompt(live))
        for prompt in (
                "Move the missing block onto the incomplete tower",
                "Drag the correct block segment onto the tower",
                "Complete the tower with the missing segment",
                "Place the missing segment onto the stack",
        ):
            self.assertTrue(hct.is_tower_prompt(prompt), prompt)
            self.assertEqual(hct.classify_from_prompt(prompt), hct.DRAG_DROP, prompt)
        self.assertFalse(hct.is_tower_prompt(
            "Please click on the animal who jumps the highest"))
        self.assertFalse(hct.is_tower_prompt(
            "Select items that are primarily metal"))
        self.assertFalse(hct.is_tower_prompt(
            "Put one of the animals into the empty spot to complete the pattern"))

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

    def test_prompt_setdown_places(self):
        live = ("Find places safe for setting down the item "
                "in the reference")
        self.assertEqual(hct.classify_from_prompt(live), hct.BINARY)
        self.assertTrue(hct.is_setdown_prompt(live))
        self.assertFalse(hct.is_tower_prompt(live))
        self.assertFalse(hct.is_attribute_prompt(live))
        # sibling wording
        for prompt in (
                "Find places that are safe to set the item down",
                "Select places safe for setting down the mug",
                "Where the reference item could be stored",
                "Find surfaces safe for the item in the reference",
        ):
            self.assertTrue(hct.is_setdown_prompt(prompt), prompt)
            self.assertEqual(hct.classify(None, None, prompt), hct.BINARY, prompt)
        # must NOT steal drag / affordance / metal grids
        for prompt in (
                "Please drag the element to the place where it fits",
                "Please pick all things you can work on with the item "
                "shown in the reference",
                "Select items that are primarily metal",
                "Please click on the animal who jumps the highest",
        ):
            self.assertFalse(hct.is_setdown_prompt(prompt), prompt)

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

    def test_setdown_mug_grid_round(self):
        # Live Discord screenshot: mug reference + 3x3 scene grid.
        payload = {"request_type": "image_label_binary",
                   "requester_question": {
                       "en": "Find places safe for setting down the item "
                             "in the reference"},
                   "requester_question_example": ["https://imgs/mug.jpg"],
                   "tasklist": [{"datapoint_uri": "https://imgs/t%d.jpg" % i}
                                for i in range(9)]}
        dom = {"tiles": 9, "examples": 1, "choices": 0, "inputs": 0,
               "canvases": 0, "images": 9, "draggables": 0,
               "move_badge": False}
        prompt = hct.question_text(payload)
        fam = hct.classify(payload, dom, prompt)
        self.assertEqual(fam, hct.BINARY)
        self.assertEqual(hct.answer_shape(fam), "tiles")
        # even when hCaptcha wraps the grid in area_select
        mixed = dict(payload)
        mixed["request_type"] = "image_label_area_select"
        self.assertEqual(hct.classify(mixed, dom, prompt), hct.BINARY)

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
            RoboflowVisionClient._parse_answer("The answer is 5", 1, "count"),
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

    def test_parse_tower_drag(self):
        # tower rounds share the drag answer shape (piece -> incomplete stack)
        got = GEO('{"drag": {"from": [0.88, 0.42], "to": [0.41, 0.61]}}',
                  "drag")
        self.assertEqual(got, {"type": "drag", "from": (0.88, 0.42),
                               "to": (0.41, 0.61)})

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

    def test_parse_loose_tile_numbers(self):
        pa = RoboflowVisionClient._parse_answer
        self.assertEqual(pa("1 3 5", 9, "tiles"),
                         {"type": "tiles", "indices": [1, 3, 5]})
        self.assertEqual(pa("tiles 1, 3 and 5", 9, "tiles"),
                         {"type": "tiles", "indices": [1, 3, 5]})
        self.assertEqual(pa("I see 9 tiles, pick 1 and 3", 9, "tiles"),
                         {"type": "tiles", "indices": [1, 3]})
        # still a count when the shape says so
        self.assertEqual(pa("1 3 5", 1, "count"),
                         {"type": "count", "count": 1})


# ── Roboflow request/response plumbing ───────────────────────────────────


class TestRoboflowHelpers(unittest.TestCase):

    def test_detection_classes_includes_noun_and_question(self):
        q = "Please click each image containing a boat"
        got = detection_classes(q)
        self.assertIn("boat", got)
        self.assertIn(q, got)

    def test_detection_classes_never_empty(self):
        self.assertTrue(detection_classes(""))

    def test_parse_yesno(self):
        self.assertIs(parse_yesno("yes"), True)
        self.assertIs(parse_yesno("Yes."), True)
        self.assertIs(parse_yesno("yep, a table"), True)
        self.assertIs(parse_yesno("no"), False)
        self.assertIs(parse_yesno("No balloon here"), False)
        self.assertIs(parse_yesno("nope"), False)
        # echoed instruction must not score as "no"
        self.assertIs(parse_yesno("Answer yes or no."), None)
        self.assertIs(parse_yesno(
            "Does this photo show a table? Answer yes or no. yes"), True)
        self.assertIs(parse_yesno(""), None)
        self.assertIs(parse_yesno("maybe a deck"), None)

    def test_tile_yes_question_setdown(self):
        live = ("Find places safe for setting down the item "
                "in the reference")
        q = tile_yes_question(live)
        self.assertIn("nightstand", q.lower())
        self.assertIn("balloon", q.lower())
        self.assertNotIn(live.lower(), q.lower())
        qref = tile_yes_question(live, has_ref=True)
        self.assertIn("first image is the item", qref.lower())

    def test_tile_yes_question_generic(self):
        q = tile_yes_question("Please click each image containing a boat")
        self.assertIn("boat", q.lower())

    def test_shape_question_carries_the_prompt(self):
        sq = RoboflowVisionClient.shape_question
        self.assertIn("how many cats", sq("How many cats", "count").lower())
        self.assertIn("count", sq("How many cats", "count").lower())
        self.assertEqual(sq("click the boat", "points"), "click the boat")

    def test_shrink_image_downscales(self):
        self.assertEqual(shrink_image(b""), b"")
        self.assertEqual(shrink_image(b"not-an-image"), b"not-an-image")
        try:
            import io
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        im = Image.new("RGB", (800, 600), (10, 20, 30))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out = shrink_image(buf.getvalue(), max_side=64)
        got = Image.open(io.BytesIO(out))
        self.assertLessEqual(max(got.size), 64)
        self.assertEqual(got.format, "JPEG")

    def test_read_response_finds_nested_predictions_and_text(self):
        body = {"outputs": [{"model_predictions": {
            "predictions": [{"x": 10, "y": 20, "width": 4, "height": 4,
                             "confidence": 0.9, "class": "boat"}]},
            "gemini_output": "yes"}]}
        preds, texts = RoboflowVisionClient.read_response(body)
        self.assertEqual(len(preds), 1)
        self.assertIn("yes", texts)

    def test_predictions_to_points_normalises_pixels(self):
        pts = RoboflowVisionClient.predictions_to_points(
            [{"x": 50, "y": 25, "width": 10, "height": 5,
              "confidence": 0.8, "class": "boat"}], (100, 50))
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(pts[0][0], 0.5)
        self.assertAlmostEqual(pts[0][1], 0.5)

    def test_predictions_to_points_drops_low_confidence(self):
        pts = RoboflowVisionClient.predictions_to_points(
            [{"x": 5, "y": 5, "confidence": 0.01}], (10, 10))
        self.assertEqual(pts, [])

    def test_detections_to_answer_shapes(self):
        d2a = RoboflowVisionClient.detections_to_answer
        pts = [(0.5, 0.5, 0.2, 0.2, 0.9, "a"), (0.9, 0.1, 0.1, 0.1, 0.7, "b")]
        self.assertEqual(d2a(pts, "points")["points"][0], (0.5, 0.5))
        bb = d2a(pts, "bbox")["bbox"]
        self.assertAlmostEqual(bb["x1"], 0.4)
        self.assertAlmostEqual(bb["y2"], 0.6)
        drag = d2a(pts, "drag")
        self.assertEqual(drag["from"], (0.5, 0.5))
        self.assertEqual(drag["to"], (0.9, 0.1))
        self.assertEqual(d2a(pts, "count"), {"type": "count", "count": 2})
        self.assertIsNone(d2a([], "points"))
        self.assertIsNone(d2a(pts[:1], "drag"))


class TestRoboflowSolve(unittest.IsolatedAsyncioTestCase):

    def _client(self):
        return RoboflowVisionClient(api_key="rf_test", log=lambda *a, **k: None)

    async def test_grid_asks_one_tile_at_a_time(self):
        client = self._client()
        calls = []
        replies = [
            [(0.5, 0.5, 0.1, 0.1, 0.9, "boat")],   # tile 1 matches
            [],                                     # tile 2 no
            [(0.4, 0.4, 0.1, 0.1, 0.8, "boat")],   # tile 3 matches
            [],
        ]

        async def fake_detect(image, question, timeout, classes=None):
            calls.append({"q": question, "classes": classes})
            return replies.pop(0), [], {"ok": True}

        client._detect = fake_detect
        png = b"\x89PNG\r\n\x1a\n"
        got = await client.solve("select boats", [png] * 4, shape="tiles")
        self.assertEqual(got, {"type": "tiles", "indices": [1, 3]})
        self.assertEqual(len(calls), 4)
        # the captcha question rides along with every image
        self.assertIn("boat", calls[0]["q"].lower())
        self.assertIn("boat", calls[0]["classes"])

    async def test_grid_falls_back_to_yes_no_text(self):
        client = self._client()

        async def fake_detect(image, question, timeout, classes=None):
            return [], ["yes"], {"ok": True}

        client._detect = fake_detect
        got = await client.solve("select boats", [b"a", b"b"], shape="tiles")
        self.assertEqual(got, {"type": "tiles", "indices": [1, 2]})

    async def test_all_errors_returns_none(self):
        client = self._client()

        async def boom(*a, **k):
            return None, None, None

        client._detect = boom
        got = await client.solve("select boats", [b"a", b"b", b"c"])
        self.assertIsNone(got)
        self.assertEqual(client.stats["failed"], 1)

    async def test_point_round_uses_detection_centre(self):
        client = self._client()

        async def fake_detect(image, question, timeout, classes=None):
            return [(0.25, 0.75, 0.1, 0.1, 0.9, "cat")], [], {"ok": True}

        client._detect = fake_detect
        got = await client.solve("click the cat", [b"canvas"], shape="points")
        self.assertEqual(got, {"type": "points", "points": [(0.25, 0.75)]})

    async def test_text_answer_falls_back_to_json_parsing(self):
        client = self._client()

        async def fake_detect(image, question, timeout, classes=None):
            return [], ['{"count": 4}'], {"ok": True}

        client._detect = fake_detect
        got = await client.solve("how many cats", [b"canvas"], shape="count")
        self.assertEqual(got, {"type": "count", "count": 4})

    async def test_unconfigured_client_returns_none(self):
        client = RoboflowVisionClient(api_key="", log=lambda *a, **k: None)
        self.assertFalse(client.configured)
        self.assertIsNone(await client.solve("x", [b"a"]))


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


# ── wooden-block tower locator ────────────────────────────────────────────


def _blank_rgb(w, h, color=(236, 226, 208)):
    return [[color] * w for _ in range(h)]


def _fill_rect(grid, x0, y0, x1, y1, color):
    h = len(grid)
    w = len(grid[0])
    for y in range(max(0, y0), min(h, y1)):
        row = list(grid[y])
        for x in range(max(0, x0), min(w, x1)):
            row[x] = color
        grid[y] = row


def _stack_blocks(grid, cx, n, bottom, bw=28, bh=16, gap=3,
                  colors=((186, 128, 62), (158, 102, 48))):
    """Paint ``n`` wooden blocks stacked upward from ``bottom``."""
    for i in range(n):
        y1 = bottom - i * (bh + gap)
        y0 = y1 - bh
        _fill_rect(grid, cx - bw // 2, y0, cx + bw // 2, y1, colors[i % 2])


class TestLocateTowerDrag(unittest.TestCase):

    def test_shortest_middle_tower(self):
        # 3 towers (4 / 2 / 4) + a 2-block Move piece on the right.
        w, h = 240, 150
        grid = _blank_rgb(w, h)
        _stack_blocks(grid, 42, 4, bottom=132)
        _stack_blocks(grid, 100, 2, bottom=132)   # incomplete
        _stack_blocks(grid, 158, 4, bottom=132)
        _stack_blocks(grid, 214, 2, bottom=88)    # floating piece
        got = hct.locate_tower_drag(grid)
        self.assertIsNotNone(got)
        fx, fy = got["from"]
        tx, ty = got["to"]
        self.assertGreater(fx, 0.78)             # piece is on the right
        self.assertGreater(tx, 0.30)
        self.assertLess(tx, 0.55)                # middle tower
        self.assertGreater(ty, 0.45)             # onto the short stack, not sky

    def test_gapped_tower(self):
        # Middle tower is 4-high with the 3rd block missing — drop in the gap.
        w, h = 240, 150
        grid = _blank_rgb(w, h)
        _stack_blocks(grid, 42, 4, bottom=132)
        _stack_blocks(grid, 158, 4, bottom=132)
        # bottom two + top one of a 4-stack, skip the 3rd
        _stack_blocks(grid, 100, 2, bottom=132)
        _stack_blocks(grid, 100, 1, bottom=132 - 3 * (16 + 3))
        _stack_blocks(grid, 214, 2, bottom=88)
        got = hct.locate_tower_drag(grid)
        self.assertIsNotNone(got)
        tx, ty = got["to"]
        self.assertGreater(tx, 0.30)
        self.assertLess(tx, 0.55)
        # gap sits between the 2nd and 4th blocks
        self.assertGreater(ty, 0.35)
        self.assertLess(ty, 0.75)

    def test_no_wood_returns_none(self):
        grid = _blank_rgb(80, 60, (240, 240, 240))
        self.assertIsNone(hct.locate_tower_drag(grid))
        self.assertIsNone(hct.locate_tower_drag(None))
        self.assertIsNone(hct.locate_tower_drag([]))

    def test_photo_highlights_and_piece_hint(self):
        # Live towers are photographs: near-white highlights used to fail
        # the strict wood mask and the piece often sits OUTSIDE the photo.
        w, h = 240, 150
        grid = _blank_rgb(w, h, (245, 238, 220))
        lights = ((228, 186, 118), (242, 214, 158), (92, 58, 32))
        _stack_blocks(grid, 42, 4, bottom=132, colors=lights)
        _stack_blocks(grid, 100, 2, bottom=132, colors=lights)
        _stack_blocks(grid, 158, 4, bottom=132, colors=lights)
        dbg = {}
        got = hct.locate_tower_drag(grid, piece_hint=(0.88, 0.48), debug=dbg)
        self.assertIsNotNone(got, dbg)
        self.assertGreater(got["from"][0], 0.75)
        self.assertGreater(got["to"][0], 0.30)
        self.assertLess(got["to"][0], 0.55)
        self.assertEqual(dbg.get("reason"), "ok")


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

    def test_setdown_mug_clicks_surfaces_not_balloons(self):
        # Live 3x3: nightstand, balloon, deck+ball, balloon, bench,
        # deck+ball, leaf, building, leaf. Click furniture/deck only.
        live = ("Find places safe for setting down the item "
                "in the reference")
        labels = ["table", "airplane", "wood", "airplane", "chair",
                  "wood", "flower", "house", "flower"]
        self.assertEqual(
            hct.resolve_semantic(live, labels, example_label="cup"),
            [1, 3, 5, 6])
        # aliases the CNN / prompt nouns actually emit
        self.assertEqual(hct.canonical("nightstand"), "table")
        self.assertEqual(hct.canonical("bench"), "chair")
        self.assertEqual(hct.canonical("wooden_deck"), "wood")
        self.assertEqual(hct.canonical("maple_leaf"), "flower")
        # no surfaces at all → vision, not an empty Verify
        self.assertIsNone(hct.resolve_semantic(
            live, ["airplane", "flower", "cup"]))

    def test_larger_than_reference(self):
        prompt = "Select items that are larger than the item in the reference"
        labels = ["snail", "dog", "elephant", "butterfly"]
        self.assertEqual(
            hct.resolve_semantic(prompt, labels, example_label="dog"),
            [3])
        self.assertEqual(
            hct.resolve_semantic(
                "Pick the items smaller than the item shown",
                labels, example_label="dog"),
            [1, 4])


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


# live browser pointer helpers + trainer ingest


class TestTrainerFrames(unittest.TestCase):

    def test_widget_vs_challenge_urls(self):
        import trainer
        eng = trainer.TrainerEngine()
        widget = ("https://newassets.hcaptcha.com/captcha/v1/x/static/"
                  "hcaptcha.html#frame=checkbox")
        challenge = ("https://newassets.hcaptcha.com/captcha/v1/x/static/"
                     "hcaptcha.html#frame=challenge")
        self.assertTrue(eng._is_widget_frame_url(widget))
        self.assertFalse(eng._is_challenge_frame_url(widget))
        self.assertTrue(eng._is_challenge_frame_url(challenge))
        self.assertFalse(eng._is_widget_frame_url(challenge))
        self.assertFalse(eng._is_widget_frame_url("https://example.com"))


class TestTrainerLiveNotes(unittest.TestCase):

    def test_note_pointer_and_challenge(self):
        import trainer
        eng = trainer.TrainerEngine()
        eng.note_pointer({"kind": "click", "x": 120.4, "y": 88.9})
        eng.note_pointer({"kind": "drag", "x1": 10, "y1": 20, "x2": 80, "y2": 90})
        self.assertEqual(len(eng.pointer_log), 2)
        joined = " ".join(eng.logs)
        self.assertIn("Operator click at (120,89)", joined)
        self.assertIn("Operator drag (10,20)", joined)
        eng.note_live_challenge("data:image/png;base64,abc", "Select all boats")
        self.assertEqual(eng.latest_screenshot, "data:image/png;base64,abc")
        self.assertEqual(eng.captured_count, 1)
        self.assertEqual(eng.latest_question, "Select all boats")
        # same prompt only refreshes the screenshot
        eng.note_live_challenge("data:image/png;base64,def", "Select all boats")
        self.assertEqual(eng.captured_count, 1)
        self.assertEqual(eng.latest_screenshot, "data:image/png;base64,def")


class TestLivePointer(unittest.TestCase):

    def test_parse_and_format(self):
        import live_control as lc
        self.assertEqual(lc.parse_xy({"x": 12.2, "y": "40"}), (12.2, 40.0))
        self.assertEqual(lc.parse_xy({"x1": 1, "y1": 2}, "x1", "y1"), (1.0, 2.0))
        self.assertEqual(lc.parse_xy({"x": None, "y": "nope"}), (0.0, 0.0))
        self.assertEqual(lc.format_click_log(432.4, 518.9),
                         "click at (432, 519)")
        self.assertIn("drag (10, 20) → (80, 90)",
                      lc.format_drag_log(10, 20, 80, 90))
        self.assertTrue(lc.is_challenge_frame_url(
            "https://hcaptcha.com/captcha#frame=challenge"))
        self.assertFalse(lc.is_challenge_frame_url(
            "https://hcaptcha.com/captcha#frame=checkbox"))
        rec = lc.pointer_entry("click", x=1.234, y=5)
        self.assertEqual(rec["kind"], "click")
        self.assertEqual(rec["x"], 1.2)
        self.assertIn("t", rec)
        rec2 = lc.pointer_entry(
            "click", x=10, y=20, selector='input[name="email"]',
            js='document.querySelector("input[name=\\"email\\"]")',
            is_input=True)
        self.assertEqual(rec2["selector"], 'input[name="email"]')
        self.assertTrue(rec2["is_input"])
        self.assertIn("querySelector", rec2["js"])
        self.assertIn('input[name="email"]', lc.format_click_log(
            10, 20, 'input[name="email"]'))
        hit = lc.sanitize_hit(
            {"selector": "#x", "js": 'document.querySelector("#x")',
             "is_input": 1})
        self.assertEqual(hit["selector"], "#x")
        self.assertTrue(hit["is_input"])
        dump = lc.format_pointer_dump([rec2])
        self.assertIn("selector:", dump)
        self.assertIn("js:", dump)


class TestMouseMovePoints(unittest.TestCase):

    def test_interpolates_from_current_not_origin(self):
        from nodriver_engine import mouse_move_points
        pts = mouse_move_points(100, 200, 140, 200, steps=4)
        self.assertEqual(len(pts), 4)
        self.assertEqual(pts[0], (110.0, 200.0))
        self.assertEqual(pts[-1], (140.0, 200.0))
        # a 1-step move is just the destination
        self.assertEqual(mouse_move_points(10, 10, 80, 90, steps=1),
                         [(80.0, 90.0)])


class TestPerformLiveAction(unittest.IsolatedAsyncioTestCase):

    async def test_click_and_drag_use_mouse(self):
        import live_control as lc

        class FakeMouse:
            def __init__(self):
                self.ops = []
                self.x = 0.0
                self.y = 0.0

            async def move(self, x, y, steps=None):
                self.x, self.y = float(x), float(y)
                self.ops.append(("move", self.x, self.y, steps))

            async def click(self, x, y):
                self.x, self.y = float(x), float(y)
                self.ops.append(("click", self.x, self.y))

            async def down(self, button="left"):
                self.ops.append(("down", self.x, self.y))

            async def up(self, button="left"):
                self.ops.append(("up", self.x, self.y))

        class FakeKeyboard:
            def __init__(self):
                self.typed = []

            async def type(self, text, delay=0):
                self.typed.append(str(text))

            async def press(self, key):
                self.typed.append("key:" + str(key))

        class FakePage:
            def __init__(self):
                self.mouse = FakeMouse()
                self.keyboard = FakeKeyboard()

            async def evaluate(self, js, arg=None):
                return {
                    "selector": 'input[name="email"]',
                    "js": 'document.querySelector("input[name=\\"email\\"]")',
                    "is_input": 1,
                }

        page = FakePage()
        rec = await lc.perform_live_action(page, {"action": "click", "x": 40, "y": 80})
        self.assertEqual(rec["kind"], "click")
        self.assertEqual(rec["x"], 40.0)
        self.assertTrue(any(op[0] == "click" for op in page.mouse.ops))
        self.assertEqual(rec.get("selector"), 'input[name="email"]')
        self.assertTrue(rec.get("is_input"))

        page = FakePage()
        typed = await lc.perform_live_action(
            page, {"action": "type", "text": "hello"})
        self.assertEqual(typed["kind"], "type")
        self.assertEqual(page.keyboard.typed, ["hello"])

        page = FakePage()
        rec = await lc.perform_live_action(
            page, {"action": "drag", "x1": 10, "y1": 20, "x2": 90, "y2": 40})
        self.assertEqual(rec["kind"], "drag")
        kinds = [op[0] for op in page.mouse.ops]
        self.assertIn("down", kinds)
        self.assertIn("up", kinds)
        self.assertLess(kinds.index("down"), kinds.index("up"))


class TestPngComplete(unittest.TestCase):

    def test_complete_vs_truncated(self):
        import io
        from PIL import Image
        from server import png_dimensions, png_is_complete

        im = Image.new("RGB", (64, 48), (10, 20, 30))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        raw = buf.getvalue()
        self.assertTrue(png_is_complete(raw))
        self.assertEqual(png_dimensions(raw), (64, 48))
        self.assertFalse(png_is_complete(raw[:-8]))
        self.assertFalse(png_is_complete(b""))
        self.assertFalse(png_is_complete(b"\x89PNG\r\n\x1a\nnot-enough"))
        self.assertFalse(png_is_complete("not-bytes"))

    def test_capture_accepts_reveal(self):
        import inspect
        from server import capture_page_screenshot
        params = inspect.signature(capture_page_screenshot).parameters
        self.assertIn("reveal", params)
        self.assertEqual(params["reveal"].default, "safe")


class TestSaveChallengePng(unittest.TestCase):

    def test_saves_two_hashes_and_reuses(self):
        import io
        import shutil
        import tempfile
        from PIL import Image
        import live_control as lc

        td = tempfile.mkdtemp()
        old = lc.CHALLENGE_DIR
        lc.CHALLENGE_DIR = td
        try:
            def png(color):
                im = Image.new("RGB", (80, 80), color)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                return buf.getvalue()

            first = lc.save_challenge_png(png((255, 0, 0)), "q1")
            second = lc.save_challenge_png(png((0, 255, 0)), "q2")
            again = lc.save_challenge_png(png((255, 0, 0)), "q1-again")
            self.assertTrue(first and second)
            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(first["id"], again["id"])
            self.assertEqual(first["file"], again["file"])
            files = [name for name in os.listdir(td) if name.endswith(".png")]
            self.assertEqual(len(files), 2)
            self.assertTrue(os.path.isfile(os.path.join(td, first["file"])))
            self.assertTrue(os.path.isfile(os.path.join(td, second["file"])))
        finally:
            lc.CHALLENGE_DIR = old
            shutil.rmtree(td, ignore_errors=True)

    def test_rejects_incomplete(self):
        import live_control as lc
        self.assertEqual(lc.save_challenge_png("data:image/png;base64,abc"), {})
        self.assertEqual(lc.save_challenge_png(b"\x89PNG\r\n\x1a\nxxxx"), {})

    def test_image_src_keeps_challenge_url(self):
        import live_control as lc
        self.assertEqual(lc.image_src("/challenges/foo.png"), "/challenges/foo.png")
        self.assertTrue(lc.image_src("abc").startswith("data:image/png;base64,"))


class TestChallengeFilePath(unittest.TestCase):

    def test_rejects_traversal(self):
        import live_control as lc
        self.assertEqual(lc.challenge_file_path("../secret.png"), "")
        self.assertEqual(lc.challenge_file_path(".."), "")
        self.assertEqual(lc.challenge_file_path("ok.txt"), "")
        self.assertEqual(lc.challenge_file_path("not-there.png"), "")
        self.assertEqual(lc.challenge_file_path(""), "")

    def test_discord_register_url(self):
        import live_control as lc
        self.assertEqual(lc.DISCORD_REGISTER_URL, "https://discord.com/register")
        self.assertTrue(os.path.isabs(lc.CHALLENGE_DIR))


class TestTrainerChallengePersist(unittest.TestCase):

    def test_clear_keeps_saved_files(self):
        import base64
        import io
        import shutil
        import tempfile
        from PIL import Image
        import live_control as lc
        import trainer

        td = tempfile.mkdtemp()
        old = lc.CHALLENGE_DIR
        lc.CHALLENGE_DIR = td
        try:
            im = Image.new("RGB", (80, 60), (1, 2, 3))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            image = ("data:image/png;base64,"
                     + base64.b64encode(buf.getvalue()).decode("ascii"))
            eng = trainer.TrainerEngine()
            eng.note_live_challenge(image, "Select all boats")
            self.assertTrue(eng.saved_challenges)
            rec = eng.saved_challenges[0]
            path = os.path.join(td, rec["file"])
            self.assertTrue(os.path.isfile(path))
            url = eng.latest_challenge_image
            self.assertTrue(url.startswith("/challenges/"))
            kept = list(eng.saved_challenges)
            eng.clear()
            self.assertEqual(eng.latest_screenshot, "")
            self.assertEqual(eng.saved_challenges, kept)
            self.assertEqual(eng.latest_challenge_image, url)
            self.assertTrue(os.path.isfile(path))
        finally:
            lc.CHALLENGE_DIR = old
            shutil.rmtree(td, ignore_errors=True)


class TestLiveUiRegister(unittest.TestCase):

    def test_register_button_and_helper(self):
        import live_ui
        html = live_ui.LIVE_INJECTION
        self.assertIn('id="liveRegBtn"', html)
        self.assertIn("function lcGoRegister", html)
        self.assertIn("https://discord.com/register", html)
        self.assertIn("/challenges/", html)
        self.assertIn("force:true", html)


if __name__ == "__main__":
    unittest.main(verbosity=1)

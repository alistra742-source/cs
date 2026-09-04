#!/usr/bin/env python3
"""Tests for vision_events: payload shape must match the API spec."""
import unittest
from datetime import datetime, timedelta, timezone

from vision_events import (VisionEvents, build_event, clean_metadata,
                           timestamp_is_valid)


class TestBuildEvent(unittest.TestCase):
    def test_required_keys_present(self):
        ev = build_event("pass")
        for k in ("eventId", "eventType", "useCaseId", "timestamp",
                  "eventData", "customMetadata"):
            self.assertIn(k, ev)

    def test_result_is_pass_or_fail_only(self):
        self.assertEqual(build_event("pass")["eventData"]["result"], "pass")
        self.assertEqual(build_event("solved")["eventData"]["result"], "pass")
        self.assertEqual(build_event("fail")["eventData"]["result"], "fail")
        self.assertEqual(build_event("anything")["eventData"]["result"], "fail")

    def test_ids_are_capped_at_256(self):
        ev = build_event("pass", use_case_id="x" * 400, event_id="y" * 400)
        self.assertLessEqual(len(ev["useCaseId"]), 256)
        self.assertLessEqual(len(ev["eventId"]), 256)

    def test_event_ids_are_unique(self):
        ids = {build_event("pass")["eventId"] for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_timestamp_is_accepted_window(self):
        self.assertTrue(timestamp_is_valid(build_event("pass")["timestamp"]))

    def test_out_of_range_timestamp_is_replaced(self):
        old = (datetime.now(timezone.utc) - timedelta(days=800)).isoformat()
        ev = build_event("pass", timestamp=old)
        self.assertTrue(timestamp_is_valid(ev["timestamp"]))


class TestMetadata(unittest.TestCase):
    def test_scalar_types_survive(self):
        got = clean_metadata({"a": "s", "b": 3, "c": 1.5, "d": True})
        self.assertEqual(got, {"a": "s", "b": 3, "c": 1.5, "d": True})

    def test_complex_values_are_stringified(self):
        got = clean_metadata({"a": [1, 2], "b": {"x": 1}})
        self.assertIsInstance(got["a"], str)
        self.assertIsInstance(got["b"], str)

    def test_none_is_dropped(self):
        self.assertNotIn("a", clean_metadata({"a": None}))

    def test_capped_at_100_entries(self):
        self.assertLessEqual(
            len(clean_metadata({f"k{i}": i for i in range(250)})), 100)

    def test_non_dict_is_safe(self):
        self.assertEqual(clean_metadata(None), {})
        self.assertEqual(clean_metadata("nope"), {})


class TestReporter(unittest.IsolatedAsyncioTestCase):
    def test_disabled_without_opt_in(self):
        self.assertFalse(VisionEvents(api_key="k", enabled=False).enabled)

    def test_enabled_requires_api_key(self):
        self.assertFalse(VisionEvents(api_key="", enabled=True).enabled)

    async def test_disabled_reporter_never_posts(self):
        ev = VisionEvents(api_key="k", enabled=False)

        async def boom(_):
            raise AssertionError("disabled reporter must not post")

        ev._post = boom
        self.assertFalse(await ev.report("pass"))

    async def test_report_passes_metadata_through(self):
        ev = VisionEvents(api_key="k", enabled=True)
        seen = {}

        async def fake(payload):
            seen.update(payload)
            return True

        ev._post = fake
        self.assertTrue(await ev.report("pass", family="drag"))
        self.assertEqual(seen["customMetadata"]["family"], "drag")
        self.assertEqual(seen["eventData"]["result"], "pass")

    def test_nowait_outside_a_loop_is_safe(self):
        VisionEvents(api_key="k", enabled=True).report_nowait("pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)

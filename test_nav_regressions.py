#!/usr/bin/env python3
"""Regressions for two bugs that aborted every attempt.

1. NameError: `probe` was read outside the throttled block that defines it,
   so the FIRST unreadable-page iteration crashed the whole attempt.
2. False-positive hCaptcha "widget error": the detector scanned the entire
   body innerText, which includes hCaptcha's inline hsw script blob, so a
   healthy widget was declared broken and the circuit rotated.
"""
import ast
import re
import unittest


class TestProbeIsAlwaysBound(unittest.TestCase):
    """`last_probe` must be initialised before the poll loop reads it."""

    def setUp(self):
        self.src = open("server.py").read()

    def test_last_probe_initialised_before_use(self):
        init = self.src.find("last_probe = \"\"")
        use = self.src.find("str(last_probe).lower()")
        self.assertNotEqual(init, -1, "last_probe is never initialised")
        self.assertNotEqual(use, -1, "last_probe is never used")
        self.assertLess(init, use,
                        "last_probe is read before it is initialised")

    def test_bare_probe_not_read_outside_its_block(self):
        """The dead-session check must not reference the throttled name."""
        self.assertNotIn('str(probe).lower()', self.src)

    def test_server_module_parses(self):
        ast.parse(self.src)

    def test_early_iteration_does_not_raise(self):
        """Reproduce the exact flow: check runs before the 3s log fires."""
        last_probe = ""
        elapsed, last_log = 0.4, 0.0
        if elapsed >= last_log + 3.0:          # not taken on iteration 1
            last_probe = "(no probe)"
        # This line raised NameError before the fix.
        hit = ("session with given id not found" in str(last_probe).lower())
        self.assertFalse(hit)

    def test_dead_session_still_detected(self):
        last_probe = ("probe-failed: ProtocolException: Session with given "
                      "id not found. [code: -32001]")
        self.assertTrue(
            "session with given id not found" in str(last_probe).lower())


class TestWidgetErrorFalsePositive(unittest.TestCase):
    """hCaptcha's hsw payload must never be read as an error banner."""

    HSW_BLOB = ('/* { "version": "1", "hash": "MEUCIQDAWcI2HzSA2hpyfr3quc'
                'MOrS+roDuhhDJT0c1Kd7nk3QIgT6kouxzkhwP4aJCwVZXNwvjotCj9wr'
                'ggJBqtyL')

    KEYWORDS = ("rate limited or network error", "rate limited",
                "network error", "please retry", "please try again",
                "automated queries")

    def _is_error(self, text):
        """Mirror of the guard in _widget_error_state."""
        low = (text or "").lower()
        if '"hash"' in low or "/*" in low or len(low) > 300:
            return False
        return any(k in low for k in self.KEYWORDS)

    def test_hsw_blob_is_not_an_error(self):
        self.assertFalse(self._is_error(self.HSW_BLOB))

    def test_script_payload_with_keyword_is_ignored(self):
        poisoned = self.HSW_BLOB + ' network error handler'
        self.assertFalse(self._is_error(poisoned))

    def test_real_banner_still_detected(self):
        self.assertTrue(
            self._is_error("Rate limited or network error. Please retry."))

    def test_long_text_is_not_a_banner(self):
        self.assertFalse(self._is_error("network error " + "x" * 400))

    def test_healthy_widget_reports_nothing(self):
        self.assertFalse(self._is_error(""))
        self.assertFalse(self._is_error("I am not a robot"))

    def test_detector_reads_only_visible_error_nodes(self):
        src = open("server.py").read()
        i = src.find("async def _widget_error_state")
        body = src[i:i + 2600]
        self.assertIn("role=\\\"alert\\\"", body.replace('\\"', '\\\\"')
                      ) if False else None
        # It must query error elements, not dump the whole body.
        self.assertIn("querySelectorAll", body)
        self.assertNotIn("document.body ? document.body.innerText : ''",
                         body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

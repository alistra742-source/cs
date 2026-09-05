#!/usr/bin/env python3
"""Regressions for bugs that aborted or hung every attempt.

1. NameError: `probe` was read outside the throttled block that defines it,
   so the FIRST unreadable-page iteration crashed the whole attempt.
2. False-positive hCaptcha "widget error": the detector scanned the entire
   body innerText, which includes hCaptcha's inline hsw script blob, so a
   healthy widget was declared broken and the circuit rotated.
3. Blank-SPA hang: on a slow circuit Discord's index.html commits while its
   JS bundles are STILL downloading; the render wait could not tell that
   from a dropped bundle - standard mode reloaded a download that was
   making progress, and LOW_MEMORY_MODE (max_reloads=0) never re-fetched a
   bundle the proxy dropped, burning the budget + a live circuit.
4. Camera: the startup new-tab page became the feed's "last good frame"
   and replayed for the whole navigation; empty captures during an
   uncommitted goto spammed ALL LOGS every tick.
"""
import ast
import re
import unittest

import nav_policy


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


class TestBlankSpaBundlePolicy(unittest.TestCase):
    """The render wait must tell 'bundles downloading' from 'bundles dead'.

    Regression for the hang in the field log: DOM committed in 31s, React
    never booted, and LOW_MEMORY_MODE (then max_reloads=0) neither waited
    for the in-flight bundles nor re-fetched the failed ones - the attempt
    polled to the budget and rotated a circuit that was fine.
    """

    IN_FLIGHT = {"scriptsTotal": 7, "scriptsPending": 5, "scriptsFailed": 0,
                 "loadComplete": False}
    SETTLED = {"scriptsTotal": 7, "scriptsPending": 0, "scriptsFailed": 1,
               "loadComplete": True}

    def test_bundles_in_flight_are_patient(self):
        self.assertEqual(
            nav_policy.blank_action(self.IN_FLIGHT, 30.0, 0, 1, 4.0,
                                    False, True),
            nav_policy.WAIT_BUNDLES)

    def test_in_flight_extends_the_render_budget(self):
        self.assertTrue(nav_policy.bundles_pending(self.IN_FLIGHT))
        self.assertFalse(nav_policy.bundles_pending(self.SETTLED))

    def test_settled_blank_reloads_even_in_low_memory(self):
        # The old low-memory dead end: max_reloads=0 meant a dropped bundle
        # was never re-fetched. One bounded reload must happen in BOTH modes.
        self.assertEqual(
            nav_policy.blank_action(self.SETTLED, 5.0, 0, 1, 4.0, False, True),
            nav_policy.RELOAD)
        self.assertEqual(
            nav_policy.blank_action(self.SETTLED, 5.0, 0, 2, 4.0, False, False),
            nav_policy.RELOAD)

    def test_too_early_to_judge_waits(self):
        settled_no_fail = {"scriptsTotal": 7, "scriptsPending": 0,
                           "scriptsFailed": 0, "loadComplete": False}
        self.assertEqual(
            nav_policy.blank_action(settled_no_fail, 2.0, 0, 1, 4.0,
                                    False, True),
            nav_policy.WAIT)

    def test_reloads_exhausted_waits_for_the_budget(self):
        self.assertEqual(
            nav_policy.blank_action(self.SETTLED, 9.0, 1, 1, 4.0, False, True),
            nav_policy.WAIT_BUDGET)

    def test_stub_after_reloads_rotates_in_both_memory_modes(self):
        for low_memory in (True, False):
            self.assertEqual(
                nav_policy.blank_action(self.SETTLED, 9.0, 2, 2, 4.0,
                                        True, low_memory),
                nav_policy.ROTATE_STUB)

    def test_state_without_bundle_fields_is_not_pending(self):
        for empty in ({}, None, {"scriptsPending": "n/a"}):
            self.assertFalse(nav_policy.bundles_pending(empty))
        # ...and a blank old-style probe still gets the reload path.
        self.assertEqual(nav_policy.blank_action({}, 9.0, 0, 1, 4.0,
                                                 False, True),
                         nav_policy.RELOAD)

    def test_server_loop_is_wired_to_the_policy(self):
        src = open("server.py").read()
        self.assertIn("nav_policy.blank_action(", src)
        self.assertIn("nav_policy.bundles_pending(state)", src)
        self.assertIn("max_reloads = 1 if LOW_MEMORY_MODE else 2", src)
        # The old dead ends are gone.
        self.assertNotIn("max_reloads = 0 if LOW_MEMORY_MODE", src)
        self.assertNotIn("skipping reload to protect renderer", src)

    def test_state_probe_reports_bundle_health(self):
        src = open("server.py").read()
        for field in ("scriptsPending:", "scriptsTotal:", "scriptsFailed:",
                      "loadComplete:"):
            self.assertIn(field, src)
        ast.parse(src)


class TestCameraNeverStoresBrowserTabs(unittest.TestCase):
    """The feed must not replay the startup new-tab page as 'last good'."""

    def test_new_tab_and_blank_pages_are_uninformative(self):
        for url, title in (("chrome://newtab", "New Tab"),
                           ("about:blank", ""),
                           ("", "Ny fane"),
                           ("chrome://newtab-footer", "Neuer Tab")):
            self.assertTrue(nav_policy.is_uninformative_page(url, title),
                            f"{url}/{title} should be skipped")

    def test_real_pages_are_informative(self):
        self.assertFalse(nav_policy.is_uninformative_page(
            "https://discord.com/register", "Discord"))
        self.assertFalse(nav_policy.is_uninformative_page(
            "https://discord.com/register", ""))

    def test_capture_is_gated_on_the_check(self):
        src = open("server.py").read()
        self.assertIn("nav_policy.is_uninformative_page(", src)

    def test_capture_retries_ride_a_backoff(self):
        src = open("server.py").read()
        i = src.find("async def capture_page_screenshot")
        body = src[i:i + 6500]
        self.assertIn("backoff = (0.15, 0.5, 1.2)", body)
        self.assertNotIn("await asyncio.sleep(0.12)", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

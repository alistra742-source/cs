#!/usr/bin/env python3
"""The CDP interceptor must actually install and rewrite the body.

Three real bugs this file locks down, all of which failed SILENTLY:
  1. the method was defined on _Context, which has no self._tab, so
     getattr(page, "intercept_request_bodies") returned None and the whole
     feature was a no-op — cdp_injections stayed 0 with cdp_note empty;
  2. Fetch.continueRequest expects post_data BASE64-encoded, so a raw
     string would have produced a body Discord could not parse;
  3. a paused request that is never continued hangs the page, so the
     continue call needs a fallback on any error.
"""
import ast
import asyncio
import json
import unittest

import nodriver_engine as ne

try:
    from nodriver import cdp
    HAS_CDP = True
except Exception:
    HAS_CDP = False


class FakeTab:
    def __init__(self):
        self.handlers = {}
        self.sent = []

    async def send(self, cmd):
        self.sent.append(cmd)
        return None


def make_page():
    page = ne._Page.__new__(ne._Page)
    page._tab = FakeTab()
    return page


class TestMethodLivesOnPage(unittest.TestCase):
    def test_defined_on_page_not_context(self):
        tree = ast.parse(open("nodriver_engine.py").read())
        owner = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for n in node.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and n.name == "intercept_request_bodies":
                        owner = node.name
        self.assertEqual(owner, "_Page",
                         "must live on the class that owns self._tab")

    def test_page_instances_expose_it(self):
        self.assertTrue(callable(
            getattr(make_page(), "intercept_request_bodies", None)))


@unittest.skipUnless(HAS_CDP, "nodriver not installed")
class TestInterception(unittest.IsolatedAsyncioTestCase):
    async def test_install_returns_true(self):
        page = make_page()
        self.assertTrue(
            await page.intercept_request_bodies(["/auth/register"],
                                                lambda u, b: None))

    async def test_handler_is_registered(self):
        page = make_page()
        await page.intercept_request_bodies(["/auth/register"],
                                            lambda u, b: None)
        self.assertIn(cdp.fetch.RequestPaused, page._tab.handlers)

    async def test_survives_a_plain_dict_handlers(self):
        """A KeyError here silently disabled the whole interceptor."""
        page = make_page()
        page._tab.handlers = {}
        self.assertTrue(
            await page.intercept_request_bodies(["/auth/register"],
                                                lambda u, b: None))

    async def _fire(self, page, url, body):
        class Req:
            pass
        req = Req()
        req.url = url
        req.post_data = body

        class Ev:
            pass
        ev = Ev()
        ev.request_id = "RID1"
        ev.request = req
        page._tab.handlers[cdp.fetch.RequestPaused][0](ev)
        await asyncio.sleep(0.15)

    async def test_mutate_is_called_for_matching_url(self):
        page = make_page()
        seen = {}

        def mutate(url, body):
            seen["url"] = url
            return None

        await page.intercept_request_bodies(["/auth/register"], mutate)
        await self._fire(page, "https://discord.com/api/v9/auth/register",
                         json.dumps({"email": "a@b.c"}))
        self.assertEqual(seen.get("url"),
                         "https://discord.com/api/v9/auth/register")

    async def test_mutate_skipped_for_other_urls(self):
        page = make_page()
        calls = []
        await page.intercept_request_bodies(
            ["/auth/register"], lambda u, b: calls.append(u))
        await self._fire(page, "https://discord.com/api/v9/science",
                         json.dumps({"x": 1}))
        self.assertEqual(calls, [])

    async def test_request_is_always_continued(self):
        """An un-continued paused request hangs the page."""
        page = make_page()

        def boom(url, body):
            raise RuntimeError("mutate exploded")

        await page.intercept_request_bodies(["/auth/register"], boom)
        await self._fire(page, "https://discord.com/api/v9/auth/register",
                         json.dumps({"email": "a@b.c"}))
        self.assertGreaterEqual(len(page._tab.sent), 2,
                                "continue_request was never sent")


class TestSourceInvariants(unittest.TestCase):
    def setUp(self):
        self.src = open("nodriver_engine.py").read()

    def test_post_data_is_base64_encoded(self):
        i = self.src.index("post_data=enc")
        self.assertIn("b64encode", self.src[max(0, i - 400):i])

    def test_continue_has_a_fallback(self):
        i = self.src.index("async def intercept_request_bodies")
        block = self.src[i:i + 3000]
        self.assertGreaterEqual(block.count("continue_request"), 3)

    def test_patterns_are_scoped_not_wildcard(self):
        i = self.src.index("async def intercept_request_bodies")
        block = self.src[i:i + 3000]
        self.assertIn('url_pattern=f"*{sub}*"', block)


@unittest.skipUnless(HAS_CDP, "nodriver not installed")
class TestNeverStallsThePage(unittest.IsolatedAsyncioTestCase):
    """Fetch.enable pauses requests; a missed continue hangs the page.

    Live regression: arming the interceptor at page creation produced
    'Form never rendered after 75s' followed by a dead-CDP rebuild loop.
    """

    async def _fire(self, page, rid="R1"):
        class Req:
            url = "https://discord.com/api/v9/auth/register"
            post_data = '{"e":1}'

        class Ev:
            request_id = rid
            request = Req()

        page._tab.handlers[cdp.fetch.RequestPaused][0](Ev())
        await asyncio.sleep(0.15)

    async def test_continues_when_mutate_raises(self):
        page = make_page()

        def boom(u, b):
            raise RuntimeError("boom")

        await page.intercept_request_bodies(["/auth/register"], boom)
        await self._fire(page)
        self.assertGreaterEqual(len(page._tab.sent), 2)

    async def test_continues_when_mutate_returns_none(self):
        page = make_page()
        await page.intercept_request_bodies(["/auth/register"],
                                            lambda u, b: None)
        await self._fire(page)
        self.assertGreaterEqual(len(page._tab.sent), 2)

    async def test_missing_request_id_is_safe(self):
        page = make_page()
        await page.intercept_request_bodies(["/auth/register"],
                                            lambda u, b: None)

        class Bad:
            request_id = None

        page._tab.handlers[cdp.fetch.RequestPaused][0](Bad())
        await asyncio.sleep(0.1)

    async def test_can_be_disabled(self):
        page = make_page()
        await page.intercept_request_bodies(["/auth/register"],
                                            lambda u, b: None)
        self.assertTrue(await page.disable_request_interception())


class TestNotArmedAtPageCreation(unittest.TestCase):
    def setUp(self):
        self.src = open("server.py").read()

    def test_early_install_does_not_arm_fetch(self):
        i = self.src.index("async def _install_captcha_hook_early")
        block = self.src[i:i + 1400]
        self.assertNotIn("_install_cdp_captcha_interceptor()", block)

    def test_disarm_exists_and_runs_after_the_submit(self):
        self.assertIn("_disarm_cdp_captcha_interceptor", self.src)
        i = self.src.index("_direct_register_with_token(\n")
        self.assertIn("_disarm_cdp_captcha_interceptor",
                      self.src[i:i + 400])

    def test_hook_failure_is_not_a_warning(self):
        i = self.src.index("Register hook deferred")
        self.assertIn('level="debug"', self.src[i:i + 200])


if __name__ == "__main__":
    unittest.main(verbosity=2)

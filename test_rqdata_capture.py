#!/usr/bin/env python3
"""Regression: enterprise rqdata must actually be captured.

Discord binds every hCaptcha token to a per-challenge rqdata blob, so a
token solved without it is rejected. The capture failed because Chrome
omits POST bodies from Network.requestWillBeSent unless Network.enable is
called with maxPostDataSize.
"""
import asyncio
import json
import unittest

from captcha_solver import extract_rqdata_from_body


class TestCdpPostDataIsRequested(unittest.TestCase):
    def setUp(self):
        self.src = open("nodriver_engine.py").read()

    def test_network_enable_asks_for_post_bodies(self):
        self.assertIn("max_post_data_size=_MAX_POST_DATA", self.src)

    def test_both_enable_sites_pass_it(self):
        self.assertEqual(
            self.src.count("max_post_data_size=_MAX_POST_DATA"), 2,
            "every Network.enable must request post bodies")

    def test_budget_is_big_enough_for_an_enterprise_payload(self):
        import re
        m = re.search(r"_MAX_POST_DATA = (\d+)", self.src)
        self.assertIsNotNone(m)
        self.assertGreaterEqual(int(m.group(1)), 65536)

    def test_request_exposes_a_cdp_body_fetch(self):
        self.assertIn("async def fetch_post_data", self.src)
        self.assertIn("get_request_post_data", self.src)

    def test_request_receives_the_tab(self):
        self.assertIn("_Request(e, self._tab)", self.src)


class TestServerUsesTheFetch(unittest.TestCase):
    def setUp(self):
        self.src = open("server.py").read()

    def test_handler_is_async(self):
        self.assertIn("async def _on_page_request", self.src)

    def test_falls_back_to_fetching_the_body(self):
        self.assertIn("fetch_post_data", self.src)

    def test_fetch_is_time_boxed(self):
        i = self.src.index("fetch_post_data")
        self.assertIn("wait_for", self.src[i:i + 400])


class TestExtraction(unittest.TestCase):
    """The parser must handle every shape hCaptcha actually sends."""

    def test_json_body(self):
        body = json.dumps({"sitekey": "a9b5", "rqdata": "RQ_1234567890abc"})
        self.assertEqual(extract_rqdata_from_body(body), "RQ_1234567890abc")

    def test_urlencoded_bytes_body(self):
        body = b"sitekey=a9b5&rqdata=RQ_abcdef1234567&host=discord.com"
        self.assertEqual(extract_rqdata_from_body(body), "RQ_abcdef1234567")

    def test_nested_enterprise_payload(self):
        body = json.dumps({"enterprisePayload": {"rqdata": "NESTED_1234567"}})
        self.assertEqual(extract_rqdata_from_body(body), "NESTED_1234567")

    def test_jwt_shaped_blob_survives(self):
        blob = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.abc-_123"
        self.assertEqual(
            extract_rqdata_from_body(json.dumps({"rqdata": blob})), blob)

    def test_absent_rqdata_returns_empty(self):
        self.assertEqual(extract_rqdata_from_body('{"sitekey":"x"}'), "")
        self.assertEqual(extract_rqdata_from_body(None), "")


class TestAsyncHandlerScheduling(unittest.TestCase):
    def test_async_handlers_are_scheduled(self):
        import nodriver_engine as ne
        seen = []

        async def handler(req):
            await asyncio.sleep(0)
            seen.append(req.url)

        async def main():
            ne._call_handler(
                handler,
                type("R", (), {"url": "https://hcaptcha.com/getcaptcha/x"})())
            await asyncio.sleep(0.1)

        asyncio.run(main())
        self.assertEqual(seen, ["https://hcaptcha.com/getcaptcha/x"])


class TestDiscordChallengeResponse(unittest.IsolatedAsyncioTestCase):
    """Discord returns rqdata in ITS OWN 400, not in hCaptcha's request."""

    BODY = {
        "captcha_key": ["captcha-required"],
        "captcha_sitekey": "a9b5fb07-92ff-493f-86fe-352a2803b3df",
        "captcha_service": "hcaptcha",
        "captcha_rqdata": "eyJhbGciOiJIUzI1NiJ9.RQ_BLOB_1234567890abcdef",
        "captcha_rqtoken": "RQTOKEN_abcdef123456",
    }

    class _Resp:
        def __init__(self, url, status, body):
            self.url = url
            self.status = status
            self._b = body

        async def json(self):
            return self._b

    class _Bot:
        _rqdata = ""
        _rqtoken = ""
        _hcaptcha_sitekey = ""
        _challenge_payload = None

        def _log(self, *a, **k):
            pass

        def _read_challenge_payload(self, data=None):
            return self._challenge_payload

    async def _run(self, url, status, body):
        import server
        bot = self._Bot()
        await server.DiscordAutomation._on_page_response(
            bot, self._Resp(url, status, body))
        return bot

    async def test_captures_rqdata_from_the_400(self):
        bot = await self._run("https://discord.com/api/v9/auth/register",
                              400, self.BODY)
        self.assertEqual(bot._rqdata, self.BODY["captcha_rqdata"])

    async def test_captures_rqtoken(self):
        bot = await self._run("https://discord.com/api/v9/auth/register",
                              400, self.BODY)
        self.assertEqual(bot._rqtoken, "RQTOKEN_abcdef123456")

    async def test_captures_the_sitekey_too(self):
        bot = await self._run("https://discord.com/api/v9/auth/register",
                              400, self.BODY)
        self.assertEqual(bot._hcaptcha_sitekey,
                         self.BODY["captcha_sitekey"])

    async def test_successful_200_is_not_a_challenge(self):
        bot = await self._run("https://discord.com/api/v9/auth/register",
                              200, {"token": "ok"})
        self.assertEqual(bot._rqdata, "")

    async def test_unrelated_host_ignored(self):
        bot = await self._run("https://example.com/api/v9/auth/register",
                              400, self.BODY)
        self.assertEqual(bot._rqdata, "")

    async def test_hcaptcha_payload_path_still_works(self):
        import server
        bot = self._Bot()
        seen = {}
        bot._read_challenge_payload = lambda data=None: seen.update(
            data or {})
        await server.DiscordAutomation._on_page_response(
            bot, self._Resp("https://hcaptcha.com/getcaptcha/x", 200,
                            {"request_type": "image_drag_drop"}))
        self.assertEqual(seen.get("request_type"), "image_drag_drop")

    async def test_429_also_carries_a_challenge(self):
        bot = await self._run("https://discord.com/api/v9/auth/register",
                              429, self.BODY)
        self.assertEqual(bot._rqdata, self.BODY["captcha_rqdata"])

    async def test_body_read_is_retried(self):
        """responseReceived can beat the body into the CDP store."""
        src = open("server.py").read()
        i = src.index("Discord's captcha challenge response")
        block = src[i:i + 2500]
        self.assertIn("for _try in range(3)", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)

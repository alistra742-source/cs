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


if __name__ == "__main__":
    unittest.main(verbosity=2)

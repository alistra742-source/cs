#!/usr/bin/env python3
"""Offline tests for the Roboflow vision client (readiness + request shape)."""

from __future__ import annotations

import asyncio
import base64
import unittest
from unittest import mock

from vision_solver import RoboflowVisionClient


class _FakeResponse:
    def __init__(self, status: int, text: str = "", payload=None):
        self.status = status
        self._text = text
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text

    async def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, response: _FakeResponse, seen: dict | None = None):
        self._response = response
        self._seen = seen if seen is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, **kwargs):
        self._seen["url"] = url
        self._seen["json"] = json
        return self._response


class TestWorkflowCheck(unittest.IsolatedAsyncioTestCase):
    def _client(self, api_key="rf_test", **kw):
        return RoboflowVisionClient(
            api_key=api_key,
            log=lambda message, level="info": None,
            **kw)

    async def _check(self, response: _FakeResponse, *, api_key="rf_test"):
        client = self._client(api_key)
        with mock.patch("vision_solver.aiohttp.ClientSession",
                        return_value=_FakeSession(response)):
            result = await client.check()
        return client, result

    async def test_endpoint_url(self):
        client = self._client(workspace="text-detectioin",
                              workflow="gemini-3-6-flash")
        self.assertEqual(
            client.endpoint,
            "https://serverless.roboflow.com/infer/workflows/"
            "text-detectioin/gemini-3-6-flash")

    async def test_success(self):
        client, result = await self._check(_FakeResponse(200, "{}"))
        self.assertEqual(result, (True, [client.model]))
        self.assertEqual(client.last_check_error, "")
        self.assertIsNone(client.last_check_http_status)

    async def test_missing_api_key_is_authentication(self):
        client, result = await self._check(_FakeResponse(200, "{}"), api_key="")
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "authentication")

    async def test_401_is_authentication(self):
        client, result = await self._check(_FakeResponse(401, "bad key"))
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "authentication")
        self.assertEqual(client.last_check_http_status, 401)

    async def test_404_is_wrong_workflow(self):
        client, result = await self._check(_FakeResponse(404, "not found"))
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "protocol")

    async def test_429_is_rate_limit(self):
        client, result = await self._check(_FakeResponse(429, "slow down"))
        self.assertEqual(client.last_check_error, "rate_limit")
        self.assertEqual(result, (False, []))

    async def test_429_is_rate_limit(self):
        client, result, _ = await self._check(_FakeResponse(429, "slow down"))
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "rate_limit")

    async def test_timeout_is_transient(self):
        client = self._client()

        class _TimeoutSession(_FakeSession):
            def post(self, *a, **k):
                raise asyncio.TimeoutError

        with mock.patch("vision_solver.aiohttp.ClientSession",
                        return_value=_TimeoutSession(_FakeResponse(200))):
            result = await client.check()
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "timeout")


class TestWorkflowRequest(unittest.IsolatedAsyncioTestCase):
    async def test_image_and_question_are_both_sent(self):
        client = RoboflowVisionClient(api_key="rf_test",
                                      workspace="text-detectioin",
                                      workflow="gemini-3-6-flash",
                                      log=lambda *a, **k: None)
        seen: dict = {}
        payload = {"outputs": [{"predictions": {"predictions": [
            {"x": 32, "y": 32, "width": 8, "height": 8,
             "confidence": 0.91, "class": "boat"}]}}]}
        response = _FakeResponse(200, payload=payload)

        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (10, 20, 30)).save(buf, format="JPEG")

        with mock.patch("vision_solver.aiohttp.ClientSession",
                        return_value=_FakeSession(response, seen)):
            got = await client.solve(
                "Please click each image containing a boat",
                [buf.getvalue()], shape="points")

        self.assertEqual(got["type"], "points")
        self.assertEqual(seen["url"], client.endpoint)
        body = seen["json"]
        # Roboflow auth goes in the body, never a header.
        self.assertEqual(body["api_key"], "rf_test")
        inputs = body["inputs"]
        # The image is attached as base64...
        self.assertEqual(inputs["image"]["type"], "base64")
        base64.b64decode(inputs["image"]["value"])
        # ...and the captcha question rides along with it.
        self.assertIn("boat", inputs["prompt"].lower())
        self.assertIn("boat", inputs["classes"])

    async def test_google_key_is_forwarded_as_model_api_key(self):
        client = RoboflowVisionClient(api_key="rf_test",
                                      google_api_key="goog_test",
                                      log=lambda *a, **k: None)
        inputs = client._inputs(b"img", "question?")
        self.assertEqual(inputs["model_api_key"], "goog_test")

    async def test_no_google_key_omits_model_api_key(self):
        client = RoboflowVisionClient(api_key="rf_test", google_api_key="",
                                      log=lambda *a, **k: None)
        inputs = client._inputs(b"img", "question?")
        self.assertNotIn("model_api_key", inputs)

    async def test_non_200_returns_none(self):
        client = RoboflowVisionClient(api_key="rf_test",
                                      log=lambda *a, **k: None)
        with mock.patch("vision_solver.aiohttp.ClientSession",
                        return_value=_FakeSession(_FakeResponse(500, "boom"))):
            got = await client.solve("x", [b"img"], shape="points")
        self.assertIsNone(got)


if __name__ == "__main__":
    unittest.main()

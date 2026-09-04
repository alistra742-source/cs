#!/usr/bin/env python3
"""Offline tests for the Hugging Face vision client (readiness + errors)."""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from vision_solver import HFVisionClient


class _FakeResponse:
    def __init__(self, status: int, text: str = ""):
        self.status = status
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, *args, **kwargs):
        return self._response


class TestVisionEndpointCheck(unittest.IsolatedAsyncioTestCase):
    async def _check(self, response: _FakeResponse, *, api_key: str = "hf_test"):
        logs = []
        client = HFVisionClient(
            base="https://api-inference.huggingface.co/models",
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            api_key=api_key,
            log=lambda message, level="info": logs.append((level, message)),
        )
        with mock.patch(
            "vision_solver.aiohttp.ClientSession",
            return_value=_FakeSession(response),
        ):
            result = await client.check()
        return client, result, logs

    async def test_endpoint_url(self):
        client = HFVisionClient(model="Qwen/Qwen2.5-VL-7B-Instruct",
                                api_key="hf_test")
        self.assertEqual(
            client.endpoint,
            "https://api-inference.huggingface.co/models/"
            "Qwen/Qwen2.5-VL-7B-Instruct/v1/chat/completions")

    async def test_success_reports_model(self):
        client, result, _ = await self._check(_FakeResponse(200, "{}"))
        self.assertEqual(result, (True, ["Qwen/Qwen2.5-VL-7B-Instruct"]))
        self.assertEqual(client.last_check_error, "")
        self.assertIsNone(client.last_check_http_status)

    async def test_cold_start_503_is_still_ok(self):
        client, result, logs = await self._check(
            _FakeResponse(503, "model is currently loading"))
        self.assertEqual(result, (True, ["Qwen/Qwen2.5-VL-7B-Instruct"]))
        self.assertEqual(client.last_check_error, "")
        self.assertTrue(any("cold start" in line for _, line in logs))

    async def test_missing_api_key_is_authentication(self):
        client, result, logs = await self._check(
            _FakeResponse(200, "{}"), api_key="")
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "authentication")
        self.assertTrue(any("API_KEY" in line for _, line in logs))

    async def test_401_is_authentication_not_unreachable(self):
        client, result, logs = await self._check(_FakeResponse(401, "bad token"))
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "authentication")
        self.assertEqual(client.last_check_http_status, 401)
        self.assertTrue(any("API_KEY" in line for _, line in logs))

    async def test_404_is_terminal_protocol_mismatch(self):
        client, result, _ = await self._check(_FakeResponse(404, "not found"))
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "protocol")
        self.assertEqual(client.last_check_http_status, 404)

    async def test_429_is_rate_limit(self):
        client, result, _ = await self._check(_FakeResponse(429, "slow down"))
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "rate_limit")

    async def test_timeout_is_transient(self):
        client = HFVisionClient(api_key="hf_test")

        class _TimeoutSession(_FakeSession):
            def get(self, *args, **kwargs):
                raise asyncio.TimeoutError

        with mock.patch(
            "vision_solver.aiohttp.ClientSession",
            return_value=_TimeoutSession(_FakeResponse(200)),
        ):
            result = await client.check()
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "timeout")
        self.assertIsNone(client.last_check_http_status)


class TestChatPayload(unittest.IsolatedAsyncioTestCase):
    async def test_images_are_sent_as_data_uri_parts(self):
        client = HFVisionClient(api_key="hf_test")
        seen = {}

        class _PostResponse(_FakeResponse):
            async def json(self):
                return {"choices": [{"message": {"content": '{"tiles": [1]}'}}]}

        class _PostSession(_FakeSession):
            def post(self, url, json=None, headers=None):
                seen["url"] = url
                seen["json"] = json
                seen["headers"] = headers
                return self._response

        with mock.patch(
            "vision_solver.aiohttp.ClientSession",
            return_value=_PostSession(_PostResponse(200)),
        ):
            got = await client.solve("select boats", [b"\x89PNG\r\n\x1a\n"])
        self.assertEqual(got, {"type": "tiles", "indices": [1]})
        self.assertEqual(seen["url"], client.endpoint)
        self.assertEqual(seen["headers"]["Authorization"], "Bearer hf_test")
        parts = seen["json"]["messages"][1]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(
            parts[1]["image_url"]["url"].startswith("data:image/"))


if __name__ == "__main__":
    unittest.main()

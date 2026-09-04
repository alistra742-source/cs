#!/usr/bin/env python3
"""Tests for the NoneCap client, including a real local HTTP server."""
import asyncio
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import nonecap_solver
from nonecap_solver import NoneCapSolver, api_key, configured


# ── a stand-in NoneCap API ──────────────────────────────────────────────
STATE = {"mode": "inline", "polls": 0, "last_payload": None,
         "last_auth": "", "cancelled": []}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        STATE["last_auth"] = self.headers.get("Authorization", "")
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if self.path.endswith("/cancel"):
            STATE["cancelled"].append(self.path)
            return self._send(200, {"status": "cancelled"})
        if "/feedback" in self.path:
            return self._send(204, {})
        STATE["last_payload"] = body
        mode = STATE["mode"]
        if mode == "inline":
            return self._send(200, {"id": "s1", "status": "solved",
                                    "token": "P1_" + "x" * 40,
                                    "credits_charged": 1})
        if mode == "poll":
            return self._send(202, {"id": "s2", "status": "pending",
                                    "token": None})
        if mode == "failed":
            return self._send(200, {"id": "s3", "status": "failed",
                                    "token": None,
                                    "error": {"message": "unsolvable"}})
        if mode == "auth":
            return self._send(401, {"error": "bad key"})
        if mode == "credits":
            return self._send(402, {"error": "no credits"})
        if mode == "ratelimit":
            return self._send(429, {"error": "slow down"})
        return self._send(500, {"error": "boom"})

    def do_GET(self):
        if self.path.startswith("/v1/me"):
            return self._send(200, {"credits_balance": 1234})
        STATE["polls"] += 1
        if STATE["polls"] >= 2:
            return self._send(200, {"id": "s2", "status": "solved",
                                    "token": "P1_" + "y" * 40,
                                    "credits_charged": 2})
        return self._send(200, {"id": "s2", "status": "solving",
                                "token": None})


class TestNoneCapAgainstServer(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.srv.server_port
        cls.t = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.t.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        STATE.update({"mode": "inline", "polls": 0, "last_payload": None,
                      "cancelled": []})

    def _client(self):
        return NoneCapSolver(key="nc_live_test", base=self.base,
                             log=lambda *a, **k: None)

    async def test_inline_token(self):
        tok = await self._client().solve("sk-uuid", "https://x.test/")
        self.assertTrue(tok.startswith("P1_"))

    async def test_payload_shape(self):
        await self._client().solve("sk-uuid", "https://x.test/login")
        p = STATE["last_payload"]
        self.assertEqual(p["type"], "hcaptcha")
        self.assertEqual(p["sitekey"], "sk-uuid")
        self.assertEqual(p["url"], "https://x.test/login")

    async def test_bearer_header(self):
        await self._client().solve("sk", "https://x.test/")
        self.assertEqual(STATE["last_auth"], "Bearer nc_live_test")

    async def test_enterprise_uses_rqdata(self):
        await self._client().solve("sk", "https://x.test/", rqdata="RQ123")
        p = STATE["last_payload"]
        self.assertEqual(p["type"], "hcaptcha_enterprise")
        self.assertEqual(p["rqdata"], "RQ123")

    async def test_202_then_poll(self):
        STATE["mode"] = "poll"
        tok = await self._client().solve("sk", "https://x.test/")
        self.assertTrue(tok.startswith("P1_"))
        self.assertGreaterEqual(STATE["polls"], 2)

    async def test_failed_solve_returns_none(self):
        STATE["mode"] = "failed"
        c = self._client()
        self.assertIsNone(await c.solve("sk", "https://x.test/"))
        self.assertIn("unsolvable", c.last_error)
        self.assertEqual(c.failures, 1)

    async def test_auth_error_is_flagged(self):
        STATE["mode"] = "auth"
        c = self._client()
        self.assertIsNone(await c.solve("sk", "https://x.test/"))
        self.assertEqual(c.last_error, "authentication")

    async def test_insufficient_credits_flagged(self):
        STATE["mode"] = "credits"
        c = self._client()
        self.assertIsNone(await c.solve("sk", "https://x.test/"))
        self.assertEqual(c.last_error, "insufficient credits")

    async def test_rate_limit_flagged(self):
        STATE["mode"] = "ratelimit"
        c = self._client()
        self.assertIsNone(await c.solve("sk", "https://x.test/"))
        self.assertEqual(c.last_error, "rate limited")

    async def test_missing_sitekey_never_calls_out(self):
        c = self._client()
        self.assertIsNone(await c.solve("", "https://x.test/"))
        self.assertIsNone(STATE["last_payload"])

    async def test_balance(self):
        self.assertEqual(await self._client().balance(), 1234)

    async def test_credits_counted(self):
        c = self._client()
        await c.solve("sk", "https://x.test/")
        self.assertEqual(c.credits_charged, 1)
        self.assertEqual(c.solves, 1)


class TestKeyHandling(unittest.TestCase):
    def setUp(self):
        for v in ("NONECAP_API", "NONECAP_API_KEY", "NONECAP_KEY"):
            os.environ.pop(v, None)

    tearDown = setUp

    def test_reads_nonecap_api(self):
        os.environ["NONECAP_API"] = "nc_live_abc"
        self.assertEqual(api_key(), "nc_live_abc")

    def test_accepts_aliases(self):
        os.environ["NONECAP_KEY"] = "nc_live_alias"
        self.assertEqual(api_key(), "nc_live_alias")

    def test_strips_pasted_name_equals_value(self):
        os.environ["NONECAP_API"] = "NONECAP_API = nc_live_pasted"
        self.assertEqual(api_key(), "nc_live_pasted")

    def test_strips_quotes(self):
        os.environ["NONECAP_API"] = "'nc_live_quoted'"
        self.assertEqual(api_key(), "nc_live_quoted")

    def test_not_configured_without_a_key(self):
        self.assertFalse(configured())

    def test_disabled_client_never_solves(self):
        c = NoneCapSolver(key="", log=lambda *a, **k: None)
        self.assertFalse(c.enabled)
        self.assertIsNone(asyncio.run(c.solve("sk", "https://x.test/")))

    def test_three_tries_by_default(self):
        self.assertEqual(nonecap_solver.NONECAP_TRIES, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

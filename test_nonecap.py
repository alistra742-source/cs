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


class TestNullCredits(unittest.IsolatedAsyncioTestCase):
    """credits_charged can be null; that must not crash or print 'None'."""

    async def test_null_credits_logged_cleanly(self):
        from nonecap_solver import NoneCapSolver
        lines = []
        c = NoneCapSolver(key="k", log=lambda m, **kw: lines.append(m))

        async def fake_await(session, solve, deadline):
            return await NoneCapSolver._await_token(c, session, solve,
                                                    deadline)

        solve = {"id": "s", "status": "solved", "token": "P1_" + "z" * 30,
                 "credits_charged": None}
        tok = await NoneCapSolver._await_token(c, None, solve, 1e18)
        self.assertTrue(tok.startswith("P1_"))
        self.assertTrue(any("credits n/a" in l for l in lines), lines)
        self.assertFalse(any("None credit" in l for l in lines), lines)

    async def test_integer_credits_still_counted(self):
        from nonecap_solver import NoneCapSolver
        c = NoneCapSolver(key="k", log=lambda *a, **k: None)
        solve = {"id": "s", "status": "solved", "token": "P1_" + "z" * 30,
                 "credits_charged": 3}
        await NoneCapSolver._await_token(c, None, solve, 1e18)
        self.assertEqual(c.credits_charged, 3)


class TestServerWiring(unittest.TestCase):
    """Guard the integration points that made the live run fail."""

    def setUp(self):
        self.src = open("server.py").read()

    def test_rqdata_is_read_live_not_just_from_the_hook(self):
        self.assertIn("_live_rqdata", self.src)
        self.assertIn("rqdata = await self._live_rqdata()", self.src)

    def test_warns_when_solving_without_rqdata(self):
        self.assertIn("No enterprise rqdata found", self.src)

    def test_token_injection_uses_the_native_react_setter(self):
        i = self.src.find("_INJECT_TOKEN_JS")
        block = self.src[i:i + 4000]
        self.assertIn("getOwnPropertyDescriptor", block)
        self.assertIn("HTMLTextAreaElement", block)

    def test_token_injection_fires_the_widget_callback(self):
        i = self.src.find("_INJECT_TOKEN_JS")
        block = self.src[i:i + 4000]
        self.assertIn("callback", block)
        self.assertIn("getConfig", block)

    def test_nonecap_runs_before_the_vision_probe(self):
        a = self.src.index("_solve_with_nonecap()")
        b = self.src.index('if not getattr(self, "_vision_ready", False)')
        self.assertLess(a, b)


class TestNoStalls(unittest.TestCase):
    """The captcha step must never hang, whatever the page does."""

    def test_live_rqdata_respects_its_budget(self):
        """A cross-origin frame can accept evaluate() and never answer."""
        import asyncio as aio
        import time as _t
        import server

        class Hanging:
            async def evaluate(self, js):
                await aio.sleep(3600)

        class Bot:
            _RQDATA_JS = server.DiscordAutomation._RQDATA_JS
            _rqdata = ""
            def __init__(self):
                self._page = Hanging()
                self._page.frames = [Hanging() for _ in range(5)]
            def _log(self, *a, **k):
                pass
            def _read_challenge_payload(self, data=None):
                return {}

        t0 = _t.time()
        out = aio.run(
            server.DiscordAutomation._live_rqdata(Bot(), budget=3.0))
        self.assertEqual(out, "")
        self.assertLess(_t.time() - t0, 9.0, "rqdata search hung")

    def test_every_page_call_in_the_path_is_timed_out(self):
        src = open("server.py").read()
        block = src[src.index("async def _solve_with_nonecap"):
                    src.index("async def _challenge_surface")]
        # Check the CALL SITES (evaluate/await), not the JS definition.
        for needle in ("extract_hcaptcha_sitekey(self._page)",
                       'evaluate("() => location.href")',
                       "evaluate(self._INJECT_TOKEN_JS, token)",
                       "self._click_form_submit()"):
            i = block.find(needle)
            self.assertNotEqual(i, -1, f"call site missing: {needle}")
            self.assertIn("wait_for", block[max(0, i - 220):i + 80],
                          f"{needle} is not time-boxed")

    def test_step_has_a_hard_budget(self):
        import server
        self.assertGreater(server.NONECAP_STEP_BUDGET, 0)
        src = open("server.py").read()
        self.assertIn("NONECAP_STEP_BUDGET", src)
        i = src.index("_solve_with_nonecap(), timeout=")
        self.assertIn("NONECAP_STEP_BUDGET", src[i:i + 80])

    def test_skip_reason_is_logged(self):
        """Silence is what made it look like NoneCap was never wired in."""
        src = open("server.py").read()
        self.assertIn("[NoneCap] Skipped:", src)
        self.assertIn("[NoneCap] Starting hosted solve", src)


class TestIpAndUaBinding(unittest.IsolatedAsyncioTestCase):
    """hCaptcha binds a token to the solving IP/UA.

    Live evidence: with the token finally reaching Discord, the verdict was
    'invalid-response' — the token was real but solved on NoneCap's IP
    while we submit from a TOR exit.
    """

    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Handler)
        cls.t = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.t.start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        STATE.update({"mode": "inline", "polls": 0, "last_payload": None})

    def _c(self):
        return NoneCapSolver(key="k", base=self.base,
                             log=lambda *a, **kw: None)

    async def test_proxy_is_forwarded(self):
        await self._c().solve("sk", "https://x.test/",
                              proxy="socks5://1.2.3.4:1080")
        self.assertEqual(STATE["last_payload"]["proxy"],
                         "socks5://1.2.3.4:1080")

    async def test_user_agent_is_forwarded(self):
        await self._c().solve("sk", "https://x.test/", user_agent="UA/1.0")
        self.assertEqual(STATE["last_payload"]["user_agent"], "UA/1.0")

    async def test_absent_by_default(self):
        await self._c().solve("sk", "https://x.test/")
        self.assertNotIn("proxy", STATE["last_payload"])
        self.assertNotIn("user_agent", STATE["last_payload"])

    async def test_env_proxy_is_used(self):
        import nonecap_solver as ns
        old = ns.NONECAP_PROXY
        ns.NONECAP_PROXY = "http://env:9"
        try:
            await self._c().solve("sk", "https://x.test/")
            self.assertEqual(STATE["last_payload"]["proxy"], "http://env:9")
        finally:
            ns.NONECAP_PROXY = old

    def test_server_sends_the_browser_ua(self):
        src = open("server.py").read()
        self.assertIn("navigator.userAgent", src)
        self.assertIn("user_agent=str(ua)", src)


class TestNoSelfInjection(unittest.TestCase):
    """Our own direct submit must not be rewritten by our interceptors."""

    def setUp(self):
        self.src = open("server.py").read()

    def test_direct_request_sets_an_inflight_flag(self):
        # The old body marker leaked to Discord as an unknown field, so
        # the guard is now an in-flight flag on both sides.
        self.assertIn("__ncDirectInflight", self.src)
        self.assertIn("_nc_direct_inflight", self.src)

    def test_marker_no_longer_leaks_into_the_body(self):
        self.assertNotIn("__nc_direct:", self.src)

    def test_cdp_mutator_skips_our_own_request(self):
        i = self.src.index("def _mutate_register_body")
        block = self.src[i:i + 1200]
        self.assertIn("_nc_direct_inflight", block)

    def test_js_hook_skips_marked_bodies(self):
        i = self.src.index("_CAPTCHA_HOOK_JS")
        block = self.src[i:i + 4000]
        self.assertIn("own-direct-request", block)


class TestEgressForwarding(unittest.TestCase):
    """rqdata is IP-bound: the solver must mint from OUR exit IP.

    Confirmed by practitioners running this exact flow — 'the rqdata blob
    is welded to discord.com AND to the exact exit IP that asked for the
    challenge'. Solver mints on its own IP -> invalid-response.
    """

    def _url(self, proxy):
        import server

        class B:
            pass

        b = B()
        b.proxy = proxy
        return server.DiscordAutomation._solver_proxy_url(b)

    def test_tor_has_no_shareable_egress(self):
        self.assertEqual(self._url(None), "")

    def test_auth_proxy_is_rendered_with_credentials(self):
        self.assertEqual(
            self._url({"server": "http://gate:8080", "username": "u",
                       "password": "p"}),
            "http://u:p@gate:8080")

    def test_inline_credentials_are_left_alone(self):
        self.assertEqual(self._url({"server": "http://u:p@gate:8080"}),
                         "http://u:p@gate:8080")

    def test_socks5_is_supported(self):
        self.assertEqual(
            self._url({"server": "socks5://gate:1080", "username": "u",
                       "password": "p"}),
            "socks5://u:p@gate:1080")

    def test_env_override_wins(self):
        import os
        os.environ["NONECAP_PROXY"] = "http://override:1"
        try:
            self.assertEqual(self._url(None), "http://override:1")
        finally:
            del os.environ["NONECAP_PROXY"]

    def test_egress_is_passed_to_the_solver(self):
        src = open("server.py").read()
        self.assertIn("proxy=egress", src)

    def test_missing_egress_is_called_out(self):
        src = open("server.py").read()
        self.assertIn("No shareable egress", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)

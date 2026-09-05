#!/usr/bin/env python3
"""Chrome cannot authenticate to a proxy from the command line.

There is no --proxy-user/--proxy-pass switch; credentials in
--proxy-server are stripped. An authenticated gateway therefore answers
407 and Chrome lands on chrome-error://chromewebdata/ in 0.0s — the exact
"PROXY SESSION DEAD" loop, on sessions that validate fine from aiohttp.

proxy_relay.py fixes it with a local unauthenticated port that injects
Proxy-Authorization upstream.
"""
import asyncio
import base64
import unittest

from proxy_relay import ProxyRelay, parse_upstream, relay_for


class TestParseUpstream(unittest.TestCase):
    def test_full_url(self):
        self.assertEqual(
            parse_upstream("http://u:p@gate.example:8080"),
            ("gate.example", 8080, "u", "p"))

    def test_scheme_optional(self):
        self.assertEqual(parse_upstream("gate.example:80")[0], "gate.example")

    def test_default_port(self):
        self.assertEqual(parse_upstream("http://gate.example")[1], 80)

    def test_junk(self):
        self.assertIsNone(parse_upstream(""))


class TestRelay(unittest.IsolatedAsyncioTestCase):
    async def _upstream(self, require_auth=True):
        seen = {}

        async def handler(reader, writer):
            try:
                head = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=3)
            except Exception:
                writer.close()
                return
            seen["head"] = head.decode()
            if require_auth and b"Proxy-Authorization" not in head:
                writer.write(b"HTTP/1.1 407 Proxy Authentication Required"
                             b"\r\n\r\n")
                await writer.drain()
                writer.close()
                return
            writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await writer.drain()
            while True:
                d = await reader.read(4096)
                if not d:
                    break
                writer.write(b"ECHO:" + d)
                await writer.drain()

        srv = await asyncio.start_server(handler, "127.0.0.1", 0)
        return srv, srv.sockets[0].getsockname()[1], seen

    async def test_chrome_connects_without_credentials(self):
        srv, port, seen = await self._upstream()
        relay = ProxyRelay("127.0.0.1", port, "myuser", "mypass")
        await relay.start()
        try:
            r, w = await asyncio.open_connection("127.0.0.1", relay.port)
            w.write(b"CONNECT discord.com:443 HTTP/1.1\r\n"
                    b"Host: discord.com:443\r\n\r\n")
            await w.drain()
            line = await asyncio.wait_for(r.readline(), timeout=5)
            self.assertIn(b"200", line)
            w.close()
        finally:
            await relay.stop()
            srv.close()

    async def test_upstream_receives_correct_credentials(self):
        srv, port, seen = await self._upstream()
        relay = ProxyRelay("127.0.0.1", port, "myuser", "mypass")
        await relay.start()
        try:
            r, w = await asyncio.open_connection("127.0.0.1", relay.port)
            w.write(b"CONNECT discord.com:443 HTTP/1.1\r\n\r\n")
            await w.drain()
            await asyncio.wait_for(r.readline(), timeout=5)
            creds = base64.b64encode(b"myuser:mypass").decode()
            self.assertIn(creds, seen["head"])
            w.close()
        finally:
            await relay.stop()
            srv.close()

    async def test_connect_target_is_preserved(self):
        srv, port, seen = await self._upstream()
        relay = ProxyRelay("127.0.0.1", port, "u", "p")
        await relay.start()
        try:
            r, w = await asyncio.open_connection("127.0.0.1", relay.port)
            w.write(b"CONNECT discord.com:443 HTTP/1.1\r\n\r\n")
            await w.drain()
            await asyncio.wait_for(r.readline(), timeout=5)
            self.assertTrue(seen["head"].startswith("CONNECT discord.com:443"))
            w.close()
        finally:
            await relay.stop()
            srv.close()

    async def test_tunnel_pipes_bytes(self):
        srv, port, _ = await self._upstream()
        relay = ProxyRelay("127.0.0.1", port, "u", "p")
        await relay.start()
        try:
            r, w = await asyncio.open_connection("127.0.0.1", relay.port)
            w.write(b"CONNECT discord.com:443 HTTP/1.1\r\n\r\n")
            await w.drain()
            await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), timeout=5)
            w.write(b"TLSDATA")
            await w.drain()
            echo = await asyncio.wait_for(r.read(32), timeout=5)
            self.assertIn(b"TLSDATA", echo)
            w.close()
        finally:
            await relay.stop()
            srv.close()

    async def test_no_relay_when_proxy_needs_no_auth(self):
        self.assertIsNone(await relay_for({"host": "h", "port": 80}))

    async def test_relay_built_from_pool_entry(self):
        srv, port, _ = await self._upstream()
        relay = await relay_for({"host": "127.0.0.1", "port": port,
                                 "username": "u", "password": "p"})
        try:
            self.assertIsNotNone(relay)
            self.assertTrue(relay.url.startswith("http://127.0.0.1:"))
        finally:
            if relay:
                await relay.stop()
            srv.close()


class TestServerUsesTheRelay(unittest.TestCase):
    def setUp(self):
        self.src = open("server.py").read()

    def test_launch_proxy_prefers_the_relay(self):
        i = self.src.index("def _launch_proxy")
        block = self.src[i:i + 1800]
        self.assertIn("_proxy_relay", block)
        self.assertIn("relay.url", block)

    def test_relay_starts_before_launch(self):
        for marker in ("args = launch_args(headless=self.headless)",):
            i = self.src.index(marker)
            self.assertIn("_start_proxy_relay()",
                          self.src[max(0, i - 200):i])

    def test_relay_is_stopped_and_restarted(self):
        self.assertIn("async def _stop_proxy_relay", self.src)
        i = self.src.index("async def _start_proxy_relay")
        self.assertIn("_stop_proxy_relay()", self.src[i:i + 400])


class TestConnectProbe(unittest.IsolatedAsyncioTestCase):
    """The probe must fail 407s that aiohttp silently passed."""

    async def _proxy(self, behaviour):
        async def h(reader, writer):
            try:
                head = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=3)
            except Exception:
                writer.close()
                return
            if behaviour == "needs_auth" and b"Proxy-Authorization" not in head:
                writer.write(b"HTTP/1.1 407 Proxy Auth Required\r\n\r\n")
            elif behaviour == "dead":
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            else:
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await writer.drain()
            writer.close()
        srv = await asyncio.start_server(h, "127.0.0.1", 0)
        return srv, srv.sockets[0].getsockname()[1]

    async def test_healthy_passes(self):
        import proxies
        srv, port = await self._proxy("ok")
        try:
            self.assertTrue(await proxies.ProxyPool._connect_probe(
                {"host": "127.0.0.1", "port": port,
                 "username": "u", "password": "p"}, timeout=3))
        finally:
            srv.close()

    async def test_407_without_credentials_fails(self):
        import proxies
        srv, port = await self._proxy("needs_auth")
        try:
            self.assertFalse(await proxies.ProxyPool._connect_probe(
                {"host": "127.0.0.1", "port": port}, timeout=3))
        finally:
            srv.close()

    async def test_dead_gateway_fails(self):
        import proxies
        srv, port = await self._proxy("dead")
        try:
            self.assertFalse(await proxies.ProxyPool._connect_probe(
                {"host": "127.0.0.1", "port": port,
                 "username": "u", "password": "p"}, timeout=3))
        finally:
            srv.close()

    async def test_unreachable_fails(self):
        import proxies
        self.assertFalse(await proxies.ProxyPool._connect_probe(
            {"host": "127.0.0.1", "port": 9,
             "username": "u", "password": "p"}, timeout=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)

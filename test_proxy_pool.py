#!/usr/bin/env python3
"""The residential pool must load AND reach the solver as the same session.

Discord's enterprise rqdata is IP-bound: the exit that requests the
challenge must be the exit that mints the token. That only holds if the
browser session and the URL forwarded to NoneCap are byte-identical.
"""
import unittest

import proxies


class TestPoolLoads(unittest.TestCase):
    def setUp(self):
        self.urls = proxies._vault_proxy_urls()

    def test_sessions_are_loaded(self):
        self.assertGreater(len(self.urls), 100)

    def test_sessions_are_unique(self):
        self.assertEqual(len(self.urls), len(set(self.urls)))

    def test_all_are_sticky(self):
        for u in self.urls:
            self.assertIn("-sess-", u)
            self.assertIn("-life-", u)

    def test_parsed_entries_are_complete(self):
        for entry in proxies.vault_proxies()[:20]:
            for key in ("host", "port", "username", "password"):
                self.assertTrue(entry.get(key), key)

    def test_pool_is_configured(self):
        self.assertTrue(proxies.configured())

    def test_tor_is_not_forced(self):
        self.assertFalse(proxies.tor_only())


class TestEgressMatchesBrowser(unittest.TestCase):
    """The whole point: solver and browser share one exit IP."""

    def test_forwarded_url_is_the_browser_session(self):
        import server
        entry = proxies.vault_proxies()[0]
        launch = {
            "server": f"{entry.get('proto', 'http')}://"
                      f"{entry['host']}:{entry['port']}",
            "username": entry["username"],
            "password": entry["password"],
        }

        class Bot:
            proxy = launch

        egress = server.DiscordAutomation._solver_proxy_url(Bot())
        self.assertIn(entry["username"], egress)
        self.assertIn(entry["password"], egress)
        self.assertIn(entry["host"], egress)

    def test_sticky_session_id_survives(self):
        import re
        import server
        entry = proxies.vault_proxies()[0]
        sess = re.search(r"-sess-([a-z0-9]+)-", entry["username"]).group(1)

        class Bot:
            proxy = {"server": f"http://{entry['host']}:{entry['port']}",
                     "username": entry["username"],
                     "password": entry["password"]}

        self.assertIn(f"-sess-{sess}-",
                      server.DiscordAutomation._solver_proxy_url(Bot()))


class TestTorIsDisabledWhenProxiesExist(unittest.TestCase):
    """A silent TOR downgrade breaks the IP binding AND flags the exit."""

    def test_app_tor_fallback_off(self):
        import app
        self.assertTrue(app.PROXY_FORCE)
        self.assertFalse(app.TOR_FALLBACK)

    def test_server_tor_not_allowed(self):
        import server
        self.assertFalse(server._tor_allowed())

    def test_tor_can_be_forced_back_on(self):
        import os
        import server
        os.environ["TOR_FALLBACK"] = "1"
        try:
            self.assertTrue(server._tor_allowed())
        finally:
            os.environ.pop("TOR_FALLBACK", None)

    def test_tor_launch_sites_are_gated(self):
        src = open("server.py").read()
        for i, line in enumerate(src.splitlines()):
            if 'socks5://127.0.0.1:9050' in line and 'launch_proxy = {' in line:
                window = "\n".join(src.splitlines()[max(0, i - 4):i])
                self.assertIn("_tor_allowed()", window,
                              f"ungated TOR launch at line {i + 1}")

    def test_dead_session_limit_scales_with_pool(self):
        """4 dead sessions out of 663 must not condemn the pool."""
        src = open("app.py").read()
        self.assertIn("_dead_limit", src)
        i = src.index("_dead_limit")
        self.assertIn("PROXY_FORCE", src[i:i + 200])


class TestNoTorInTheSolverPath(unittest.TestCase):
    def test_egress_is_never_a_local_socks_port(self):
        import server
        entry = proxies.vault_proxies()[0]

        class Bot:
            proxy = {"server": f"http://{entry['host']}:{entry['port']}",
                     "username": entry["username"],
                     "password": entry["password"]}

        egress = server.DiscordAutomation._solver_proxy_url(Bot())
        self.assertNotIn("127.0.0.1", egress)
        self.assertNotIn("9050", egress)
        self.assertIn(entry["host"], egress)


class TestWorkerPassesTheProxyToTheBrowser(unittest.TestCase):
    """The bug: the worker picked a session then built the bot without it.

    DiscordAutomation(headless=..., domain=...) left self.proxy None, so
    _launch_proxy() returned None and Chrome silently launched on TOR —
    one second after logging '663 proxy sessions loaded'.
    """

    def test_worker_constructs_the_bot_with_proxy(self):
        src = open("app.py").read()
        i = src.index("bot = DiscordAutomation(\n"
                      "                headless=cfg.get(\"headless\", True),")
        block = src[i:i + 260]
        self.assertIn("proxy=proxy", block,
                      "worker must hand the session to the browser")

    def test_launch_proxy_is_populated(self):
        import server
        entry = proxies.vault_proxies()[0]
        bot = server.DiscordAutomation(headless=True, domain="x",
                                       proxy=entry)
        lp = bot._launch_proxy()
        self.assertIsNotNone(lp, "None here means TOR")
        self.assertIn(entry["host"], lp["server"])
        self.assertEqual(lp["username"], entry["username"])

    def test_without_a_proxy_launch_is_none(self):
        import server
        bot = server.DiscordAutomation(headless=True, domain="x")
        self.assertIsNone(bot._launch_proxy())


class TestSolverEgressAcceptsBothShapes(unittest.TestCase):
    """_solver_proxy_url read only `server` and returned "" for pool
    entries, silently dropping the solver back to its own IP."""

    def setUp(self):
        self.entry = proxies.vault_proxies()[0]

    def _url(self, proxy):
        import server

        class B:
            pass

        b = B()
        b.proxy = proxy
        return server.DiscordAutomation._solver_proxy_url(b)

    def test_pool_entry_shape(self):
        url = self._url(self.entry)
        self.assertIn(self.entry["username"], url)
        self.assertIn(self.entry["host"], url)

    def test_playwright_shape(self):
        url = self._url({
            "server": f"http://{self.entry['host']}:{self.entry['port']}",
            "username": self.entry["username"],
            "password": self.entry["password"]})
        self.assertIn(self.entry["username"], url)

    def test_both_shapes_agree(self):
        a = self._url(self.entry)
        b = self._url({
            "server": f"http://{self.entry['host']}:{self.entry['port']}",
            "username": self.entry["username"],
            "password": self.entry["password"]})
        self.assertEqual(a, b)

    def test_browser_and_solver_share_the_session(self):
        import server
        bot = server.DiscordAutomation(headless=True, domain="x",
                                       proxy=self.entry)
        lp = bot._launch_proxy()
        eg = server.DiscordAutomation._solver_proxy_url(bot)
        self.assertIn(lp["username"], eg)
        self.assertIn(self.entry["host"], eg)


if __name__ == "__main__":
    unittest.main(verbosity=2)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)

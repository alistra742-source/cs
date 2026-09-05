#!/usr/bin/env python3
"""The register-request hook: Discord reads the token from ITS OWN request.

Discord's React app never reads textarea[name=h-captcha-response]. It POSTs
/api/v9/auth/register with {captcha_key, captcha_rqtoken} in the JSON body,
so the solved token has to go into that request.

These tests execute the REAL hook source in node.
"""
import json
import re
import shutil
import subprocess
import tempfile
import unittest

NODE = shutil.which("node") or shutil.which("nodejs")


def hook_source() -> str:
    src = open("server.py").read()
    marker = '_CAPTCHA_HOOK_JS = r"""'
    i = src.index(marker) + len(marker)
    j = src.index('"""', i)
    return src[i:j]


HARNESS = """
global.window = global;
const captured = [];
global.fetch = async function (input, init) {
  captured.push({via:'fetch', url:(typeof input==='string')?input:input.url,
                 body: init && init.body});
  return {ok:true};
};
class XHR {
  open(m,u){ this.m=m; this.u=u; }
  send(b){ captured.push({via:'xhr', url:this.u, body:b}); }
}
global.XMLHttpRequest = XHR;
const install = %s;
const first = install();
const second = install();
window.__ncToken = %s;
window.__ncRqToken = 'RQTOK_1';
(async () => {
  await fetch('https://discord.com/api/v9/auth/register',
    {method:'POST', body: JSON.stringify({email:'a@b.c'})});
  await fetch('https://discord.com/api/v9/experiments',
    {method:'POST', body: JSON.stringify({x:1})});
  const x = new XMLHttpRequest();
  x.open('POST','https://discord.com/api/v9/auth/register');
  x.send(JSON.stringify({email:'a@b.c'}));
  console.log(JSON.stringify({first, second, captured,
                              injections: window.__ncInjections}));
})();
"""


def run_hook(token="'P1_TESTTOKEN'"):
    script = HARNESS % (hook_source(), token)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(script)
        path = f.name
    out = subprocess.run([NODE, path], capture_output=True, text=True,
                         timeout=60)
    if out.returncode != 0:
        raise AssertionError(out.stderr[:400])
    return json.loads(out.stdout.strip().splitlines()[-1])


@unittest.skipUnless(NODE, "node not available")
class TestHookBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_hook()

    def _body(self, via, path):
        for c in self.r["captured"]:
            if c["via"] == via and path in c["url"]:
                return json.loads(c["body"])
        self.fail(f"no {via} call to {path}")

    def test_installs_once(self):
        self.assertEqual(self.r["first"], "installed")
        self.assertEqual(self.r["second"], "already")

    def test_fetch_register_carries_the_token(self):
        self.assertEqual(self._body("fetch", "/auth/register")["captcha_key"],
                         "P1_TESTTOKEN")

    def test_fetch_register_carries_the_rqtoken(self):
        self.assertEqual(
            self._body("fetch", "/auth/register")["captcha_rqtoken"],
            "RQTOK_1")

    def test_xhr_register_carries_the_token(self):
        self.assertEqual(self._body("xhr", "/auth/register")["captcha_key"],
                         "P1_TESTTOKEN")

    def test_original_fields_survive(self):
        self.assertEqual(self._body("fetch", "/auth/register")["email"],
                         "a@b.c")

    def test_unrelated_requests_untouched(self):
        self.assertNotIn("captcha_key", self._body("fetch", "/experiments"))

    def test_injection_count(self):
        self.assertEqual(self.r["injections"], 2)


@unittest.skipUnless(NODE, "node not available")
class TestHookWithoutAToken(unittest.TestCase):
    def test_no_token_means_no_mutation(self):
        r = run_hook(token="''")
        for c in r["captured"]:
            self.assertNotIn("captcha_key", json.loads(c["body"]))


class TestWiring(unittest.TestCase):
    def setUp(self):
        self.src = open("server.py").read()

    def test_hook_installed_before_submitting(self):
        i = self.src.index("_install_captcha_hook()")
        j = self.src.index("_click_form_submit(), timeout=15.0")
        self.assertLess(i, j, "hook must be installed before submit")

    def test_token_is_published_to_the_page(self):
        self.assertIn("window.__ncToken = a[0]", self.src)
        self.assertIn("window.__ncRqToken = a[1]", self.src)

    def test_register_2xx_marks_success(self):
        self.assertIn("self._register_accepted = True", self.src)
        i = self.src.index("self._register_accepted = True")
        self.assertIn("/auth/register", self.src[max(0, i - 400):i])

    def test_past_captcha_honours_the_flag(self):
        i = self.src.index("async def _past_captcha")
        self.assertIn("_register_accepted", self.src[i:i + 900])

    def test_flag_resets_each_attempt(self):
        self.assertGreaterEqual(
            self.src.count("self._register_accepted = False"), 2)


@unittest.skipUnless(NODE, "node not available")
class TestDiagnostics(unittest.TestCase):
    """The hook must report WHY a token did not ride the request."""

    SCRIPT = """
global.window = global;
global.fetch = async () => ({ok:true});
class XHR { open(m,u){this.u=u;} send(b){} }
global.XMLHttpRequest = XHR;
const install = %s;
install();
(async () => {
  await fetch('https://discord.com/api/v9/auth/register',
    {method:'POST', body: JSON.stringify({email:'a@b.c'})});
  const before = {injections: window.__ncInjections,
                  seen: window.__ncSeen.slice()};
  window.__ncToken = 'P1_X'; window.__ncRqToken = 'RQ';
  await fetch('https://discord.com/api/v9/auth/register',
    {method:'POST', body: JSON.stringify({email:'a@b.c'})});
  console.log(JSON.stringify({before, after: {
    injections: window.__ncInjections, seen: window.__ncSeen,
    bodyKeys: window.__ncLastBodyKeys}}));
})();
"""

    def _run(self):
        script = self.SCRIPT % hook_source()
        with tempfile.NamedTemporaryFile("w", suffix=".js",
                                         delete=False) as f:
            f.write(script)
            path = f.name
        out = subprocess.run([NODE, path], capture_output=True, text=True,
                             timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr[:300])
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_records_a_request_that_arrived_before_the_token(self):
        r = self._run()
        self.assertEqual(r["before"]["injections"], 0)
        self.assertIn("no-token-yet", r["before"]["seen"])

    def test_records_a_successful_injection(self):
        r = self._run()
        self.assertEqual(r["after"]["injections"], 1)
        self.assertIn("injected", r["after"]["seen"])

    def test_reports_the_outgoing_body_keys(self):
        r = self._run()
        self.assertIn("captcha_key", r["after"]["bodyKeys"])
        self.assertIn("captcha_rqtoken", r["after"]["bodyKeys"])


class TestRejectionHints(unittest.TestCase):
    """Discord's captcha_key values must be translated, not just echoed."""

    def setUp(self):
        import server
        self.hints = server._CAPTCHA_KEY_HINTS

    def test_covers_the_common_reasons(self):
        for key in ("captcha-required", "invalid-response",
                    "sitekey-secret-mismatch", "rqdata-mismatch",
                    "expired", "ip-blocked"):
            self.assertIn(key, self.hints)

    def test_hints_are_actionable_text(self):
        for key, hint in self.hints.items():
            self.assertGreater(len(hint), 15, key)

    def test_reasons_are_logged(self):
        src = open("server.py").read()
        self.assertIn("[Captcha] Discord says:", src)
        self.assertIn("_CAPTCHA_KEY_HINTS.get", src)

    def test_hook_installed_at_page_creation(self):
        src = open("server.py").read()
        self.assertIn("_install_captcha_hook_early", src)
        i = src.index("def _attach_rqdata_capture")
        j = src.index("_install_captcha_hook_early")
        self.assertTrue(j > 0)


class TestCdpBodyInjection(unittest.TestCase):
    """Network-layer injection: the JS hook never saw the request.

    Live evidence: {'installed': True, 'injections': 0, 'seen': []} — the
    patched fetch/XHR observed NO register request, so Discord issued it
    outside our context. Fetch.requestPaused sits below that.
    """

    class _Bot:
        _pending_captcha_token = "P1_REALTOKEN"
        _rqtoken = "RQTOK9"
        _cdp_injections = 0
        _cdp_inject_note = ""

        def _log(self, *a, **k):
            pass

    def _mutate(self, bot, url, body):
        import server
        return server.DiscordAutomation._mutate_register_body(bot, url, body)

    def test_injects_into_register(self):
        bot = self._Bot()
        out = self._mutate(bot, "https://discord.com/api/v9/auth/register",
                           json.dumps({"email": "a@b.c"}))
        obj = json.loads(out)
        self.assertEqual(obj["captcha_key"], "P1_REALTOKEN")
        self.assertEqual(obj["captcha_rqtoken"], "RQTOK9")

    def test_preserves_original_fields(self):
        bot = self._Bot()
        out = self._mutate(bot, "https://discord.com/api/v9/auth/register",
                           json.dumps({"email": "a@b.c", "username": "u"}))
        obj = json.loads(out)
        self.assertEqual(obj["email"], "a@b.c")
        self.assertEqual(obj["username"], "u")

    def test_ignores_unrelated_endpoints(self):
        bot = self._Bot()
        self.assertIsNone(
            self._mutate(bot, "https://discord.com/api/v9/experiments",
                         json.dumps({"x": 1})))

    def test_no_token_means_no_mutation(self):
        bot = self._Bot()
        bot._pending_captcha_token = ""
        self.assertIsNone(
            self._mutate(bot, "https://discord.com/api/v9/auth/register",
                         json.dumps({"email": "a@b.c"})))

    def test_non_json_body_is_left_alone(self):
        bot = self._Bot()
        self.assertIsNone(
            self._mutate(bot, "https://discord.com/api/v9/auth/register",
                         "not json"))

    def test_counts_injections(self):
        bot = self._Bot()
        self._mutate(bot, "https://discord.com/api/v9/auth/register",
                     json.dumps({"email": "a@b.c"}))
        self.assertEqual(bot._cdp_injections, 1)
        self.assertIn("captcha_key", bot._cdp_inject_note)

    def test_engine_exposes_the_interceptor(self):
        src = open("nodriver_engine.py").read()
        self.assertIn("async def intercept_request_bodies", src)
        self.assertIn("cdp.fetch.RequestPaused", src)
        self.assertIn("continue_request", src)
        # post_data must be base64-encoded before it is sent.
        self.assertIn("post_data=enc", src)
        self.assertIn("b64encode", src)

    def test_interceptor_installed_before_submit(self):
        src = open("server.py").read()
        i = src.index("_install_cdp_captcha_interceptor()\n        await "
                      "self._install_captcha_hook()")
        j = src.index("_click_form_submit(), timeout=15.0")
        self.assertLess(i, j)

    def test_diagnostics_report_cdp_injections(self):
        src = open("server.py").read()
        self.assertIn("cdp_injections", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)

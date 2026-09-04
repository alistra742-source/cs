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


if __name__ == "__main__":
    unittest.main(verbosity=2)

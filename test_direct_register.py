#!/usr/bin/env python3
"""Submit /auth/register ourselves, with the captcha token attached.

Clicking Create Account a second time does NOT produce a register request:
after the first captcha-required response Discord's client holds the
challenge open and never re-POSTs. Proven by the live logs — with BOTH the
JS hook and the CDP Fetch interceptor confirmed active, the diagnostics
were {'injections': 0, 'seen': [], 'cdp_injections': 0}: no register
request existed to intercept. So the token has to be sent directly.
"""
import json
import shutil
import subprocess
import tempfile
import unittest

NODE = shutil.which("node") or shutil.which("nodejs")


def register_js() -> str:
    src = open("server.py").read()
    marker = '_DIRECT_REGISTER_JS = r"""'
    i = src.index(marker) + len(marker)
    j = src.index('"""', i)
    return src[i:j]


HARNESS = """
let seen = null;
global.fetch = async (url, init) => {
  seen = {url, init};
  return {status: %d, text: async () => %s};
};
const f = %s;
(async () => {
  const out = await f({
    email:'a@b.c', username:'user1', global_name:'User One',
    password:'pw', dob:'1998-05-01', token:'P1_TOKEN',
    rqtoken:'RQT1', fingerprint:'FP123'
  });
  console.log(JSON.stringify({seen: {url: seen.url,
    method: seen.init.method, credentials: seen.init.credentials,
    headers: seen.init.headers, body: JSON.parse(seen.init.body)}, out}));
})();
"""


def run(status=201, body='JSON.stringify({token:"SESSION"})'):
    script = HARNESS % (status, body, register_js())
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(script)
        path = f.name
    p = subprocess.run([NODE, path], capture_output=True, text=True,
                       timeout=60)
    if p.returncode != 0:
        raise AssertionError(p.stderr[:400])
    return json.loads(p.stdout.strip().splitlines()[-1])


@unittest.skipUnless(NODE, "node not available")
class TestRegisterPayload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run()

    def test_posts_to_the_register_endpoint(self):
        self.assertEqual(self.r["seen"]["url"], "/api/v9/auth/register")
        self.assertEqual(self.r["seen"]["method"], "POST")

    def test_sends_cookies(self):
        self.assertEqual(self.r["seen"]["credentials"], "include")

    def test_carries_the_captcha_token(self):
        self.assertEqual(self.r["seen"]["body"]["captcha_key"], "P1_TOKEN")

    def test_carries_the_rqtoken(self):
        self.assertEqual(self.r["seen"]["body"]["captcha_rqtoken"], "RQT1")

    def test_carries_the_fingerprint(self):
        self.assertEqual(self.r["seen"]["body"]["fingerprint"], "FP123")

    def test_sends_the_required_account_fields(self):
        b = self.r["seen"]["body"]
        self.assertEqual(b["email"], "a@b.c")
        self.assertEqual(b["username"], "user1")
        self.assertEqual(b["password"], "pw")
        self.assertEqual(b["date_of_birth"], "1998-05-01")
        self.assertTrue(b["consent"])

    def test_json_content_type(self):
        self.assertEqual(self.r["seen"]["headers"]["Content-Type"],
                         "application/json")

    def test_returns_status_and_body(self):
        self.assertEqual(self.r["out"]["status"], 201)
        self.assertIn("SESSION", self.r["out"]["body"])

    def test_reports_a_rejection(self):
        r = run(status=400,
                body='JSON.stringify({captcha_key:["invalid-response"]})')
        self.assertEqual(r["out"]["status"], 400)
        self.assertIn("invalid-response", r["out"]["body"])

    def test_network_error_is_caught(self):
        script = ("global.fetch = async () => { throw new Error('boom'); };\n"
                  "const f = %s;\n"
                  "f({email:'a',username:'u',password:'p',dob:'2000-01-01',"
                  "token:'T'}).then(o => console.log(JSON.stringify(o)));"
                  % register_js())
        with tempfile.NamedTemporaryFile("w", suffix=".js",
                                         delete=False) as f:
            f.write(script)
            path = f.name
        p = subprocess.run([NODE, path], capture_output=True, text=True,
                           timeout=60)
        out = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(out["status"], -1)


class TestWiring(unittest.TestCase):
    def setUp(self):
        self.src = open("server.py").read()

    def test_direct_submit_runs_after_the_click_path(self):
        i = self.src.index("_click_form_submit(), timeout=15.0")
        j = self.src.index("_direct_register_with_token(token)")
        self.assertLess(i, j)

    def test_success_sets_the_accepted_flag(self):
        i = self.src.index("async def _direct_register_with_token")
        block = self.src[i:i + 4000]
        self.assertIn("self._register_accepted = True", block)

    def test_missing_credentials_abort_early(self):
        i = self.src.index("async def _direct_register_with_token")
        block = self.src[i:i + 4000]
        self.assertIn("Cannot submit directly", block)

    def test_dob_is_recorded_in_iso_form(self):
        self.assertIn("self._dob_iso = (f\"{int(year_val):04d}-", self.src)

    def test_fingerprint_is_captured(self):
        self.assertIn("_discord_fingerprint", self.src)
        self.assertIn("/experiments", self.src)

    def test_rejection_refreshes_rqdata(self):
        i = self.src.index("async def _direct_register_with_token")
        block = self.src[i:i + 4000]
        self.assertIn("captcha_rqdata", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)

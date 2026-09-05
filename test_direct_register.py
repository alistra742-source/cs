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
global.window = global;
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
    rqtoken:'RQT1', session:'SESS1', fingerprint:'FP123'
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

    def test_token_goes_in_the_header(self):
        """Discord reads X-Captcha-Key, NOT body.captcha_key."""
        self.assertEqual(self.r["seen"]["headers"]["X-Captcha-Key"],
                         "P1_TOKEN")

    def test_rqtoken_goes_in_the_header(self):
        self.assertEqual(self.r["seen"]["headers"]["X-Captcha-Rqtoken"],
                         "RQT1")

    def test_session_id_goes_in_the_header(self):
        self.assertEqual(self.r["seen"]["headers"]["X-Captcha-Session-Id"],
                         "SESS1")

    def test_body_no_longer_carries_the_token(self):
        self.assertNotIn("captcha_key", self.r["seen"]["body"])

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
        script = ("global.window = global;\n"
                  "global.fetch = async () => { throw new Error('boom'); };\n"
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


class TestPromiseAwaiting(unittest.TestCase):
    """`HTTP 0` was the engine not awaiting the register promise.

    _wrap_js only set await_promise for JS literally starting with
    `async`. The direct-register arrow returns fetch(...).then(...), so
    Runtime.evaluate handed back the UNRESOLVED Promise, which marshals to
    an empty value — reported as status 0. Nothing to do with proxies.
    """

    def setUp(self):
        from nodriver_engine import _wrap_js
        self._wrap = _wrap_js

    def test_promise_returning_arrow_is_awaited(self):
        js = '(p) => { return fetch("/x").then(r => ({status: r.status})); }'
        self.assertTrue(self._wrap(js, {"a": 1})[1])

    def test_the_real_register_js_is_awaited(self):
        self.assertTrue(self._wrap(register_js(), {"email": "a"})[1])

    def test_async_arrow_still_awaited(self):
        js = 'async () => { const r = await fetch("/x"); return r.status; }'
        self.assertTrue(self._wrap(js, None)[1])

    def test_plain_arrow_not_awaited(self):
        self.assertFalse(self._wrap("() => document.title", None)[1])

    def test_bare_expression_not_awaited(self):
        self.assertFalse(self._wrap("location.href", None)[1])

    def test_bare_fetch_expression_is_awaited(self):
        self.assertTrue(self._wrap('fetch("/x")', None)[1])


class TestFreshChallengeOnRejection(unittest.TestCase):
    """invalid-response comes with a NEW challenge; reuse is guaranteed fail."""

    def setUp(self):
        self.src = open("server.py").read()

    def test_session_id_is_captured(self):
        self.assertIn("captcha_session_id", self.src)
        self.assertIn("self._captcha_session_id", self.src)

    def test_rejection_refreshes_all_three(self):
        i = self.src.index("Fresh challenge issued")
        block = self.src[max(0, i - 1800):i]
        for key in ("captcha_rqdata", "captcha_rqtoken",
                    "captcha_session_id"):
            self.assertIn(key, block)

    def test_refresh_is_logged(self):
        self.assertIn("Fresh challenge issued", self.src)


class TestInvisibleMode(unittest.TestCase):
    """Discord's 400 says should_serve_invisible:true for registration.

    An invisible hCaptcha token is minted differently from a checkbox one.
    Solving the wrong mode yields a token hCaptcha refuses — which is
    exactly the invalid-response we kept getting.
    """

    def setUp(self):
        self.src = open("server.py").read()

    def test_flag_is_captured(self):
        self.assertIn("should_serve_invisible", self.src)
        self.assertIn("self._captcha_invisible", self.src)

    def test_flag_is_passed_to_the_solver(self):
        self.assertIn("invisible=invisible", self.src)

    def test_mode_is_logged(self):
        self.assertIn("INVISIBLE", self.src)

    def test_solver_forwards_it(self):
        import inspect
        import nonecap_solver
        src = inspect.getsource(nonecap_solver.NoneCapSolver.solve)
        self.assertIn('payload["invisible"] = True', src)


class TestNoMarkerLeak(unittest.TestCase):
    """__nc_direct was being POSTed to Discord as an unknown field."""

    def setUp(self):
        self.src = open("server.py").read()

    def test_marker_is_gone_from_the_body(self):
        self.assertNotIn("__nc_direct:", self.src)
        self.assertNotIn('"__nc_direct"', self.src)

    def test_inflight_flag_replaces_it(self):
        self.assertIn("__ncDirectInflight", self.src)
        self.assertIn("_nc_direct_inflight", self.src)

    def test_flag_is_cleared_on_every_path(self):
        i = self.src.index("async def _direct_register_with_token")
        block = self.src[i:i + 5000]
        self.assertGreaterEqual(
            block.count("self._nc_direct_inflight = False"), 3)

    @unittest.skipUnless(NODE, "node not available")
    def test_body_has_no_marker(self):
        r = run()
        self.assertNotIn("__nc_direct", json.dumps(r["seen"]["body"]))


@unittest.skipUnless(NODE, "node not available")
class TestHookDoesNotTouchOurOwnRequest(unittest.TestCase):
    """The JS hook must not re-add captcha_key to our direct submit.

    Live evidence: seen=['injected'] with bodyKeys ending
    '...,captcha_key,captcha_rqtoken' — our own request was rewritten,
    putting the token in the BODY as well as the headers. The in-flight
    flag has to be set synchronously BEFORE fetch() is called, because the
    patched fetch runs inline.
    """

    SCRIPT = """
global.window = global;
let sent = null;
global.fetch = async (u, i) => { sent = {u, i};
  return {status: 400, text: async () => '{}'}; };
class XHR { open(m, u) { this.u = u; } send(b) {} }
global.XMLHttpRequest = XHR;
(%s)();
window.__ncToken = 'P1_TOK';
window.__ncRqToken = 'RQ1';
const direct = %s;
direct({email:'a@b.c', username:'u', global_name:'U', password:'p',
        dob:'1994-06-04', token:'P1_TOK', rqtoken:'RQ1', session:'S1',
        fingerprint:'FP'}).then(() => {
  const b = JSON.parse(sent.i.body);
  console.log(JSON.stringify({
    seen: window.__ncSeen, injections: window.__ncInjections,
    bodyHasCaptchaKey: ('captcha_key' in b),
    header: sent.i.headers['X-Captcha-Key'],
    cleared: (window.__ncDirectInflight === false)}));
});
"""

    @classmethod
    def setUpClass(cls):
        src = open("server.py").read()
        hm = '_CAPTCHA_HOOK_JS = r"""'
        i = src.index(hm) + len(hm)
        hook = src[i:src.index('"""', i)]
        script = cls.SCRIPT % (hook, register_js())
        with tempfile.NamedTemporaryFile("w", suffix=".js",
                                         delete=False) as f:
            f.write(script)
            path = f.name
        p = subprocess.run([NODE, path], capture_output=True, text=True,
                           timeout=60)
        assert p.returncode == 0, p.stderr[:400]
        cls.r = json.loads(p.stdout.strip().splitlines()[-1])

    def test_hook_recognises_our_own_request(self):
        self.assertIn("own-direct-request", self.r["seen"])

    def test_hook_does_not_inject(self):
        self.assertEqual(self.r["injections"], 0)

    def test_body_stays_clean(self):
        self.assertFalse(self.r["bodyHasCaptchaKey"])

    def test_token_is_still_in_the_header(self):
        self.assertEqual(self.r["header"], "P1_TOK")

    def test_flag_is_cleared_afterwards(self):
        self.assertTrue(self.r["cleared"])

    def test_flag_is_set_before_fetch(self):
        src = open("server.py").read()
        i = src.index("window.__ncDirectInflight = true")
        j = src.index("return fetch('/api/v9/auth/register'", i)
        self.assertLess(i, j, "flag must be set before the call")


if __name__ == "__main__":
    unittest.main(verbosity=2)

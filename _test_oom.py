"""Harness: verify the OOM-aware crash recovery + cgroup OOM detection.

Run: python3 _test_oom.py   (no browser needed)
"""
import asyncio
import sys
import types


def _stub(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    try:
        __import__(name)
        return sys.modules[name]
    except Exception:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m


class _Any:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, item):
        return _Any()


for _mod in ("aiohttp", "PIL", "PIL.Image", "numpy", "torch"):
    _stub(_mod, ClientSession=_Any, ClientTimeout=_Any, Image=_Any,
          __getattr__=lambda n: _Any())
_stub("requests", post=_Any, get=_Any)
try:
    import camoufox.async_api  # noqa: F401
except Exception:
    _stub("camoufox")
    _stub("camoufox.async_api",
          AsyncCamoufox=_Any, AsyncNewContext=_Any, AsyncNewBrowser=_Any)
    _stub("camoufox.exceptions",
          NotInstalledGeoIPExtra=Exception, UnknownIPLocation=Exception)

sys.path.insert(0, "/home/user/cs")
import server  # noqa: E402


class FakePage:
    async def goto(self, url, **k):
        return None


class FakeBot:
    """Bare object carrying the real _recover_crashed_page method."""
    _recover_crashed_page = server.DiscordAutomation._recover_crashed_page

    def __init__(self, oom_now=0, fail_first_n=99):
        self._stopped = types.SimpleNamespace(is_set=lambda: False)
        self._oom_kills_at_launch = 0
        self._oom_crash_times = []
        self._page_crashed = True
        self._page = None
        self._relaunch_calls = 0
        self.fail_first_n = fail_first_n
        self.logs = []
        self.sleeps = []
        self._fake_oom = oom_now

    def _log(self, message, level="info"):
        self.logs.append((level, message))

    async def _relaunch_browser(self):
        self._relaunch_calls += 1
        if self._relaunch_calls <= self.fail_first_n:
            raise RuntimeError("launch boom")
        self._page = FakePage()

    async def _log_browser_diagnostics(self, *a, **k):
        self.logs.append(("error", "[Diag] " + str(a[0])))


def _make_bot(oom_now, fail_first_n=99):
    b = FakeBot(oom_now=oom_now, fail_first_n=fail_first_n)
    # route the module-level counter + sleep through the fake
    server._cgroup_oom_kills = lambda: b._fake_oom
    return b


async def main():
    sleeps = []
    real_sleep = asyncio.sleep

    async def fake_sleep(t):
        sleeps.append(t)
        # don't actually wait 60s in the test
    server.asyncio.sleep = fake_sleep
    # gc.collect is real but cheap
    try:
        real_oom_fn = server._cgroup_oom_kills
        # 1) plain crash (no OOM delta): exactly ONE relaunch attempt
        sleeps.clear()
        b = _make_bot(oom_now=5)  # baseline 0, now 5? -> that's OOM. use 0
        b = FakeBot(oom_now=0, fail_first_n=99)
        b._page = None
        server._cgroup_oom_kills = lambda: 0
        ok = await b._recover_crashed_page("https://x")
        assert ok is False and b._relaunch_calls == 1, (ok, b._relaunch_calls)
        assert not any("memory pressure" in m for _, m in b.logs)
        print("case1 ok: non-OOM crash -> single restart attempt, then rotate")

        # 2) OOM crash, all 3 attempts fail -> False + explicit memory log
        sleeps.clear()
        b = FakeBot(oom_now=0, fail_first_n=99)
        server._cgroup_oom_kills = lambda: 4  # +4 since launch baseline 0
        ok = await b._recover_crashed_page("https://x")
        assert ok is False and b._relaunch_calls == 3, (ok, b._relaunch_calls)
        assert any("memory pressure" in m for _, m in b.logs)
        assert any("memory limit" in m for lvl, m in b.logs if lvl == "error")
        assert any(s >= 4.0 for s in sleeps), sleeps  # reclaim pauses happened
        print("case2 ok: OOM crash -> 3 same-transport retries with reclaim pauses")

        # 3) OOM crash, recovery succeeds on attempt 2
        sleeps.clear()
        b = FakeBot(oom_now=0, fail_first_n=1)  # first relaunch fails, 2nd ok
        server._cgroup_oom_kills = lambda: 2
        ok = await b._recover_crashed_page("https://x")
        assert ok is True and b._relaunch_calls == 2, (ok, b._relaunch_calls)
        assert b._page is not None
        print("case3 ok: OOM crash -> recovered on retry 2, same transport kept")

        # 4) third OOM crash in the window -> 60s backoff
        sleeps.clear()
        b = FakeBot(oom_now=0, fail_first_n=99)
        # simulate two prior OOM crashes in the last 5 min
        import time as _t
        b._oom_crash_times = [_t.time() - 60, _t.time() - 30]
        server._cgroup_oom_kills = lambda: 3
        ok = await b._recover_crashed_page("https://x")
        assert ok is False
        assert 60 in sleeps, sleeps
        assert any("container memory limit is too small" in m for _, m in b.logs)
        print("case4 ok: 3 OOM crashes / 5min -> 60s backoff + operator warning")

        # 5) _cgroup_oom_kills parses v2 events + v1 oom_control
        server._cgroup_oom_kills = real_oom_fn
        real_open = open

        class FakeFile:
            def __init__(self, content):
                self.c = content
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def __iter__(self):
                return iter(self.c.splitlines())

        def fake_open(path, *a, **k):
            if path == "/sys/fs/cgroup/memory.events":
                raise FileNotFoundError(path)
            if path == "/sys/fs/cgroup/memory/memory.oom_control":
                return FakeFile("oom_kill 7\noom_kill_counter 7\n")
            return real_open(path, *a, **k)
        server.open = fake_open
        try:
            assert server._cgroup_oom_kills() == 7, server._cgroup_oom_kills()
        finally:
            server.open = real_open
        print("case5 ok: _cgroup_oom_kills parses v1/v2 counters")
    finally:
        server.asyncio.sleep = real_sleep
        server._cgroup_oom_kills = real_oom_fn

    print("ALL PASS")


asyncio.run(main())

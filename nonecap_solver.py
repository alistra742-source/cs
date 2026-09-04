#!/usr/bin/env python3
"""nonecap_solver.py — hosted hCaptcha solving via the NoneCap API.

NoneCap returns a REAL hCaptcha token (a ``P1_…`` string that passes
siteverify) rather than image coordinates, so it sidesteps every problem
the local pipeline has with drag/pattern rounds: no glyph matching, no
gesture channel, no per-round guessing.

API shape (https://nonecap.com):

    POST https://api.nonecap.com/v1/solves?wait=90
    Authorization: Bearer <NONECAP_API>
    Content-Type: application/json
    {"type": "hcaptcha", "sitekey": "<uuid>", "url": "<page url>"}

    -> 200 {"id": "...", "status": "solved", "token": "P1_…",
            "credits_charged": 1}
    -> 202 {"id": "...", "status": "pending", "token": null}
       ...then poll GET /v1/solves/{id} until it leaves pending/solving.

Enterprise sitekeys bind the token to a per-challenge ``rqdata`` blob;
pass ``type="hcaptcha_enterprise"`` with that value. The bot already
captures rqdata, so this is wired automatically.

Failed solves are not charged, so a retry costs nothing but time.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

NONECAP_BASE = (os.environ.get("NONECAP_BASE")
                or "https://api.nonecap.com").rstrip("/")
# The user stores the key as NONECAP_API; accept the common aliases too.
_KEY_VARS = ("NONECAP_API", "NONECAP_API_KEY", "NONECAP_KEY")
# Seconds the API holds the connection open waiting for a token (max 90).
NONECAP_WAIT = int(float(os.environ.get("NONECAP_WAIT", "90")))
# How many solves to attempt per challenge.
NONECAP_TRIES = int(float(os.environ.get("NONECAP_TRIES", "3")))
# Overall ceiling for one solve attempt, including polling.
NONECAP_TIMEOUT = float(os.environ.get("NONECAP_TIMEOUT", "180"))
NONECAP_ENABLED = (os.environ.get("NONECAP_ENABLED", "1").strip().lower()
                   not in ("0", "false", "no", "off"))

_TERMINAL_BAD = ("failed", "error", "cancelled", "canceled", "expired")


def api_key() -> str:
    for var in _KEY_VARS:
        val = (os.environ.get(var) or "").strip()
        if val:
            # Tolerate a pasted "NONECAP_API = nc_live_..." line.
            if "=" in val and val.split("=", 1)[0].strip().isupper():
                val = val.split("=", 1)[1].strip()
            return val.strip("'\"")
    return ""


def configured() -> bool:
    """True when NoneCap can actually be used."""
    return bool(NONECAP_ENABLED and api_key() and aiohttp is not None)


class NoneCapSolver:
    """Minimal async client: submit a solve, return the token."""

    def __init__(self, key: str = "", log=None, base: str = ""):
        self._key = (key or api_key()).strip()
        self._base = (base or NONECAP_BASE).rstrip("/")
        self._log = log or (lambda *a, **k: None)
        self.solves = 0
        self.failures = 0
        self.credits_charged = 0
        self.last_error = ""
        self.last_solve_id = ""

    @property
    def enabled(self) -> bool:
        return bool(self._key) and aiohttp is not None and NONECAP_ENABLED

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json"}

    async def _read(self, resp):
        try:
            return await resp.json(content_type=None)
        except Exception:
            return {"_raw": (await resp.text())[:300]}

    async def solve(self, sitekey: str, url: str, rqdata: str = "",
                    invisible: bool = False,
                    timeout: float = NONECAP_TIMEOUT) -> Optional[str]:
        """Return an hCaptcha token, or None. Never raises."""
        if not self.enabled:
            self.last_error = "not configured"
            return None
        if not sitekey or not url:
            self.last_error = f"missing sitekey/url (sitekey={bool(sitekey)})"
            self._log(f"[NoneCap] {self.last_error}", level="warn")
            return None

        payload = {
            "type": "hcaptcha_enterprise" if rqdata else "hcaptcha",
            "sitekey": sitekey,
            "url": url,
        }
        if rqdata:
            payload["rqdata"] = rqdata
        if invisible:
            payload["invisible"] = True

        deadline = asyncio.get_event_loop().time() + timeout
        try:
            cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=cfg) as session:
                self._log(f"[NoneCap] Solving {payload['type']} for "
                          f"{sitekey[:8]}… (wait={NONECAP_WAIT}s)")
                async with session.post(
                        f"{self._base}/v1/solves",
                        params={"wait": NONECAP_WAIT},
                        json=payload, headers=self._headers()) as r:
                    data = await self._read(r)
                    if r.status == 401 or r.status == 403:
                        self.last_error = "authentication"
                        self._log("[NoneCap] Key rejected (HTTP "
                                  f"{r.status}) — check NONECAP_API",
                                  level="error")
                        return None
                    if r.status == 402:
                        self.last_error = "insufficient credits"
                        self._log("[NoneCap] Out of credits — top up at "
                                  "dashboard.nonecap.com", level="error")
                        return None
                    if r.status == 429:
                        self.last_error = "rate limited"
                        self._log("[NoneCap] Rate limited", level="warn")
                        return None
                    if r.status not in (200, 201, 202):
                        self.last_error = f"http {r.status}"
                        self._log(f"[NoneCap] HTTP {r.status}: "
                                  f"{str(data)[:200]}", level="warn")
                        return None

                solve = data if isinstance(data, dict) else {}
                self.last_solve_id = str(solve.get("id") or "")
                token = await self._await_token(session, solve, deadline)
                return token
        except asyncio.TimeoutError:
            self.last_error = "timeout"
            self._log(f"[NoneCap] Timed out after {timeout:.0f}s",
                      level="warn")
            return None
        except Exception as e:
            self.last_error = type(e).__name__
            self._log(f"[NoneCap] {type(e).__name__}: {e}", level="warn")
            return None

    async def _await_token(self, session, solve: dict,
                           deadline: float) -> Optional[str]:
        """Poll GET /v1/solves/{id} until the solve settles."""
        while True:
            status = str(solve.get("status") or "").lower()
            token = solve.get("token") or ""
            if status == "solved" and token:
                self.solves += 1
                try:
                    self.credits_charged += int(
                        solve.get("credits_charged") or 0)
                except Exception:
                    pass
                charged = solve.get("credits_charged")
                charged_txt = (f"{charged} credit(s)"
                               if charged is not None else "credits n/a")
                self._log(f"[NoneCap] Token received "
                          f"({len(token)} chars, {charged_txt})")
                return token
            if status in _TERMINAL_BAD:
                err = solve.get("error") or {}
                msg = (err.get("message") if isinstance(err, dict)
                       else str(err)) or status
                self.failures += 1
                self.last_error = str(msg)[:160]
                self._log(f"[NoneCap] Solve {status}: {self.last_error} "
                          "(not charged)", level="warn")
                return None
            # pending / solving
            sid = solve.get("id")
            if not sid:
                self.last_error = "no solve id"
                return None
            if asyncio.get_event_loop().time() >= deadline:
                self.last_error = "timeout while polling"
                self._log("[NoneCap] Gave up waiting for the token",
                          level="warn")
                await self.cancel(session, str(sid))
                return None
            await asyncio.sleep(2.0)
            try:
                async with session.get(f"{self._base}/v1/solves/{sid}",
                                       headers=self._headers()) as r:
                    solve = await self._read(r) or {}
            except Exception as e:
                self.last_error = type(e).__name__
                self._log(f"[NoneCap] poll failed: {type(e).__name__}",
                          level="debug")
                return None

    async def cancel(self, session, solve_id: str) -> bool:
        """Stop a solve we no longer need, so it does not run on."""
        try:
            async with session.post(
                    f"{self._base}/v1/solves/{solve_id}/cancel",
                    headers=self._headers()) as r:
                return r.status in (200, 202, 204)
        except Exception:
            return False

    async def report(self, solve_id: str, accepted: bool,
                     reason: str = "") -> bool:
        """Tell NoneCap whether the token was accepted (improves routing)."""
        if not self.enabled or not solve_id:
            return False
        body = {"outcome": "accepted" if accepted else "rejected"}
        if reason and not accepted:
            body["reason"] = reason[:200]
        try:
            cfg = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=cfg) as s:
                async with s.post(
                        f"{self._base}/v1/solves/{solve_id}/feedback",
                        json=body, headers=self._headers()) as r:
                    return r.status in (200, 201, 202, 204)
        except Exception:
            return False

    async def balance(self) -> Optional[int]:
        """Remaining credits, or None when unavailable."""
        if not self.enabled:
            return None
        try:
            cfg = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=cfg) as s:
                async with s.get(f"{self._base}/v1/me",
                                 headers=self._headers()) as r:
                    if r.status != 200:
                        return None
                    data = await self._read(r) or {}
            for key in ("credits_balance", "credits", "balance"):
                if key in data:
                    return int(data[key])
        except Exception:
            return None
        return None

    def stats(self) -> dict:
        return {"enabled": self.enabled, "solves": self.solves,
                "failures": self.failures,
                "credits_charged": self.credits_charged,
                "last_error": self.last_error}


if __name__ == "__main__":  # pragma: no cover
    print("configured:", configured())
    print("base:", NONECAP_BASE, "tries:", NONECAP_TRIES,
          "wait:", NONECAP_WAIT)

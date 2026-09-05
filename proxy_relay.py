#!/usr/bin/env python3
"""proxy_relay.py — local, unauthenticated front-end for an authenticated proxy.

WHY THIS EXISTS
---------------
Chrome has NO ``--proxy-user`` / ``--proxy-pass`` switches. Those flags are
silently ignored, and credentials embedded in ``--proxy-server`` are
stripped. So launching Chrome against an authenticated gateway produces an
immediate ``407 Proxy Authentication Required`` -> ``chrome-error://
chromewebdata/`` in 0.0s, even though the very same session validates fine
from aiohttp (which does send the credentials).

The usual CDP workaround (``Fetch.enable(handleAuthRequests=True)``) pauses
every request in the page and would collide with the captcha interceptor
that already owns the Fetch domain. So instead this runs a tiny local relay:

    Chrome --proxy-server=http://127.0.0.1:<port>   (no auth needed)
        -> relay adds Proxy-Authorization: Basic <...>
        -> upstream gate-eu.vaultproxies.com:80

Handles both proxy modes:
  * ``CONNECT host:port``  (HTTPS — the one that matters) then blind pipe
  * absolute-URI requests  (plain HTTP)

One relay per upstream session; the exit IP is unchanged, so the
IP-binding contract with the captcha solver still holds.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Optional, Tuple
from urllib.parse import urlsplit

_PIPE_CHUNK = 65536


def parse_upstream(url: str) -> Optional[Tuple[str, int, str, str]]:
    """``http://user:pass@host:port`` -> ``(host, port, user, pass)``."""
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    try:
        parts = urlsplit(url)
    except Exception:
        return None
    host = parts.hostname or ""
    if not host:
        return None
    port = int(parts.port or 80)
    return host, port, (parts.username or ""), (parts.password or "")


class ProxyRelay:
    """Local relay that injects Proxy-Authorization for Chrome."""

    def __init__(self, host: str, port: int, username: str = "",
                 password: str = "", log=None):
        self.up_host = host
        self.up_port = int(port)
        self.username = username or ""
        self.password = password or ""
        self._log = log or (lambda *a, **k: None)
        self._server: Optional[asyncio.AbstractServer] = None
        self.port: int = 0
        self.requests = 0
        self.failures = 0

    @property
    def auth_header(self) -> bytes:
        if not self.username:
            return b""
        raw = f"{self.username}:{self.password}".encode("utf-8")
        return b"Proxy-Authorization: Basic " + base64.b64encode(raw) + b"\r\n"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}" if self.port else ""

    async def start(self) -> str:
        """Bind an ephemeral local port. Returns the URL for Chrome."""
        self._server = await asyncio.start_server(
            self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self._log(f"[Relay] Local proxy relay on 127.0.0.1:{self.port} "
                  f"-> {self.up_host}:{self.up_port} "
                  f"(auth={'yes' if self.username else 'no'})")
        return self.url

    async def stop(self) -> None:
        if self._server is None:
            return
        try:
            self._server.close()
            await self._server.wait_closed()
        except Exception:
            pass
        self._server = None
        self.port = 0

    # ── connection handling ─────────────────────────────────────────────

    async def _handle(self, creader: asyncio.StreamReader,
                      cwriter: asyncio.StreamWriter) -> None:
        ureader = uwriter = None
        try:
            head = await asyncio.wait_for(
                creader.readuntil(b"\r\n\r\n"), timeout=30)
        except Exception:
            self._close(cwriter)
            return
        self.requests += 1
        try:
            ureader, uwriter = await asyncio.wait_for(
                asyncio.open_connection(self.up_host, self.up_port),
                timeout=20)
        except Exception as e:
            self.failures += 1
            self._log(f"[Relay] upstream connect failed: {type(e).__name__}",
                      level="debug")
            try:
                cwriter.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await cwriter.drain()
            except Exception:
                pass
            self._close(cwriter)
            return

        try:
            uwriter.write(self._with_auth(head))
            await uwriter.drain()
            await asyncio.gather(
                self._pipe(creader, uwriter),
                self._pipe(ureader, cwriter),
                return_exceptions=True)
        except Exception:
            pass
        finally:
            self._close(cwriter)
            self._close(uwriter)

    def _with_auth(self, head: bytes) -> bytes:
        """Insert Proxy-Authorization into the request head."""
        auth = self.auth_header
        if not auth:
            return head
        # Drop any existing header, then add ours right after the request
        # line so the upstream always sees exactly one.
        lines = head.split(b"\r\n")
        kept = [l for l in lines
                if not l.lower().startswith(b"proxy-authorization:")]
        if not kept:
            return head
        return kept[0] + b"\r\n" + auth + b"\r\n".join(kept[1:])

    @staticmethod
    async def _pipe(reader: asyncio.StreamReader,
                    writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                chunk = await reader.read(_PIPE_CHUNK)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.write_eof()
            except Exception:
                pass

    @staticmethod
    def _close(writer) -> None:
        if writer is None:
            return
        try:
            writer.close()
        except Exception:
            pass


async def relay_for(proxy: dict, log=None) -> Optional["ProxyRelay"]:
    """Build and start a relay for a pool entry / Playwright proxy dict.

    Returns None when the proxy needs no authentication (Chrome can use it
    directly) or when the dict is unusable.
    """
    if not isinstance(proxy, dict):
        return None
    user = str(proxy.get("username") or "").strip()
    if not user:
        return None                      # no auth -> Chrome is fine alone
    pwd = str(proxy.get("password") or "")
    host = str(proxy.get("host") or "").strip()
    port = proxy.get("port")
    if not host:
        parsed = parse_upstream(str(proxy.get("server") or ""))
        if not parsed:
            return None
        host, port = parsed[0], parsed[1]
    try:
        port = int(port or 80)
    except Exception:
        port = 80
    relay = ProxyRelay(host, port, user, pwd, log=log)
    await relay.start()
    return relay


if __name__ == "__main__":  # pragma: no cover
    print(parse_upstream("http://u:p@gate-eu.vaultproxies.com:80"))

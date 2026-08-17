"""
db.py — Postgres persistence for EY3.

Uses DATABASE_URL (env).  Auto-migrates on startup (CREATE TABLE IF NOT EXISTS).
Stores every generated account: email, username, password, FULL token, proxy
used, Discord user id, avatar, bio, humanization flag and validation status.
Also validates tokens against the Discord API so the dashboard can show a
live "valid tokens" count.
"""
import asyncio
import os
from typing import Dict, List, Optional

import aiohttp
import asyncpg

DATABASE_URL = os.environ.get("DATABASE_URL", "")

_pool: Optional[asyncpg.Pool] = None
_db_ready = False


async def init_db() -> bool:
    """Connect + auto-migrate. Safe to call at startup; no-ops without URL."""
    global _pool, _db_ready
    if not DATABASE_URL:
        print("[DB] DATABASE_URL not set - token saving disabled", flush=True)
        _db_ready = False
        return False
    try:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        async with _pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id          SERIAL PRIMARY KEY,
                    email       TEXT,
                    username    TEXT,
                    password    TEXT,
                    token       TEXT,
                    status      TEXT DEFAULT 'pending',
                    proxy       TEXT,
                    worker_id   TEXT,
                    created_at  TIMESTAMPTZ DEFAULT now()
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_accounts_status
                ON accounts (status);
            """)
            # EY3 additions (safe on existing databases)
            for col, ddl in (
                ("user_id", "TEXT DEFAULT ''"),
                ("avatar", "TEXT DEFAULT ''"),
                ("bio", "TEXT DEFAULT ''"),
                ("humanized", "BOOLEAN DEFAULT FALSE"),
            ):
                try:
                    await conn.execute(
                        f"ALTER TABLE accounts ADD COLUMN IF NOT EXISTS {col} {ddl}"
                    )
                except Exception:
                    pass
            # used_proxies — every proxy session handed to a worker, so a
            # redeploy can skip sticky IPs that were already used.
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS used_proxies (
                    key        TEXT PRIMARY KEY,
                    exit_ip    TEXT DEFAULT '',
                    status     TEXT DEFAULT 'used',
                    first_seen TIMESTAMPTZ DEFAULT now(),
                    last_used  TIMESTAMPTZ DEFAULT now()
                );
            """)
        _db_ready = True
        print("[DB] Connected - accounts table ready", flush=True)
        return True
    except Exception as e:
        print(f"[DB] init failed: {e}", flush=True)
        _db_ready = False
        return False


def db_ok() -> bool:
    return _db_ready and _pool is not None


async def save_account(email: str, username: str, password: str,
                       token: str, proxy: str = "", worker_id: str = "",
                       user_id: str = "", avatar: str = "", bio: str = "",
                       humanized: bool = False) -> bool:
    """Persist a generated account. Never raises."""
    if not db_ok():
        return False
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (email, username, password, token,
                                      status, proxy, worker_id, user_id,
                                      avatar, bio, humanized)
                VALUES ($1, $2, $3, $4, 'pending', $5, $6, $7, $8, $9, $10)
                """,
                email, username, password, token, proxy, worker_id,
                user_id, avatar, bio, humanized,
            )
        return True
    except Exception as e:
        print(f"[DB] save_account error: {e}", flush=True)
        return False


async def list_accounts(limit: int = 200) -> List[dict]:
    if not db_ok():
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, email, username, password, token, status, proxy,
                          worker_id, created_at, user_id, avatar, bio, humanized
                   FROM accounts ORDER BY id DESC LIMIT $1""",
                limit,
            )
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] list_accounts error: {e}", flush=True)
        return []


async def update_account_status(token: str, status: str) -> None:
    if not db_ok():
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                "UPDATE accounts SET status = $1 WHERE token = $2",
                status, token,
            )
    except Exception:
        pass


async def delete_accounts(ids: List[int]) -> int:
    """Delete accounts by primary key id. Returns the number requested."""
    if not db_ok() or not ids:
        return 0
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM accounts WHERE id = ANY($1::int[])", ids)
        return len(ids)
    except Exception as e:
        print(f"[DB] delete_accounts error: {e}", flush=True)
        return 0


async def validate_token(token: str) -> bool:
    """Check a Discord token against the API. 200 => valid."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://discord.com/api/v9/users/@me",
                headers={"Authorization": token},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                return r.status == 200
    except Exception:
        return False


async def validate_all_tokens(accounts: List[dict]) -> int:
    """Validate every account's token concurrently. Returns valid count."""
    if not accounts:
        return 0
    valid = 0
    sem = asyncio.Semaphore(20)

    async def _check(acc: dict) -> bool:
        if not acc.get("token"):
            return False
        async with sem:
            ok = await validate_token(acc["token"])
            await update_account_status(acc["token"], "valid" if ok else "invalid")
            return ok

    results = await asyncio.gather(
        *[_check(a) for a in accounts], return_exceptions=True
    )
    for r in results:
        if r is True:
            valid += 1


# ── used_proxies: persistent record of proxy sessions handed to workers ──

def _proxy_upsert_sql() -> str:
    return """
        INSERT INTO used_proxies (key, status, exit_ip, last_used)
        VALUES ($1, $2, $3, now())
        ON CONFLICT (key) DO UPDATE SET
            status = EXCLUDED.status,
            exit_ip = CASE WHEN EXCLUDED.exit_ip <> '' THEN EXCLUDED.exit_ip
                           ELSE used_proxies.exit_ip END,
            last_used = now()
    """


async def record_proxy(key: str, status: str = "used", exit_ip: str = "") -> bool:
    """Upsert one used-proxy record. Never raises."""
    if not db_ok() or not key:
        return False
    try:
        async with _pool.acquire() as conn:
            await conn.execute(_proxy_upsert_sql(), key, status, exit_ip or "")
        return True
    except Exception:
        return False


async def record_proxies(items) -> bool:
    """Batch upsert [(key, status, exit_ip), ...]. Never raises."""
    if not db_ok() or not items:
        return False
    try:
        async with _pool.acquire() as conn:
            for key, status, exit_ip in items:
                if key:
                    await conn.execute(_proxy_upsert_sql(), key, status,
                                       exit_ip or "")
        return True
    except Exception:
        return False


async def list_proxies(limit: int = 3000) -> List[dict]:
    """All used-proxy records, newest first. Never raises."""
    if not db_ok():
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT key, exit_ip, status, first_seen, last_used
                   FROM used_proxies ORDER BY last_used DESC LIMIT $1""",
                limit,
            )
        return [dict(r) for r in rows]
    except Exception:
        return []

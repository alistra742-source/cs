#!/usr/bin/env python3
"""vision_events.py — report captcha outcomes to Roboflow Vision Events.

Every solve attempt is a labelled data point we currently throw away. This
posts each one as pass/fail so the dashboard shows REAL accuracy per
challenge family instead of guesses from logs:

    POST https://api.roboflow.com/vision-events
    Authorization: Bearer <API_KEY>
    {
      "eventId":        "<uuid>",          unique per use case, <=256 chars
      "eventType":      "quality_check",
      "useCaseId":      "<id>",            <=256 chars
      "timestamp":      "<ISO 8601>",      last year .. tomorrow
      "eventData":      {"result": "pass" | "fail"},
      "customMetadata": {k: str|num|bool}  optional, <=100 entries
    }

Design rules:
  * NEVER let telemetry break a solve — every failure is swallowed and
    logged at debug.
  * Fire-and-forget: posts run as background tasks so the solver never
    waits on the network mid-challenge.
  * Off unless VISION_EVENTS=1, so nothing is sent without opting in.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

VISION_EVENTS_URL = (os.environ.get("VISION_EVENTS_URL")
                     or "https://api.roboflow.com/vision-events")
VISION_EVENTS_ENABLED = (os.environ.get("VISION_EVENTS", "")
                         .strip().lower() in ("1", "true", "yes", "on"))
VISION_EVENTS_USE_CASE = (os.environ.get("VISION_EVENTS_USE_CASE")
                          or "hcaptcha-solver").strip()[:256]
VISION_EVENTS_TIMEOUT = float(os.environ.get("VISION_EVENTS_TIMEOUT", "8"))

# Spec limits.
_MAX_ID = 256
_MAX_METADATA = 100


def _iso_now() -> str:
    """ISO 8601 UTC timestamp, guaranteed inside the accepted window."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def timestamp_is_valid(ts: str) -> bool:
    """True when ``ts`` falls within the last year through tomorrow."""
    try:
        cleaned = (ts or "").strip().replace("Z", "+00:00")
        when = datetime.fromisoformat(cleaned)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
    except Exception:
        return False
    now = datetime.now(timezone.utc)
    return (now - timedelta(days=365)) <= when <= (now + timedelta(days=1))


def clean_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Coerce metadata to the accepted shape: str/int/float/bool, <=100 keys.

    Anything else is stringified rather than dropped — a slightly lossy
    value beats a rejected request.
    """
    out: Dict[str, Any] = {}
    if not isinstance(meta, dict):
        return out
    for key, value in meta.items():
        if len(out) >= _MAX_METADATA:
            break
        k = str(key)[:_MAX_ID]
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float)):
            out[k] = value
        else:
            out[k] = str(value)[:512]
    return out


def build_event(result: str,
                use_case_id: Optional[str] = None,
                event_id: Optional[str] = None,
                timestamp: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None,
                event_type: str = "quality_check") -> Dict[str, Any]:
    """Assemble a spec-valid payload. ``result`` is coerced to pass/fail."""
    res = "pass" if str(result).strip().lower() in (
        "pass", "true", "ok", "solved", "1") else "fail"
    ts = timestamp or _iso_now()
    if not timestamp_is_valid(ts):
        ts = _iso_now()
    return {
        "eventId": (event_id or str(uuid.uuid4()))[:_MAX_ID],
        "eventType": event_type,
        "useCaseId": (use_case_id or VISION_EVENTS_USE_CASE)[:_MAX_ID],
        "timestamp": ts,
        "eventData": {"result": res},
        "customMetadata": clean_metadata(metadata),
    }


class VisionEvents:
    """Fire-and-forget reporter. Safe to construct even when disabled."""

    def __init__(self, api_key: str = "", use_case_id: str = "",
                 log=None, enabled: Optional[bool] = None):
        self._api_key = (api_key or os.environ.get("API_KEY", "")).strip()
        self.use_case_id = (use_case_id or VISION_EVENTS_USE_CASE)[:_MAX_ID]
        self._log = log or (lambda *a, **k: None)
        self.enabled = (VISION_EVENTS_ENABLED if enabled is None else enabled)
        if self.enabled and not self._api_key:
            self._log("[Events] VISION_EVENTS=1 but API_KEY is empty — "
                      "telemetry disabled", level="warn")
            self.enabled = False
        self.sent = 0
        self.failed = 0
        self._tasks: set = set()

    async def _post(self, payload: Dict[str, Any]) -> bool:
        if aiohttp is None:
            return False
        try:
            cfg = aiohttp.ClientTimeout(total=VISION_EVENTS_TIMEOUT)
            headers = {"Authorization": f"Bearer {self._api_key}",
                       "Content-Type": "application/json"}
            async with aiohttp.ClientSession(timeout=cfg) as s:
                async with s.post(VISION_EVENTS_URL, json=payload,
                                  headers=headers) as r:
                    if r.status in (200, 201, 202, 204):
                        self.sent += 1
                        return True
                    body = (await r.text())[:180]
                    self.failed += 1
                    self._log(f"[Events] HTTP {r.status}: {body}",
                              level="debug")
                    return False
        except Exception as e:
            self.failed += 1
            self._log(f"[Events] post failed: {type(e).__name__}",
                      level="debug")
            return False

    async def report(self, result: str, **metadata) -> bool:
        """Await the post. Returns True when accepted."""
        if not self.enabled:
            return False
        return await self._post(build_event(
            result, use_case_id=self.use_case_id, metadata=metadata))

    def report_nowait(self, result: str, **metadata) -> None:
        """Schedule the post without blocking the solve."""
        if not self.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.report(result, **metadata))
        # Hold a reference so the task is not garbage-collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def stats(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "sent": self.sent,
                "failed": self.failed, "use_case": self.use_case_id}


if __name__ == "__main__":  # pragma: no cover
    ev = build_event("pass", use_case_id="hcaptcha-solver",
                     metadata={"family": "drag", "gesture": "html5-dnd",
                               "confidence": 0.97, "rounds": 2})
    import json
    print(json.dumps(ev, indent=2))
    print("timestamp valid:", timestamp_is_valid(ev["timestamp"]))

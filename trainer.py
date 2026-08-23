"""Live hCaptcha demo runner.

This module deliberately does *not* manufacture challenges, issue tokens, or
solve CAPTCHA challenges.  It drives the official hCaptcha demo in a real
Chrome tab, fills its optional sample field, clicks the real checkbox, and
then pauses when hCaptcha asks for a human.  The actual challenge iframe is
captured and its visible prompt is copied into the dashboard so an operator
can inspect it and complete the check manually.
"""

from __future__ import annotations

import asyncio
import base64
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


# This is hCaptcha's public demo page, not a locally rendered approximation.
TARGET_DEMO_URL = "https://accounts.hcaptcha.com/demo"
# How often the dashboard challenge pane gets a fresh Chrome frame.
SCREENSHOT_INTERVAL_S = 3.0

# The demo currently exposes one optional text field, but these fallbacks make
# the runner tolerant of harmless markup changes on the official page.
FORM_FIELD_SELECTORS = (
    'input[type="text"]',
    'input:not([type])',
    'textarea',
)
WIDGET_IFRAME_SELECTOR = (
    'iframe[title*="hCaptcha"], '
    'iframe[src*="hcaptcha.com"]'
)
CHALLENGE_IFRAME_SELECTOR = (
    'iframe[title*="hCaptcha challenge"], '
    'iframe[src*="hcaptcha-challenge"]'
)
CHECKBOX_SELECTOR = (
    '#checkbox',
    '[role="checkbox"]',
    '.checkbox',
    '[aria-checked]',
    'input[type="checkbox"]',
)

FIRST_NAMES = (
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Riley",
    "Casey", "Logan", "Avery", "Jamie", "Dakota", "Reese", "Skyler",
)
LAST_NAMES = (
    "Vance", "Frost", "Sterling", "Cross", "Stone", "Hayes", "Drake",
    "Rivers", "Mercer", "Black", "Fox", "Ray", "Knight", "Cole",
)
COMMENTS = (
    "Quick test", "Demo check", "Form test", "Live sample", "Hello demo",
    "Short note", "Ready now", "Browser check",
)


def generate_form_words(rng: Optional[random.Random] = None) -> Dict[str, str]:
    """Return short, non-sensitive values for the demo's optional field.

    ``page.locator(...).fill`` inserts the value in one operation; no local
    form is rendered in the dashboard and no account or token is generated.
    """
    rng = rng or random
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    sample = rng.choice(COMMENTS)
    return {
        "name": name,
        "email": "",
        "comment": sample,
        "field": sample,
        "url": TARGET_DEMO_URL,
    }


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _question_from_text(text: str) -> str:
    """Pick the visible instruction from an hCaptcha frame's text.

    hCaptcha localizes and changes its markup, so this intentionally uses
    wording rather than a brittle class name.  The complete frame text is
    retained separately for debugging/display when no instruction line is
    identifiable.
    """
    raw = str(text or "")
    compact = _clean_text(raw)
    if not compact:
        return ""
    parts = [
        _clean_text(p) for p in re.split(r"(?<=[.!?])\s+|\n+", raw)
        if _clean_text(p)
    ]
    instruction = re.compile(
        r"\b(select|click|choose|pick|mark|check|tap|identify|selecte|wählen|wähle|"
        r"seleccione|selecciona|cliquez|choisissez|画像|画像を|選択|выберите|выбери|"
        r"请|点击|เลือก|chọn)\b",
        re.IGNORECASE,
    )
    # A body-only read can flatten a short set of chrome labels into one
    # string. Do not mistake that chrome for the challenge question.
    chrome_only = re.compile(
        r"^(?:h?captcha|verify|privacy|terms|audio|accessibility|reload|next|"
        r"please try again|i am human|english|en)(?:\s+(?:h?captcha|verify|privacy|"
        r"terms|audio|accessibility|reload|next|please try again|i am human|english|en))*$",
        re.IGNORECASE,
    )
    for part in parts:
        if len(part) >= 8 and instruction.search(part) and not chrome_only.fullmatch(part):
            return part[:500]
    for part in parts:
        if len(part) >= 8 and not chrome_only.fullmatch(part):
            return part[:500]
    if chrome_only.fullmatch(compact):
        return ""
    return compact[:500]


class TrainerEngine:
    """Run the real official demo in a human-in-the-loop browser session."""

    def __init__(self):
        self._lock = threading.RLock()
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._one_shot = False
        self.speed = 2.0
        self.headless = True

        self.current_stage = "idle"
        self.status_text = "Ready. Start the real hCaptcha demo runner."
        self.current_form: Dict[str, str] = {
            "name": "",
            "email": "",
            "comment": "",
            "field": "",
            "url": TARGET_DEMO_URL,
        }

        self.captured_count = 0
        self.pass_count = 0
        self.total_cycles = 0
        self.start_time: Optional[float] = None

        self.latest_challenge: Dict[str, Any] = {}
        self.latest_screenshot = ""
        self.latest_question = ""
        self.questions: List[Dict[str, Any]] = []
        self.logs: List[str] = []

        # Owned by the runner thread/event loop.  They are kept here only so
        # shutdown is observable and no second browser is spawned by a second
        # START click.
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._external_task = None
        self._external_page = None
        self._add_log("Live demo runner initialized; synthetic challenges disabled.")

    # ── State/log helpers ────────────────────────────────────────────────

    def _add_log(self, msg: str) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > 80:
                self.logs = self.logs[-60:]

    def _set_state(self, stage: str, text: str) -> None:
        with self._lock:
            self.current_stage = stage
            self.status_text = text

    def _stopped(self) -> bool:
        return self._stop_event.is_set()

    # ── Public lifecycle ─────────────────────────────────────────────────

    def start(self, speed: float = 2.0, headless: Optional[bool] = None) -> dict:
        """Start the live browser loop.

        The loop stops at a real challenge and waits for the human to finish
        it.  It never fabricates a prompt or treats a generated value as an
        hCaptcha token.
        """
        with self._lock:
            if self.running or (self._thread and self._thread.is_alive()):
                return {"ok": True, "message": "Live demo runner already running or stopping"}
            if self._external_task and not self._external_task.done():
                return {"ok": True, "message": "Live demo runner already running or stopping"}
            self.speed = max(0.5, float(speed))
            if headless is not None:
                self.headless = bool(headless)
            self.running = True
            self._one_shot = False
            self._stop_event.clear()
            if self.start_time is None:
                self.start_time = time.time()
            self._thread = threading.Thread(
                target=self._thread_entry,
                name="hcaptcha-demo-runner",
                daemon=True,
            )
            self._thread.start()
            self._add_log(
                f"Started real Chrome runner for {TARGET_DEMO_URL} "
                f"(poll delay={self.speed:.1f}s)."
            )
            return {"ok": True, "message": "Real hCaptcha demo runner started"}

    def step(self, headless: Optional[bool] = None) -> dict:
        """Queue one real demo cycle without starting a continuous loop."""
        with self._lock:
            if (self.running or (self._thread and self._thread.is_alive())
                    or (self._external_task and not self._external_task.done())):
                return {"ok": False, "message": "Runner already running or stopping"}
            self.speed = max(0.5, self.speed)
            if headless is not None:
                self.headless = bool(headless)
            self.running = True
            self._one_shot = True
            self._stop_event.clear()
            if self.start_time is None:
                self.start_time = time.time()
            self._thread = threading.Thread(
                target=self._thread_entry,
                name="hcaptcha-demo-step",
                daemon=True,
            )
            self._thread.start()
            self._add_log("Queued one real hCaptcha demo cycle.")
            return {"ok": True, "message": "Real demo cycle queued"}

    def start_external(self, page, speed: float = 2.0,
                       one_shot: bool = False) -> dict:
        """Run against the app's live page so the operator can take over.

        The Flask app already exposes the shared B1 browser through its LIVE
        camera/action controls.  Reusing that page keeps the real challenge
        visible and avoids launching a second hidden browser session.
        This method must be called from the page's asyncio event loop.
        """
        if page is None:
            return {"ok": False, "message": "Live browser page is not available"}
        with self._lock:
            if (self.running or (self._thread and self._thread.is_alive())
                    or (self._external_task and not self._external_task.done())):
                return {"ok": False, "message": "Runner already running or stopping"}
            self.speed = max(0.5, float(speed))
            self.running = True
            self._one_shot = bool(one_shot)
            self._stop_event.clear()
            self._external_page = page
            self._page = page
            if self.start_time is None:
                self.start_time = time.time()
            self._external_task = asyncio.create_task(
                self._run_attached_async(), name="hcaptcha-demo-attached-runner"
            )
            self._add_log("Attached runner to the shared LIVE browser page.")
            return {"ok": True, "message": "Real demo runner attached to LIVE browser"}

    def is_busy(self) -> bool:
        with self._lock:
            return bool(
                self.running
                or (self._thread and self._thread.is_alive())
                or (self._external_task and not self._external_task.done())
            )

    def set_speed(self, speed: float) -> dict:
        with self._lock:
            self.speed = max(0.5, float(speed))
            return {"ok": True, "speed": self.speed}

    def stop(self) -> dict:
        with self._lock:
            if not self.running:
                return {"ok": True, "message": "Live demo runner already stopped"}
            self.running = False
            self._stop_event.set()
            self.current_stage = "idle"
            self.status_text = "Runner stopped."
            self._add_log("Live demo runner stopped by user.")
            return {"ok": True, "message": "Live demo runner stopped"}

    def clear(self) -> dict:
        with self._lock:
            self.questions.clear()
            self.captured_count = 0
            self.pass_count = 0
            self.total_cycles = 0
            self.latest_screenshot = ""
            self.latest_question = ""
            self.latest_challenge = {}
            self._add_log("Captured question list cleared.")
            return {"ok": True, "message": "Captured questions cleared"}

    def get_state(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "speed": self.speed,
                "stage": self.current_stage,
                "status_text": self.status_text,
                "form": dict(self.current_form),
                "captured_count": self.captured_count,
                "pass_count": self.pass_count,
                "total_cycles": self.total_cycles,
                "latest_question": self.latest_question,
                "latest_challenge": dict(self.latest_challenge),
                "latest_screenshot": self.latest_screenshot,
                "questions": [dict(q) for q in self.questions],
                "logs": list(self.logs[-20:]),
                "target_url": TARGET_DEMO_URL,
                "human_in_the_loop": True,
                "screenshot_interval": SCREENSHOT_INTERVAL_S,
            }

    # ── Browser helpers ───────────────────────────────────────────────────

    async def _wait_for_content(self, page, timeout: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stopped():
            try:
                body = await asyncio.wait_for(
                    page.evaluate(
                        "() => !!(document.body && (document.body.innerText || '').trim())"
                    ),
                    timeout=3.0,
                )
                if body:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.25)
        return False

    async def _visible_locators(self, page, selector: str) -> list:
        try:
            return [loc for loc in await page.locator(selector).all()
                    if await loc.is_visible()]
        except Exception:
            return []

    async def _fill_demo_field(self, page, value: str) -> bool:
        for selector in FORM_FIELD_SELECTORS:
            try:
                locs = await self._visible_locators(page, selector)
                for loc in locs:
                    # Ignore the hidden hCaptcha response and any submit-like
                    # controls that happen to be represented as inputs.
                    typ = (await loc.get_attribute("type") or "text").lower()
                    if typ in {"hidden", "submit", "button", "checkbox", "radio"}:
                        continue
                    await loc.fill(value, timeout=5000)
                    return True
            except Exception:
                continue
        return False

    async def _find_frame(self, page, selector: str):
        """Return (iframe locator, frame) for the first visible matching frame."""
        try:
            refresh_frames = getattr(page, "_refresh_frames", None)
            if callable(refresh_frames):
                await refresh_frames(force=True)
        except Exception:
            pass
        try:
            locs = await self._visible_locators(page, selector)
        except Exception:
            locs = []
        for iframe in locs:
            try:
                handle = await iframe.element_handle(timeout=2500)
                frame = await handle.content_frame() if handle else None
                if frame is not None:
                    return iframe, frame
            except Exception:
                pass
        return None, None

    async def _wait_for_checkbox(self, page, timeout: float = 35.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stopped():
            iframe, frame = await self._find_frame(page, WIDGET_IFRAME_SELECTOR)
            if frame is not None:
                try:
                    controls = frame.locator(", ".join(CHECKBOX_SELECTOR))
                    if await controls.count() > 0:
                        return iframe, frame, controls.first
                except Exception:
                    pass
            # The frame cache is refreshed by the engine after evaluate calls;
            # this also gives hCaptcha time to attach its iframe after load.
            await asyncio.sleep(0.35)
        return None, None, None

    async def _click_real_checkbox(self, page) -> bool:
        iframe, frame, checkbox = await self._wait_for_checkbox(page)
        if iframe is None or frame is None or checkbox is None:
            return False
        try:
            await checkbox.click(timeout=5000)
            self._add_log("Clicked the real hCaptcha checkbox.")
            return True
        except Exception as first_error:
            # A few Chrome builds expose the checkbox through frame_locator
            # before content_frame() has refreshed.  Retry with a trusted
            # frame-scoped locator; do not use JS click or mutate hCaptcha DOM.
            try:
                src = await iframe.get_attribute("src") or ""
                if src:
                    fl = page.frame_locator(f'iframe[src="{src}"]')
                    controls = fl.locator(", ".join(CHECKBOX_SELECTOR))
                    await controls.first.click(timeout=5000)
                    self._add_log("Clicked the real hCaptcha checkbox (frame locator).")
                    return True
            except Exception as second_error:
                self._add_log(
                    f"Real checkbox click failed: {type(first_error).__name__}; "
                    f"frame retry: {type(second_error).__name__}"
                )
        return False

    async def _read_response_token(self, page) -> bool:
        """Detect hCaptcha's own response, without creating or injecting one."""
        try:
            return bool(await page.evaluate("""() => {
                const ta = document.querySelector('textarea[name="h-captcha-response"]');
                if (ta && String(ta.value || '').trim()) return true;
                try {
                    return !!(window.hcaptcha && typeof window.hcaptcha.getResponse === 'function'
                        && String(window.hcaptcha.getResponse() || '').trim());
                } catch (e) { return false; }
            }"""))
        except Exception:
            return False

    async def _checkbox_is_checked(self, page) -> bool:
        try:
            for _iframe, frame in await self._all_frames(page):
                try:
                    checked = await frame.evaluate("""() => {
                        const el = document.querySelector('[aria-checked], input[type="checkbox"]');
                        return !!el && (el.getAttribute('aria-checked') === 'true' || el.checked === true);
                    }""")
                    if checked:
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        return False

    async def _all_frames(self, page) -> list:
        out = []
        try:
            # Refreshing with a harmless evaluation makes the nodriver facade
            # update its frame cache after the challenge iframe is mounted.
            await page.evaluate("document.readyState")
            for frame in list(page.frames):
                if frame is not None:
                    out.append((None, frame))
        except Exception:
            pass
        return out

    async def _challenge_ready(self, frame) -> bool:
        """Avoid treating hCaptcha's empty iframe shell as a challenge."""
        try:
            state = await frame.evaluate("document.readyState")
            if state not in {"interactive", "complete"}:
                return False
            text = _clean_text(await frame.locator("body").inner_text())
            if len(text) >= 8:
                return True
            # Some locales/rendering paths paint the instruction as an image
            # or canvas. Require a real painted control/media node instead of
            # accepting the iframe's initial blank document.
            for selector in ("img", "canvas", "button", "[role=button]", "[role=img]"):
                if await frame.locator(selector).count() > 0:
                    return True
        except Exception:
            pass
        return False

    async def _wait_for_challenge_or_success(self, page, timeout: float = 60.0):
        deadline = time.monotonic() + timeout
        checked_since: Optional[float] = None
        while time.monotonic() < deadline and not self._stopped():
            _iframe, frame = await self._find_frame(page, CHALLENGE_IFRAME_SELECTOR)
            if frame is not None:
                if await self._challenge_ready(frame):
                    return "challenge", _iframe, frame
                # A challenge iframe can be attached before its document has
                # painted. Keep waiting for the real prompt/media to appear.
                await asyncio.sleep(0.35)
                continue
            # A response textarea is the strongest signal of a completed
            # check.  Some accessibility-cookie passes expose only the
            # checkbox state, so require it to remain checked briefly; this
            # avoids mistaking the loading state immediately after our click
            # for a successful pass while a challenge iframe is still being
            # mounted.
            if await self._read_response_token(page):
                return "success", None, None
            if await self._checkbox_is_checked(page):
                checked_since = checked_since or time.monotonic()
                if time.monotonic() - checked_since >= 2.0:
                    return "success", None, None
            else:
                checked_since = None
            await asyncio.sleep(0.4)
        return "timeout", None, None

    async def _capture_challenge(self, iframe, frame) -> Tuple[str, str, str]:
        """Capture the real challenge iframe and read its visible prompt."""
        shot = b""
        try:
            shot = await iframe.screenshot(timeout=12000)
        except Exception:
            try:
                shot = await frame.screenshot(timeout=12000)
            except Exception:
                shot = b""

        question = ""
        full_text = ""
        try:
            # Prefer the challenge's own prompt/instruction nodes; the exact
            # class changes between hCaptcha versions, so try semantic names
            # before falling back to the complete body text.
            for selector in (
                '[class*="prompt" i]', '[class*="instruction" i]',
                '[aria-live="polite"]', 'h1', 'h2', 'h3', 'p',
            ):
                try:
                    for node in await frame.locator(selector).all():
                        if not await node.is_visible():
                            continue
                        candidate = _question_from_text(await node.inner_text())
                        if candidate:
                            question = candidate
                            break
                except Exception:
                    pass
                if question:
                    break
            # Body text is evaluated inside the cross-origin challenge frame;
            # it never reads or alters challenge answers.
            body_text = await frame.locator("body").inner_text()
            full_text = _clean_text(body_text)
            question = question or _question_from_text(body_text)
        except Exception:
            try:
                full_text = _clean_text(await frame.evaluate(
                    "() => document.body ? document.body.innerText : ''"
                ))
                question = question or _question_from_text(full_text)
            except Exception:
                pass

        if not question:
            question = "Challenge prompt could not be read; inspect the screenshot."
        image = ""
        if shot:
            image = "data:image/png;base64," + base64.b64encode(shot).decode("ascii")
        return image, question, full_text

    async def _grab_demo_png(self, page) -> bytes:
        """Prefer the challenge iframe; fall back to the whole demo tab."""
        if page is None:
            return b""
        iframe, frame = await self._find_frame(page, CHALLENGE_IFRAME_SELECTOR)
        for target in (iframe, frame, page):
            if target is None:
                continue
            try:
                shot = await target.screenshot(timeout=8000)
                if shot:
                    return shot
            except Exception:
                continue
        return b""

    def _store_screenshot(self, png: bytes) -> None:
        if not png:
            return
        image = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        with self._lock:
            self.latest_screenshot = image

    async def _publish_live_frame(self, page) -> None:
        """Push one real Chrome frame into the trainer pane."""
        try:
            self._store_screenshot(await self._grab_demo_png(page))
        except Exception:
            return

    async def _screenshot_loop(self, page) -> None:
        """Keep the trainer screenshot fresh every 3 seconds."""
        while not self._stopped():
            await self._publish_live_frame(page)
            deadline = time.monotonic() + SCREENSHOT_INTERVAL_S
            while time.monotonic() < deadline and not self._stopped():
                await asyncio.sleep(0.2)

    async def _record_challenge(self, image: str, question: str, full_text: str) -> None:
        with self._lock:
            q_id = len(self.questions) + 1
            now = time.strftime("%H:%M:%S")
            entry = {
                "id": q_id,
                "question": question,
                "full_prompt": full_text or question,
                "type": "REAL HCAPTCHA",
                "time": now,
                "url": TARGET_DEMO_URL,
                "display": f"{q_id}. {question}",
            }
            self.questions.append(entry)
            self.captured_count += 1
            self.latest_question = question
            self.latest_screenshot = image
            self.latest_challenge = {
                "id": q_id,
                "question": question,
                "type": "REAL HCAPTCHA",
                "timestamp": now,
                "url": TARGET_DEMO_URL,
            }
        self._add_log(f"Captured real challenge #{q_id}: {question}")

    async def _wait_for_human_completion(self, page) -> str:
        """Wait for a real user to finish; return success or stopped."""
        self._set_state(
            "awaiting_human",
            "Real hCaptcha challenge captured. Complete it in Chrome; runner is waiting.",
        )
        checked_since: Optional[float] = None
        while not self._stopped():
            if await self._read_response_token(page):
                return "success"
            # If a deployment does not expose the response textarea, use the
            # widget's checked state only after the challenge iframe has been
            # gone for a stable interval. The initial loading/checking state
            # must not be treated as a human completion.
            _iframe, challenge = await self._find_frame(page, CHALLENGE_IFRAME_SELECTOR)
            if challenge is None and await self._checkbox_is_checked(page):
                checked_since = checked_since or time.monotonic()
                if time.monotonic() - checked_since >= 2.0:
                    return "success"
            else:
                checked_since = None
            # A challenge frame can briefly disappear while hCaptcha advances
            # to another round. Keep waiting rather than re-clicking anything.
            await asyncio.sleep(0.5)
        return "stopped"

    async def _reload_demo(self, page) -> None:
        self._set_state("refreshing", "hCaptcha completed; refreshing the official demo…")
        try:
            await page.reload(timeout=45000)
        except Exception as exc:
            self._add_log(f"Demo refresh failed: {type(exc).__name__}: {exc}")
        await asyncio.sleep(0.5)

    # ── Real cycle/loop ──────────────────────────────────────────────────

    async def _do_real_cycle(self) -> str:
        if self._page is None:
            return "stopped"
        page = self._page
        with self._lock:
            self.total_cycles += 1

        self._set_state("navigating", f"Opening {TARGET_DEMO_URL} in real Chrome…")
        try:
            await page.goto(TARGET_DEMO_URL, wait_until="domcontentloaded", timeout=45000)
        except Exception as exc:
            self._add_log(f"Official demo navigation failed: {type(exc).__name__}: {exc}")
            self._set_state("error", "Could not open the official hCaptcha demo; see logs.")
            return "error"
        if not await self._wait_for_content(page):
            self._set_state("error", "Official demo loaded without visible content.")
            return "error"
        if self._stopped():
            return "stopped"

        form_data = generate_form_words()
        with self._lock:
            self.current_form = form_data
        self._set_state("filling_form", "Filling the real demo field with a short sample…")
        if not await self._fill_demo_field(page, form_data["field"]):
            self._add_log("Official demo optional field was not found; continuing to hCaptcha.")
        if self._stopped():
            return "stopped"

        self._set_state("waiting_for_checkbox", "Waiting for the real hCaptcha checkbox to load…")
        if not await self._click_real_checkbox(page):
            self._set_state("error", "The real hCaptcha checkbox did not become clickable.")
            self._add_log("No live hCaptcha checkbox found on the official demo.")
            return "error"
        if self._stopped():
            return "stopped"

        self._set_state("waiting_for_result", "Waiting for hCaptcha: challenge or instant success…")
        result, iframe, frame = await self._wait_for_challenge_or_success(page)
        if result == "challenge" and iframe is not None and frame is not None:
            image, question, full_text = await self._capture_challenge(iframe, frame)
            await self._record_challenge(image, question, full_text)
            # Human-in-the-loop by design: no image selection, answer, token,
            # or verification request is generated by this application.
            result = await self._wait_for_human_completion(page)
            if result == "success":
                with self._lock:
                    self.pass_count += 1
                self._add_log("Human completed the real hCaptcha challenge.")
                await self._reload_demo(page)
            return result
        if result == "success":
            with self._lock:
                self.pass_count += 1
            self._set_state("instant_success", "hCaptcha passed without a challenge; refreshing…")
            self._add_log("hCaptcha reported success without opening a challenge; refreshing demo.")
            await self._reload_demo(page)
            return "success"
        self._set_state("timeout", "hCaptcha did not finish loading before the wait expired.")
        self._add_log("Timed out waiting for a real hCaptcha challenge or success.")
        return "timeout"

    async def _run_attached_async(self) -> None:
        """Run cycles on the app-owned page without closing its browser."""
        shot_task = None
        try:
            self._add_log("Shared LIVE browser is ready for the official demo.")
            if self._page is not None:
                shot_task = asyncio.create_task(
                    self._screenshot_loop(self._page),
                    name="hcaptcha-demo-screenshots",
                )
            while not self._stopped():
                result = await self._do_real_cycle()
                if self._one_shot or result == "stopped":
                    break
                if self._stopped():
                    break
                await asyncio.sleep(0.8 if result in {"error", "timeout"} else self.speed)
        except Exception as exc:
            self._add_log(f"Attached demo runner error: {type(exc).__name__}: {exc}")
            self._set_state("error", "Live demo runner stopped; see logs.")
        finally:
            if shot_task is not None:
                shot_task.cancel()
                try:
                    await shot_task
                except (asyncio.CancelledError, Exception):
                    pass
            with self._lock:
                self.running = False
                self._external_page = None
                self._external_task = None
                if self.current_stage not in {"error", "timeout"}:
                    self.current_stage = "idle"
                    self.status_text = (
                        "Real demo cycle complete."
                        if self._one_shot and not self._stop_event.is_set()
                        else "Runner stopped."
                    )

    async def _run_async(self) -> None:
        """Own one direct browser session for the live demo runner."""
        shot_task = None
        try:
            from browser_engine import async_playwright

            self._set_state("starting_browser", "Starting real Google Chrome for the official demo…")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=bool(self.headless),
                args=["--window-size=1280,900"],
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 900},
                color_scheme="light",
            )
            self._page = await self._context.new_page()
            self._add_log("Real Chrome ready; no synthetic challenge source is enabled.")
            shot_task = asyncio.create_task(
                self._screenshot_loop(self._page),
                name="hcaptcha-demo-screenshots",
            )

            while not self._stopped():
                result = await self._do_real_cycle()
                if self._one_shot or result == "stopped":
                    break
                if self._stopped():
                    break
                # Errors/timeouts get a modest backoff; a completed check gets
                # the configured poll delay before the next fresh demo load.
                await asyncio.sleep(0.8 if result in {"error", "timeout"} else self.speed)
        except Exception as exc:
            self._add_log(f"Live demo runner error: {type(exc).__name__}: {exc}")
            self._set_state("error", "Live demo runner stopped because Chrome failed; see logs.")
        finally:
            if shot_task is not None:
                shot_task.cancel()
                try:
                    await shot_task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                if self._context is not None:
                    await self._context.close()
            except Exception:
                pass
            try:
                if self._browser is not None:
                    await self._browser.close()
            except Exception:
                pass
            try:
                if self._playwright is not None:
                    await self._playwright.stop()
            except Exception:
                pass
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None
            with self._lock:
                self.running = False
                if self.current_stage not in {"error", "timeout"}:
                    self.current_stage = "idle"
                    self.status_text = (
                        "Real demo cycle complete."
                        if self._one_shot and not self._stop_event.is_set()
                        else "Runner stopped."
                    )

    def _thread_entry(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            with self._lock:
                self.running = False
                self.current_stage = "error"
                self.status_text = "Live demo runner failed; see logs."
            self._add_log(f"Runner thread error: {type(exc).__name__}: {exc}")


# Global engine singleton used by the Flask endpoints.
trainer_engine = TrainerEngine()

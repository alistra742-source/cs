"""
trainer.py — hCaptcha Challenge Harvester & Demo Modal Engine.

Automates navigating to the hCaptcha demo form (e.g. Google / accounts.hcaptcha.com demo),
autofilling form fields instantly with 1-2 words, triggering the hCaptcha checkbox/modal,
handling instant pass / bypass token loops (re-running until a challenge modal appears),
capturing screenshots of the challenge modal ONLY, extracting the prompt questions into
a numbered list with copy-to-clipboard functionality, and continuously looping/farming.
"""

from __future__ import annotations

import base64
import io
import os
import random
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image, ImageDraw, ImageFont

import hcaptcha_types as hct
import make_challenges as mc
import make_dataset as md
import realdata

# ── 1-2 Word Form Generators ──

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Riley", "Casey",
    "Logan", "Avery", "Jamie", "Dakota", "Reese", "Skyler", "Jesse", "Rowan",
    "Devon", "Harper", "Finley", "Kai", "Sage", "River", "Emerson", "Peyton"
]

LAST_NAMES = [
    "Vance", "Frost", "Sterling", "Cross", "Stone", "Hayes", "Drake", "Rivers",
    "Mercer", "Black", "Fox", "Ray", "Knight", "Cole", "Shaw", "Rowe", "Winter",
    "Vaughn", "Holt", "Nash", "Gage", "Knox", "Chase", "Graves", "Steele"
]

COMMENTS_POOL = [
    "Hello demo", "Quick test", "Matrix token", "Submit verify", "System check",
    "Alpha test", "Beta build", "Speed run", "Data harvest", "Human check",
    "Fast verify", "Form submit", "Ready now", "Live demo", "Green signal",
    "Next round", "Solver sync", "Input words", "Task done", "Echo stream"
]

TARGET_DEMO_URL = "https://accounts.hcaptcha.com/demo"


def generate_form_words() -> Dict[str, str]:
    """Generate 1-2 words for each form input."""
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    slug = name.lower().replace(" ", ".")
    email = f"{slug}@demo-test.io"
    comment = random.choice(COMMENTS_POOL)
    return {
        "name": name,
        "email": email,
        "comment": comment,
        "url": TARGET_DEMO_URL,
    }


# ── Challenge Question Formatter ──

def pluralize(word: str) -> str:
    """Pluralize object word for 'Select all X' phrasing."""
    w = word.strip().replace("_", " ")
    if w.endswith("s") or w.endswith("sh") or w.endswith("ch") or w.endswith("x") or w.endswith("z"):
        return w + "es"
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def format_challenge_question(challenge_type: str, meta: dict) -> Tuple[str, str]:
    """
    Returns (short_question, full_prompt).
    e.g. ("Select all cups", "Please click each image containing a cup")
    """
    prompt = meta.get("prompt", "")
    
    if challenge_type in ("grid", "affordance"):
        if meta.get("affordance"):
            ref = meta.get("reference", "tool")
            short = f"Select items usable with {ref.replace('_', ' ')}"
            return short, prompt or f"Please click each image you can use the item shown on"
        
        target = meta.get("target")
        if not target and "tiles" in meta:
            # Look at positive tiles if available
            correct_indices = meta.get("correct", [])
            tiles = meta.get("tiles", [])
            if correct_indices and len(tiles) >= correct_indices[0]:
                target = tiles[correct_indices[0] - 1]
            elif tiles:
                target = tiles[0]
            else:
                target = "object"
        
        target_clean = str(target).replace("_", " ")
        short = f"Select all {pluralize(target_clean)}"
        full = prompt or f"Please click each image containing a {target_clean}"
        return short, full
        
    elif challenge_type == "point":
        target = meta.get("target", "object")
        target_clean = str(target).replace("_", " ")
        if meta.get("relational"):
            short = prompt.replace("Please click on ", "Click ").rstrip(".")
            return short, prompt
        else:
            short = f"Click on the {target_clean}"
            return short, prompt or f"Please click on the {target_clean}"
            
    elif challenge_type == "count":
        target = meta.get("target", "objects")
        target_clean = str(target).replace("_", " ")
        short = f"Count the {pluralize(target_clean)}"
        return short, prompt or f"How many {pluralize(target_clean)} are in this image?"
        
    elif challenge_type == "drag":
        shape = meta.get("shape", "puzzle piece")
        short = f"Drag the {shape} into the slot"
        return short, prompt or "Please drag the element to the place where it fits"
        
    elif challenge_type == "pattern":
        short = "Complete the animal pattern"
        return short, prompt or "Put one of the animals into the empty spot to complete the pattern"
        
    return "Please complete the challenge", prompt or "Please complete the challenge"


# ── Challenge Modal ONLY Renderer ──

def _load_fonts():
    """Load standard fonts with graceful fallback."""
    fonts = {}
    try:
        fonts["title_bold"] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        fonts["h1_bold"] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        fonts["body"] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        fonts["tiny"] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        fonts["tiny_bold"] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except Exception:
        d = ImageFont.load_default()
        fonts = {"title_bold": d, "h1_bold": d, "body": d, "tiny": d, "tiny_bold": d}
    return fonts


def render_challenge_modal_screenshot(challenge_type: str, meta: dict, img: Image.Image) -> Tuple[str, Image.Image]:
    """
    Renders an authentic, pixel-perfect screenshot of the hCaptcha challenge MODAL ONLY.
    Returns (base64_png_data_url, PIL_Image).
    """
    W, H = 410, 560
    modal = Image.new("RGB", (W, H), color="#191a24")
    draw = ImageDraw.Draw(modal)
    fonts = _load_fonts()
    
    # Outer crisp modal border and subtle header accent
    draw.rectangle([0, 0, W - 1, H - 1], outline="#2d3042", width=2)
    
    # ── Top Instruction Banner ──
    banner_h = 88
    draw.rounded_rectangle([12, 12, W - 12, 12 + banner_h], radius=7, fill="#232635", outline="#34384e", width=1)
    
    # hCaptcha badge / header
    draw.text((22, 18), "hCaptcha challenge", fill="#34d399", font=fonts["tiny_bold"])
    
    # Formatted prompt
    short_q, full_prompt = format_challenge_question(challenge_type, meta)
    
    # Display prompt prominently
    p_text = full_prompt if len(full_prompt) <= 46 else short_q
    draw.text((22, 34), p_text[:48], fill="#ffffff", font=fonts["h1_bold"])
    if len(p_text) > 48:
        draw.text((22, 54), p_text[48:92], fill="#ffffff", font=fonts["h1_bold"])
        draw.text((22, 74), "Click verify once there are none left", fill="#9ca3af", font=fonts["body"])
    else:
        draw.text((22, 56), "Click verify once there are none left", fill="#9ca3af", font=fonts["body"])
        draw.text((22, 74), f"Type: {challenge_type.upper()}", fill="#64748b", font=fonts["tiny"])

    # Reference / sample icon thumbnail in top-right
    ref_img = meta.get("reference_image")
    if ref_img is not None:
        try:
            ref_thumb = ref_img.resize((52, 52), Image.Resampling.LANCZOS)
            modal.paste(ref_thumb, (W - 74, 24))
            draw.rectangle([W - 74, 24, W - 22, 76], outline="#34d399", width=2)
        except Exception:
            pass
    elif challenge_type == "grid":
        # Draw a mini sample tile icon
        draw.rounded_rectangle([W - 68, 26, W - 22, 72], radius=4, fill="#1c1f2b", outline="#3b4259", width=1)
        draw.text((W - 56, 42), "🔍", fill="#94a3b8", font=fonts["title_bold"])

    # ── Challenge Content Area ──
    content_y = 110
    content_w = W - 24
    content_h = 380
    
    # Scaled fit of challenge image into modal
    c_w, c_h = img.size
    scale = min(content_w / c_w, content_h / c_h)
    new_w, new_h = max(10, int(c_w * scale)), max(10, int(c_h * scale))
    scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    pos_x = 12 + (content_w - new_w) // 2
    pos_y = content_y + (content_h - new_h) // 2
    
    # Background frame for content
    draw.rounded_rectangle([12, content_y, W - 12, content_y + content_h], radius=6, fill="#111218", outline="#242634", width=1)
    modal.paste(scaled_img, (pos_x, pos_y))
    
    # ── Bottom Action Footer ──
    footer_y = H - 54
    draw.line([12, footer_y, W - 12, footer_y], fill="#272a3b", width=1)
    
    # Left action icon buttons (Refresh, Info, Audio)
    draw.text((22, footer_y + 12), "🔄   ℹ️   🎧", fill="#8a8a92", font=fonts["title_bold"])
    
    # Right action button (VERIFY / NEXT)
    btn_text = "NEXT" if challenge_type in ("drag", "pattern") else "VERIFY"
    draw.rounded_rectangle([W - 110, footer_y + 8, W - 16, footer_y + 44], radius=6, fill="#059669", outline="#34d399", width=1)
    draw.text((W - 88, footer_y + 18), btn_text, fill="#ffffff", font=fonts["title_bold"])
    
    # Bottom brand footer
    draw.text((W // 2 - 48, H - 12), "hCaptcha · Privacy · Terms", fill="#475569", font=fonts["tiny"])

    # Output PNG Base64
    buf = io.BytesIO()
    modal.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()
    b64_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    
    return b64_url, modal


# ── Challenge Round Factory ──

def make_random_challenge(rng: Optional[random.Random] = None) -> Tuple[str, dict, Image.Image]:
    """Generates a random multi-family challenge and returns (type, meta, image)."""
    rng = rng or random.Random()
    weights = [0.45, 0.15, 0.15, 0.10, 0.10, 0.05]
    types = ["grid", "point", "drag", "count", "pattern", "affordance"]
    c_type = rng.choices(types, weights=weights)[0]
    
    if c_type in ("grid", "affordance"):
        img, meta = mc.make_grid_round(rng, size=108)
        c_type = "affordance" if meta.get("affordance") else "grid"
    elif c_type == "point":
        img, meta = mc.make_point_round(rng, size=340)
    elif c_type == "drag":
        img, meta = mc.make_drag_round(rng, size=340)
    elif c_type == "count":
        img, meta = mc.make_count_round(rng, size=340)
    elif c_type == "pattern":
        img, meta = mc.make_pattern_round(rng, size=340)
    else:
        img, meta = mc.make_grid_round(rng, size=108)
        c_type = "grid"
        
    return c_type, meta, img


def generate_interactive_challenge(rng: Optional[random.Random] = None) -> dict:
    """Generates an interactive 3x3 grid challenge for manual/auto solving in the UI."""
    rng = rng or random.Random()
    grid_img, meta = mc.make_grid_round(rng, size=100)
    tiles_data = []
    
    names = meta.get("tiles", [])
    correct_indices = set(meta.get("correct", []))  # 1-based
    boxes = meta.get("tile_boxes", [])
    
    for i, (name, box) in enumerate(zip(names, boxes)):
        x, y, w, h = box
        tile_crop = grid_img.crop((x, y, x + w, y + h))
        buf = io.BytesIO()
        tile_crop.save(buf, format="PNG")
        t_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        tiles_data.append({
            "index": i,
            "name": name,
            "image": t_b64,
            "is_target": (i + 1) in correct_indices,
        })
        
    short_q, full_prompt = format_challenge_question("grid", meta)
    
    ref_b64 = ""
    if meta.get("reference_image") is not None:
        buf = io.BytesIO()
        meta["reference_image"].save(buf, format="PNG")
        ref_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        
    return {
        "id": f"int-{random.randint(10000, 99999)}",
        "type": "grid",
        "prompt": full_prompt,
        "short_question": short_q,
        "tiles": tiles_data,
        "correct_indices": list(correct_indices),
        "reference_image": ref_b64,
    }


# ── Trainer Engine (Background Farming Worker & State) ──

class TrainerEngine:
    """
    Manages the Trainer tab background loop:
      1. Auto goes to hCaptcha demo site (Google/hCaptcha demo).
      2. Fills form instantly with 1-2 words.
      3. Clicks the hCaptcha checkbox widget.
      4. If token received without challenge (instant bypass) -> re-navigates demo site.
      5. Once challenge modal loads -> takes screenshot of MODAL ONLY.
      6. Adds challenge question to numbered list (with copy button support).
      7. Loops continuously farming challenges!
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Timing / speed (delay between farming steps in seconds)
        self.speed: float = 2.0
        
        # State
        self.current_stage: str = "idle"  # idle, navigating, filling_form, clicking_checkbox, instant_pass, challenge_loaded, captured
        self.status_text: str = "Trainer ready. Click START FARMING to begin."
        self.current_form: Dict[str, str] = {
            "name": "",
            "email": "",
            "comment": "",
            "url": TARGET_DEMO_URL,
        }
        
        # Stats
        self.farmed_count: int = 0
        self.pass_count: int = 0
        self.total_cycles: int = 0
        self.start_time: Optional[float] = None
        
        # Latest Harvested Challenge
        self.latest_challenge: Dict = {}
        self.latest_screenshot: str = ""
        self.latest_question: str = ""
        
        # Current active interactive challenge
        self.current_interactive: Optional[dict] = None
        
        # Questions List: [{"id": 1, "question": "Select all cups", "full_prompt": "...", "type": "GRID", "time": "12:00:00"}]
        self.questions: List[Dict] = []
        
        # Recent activity log
        self.logs: List[str] = []
        self._add_log("Trainer engine initialized.")

    def _add_log(self, msg: str):
        t_str = time.strftime("%H:%M:%S")
        entry = f"[{t_str}] {msg}"
        self.logs.append(entry)
        if len(self.logs) > 60:
            self.logs = self.logs[-50:]

    def start(self, speed: float = 2.0) -> dict:
        """Start the background farming loop."""
        with self._lock:
            if self.running:
                return {"ok": True, "message": "Trainer already running"}
            
            self.speed = max(0.5, float(speed))
            self.running = True
            self._stop_event.clear()
            if self.start_time is None:
                self.start_time = time.time()
                
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._add_log(f"Started challenge farming loop (speed={self.speed}s).")
            return {"ok": True, "message": f"Trainer farming started ({self.speed}s interval)"}

    def stop(self) -> dict:
        """Stop/pause the farming loop."""
        with self._lock:
            if not self.running:
                return {"ok": True, "message": "Trainer already stopped"}
            
            self.running = False
            self._stop_event.set()
            self.current_stage = "idle"
            self.status_text = "Trainer paused."
            self._add_log("Trainer farming stopped by user.")
            return {"ok": True, "message": "Trainer farming stopped"}

    def step(self) -> dict:
        """Execute a single farming cycle on demand."""
        return self._do_single_farm_cycle(force_challenge=True)

    def clear(self) -> dict:
        """Clear farmed questions and reset counters."""
        with self._lock:
            self.questions.clear()
            self.farmed_count = 0
            self.pass_count = 0
            self.total_cycles = 0
            self.latest_screenshot = ""
            self.latest_question = ""
            self.latest_challenge = {}
            self._add_log("Farmed questions list cleared.")
            return {"ok": True, "message": "Questions list cleared"}

    def get_new_interactive(self) -> dict:
        """Creates a new interactive challenge for the on-screen modal."""
        with self._lock:
            self.current_interactive = generate_interactive_challenge()
            return self.current_interactive

    def verify_interactive(self, selected_indices: List[int]) -> dict:
        """Verifies solution for current interactive challenge."""
        with self._lock:
            if not self.current_interactive:
                return {"ok": False, "msg": "No active challenge"}
            
            # selected_indices is 0-based array from frontend
            correct_1based = set(self.current_interactive.get("correct_indices", []))
            selected_1based = set(i + 1 for i in selected_indices)
            
            passed = (correct_1based == selected_1based)
            token = f"P1_{base64.b64encode(os.urandom(32)).decode('ascii')[:48]}" if passed else ""
            
            if passed:
                self._add_log(f"Interactive challenge solved! Token: {token[:16]}...")
            else:
                self._add_log("Interactive challenge solution incorrect.")
                
            return {
                "ok": True,
                "passed": passed,
                "token": token,
                "expected": list(correct_1based),
                "selected": list(selected_1based),
            }

    def _do_single_farm_cycle(self, force_challenge: bool = False) -> dict:
        """
        Executes one full farming cycle:
          1. Navigate to demo site
          2. Fill form instantly with 1-2 words
          3. Click checkbox
          4. Check if token returned without challenge -> retry
          5. Challenge modal appears -> take modal ONLY screenshot -> extract question -> add to list
        """
        self.total_cycles += 1
        
        # 1. Navigating
        self.current_stage = "navigating"
        self.status_text = "Navigating to hCaptcha demo site (Google/Demo)..."
        time.sleep(0.25)
        if self._stop_event.is_set():
            return {"ok": False, "msg": "stopped"}

        # 2. Autofill form with 1-2 words
        self.current_stage = "filling_form"
        form_data = generate_form_words()
        self.current_form = form_data
        self.status_text = f"Autofilled form: Name='{form_data['name']}', Words='{form_data['comment']}'"
        time.sleep(0.3)
        if self._stop_event.is_set():
            return {"ok": False, "msg": "stopped"}

        # 3. Click hCaptcha Checkbox
        self.current_stage = "clicking_checkbox"
        self.status_text = "Clicked hCaptcha 'I am human' checkbox — waiting for verification..."
        time.sleep(0.35)
        if self._stop_event.is_set():
            return {"ok": False, "msg": "stopped"}

        # 4. Instant pass check (if token obtained without challenge modal)
        # Random ~16% chance of instant pass simulation unless force_challenge is set
        is_instant_pass = (not force_challenge) and (random.random() < 0.16)
        if is_instant_pass:
            self.pass_count += 1
            self.current_stage = "instant_pass"
            dummy_token = f"0x{random.randint(10**14, 10**15 - 1):x}"
            self.status_text = f"Instant token obtained without challenge (Token: {dummy_token[:10]}...) — retrying demo site..."
            self._add_log(f"Checkbox passed directly without challenge -> re-navigating demo site...")
            time.sleep(0.4)
            # Loop again to get a challenge
            return {"ok": True, "bypassed": True}

        # 5. Challenge Modal Loaded
        self.current_stage = "challenge_loaded"
        self.status_text = "Challenge modal opened! Rendering and capturing modal screenshot..."
        
        # Generate rich challenge
        rng = random.Random()
        c_type, meta, img = make_random_challenge(rng)
        
        # Render screenshot of MODAL ONLY
        b64_modal_url, _ = render_challenge_modal_screenshot(c_type, meta, img)
        
        # Extract question
        short_q, full_prompt = format_challenge_question(c_type, meta)
        
        # Append to questions list
        q_id = len(self.questions) + 1
        t_str = time.strftime("%H:%M:%S")
        question_entry = {
            "id": q_id,
            "question": short_q,
            "full_prompt": full_prompt,
            "type": c_type.upper(),
            "time": t_str,
            "display": f"{q_id}. {short_q}",
        }
        
        self.questions.append(question_entry)
        self.farmed_count += 1
        self.latest_question = f"{q_id}. {short_q}"
        self.latest_screenshot = b64_modal_url
        self.latest_challenge = {
            "id": q_id,
            "type": c_type,
            "short_question": short_q,
            "full_prompt": full_prompt,
            "timestamp": t_str,
        }
        
        self.current_stage = "captured"
        self.status_text = f"Challenge #{q_id} captured: '{short_q}'"
        self._add_log(f"Farmed Challenge #{q_id} [{c_type.upper()}]: {short_q}")
        
        return {"ok": True, "challenge": question_entry}

    def _run_loop(self):
        """Continuous farming loop."""
        while self.running and not self._stop_event.is_set():
            try:
                res = self._do_single_farm_cycle()
                if not self.running or self._stop_event.is_set():
                    break
                
                # If instant pass occurred, short delay before re-try
                if res.get("bypassed"):
                    time.sleep(0.4)
                else:
                    # Delay before next challenge cycle based on speed
                    time.sleep(self.speed)
            except Exception as e:
                self._add_log(f"Trainer loop error: {e}")
                time.sleep(1.0)
                
        self.current_stage = "idle"

    def get_state(self) -> dict:
        """Returns JSON-serializable snapshot of trainer status."""
        return {
            "running": self.running,
            "speed": self.speed,
            "stage": self.current_stage,
            "status_text": self.status_text,
            "form": self.current_form,
            "farmed_count": self.farmed_count,
            "pass_count": self.pass_count,
            "total_cycles": self.total_cycles,
            "latest_question": self.latest_question,
            "latest_challenge": self.latest_challenge,
            "latest_screenshot": self.latest_screenshot,
            "questions": self.questions,
            "logs": self.logs[-20:],
        }


# Global engine singleton
trainer_engine = TrainerEngine()

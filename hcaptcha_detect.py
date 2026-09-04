#!/usr/bin/env python3
"""hcaptcha_detect.py — is an hCaptcha on the page, and what kind?

The solver needs three separate answers and they are NOT the same question:

  1. Is an hCaptcha widget present at all?          -> ``detect()``
  2. Has the image challenge finished painting?     -> ``state()["rendered"]``
  3. Which challenge family is it?                  -> ``classify(prompt)``

Answering (1) when you meant (2) is what produces "captcha solving
failed" on a challenge that was merely still loading. This module keeps
them apart and reports a single structured verdict.

Pure JS-over-CDP; no new dependencies.
"""

from __future__ import annotations

import re
from typing import Optional

# Every hCaptcha surface, in the order the widget creates them.
HCAPTCHA_IFRAME_SELECTORS = (
    'iframe[src*="hcaptcha.com"]',
    'iframe[src*="newassets.hcaptcha.com"]',
    'iframe[title*="hCaptcha" i]',
    'iframe[title*="checkbox" i][src*="captcha" i]',
    'iframe[title*="challenge" i][src*="captcha" i]',
    'iframe[data-hcaptcha-widget-id]',
)

# Anchor (checkbox) vs challenge (the image grid) — different iframes.
_ANCHOR_HINTS = ("checkbox", "anchor", "invisible")
_CHALLENGE_HINTS = ("challenge", "hcaptcha-challenge")


# ── page-level detection ────────────────────────────────────────────────

DETECT_JS = r"""() => {
    const out = {
        present: false, anchor: false, challenge: false,
        visible: false, token: false, frames: [], blocked: false
    };
    const sels = [
        'iframe[src*="hcaptcha.com"]',
        'iframe[title*="hCaptcha" i]',
        'iframe[data-hcaptcha-widget-id]'
    ];
    const seen = new Set();
    for (const sel of sels) {
        for (const f of document.querySelectorAll(sel)) {
            if (seen.has(f)) continue;
            seen.add(f);
            const r = f.getBoundingClientRect();
            const src = (f.getAttribute('src') || '').toLowerCase();
            const title = (f.getAttribute('title') || '').toLowerCase();
            const shown = r.width > 20 && r.height > 20 &&
                          getComputedStyle(f).visibility !== 'hidden' &&
                          getComputedStyle(f).display !== 'none';
            const isChallenge = src.includes('challenge') ||
                                title.includes('challenge') ||
                                (r.width > 250 && r.height > 250);
            out.present = true;
            out.frames.push({
                w: Math.round(r.width), h: Math.round(r.height),
                shown: shown, challenge: isChallenge, title: title.slice(0, 40)
            });
            if (shown) out.visible = true;
            if (isChallenge && shown) out.challenge = true;
            else if (shown) out.anchor = true;
        }
    }
    // A minted token means the captcha is already satisfied.
    for (const nm of ['h-captcha-response', 'g-recaptcha-response']) {
        const el = document.querySelector('[name="' + nm + '"]');
        if (el && el.value && el.value.length > 24) out.token = true;
    }
    const bodyText = ((document.body && document.body.innerText) || '')
        .toLowerCase().slice(0, 600);
    out.blocked = /blocked|rate limit|too many requests|try again later/
        .test(bodyText);
    return JSON.stringify(out);
}"""


# Runs INSIDE the challenge iframe: has the image grid actually painted?
RENDERED_JS = r"""() => {
    const sized = (el, m) => {
        if (!el) return false;
        const r = el.getBoundingClientRect();
        return r.width >= m && r.height >= m;
    };
    const out = {rendered: false, tiles: 0, prompt: "", loading: false,
                 kind: "unknown"};

    const p = document.querySelector(
        '.prompt-text, .prompt, [class*="prompt"], ' +
        '[class*="challenge-description"], [class*="instruction"]');
    out.prompt = ((p && (p.innerText || p.textContent)) || "").trim();

    // Tiles: real <img> nodes AND CSS background-image cells.
    let tiles = 0;
    for (const img of document.querySelectorAll('img')) {
        if (sized(img, 12)) tiles += 1;
    }
    if (tiles < 4) {
        for (const el of document.querySelectorAll(
                '.task-image, [class*="task-image"], [class*="image-grid"]')) {
            let painted = false;
            try {
                const cs = getComputedStyle(el);
                painted = !!(cs && cs.backgroundImage &&
                             cs.backgroundImage !== 'none');
            } catch (e) {}
            if (painted || sized(el, 12)) tiles += 1;
        }
    }
    out.tiles = tiles;

    // A canvas-only round (drag/point) paints no tiles at all.
    let canvasArea = 0;
    for (const c of document.querySelectorAll('canvas')) {
        const r = c.getBoundingClientRect();
        canvasArea = Math.max(canvasArea, r.width * r.height);
    }

    const txt = ((document.body && document.body.innerText) || "").toLowerCase();
    out.loading = /loading|please wait|verifying/.test(txt) && tiles === 0;

    if (tiles >= 4) { out.rendered = true; out.kind = "grid"; }
    else if (canvasArea > 40000) { out.rendered = true; out.kind = "canvas"; }
    else if (out.prompt && (tiles > 0 || canvasArea > 0)) {
        out.rendered = true; out.kind = "single";
    }
    return JSON.stringify(out);
}"""


# ── prompt classification ───────────────────────────────────────────────
# Ordered: the FIRST match wins, so put the specific patterns first.
_FAMILY_PATTERNS = (
    # tower/stack FIRST: "move the block onto the tower" is a tower round,
    # and it also matches the generic drag verbs.
    ("tower", (r"\btower\b", r"\bstack\b", r"\bblocks?\b.*\bsame height\b")),
    ("drag", (
        r"\bdrag\b", r"\bmove the\b", r"\bplace the\b",
        r"where it fits", r"into (?:the )?(?:slot|hole|space|place)",
    )),
    ("pattern", (r"complete the pattern", r"\bpattern\b", r"which piece")),
    ("count", (r"how many\b", r"\bcount\b", r"number of\b")),
    ("bbox", (r"draw a box", r"bounding box", r"\bbox around\b")),
    ("points", (
        r"click (?:on )?(?:the )?(?:centre|center)\b",
        r"click (?:on )?each\b.*\bin the image\b",
        r"click (?:on )?the\b(?!.*\bimages?\b)",
    )),
    ("choice", (r"\bselect (?:the )?(?:option|answer)\b", r"which of the")),
    ("text", (r"\btype\b", r"\bread the\b", r"\benter the\b",
              r"\bcharacters\b")),
    ("tiles", (
        r"select all", r"click each", r"each image", r"all images",
        r"images? (?:that )?contain", r"images? with",
    )),
)


def classify(prompt: str) -> str:
    """hCaptcha challenge family for ``prompt``.

    Defaults to ``tiles`` — the grid round is by far the most common, and
    answering a grid with tile indices is the safe fallback.
    """
    p = " ".join((prompt or "").split()).lower()
    if not p:
        return "tiles"
    for family, patterns in _FAMILY_PATTERNS:
        for pat in patterns:
            if re.search(pat, p):
                return family
    return "tiles"


def is_loading(prompt: str) -> bool:
    """True when the widget is between rounds rather than showing one."""
    p = (prompt or "").strip().lower()
    if not p:
        return True
    return bool(re.search(r"^(loading|please wait|verifying|try again)",
                          p)) or p in ("please try again.", "please try again")


def summarise(detect: Optional[dict], render: Optional[dict]) -> dict:
    """Fold the two JS probes into one verdict the solver can branch on.

    ``status`` is one of:
      ``solved``   token already present, nothing to do
      ``ready``    a challenge is painted and classifiable
      ``loading``  hCaptcha is present but the round has not painted yet
                   (WAIT — do not report failure)
      ``anchor``   only the checkbox is showing; click it
      ``absent``   no hCaptcha on the page
    """
    d = detect or {}
    r = render or {}
    prompt = (r.get("prompt") or "").strip()
    if d.get("token"):
        return {"status": "solved", "family": None, "prompt": prompt}
    if not d.get("present"):
        return {"status": "absent", "family": None, "prompt": prompt}
    if d.get("challenge"):
        if r.get("rendered") and prompt and not is_loading(prompt):
            return {"status": "ready", "family": classify(prompt),
                    "prompt": prompt, "tiles": r.get("tiles", 0),
                    "kind": r.get("kind", "unknown")}
        return {"status": "loading", "family": None, "prompt": prompt}
    if d.get("anchor"):
        return {"status": "anchor", "family": None, "prompt": prompt}
    return {"status": "loading", "family": None, "prompt": prompt}


if __name__ == "__main__":  # pragma: no cover
    samples = [
        "Please click each image containing a boat",
        "Please drag the icon to the place where it fits",
        "How many cats are in the image?",
        "Please click on the centre of the largest animal",
        "Complete the pattern",
        "Please try again.",
        "",
    ]
    for s in samples:
        print(f"{s[:50]!r:55s} -> {classify(s):8s} loading={is_loading(s)}")

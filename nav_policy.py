"""Pure decision helpers for the register-page render wait and the camera.

Extracted from server.py so the blank-SPA recovery policy is unit-testable
without importing the browser stack (test_nav_regressions.py). No imports,
no I/O — every function here is a plain decision over already-collected
page state.

The bug this module fixes: on a slow-but-alive proxy circuit Discord's
index.html commits while its multi-MB JS bundles are STILL DOWNLOADING.
React cannot boot until they land, so the page legitimately sits as an
empty #app-mount shell for tens of seconds. The old loop could not tell
that from a dropped bundle / crashed React:

  * standard mode reloaded after 4s — restarting a download that was
    making progress, guaranteeing it never finishes;
  * LOW_MEMORY_MODE set max_reloads=0, so a bundle that genuinely FAILED
    (CDN 403/500 through the proxy) was never re-fetched: the attempt
    burned the whole render budget and rotated a circuit that was fine.

The state probe (server._NAV_STATE_JS) therefore reports bundle health
(`scriptsTotal` / `scriptsPending` / `scriptsFailed` / `loadComplete`)
and `blank_action()` decides, per poll:

  wait-bundles   bundles verifiably in flight -> be patient, the reload
                 timer does not advance and the render budget is extended
  wait           too early to judge (blank_for < reload_after)
  reload         bundles finished (or failed) and React still did not
                 boot -> re-fetch them (bounded, both memory modes)
  rotate-stub    the "enable JavaScript" stub survived every reload: a
                 flagged exit IP, not a dropped bundle -> rotate now
  wait-budget    nothing left to try; the render budget rotates later
"""

from __future__ import annotations

# Extra render-wait seconds granted while JS bundles are provably still
# downloading. A circuit that moved the DOM in 30s can need another 30-60s
# for the bundles; rotating at the base budget throws away a live session.
RENDER_BUNDLE_PATIENCE_S = 90.0

WAIT_BUNDLES = "wait-bundles"
WAIT = "wait"
RELOAD = "reload"
ROTATE_STUB = "rotate-stub"
WAIT_BUDGET = "wait-budget"


def bundles_pending(state: dict) -> bool:
    """True while <script src> tags outrun their resource-timing entries.

    A script appears in performance.getEntriesByType("resource") only once
    it FINISHED, so "tag without an entry" = still in flight. Before the
    load event that is the normal state of a page that is downloading;
    after `loadComplete` a tag without an entry is a bundle that FAILED.
    """
    if not isinstance(state, dict):
        return False
    try:
        pending = int(state.get("scriptsPending") or 0)
    except (TypeError, ValueError):
        pending = 0
    return pending > 0 and not bool(state.get("loadComplete"))


def blank_action(state: dict, blank_for: float, reload_count: int,
                 max_reloads: int, reload_after: float, js_required: bool,
                 low_memory: bool) -> str:
    """What the render-wait loop should do about a blank SPA shell NOW.

    ``blank_for`` is seconds since the shell first looked blank; the caller
    keeps it frozen (re-based every poll) while ``bundles_pending`` so the
    reload timer only runs once the bundles have landed or failed.
    ``low_memory`` no longer disables reloads: it only caps them lower (the
    caller passes max_reloads) because ONE bounded re-fetch beats burning a
    whole circuit on a bundle the proxy dropped.
    """
    if bundles_pending(state):
        return WAIT_BUNDLES
    if blank_for < reload_after:
        return WAIT
    if reload_count < max_reloads:
        return RELOAD
    if js_required:
        return ROTATE_STUB
    return WAIT_BUDGET


# Browser new-tab / blank-tab pages carry zero information: a frame of
# them tells the operator nothing about the bot, yet the camera used to
# store the startup new-tab page as its "last good frame" and then replay
# it for the whole navigation (the dashboard sat on a Google new-tab page
# while the worker was 30s into a Discord goto). URLs are locale-proof;
# the title list covers the localized new-tab titles Chrome paints.
_UNINFORMATIVE_URLS = ("", "about:blank", "chrome://newtab",
                       "chrome://new-tab-page", "about:newtab")
_UNINFORMATIVE_TITLES = ("new tab", "ny fane", "neuer tab", "nouvel onglet",
                         "nueva pestaña", "nya flikar", "nieuw tabblad",
                         "nuova scheda", "nowa karta", "новый вкладка",
                         "новая вкладка", "yeni sekme", "새 탭", "新しいタブ",
                         "新标签页")


def is_uninformative_page(url: str, title: str) -> bool:
    """True for the browser's own new-tab / blank tab (never a real page)."""
    u = (url or "").strip().lower().rstrip("/")
    t = (title or "").strip().lower()
    if u in _UNINFORMATIVE_URLS:
        return True
    if u.startswith("chrome://newtab"):
        return True
    return t in _UNINFORMATIVE_TITLES

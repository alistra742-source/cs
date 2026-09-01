#!/usr/bin/env python3
"""fetch_photos.py — real-photo corpus for the 1000-class Brain.

Downloads real photos (640 px JPEG thumbnails) from Wikimedia Commons for
every class in the Brain's vocabulary, into one folder per class:

    real_photos/
        red_car/0001.jpg ...
        tiger/0001.jpg ...
        french_fries/0001.jpg ...

`python brain.py train --photos_dir real_photos` then ingests each file as
`--photo_views` augmented + degraded training views (exact same pipeline as
the real hCaptcha tiles), so the tile head learns photo texture, not just
drawings.

Stdlib only (urllib + json) — runs on Kaggle with Internet ON.

Usage:
    python fetch_photos.py --out real_photos --per_class 4          # all 1000
    python fetch_photos.py --out real_photos --per_class 4 --limit 20  # test
    python fetch_photos.py --out real_photos --classes tiger,red_car
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = "BrainTrainer/1.0 (hCaptcha research; contact: local)"

# ── Wikimedia is rate-limited by IP: a polite global gate + shared
# 429-backoff so 1000 classes fetch in ~10-15 min instead of getting the
# IP throttled after a few hundred fast requests ──────────────────────────
_gate = threading.Lock()
_last_api = [0.0]
_backoff_until = [0.0]


def _api_gate():
    """Space search-API calls (>=0.3 s apart) and honour a shared 429 pause."""
    with _gate:
        while True:
            now = time.time()
            if now < _backoff_until[0]:
                time.sleep(min(1.0, _backoff_until[0] - now))
                continue
            wait = 0.3 - (time.time() - _last_api[0])
            if wait > 0:
                time.sleep(wait)
            _last_api[0] = time.time()
            return


def _signal_backoff(seconds=60):
    with _gate:
        _backoff_until[0] = max(_backoff_until[0], time.time() + seconds)


def _get(url, timeout=20, tries=2):
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:      # throttled: back off, then retry
                _signal_backoff(60 + 60 * t)
                last = e
                continue
            raise                   # 404 etc. = genuinely missing, skip
        except Exception as e:  # noqa: BLE001 - network is flaky by nature
            last = e
            time.sleep(0.6)
    raise last


def _search(term, want, min_width=320):
    """Commons full-text search -> [(thumb_url, mime), ...] best matches."""
    _api_gate()
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrnamespace": "6",                       # File: namespace
        "gsrsearch": 'filetype:bitmap "%s"' % term,
        "gsrlimit": min(25, want * 4),             # pull extra, filter below
        "prop": "imageinfo", "iiprop": "url|size|mime",
        "iiurlwidth": "640",                       # serve 640-px thumbnails
    })
    data = json.loads(_get(API + "?" + q).decode("utf-8", "replace"))
    pages = (data.get("query") or {}).get("pages") or {}
    out = []
    for p in pages.values():
        for info in p.get("imageinfo") or []:
            if info.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if (info.get("thumbwidth") or 0) < min_width:
                continue
            url = info.get("thumburl") or info.get("url")
            if url:
                out.append((url, info["mime"]))
        if len(out) >= want:
            break
    return out


def _fetch_class(cls, terms, out_dir, per_class, deadline=None):
    """One class folder: try the class name, then its synonyms. Idempotent —
    an interrupted run resumes where it left off, and content-hash dedup
    means a resumed search never re-saves a photo it already has."""
    d = os.path.join(out_dir, cls)
    os.makedirs(d, exist_ok=True)
    have = len([f for f in os.listdir(d)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if have >= per_class:
        return
    # timebox: once past the deadline, don't START classes we haven't
    # touched yet (in-flight ones still finish). A re-run tops these up —
    # the fetch is idempotent, so nothing is lost.
    if deadline is not None and have == 0 and time.time() > deadline:
        return
    need = per_class - have
    seen = set()                       # sha1 of everything already on disk
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                seen.add(hashlib.sha1(open(p, "rb").read()).hexdigest())
            except OSError:
                pass
    for term in terms:
        try:
            hits = _search(term, max(need * 2, 4))
        except Exception:
            continue
        for url, mime in hits:
            if need <= 0:
                break
            try:
                blob = _get(url, timeout=25)
            except Exception:
                continue
            if len(blob) < 8192:       # not a real photo
                continue
            h = hashlib.sha1(blob).hexdigest()
            if h in seen:              # already have this exact photo
                continue
            fn = "%04d.%s" % (have + 1,
                              "png" if mime == "image/png" else "jpg")
            with open(os.path.join(d, fn), "wb") as f:
                f.write(blob)
            seen.add(h)
            have += 1
            need -= 1
        if need <= 0:
            break
        time.sleep(0.15)              # be polite between search rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="real_photos")
    ap.add_argument("--per_class", type=int, default=4,
                    help="photos per class (default 4; 16x views makes each "
                         "photo ~4-64 training images)")
    ap.add_argument("--classes", default="",
                    help="comma list of class names (default: all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N classes (testing)")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel downloaders (the search API stays "
                         "polite — it is rate-gated at 0.3 s — so extra "
                         "workers only parallelise the photo downloads)")
    ap.add_argument("--max_minutes", type=int, default=0,
                    help="hard timebox: stop STARTING new classes after N "
                         "minutes (in-flight ones finish), so a slow fetch "
                         "never eats the training session; re-run next "
                         "session to finish — it's idempotent. 0 = no limit")
    a = ap.parse_args()

    import make_dataset as md
    import hcaptcha_types as hct

    classes = ([c.strip() for c in a.classes.split(",") if c.strip()]
               if a.classes else list(md.CLASSES))
    if a.limit:
        classes = classes[:a.limit]
    valid = set(md.CLASSES)
    unknown = [c for c in classes if c not in valid]
    if unknown:
        print("warning: skipping %d unknown classes: %s..." %
              (len(unknown), unknown[:5]))
    classes = [c for c in classes if c in valid]

    # search terms per class: the plain name, then up to 2 long synonyms
    # (reverse-lookup in hct.SYNONYMS: alias -> canonical).
    rev = {}
    for alias, can in hct.SYNONYMS.items():
        rev.setdefault(can, []).append(alias)
    terms = {}
    for c in classes:
        t = [c.replace("_", " ")]
        t += sorted((s for s in rev.get(c, [])
                     if " " in s.replace("_", " ") and s != c),
                    key=len, reverse=True)[:2]
        terms[c] = t

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    done = 0
    n_saved = 0
    n_files_before = 0
    for sub in os.listdir(a.out):
        p = os.path.join(a.out, sub)
        if os.path.isdir(p):
            n_files_before += len(os.listdir(p))

    deadline = (t0 + a.max_minutes * 60) if a.max_minutes > 0 else None
    print("fetching %d classes x %d photos -> %s  (%d workers%s)" %
          (len(classes), a.per_class, a.out, a.workers,
           ", timebox %d min" % a.max_minutes if deadline else ""))
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_fetch_class, c, terms[c], a.out, a.per_class,
                          deadline): c
                for c in classes}
        for fut in cf.as_completed(futs):
            c = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - per-class isolation
                print("  %s: FAILED (%s)" % (c, e))
            done += 1
            if done % 50 == 0 or done == len(classes):
                n_now = 0
                for sub in os.listdir(a.out):
                    p = os.path.join(a.out, sub)
                    if os.path.isdir(p):
                        n_now += len(os.listdir(p))
                print("  %d/%d classes done, %d photos on disk (%.0fs)" %
                      (done, len(classes), n_now, time.time() - t0))

    # final summary
    covered, total = 0, 0
    for c in classes:
        d = os.path.join(a.out, c)
        n = (len([f for f in os.listdir(d)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))])
             if os.path.isdir(d) else 0)
        total += n
        covered += int(n > 0)
    dt = time.time() - t0
    print("== done: %d/%d classes have photos, %d total "
         "(+%d new) in %.0fs ==" % (covered, len(classes), total,
                                    total - n_files_before, dt))
    if deadline is not None and time.time() >= deadline - 1 and covered \
            < len(classes):
        print("timebox hit: %d classes still photo-less (they fall back to "
              "renders under --real_only and are named in the train log)." %
              (len(classes) - covered))
        print("re-run this fetch in the next session to top them up — it is "
              "idempotent and skips what's already on disk.")
    print("train with:  --photos_dir %s" % a.out)


if __name__ == "__main__":
    main()

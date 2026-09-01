# Training the 1000-class Giga Brain (Kaggle)

The Brain v2 (`brain.py`) is a single unified network that solves all nine
hCaptcha challenge families offline: tile grids (incl. **colour grids** —
"red car"), point ("jumps the highest"), bbox, count, text, drag (pipe /
tower / shape), pattern completion and the family router.

The vocabulary is **1000 classes**:

| block | classes | source |
|---|---|---|
| core | 60 | `make_dataset.py` + `synth_shapes.py` painters (ids 0-59) |
| long-tail | 454 | `make_longtail.py` recipe painters (animals, food, vehicles, tools, furniture, electronics, clothing, sports, nature, street, household) (ids 60-513) |
| colour compounds | 486 | 54 core objects × 9 colours, recoloured object layer (`red_car`, `blue_dog`, …) (ids 514-999) |

The checkpoint budget is **~1.1 GB (fp32)** — the `giga` preset:
width 160 backbone, 13-layer 1024-d prompt transformer, 512-d class
concepts, 8-layer 640-d pattern reasoner → **282 M parameters ≈ 1.10 GB**.

## 1. Kaggle notebook setup

1. New **Python 3 notebook**, Internet ON (for the hCaptcha data download),
   accelerator **T4 16 GB** (use `--batch 8` on T4). Do NOT pick the P100:
   current PyTorch builds no longer contain CUDA kernels for it (sm_60), so
   it crashes with "no kernel image is available for execution on the
   device" — T4 (sm_75) is fully supported and is what this runbook targets.
2. First cell. Notes: use `%cd` (Jupyter magic, not `cd`), and clone into
   `/kaggle/working` — `/kaggle/input` is read-only. Kaggle already ships a
   current CUDA build of torch, so do NOT pin an old one.

```bash
!pip install -q numpy Pillow
!git clone --depth 1 -b arena/01a051bf-cs https://github.com/alistra742-source/cs /kaggle/working/cs
%cd /kaggle/working/cs
```

3. Real hCaptcha tiles (the "100k challenges" family data). Two public
   datasets, both already aliased into the 1000-class vocabulary
   (motorbus→bus, seaplane→airplane, …). Clone into `/kaggle/working`
   (the input dir is read-only):

```bash
# orlov-ai: 4,068 labelled vehicle tiles (quick, ~5 MB)
!git clone --depth 1 https://github.com/orlov-ai/hcaptcha-dataset /kaggle/working/hcap

# drandule: ~100,000 sorted vehicle tiles (the big one, ~1-2 GB)
# !git clone --depth 1 https://github.com/drandule/hcaptcha_dataset /kaggle/working/hcap
```

4. **Real photos for the non-vehicle classes** (Wikimedia Commons, ~10-20
   min, ~300 MB). One folder per class; each photo becomes 16 augmented +
   degraded training views, exactly like the hCaptcha tiles. The fetch is
   idempotent — re-running it resumes/skips, so a kill + re-run is safe:

```bash
# all 1000 classes x 4 photos (test with --limit 20 first if you like)
!python fetch_photos.py --out /kaggle/working/photos --per_class 4
```

## 2. Train the giga brain

Kaggle kills any notebook run after **12 h (43200 s)** — this is a hard cap,
not a crash. A full giga run (14 + 4 epochs, batch 8 on a T4) is ~2 h/epoch,
so it spans **multiple 12 h sessions**. Every epoch writes a full checkpoint
(weights + optimizer + scheduler position) to `--models_dir`; the next
session picks up where the last one stopped with `--resume`. Point
`--models_dir` at `/kaggle/output/ckpt` so the checkpoint survives the kill
and is downloadable.

**Session 1** (first run):

```bash
!python brain.py train \
  --preset giga \
  --epochs 14 --phase2 4 \
  --batch 8 \
  --per_class 310 \
  --n_point 18000 --n_count 12000 --n_drag 14000 --n_grid 9000 \
  --n_pattern 12000 --n_bbox 10000 --n_pipe 7000 --n_tower 7000 \
  --n_shape 7000 --n_text 6000 \
  --hard_frac 0.45 --degrade_frac 0.5 --clutter_frac 0.55 \
  --hcap_dir /kaggle/working/hcap --hcap_views 16 \
  --photos_dir /kaggle/working/photos --photo_views 16 \
  --amp 1 \
  --models_dir /kaggle/output/ckpt --resume \
  --split_parts --part_max_mb 96
```
It trains ~5 epochs, then Kaggle stops it at 12 h. The `Logs` tab will say
`canceled after ~43200s (timeout exceeded)` — **that is expected, not an
error.** The corpus is rebuilt each session (~1 h, Kaggle storage is
ephemeral), so budget ~1 h of that 12 h on data, not training.

**Continuing (sessions 2, 3, …)** — each time Kaggle stops you:
1. **Output** tab → download `ckpt/resume.pt` (~3.5 GB for giga).
2. Kaggle → **Datasets** → *New Dataset* → upload `resume.pt` (any name; the
   folder it lands in doesn't matter — `--resume` scans all of
   `/kaggle/input`).
3. Add that dataset to this notebook's **Settings → Add Input**.
4. **Re-run the exact same training cell.** `--resume` auto-finds the newest
   `/kaggle/input/**/resume.pt` and prints
   `resume: ... continuing from epoch N/18` — it trains only the remaining
   epochs. Repeat until a run prints the final metrics + split parts.

`--resume` with no file found just starts fresh (a bare `--resume` is safe on
the first run). A checkpoint whose arch doesn't match the current
`--preset`/widths is rejected with a clear log line and it starts fresh
instead of corrupting the run. The final run writes `brain_part_00…11` +
`brain_arch.json` to `/kaggle/output/ckpt` — download those (see §3).

**Want it to finish in a single 12 h session?** Use a smaller-but-strong
config that fits ~9.5 h (≈4 epochs at ~2 h/epoch + ~1.5 h corpus):

```bash
!python brain.py train \
  --preset giga --batch 8 --per_class 310 \
  --epochs 4 --phase2 0 \
  --n_point 18000 --n_count 12000 --n_drag 14000 --n_grid 9000 \
  --n_pattern 12000 --n_bbox 10000 --n_pipe 7000 --n_tower 7000 \
  --n_shape 7000 --n_text 6000 \
  --hard_frac 0.45 --degrade_frac 0.5 --clutter_frac 0.55 \
  --hcap_dir /kaggle/working/hcap --hcap_views 16 \
  --photos_dir /kaggle/working/photos --photo_views 16 \
  --amp 1 --models_dir /kaggle/output/ckpt \
  --split_parts --part_max_mb 96
```
Same 100k+ rounds + 310k tiles + real hcap + real photos, just fewer epochs —
a solid one-shot brain if you don't want to juggle resume across sessions.

(`--batch 8` is the T4 setting; on a 16 GB card that's what fits the 1.1 GB
giga model + optimizer states + the 160-char prompt transformer comfortably.
Skip the `--photos_dir` line if you didn't run the fetch in step 4.)

What this is:

- **~310k+ synthetic tiles** (310 × 1000 classes) across day/dusk/night,
  flips, rotation, blur, JPEG, noise, motion blur, dark-mode tint,
  vignette, plus **~25k–300k real hCaptcha vehicle views** from the dataset
  (16 augmented views per real tile) and **~64k real-photo views** for the
  other classes (4 photos x 16 views x 1000 classes, Wikimedia Commons).
- **100k+ challenge rounds in total**: 9k binary grids + 18k point +
  12k count + 14k drag (pipe/tower/shape mixed) + 9k mixed binary+point
  + 12k pattern + 10k bbox + 7k pipe + 7k tower + 7k shape + 6k text +
  real-data views — every round rendered under the hard-degradation
  recipe (screen-tear motion blur, Gaussian blur, salt&pepper,
  low-contrast dusk/night, colour crush) 45-50% of the time, with scene
  clutter 55% of the time.
- `--phase2 4` fine-tunes the heads on the hardest 20% of rounds
  (harder degradation + more clutter) for 4 extra epochs.
- `--amp 1` (GPU only) halves activation memory; drop it on CPU.
- `--split_parts --part_max_mb 96` writes `brain_part_00…NN` (≤96 MB
  each — 12 parts for the 1.1 GB giga) **and `brain_arch.json`**
  (the exact class list + architecture + metrics) next to them.

GPU sizing: 282 M params × 4 B = 1.13 GB weights, ×3 with Adam states;
on the T4 use `--batch 8` (the 160-char prompt transformer's backward
intermediates are the peak — a 256-prompt router chunk OOMs the 16 GB
card, so the trainer already caps router steps at 64 prompts). The prompt
LRU and the in-memory corpus still fit: ~6-8 GB RAM — add a 32 GB RAM
dataset if you push `per_class` above 350.

Rough wall time: corpus build ~30-60 min on the Kaggle CPU (Pillow-bound),
training ~6-12 h (14 + 4 phases, fp16). If you only have 8 h, use
`--epochs 10 --phase2 3 --per_class 260` — still >260k tiles + 100k rounds.

**Corpus disk cache (free speed-up):** after the first build the rendered
corpus (~9 GB, `corpus_cache/corpus_<hash>.npz` next to `brain.py`) is kept
on the notebook's working disk. Any re-run with the SAME settings (session
restart, OOM retry, parameter tweak that leaves the generation config
unchanged) prints `corpus cache HIT` and skips the whole render step —
training starts in ~2 min instead of ~1 h. Change any generation parameter
(`per_class`, `--n_*`, fracs, seed, hcap dir) and a fresh corpus is built
and cached under a new hash. To force a rebuild, delete `corpus_cache/`.

## 3. Publish the parts (so the Test tab picks them up)

The Test tab (`brain_test.py`) reassembles `models/brain.pt` from:

1. loose `brain_part_NN` files in the repo folder,
2. any git ref that contains them,
3. a GitHub raw download of the committed SHA.

So after training:

```bash
# inside the notebook, download the parts to your machine, then:
cd cs
# brain_part_00 … brain_part_11 + brain_arch.json are in the working dir
git add brain_part_* brain_arch.json
git commit -m "brain: 1000-class giga (1.1GB) split"
git push
```

- Keep the OLD `brain_part_00..06` (60-class, 149 MB) available until you
  are happy with the new one — the loader auto-detects v1 parts
  (single-resolution heads are adapted to the dual-resolution layout;
  at load time the fused output equals the v1 output exactly).
- If you publish to a fresh branch/tag, update `_BRAIN_SHA` at the top of
  `brain_test.py` to that commit SHA (it is the GitHub raw fallback).
- `brain_arch.json` at the repo root is the authoritative sidecar for
  loose/gig parts — the Test tab reads it before falling back to
  `brain.json` in git.

## 4. Verify locally after pulling the new parts

```bash
pip install torch numpy Pillow
python brain.py eval            # loads models/brain.pt (reassembled)
python brain_test.py            # or hit the Test tab in the app:
                                # it reassembles, loads, and solves live
```

`brain.py eval` prints per-family metrics on a held-out round set
(seed 999, disjoint from training): tile accuracy, point hit@10%,
drag both@10%, bbox, count (self-gated exact), text (exact + per-char),
pattern, router. The target for a 99% end-to-end solve rate is
tile ≥ 0.97 AND point ≥ 0.95 AND every gated family ≥ 0.90 — the
confidence gates then only defer the genuinely ambiguous 1-2%.

## 5. CPU quick-check (any machine, no GPU)

```bash
python brain.py smoke           # tiny corpus, 1 epoch, all heads run
python brain.py train --preset small --epochs 2 --phase2 1 \
  --per_class 60 --n_point 400 --n_count 200 --n_drag 400 --n_grid 200 \
  --n_pattern 200 --n_bbox 300 --n_pipe 100 --n_tower 100 --n_shape 100 \
  --n_text 200 --split_parts
```

~10 min on a laptop: proves the 1000-class pipeline end-to-end.

## Preset size table (fp32, 1000-class heads)

| preset | params | checkpoint |
|---|---|---|
| small | 17.9 M | ~70 MB |
| medium | 35.3 M | ~140 MB |
| large | 67.8 M | ~270 MB |
| mega | 115.2 M | ~460 MB |
| **giga** | **282.1 M** | **~1.10 GB** |

## What the 1000-class vocabulary buys

- **Colour grids** ("click each image containing a **red car**"): the 486
  compound classes are first-class labels; `hcaptcha_types` canonicalises
  "red car" → `red_car` (and "navy car" → `blue_car` via colour aliases),
  so the offline semantic path answers them directly.
- **Long-tail nouns** ("tiger", "french fries", "lighthouse"): the router
  bank now contains ~60k (prompt, family) pairs — every class × every
  synonym × article/plural forms — so the language brain has seen the
  wording, and the tile head can emit the class.
- **Relational rounds** ("the largest vehicle"): every longtail class
  carries a size rank in the shared `SIZE_RANK` table, so the offline
  resolver and the generator agree by construction.
- **Set predicates** ("animals", "metal", "edible"): `ANIMALS`/`METAL`/
  `EDIBLE`/`WHEELED`/`MOTORISED`/`PLANTS` all extend automatically with
  the new categories (260 animals, 290 metal items, …).

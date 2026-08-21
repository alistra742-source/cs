# hCaptcha multi-family solver

The solver started as a single path: screenshot `div.task-image` tiles → ask
a vision model which match → click them. That is one of **five** challenge
families hCaptcha serves; the other four were answered with tile indices and
could never pass. This document describes the rewrite: a family router, a
semantic knowledge base, humanized pointer telemetry, and three small CNN
models that solve the whole thing **offline** (no model server needed) with
the vision model as fallback.

## Challenge families

| family | prompt | correct answer | old behaviour |
|---|---|---|---|
| `image_label_binary` | "click each image containing a bus" | tile indices | worked |
| binary + reference | "pick all things you can work on with the item shown" | tile indices | blind — reference image never captured |
| `area_select` point | "click on the animal who jumps the highest" | (x, y) | clicked image centre |
| `area_select` bbox | "draw a box around the cat's head" | rectangle | unhandled |
| `image_drag_drop` | "drag the element to the place where it fits" | press/move/release | unreachable — `DragSolver` matches only Arkose iframes |
| `multiple_choice` | "select the most accurate description" | one option | unhandled |

## Architecture

```
                 hCaptcha /getcaptcha
                       │
         ┌─────────────┴──────────────┐
         │  payload (request_type)    │  tier 1  ─┐
         │  DOM probe (visible nodes) │  tier 2   ├─ hcaptcha_types.classify()
         │  prompt wording (regexes)  │  tier 3  ─┘         │
         └─────────────┬──────────────┘                     ▼
                       │                          family + answer shape
                       ▼
        ┌───────────────────────────────────────────────────┐
        │ server.py round loop                               │
        │  BINARY     → _solve_binary_round  (tiles)         │
        │  AREA_POINT → _solve_point_round   (points)        │
        │  AREA_BBOX  → _solve_point_round   (bbox drag)     │
        │  DRAG_DROP  → _solve_drag_round    (hm.drag)       │
        │  CHOICE     → _solve_choice_round  (hm.click_box)  │
        │  TEXT_ENTRY → _solve_text_round                    │
        └───────────────────────────────────────────────────┘
                       │
        ┌──────────────┴───────────────┐
        │ OFFLINE path (models/*.pt)   │  confidence-gated
        │  TileClassifier / PointLocator│  (SOLVER_CNN_MIN_CONF, 0.62)
        │  DragLocator + knowledge base │
        └──────────────┬───────────────┘
                       │ fallback
        ┌──────────────┴───────────────┐
        │ vision model (vision_solver) │  per-shape system prompts,
        │  Ollama / OpenAI-compatible   │  strict JSON-repair parsing
        └──────────────────────────────┘
                       │
              human_mouse.py on every
              pointer interaction
```

### Routing tiers (`hcaptcha_types.py`)

1. **payload** — `classify_from_payload` reads `request_type` from the
   `/getcaptcha` JSON (captured by the page's response hook); `area_select`
   is split point/bbox by question wording or `request_config`. Helpers:
   `question_text()`, `example_urls()` (the `requester_question_example`
   reference images), `task_urls()` (`tasklist[].datapoint_uri`).
2. **DOM** — `classify_from_dom` consumes `DOM_PROBE_JS` facts (visible
   tiles, example thumbnails, canvases, big images, draggables, a "Move"
   badge, choice buttons, text inputs): draggable/Move + ≤1 tile → drag;
   ≥4 tiles → binary; ≥2 choices + ≤1 tile → choice; bare text field →
   text; 1 tile/canvas → point (bbox when the prompt says "draw a box").
3. **prompt** — `classify_from_prompt` wording regexes, binary before
   point ("click each image containing…" vs "click on the…").

### Knowledge base (`hcaptcha_types.py`)

* `SIZE_RANK` / `JUMP_RANK` / `SPEED_RANK` / `TEMP_RANK` — superlative
  tables powering "click the animal who jumps the highest", "largest",
  "slowest"… `superlative_table(prompt) → (table, max|min)`.
* `TOOL_AFFORDANCE` — drill → wood/wall/table/chair/house, hammer →
  nail/wood/wall, saw → wood/tree, wrench → bolt/bicycle/car/truck,
  paintbrush → wall/canvas/house, screwdriver → screw/wood. This is what
  resolves the reference-image affordance grids.
* `SYNONYMS` + `canonical()` + `extract_target()` — surface nouns →
  canonical classes. `red_light` and `traffic_light` are deliberately kept
  exclusive (opposite labels).
* `EDIBLE` / `WHEELED` / `MOTORISED` / `ANIMALS` / `TOOLS` — set
  predicates for "click each image containing an animal" style prompts.
* `resolve_semantic(prompt, tile_labels, example_label) →` 1-based
  indices, trying superlatives → affordance → same-category-as-example →
  set predicates → plain noun. Returns **`None`** when the prompt is not
  understood (server falls back to the vision model) and **`[]`** for a
  legitimately empty round (they exist — clicking nothing and Verify is the
  right answer).

### Offline models (`train_models.py` + `tile_classifier.py`)

One conv backbone everywhere: 4 blocks of (3×3 conv → BatchNorm → ReLU),
max-pool after blocks 1–3; channel widths `w, 2w, 4w, 8w`.

| model | head | input | width | output |
|---|---|---|---|---|
| TileNet | adaptive-avg-pool → FC | 64 px | 16 | 60-class softmax |
| PointNet | 1×1 conv → 60-channel heatmap | 96 px | 24 | target point |
| DragNet | 1×1 conv → 2-channel heatmap | 96 px | 24 | piece + slot points |

PointNet/DragNet are **heatmap** models: `heatmap(x, onehot)` selects the
target-class channel and the point is decoded with soft-argmax. Training
loss is spatial cross-entropy on the target cell + `4.0 ×` soft-argmax L1.
A flattened FC coordinate head was tried first and plateaued at 0.36 median
error (≈ random, hit@10% 0.07); the heatmap head beat that in one epoch and
converges to ~0.03.

* `PointLocator.scan(image)` — one pass, presence + location for every
  class. Presence is the per-cell softmax **across classes**, so classes
  compete for each cell (a raw per-channel peak can't discriminate).
* `PointLocator.locate_relational(image, prompt, verifier=TileClassifier())`
  — scan, keep classes with presence ≥ 0.30, crop each candidate peak and
  confirm it with the tile classifier, rank the survivors through the
  superlative table. This is what makes "click the animal who jumps the
  highest" work with no vision model.

### Pointer realism (`human_mouse.py`)

hCaptcha grades pointer telemetry, and the old code clicked DOM nodes in a
tight loop. Now: cubic-Bezier paths with a perpendicular bow, cubic
ease-in-out timing, sub-pixel tremor (12–60 samples), overshoot-and-correct
on long glides, a 45–130 ms press dwell, gaussian (never dead-centre)
landing points inside tile boxes, 0.18–0.55 s between tiles, and real
press/travel/micro-adjust/release drags (a synthetic click does nothing on
a drag round).

## Data (real photographs + procedural filler, seeded)

**Real photographs.** `realdata.py` organises image-search downloads into
`data_real/`: 96×96 real photo tiles for **59 of the 60 classes** (~3–5 per
class, split 90/10 by file so an oversampled photo can never leak into
validation), plus 25 real background scenes (streets, meadows, beaches,
desks, asphalt). The one deliberately synthetic class is **`red_light`**:
"red light lit" photo searches return mostly *green/amber* signals, which is
exactly the confusion the `red_light` vs `traffic_light` label split exists
to prevent (the same negative examples do serve `traffic_light`, after
manually dropping any red-lit frames).

`make_dataset.py` + `synth_shapes.py` still draw the 60-class procedural
tiles (deterministic per-class seeds, `"seed|class|index"`, never `hash()`);
the trainer loads painted and real tiles together, repeating the real
photos ~30× with augmentation so a few photos stand up to hundreds of
paintings.

`make_challenges.py` composes full rounds with ground truth — point rounds
(3–5 objects, named or relational prompt), drag rounds (punched slot +
loose piece with a "Move" badge), grid rounds (9 tiles + correct indices,
incl. affordance rounds with a tool reference) — and mixes the domains:
~60% of backgrounds are real photo crops, ~60% of placed objects are real
photo tiles when the class has them, so the heatmap models train on
photographs, not cartoons.

```bash
pip install torch numpy Pillow
# real corpus (workspace image-search downloads -> data_real/):
python realdata.py organize
# painted base + hybrid challenge rounds:
python make_dataset.py --per_class 400 --out data_v2/tiles --size 96
python make_challenges.py --out data_v2/challenges \
    --n_point 7000 --n_drag 4000 --n_grid 1500
# data_v2/, data_real/ and image-search/ are gitignored — regenerable,
# and the photos are third-party stock that must not be committed
```

## Training

```bash
python realdata.py organize            # image-search/ -> data_real/
python realdata.py composites          # object cutouts -> composite tiles
python train_models.py --task tile  --epochs 14 --batch 96 --size 64 --width 24
python train_models.py --task point --epochs 12 --batch 48 --size 96 --width 24
python train_models.py --task drag  --epochs 8  --batch 48 --size 96 --width 24
```

≈55 min on 2 CPU cores. The tile classifier trains on three views of every
class: painted tiles (400/class), the real photos themselves (12 augmented
repeats each), and composite tiles (real object cutouts pasted onto real
scene backgrounds — what teaches background-invariance). Checkpoints →
`models/<task>.pt` + `models/<task>.json` sidecar (class list, input size,
width, held-out metrics). The models directory IS committed (~7.5 MB
total).

## Configuration (env vars)

| var | meaning | default |
|---|---|---|
| `VISION_API_BASE` | vision endpoint (Ollama-compatible) | `http://localhost:11434` |
| `VISION_API_KEY` | Bearer token for the endpoint | — |
| `OLLAMA_MODEL` | vision model name | `qwen3-vl:2b` |
| `SOLVER_CNN_MIN_CONF` | mean per-tile confidence to trust the offline grid path | `0.62` |
| `SOLVER_MODELS_DIR` | where the `.pt`/`.json` live | `./models` |

## Measured results (held-out hybrid rounds — real photos composited onto
real scene backgrounds, disjoint seeds)

| metric | result |
|---|---|
| tile classifier, 60 classes (painted tiles) | 95.2% |
| tile classifier, NEVER-TRAINED real photographs | 31.6% |
| grid rounds solved exactly, end-to-end (hybrid tiles) | 45 / 60 |
| point localiser, named target, within 10% | 72% |
| relational point, right class / click landed | 28% / 28% |
| drag localiser, both ends within 10% | 59 / 60 (98%) |
| offline suite (`python test_solver.py`) | 47/47 |

Read the table carefully: the first column of generations (fully-synthetic
train AND test) produced flattering 95–99% everywhere because test matched
training perfectly. The numbers above are against hybrid real-photo content
— lower, but honest. The drag piece/slot task is geometric, so it holds at
98%; class-identity tasks degrade with a ~330-photo corpus; this is exactly
why the grid path keeps the confidence gate and falls back to the vision
model below `SOLVER_CNN_MIN_CONF`.

## Honest limits

* The real-photo corpus is **small** (3–5 usable photos/class from image
  search) and query-labelled, so it carries label noise; the procedural
  painters still provide most of the sample mass. That's exactly why the
  grid path is gated by `SOLVER_CNN_MIN_CONF` — low confidence falls back
  to the vision model. A larger curated real corpus (e.g. COCO crops) would
  raise the real-photo accuracy further, but it wasn't fetchable in this
  sandbox (only PyPI egress).
* Invisible bbox tolerance: the bbox answer is graded against an invisible
  ground-truth rectangle; being a few pixels off can still fail.
* One bad round loses a multi-round challenge (2–3 rounds per challenge —
  per-round accuracy compounds).
* Proxy/fingerprint quality still dominates the overall pass rate: a
  flagged IP never even sees a solvable challenge.

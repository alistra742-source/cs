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
| `counting` | "how many X are in this image?" | one number | **unhandled** — a photo + numeric options was misread as multiple choice |
| **pattern completion** | "put one of the animals into the empty spot to complete the pattern" | drag candidate → empty cell | **misrouted** — prompt regex missed it and the tile-rich DOM fell through to binary |
| **mixed binary + point** | "click each image containing X, then click on Y" | tiles then (x, y) | **misrouted** — payload tier saw `area_select` and treated the grid stage as a point round |

The mixed round shares the `image_label_area_select` request type: hCaptcha
serves a binary tile-grid stage followed by an area stage under one payload.
The payload tier now **defers** when the payload question is binary-grid
wording, so the live DOM/prompt tiers classify each stage as it renders
(grid → binary, single surface → point), and `_click_challenge_verify`
also clicks the intermediate "Next" arrow button.

Counting routes from the payload (`image_count`/`*count*`), from counting
wording + numeric option buttons in the DOM ("How many…", "count the…",
"number of…"), or from the prompt alone; the answer is graded exactly, so
the offline counter self-gates to the vision model whenever it is unsure
and the round is clicked through `_click_number_option`.

Pattern-completion rounds ("put one of the animals into the empty spot to
complete the pattern") are `image_drag_drop` under the hood, but the
dragged candidate is chosen by the **pattern**, not by geometry: the
prompt tier and the DOM tier (pattern wording + draggables / many tiles)
route them to DRAG_DROP, `is_pattern_prompt()` flags them, and the round
loop dispatches to `_solve_pattern_round` instead of the geometric
`_solve_drag_round`. The solver crops the grid cells and candidates from
the surface screenshot, finds the empty cell as the brightest one
(near-white hole vs painted tiles), labels everything with the tile
classifier, and runs the Latin-square resolver — confidence-gated, with
the vision model as fallback (which answers as a candidate→hole drag and
can also handle multi-candidate variants the offline logic refuses).

The prompt tier also understands the **select-all** wording variants
("select/choose/pick/check/mark all the images/tiles with…") and the
**drag-puzzle** wording variants ("complete the puzzle", "missing piece",
"matching outline", "empty space", "move the piece…").

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
        │  COUNT      → _solve_count_round (number options)  │
        │  PATTERN*   → _solve_pattern_round (Latin square)  │
        │   (* DRAG_DROP flagged by is_pattern_prompt)       │
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
   is split point/bbox by question wording or `request_config`, and **mixed
   binary+point rounds defer to the DOM tier** when the payload question is
   binary-grid wording. Helpers: `question_text()`, `example_urls()` (the
   `requester_question_example` reference images), `task_urls()`
   (`tasklist[].datapoint_uri`).
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
  "slowest", "coldest/hottest place". SIZE now covers **all 60 classes**
  (tools, street furniture and surfaces included — "largest tool" rounds
  resolve offline too); TEMP carries the animal-by-habitat-warmth ranks.
  `superlative_table(prompt) → (table, max|min)`.
* `TOOL_AFFORDANCE` — drill → wood/wall/table/chair/house, hammer →
  nail/wood/wall, saw → wood/tree, wrench → bolt/bicycle/car/truck,
  paintbrush → wall/canvas/house, screwdriver → screw/wood. This is what
  resolves the reference-image affordance grids.
* `SYNONYMS` + `canonical()` + `extract_target()` — surface nouns →
  canonical classes, with a **long-tail alias table** (~255 entries) that
  maps the hundreds of real-world prompt nouns onto the 60 classes the
  offline models can emit *where the visuals are defensible at tile
  scale*: helicopter/seaplane → airplane, police car/taxi → car, fire
  truck/semi → truck, subway/tram → train, sailboat/ferry → boat,
  owl/parrot/penguin/chicken → bird, shark/dolphin/whale → fish,
  deer/donkey/camel → horse, goat/alpaca → sheep, bison → cow,
  panda/koala → bear, palm tree/pine tree/forest → tree, rose/sunflower →
  flower, volcano/cliff → mountain, barn/apartment → house, traffic
  signal → traffic_light, pedestrian crossing → crosswalk, bench/sofa →
  chair, desk → table, pie → pizza, watch/alarm clock → clock, nut →
  bolt, … Plurals resolve too ("pandas" → panda). Everything deliberately
  unmapped (tiger, monkey, skyscraper, waterfall, smartphone, ladder, …)
  stays `None` so the server falls back to the vision model — which reads
  arbitrary prompt text — instead of trusting a wrong offline label.
  `red_light` and `traffic_light` are deliberately kept exclusive
  (opposite labels).
* `EDIBLE` / `WHEELED` / `MOTORISED` / `ANIMALS` / `TOOLS` — set
  predicates for "click each image containing an animal" style prompts.
* `resolve_semantic(prompt, tile_labels, example_label) →` 1-based
  indices, trying superlatives → affordance → same-category-as-example →
  set predicates → plain noun. Returns **`None`** when the prompt is not
  understood (server falls back to the vision model) and **`[]`** for a
  legitimately empty round (they exist — clicking nothing and Verify is the
  right answer).
* `resolve_pattern(grid_labels, hole_index, candidates) →` candidate
  index completing every row AND column of a 3×3 grid with distinct
  labels (Latin square; rows-only rule tried second). Returns **`None`**
  on any ambiguity — a wrong candidate fails the round outright, so the
  resolver never guesses and the server falls back to the vision model.

### Offline models (`train_models.py` + `tile_classifier.py`)

One conv backbone everywhere: 4 blocks of (3×3 conv → BatchNorm → ReLU),
max-pool after blocks 1–3; channel widths `w, 2w, 4w, 8w`.

| model | head | input | width | output |
|---|---|---|---|---|
| TileNet | adaptive-avg-pool → FC | 64 px | 24 | 60-class softmax |
| PointNet | 1×1 conv → 60-channel heatmap | 96 px | 24 | target point |
| DragNet | 1×1 conv → 2-channel heatmap | 96 px | 24 | piece + slot points |

PointNet/DragNet are **heatmap** models: `heatmap(x, onehot)` selects the
target-class channel and the point is decoded with soft-argmax. Training
loss is spatial cross-entropy on the target cell (with the per-cell
background competition for PointNet) + `4.0 ×` soft-argmax L1. A flattened
FC coordinate head was tried first and plateaued at 0.36 median error
(≈ random, hit@10% 0.07); the heatmap head beat that in one epoch and
converges to ~0.03. **Gaussian-softened spatial targets were A/B-tested
later and regressed hard** (loss plateaus at centre-prediction) — the hard
single-cell signal is what trains the peak; the experiment is documented
in the training log.

* `PointLocator.scan(image)` — one pass, presence + location for every
  class. Presence is the per-cell softmax **across classes**, so classes
  compete for each cell (a raw per-channel peak can't discriminate).
* `PointLocator.locate_relational(image, prompt, verifier=TileClassifier())`
  — scan, keep classes with presence ≥ 0.30, NMS the location clusters,
  crop each candidate peak and confirm it with the tile classifier, rank
  the survivors through the superlative table. This is what makes "click
  the animal who jumps the highest" work with no vision model.
* `PointLocator.count(image, target)` — counting rounds: the target
  class's presence map is peak-found (local maxima above `min_peak=0.08`,
  NMS-clustered) and the cluster count is the answer. The point model is
  trained on multi-instance count rounds (k instances of one class per
  scene, one supervised cell each) so every instance lights its own peak.
  A count answer is graded EXACTLY, so the counter self-gates hard —
  border-touching peaks, over-fragmented maps, or a weakest kept peak
  below `weak_gate=0.20` return `None` and the server falls back to the
  vision model (measured: ~72% answered exactly offline, ~22% gated).

### Pointer realism (`human_mouse.py`)

hCaptcha grades pointer telemetry, and the old code clicked DOM nodes in a
tight loop. Now: cubic-Bezier paths with a perpendicular bow, cubic
ease-in-out timing, sub-pixel tremor (12–60 samples), overshoot-and-correct
on long glides, a 45–130 ms press dwell, gaussian (never dead-centre)
landing points inside tile boxes, 0.18–0.55 s between tiles, and real
press/travel/micro-adjust/release drags (a synthetic click does nothing on
a drag round).

## Data

**Real photographs.** `image-search/` (gitignored) holds the image-search
downloads; `realdata.py organize` curates them into `data_real/`: 96×96
photo tiles for **59 of the 60 classes** (~2–6 per class, split by file
into train + a 2/class `val/` holdout), plus ~29 real background scenes
(streets, meadows, beaches, desks, asphalt). The one deliberately synthetic
class is **`red_light`**: "red light lit" photo searches return mostly
*green/amber* signals, which is exactly the confusion the `red_light` vs
`traffic_light` label split exists to prevent. `realdata.py composites`
pastes object cutouts (uniform-background photos) onto real scene crops
into the painted corpus, teaching background-invariance.

**Procedural filler.** `make_dataset.py` + `synth_shapes.py` draw the
60-class painted tiles (deterministic per-class seeds, `"seed|class|index"`,
never `hash()`). The trainer loads painted and real tiles together,
repeating the real photos ~30× with distinct random views (resized crop +
flip + small rotation) so a few photos stand up to hundreds of paintings;
a real-weighted `--resume` fine-tune pass lifts the photo transfer further.

`make_challenges.py` composes full rounds with ground truth — point rounds
(3–5 objects, named or relational prompt, now including TEMP
coldest/warmest rounds restricted to animals), drag rounds (punched slot +
loose piece with a "Move" badge), grid rounds (9 tiles + correct indices,
incl. affordance rounds with a tool reference), **count rounds** (2–5
separated instances of ONE class, prompt "How many X are in this
image?", ground-truth count), and **pattern rounds** (3×3 Latin-square
animal grid with one empty cell + 3 candidates, landscape canvas with
~40px cells — the size the tile classifier can label at ~94%) — and
mixes the domains: ~60% of backgrounds are real photo crops, ~60% of
placed objects are real photo tiles when the class has them, so the
heatmap models train on photographs, not cartoons.

```bash
pip install torch numpy Pillow
# real corpus (workspace image-search downloads -> data_real/):
python realdata.py organize --holdout 2
python realdata.py composites
# painted base + hybrid challenge rounds:
python make_dataset.py --per_class 600 --out data_v2/tiles --size 96
python make_challenges.py --out data_v2/challenges \
    --n_point 9000 --n_drag 6000 --n_grid 2000 --n_count 4000 \
    --n_pattern 300
# data_v2/, data_real/ and image-search/ are gitignored — regenerable,
# and the photos are third-party stock that must not be committed
```

## Training

```bash
python realdata.py organize            # image-search/ -> data_real/
python realdata.py composites          # object cutouts -> composite tiles
python train_models.py --task tile  --epochs 14 --batch 128 --size 64 --width 24
python train_models.py --task point --epochs 12 --batch 48 --size 96 --width 24 \
    --data data_v2/challenges/manifest.jsonl
python train_models.py --task drag  --epochs 10 --batch 48 --size 96 --width 24 \
    --data data_v2/challenges/manifest.jsonl
# real-photo transfer pass (starts from models/tile.pt):
python train_models.py --task tile --epochs 3 --lr 3e-4 --resume models/tile.pt \
    --real_repeat 80
```

≈1.5–2 h on 2 CPU cores. The tile classifier trains on three views of
every class: painted tiles (600/class), the real photos themselves
(augmented repeats), and composite tiles (real object cutouts pasted onto
real scene backgrounds). The coordinate tasks train with
**coordinate-mapped geometric augmentation** (`_prep_geom`): every batch is
rotated ±15°, scaled 0.78–1.28, translated ±10% and flipped, and the click
targets are carried through the SAME affine — one round now teaches a
continuum of poses instead of one, which is the single biggest lever for
the point localiser (held-out named-point hit went 58 → 70/100 on the same
corpus version; verified sub-pixel label fidelity by dot-tracking).
`train_point` pulls the **count rounds from the same manifest** and
supervises them with a per-instance cell mask (all instance points ride
the same geometric affine), so the one point model localises single
targets AND counts multi-instance scenes; the L1 term applies only to
single-instance rows. Checkpoints → `models/<task>.pt` +
`models/<task>.json` sidecar (class list, input size, width, held-out
metrics). The models directory IS committed (~2.7 MB total).

## Configuration (env vars)

| var | meaning | default |
|---|---|---|
| `VISION_API_BASE` | vision endpoint (Ollama-compatible) | `http://localhost:11434` |
| `VISION_API_KEY` | Bearer token for the endpoint | — |
| `OLLAMA_MODEL` | vision model name | `qwen3-vl:2b` |
| `SOLVER_CNN_MIN_CONF` | mean per-tile confidence to trust the offline grid path | `0.62` |
| `SOLVER_MODELS_DIR` | where the `.pt`/`.json` live | `./models` |

## Measured results (`python test_solver.py`, held-out rounds, disjoint
seeds — the figures below are painted-only content: the real-photo corpus
is gitignored and regenerable, and when `data_real/` is absent the
generators and the suite fall back to their procedural output, which is
easier than hybrid real-photo content. The real-photo rows re-appear once
`python realdata.py organize` has run.)

| metric | result | previous |
|---|---|---|
| tile classifier, 60 classes (painted tiles) | **98.1%** | 77.7%¹ |
| tile classifier, NEVER-TRAINED real photographs | *(requires data_real/)* | 51.3% (59/115) |
| grid rounds solved exactly, end-to-end (hybrid tiles) | **56 / 60** painted-only | 51 / 60 hybrid |
| point localiser, named target, within 10% | **92%** painted-only | 70% hybrid |
| relational point, right class / click landed | **64% / 65%** painted-only | 50% / 58% hybrid |
| **counting, exact count** | **43 / 60 offline + 13 self-gated to vision** | unhandled |
| **pattern completion, correct candidate** | **40 / 60 offline + 20 self-gated to vision (0 wrong)** | unhandled |
| drag localiser, both ends within 10% | **60 / 60 (100%)** | 59 / 60 |
| offline suite (`python test_solver.py`) | **65 (64 + 1 corpus-skip)** | 48/48 |

¹ previous column = the same held-out harness run against the previously
committed weights in this checkout (the SOLVER.md numbers before this
revision were stale in two ways: the weights predated the 49→60 class
expansion, so 11 classes were automatic misses, and the generator crashed
on current Pillow).

Read the table carefully: the fully-synthetic train AND test generations
produce flattering 95–99% everywhere because test matches training
perfectly — the painted-only rows this run are exactly that regime, so
they overstate live hybrid performance; the hybrid rows (previous column)
are the honest reference once the real corpus is regenerated. Counting is
a new capability: ~72% of held-out count rounds answered exactly offline,
~22% self-gated to the vision model, a small tail still wrong (a count
answer is graded exactly, which is why the gates are strict).
The drag piece/slot task is geometric, so it holds at 100%; class-identity
tasks degrade with the photo corpus size; this is exactly why the grid
path keeps the confidence gate and falls back to the vision model below
`SOLVER_CNN_MIN_CONF`.

## Honest limits

* The real-photo corpus is still small (~240 train photos for 59 classes)
  and query-labelled, so it carries label noise; the procedural painters
  still provide most of the sample mass. The real-photo transfer number
  (51.3%) doubled when the corpus doubled — it scales with corpus size.
* Invisible bbox tolerance: the bbox answer is graded against an invisible
  ground-truth rectangle; being a few pixels off can still fail.
* One bad round loses a multi-round challenge (2–3 rounds per challenge —
  per-round accuracy compounds).
* Mixed binary+point rounds are routed per stage, but both stages must
  pass in one challenge.
* Counting answers are graded exactly: a count that is off by one fails
  the round outright, so the offline counter gates aggressively to the
  vision model; on cluttered real photos (which the painted-only counter
  has not seen in training) most counts will be gated.
* Pattern rounds are graded exactly too: the Latin-square resolver
  refuses ambiguity (never guesses), the offline path needs the tile
  classifier to label ~40px+ cells, and the DOM probe (lattice clustering
  of candidate vs grid elements) is conservative — any doubt and the
  vision model answers the drag instead. Icon styles outside the 60
  painted classes are vision territory.
* The alias table is deliberately conservative: nouns that are NOT
  visually defensible at tile scale are left unmapped so the vision model
  (which reads arbitrary prompt text) answers them. The offline models
  still only emit the 60 trained classes — a "helicopter" tile only
  resolves offline when the classifier genuinely sees it as an airplane.
* Proxy/fingerprint quality still dominates the overall pass rate: a
  flagged IP never even sees a solvable challenge.

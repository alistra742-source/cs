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
| **wooden-block tower** | "move the correct missing block segment onto the incomplete tower" | drag piece → short/gapped stack | **misrouted** — payload `image_label_area_select` committed to a point click; the Move badge (`+ Move`) was missed; DragLocator punched-slot geometry is the wrong puzzle |
| **set-down / spatial-ref** | "find places safe for setting down the item in the reference" | tile indices (surfaces) | **misread** — treated as a tool-affordance or point click; clicked balloons/leaves instead of nightstand/bench/deck |

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

Wooden-block tower rounds ("Move the correct missing block segment onto
the incomplete tower") are also served under `image_label_area_select`
even though the answer is a Move-badge drag: three wood stacks plus a
1–2 block piece on the right. The live prompt forces `DRAG_DROP` (the
payload tier only commits when the question is *only* the tower, so a
mixed "select items, then move the block…" payload still defers). The
DOM "Move" probe accepts `+ Move` / short labels with an icon child.
The round loop dispatches to `_solve_tower_round` — **not**
`DragLocator`. It screenshots the **whole challenge iframe** (the Move
piece is often a separate DOM node beside the photo), takes a Move-badge
hint from the DOM, then finds warm/wood columns and drops onto the
shortest stack or the largest internal gap. Vision (`shape="tower"`) is
a 18s last resort only — a long 504 expires the challenge.

Set-down / spatial-reference grids ("Find places safe for setting down
the item in the reference") are a **binary tile grid with a header
photo** (a mug, a tool, …). The live prompt forces `BINARY` (it is not
a point click). Offline, `is_setdown_prompt()` + `FLAT_SURFACES`
(`table` / `chair` / `wood`, with nightstand/dresser/desk → table,
bench/sofa → chair, deck → wood) clicks every furniture/lumber tile
and skips balloons, balls, leaves, and building facades. No matching
surface returns `None` so the vision model answers — an empty Verify
almost never wins this family. The wording is tight on purpose:
`"place where it fits"` is a drag puzzle and bare `"in the reference"`
matches every affordance grid. The default vision model is
**SmolVLM2-256M** (`ahmadwaqar/smolvlm2-256m-video:q8_0`): it cannot
do a 9-image JSON contract (that 504s in ~180s), so the client asks
**one tile at a time** ("is this a table/nightstand/bench/deck?") at
256 px / 12s / no `format: json`. Offline surfaces stay the first path
— a 256M yes/no on a tennis-ball-on-deck photo is unreliable.

This is how the long tail of the ~1000-prompt catalog is covered:
**routing + aliases + the 60-class CNN + vision**, not a 1000-class
retrain. The offline tile model still only emits the 60 trained
classes; unmapped nouns (tennis ball, hot-air balloon, plastic, glass,
3-D views, odd-one-out, actions) fall through to the vision model.

The prompt tier also understands the **select-all** wording variants
("select/choose/pick/check/mark all the images/tiles with…"), the
**attribute/material** wording variants ("select items that are primarily
metal", "made of wood", "have fur" — resolved offline against METAL /
WOODEN / FURRY class sets, unknown materials like plastic/glass fall
through to the vision model), and the **drag-puzzle** wording variants
("complete the puzzle", "missing piece", "matching outline", "empty
space", "move the piece…"). After every answered round the solver clicks
the enabled **Next** or **Verify** control and waits for the next
challenge to paint (it does not treat the brief iframe-loader dip as
"challenge over", and it will not re-click the same tiles — that would
toggle them off).

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
        │  TOWER*     → _solve_tower_round (wood-mask drag)  │
        │   (* DRAG_DROP flagged by is_pattern_prompt /      │
        │      is_tower_prompt; tower never uses DragLocator)│
        └───────────────────────────────────────────────────┘
                       │
        ┌──────────────┴───────────────┐
        │ OFFLINE path (models/*.pt)   │  confidence-gated
        │  TileClassifier / PointLocator│  (SOLVER_CNN_MIN_CONF, 0.62)
        │  DragLocator + knowledge base │
        └──────────────┬───────────────┘
                       │ fallback
        ┌──────────────┴───────────────┐
        │ vision model (vision_solver) │  SmolVLM2: per-tile yes/no;
        │  Ollama / OpenAI-compatible   │  larger VLMs: JSON + repair
        └──────────────────────────────┘
                       │
              human_mouse.py on every
              pointer interaction
```

### Routing tiers (`hcaptcha_types.py`)

1. **payload** — `classify_from_payload` reads `request_type` from the
   `/getcaptcha` JSON (captured by the page's response hook); `area_select`
   is split point/bbox by question wording or `request_config`; **tower /
   pattern-only** area_select payloads route to drag; and **mixed
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
* `METAL` / `WOODEN` / `FURRY` / `PLANTS` — dominant-material sets for
  "select items that are primarily metal / made of wood / have fur".
  `is_attribute_prompt()` / `attribute_members()` gate them; unknown
  materials return `None` so the vision model (which reads the object
  itself, not the background) answers.
* `resolve_semantic(prompt, tile_labels, example_label) →` 1-based
  indices, trying superlatives → set-down surfaces → comparative vs the
  reference ("larger than the item shown") → affordance →
  same-category-as-example → material/attribute sets → set predicates →
  plain noun. Returns **`None`** when the prompt is not understood
  (server falls back to the vision model) and **`[]`** for a
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

### Arkose block-stacking rounds (`drag_solver.py` STACK)

Arkose/FunCaptcha also serves a **block-stacking** game: vertical columns
of blocks (e.g. one 3 tall, one 2, one 1) plus loose draggable blocks, and
the answer is every drag that levels the columns ("3, 3, 3"). `DragSolver`
now has a fourth challenge family besides slider/tiles/match:

* **routing** — `_STACK_KEYWORDS` wording ("same height", "stack",
  "tower", "balance", …) is checked *before* the slider keywords because
  stack instructions also contain "drag"; the vision classifier fallback
  offers "stack" as a fourth one-word answer, so cross-origin frames
  (where `innerText` is unreadable) still route correctly.
* **answering** — `_solve_stack` screenshots the iframe and asks the
  vision model with the new `shape="stack"` answer contract
  (`vision_solver._SYSTEM_STACK` + `_parse_stack_geometry`): a
  `{"drags": [[sx, sy, tx, ty], ...]}` plan in 0–100 **percent** iframe
  coordinates. The parser accepts every transport a small model emits
  (per-drag from/to dicts, bare 4-lists, "moves" aliases, fenced JSON,
  0–1 fractions rescaled ×100, mixed unit repair), rejects hallucinated
  pixel coordinates (any value beyond ±3 of the 0–100 range drops that
  drag), and caps plans at 12 drags. `DragSolver._parse_stack_plan`
  re-parses the degraded transports (single `drag`, flat `tiles`
  indices in chunks of four, `points` grab/drop pairs) for facades
  without the `shape` kwarg.
* **execution** — each plan drag is converted percent → absolute page
  coordinates via the iframe box (with ±1.5 % human jitter inside the
  grabbed block) and replayed through the humanised drag path; a failed
  round just returns to `solve()`'s retry loop, which re-screenshots and
  re-asks. Samples save to the training collector as type `stack`.

Verified offline: 26 self-test cases (`python drag_solver.py`) cover the
plan parser (all transports, clamps, rejections) and the stack/slider/
tiles wording router; the vision-side geometry parser has its own case
table and the other shapes (drag/points/tiles) parse identically before
and after.

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
| `OLLAMA_MODEL` | vision model name | `ahmadwaqar/smolvlm2-256m-video:q8_0` |
| `OLLAMA_TIMEOUT` | per-solve timeout (seconds) | `30` |
| `OLLAMA_TILE_TIMEOUT` | per-tile yes/no timeout for tiny VLMs | `12` |
| `OLLAMA_IMAGE_SIDE` | max image side (px) sent to tiny VLMs | `256` |
| `SOLVER_CNN_MIN_CONF` | mean per-tile confidence to trust the offline grid path | `0.62` |
| `SOLVER_MODELS_DIR` | where the `.pt`/`.json` live | `./models` |
| `FULLPAGE_SHOTS` | whole scrollable page camera frames (default: full browser-view frames with the register form revealed when out of sight) | `0` |
| `FULLPAGE_MAX_PX` | max page height (px) worth a full-page frame; taller pages fall back to viewport frames | `8000` |

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
* Tower rounds are a geometric heuristic on wood-coloured pixels: unusual
  palettes (grey stone, neon plastic), four-plus equal-height stacks, or
  a piece that is not in the right strip fall through to the vision
  model (`shape="tower"`). The locator never guesses when every stack is
  the same height and there is no gap.
* Set-down grids need the CNN to label nightstands as `table`, benches
  as `chair`, and wooden decks as `wood`. A tennis-ball-on-deck photo
  whose subject the 60-class model does not know is vision territory.
  Do **not** retrain a 1000-class tile CNN for the prompt catalog — there
  is no corpus, and the existing 60-class pass already takes 1.5–2 h.
* The alias table is deliberately conservative: nouns that are NOT
  visually defensible at tile scale are left unmapped so the vision model
  (which reads arbitrary prompt text) answers them. The offline models
  still only emit the 60 trained classes — a "helicopter" tile only
  resolves offline when the classifier genuinely sees it as an airplane.
* Proxy/fingerprint quality still dominates the overall pass rate: a
  flagged IP never even sees a solvable challenge.

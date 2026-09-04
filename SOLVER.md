# hCaptcha multi-family solver

The solver started as a single path: screenshot `div.task-image` tiles → ask
a vision model which match → click them. That is one of **five** challenge
families hCaptcha serves; the other four were answered with tile indices and
could never pass. This document describes the rewrite: a family router, a
semantic knowledge base, humanized pointer telemetry, and a **Roboflow
Workflow running Google Gemini 3.6 Flash** that answers every family.

All visual reasoning is remote: there is no local checkpoint, no CNN
weights and no self-hosted model server in this repo. Every call sends
the tile/canvas image **and the challenge question itself** to the
workflow. Configure `API_KEY` with your Roboflow key.

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
| **wooden-block tower** | "move the correct missing block segment onto the incomplete tower" | drag piece → short/gapped stack | **misrouted** — payload `image_label_area_select` committed to a point click and the Move badge (`+ Move`) was missed |
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
the workflow counts the detections and the round is clicked through
`_click_number_option`.

Pattern-completion rounds ("put one of the animals into the empty spot to
complete the pattern") are `image_drag_drop` under the hood, but the
dragged candidate is chosen by the **pattern**, not by geometry: the
prompt tier and the DOM tier (pattern wording + draggables / many tiles)
route them to DRAG_DROP, `is_pattern_prompt()` flags them, and the round
loop dispatches to `_solve_pattern_round` instead of the geometric
`_solve_drag_round`, which asks the workflow for the candidate and the
empty cell and replays the answer as a candidate→hole drag.

Wooden-block tower rounds ("Move the correct missing block segment onto
the incomplete tower") are also served under `image_label_area_select`
even though the answer is a Move-badge drag: three wood stacks plus a
1–2 block piece on the right. The live prompt forces `DRAG_DROP` (the
payload tier only commits when the question is *only* the tower, so a
mixed "select items, then move the block…" payload still defers). The
DOM "Move" probe accepts `+ Move` / short labels with an icon child.
The round loop dispatches to `_solve_tower_round`. It screenshots the **whole challenge iframe** (the Move
piece is often a separate DOM node beside the photo), takes a Move-badge
hint from the DOM, then finds warm/wood columns and drops onto the
shortest stack or the largest internal gap. Vision (`shape="tower"`) is
a 18s last resort only — a long stall expires the challenge.

Set-down / spatial-reference grids ("Find places safe for setting down
the item in the reference") are a **binary tile grid with a header
photo** (a mug, a tool, …). The live prompt forces `BINARY` (it is not
a point click), and `is_setdown_prompt()` rewrites the per-tile question
into something concrete to detect ("does this photo show a table,
nightstand, bench, wooden deck, counter or shelf a mug could sit on?" —
explicitly not a balloon, ball, leaf or sky). The wording is tight on
purpose: `"place where it fits"` is a drag puzzle and bare `"in the
reference"` matches every affordance grid.

This is how the long tail of the ~1000-prompt catalog is covered:
**routing + aliases + a general-purpose VLM**, not a class-limited
classifier. Any noun the prompt can name (tennis ball, hot-air balloon,
plastic, glass, 3-D views, odd-one-out, actions) is the model's job.

The prompt tier also understands the **select-all** wording variants
("select/choose/pick/check/mark all the images/tiles with…"), the
**attribute/material** wording variants ("select items that are primarily
metal", "made of wood", "have fur" — resolved offline against METAL /
WOODEN / FURRY class sets), and the **drag-puzzle** wording variants
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
        │      is_tower_prompt)                              │
        └───────────────────────────────────────────────────┘
                       │
        ┌──────────────┴────────────────────────┐
        │ Roboflow Workflow (vision_solver.py)   │
        │  serverless.roboflow.com/infer/        │
        │    workflows/<ws>/<workflow>           │
        │  Gemini 3.6 Flash object detection     │
        │  body: api_key + image(base64)         │
        │        + the challenge question        │
        │  out:  predictions[] (px) -> 0-1       │
        └───────────────────────────────────────┘
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

The knowledge base now serves **routing and prompt understanding** (which
family a round is, what the target noun is, which wording variant it is),
not offline answering: the answer itself always comes from the workflow.
The helpers below are retained and unit-tested because the routers, the
`classes` list and the set-down question rewriter build on this vocabulary.

* `SIZE_RANK` / `JUMP_RANK` / `SPEED_RANK` / `TEMP_RANK` — superlative
  tables powering "click the animal who jumps the highest", "largest",
  "slowest", "coldest/hottest place". SIZE now covers **all 60 classes**
  (tools, street furniture and surfaces included — "largest tool" rounds
  handled here too); TEMP carries the animal-by-habitat-warmth ranks.
  `superlative_table(prompt) → (table, max|min)`.
* `TOOL_AFFORDANCE` — drill → wood/wall/table/chair/house, hammer →
  nail/wood/wall, saw → wood/tree, wrench → bolt/bicycle/car/truck,
  paintbrush → wall/canvas/house, screwdriver → screw/wood. This is what
  resolves the reference-image affordance grids.
* `SYNONYMS` + `canonical()` + `extract_target()` — surface nouns →
  canonical classes, with a **long-tail alias table** (~255 entries) that
  maps the hundreds of real-world prompt nouns onto canonical classes
  *where the visuals are defensible at tile scale*: helicopter/seaplane → airplane, police car/taxi → car, fire
  truck/semi → truck, subway/tram → train, sailboat/ferry → boat,
  owl/parrot/penguin/chicken → bird, shark/dolphin/whale → fish,
  deer/donkey/camel → horse, goat/alpaca → sheep, bison → cow,
  panda/koala → bear, palm tree/pine tree/forest → tree, rose/sunflower →
  flower, volcano/cliff → mountain, barn/apartment → house, traffic
  signal → traffic_light, pedestrian crossing → crosswalk, bench/sofa →
  chair, desk → table, pie → pizza, watch/alarm clock → clock, nut →
  bolt, … Plurals resolve too ("pandas" → panda). Everything deliberately
  unmapped (tiger, monkey, skyscraper, waterfall, smartphone, ladder, …)
  stays `None` — the raw question is sent to the model anyway.
  `red_light` and `traffic_light` are deliberately kept exclusive
  (opposite labels).
* `EDIBLE` / `WHEELED` / `MOTORISED` / `ANIMALS` / `TOOLS` — set
  predicates for "click each image containing an animal" style prompts.
* `METAL` / `WOODEN` / `FURRY` / `PLANTS` — dominant-material sets for
  "select items that are primarily metal / made of wood / have fur".
  `is_attribute_prompt()` / `attribute_members()` gate them; unknown
  materials return `None`.
* `resolve_semantic(prompt, tile_labels, example_label) →` 1-based
  indices, trying superlatives → set-down surfaces → comparative vs the
  reference ("larger than the item shown") → affordance →
  same-category-as-example → material/attribute sets → set predicates →
  plain noun. Returns **`None`** when the prompt is not understood
  and **`[]`** for a
  legitimately empty round (they exist — clicking nothing and Verify is the
  right answer).
* `resolve_pattern(grid_labels, hole_index, candidates) →` candidate
  index completing every row AND column of a 3×3 grid with distinct
  labels (Latin square; rows-only rule tried second). Returns **`None`**
  on any ambiguity — a wrong candidate fails the round outright, so the
  resolver never guesses.

### Vision client (`vision_solver.py`)

`RoboflowVisionClient` posts one image per request to a Roboflow Workflow:

```
POST https://serverless.roboflow.com/infer/workflows/<workspace>/<workflow>
{
  "api_key": "<API_KEY>",
  "inputs": {
    "image":   {"type": "base64", "value": "<jpeg>"},
    "prompt":  "Please click each image containing a boat",
    "classes": ["boat", "Please click each image containing a boat"]
  }
}
```

The image **and the question** go up together on every call. `prompt` is
mirrored to `query`/`question` because Gemini workflow templates name that
input differently and a workflow ignores inputs it does not declare;
`classes` carries the canonical noun (via `hcaptcha_types.extract_target`)
plus the raw question so an object-detection workflow has something
concrete to find.

The reply is read shape-agnostically: `read_response()` walks the whole
JSON body collecting every `predictions[]` list and every string leaf, so
detection blocks and VLM/caption blocks both work.
`predictions_to_points()` converts Roboflow's pixel centre boxes to
normalised 0-1 `(x, y, w, h, conf, label)` and drops anything below
`ROBOFLOW_MIN_CONF`. `detections_to_answer()` then maps them per shape:

| shape | mapping |
|---|---|
| `tiles` | one request **per tile**; a tile with ≥1 detection (or a "yes" text answer) is selected |
| `points` | detection centres, best confidence first |
| `bbox` | highest-confidence box, centre+size → x1/y1/x2/y2 |
| `drag` / `pattern` / `tower` | best detection = the piece, second = the destination |
| `count` | number of detections above threshold |

When the workflow returns free text instead (a VLM step), the existing
JSON repair layer (`_parse_geometry` / `_parse_answer`) handles it —
markdown fences, bare-dot decimals, percent-vs-fraction units, loose prose
tile numbers.

`check()` POSTs `describe_interface`: 200 = ready, 401/403 = bad
`API_KEY`, 404 = wrong workspace/workflow, 429 = rate limited. The server
retries only transient classes. `app.py` re-probes every 10 minutes so a
scaled-to-zero workflow stays warm and misconfiguration shows up in the
log rather than mid-solve.

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
  Roboflow workflow with the new `shape="stack"` answer contract
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

## Configuration (env vars)

| var | meaning | default |
|---|---|---|
| `API_KEY` | **Roboflow API key — required** (sent as `api_key` in the body) | — |
| `ROBOFLOW_WORKSPACE` | workspace slug | `alistra742-gmail-com` |
| `ROBOFLOW_WORKFLOW` | pin ONE workflow id (skips the search) | — |
| `ROBOFLOW_WORKFLOWS` | comma-separated ids to try in order | `gemini-3-6-flash-object-detection,coco-50-object-counter-1788536417919,gemini-3-6-flash,coco-50-object-counter` |
| `ROBOFLOW_API_BASE` | serverless base URL | `https://serverless.roboflow.com` |
| `GOOGLE_API_KEY` | optional BYO Google AI Studio key (sent as `model_api_key`; omitted when unset, so inference bills Roboflow credits) | — |
| `ROBOFLOW_TIMEOUT` | per-solve timeout (seconds) | `60` |
| `ROBOFLOW_TILE_TIMEOUT` | per-tile timeout for grid rounds | `25` |
| `ROBOFLOW_IMAGE_SIDE` | max image side (px) sent up | `640` |
| `ROBOFLOW_MIN_CONF` | minimum detection confidence kept | `0.30` |
| `ROBOFLOW_CHECK_TIMEOUT` | readiness-probe timeout (seconds) | `60` |
| `RTDETR_ENABLED` | RT-DETR backup detector on/off | `1` |
| `RTDETR_MODEL_ID` | backup model alias | `rfdetr-small` |
| `RTDETR_TIMEOUT` | per-image backup timeout (seconds) | `30` |
| `RTDETR_MIN_CONF` | minimum backup detection confidence | `0.35` |

## NoneCap (hosted solver) — tried FIRST

NoneCap returns a real hCaptcha token (`P1_…`, the kind that passes
siteverify) instead of image coordinates, so there is nothing to click,
drag or shape-match. It runs before the local vision pipeline; anything it
cannot do falls through to the tiers below.

```
NONECAP_API = nc_live_...
```

That is the only required variable. Optional:

| variable | meaning | default |
|---|---|---|
| `NONECAP_TRIES` | solve attempts per challenge | `3` |
| `NONECAP_WAIT` | seconds the API holds the connection (max 90) | `90` |
| `NONECAP_TIMEOUT` | overall ceiling per attempt, incl. polling | `180` |
| `NONECAP_ENABLED` | set 0 to disable and use vision only | `1` |
| `NONECAP_BASE` | API base URL | `https://api.nonecap.com` |

Flow: `POST /v1/solves?wait=90` with `{type, sitekey, url}`. A token that
lands inside the wait window comes back inline; a slower solve returns
`202` and is polled on `GET /v1/solves/{id}`. Enterprise sitekeys are
detected automatically — when the bot has captured an `rqdata` blob the
request switches to `type: hcaptcha_enterprise` and includes it.

The token is injected into every `h-captcha-response` field, known widget
callbacks are fired, the form is submitted, and the page is checked for up
to 10s. **The attempt only counts as a success if the page actually moves
past the captcha** — a minted-but-rejected token is reported back to
NoneCap as `rejected` and the next attempt runs. Failed solves are never
charged.

Falls through to the local solver on: no key, no sitekey, authentication
failure, or exhausted credits.

### Drag rounds: OpenCV contour matching

"Drag the icon to the place where it fits" is answered BEFORE the vision
tiers, because no detector can answer it (every candidate is the same kind
of outlined glyph, and the question is relational).

`shape_match_cv.py` uses the standard tool for this:

| step | function |
|---|---|
| binarise the busy gradient | `cv2.adaptiveThreshold` (12 variants, scored) |
| outline every glyph | `cv2.findContours` |
| compare shapes | `cv2.matchShapes` (Hu moments) |
| break same-family ties | `cv2.approxPolyDP` vertex count |

Hu moments are invariant to translation, rotation and scale — the exact
property needed, since the piece is the same glyph drawn elsewhere at a
different angle. Two details matter in practice:

* the piece sits on a light PANEL, whose contour is bigger than the glyph
  inside it; comparing that rounded rectangle to flowers makes every
  distance meaningless, so container contours are dropped and the glyph
  inside the panel is re-detected;
* Hu distance alone barely separates same-family glyphs (all flowers score
  ~0.17), so the lobe count from `approxPolyDP` dominates the score.

`shape_drag.py` (radial-FFT) stays as the fallback when OpenCV is absent.

### Tier 3: Gemma VLM (`gemma3:4b`)

Last resort, tried only after BOTH the Gemini workflow and `rfdetr-small`
have failed or abstained. Unlike RT-DETR it is a real vision-language
model, so it reads the prompt and can answer the reasoning rounds COCO
has no class for ("odd one out", most trees/tools/terrain).

**Ollama cannot run inside the app container** — that container only has
Python, Chrome and Tor. Run it as a SECOND service and point the app at
it over the private network.

#### Railway (recommended)

1. In the same Railway project: **New → Docker Image → `ollama/ollama`**.
2. Name the service `ollama`.
3. **Variables** on that service:

   ```
   OLLAMA_HOST = [::]:11434
   ```

   Railway's private network is **IPv6-only**. Ollama binds `127.0.0.1`
   by default, and even `0.0.0.0` only listens on IPv4 — either way the
   app service cannot reach it. `[::]` is required.

4. **Settings → Volumes**: mount a volume at `/root/.ollama`, or the
   ~3.3 GB of weights are re-downloaded on every restart.
5. Pull the model once from that service's shell:

   ```bash
   ollama pull gemma3:4b
   ```

6. **Variables** on the APP service:

   ```
   GEMMA_BASE = http://ollama.railway.internal:11434
   ```

   Use the private hostname, not a public domain: it stays off the public
   internet and does not incur egress. Do NOT expose Ollama publicly —
   an open Ollama endpoint is free compute for anyone who finds it.

Memory: `gemma3:4b` needs ~4-6 GB of RAM. Give the Ollama service its own
plan headroom; it must NOT share the app container, which is already
tight enough that Chrome is being OOM-killed.

#### Self-hosted alternative

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:4b
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

```
GEMMA_BASE=http://<host>:11434
```

Open port 11434 in the firewall and put an allowlist in front of it.

Leaving `GEMMA_BASE` unset disables the tier cleanly — the solver just
stops after RT-DETR.

| variable | meaning | default |
|---|---|---|
| `GEMMA_BASE` | Ollama base URL; empty disables tier 3 | — |
| `GEMMA_MODEL` | model tag | `gemma3:4b` |
| `GEMMA_TIMEOUT` | per-solve timeout (s) | `90` |
| `GEMMA_TILE_TIMEOUT` | per-tile timeout for grids (s) | `30` |
| `GEMMA_ENABLED` | set 0 to disable | `1` |

### Backup detector: RT-DETR small

The Gemini workflow is primary. When it fails — host unreachable, 429, out
of credits, or an answer that will not parse — `solve()` automatically
retries the round through `solve_rtdetr()` against **`rfdetr-small`**.

**There is no service to create and nothing to pull.** `rfdetr-small` is a
built-in COCO-pretrained alias hosted on the same serverless endpoint and
authenticated with the same `API_KEY`, so it works the moment your key is
set:

```
POST https://serverless.roboflow.com/infer/object_detection
{"api_key": "...", "model_id": "rfdetr-small",
 "image": {"type": "base64", "value": "..."}}
```

It is ~13 ms of model time versus Gemini's ~13 s, but it only knows the 80
COCO classes and cannot read a prompt. **The knowledge base closes that
gap**: `coco_targets()` resolves the question through `hcaptcha_types`'
~1700-entry alias table and set predicates down to COCO labels —
`helicopter` → `airplane`, `nightstand` → `dining table`, "an animal" →
every COCO animal, "setting down" → the flat surfaces. A prompt with no
COCO equivalent (`odd one out`, most trees and tools) makes the backup
**abstain** instead of answering with the wrong object.

Optional self-hosting (a local GPU/CPU server instead of the cloud):

```bash
pip install inference-cli
inference server start          # pulls the right image, listens on :9001
export ROBOFLOW_API_BASE=http://localhost:9001
```
| `FULLPAGE_SHOTS` | whole scrollable page camera frames (default: full browser-view frames with the register form revealed when out of sight) | `0` |
| `FULLPAGE_MAX_PX` | max page height (px) worth a full-page frame; taller pages fall back to viewport frames | `8000` |

## Verification

```bash
python test_solver.py         # routing, knowledge base, pointer realism,
                              # detection->answer mapping, per-tile grid path
python test_vision_client.py  # Roboflow readiness + exact request body
python drag_solver.py         # Arkose stack-plan parser self-tests
python solver.py --check      # live probe: API_KEY valid? workflow found?
```

`test_solver.py` and `test_vision_client.py` are fully offline — no network
and no API key needed; the HTTP layer is mocked.

## Honest limits

* Accuracy is the accuracy of the workflow behind `ROBOFLOW_WORKFLOW`.
  Gemini 3.6 Flash scores 83.0% overall on Roboflow's Vision Evals but
  only 57.1% on object detection at low thinking effort — box placement
  is its weakest axis, which is exactly what point/bbox/drag rounds need.
* Latency: Gemini 3.6 Flash averages ~13-15 s per sample. A 9-tile grid is
  9 sequential requests, so a grid round can approach the challenge
  timeout. Shrink `ROBOFLOW_IMAGE_SIDE` or use a lighter workflow if
  rounds start expiring.
* Cost: one request per tile means a 9-tile round is 9 billed inferences.
* A detection-only workflow cannot answer `choice`, `text` or `stack`
  rounds from boxes alone — those need the workflow to emit text, which
  the client parses if present, and otherwise the round fails over to the
  next attempt.
* Serverless workflows scale to zero; the first call after idle pays a
  cold start. The 10-minute warmup ping in `app.py` mitigates it.
* Invisible bbox tolerance: the bbox answer is graded against an invisible
  ground-truth rectangle; being a few pixels off can still fail.
* One bad round loses a multi-round challenge (2-3 rounds per challenge —
  per-round accuracy compounds).
* Counting answers are graded exactly: off by one fails the round.
* Never put `API_KEY` in logs, screenshots or committed config — it is
  read from the environment only.
* Proxy/fingerprint quality still dominates the overall pass rate: a
  flagged IP never even sees a solvable challenge.

## Outcome telemetry (Roboflow Vision Events)

Every round is reported as pass/fail so accuracy is measured instead of
guessed. Off by default; opt in with:

```
VISION_EVENTS = 1
VISION_EVENTS_USE_CASE = hcaptcha-solver
```

Uses the same `API_KEY` as a bearer token. Two event streams are sent:

| metadata | meaning |
|---|---|
| `family`, `round`, `prompt`, `tiles` | per-round solve outcome |
| `stage=drag_gesture`, `gesture` | WHICH drag channel actually stuck |

The second is the important one for drag rounds: `gesture` names the
winning strategy (`pointer`, `slow-pointer`, `html5-dnd`,
`pointer-events`) or `all-failed`. Once one channel proves itself, promote
it to first in `_drag_verified` and drop the wasted attempts.

Telemetry never blocks or breaks a solve — posts are fire-and-forget and
every error is swallowed at debug level.

# hCaptcha multi-family solver

The solver started as a single path: screenshot `div.task-image` tiles → ask
a vision model which match → click them. That is one of **five** challenge
families hCaptcha serves; the other four were answered with tile indices and
could never pass. This document describes the rewrite: a family router, a
semantic knowledge base, humanized pointer telemetry, and a **Hugging Face
vision model** that answers every family through one shaped JSON contract.

All visual reasoning is remote: there is no local checkpoint, no CNN
weights and no self-hosted model server in this repo. Configure `API_KEY`
(a Hugging Face token) and, optionally, `HF_MODEL`.

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
the vision model answers with `shape="count"` and the round is clicked
through `_click_number_option`.

Pattern-completion rounds ("put one of the animals into the empty spot to
complete the pattern") are `image_drag_drop` under the hood, but the
dragged candidate is chosen by the **pattern**, not by geometry: the
prompt tier and the DOM tier (pattern wording + draggables / many tiles)
route them to DRAG_DROP, `is_pattern_prompt()` flags them, and the round
loop dispatches to `_solve_pattern_round` instead of the geometric
`_solve_drag_round`, which asks the vision model with `shape="pattern"`
and replays the answer as a candidate→hole drag.

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
into something a VLM answers reliably ("does this photo show a table,
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
        ┌──────────────┴────────────────────┐
        │ Hugging Face vision (vision_solver)│  shaped system prompt
        │  api-inference.huggingface.co      │  + JSON repair;
        │  chat/completions + image data URIs│  small VLMs: per-tile
        │  auth: API_KEY (hf_...)            │  yes/no
        └───────────────────────────────────┘
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
not offline answering: the answer itself always comes from the vision
model. The resolver helpers below are retained and unit-tested because the
routers and the set-down question rewriter build on the same vocabulary.

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
  stays `None` — the vision model reads arbitrary prompt text anyway.
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

`HFVisionClient` speaks the OpenAI-compatible chat schema of the Hugging
Face serverless Inference API:

```
POST https://api-inference.huggingface.co/models/<HF_MODEL>/v1/chat/completions
Authorization: Bearer $API_KEY
```

Tile screenshots are downscaled (`HF_IMAGE_SIDE`, default 512) and attached
as `image_url` data URIs. Each answer shape has its own system prompt
(`tiles`, `points`, `bbox`, `drag`, `pattern`, `tower`, `stack`, `choice`,
`count`, `text`) and the reply is parsed by `_parse_geometry` /
`_parse_answer`, which repair everything small models emit: markdown
fences, bare-dot decimals, trailing commas, percent-vs-fraction units,
loose prose tile numbers.

Small captioning VLMs (SmolVLM, moondream, or any model with
`HF_PER_TILE=1`) cannot follow a 9-image JSON contract, so grids are asked
one tile at a time as a yes/no question (`tile_yes_question` +
`parse_yesno`).

`check()` probes the model route: HTTP 200/405 = warm, 503 = cold-starting
(still healthy), 401 = bad `API_KEY`, 403 = licence/permission, 404 = wrong
`HF_MODEL`. The server retries only the transient classes; auth/protocol
errors fail the round immediately instead of burning three probes.
`app.py` pings the model every 10 minutes so serverless keeps it resident.

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
  Hugging Face vision model with the new `shape="stack"` answer contract
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
| `API_KEY` | **Hugging Face access token (`hf_...`) — required** | — |
| `HF_MODEL` | vision model repo id | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `HF_API_BASE` | Inference API base URL | `https://api-inference.huggingface.co/models` |
| `HF_TIMEOUT` | per-solve timeout (seconds) | `60` |
| `HF_TILE_TIMEOUT` | per-tile yes/no timeout for small VLMs | `20` |
| `HF_IMAGE_SIDE` | max image side (px) sent to the model | `512` |
| `HF_PER_TILE` | force the per-tile yes/no grid path | `0` |
| `HF_CHECK_TIMEOUT` | readiness-probe timeout (seconds) | `60` |
| `FULLPAGE_SHOTS` | whole scrollable page camera frames (default: full browser-view frames with the register form revealed when out of sight) | `0` |
| `FULLPAGE_MAX_PX` | max page height (px) worth a full-page frame; taller pages fall back to viewport frames | `8000` |

## Verification

```bash
python test_solver.py         # routing, knowledge base, pointer realism,
                              # vision answer parsing, per-tile path
python test_vision_client.py  # Hugging Face readiness + request payload
python drag_solver.py         # Arkose stack-plan parser self-tests
python solver.py --check      # live probe: API_KEY valid? model up?
```

`test_solver.py` and `test_vision_client.py` are fully offline — no network
and no API key needed; the HTTP layer is mocked.

## Honest limits

* Accuracy is the accuracy of whatever `HF_MODEL` you point at. A 7B-class
  VLM (Qwen2.5-VL and friends) handles grids, points and counting
  reasonably; 256M captioning models only manage per-tile yes/no.
* Serverless cold starts: the first solve after an idle period can cost
  20–60 s (HTTP 503 "model is loading"), which can expire a challenge. The
  10-minute warmup ping in `app.py` mitigates this; a dedicated Inference
  Endpoint removes it entirely (point `HF_API_BASE` at it).
* Rate limits: the free serverless tier throttles hard (HTTP 429). A
  9-tile per-tile round is 9 requests — budget accordingly.
* Every round costs a network round trip, so a multi-round challenge is
  latency-bound; tower/pattern rounds keep short timeouts specifically so
  a slow answer does not expire the challenge.
* Invisible bbox tolerance: the bbox answer is graded against an invisible
  ground-truth rectangle; being a few pixels off can still fail.
* One bad round loses a multi-round challenge (2–3 rounds per challenge —
  per-round accuracy compounds).
* Counting answers are graded exactly: off by one fails the round.
* Never put `API_KEY` in logs, screenshots or committed config — it is read
  from the environment only.
* Proxy/fingerprint quality still dominates the overall pass rate: a
  flagged IP never even sees a solvable challenge.

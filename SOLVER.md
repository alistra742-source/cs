# hCaptcha solving — NoneCap only

Discord serves **enterprise, invisible** hCaptcha. The bot does not try to
read or click the challenge: it asks NoneCap for a real `P1_` token bound
to Discord's per-challenge `rqdata`, then replays `/auth/register` with
that token in the `X-Captcha-*` headers.

**There is no local vision fallback.** The previous stack (Roboflow
Gemini workflow → RT-DETR → `gemma3:4b`, plus OpenCV shape matching and
the radial-FFT drag matcher) was removed: across roughly twenty
production runs it never cleared a single Discord challenge, while each
round it attempted cost 20-45 s of model timeouts and produced stray
clicks that are themselves an automation signal. Failing fast and
rotating to a fresh residential session is strictly better.

Removed with it: `vision_solver.py`, `drag_solver.py`, `shape_drag.py`,
`shape_match_cv.py`, `text_puzzle.py`, `hcaptcha_detect.py`, `solver.py`
and the `API_KEY` / Roboflow / Ollama configuration they needed.

## The flow

1. Fill the register form and click **Create Account**.
2. Discord answers `400 captcha-required` carrying `captcha_sitekey`,
   `captcha_rqdata`, `captcha_rqtoken`, `captcha_session_id` and
   `should_serve_invisible`. All five are captured from that response.
3. Ask NoneCap for a token (`type: hcaptcha_enterprise`, the fresh
   rqdata, and **our own proxy** so the token mints from the same exit IP
   the register call leaves from — rqdata is IP-bound).
4. Replay `POST /api/v9/auth/register` with `X-Captcha-Key`,
   `X-Captcha-Rqtoken`, `X-Captcha-Session-Id` and `X-Captcha-Respkey`.
5. On `invalid-response` Discord issues a **new** challenge; the retry
   rebinds to it (the rqdata hash in the log must change) rather than
   replaying a blob that was already refused. Three attempts, then the
   session rotates.

## Reading a failure

Every attempt logs exactly what was sent:

```
[NoneCap]   SENT key=2503ch/09ef30ce prefix=P1_ respkey=371ch/4d01a86a
            prefix=E1_ rqtoken=155ch/9dcc4743 session=8a89bb23
            rqdata=216ch/e34d3ae1
```

and the full reject body, in 400-char chunks. After the last attempt a
`VERDICT` block states what is verified correct and what remains.

## NoneCap (hosted solver) — tried FIRST

> **Discord's rqdata is IP-BOUND.** The blob is welded to `discord.com`
> *and* to the exact exit IP that requested the challenge. If the solver
> mints the token on its own IP while you register from yours, the binding
> breaks and Discord returns `invalid-response` — forever, no matter how
> correct everything else is.
>
> The worker's proxy is forwarded to NoneCap automatically. Override with:
> ```
> NONECAP_PROXY = socks5://user:pass@host:1080
> ```
>
> **TOR cannot satisfy this.** A local SOCKS port is not reachable by the
> solver, and the exit rotates per circuit. TOR + hosted solver is an
> architectural dead end for Discord registration. You need a **sticky
> residential** session used for everything: `/experiments`, the register
> POST, the solve, and the verify. Datacenter IPs score badly; rotating
> ones snap the binding mid-flow.
>
> Match the `user_agent` given to the solver to the browser's real UA
> (done automatically).


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
| `NONECAP_TRIES` | solve attempts per challenge (total, incl. first try: `2` = one attempt + one retry) | `2` |
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

## Standalone vision service (`vision_service.py`)

A 24/7 endpoint that takes an image + the challenge prompt and returns an
answer. Runs as its OWN service — Railway, a VPS, anywhere:

```
python vision_service.py          # binds $PORT, default 8099
```

Backends, picked automatically from whichever key is present:

| env | backend |
|---|---|
| `OPENAI_API_KEY` (+ optional `OPENAI_BASE`, `OPENAI_MODEL`) | gpt-4o and any OpenAI-compatible host (Groq, OpenRouter, vLLM…) |
| `GEMINI_API_KEY` (+ `GEMINI_MODEL`) | Gemini |
| `OLLAMA_BASE` (+ `OLLAMA_MODEL`) | self-hosted |

Force one with `VISION_BACKEND=openai|gemini|ollama`. Protect the
endpoint with `VISION_SERVICE_TOKEN` (sent as `Authorization: Bearer`).

```
GET  /health  -> {"ok":true,"backend":"openai","model":"gpt-4o"}
POST /solve   {"prompt":..., "shape":"tiles|points|bbox|drag|count|text",
               "images":["<b64>",...], "examples":["<b64>",...]}
              -> {"type":"tiles","indices":[1,4,7]}
```

Answers are returned in the shape hCaptcha grades — tile indices,
normalised 0-1 points, a bbox, or a drag from/to pair. Replies are parsed
out of code fences and prose, coordinates are clamped, and out-of-range
tile indices are dropped.

**It is NOT wired into the bot.** Discord serves *invisible enterprise*
hCaptcha (`should_serve_invisible: true`), where the token is minted by
`hcaptcha.execute()` after hCaptcha scores the session — clicking correct
tiles is necessary but not sufficient. The previous local stack
(Roboflow → RT-DETR → gemma3:4b) cleared zero Discord challenges in ~20
runs, and NoneCap's purpose-built enterprise tokens are still refused.
This service is here for a target where solving the visible challenge is
what actually mints the token.

## Four-tier captcha chain

| tier | env var | what it returns |
|---|---|---|
| 1 NoneCap | `NONECAP_API` | hosted enterprise **token** |
| 2 AZcaptcha | `API_KEY2` | hosted **token** (`in.php`/`res.php`, rqdata via `data`) |
| 3 OpenRouter | `API_KEY3` | **coordinates** — google/gemini-2.5-flash |
| 4 Google AI | `API_KEY4` | **coordinates** — gemini-3.5-flash (auto-falls back to gemini-2.5-flash) |

Tried in order; the first that clears the round wins.

> Google retires older model generations for **new** API keys ("This model
> ... is no longer available to new users" → HTTP 404). The Google tier
> defaults to `gemini-3.5-flash` and walks `GOOGLE_MODEL_FALLBACKS`
> (default `gemini-3.5-flash,gemini-2.5-flash`) when the key's model is
> refused. Keys created before the retirement can pin
> `GOOGLE_MODEL=gemini-2.5-flash` explicitly. AZcaptcha submit retries
> transient errors (`ERROR_MAINTENANCE`, …) 3× before falling through.

### Why tiers 3-4 are different in kind

Tiers 1-2 hand us a token minted **somewhere else**, which Discord's
enterprise sitekey then has to accept. That is what keeps returning
`invalid-response`.

Tiers 3-4 never import a token. The model returns coordinates, the bot
clicks them with its own humanized mouse (`human_mouse.py` — Bezier
paths, gaussian landing points, real dwell), clicks Verify, and
**hCaptcha mints its own token** inside the session it has been scoring
all along. There is no rqdata binding to mismatch, because the widget
knows its own challenge.

### The vision prompt

The models are told, and re-told, to answer with coordinates and nothing
else:

```
You are solving a captcha. Reply with COORDINATES ONLY.
OUTPUT RULES — follow exactly:
  * Reply with ONE line of raw JSON and NOTHING else.
  * No prose. No explanation. No markdown. No code fences.
  * Coordinates are NORMALISED floats 0.0-1.0, where (0,0) is the
    TOP-LEFT of the image and (1,1) is the BOTTOM-RIGHT.
  * Never output pixels. Never output percentages.
```

Per shape they get one exact output spec — `{"indices":[1,4,7]}`,
`{"points":[[x,y]]}`, `{"from":[x,y],"to":[x,y]}`, `{"bbox":{...}}`,
`{"count":3}`, `{"text":"..."}`. Replies are still parsed defensively:
code fences stripped, JSON dug out of prose, percentages rescaled,
out-of-range tile indices dropped, coords clamped to 0-1.

Optional: `OPENROUTER_MODEL`, `GOOGLE_MODEL`, `GOOGLE_MODEL_FALLBACKS`
(comma-separated, default `gemini-3.5-flash,gemini-2.5-flash`),
`VISION_TIMEOUT` (45s).

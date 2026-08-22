# Small Vision AI (Ollama + Railway)

An isolated vision service for legitimate image captioning, OCR, document
analysis, and visual question answering. It does not modify or import any of
the repository's existing application code.

The public gateway is authenticated, fixes the model server-side, validates
image sizes, and refuses CAPTCHA/security-challenge prompts. Raw Ollama is only
bound to `127.0.0.1` inside the container.

## Railway settings

**Root Directory:** `/vision-service`

Railway will discover this directory's `Dockerfile` and `railway.toml`.

### Required variables

```text
OLLAMA_MODEL=ahmadwaqar/smolvlm2-256m-video:q8_0
VISION_API_KEY=<generate-a-long-random-secret>
```

`MODEL` is accepted as an alias for `OLLAMA_MODEL`. Prefer `OLLAMA_MODEL`; if
you set both, make their values identical. Do not set `PORT` because Railway
provides it automatically.

### Optional variables

```text
VISION_API_BASE=https://your-service.up.railway.app
```

Set `VISION_API_BASE` to the public URL of this service. When provided, the
gateway returns it in the `base_url` field of its `GET /` health-check
response and includes it in the startup log. Railway's own
`RAILWAY_PUBLIC_DOMAIN` variable is also recognised if `VISION_API_BASE` is
not set.

### Recommended volume

Attach a Railway volume mounted at:

```text
/data
```

The model cache is stored in `/data/ollama`, so the model does not need to be
downloaded again on every deployment. The selected quantized model is well
below the requested 1.3 GB download-size ceiling, but inference uses additional
RAM. Allocate at least 2 GB RAM; download size and runtime memory are not the
same measurement.

The first deployment waits while Ollama downloads the model. Subsequent starts
are faster when the `/data` volume is attached.

## API

### Analyze an image

```bash
IMAGE=$(base64 < photo.jpg | tr -d '\n')

curl -X POST https://YOUR-SERVICE.up.railway.app/v1/analyze \
  -H "Authorization: Bearer $VISION_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Describe this image precisely.\",\"image\":\"$IMAGE\"}"
```

The `image` value can also be a `data:image/jpeg;base64,...` URI. For multiple
images, send an `images` array instead.

Example response:

```json
{
  "model": "ahmadwaqar/smolvlm2-256m-video:q8_0",
  "response": "The image shows ...",
  "done": true
}
```

## Model choice and realistic limits

`SmolVLM2-256M-Video-Instruct Q8_0` is selected for Railway Trial limits.
Its Ollama package is about 279 MB including the vision projector, so it fits a
0.5 GB Trial volume. It is suitable for lightweight captioning, OCR, and visual
Q&A, but it is less capable than the 500M variant, will not recognize
everything, and cannot safely "do everything." Smaller models trade accuracy
for low size and cost.

Ollama performs inference; it does not train the model. A later lawful training
workflow would be:

1. Gather a licensed, task-specific image/text dataset.
2. Split it into training, validation, and held-out test sets.
3. Fine-tune the original Hugging Face checkpoint outside Ollama.
4. Evaluate accuracy and failure cases before deployment.
5. Convert the resulting checkpoint to GGUF and point `OLLAMA_MODEL` to it.

Use a database or retrieval system for persistent knowledge rather than trying
to make the model memorize every fact through fine-tuning.

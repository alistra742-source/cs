FROM ollama/ollama:latest

# The official Ollama image contains the runtime. Python and curl power the
# small authenticated HTTP gateway and local Ollama startup checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY server.py start.sh ./
RUN chmod +x /service/start.sh

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/data/ollama \
    DEFAULT_OLLAMA_MODEL=hf.co/ggml-org/SmolVLM-500M-Instruct-GGUF:Q8_0 \
    OLLAMA_KEEP_ALIVE=10m \
    OLLAMA_REQUEST_TIMEOUT=180 \
    MAX_IMAGE_BYTES=10485760

# Railway manages the persistent /data mount from the service's Volume UI.
# Do not declare Docker VOLUME here; Railpack/Railway rejects that instruction.
EXPOSE 8080

# ollama/ollama defines its own `ollama` entrypoint. Replace it so Railway
# executes our startup script instead of passing the script path to Ollama.
ENTRYPOINT ["/service/start.sh"]
CMD []

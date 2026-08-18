FROM ollama/ollama:latest

# The official Ollama image contains the runtime. Python and curl power the
# small authenticated HTTP gateway and startup health checks.
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

VOLUME ["/data"]
EXPOSE 8080

CMD ["/service/start.sh"]

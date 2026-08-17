FROM python:3.11-slim

WORKDIR /app

# System dependencies for the Camoufox engine (a debloated Firefox fork) + Tor
RUN apt-get update && apt-get install -y \
    wget gnupg curl ca-certificates tor unzip \
    libglib2.0-0 libnss3 libnspr4 libdbus-1-3 \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libasound2 libcairo2 libpango-1.0-0 libgtk-3-0 \
    libexpat1 libx11-6 libxcb1 libxext6 \
    fonts-liberation libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
# --retries/--timeout: builds pulling deps over the wire can hit transient
# network blips; retry instead of killing the whole build.
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

# Fetch the Camoufox browser binary once at build time so the image is
# self-contained (engine launches are instant at runtime). Falls back to
# fetching at first launch if the build has no network.
RUN python -m camoufox fetch || echo "[Camoufox] fetch skipped - will fetch at first launch"

# Copy ALL application files
COPY *.py ./
COPY *.txt ./
COPY config.json ./
COPY torrc /etc/tor/torrc
COPY start.sh ./
RUN chmod +x start.sh

EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
CMD ["./start.sh"]

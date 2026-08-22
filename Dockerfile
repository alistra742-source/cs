FROM python:3.11-slim

WORKDIR /app

# Base runtime dependencies for the web application, Chromium rendering, and Tor.
# ``chromium`` is the browser binary truedriver launches (found on PATH).
RUN apt-get update && apt-get install -y \
    wget gnupg curl ca-certificates tor unzip chromium \
    libglib2.0-0 libnss3 libnspr4 libdbus-1-3 \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libasound2 libcairo2 libpango-1.0-0 libgtk-3-0 \
    libexpat1 libx11-6 libxcb1 libxext6 \
    fonts-liberation libatspi2.0-0 \
    libgl1 libgl1-mesa-dri mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install the Python driver first (truedriver pulls websockets/deprecated/etc.
# as its own dependencies).
COPY requirements.txt ./
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

# Run the browser and Tor under a non-root account. Chromium is designed to
# run as an unprivileged process; browser files remain readable by that user.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser /app

# Copy application files, then give the runtime account ownership.
COPY *.py ./
COPY *.txt ./
COPY config.json ./
COPY torrc /etc/tor/torrc
COPY start.sh ./
RUN chmod +x start.sh && chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser
ENV XDG_CACHE_HOME=/home/appuser/.cache
ENV LIBGL_ALWAYS_SOFTWARE=1

EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
CMD ["./start.sh"]

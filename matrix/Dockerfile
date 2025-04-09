FROM python:3.10-slim

# Install required packages (libolm, ffmpeg, build tools)
RUN apt-get update && apt-get install -y \
    libolm-dev \
    ffmpeg \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Pre-install pip, wheel, and matrix-nio[e2e]
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --upgrade build \
    && pip install --no-cache-dir "matrix-nio[e2e]" aiohttp \
    && pip install --no-cache-dir -r requirements.txt


# Create Matrix store dir with safer permissions
RUN mkdir -p /data/matrix-store && chmod -R 755 /data

# Copy rest of app code
COPY . .

EXPOSE 5004

CMD ["python", "-u", "bot.py"]


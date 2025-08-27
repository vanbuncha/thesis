FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    espeak \
    ffmpeg \
    libsndfile1 \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (for TTS dependency)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 5003

CMD ["uvicorn", "tts_server:app", "--host", "0.0.0.0", "--port", "5003"]


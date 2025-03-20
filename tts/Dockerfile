FROM --platform=linux/arm64 python:3.10-slim

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    espeak \
    ffmpeg \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install the latest rust and pip
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --upgrade pip

# Copy Python dependency file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 5003 for the TTS service
EXPOSE 5003

CMD ["python3", "tts_server.py"]


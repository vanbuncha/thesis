# STT Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    espeak \
    ffmpeg \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install vosk flask gunicorn  

# Copy STT code and model
COPY . /app
COPY models /app/models

WORKDIR /app

# Expose port
EXPOSE 5001

# Run STT server with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "stt_server:app"]

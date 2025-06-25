FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    espeak \
    ffmpeg \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for FastAPI STT service
EXPOSE 5001

# Launch FastAPI app
CMD ["uvicorn", "stt_server:app", "--host", "0.0.0.0", "--port", "5001", "--log-level", "debug"]



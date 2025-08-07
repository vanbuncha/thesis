FROM python:3.13-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade torch torchvision torchaudio


# Copy rest of app
COPY . .

# Expose port
EXPOSE 5001

# Run STT server
CMD ["uvicorn", "stt_server:app", "--host", "0.0.0.0", "--port", "5001", "--log-level", "debug"]



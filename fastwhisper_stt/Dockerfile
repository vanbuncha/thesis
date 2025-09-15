FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04


# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libsndfile1 ffmpeg \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install --no-cache-dir faster-whisper

# App
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001
CMD ["uvicorn", "stt_server:app", "--host", "0.0.0.0", "--port", "5001", "--log-level", "debug"]


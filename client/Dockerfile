FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pi_client.py ./
COPY .env.client ./
COPY hello-friend.ppn porcupine_params.pv ./

CMD ["python", "pi_client.py"]


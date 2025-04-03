FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    espeak \
    ffmpeg \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set workdir 
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app 
COPY . .

# Install any extra dependencies
RUN pip install vosk flask pydub

# Expose port
EXPOSE 5001

# Run STT server
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "stt_server:app"]


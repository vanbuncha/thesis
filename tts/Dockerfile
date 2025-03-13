# TTS Dockerfile
FROM python:3.9-slim

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CARGO_HOME=/root/.cargo

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install Rust (latest stable version)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Verify Rust installation
RUN rustc --version && cargo --version

# Install setuptools-rust (fixes sudachipy build issue)
RUN pip install setuptools-rust

# Install Cython (specific version to avoid compatibility issues)
RUN pip install "Cython<3.0"

# Install spaCy separately to avoid compilation errors
RUN pip install "spacy<3.6" numpy

# Install TTS and Flask with no build isolation
RUN pip install TTS flask --no-build-isolation

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

# Expose the necessary port
EXPOSE 5000

# Define the entry point
CMD ["python", "app.py"]


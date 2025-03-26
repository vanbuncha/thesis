# Use slim Python 3.10 as base
FROM python:3.10-slim

# Install libolm and compiler tools
RUN apt-get update && \
    apt-get install -y libolm-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies including matrix-nio[e2e]
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "matrix-nio[e2e]" aiohttp
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /data/matrix-store && chmod -R 777 /data


# Copy the rest of the code
COPY . .

# Optional: expose healthcheck or debug port if needed
EXPOSE 5004

# Default command
CMD ["python", "-u", "bot.py"]


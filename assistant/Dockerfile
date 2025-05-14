FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the assistant source code
COPY . .

# Entry point: Run FastAPI with Uvicorn
CMD ["uvicorn", "ws_server:app", "--host", "0.0.0.0", "--port", "8000"]


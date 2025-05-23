FROM python:3.9-slim

WORKDIR /app

# Install build essentials to support uvicorn deps
RUN apt-get update && apt-get install -y build-essential gcc libffi-dev

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 5001

CMD uvicorn llm_server:app --host 0.0.0.0 --port 5001 


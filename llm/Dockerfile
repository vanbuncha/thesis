FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl && \
    rm -rf /var/lib/apt/lists/*

COPY llm/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY llm/llm_server.py /app/llm_server.py
COPY assistant/db /app/db
ENV PYTHONPATH=/app

EXPOSE 5001
CMD ["uvicorn", "llm_server:app", "--host", "0.0.0.0", "--port", "5001"]


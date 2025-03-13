# LLM Dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install transformers flask

# Copy LLM code
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 5002

# Run LLM server
CMD ["python", "llm_server.py"]

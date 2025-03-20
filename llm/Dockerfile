FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app files
COPY . .

# Expose port 5002
EXPOSE 5002

# Run the Flask app
CMD ["python", "llm_server.py"]


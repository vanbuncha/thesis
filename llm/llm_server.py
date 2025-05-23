from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import json
import logging

app = FastAPI()

# Logging setup
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://ollama:11434/api/generate"  # Ollama API


@app.get("/health")
async def health():
    return "OK"


@app.post("/generate")
async def generate(request: Request):
    try:
        raw_data = await request.body()
        raw_text = raw_data.decode("utf-8")
        logger.debug("Raw Data Received: %s", raw_text)

        json_data = json.loads(raw_text)
        logger.debug("Parsed JSON Data: %s", json_data)

        user_input = json_data.get("prompt", "")

        response = requests.post(
            OLLAMA_URL,
            json={"model": "mistral", "prompt": user_input, "stream": True},
            stream=True,
        )

        if response.status_code == 200:
            full_response = ""

            for chunk in response.iter_lines():
                if chunk:
                    try:
                        decoded_chunk = json.loads(chunk.decode("utf-8"))
                        token = decoded_chunk.get("response", "")
                        full_response += token
                    except json.JSONDecodeError:
                        logger.error("Error decoding JSON chunk: %s", chunk)

            return JSONResponse(content={"response": full_response})
        else:
            logger.error(f"Ollama error: {response.status_code} - {response.text}")
            return JSONResponse(
                content={"error": "Ollama API returned an error"}, status_code=500
            )

    except Exception as e:
        logger.error("Exception: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

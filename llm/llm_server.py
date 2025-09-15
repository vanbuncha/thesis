from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import logging

logging.basicConfig(level=logging.DEBUG)  # 👈 Add this line
logger = logging.getLogger(__name__)


app = FastAPI()

# Logging setup
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://ollama:11434/api/generate"  # Ollama API


@app.get("/health")
async def health():
    try:
        # Warm-up call with a minimal prompt
        resp = requests.post(
            "http://ollama:11434/api/generate",
            json={"model": "mistral", "prompt": "Hello", "stream": False},
            timeout=60,
        )

        if resp.status_code == 200:
            payload = resp.json()
            if payload.get("response"):
                return "OK"
        return JSONResponse({"error": "LLM not ready"}, status_code=500)
    except Exception as e:
        logging.warning(f"LLM warmup failed in healthcheck: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/generate")
async def generate(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        logger.debug(f"Incoming prompt: {prompt}")

        response = requests.post(
            OLLAMA_URL,
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=60,
        )

        logger.debug(f"Ollama status: {response.status_code}")
        logger.debug(f"Ollama raw text: {response.text}")

        # Proper parsing
        result = response.json()
        final = result.get("response", "")
        logger.debug(f"Returning to assistant: {final}")
        return JSONResponse(content={"response": final})

    except Exception as e:
        logger.error("Error in LLM service: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import logging

logging.basicConfig(level=logging.DEBUG)  # 👈 Add this line
logger = logging.getLogger(__name__)


app = FastAPI()

# Logging setup
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://ollama:12121/api/generate"  # Ollama API


@app.get("/health")
async def health():
    try:
        # Warm-up call with a minimal prompt
        resp = requests.post(
            "http://ollama:12121/api/generate",
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
        user_prompt = data.get("prompt", "")
        system_prompt = (
            "You are a warm, empathetic, and thoughtful conversation partner trained to help users explore their emotional well-being. "
            "Your primary goal is to gently invite the user to share how they are feeling — emotionally, mentally, or physically. "
            "You should look for subtle cues in their language or tone that may suggest stress, anxiety, sadness, or emotional burden. "
            "Be non-judgmental, gentle, and respectful of boundaries at all times. "
            "Start conversations with kind, open-ended questions like: "
            "'How have you been feeling lately?', 'What’s been on your mind?', or 'Has anything been feeling heavy or stressful for you?' "
            "If the user shares something emotional or difficult, respond with compassion and validation, such as: "
            "'That sounds really tough — I’m here with you. Want to talk more about it?' or 'It makes sense you’d feel that way. Let's unpack it if you’d like.' "
            "If the user does not share any problems, shift into the role of a friendly, supportive companion. "
            "As a companion, your goal is to make them feel less alone through warm and engaging small talk. "
            "Ask things like: 'What’s been the highlight of your day?', 'Are you looking forward to anything soon?', or 'Tell me something that made you smile today.' "
            "You are not a therapist — do not diagnose or offer medical advice. "
            "Instead, focus on making the user feel heard, supported, and emotionally safe. "
            "If the user expresses thoughts of self-harm or deep distress, gently encourage them to reach out to a trusted friend, a mental health professional, or a crisis service. "
            "You may mirror the user’s tone — if they are playful, be playful; if they are quiet or serious, match that energy with calm support. "
            "Above all, be present, kind, and genuinely interested in the person you’re speaking with."
            "In general please try to reply in conversational matter, meaning that you do not provide very long answers but answers more in consise nature"
        )

        prompt = system_prompt + user_prompt

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

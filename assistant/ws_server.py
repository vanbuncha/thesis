import os
import json
import tempfile
import aiohttp
from fastapi import FastAPI, WebSocket
from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles

import asyncio

STT_URL = "http://stt_vosk:5002/transcribe"
# STT_URL = "http://stt_fastwhisper:5001/transcribe"
LLM_URL = "http://llm:5001/generate"
TTS_URL = "http://tts:5003/synthesize"


services = {
    "STT (Vosk)": "http://stt_vosk:5002/transcribe",
    "STT (FastWhisper)": "http://stt_fastwhisper:5001/transcribe",
    "LLM": "http://llm:5001/generate",
    "TTS": "http://tts:5003/synthesize",
    "Ollama": "http://ollama:11434/",
    "Database": "http://database:5432/",
}

app = FastAPI()
router = APIRouter()


# -------- HEALTHCHECK ------------


@router.get("/health")
async def health_check():
    results = {}

    async with aiohttp.ClientSession() as session:
        for name, url in services.items():
            if url.startswith("http"):
                try:
                    # Decide method type based on endpoint
                    if "/transcribe" in url:
                        payload = aiohttp.FormData()
                        payload.add_field(
                            "audio", b"", filename="dummy.wav", content_type="audio/wav"
                        )
                        async with session.post(url, data=payload, timeout=2) as resp:
                            results[name] = resp.status in (
                                200,
                                400,
                            )  # Accept 400 for empty input
                    elif "/generate" in url:
                        async with session.post(
                            url, json={"prompt": ""}, timeout=2
                        ) as resp:
                            results[name] = resp.status in (200, 400)
                    elif "/synthesize" in url:
                        async with session.post(
                            url, json={"text": ""}, timeout=2
                        ) as resp:
                            results[name] = resp.status in (200, 400)
                    else:
                        async with session.get(url, timeout=2) as resp:
                            results[name] = resp.status == 200
                except Exception:
                    results[name] = False
            else:
                results[name] = None  # Non-HTTP

    return results


# -------- HEALTHCHECK ------------


def extract_text_from_stt(stt_raw):
    try:
        outer = json.loads(stt_raw)
        inner_json = outer.get("text", "")
        if not inner_json:
            return ""
        import re

        matches = re.findall(r'{\s*"text"\s*:\s*"([^"]*)"\s*}', inner_json)
        return " ".join(matches).strip()
    except Exception as e:
        print(f"❌ Failed to extract text: {e}")
        return ""


async def transcribe_audio(audio_file_path):
    for _ in range(5):
        try:
            async with aiohttp.ClientSession() as session:
                with open(audio_file_path, "rb") as f:
                    data = {"audio": f}
                    async with session.post(STT_URL, data=data) as resp:
                        return await resp.text()
        except aiohttp.ClientConnectorError:
            print("❌ STT service not ready yet, retrying...")
            await asyncio.sleep(2)
    raise RuntimeError("Failed to connect to STT after several retries.")


async def generate_response(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_URL, json={"prompt": prompt}) as resp:
            result = await resp.json()
            print("raw response:", result)
            return result.get("response", "")


async def synthesize_speech(text, out_path):
    if not text.strip():
        print("Empty or invalid text for TTS, skipping synthesis.")
        return

    async with aiohttp.ClientSession() as session:
        async with session.post(TTS_URL, json={"text": text}) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"TTS failed: {resp.status} — {error}")
            audio = await resp.read()
            with open(out_path, "wb") as f:
                f.write(audio)


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connection established")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        print(f"📥 Receiving audio chunks to {tmp_audio.name}")
        while True:
            data = await websocket.receive_bytes()

            # 🔑 Check for end-of-stream marker
            if data == b"\x00":
                print("🛑 End of stream signal received.")
                break

            tmp_audio.write(data)

        tmp_audio.flush()

    print("🎧 Audio received, running pipeline...")
    stt_result = await transcribe_audio(tmp_audio.name)
    prompt = extract_text_from_stt(stt_result)
    reply = await generate_response(prompt)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_audio:
        print(f"📝 Synthesizing reply: {reply}")
        await synthesize_speech(reply, tts_audio.name)

        if not os.path.exists(tts_audio.name) or os.path.getsize(tts_audio.name) < 1000:
            print("❌ TTS output is empty or too small — skipping send.")
            await websocket.send_bytes(b"")
            return

        print(f"📤 Sending audio response: {tts_audio.name}")
        with open(tts_audio.name, "rb") as f:
            await websocket.send_bytes(f.read())

    print("✅ Response sent, closing WebSocket")
    await websocket.close()


# main
app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")

import os
import json
import tempfile
import aiohttp
import time
import wave
from fastapi import FastAPI, WebSocket
from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles

from aiohttp import ClientTimeout

import asyncio

timeout = ClientTimeout(total=2)


# STT_URL = "http://stt_vosk:5002/transcribe"
STT_URL = "http://stt_fastwhisper:5001/transcribe"
LLM_URL = "http://llm:5001/generate"
TTS_URL = "http://tts:5003/synthesize"


services = {
    # "STT (Vosk)": "http://stt_vosk:5002/health",
    "STT (FastWhisper)": "http://stt_fastwhisper:5001/health",
    "LLM": "http://llm:5001/health",
    "TTS": "http://tts:5003/health",
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
                        async with session.post(
                            url, data=payload, timeout=timeout
                        ) as resp:
                            results[name] = resp.status in (
                                200,
                                400,
                            )  # Accept 400 for empty input
                    elif "/generate" in url:
                        async with session.post(
                            url, json={"prompt": ""}, timeout=timeout
                        ) as resp:
                            results[name] = resp.status in (200, 400)
                    elif "/synthesize" in url:
                        async with session.post(
                            url, json={"text": ""}, timeout=timeout
                        ) as resp:
                            results[name] = resp.status in (200, 400)
                    else:
                        async with session.get(url, timeout=timeout) as resp:
                            results[name] = resp.status == 200
                except Exception:
                    results[name] = False
            else:
                results[name] = None  # Non-HTTP

    return results


# -------- HEALTHCHECK ------------


def save_as_wav(raw_path, sample_rate=16000, channels=1):
    with open(raw_path, "rb") as pcm_file:
        raw_data = pcm_file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        with wave.open(wav_file.name, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_data)

        return wav_file.name


def extract_text_from_stt(stt_raw):
    try:
        data = json.loads(stt_raw)
        return data.get("text", "").strip()
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


async def generate_response(prompt, retries=3):
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(LLM_URL, json={"prompt": prompt}) as resp:
                    text = await resp.text()
                    result = json.loads(text)
                    return result.get("response", "")
        except Exception as e:
            print(f"⚠️ LLM retry {attempt + 1}: {e}")
            await asyncio.sleep(2)
    return ""


async def synthesize_speech(text, out_path, retries=3):
    if not text.strip():
        print("❌ Empty or invalid text for TTS.")
        return False

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(TTS_URL, json={"text": text}) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"TTS failed: {await resp.text()}")
                    audio = await resp.read()
                    with open(out_path, "wb") as f:
                        f.write(audio)
                    return True
        except Exception as e:
            print(f"⚠️ TTS retry {attempt + 1}: {e}")
            await asyncio.sleep(2)
    return False


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    start_total = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        print(f"Receiving audio chunks to {tmp_audio.name}")
        while True:
            data = await websocket.receive_bytes()
            if data == b"\x00":
                print("🛑 End of stream signal received.")
                break
            tmp_audio.write(data)
        tmp_audio.flush()

    print("Audio received, running pipeline...")

    # Time STT
    stt_start = time.perf_counter()
    stt_result = await transcribe_audio(tmp_audio.name)
    stt_duration = time.perf_counter() - stt_start
    print(f"STT took {stt_duration:.2f} seconds")

    prompt = extract_text_from_stt(stt_result)

    # Filter out empty or short prompts
    if not prompt or len(prompt.strip()) < 5:
        print(f"⚠️ Ignored short or empty prompt: '{prompt}'")
        await websocket.send_bytes(b"")  # Send empty response to client
        await websocket.close()
        return

    # Time LLM
    llm_start = time.perf_counter()
    reply = await generate_response(prompt)
    llm_duration = time.perf_counter() - llm_start
    print(f"LLM took {llm_duration:.2f} seconds")

    # Time TTS
    tts_start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_audio:
        print(f"Synthesizing reply: {reply}")
        await synthesize_speech(reply, tts_audio.name)
        tts_duration = time.perf_counter() - tts_start
        print(f"TTS took {tts_duration:.2f} seconds")

        if not os.path.exists(tts_audio.name) or os.path.getsize(tts_audio.name) < 1000:
            print("❌ TTS output is empty or too small — skipping send.")
            await websocket.send_bytes(b"")
            await websocket.close()
            return

        print(f"Sending audio response: {tts_audio.name}")
        with open(tts_audio.name, "rb") as f:
            await websocket.send_bytes(f.read())

    total_duration = time.perf_counter() - start_total
    print(f"✅ Pipeline finished in {total_duration:.2f} seconds")
    await websocket.close()


# main
app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")

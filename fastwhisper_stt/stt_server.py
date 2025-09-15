from fastapi import FastAPI

from fastapi.responses import JSONResponse
from fastapi import File, UploadFile
import os
import wave
import tempfile
import logging
from faster_whisper import WhisperModel

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
model = WhisperModel("medium.en", device="cuda", compute_type="int8_float16")


@app.get("/health")
async def health():
    return "OK"


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_temp:
            input_pcm_path = wav_temp.name
            content = await audio.read()
            with wave.open(input_pcm_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(content)

        # Transcribe
        segments, _ = model.transcribe(input_pcm_path)
        text = " ".join([seg.text.strip() for seg in segments])
        logging.debug(f"Fast Whisper STT result: {text}")

        os.remove(input_pcm_path)
        return JSONResponse(content={"text": text})

    except Exception as e:
        logging.error(f"Fast Whisper STT Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

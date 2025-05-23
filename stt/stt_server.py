from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile
import os
import logging
import wave
from vosk import Model, KaldiRecognizer
from functools import lru_cache

app = FastAPI()

# Load model
# -------------------------------


@lru_cache()
def get_model():
    return Model("models/vosk-model-small-en-us-0.15")


# -------------------------------


@app.get("/health")
async def health():
    return "OK"


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    input_pcm_path = "temp_input.raw"
    output_wav_path = "temp_output.wav"

    try:
        with open(input_pcm_path, "wb") as f:
            f.write(await audio.read())

        file_size = os.path.getsize(input_pcm_path)
        logging.debug(f"Saved input audio as {input_pcm_path} ({file_size} bytes)")

        if file_size < 1000:
            logging.error("Audio file too small or empty.")
            return JSONResponse(
                content={"error": "Audio file is empty or corrupted"}, status_code=400
            )

        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            with open(input_pcm_path, "rb") as pcm:
                wf.writeframes(pcm.read())

        wf = wave.open(output_wav_path, "rb")
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getframerate() not in [8000, 16000]
        ):
            logging.error("Invalid WAV format")
            return JSONResponse(
                content={"error": "WAV must be mono PCM, 16-bit, 8kHz or 16kHz"},
                status_code=400,
            )

        rec = KaldiRecognizer(get_model(), wf.getframerate())
        result_text = ""

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result_text += rec.Result()

        result_text += rec.FinalResult()
        logging.debug("Final STT result: %s", result_text)
        wf.close()

        return JSONResponse(content={"text": result_text})

    except Exception as e:
        logging.error(f"STT Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        for path in [input_pcm_path, output_wav_path]:
            if os.path.exists(path):
                os.remove(path)

import os
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from TTS.api import TTS
import tempfile
import traceback

# Environment configuration
os.environ["NNPACK_DISABLE"] = "1"
os.environ["ATEN_DISABLE_NNPACK"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_USE_CUDA"] = "0"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/en/vctk/vits").to(device)


class TTSRequest(BaseModel):
    text: str
    speaker: str | None = None


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/synthesize")
def synthesize(request_data: TTSRequest):
    text = request_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        speaker = request_data.speaker or tts.speakers[0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            path = tmp_file.name

        tts.tts_to_file(text=text, speaker=speaker, file_path=path)

        return FileResponse(
            path,
            media_type="audio/wav",
            filename="speech.wav",
            background=lambda: os.remove(path),
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


print(">>> FastAPI TTS service is starting...")

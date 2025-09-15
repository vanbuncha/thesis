import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask  # <-- add this
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
        # Fallback to the first available speaker if none provided
        speaker = request_data.speaker or (
            tts.speakers[0] if hasattr(tts, "speakers") else None
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            path = tmp_file.name

        tts.tts_to_file(text=text, speaker=speaker, file_path=path)

        # Use BackgroundTask so Starlette can await it
        cleanup = BackgroundTask(os.remove, path)
        return FileResponse(
            path, media_type="audio/wav", filename="speech.wav", background=cleanup
        )

    except Exception as e:
        traceback.print_exc()
        # Best-effort cleanup if tts_to_file failed after creating path
        try:
            if "path" in locals() and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": str(e)})


print(">>> FastAPI TTS service is starting...")

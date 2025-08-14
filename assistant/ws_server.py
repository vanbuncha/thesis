import os
import json
import tempfile
import aiohttp
import time
import wave
import asyncio
import logging
from contextlib import contextmanager

from fastapi import FastAPI, WebSocket, Depends
from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles

from aiohttp import ClientTimeout

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session as DBSession

from db.models import User, Session as SessionModel, Interaction, Base
from db.database import get_db, SessionLocal, engine

# ---------------------------------------------------------
# Logging / timing utils
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assistant")


@contextmanager
def Timer(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        logger.info(f"[TIMER] {label}: {dur:.3f}s")


def now_ms() -> int:
    return int(time.time() * 1000)


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024**2):.2f} MB"


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
timeout = ClientTimeout(total=2)

STT_URL = os.getenv("STT_URL", "")
LLM_URL = os.getenv("LLM_URL", "")
TTS_URL = os.getenv("TTS_URL", "")

services = {
    "STT (FastWhisper)": os.getenv("SERVICE_STT", ""),
    "LLM": os.getenv("SERVICE_LLM", ""),
    "TTS": os.getenv("SERVICE_TTS", ""),
    "Ollama": os.getenv("SERVICE_OLLAMA", ""),
    "Database": os.getenv("SERVICE_DATABASE", ""),
}

app = FastAPI(title="Assistant API")
router = APIRouter()


# ---------------------------------------------------------
# Startup: wait for DB, then create tables once
# ---------------------------------------------------------
@app.on_event("startup")
def on_startup():
    with Timer("startup: wait_for_db"):
        retries = 10
        delay = 1.0
        for i in range(retries):
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                break
            except OperationalError as e:
                logger.warning(f"DB not ready ({i + 1}/{retries}): {e}")
                time.sleep(delay)
                delay = min(delay * 1.5, 5.0)
        else:
            raise RuntimeError("Database never became ready.")
    with Timer("startup: metadata.create_all"):
        Base.metadata.create_all(bind=engine)
    logger.info("Database connected and metadata created.")


# ---------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------
@router.get("/health")
async def health_check():
    results = {}
    async with aiohttp.ClientSession() as session:
        for name, url in services.items():
            if url.startswith("http"):
                try:
                    if "/transcribe" in url:
                        payload = aiohttp.FormData()
                        payload.add_field(
                            "audio", b"", filename="dummy.wav", content_type="audio/wav"
                        )
                        async with session.post(
                            url, data=payload, timeout=timeout
                        ) as resp:
                            results[name] = resp.status in (200, 400)
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
                except Exception as e:
                    logger.warning(f"health_check error for {name}: {e}")
                    results[name] = False
            else:
                results[name] = None  # Non-HTTP
    return results


# ---------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------
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


def extract_text_from_stt(stt_raw: str) -> str:
    try:
        data = json.loads(stt_raw)
        return data.get("text", "").strip()
    except Exception as e:
        logger.error(f"Failed to extract text from STT JSON: {e}")
        return ""


async def transcribe_audio(audio_file_path: str) -> tuple[str, int]:
    """
    Returns (text_json, http_status) or raises after retries.
    """
    for attempt in range(1, 6):
        try:
            async with aiohttp.ClientSession() as session:
                with open(audio_file_path, "rb") as f:
                    data = {"audio": f}
                    t0 = now_ms()
                    async with session.post(STT_URL, data=data) as resp:
                        body = await resp.text()
                        dt = now_ms() - t0
                        logger.info(
                            f"[STT] status={resp.status} latency={dt}ms bytes_in={fmt_bytes(os.path.getsize(audio_file_path))}"
                        )
                        return body, resp.status
        except aiohttp.ClientConnectorError as e:
            logger.warning(f"[STT] connect error, retry {attempt}/5: {e}")
            await asyncio.sleep(2)
    raise RuntimeError("Failed to connect to STT after several retries.")


async def generate_response(prompt: str, retries: int = 3) -> tuple[str, int]:
    """
    Returns (llm_text, http_status). On failure returns ("", status_or_0).
    """
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                t0 = now_ms()
                async with session.post(LLM_URL, json={"prompt": prompt}) as resp:
                    text_resp = await resp.text()
                    dt = now_ms() - t0
                    logger.info(
                        f"[LLM] status={resp.status} latency={dt}ms prompt_chars={len(prompt)}"
                    )
                    try:
                        result = json.loads(text_resp)
                        return result.get("response", ""), resp.status
                    except Exception as e:
                        logger.error(
                            f"[LLM] JSON parse error: {e} body[:400]={text_resp[:400]}"
                        )
                        return "", resp.status
        except Exception as e:
            logger.warning(f"[LLM] retry {attempt}/{retries}: {e}")
            await asyncio.sleep(2)
    return "", 0


async def synthesize_speech(
    text_in: str, out_path: str, retries: int = 3
) -> tuple[bool, int]:
    """
    Returns (ok, http_status). Writes audio to out_path when ok=True.
    """
    if not text_in.strip():
        logger.error("[TTS] Empty or invalid text.")
        return False, 0

    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                t0 = now_ms()
                async with session.post(TTS_URL, json={"text": text_in}) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        dt = now_ms() - t0
                        logger.error(
                            f"[TTS] status={resp.status} latency={dt}ms err={body[:200]}"
                        )
                        raise RuntimeError(f"TTS failed: {body}")
                    audio = await resp.read()
                    dt = now_ms() - t0
                    with open(out_path, "wb") as f:
                        f.write(audio)
                    logger.info(
                        f"[TTS] status=200 latency={dt}ms bytes_out={fmt_bytes(len(audio))}"
                    )
                    return True, 200
        except Exception as e:
            logger.warning(f"[TTS] retry {attempt}/{retries}: {e}")
            await asyncio.sleep(2)
    return False, 0


# ---------------------------------------------------------
# WebSocket entry point
# ---------------------------------------------------------
@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket, db: DBSession = Depends(get_db)):
    with Timer("ws:accept"):
        await websocket.accept()

    # 1) Identify user
    user_identifier = websocket.query_params.get("user", "anonymous")

    # 2) Ensure user exists
    with Timer("db:get_or_create_user"):
        user = db.query(User).filter_by(identifier=user_identifier).first()
        if not user:
            user = User(identifier=user_identifier)
            db.add(user)
            db.commit()
            db.refresh(user)

    # 3) Start session
    with Timer("db:start_session"):
        session_entry = SessionModel(user_id=user.id)
        db.add(session_entry)
        db.commit()
        db.refresh(session_entry)

    start_total = time.perf_counter()

    # Receive raw audio stream (ending with 0x00)
    bytes_received = 0
    recv_start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        logger.info(f"[WS] receiving audio chunks -> {tmp_audio.name}")
        while True:
            data = await websocket.receive_bytes()
            if data == b"\x00":
                logger.info("[WS] end-of-stream marker received")
                break
            tmp_audio.write(data)
            bytes_received += len(data)
        tmp_audio.flush()
    recv_duration = time.perf_counter() - recv_start
    logger.info(
        f"[WS] audio received: size={fmt_bytes(bytes_received)} time={recv_duration:.3f}s"
    )

    # STT
    with Timer("stage:STT"):
        stt_text, stt_status = await transcribe_audio(tmp_audio.name)
    prompt = extract_text_from_stt(stt_text)
    logger.info(f"[STT] extracted prompt chars={len(prompt)} status={stt_status}")

    # Filter out empty / short prompts
    if not prompt or len(prompt.strip()) < 5:
        logger.info(f"[FILTER] short/empty prompt ignored: '{prompt}'")
        with Timer("ws:send_empty_and_close"):
            await websocket.send_bytes(b"")
            await websocket.close()
        return

    # LLM
    with Timer("stage:LLM"):
        reply, llm_status = await generate_response(prompt)
    logger.info(f"[LLM] reply chars={len(reply)} status={llm_status}")

    # Log interaction
    with Timer("db:log_interaction"):
        try:
            interaction = Interaction(
                session_id=session_entry.id,
                user_input=prompt,
                llm_response=reply,
            )
            db.add(interaction)
            db.commit()
        except Exception as e:
            logger.error(f"[DB] logging failed: {e}")

    # TTS
    with Timer("stage:TTS"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_audio:
            logger.info(f"[TTS] synthesizing reply -> {tts_audio.name}")
            ok, tts_status = await synthesize_speech(reply, tts_audio.name)
            if (
                not ok
                or not os.path.exists(tts_audio.name)
                or os.path.getsize(tts_audio.name) < 1000
            ):
                logger.error(f"[TTS] output too small or failed; status={tts_status}")
                with Timer("ws:send_empty_and_close_after_tts_fail"):
                    await websocket.send_bytes(b"")
                    await websocket.close()
                return

            # Send audio back
            with Timer("ws:send_audio"):
                with open(tts_audio.name, "rb") as f:
                    payload = f.read()
                    await websocket.send_bytes(payload)
                    logger.info(f"[WS] sent audio bytes={fmt_bytes(len(payload))}")

    total_duration = time.perf_counter() - start_total
    logger.info(f"[PIPELINE] total: {total_duration:.3f}s user={user_identifier}")
    with Timer("ws:close"):
        await websocket.close()


# ---------------------------------------------------------
# Mount router and static files
# ---------------------------------------------------------
app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")

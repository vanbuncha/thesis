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

# -------- CSV logging helpers --------
import csv
from pathlib import Path
from datetime import datetime

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
# CSV logging helpers (MINIMAL EVAL INSTRUMENTATION)
# ---------------------------------------------------------
LOG_DIR = Path(os.getenv("LOG_DIR", "/app/logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILES = {
    "stt": LOG_DIR / "stt.csv",
    "llm": LOG_DIR / "llm.csv",
    "tts": LOG_DIR / "tts.csv",
    "pipeline": LOG_DIR / "pipeline.csv",
}

CSV_HEADERS = {
    "stt": [
        "ts_iso",
        "user",
        "session_id",
        "audio_bytes",
        "status",
        "latency_ms",
        "text_chars",
    ],
    "llm": [
        "ts_iso",
        "user",
        "session_id",
        "prompt_chars",
        "status",
        "latency_ms",
        "reply_chars",
    ],
    "tts": [
        "ts_iso",
        "user",
        "session_id",
        "text_chars",
        "status",
        "latency_ms",
        "audio_bytes",
    ],
    "pipeline": [
        "ts_iso",
        "user",
        "session_id",
        "bytes_in",
        "recv_ms",
        "stt_ms",
        "llm_ms",
        "tts_ms",
        "total_ms",
        "sent_audio_bytes",
        "result",  # "ok" | "empty" | "tts_fail" | "filter_short"
    ],
}


def _ensure_header(path: Path, header: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(header)


def write_csv(which: str, row: list):
    path = CSV_FILES[which]
    _ensure_header(path, CSV_HEADERS[which])
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(row)


def ts_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
# Short health-check timeout, but set proper timeouts for services below
timeout = ClientTimeout(total=2)

# Per-service timeouts for real calls (adjust if needed)
STT_TIMEOUT = aiohttp.ClientTimeout(total=15)
LLM_TIMEOUT = aiohttp.ClientTimeout(total=25)
TTS_TIMEOUT = aiohttp.ClientTimeout(total=15)

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
# Startup: validate config, wait for DB, create tables
# ---------------------------------------------------------
@app.on_event("startup")
def on_startup():
    # Fail fast if critical URLs are missing
    for k, v in {"STT_URL": STT_URL, "LLM_URL": LLM_URL, "TTS_URL": TTS_URL}.items():
        if not v:
            raise RuntimeError(f"{k} is not configured")
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


@app.on_event("startup")
def init_logs():
    for which, path in CSV_FILES.items():
        _ensure_header(path, CSV_HEADERS[which])


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
def save_as_wav(raw_pcm_path: str, sample_rate=16000, channels=1) -> str:
    with open(raw_pcm_path, "rb") as pcm_file:
        raw_data = pcm_file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        with wave.open(wav_file.name, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit PCM
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


# ---------------------------------------------------------
# Service calls (instrumented)
# ---------------------------------------------------------
async def transcribe_audio(
    audio_file_path: str,
    user_identifier: str = "anonymous",
    session_id: int | None = None,
) -> tuple[str, int, int]:
    """
    Returns (text_json, http_status, latency_ms) or raises after retries.
    """
    size_bytes = os.path.getsize(audio_file_path)
    form = aiohttp.FormData()
    form.add_field(
        "audio",
        open(audio_file_path, "rb"),
        filename="input.wav",
        content_type="audio/wav",
    )
    for attempt in range(1, 6):
        try:
            async with aiohttp.ClientSession() as session:
                t0 = now_ms()
                async with session.post(
                    STT_URL, data=form, timeout=STT_TIMEOUT
                ) as resp:
                    body = await resp.text()
                    dt = now_ms() - t0
                    logger.info(
                        f"[STT] status={resp.status} latency={dt}ms bytes_in={fmt_bytes(size_bytes)}"
                    )
                    try:
                        txt = json.loads(body).get("text", "") or ""
                        text_chars = len(txt)
                    except Exception:
                        text_chars = 0
                    write_csv(
                        "stt",
                        [
                            ts_iso(),
                            user_identifier,
                            session_id,
                            size_bytes,
                            resp.status,
                            dt,
                            text_chars,
                        ],
                    )
                    return body, resp.status, dt
        except aiohttp.ClientConnectorError as e:
            logger.warning(f"[STT] connect error, retry {attempt}/5: {e}")
            await asyncio.sleep(2)
    raise RuntimeError("Failed to connect to STT after several retries.")


async def generate_response(
    prompt: str, user_identifier: str, session_id: int | None = None, retries: int = 3
) -> tuple[str, int, int]:
    """
    Returns (llm_text, http_status, latency_ms). On failure returns ("", status_or_0, dt=0).
    """
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                t0 = now_ms()
                payload = {"prompt": prompt, "user": user_identifier}
                async with session.post(
                    LLM_URL, json=payload, timeout=LLM_TIMEOUT
                ) as resp:
                    text_resp = await resp.text()
                    dt = now_ms() - t0
                    logger.info(
                        f"[LLM] status={resp.status} latency={dt}ms prompt_chars={len(prompt)} user={user_identifier}"
                    )
                    reply = ""
                    try:
                        result = json.loads(text_resp)
                        reply = result.get("response", "") or ""
                    except Exception as e:
                        logger.error(
                            f"[LLM] JSON parse error: {e} body[:400]={text_resp[:400]}"
                        )
                    write_csv(
                        "llm",
                        [
                            ts_iso(),
                            user_identifier,
                            session_id,
                            len(prompt),
                            resp.status,
                            dt,
                            len(reply),
                        ],
                    )
                    return reply, resp.status, dt
        except Exception as e:
            logger.warning(f"[LLM] retry {attempt}/{retries}: {e}")
            await asyncio.sleep(2)
    write_csv("llm", [ts_iso(), user_identifier, session_id, len(prompt), 0, 0, 0])
    return "", 0, 0


async def synthesize_speech(
    text_in: str,
    out_path: str,
    user_identifier: str = "anonymous",
    session_id: int | None = None,
    retries: int = 3,
) -> tuple[bool, int, int, int]:
    """
    Returns (ok, http_status, latency_ms, audio_bytes). Writes audio to out_path when ok=True.
    """
    if not text_in.strip():
        logger.error("[TTS] Empty or invalid text.")
        write_csv("tts", [ts_iso(), user_identifier, session_id, 0, 0, 0, 0])
        return False, 0, 0, 0

    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                t0 = now_ms()
                async with session.post(
                    TTS_URL, json={"text": text_in}, timeout=TTS_TIMEOUT
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        dt = now_ms() - t0
                        logger.error(
                            f"[TTS] status={resp.status} latency={dt}ms err={body[:200]}"
                        )
                        write_csv(
                            "tts",
                            [
                                ts_iso(),
                                user_identifier,
                                session_id,
                                len(text_in),
                                resp.status,
                                dt,
                                0,
                            ],
                        )
                        raise RuntimeError(f"TTS failed: {body}")
                    audio = await resp.read()
                    dt = now_ms() - t0
                    with open(out_path, "wb") as f:
                        f.write(audio)
                    audio_bytes = len(audio)
                    logger.info(
                        f"[TTS] status=200 latency={dt}ms bytes_out={fmt_bytes(audio_bytes)}"
                    )
                    write_csv(
                        "tts",
                        [
                            ts_iso(),
                            user_identifier,
                            session_id,
                            len(text_in),
                            200,
                            dt,
                            audio_bytes,
                        ],
                    )
                    return True, 200, dt, audio_bytes
        except Exception as e:
            logger.warning(f"[TTS] retry {attempt}/{retries}: {e}")
            await asyncio.sleep(2)
    write_csv("tts", [ts_iso(), user_identifier, session_id, len(text_in), 0, 0, 0])
    return False, 0, 0, 0


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

    # Receive raw audio stream (ending with 0x00) -> buffer PCM, then normalize to WAV
    bytes_received = 0
    recv_start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as tmp_pcm:
        logger.info(f"[WS] receiving audio chunks -> {tmp_pcm.name}")
        try:
            while True:
                data = await websocket.receive_bytes()
                if data == b"\x00":
                    logger.info("[WS] end-of-stream marker received")
                    break
                tmp_pcm.write(data)
                bytes_received += len(data)
            tmp_pcm.flush()
        except Exception as e:
            logger.error(f"[WS] receive error: {e}")
            total_duration = time.perf_counter() - start_total
            write_csv(
                "pipeline",
                [
                    ts_iso(),
                    user_identifier,
                    session_entry.id,
                    bytes_received,
                    int((time.perf_counter() - recv_start) * 1000),
                    0,
                    0,
                    0,
                    int(total_duration * 1000),
                    0,
                    "empty",
                ],
            )
            await websocket.send_bytes(b"")
            await websocket.close()
            return

    recv_duration = time.perf_counter() - recv_start
    logger.info(
        f"[WS] audio received: size={fmt_bytes(bytes_received)} time={recv_duration:.3f}s"
    )

    # Normalize PCM -> WAV
    wav_path = save_as_wav(tmp_pcm.name, sample_rate=16000, channels=1)

    # STT
    with Timer("stage:STT"):
        try:
            stt_text, stt_status, stt_ms = await transcribe_audio(
                wav_path, user_identifier, session_entry.id
            )
        except Exception as e:
            logger.error(f"[STT] failure: {e}")
            total_duration = time.perf_counter() - start_total
            write_csv(
                "pipeline",
                [
                    ts_iso(),
                    user_identifier,
                    session_entry.id,
                    bytes_received,
                    int(recv_duration * 1000),
                    0,
                    0,
                    0,
                    int(total_duration * 1000),
                    0,
                    "empty",
                ],
            )
            await websocket.send_bytes(b"")
            await websocket.close()
            return

    prompt = extract_text_from_stt(stt_text)
    logger.info(f"[STT] extracted prompt chars={len(prompt)} status={stt_status}")

    # Filter out empty / short prompts
    if not prompt or len(prompt.strip()) < 5:
        logger.info(f"[FILTER] short/empty prompt ignored: '{prompt}'")
        total_duration = time.perf_counter() - start_total
        write_csv(
            "pipeline",
            [
                ts_iso(),
                user_identifier,
                session_entry.id,
                bytes_received,
                int(recv_duration * 1000),
                0,
                0,
                0,
                int(total_duration * 1000),
                0,
                "filter_short",
            ],
        )
        with Timer("ws:send_empty_and_close"):
            await websocket.send_bytes(b"")
            await websocket.close()
        return

    # LLM
    with Timer("stage:LLM"):
        reply, llm_status, llm_ms = await generate_response(
            prompt, user_identifier, session_entry.id
        )
    logger.info(f"[LLM] reply chars={len(reply)} status={llm_status}")

    # Log interaction to DB (optional but useful)
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
            ok, tts_status, tts_ms, sent_audio_bytes = await synthesize_speech(
                reply, tts_audio.name, user_identifier, session_entry.id
            )
            if (
                not ok
                or not os.path.exists(tts_audio.name)
                or os.path.getsize(tts_audio.name) < 1000
            ):
                logger.error(f"[TTS] output too small or failed; status={tts_status}")
                total_duration = time.perf_counter() - start_total
                write_csv(
                    "pipeline",
                    [
                        ts_iso(),
                        user_identifier,
                        session_entry.id,
                        bytes_received,
                        int(recv_duration * 1000),
                        stt_ms,
                        llm_ms,
                        tts_ms,
                        int(total_duration * 1000),
                        0,
                        "tts_fail",
                    ],
                )
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

    # Happy path: pipeline summary row
    total_duration = time.perf_counter() - start_total
    write_csv(
        "pipeline",
        [
            ts_iso(),
            user_identifier,
            session_entry.id,
            bytes_received,
            int(recv_duration * 1000),
            stt_ms,
            llm_ms,
            tts_ms,
            int(total_duration * 1000),
            sent_audio_bytes,
            "ok",
        ],
    )
    logger.info(f"[PIPELINE] total: {total_duration:.3f}s user={user_identifier}")
    with Timer("ws:close"):
        await websocket.close()


# ---------------------------------------------------------
# Mount router and static files
# ---------------------------------------------------------
app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")

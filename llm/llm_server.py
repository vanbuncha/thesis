# llm_server.py
import os
import json
import logging
import requests
from typing import List

from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import desc

from db.models import (
    Interaction,
    Session as SessionModel,
    User,
    UserProfile,
)
from db.database import get_db

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_server")

# ---------------------------------------------------------------------
# Environment / Config
# ---------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:12121/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")

# Short “hot” window for local coherence (entity memory supplies long-term facts)
HOT_TURNS = int(os.getenv("HOT_TURNS", "4"))

# Token budget controls (lightweight guardrails; entity memory is compact)
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "2600"))
RESERVED_COMPLETION_TOKENS = int(os.getenv("RESERVED_COMPLETION_TOKENS", "800"))

# HTTP timeout to Ollama (seconds)
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# ---------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a warm, empathetic, and thoughtful conversation partner trained to help users explore their emotional well-being. "
    "Your primary goal is to gently invite the user to share how they are feeling — emotionally, mentally, or physically. "
    "Be non-judgmental, gentle, and respectful of boundaries at all times. "
    "Use kind, open-ended questions. If the user shares something difficult, respond with compassion and validation. "
    "You are not a therapist — do not diagnose or offer medical advice. Focus on making the user feel heard, supported, and emotionally safe. "
    "If the user expresses thoughts of self-harm or deep distress, encourage them to reach out to a trusted friend, a mental health professional, or a crisis service. "
    "Mirror the user's tone appropriately. "
    "Keep responses warm and empathetic, but limit to 2–3 sentences unless more detail is essential."
    "In general, reply in a conversational manner with concise, human-like answers.\n"
)

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="LLM Service with Entity/Slot Memory")


# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    user: str | None = "anonymous"


# ---------------------------------------------------------------------
# Utilities: token estimate + formatting
# ---------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token + newlines)."""
    if not text:
        return 0
    return max(1, int(len(text) / 4) + text.count("\n"))


def format_turn(user_text: str, assistant_text: str) -> str:
    """Format one past turn into the prompt."""
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()
    return f"User: {user_text}\nAssistant: {assistant_text}\n"


# ---------------------------------------------------------------------
# Entity/slot memory helpers
# ---------------------------------------------------------------------
def get_or_create_profile(db: DBSession, user_id: int) -> UserProfile:
    prof = db.query(UserProfile).filter_by(user_id=user_id).first()
    if prof is None:
        prof = UserProfile(user_id=user_id, likes=[], dislikes=[], preferences={})
        db.add(prof)
        db.commit()
        db.refresh(prof)
    return prof


def render_profile_block(prof: UserProfile) -> str:
    """Compact profile rendering; include only non-empty fields."""
    lines = ["[User Profile]"]
    if prof.display_name:
        lines.append(f"name: {prof.display_name}")
    if prof.timezone:
        lines.append(f"timezone: {prof.timezone}")
    if prof.likes:
        lines.append("likes: " + ", ".join(map(str, prof.likes)))
    if prof.dislikes:
        lines.append("dislikes: " + ", ".join(map(str, prof.dislikes)))
    if prof.preferences:
        prefs = ", ".join(f"{k}: {v}" for k, v in prof.preferences.items())
        lines.append(f"preferences: {prefs}")
    if prof.notes:
        lines.append(f"notes: {prof.notes[:300]}")
    return "\n".join(lines) + "\n"


SLOT_EXTRACTION_SYSTEM = (
    "Extract user-specific facts from the conversation into strict JSON. "
    "Only include fields if you are confident. Respond with JSON ONLY, no prose."
)

SLOT_EXTRACTION_INSTRUCTIONS = """
Return a JSON object with any of these keys if present:

- display_name: string
- timezone: string (IANA like 'Europe/Amsterdam' if stated or implied)
- likes: array of strings (additive; e.g., ["tea","morning walks"])
- dislikes: array of strings
- preferences: object of key->value pairs (e.g., {"music":"jazz","reminders":"evening"})
- notes: short string with salient facts (concise)

If unsure, return {}.
"""


def extract_slots_with_llm(user_utt: str, assistant_utt: str) -> dict:
    prompt = (
        f"{SLOT_EXTRACTION_SYSTEM}\n\n"
        f"{SLOT_EXTRACTION_INSTRUCTIONS}\n\n"
        f"Conversation Turn:\n"
        f"User: {user_utt}\nAssistant: {assistant_utt}\n\n"
        f"JSON:"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        data = resp.json().get("response", "").strip()
        # Tolerate accidental code fences
        if data.startswith("```"):
            data = data.strip("`")
            if data.startswith("json"):
                data = data[4:].strip()
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        logger.warning(f"slot extraction failed: {e}")
        return {}


def merge_slots_into_profile(prof: UserProfile, slots: dict) -> None:
    import datetime

    # Scalars
    if isinstance(slots.get("display_name"), str) and slots["display_name"].strip():
        prof.display_name = slots["display_name"].strip()
    if isinstance(slots.get("timezone"), str) and slots["timezone"].strip():
        prof.timezone = slots["timezone"].strip()

    # Arrays (additive, dedup)
    if isinstance(slots.get("likes"), list):
        cur = set((prof.likes or []))
        for v in slots["likes"]:
            s = str(v).strip()
            if s:
                cur.add(s)
        prof.likes = list(cur)

    if isinstance(slots.get("dislikes"), list):
        cur = set((prof.dislikes or []))
        for v in slots["dislikes"]:
            s = str(v).strip()
            if s:
                cur.add(s)
        prof.dislikes = list(cur)

    # Dict (shallow merge)
    if isinstance(slots.get("preferences"), dict):
        prefs = dict(prof.preferences or {})
        for k, v in slots["preferences"].items():
            k2 = str(k).strip()
            if k2:
                prefs[k2] = v
        prof.preferences = prefs

    # Notes (append compactly with date)
    if isinstance(slots.get("notes"), str):
        new_note = slots["notes"].strip()
        if new_note:
            existing = prof.notes or ""
            stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d")
            snippet = f"[{stamp}] {new_note}"
            prof.notes = (existing + "\n" + snippet).strip() if existing else snippet


# ---------------------------------------------------------------------
# Small history window
# ---------------------------------------------------------------------
def build_hot_window(db: DBSession, user_identifier: str, turns: int) -> str:
    if turns <= 0:
        return ""
    recent: List[Interaction] = (
        db.query(Interaction)
        .join(SessionModel, Interaction.session_id == SessionModel.id)
        .join(User, SessionModel.user_id == User.id)
        .filter(User.identifier == user_identifier)
        .order_by(desc(Interaction.created_at))
        .limit(turns)
        .all()
    )
    recent.reverse()  # oldest -> newest
    return "".join(format_turn(i.user_input, i.llm_response) for i in recent)


# ---------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------
def assemble_prompt(profile_block: str, hot_window: str, current_user_text: str) -> str:
    budget = MAX_PROMPT_TOKENS - RESERVED_COMPLETION_TOKENS
    if budget < 400:
        budget = 400

    parts = [SYSTEM_PROMPT, profile_block]
    used = estimate_tokens(parts[0]) + estimate_tokens(parts[1])

    if hot_window:
        hw_cost = estimate_tokens(hot_window)
        if used + hw_cost <= budget:
            parts.append(hot_window)
            used += hw_cost
        else:
            # If hot window would overflow, drop
            logger.info("Dropping hot window due to token budget")

    current_block = f"User: {current_user_text.strip()}\nAssistant:"
    parts.append(current_block)
    return "".join(parts)


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/health")
def health() -> JSONResponse:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": "ping", "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        ok = resp.status_code == 200 and bool(resp.json().get("response"))
        return JSONResponse(
            {"status": "ok" if ok else "not-ready"}, status_code=200 if ok else 500
        )
    except Exception as e:
        logger.warning(f"LLM warmup failed in healthcheck: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/generate")
def generate(req: GenerateRequest, db: DBSession = Depends(get_db)) -> JSONResponse:
    try:
        user_identifier = (req.user or "anonymous").strip() or "anonymous"
        user_prompt = req.prompt or ""

        # Ensure user exists so we can attach a profile
        user = db.query(User).filter(User.identifier == user_identifier).first()
        if not user:
            user = User(identifier=user_identifier)
            db.add(user)
            db.commit()
            db.refresh(user)

        # Load/create profile and render block
        profile = get_or_create_profile(db, user.id)
        profile_block = render_profile_block(profile)

        # Short local context window
        hot_window = build_hot_window(db, user_identifier, HOT_TURNS)

        # Assemble final prompt
        full_prompt = assemble_prompt(profile_block, hot_window, user_prompt)
        logger.debug("Full prompt (truncated to 2k chars): %s", full_prompt[:2000])

        # Call Ollama
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": full_prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )

        if response.status_code != 200:
            logger.error("Ollama error %s: %s", response.status_code, response.text)
            return JSONResponse(
                {
                    "error": f"Ollama returned {response.status_code}",
                    "details": response.text,
                },
                status_code=502,
            )

        payload = response.json()
        final = payload.get("response", "")

        # Slot extraction & upsert (best-effort; non-fatal)
        try:
            slots = extract_slots_with_llm(user_prompt, final)
            if slots:
                merge_slots_into_profile(profile, slots)
                db.add(profile)
                db.commit()
        except Exception as e:
            logger.warning(f"slot merge/upsert failed: {e}")

        return JSONResponse({"response": final})

    except Exception as e:
        logger.exception("Error in /generate: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

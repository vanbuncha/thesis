import os
import re
import json
import mimetypes
import aiohttp
import asyncio
import tempfile
import traceback


from aiohttp import ClientSession
from nio import (
    AsyncClient,
    LoginResponse,
    RoomMessageAudio,
    InviteMemberEvent,
    RoomInviteError,
    UploadResponse,
    AsyncClientConfig,
)
from nio.responses import DownloadResponse
from nio.events.room_events import RoomMessage, RoomEncryptedAudio
from nio.crypto.attachments import decrypt_attachment


# --- Configuration ---
MATRIX_HOMESERVER = os.getenv("MATRIX_HOMESERVER", "https://matrix.org")
MATRIX_USER = os.getenv("MATRIX_USER")
MATRIX_PASSWORD = os.getenv("MATRIX_PASSWORD")

STT_URL = "http://stt:5002/transcribe"
LLM_URL = "http://llm:5001/generate"
TTS_URL = "http://tts:5003/synthesize"

STORE_PATH = "/data/matrix-store"
DEVICE_NAME = "e2ee-bot-device"


# --- Helper functions ---


# --- Matrix client setup ---
print("🚀 Bot starting...")

client = AsyncClient(
    MATRIX_HOMESERVER,
    MATRIX_USER,
    device_id="BOTDEVICE",
    store_path=STORE_PATH,
    config=AsyncClientConfig(encryption_enabled=True),
)
client.device_name = DEVICE_NAME


async def login():
    print("🔐 Attempting login...")
    resp = await client.login(MATRIX_PASSWORD)
    print("📡 Login response:", resp)
    print(f"🪪 Access token: {client.access_token}")

    if isinstance(resp, LoginResponse):
        print("✅ Logged in as", MATRIX_USER)
        client.access_token = resp.access_token
        await client.sync(full_state=True)
    else:
        print("❌ Login failed:", resp)
        exit(1)


async def invite_callback(room, event):
    print(f"📩 Invited to room {room.room_id}, joining...")
    try:
        await client.join(room.room_id)
        print(f"✅ Joined {room.room_id}")
    except RoomInviteError as e:
        print(f"❌ Failed to join: {e}")


def extract_text_from_stt(stt_raw):
    try:
        outer = json.loads(stt_raw)
        inner_json = outer.get("text", "")
        if not inner_json:
            return ""
        matches = re.findall(r'{\s*"text"\s*:\s*"([^"]*)"\s*}', inner_json)
        return " ".join(matches).strip()
    except Exception as e:
        print(f"❌ Failed to extract text: {e}")
        return ""


async def transcribe_audio(audio_file_path):
    for _ in range(5):
        try:
            async with ClientSession() as session:
                with open(audio_file_path, "rb") as f:
                    data = {"audio": f}
                    async with session.post(STT_URL, data=data) as resp:
                        return await resp.text()
        except aiohttp.ClientConnectorError:
            print("❌ STT service not ready yet, retrying...")
            await asyncio.sleep(2)
    raise RuntimeError("Failed to connect to STT after several retries.")


async def generate_response(prompt):
    async with ClientSession() as session:
        async with session.post(LLM_URL, json={"prompt": prompt}) as resp:
            result = await resp.json()
            print("raw response:", result)
            return result.get("response", "")


async def synthesize_speech(text, out_path):
    if not text.strip():
        print("⚠️ Empty or invalid text for TTS, skipping synthesis.")
        return

    async with ClientSession() as session:
        async with session.post(TTS_URL, json={"text": text}) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"TTS failed: {resp.status} — {error}")
            audio = await resp.read()
            with open(out_path, "wb") as f:
                f.write(audio)


async def send_audio_response(room_id, audio_path):
    mime_type, _ = mimetypes.guess_type(audio_path)
    try:
        with open(audio_path, "rb") as f:
            upload_response, encrypted = await client.upload(
                f,
                content_type=mime_type or "audio/wav",
                filename="response.wav",
                encrypt=client.rooms[room_id].encrypted,
            )

        if isinstance(upload_response, UploadResponse):
            content = {
                "body": "response.wav",
                "msgtype": "m.audio",
                "info": {
                    "mimetype": mime_type,
                    "size": os.path.getsize(audio_path),
                },
            }

            if encrypted:
                content["file"] = {
                    "url": upload_response.content_uri,
                    "key": {
                        "alg": encrypted["key"]["alg"],
                        "k": encrypted["key"]["k"],
                    },
                    "iv": encrypted["iv"],
                    "hashes": {
                        "sha256": encrypted["hashes"]["sha256"],
                    },
                }
            else:
                content["file"] = {
                    "url": upload_response.content_uri,
                    "key": {"alg": "A256CTR", "k": "dummy"},
                    "iv": "dummy",
                    "hashes": {"sha256": "dummy"},
                }

            await client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=True,
            )
            print("📤 Sent audio reply")
        else:
            print("❌ Upload failed:", upload_response)
    except Exception as e:
        print("❌ Upload error:", str(e))


async def handle_voice_event(room, audio_event):
    if audio_event.sender == client.user:
        print("🤖 Ignoring own audio message.")
        return

    mxc_url = audio_event.url
    if not mxc_url:
        print("⚠️ No MXC URL found in event.")
        return

    print(f"🎙️ Voice message in {room.display_name}")
    print(f"🔗 Raw MXC URL: {mxc_url}")

    try:
        response = await client.download(mxc_url)
        if not isinstance(response, DownloadResponse):
            print(f"❌ Matrix client download failed: {response}")
            return

        file_data = response.body

        # If it's an encrypted event and marked as decrypted, attempt to decrypt
        if isinstance(audio_event, RoomEncryptedAudio) and getattr(
            audio_event, "decrypted", False
        ):
            try:
                print("🔓 Decrypting downloaded attachment...")
                file_data = decrypt_attachment(
                    file_data,
                    audio_event.key["k"],
                    audio_event.hashes["sha256"],
                    audio_event.iv,
                )

            except Exception as ex:
                print("❌ Failed to decrypt attachment:", ex)
                traceback.print_exc()
                return

        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_audio:
            tmp_audio.write(file_data)
            tmp_audio.flush()

            file_size = os.path.getsize(tmp_audio.name)
            print(f"🔊 Downloaded audio ({file_size} bytes) → {tmp_audio.name}")

            if file_size < 1000:
                print("⚠️ Audio file too small, skipping.")
                return

            # Transcribe, Generate, Synthesize, Send
            stt_raw = await transcribe_audio(tmp_audio.name)
            print("📝 Raw STT:", stt_raw)

            clean_text = extract_text_from_stt(stt_raw)
            print("🧼 Extracted Text:", clean_text)

            if not clean_text:
                print("⚠️ No meaningful text extracted, skipping response.")
                return

            reply = await generate_response(clean_text)
            print(f"💬 LLM reply: {repr(reply)}")
            print("💬 LLM:", reply)

            if not reply.strip():
                reply = "Sorry, I didn't catch that. Could you please repeat?"

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as reply_audio:
                print(f"📢 Sending to TTS: {repr(reply)}")
                await synthesize_speech(reply, reply_audio.name)
                await send_audio_response(room.room_id, reply_audio.name)

    except Exception as e:
        print(f"❌ Error in voice handler: {e}")
        traceback.print_exc()


async def encrypted_message_callback(room, event):
    print(f"🔐 Received encrypted event from {event.sender} in {room.display_name}")
    print("🔎 Initial event state:", event)

    if getattr(event, "decrypted", False):
        print("🔓 Event already decrypted. Processing it as a plain audio message.")
        await handle_voice_event(room, event)
        return

    print(
        "⚠️ Event is not decrypted and cannot be processed with decrypt_event for this event type."
    )


async def plain_audio_callback(room, event):
    if event.sender == client.user:
        return
    if isinstance(event, RoomMessageAudio):
        print(f"🎧 Received plain audio message in {room.display_name}")
        await handle_voice_event(room, event)


async def debug_callback(room, event):
    print(f"🐛 Debug: Got event of type {type(event)} from {event.sender}")


async def main():
    await login()
    print("🔍 Rooms joined:")
    for room_id, room in client.rooms.items():
        print(f" - {room.display_name} (Encrypted: {room.encrypted})")

    client.add_event_callback(invite_callback, InviteMemberEvent)
    client.add_event_callback(encrypted_message_callback, RoomEncryptedAudio)
    client.add_event_callback(debug_callback, RoomMessage)
    client.add_event_callback(plain_audio_callback, RoomMessageAudio)

    await client.sync_forever(timeout=30000, full_state=True)


if __name__ == "__main__":
    asyncio.run(main())

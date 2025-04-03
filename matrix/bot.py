import os
from nio.responses import DownloadResponse
import json
import mimetypes
import aiohttp
import asyncio
import tempfile
from nio.events.room_events import RoomMessage
from nio import (
    AsyncClient,
    LoginResponse,
    RoomMessageAudio,
    InviteMemberEvent,
    RoomInviteError,
    UploadResponse,
    MegolmEvent,
)
from aiohttp import ClientSession
from nio import AsyncClientConfig
from nio.exceptions import EncryptionError

# --- Configuration ---
MATRIX_HOMESERVER = os.getenv("MATRIX_HOMESERVER", "https://matrix.org")
MATRIX_USER = os.getenv("MATRIX_USER")
MATRIX_PASSWORD = os.getenv("MATRIX_PASSWORD")

STT_URL = "http://stt:5002/transcribe"
LLM_URL = "http://llm:5001/generate"
TTS_URL = "http://tts:5003/synthesize"

STORE_PATH = "/data/matrix-store"
DEVICE_NAME = "e2ee-bot-device"

# --- Matrix setup ---
print("🚀 Bot starting...")

store_path = "/data/matrix-store"

client = AsyncClient(
    MATRIX_HOMESERVER,
    MATRIX_USER,
    device_id="BOTDEVICE",
    store_path=store_path,
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


# --- Join room if invited ---
async def invite_callback(room, event):
    print(f"📩 Invited to room {room.room_id}, joining...")
    try:
        await client.join(room.room_id)
        print(f"✅ Joined {room.room_id}")
    except RoomInviteError as e:
        print(f"❌ Failed to join: {e}")


# --- Download + Transcribe ---
async def download_media(url: str, dest: str):
    async with ClientSession() as session:
        async with session.get(url) as resp:
            with open(dest, "wb") as f:
                f.write(await resp.read())


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
    async with ClientSession() as session:
        async with session.post(LLM_URL, json={"prompt": prompt}) as resp:
            result = await resp.json()
            return result.get("response", "")


async def synthesize_speech(text, out_path):
    async with ClientSession() as session:
        async with session.post(TTS_URL, json={"text": text}) as resp:
            audio = await resp.read()
            with open(out_path, "wb") as f:
                f.write(audio)


# --- Respond with audio ---


async def send_audio_response(room_id, audio_path):
    mime_type, _ = mimetypes.guess_type(audio_path)

    try:
        with open(audio_path, "rb") as f:
            upload_response, _ = await client.upload(
                f,
                content_type=mime_type or "audio/wav",
                filename="response.wav",
            )

        if isinstance(upload_response, UploadResponse):
            content = {
                "body": "response.wav",
                "msgtype": "m.audio",
                "url": upload_response.content_uri,
                "info": {
                    "mimetype": mime_type,
                    "size": os.path.getsize(audio_path),
                },
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
    mxc_url = audio_event.url
    if not mxc_url:
        print("⚠️ No MXC URL found in event.")
        return

    print(f"🎙️ Voice message in {room.display_name}")
    print(f"🔗 Raw MXC URL: {mxc_url}")

    try:
        # Attempt to download using authenticated matrix-nio client
        response = await client.download(mxc_url)
        if not isinstance(response, DownloadResponse):
            print(f"❌ Matrix client download failed: {response}")
            return

        # Write audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_audio:
            tmp_audio.write(response.body)
            tmp_audio.flush()

            file_size = os.path.getsize(tmp_audio.name)
            print(f"🔊 Downloaded audio ({file_size} bytes) → {tmp_audio.name}")

            if file_size < 1000:
                print("⚠️ Downloaded audio file is too small, skipping.")
                return

            # Run STT → LLM → TTS flow
            try:
                text = await transcribe_audio(tmp_audio.name)
                print("📝 Transcribed:", text)

                reply = await generate_response(text)
                print("💬 LLM:", reply)

                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as reply_audio:
                    await synthesize_speech(reply, reply_audio.name)
                    await send_audio_response(room.room_id, reply_audio.name)

            except Exception as e:
                print(f"❌ Failed to handle voice message: {e}")

    except Exception as e:
        print(f"❌ Unexpected error while downloading audio: {e}")


# --- Main event handler ---


async def encrypted_message_callback(room, event):
    print(f"🔐 Received MegolmEvent from {event.sender} in {room.display_name}")
    if not isinstance(event, MegolmEvent):
        return

    print(f"📥 Received encrypted event in {room.display_name} from {event.sender}")

    try:
        # Try to decrypt and handle voice messages
        if isinstance(event.decrypted, RoomMessageAudio):
            print("🔓 Decrypted event is a voice message (RoomMessageAudio)")
            await handle_voice_event(room, event.decrypted)
        else:
            print(f"ℹ️ Decrypted event is not a voice message: {type(event.decrypted)}")

    except EncryptionError as e:
        print(f"⚠️ Encryption error: {e}. Trying key query...")

        sender = event.sender
        print(f"🔑 Performing key query for {sender}")
        await client.keys_query([sender])

        await asyncio.sleep(1)  # give time for key sync

        try:
            await client.decrypt_event(event)
            if isinstance(event.decrypted, RoomMessageAudio):
                print("🔁 Retry successful: voice message decrypted")
                await handle_voice_event(room, event.decrypted)
            else:
                print(
                    f"⚠️ Retry decrypted to unsupported event type: {type(event.decrypted)}"
                )
        except Exception as e2:
            print(f"❌ Retry decryption failed: {e2}")


async def plain_audio_callback(room, event):
    if event.sender == client.user:  # avoid replying to self
        return

    if isinstance(event, RoomMessageAudio):
        print(f"🎧 Received plain audio message in {room.display_name}")
        await handle_voice_event(room, event)


async def debug_callback(room, event):
    print(f"🐛 Debug: Got event of type {type(event)} from {event.sender}")


# --- Run bot ---
async def main():
    await login()
    print("🔍 Joined rooms and encryption state:")
    for room_id, room in client.rooms.items():
        print(f" - {room.display_name} (Encrypted: {room.encrypted})")

    client.add_event_callback(invite_callback, InviteMemberEvent)
    client.add_event_callback(encrypted_message_callback, MegolmEvent)
    client.add_event_callback(debug_callback, RoomMessage)
    client.add_event_callback(plain_audio_callback, RoomMessageAudio)

    await client.sync_forever(timeout=30000, full_state=True)


if __name__ == "__main__":
    asyncio.run(main())

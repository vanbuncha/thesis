import os
import asyncio
import pyaudio
import wave
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from pvporcupine import create as create_porcupine
from nio import AsyncClient, RoomMessageAudio, UploadResponse

# Load environment
load_dotenv("../.env.client")
HOMESERVER = os.getenv("MATRIX_HOMESERVER")
USER = os.getenv("MATRIX_USER")
PASSWORD = os.getenv("MATRIX_PASSWORD")
ROOM_ID = os.getenv("MATRIX_ROOM_ID")
WAKE_WORD_PATH = os.getenv("WAKE_WORD_PATH")
MODEL_PATH = os.getenv("PORCUPINE_MODEL_PATH")

client = AsyncClient(HOMESERVER, USER)


def record_audio(filename="input.wav", duration=5):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
    )
    frames = []
    print("🎙️ Recording...")
    for _ in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("🛑 Done recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b"".join(frames))
    wf.close()


async def upload_audio(filepath):
    mime_type = "audio/wav"
    with open(filepath, "rb") as f:
        resp = await client.upload(f, content_type=mime_type, filename="question.wav")
    if isinstance(resp, UploadResponse):
        content = {
            "body": "question.wav",
            "msgtype": "m.audio",
            "url": resp.content_uri,
            "info": {"mimetype": mime_type},
        }
        await client.room_send(ROOM_ID, message_type="m.room.message", content=content)
        print("✅ Audio sent.")
    else:
        print("❌ Upload failed.")


def play_audio_file(wav_path):
    print("🔊 Playing response...")
    data, fs = sf.read(wav_path, dtype="float32")
    sd.play(data, fs)
    sd.wait()


async def listen_for_reply():
    async def callback(room, event):
        if isinstance(event, RoomMessageAudio) and event.sender != client.user:
            print("📥 Received reply.")
            mxc_url = event.url
            resp = await client.download(mxc_url)
            with open("response.wav", "wb") as f:
                f.write(resp.body)
            play_audio_file("response.wav")

    client.add_event_callback(callback, RoomMessageAudio)


async def main():
    await client.login(PASSWORD)
    await listen_for_reply()

    porcupine = create_porcupine(
        keyword_paths=[WAKE_WORD_PATH],
        model_path=MODEL_PATH,
        sensitivities=[0.6],
    )
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )

    print("👂 Waiting for wake word...")
    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = [
            int.from_bytes(pcm[i : i + 2], "little", signed=True)
            for i in range(0, len(pcm), 2)
        ]
        if porcupine.process(pcm) >= 0:
            print("🟢 Wake word detected!")
            record_audio("input.wav", duration=5)
            await upload_audio("input.wav")

        await client.sync(timeout=3000)


if __name__ == "__main__":
    asyncio.run(main())

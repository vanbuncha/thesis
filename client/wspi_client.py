import asyncio
import websockets
import sounddevice as sd
import numpy as np
import simpleaudio as sa
import wave
import tempfile

WS_URL = "ws://localhost:8000/ws/audio"  # Replace with your server's IP if remote
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)


def record_audio_chunk():
    audio = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    sd.wait()
    return audio.tobytes()


def play_audio_bytes(audio_bytes):
    if len(audio_bytes) < 100:
        print("❌ No audio received or invalid WAV format.")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        try:
            with wave.open(f.name, "rb") as wf:
                audio_data = wf.readframes(wf.getnframes())
                play_obj = sa.play_buffer(
                    audio_data, wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
                )
                play_obj.wait_done()
        except wave.Error as e:
            print(f"❌ Failed to play audio: {e}")


async def stream_audio():
    async with websockets.connect(WS_URL, max_size=2_000_000) as websocket:
        print("🎙️ Recording... Speak now.")

        for _ in range(6):  # ~3 seconds of audio
            chunk = record_audio_chunk()
            await websocket.send(chunk)

        await websocket.send(b"\x00")  # end of stream signal
        print("📤 Audio sent. Waiting for response...")

        response_audio = await websocket.recv()
        print(
            f"🔊 Playing response...\n🔍 Received {len(response_audio)} bytes from server"
        )
        play_audio_bytes(response_audio)


if __name__ == "__main__":
    asyncio.run(stream_audio())

import asyncio
import websockets
import sounddevice as sd
import simpleaudio as sa
import wave
import tempfile

WS_URL = "ws://localhost:8000/ws/audio"
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


async def stream_audio_from_file(file_path):
    async with websockets.connect(WS_URL, max_size=2_000_000) as websocket:
        print(f"📁 Sending audio from file: {file_path}")

        with wave.open(file_path, "rb") as wf:
            while True:
                frames = wf.readframes(CHUNK_SIZE)
                if not frames:
                    break
                await websocket.send(frames)

        await websocket.send(b"\x00")  # end of stream signal
        print("📤 Audio sent. Waiting for response...")

        response_audio = await websocket.recv()
        print(
            f"🔊 Playing response...\n🔍 Received {len(response_audio)} bytes from server"
        )
        play_audio_bytes(response_audio)


#     asyncio.run(stream_audio())

if __name__ == "__main__":
    test_wav = "voice.wav"
    #     asyncio.run(stream_audio())
    asyncio.run(stream_audio_from_file(test_wav))

import asyncio
import websockets
import sounddevice as sd
import simpleaudio as sa
import wave
import tempfile
import pvporcupine
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

WS_URL = "ws://localhost:8000/ws/audio"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Path to your wake word file
WAKE_WORD_PATH = "hello-friend.ppn"


def play_audio_bytes(audio_bytes):
    if len(audio_bytes) < 100:
        print("❌ No audio received or invalid WAV format.")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()

        try:
            with wave.open(f.name, "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

                dtype = "int16" if sampwidth == 2 else "int8"
                audio_np = np.frombuffer(frames, dtype=dtype)

                if channels == 2:
                    audio_np = np.reshape(audio_np, (-1, 2))

                sd.play(audio_np, samplerate=sample_rate)
                sd.wait()

        except wave.Error as e:
            print(f"❌ Failed to play audio: {e}")


async def stream_audio():
    async with websockets.connect(WS_URL, max_size=10_000_000) as websocket:
        print("🎙️ Recording... Speak now.")

        for _ in range(6):  # ~3 seconds of audio
            audio = sd.rec(
                CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16"
            )
            sd.wait()
            await websocket.send(audio.tobytes())

        await websocket.send(b"\x00")  # end of stream signal
        print("📤 Audio sent. Waiting for response...")

        response_audio = await websocket.recv()
        print(
            f"🔊 Playing response...\n🔍 Received {len(response_audio)} bytes from server"
        )
        play_audio_bytes(response_audio)


def listen_for_wake_word():
    print("🎤 Passive mode: Listening for wake word...")

    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY, keyword_paths=[WAKE_WORD_PATH]
    )

    try:
        with sd.RawInputStream(
            samplerate=porcupine.sample_rate,
            blocksize=porcupine.frame_length,
            dtype="int16",
            channels=1,
        ) as stream:
            while True:
                raw_pcm = stream.read(porcupine.frame_length)[
                    0
                ]  # this is already bytes-like
                pcm = np.frombuffer(raw_pcm, dtype=np.int16).tolist()

                result = porcupine.process(pcm)
                if result >= 0:
                    print("🔔 Wake word 'hello-friend' detected!")
                    return
    finally:
        porcupine.delete()


async def stream_audio_from_file(file_path):
    async with websockets.connect(WS_URL, max_size=2_000_000) as websocket:
        print(f"📁 Sending audio from file: {file_path}")

        with wave.open(file_path, "rb") as wf:
            while True:
                frames = wf.readframes(CHUNK_SIZE)
                if not frames:
                    break
                await websocket.send(frames)

        await websocket.send(b"\x00")
        response_audio = await websocket.recv()
        print(f"🔊 Playing response...\n🔍 Received {len(response_audio)} bytes")
        play_audio_bytes(response_audio)


async def main():
    while True:
        listen_for_wake_word()
        # await stream_audio_from_file("voice.wav")
        await stream_audio()


if __name__ == "__main__":
    asyncio.run(main())

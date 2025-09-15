#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv
from sounddevice import PortAudioError
import wave
import tempfile
import pvporcupine
import torch
from silero_vad import load_silero_vad


# --------------------------
# Config
# --------------------------

# Load secrets
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

# WS server
BASE_WS_URL = os.getenv("BASE_WS_URL", "ws://localhost:8000/ws/audio")
USER_IDENTIFIER = os.getenv("USER_IDENTIFIER", "pi-1235")
URI = f"{BASE_WS_URL}?user={USER_IDENTIFIER}"

# STT server expects 16k mono int16 PCM
SAMPLE_RATE = 16000
CHANNELS = 1

# Chunking (half-second frames at 16k = 8,000 samples/frame)
CHUNK_DURATION = 0.5
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Active conversation behavior
ACTIVE_TIMEOUT = 15  # seconds
TOTAL_RECORD_SECONDS = 5.0  # total audio to send per request
TARGET_FRAMES = int(round(TOTAL_RECORD_SECONDS / CHUNK_DURATION))  # e.g., 3.0 / 0.5 = 6

# Porcupine keyword
WAKE_WORD_PATH = os.getenv(
    "WAKE_WORD_PATH", "hello-friend-linux.ppn"
)  # Linux x86_64 .ppn

# Voice activation detection parameters
SILENCE_TIMEOUT = 1.0
MAX_RECORD_SECONDS = 20.0
MIN_AUDIO_LENGTH = 8000

# Silero VAD model
torch.set_num_threads(1)
vad_model = load_silero_vad()


# --------------------------
# Audio helpers
# --------------------------


def play_beep(beep_path="beep.wav"):
    try:
        with open(beep_path, "rb") as f:
            play_audio_bytes(f.read())
    except Exception as e:
        print(f"Failed to play beep: {e}")


def _resample_block_linear_1d(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resample fallback for 1-D float32 array in [-1, 1]."""
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_in = x.shape[0]
    n_out = int(round(n_in * sr_out / sr_in))
    t_in = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    y = np.interp(t_out, t_in, x.astype(np.float32))
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32, copy=False)


def resample_float_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float32 [-1,1], prefer scipy.signal.resample_poly, fallback to linear."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        g = math.gcd(sr_out, sr_in)
        up, down = sr_out // g, sr_in // g
        y = resample_poly(x.astype(np.float32), up, down)
        y = np.clip(y, -1.0, 1.0)
        return y.astype(np.float32, copy=False)
    except Exception:
        return _resample_block_linear_1d(x, sr_in, sr_out)


def resample_int16_to_16k(
    x_i16: np.ndarray, sr_in: int, sr_out: int = 16000
) -> np.ndarray:
    """Resample mono int16 at sr_in -> int16 at 16 kHz."""
    if sr_in == sr_out:
        return x_i16.astype(np.int16, copy=False)
    # convert to float, resample, back to int16
    x = x_i16.astype(np.float32) / 32768.0
    y = resample_float_mono(x, sr_in, sr_out)
    y = np.clip(y, -1.0, 1.0)
    return (y * 32768.0).astype(np.int16, copy=False)


# --------------------------
# Playback
# --------------------------


def play_audio_bytes(audio_bytes: bytes) -> None:
    """Always resample to the output device's default samplerate for clean playback."""
    if not audio_bytes or len(audio_bytes) < 100:
        print("No audio received or invalid WAV format.")
        return

    def read_wav_bytes_to_array(b: bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b)
            f.flush()
            path = f.name
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        # int8/16 -> float32 [-1,1], else try float32 direct
        if sw == 2:
            x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 1:
            x = (
                np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
            ) / 128.0
        else:
            x = np.frombuffer(frames, dtype=np.float32)

        if ch > 1:
            # mixdown to mono
            x = x.reshape(-1, ch).mean(axis=1)
        else:
            x = x.reshape(-1)
        return x.astype(np.float32, copy=False), int(sr)

    x, sr_in = read_wav_bytes_to_array(audio_bytes)

    try:
        sr_out = int(sd.query_devices(None, "output")["default_samplerate"])
    except Exception:
        sr_out = 48000  # safe default on Linux

    y = resample_float_mono(x, sr_in, sr_out)

    try:
        sd.play(y, samplerate=sr_out, blocking=True)
        sd.wait()
    except PortAudioError as e:
        print(f"Output device error: {e}")


# --------------------------
# Capture & send (active)
# --------------------------


async def stream_audio() -> bool:
    """Record and send audio using VAD, return True if meaningful response received."""
    try:
        play_beep()
    except Exception as e:
        print(f"Failed to play beep: {e}")

    try:
        async with websockets.connect(URI, max_size=10_000_000) as websocket:
            print("Recording (VAD enabled)...")

            dinfo = sd.query_devices(None, "input")
            sr_dev = int(dinfo.get("default_samplerate") or 48000)
            blocksize = max(512, int(sr_dev * 0.05))

            audio_buffer = deque()
            silence_start = None
            start_time = time.time()

            with sd.RawInputStream(
                samplerate=sr_dev,
                blocksize=blocksize,
                dtype="int16",
                channels=1,
            ) as stream:
                while True:
                    raw = stream.read(blocksize)[0]
                    x = np.frombuffer(raw, dtype=np.int16)
                    y = resample_int16_to_16k(x, sr_dev, SAMPLE_RATE)

                    # Get VAD probability
                    # Only pass 512 samples to the VAD model
                    if len(y) >= 512:
                        y_vad = y[-512:] / 32768.0
                        y_vad = np.clip(y_vad, -1.0, 1.0).astype(np.float32)

                        speech_prob = vad_model(
                            torch.from_numpy(y_vad.astype(np.float32)), SAMPLE_RATE
                        ).item()
                        is_speech = speech_prob > 0.3
                        print(f"Speech prob: {speech_prob:.2f}")
                    else:
                        is_speech = False  # too short, assume no speech

                    if is_speech:
                        silence_start = None
                        audio_buffer.extend(y.tolist())
                    else:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_TIMEOUT:
                            print(" Silence detected, stopping recording.")
                            break

                    if time.time() - start_time > MAX_RECORD_SECONDS:
                        print("Max recording time reached.")
                        break

            if len(audio_buffer) < MIN_AUDIO_LENGTH:
                print(" Too little speech detected.")
                return False

            print(" Sending audio to server...")

            # Send chunks in 0.5s (8000-sample) frames
            fifo = deque(audio_buffer)
            while len(fifo) >= CHUNK_SIZE:
                chunk = [fifo.popleft() for _ in range(CHUNK_SIZE)]
                await websocket.send(np.asarray(chunk, dtype=np.int16).tobytes())

            await websocket.send(b"\x00")  # end of audio signal

            print("Awaiting response from server...")
            response_audio = await websocket.recv()

            if response_audio and len(response_audio) > 100:
                print(f"Playing response ({len(response_audio)} bytes)")
                play_audio_bytes(response_audio)
                return True
            else:
                print("No audio received or invalid WAV format.")
                return False

    except PortAudioError as e:
        print(f" Mic error: {e}")
        return False
    except Exception as e:
        print(f" General error: {e}")
        return False


# --------------------------
# File mode (for testing)
# --------------------------


async def stream_audio_from_file(file_path: str) -> None:
    """Send WAV frames to server (downsample to 16k if needed)."""
    async with websockets.connect(URI, max_size=10_000_000) as websocket:
        print(f"🔊 Sending audio from file: {file_path}")

        with wave.open(file_path, "rb") as wf:
            sr_in = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()

            # read in blocks ~ CHUNK_DURATION
            frames_per_read = int(round(CHUNK_DURATION * sr_in))
            while True:
                frames = wf.readframes(frames_per_read)
                if not frames:
                    break

                # to mono int16 at sr_in
                if sw == 2:
                    x = np.frombuffer(frames, dtype=np.int16)
                else:
                    # normalize to int16 if needed
                    x = (
                        np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128
                    ) * 256

                if ch > 1:
                    x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)

                y16 = resample_int16_to_16k(x, sr_in, SAMPLE_RATE)

                # send in exact CHUNK_SIZE blocks
                fifo = deque(y16.tolist())
                while len(fifo) >= CHUNK_SIZE:
                    chunk = [fifo.popleft() for _ in range(CHUNK_SIZE)]
                    await websocket.send(np.asarray(chunk, dtype=np.int16).tobytes())

        await websocket.send(b"\x00")
        print("Audio sent. Waiting for response...")

        response_audio = await websocket.recv()
        print(f"Playing response... Received {len(response_audio)} bytes")
        play_audio_bytes(response_audio)


# --------------------------
# Wake word loop (Porcupine @ 16k)
# --------------------------


def listen_for_wake_word() -> None:
    """Listen at device rate, resample to 16k, feed exact Porcupine frames."""
    print("Passive mode: Listening for wake word...")

    # Create porcupine (expects 16 kHz, 16-bit, mono)
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keyword_paths=[WAKE_WORD_PATH],
    )

    dev_info = sd.query_devices(None, "input")
    sr_dev = int(dev_info.get("default_samplerate") or 48000)  # e.g., 44100 or 48000
    sr_pv = porcupine.sample_rate  # usually 16000
    frame_pv = porcupine.frame_length  # e.g., 512 samples at 16k
    block_dev = max(128, int(round(frame_pv * sr_dev / sr_pv)))

    buf = deque()

    try:
        with sd.RawInputStream(
            samplerate=sr_dev,
            blocksize=block_dev,
            dtype="int16",
            channels=1,
        ) as stream:
            while True:
                raw = stream.read(block_dev)[0]
                x_i16 = np.frombuffer(raw, dtype=np.int16)
                y_i16 = resample_int16_to_16k(x_i16, sr_dev, sr_pv)

                buf.extend(y_i16.tolist())
                while len(buf) >= frame_pv:
                    frame = [buf.popleft() for _ in range(frame_pv)]
                    result = porcupine.process(frame)
                    if result >= 0:
                        print("Wake word detected!")
                        return
    finally:
        porcupine.delete()


# --------------------------
# Device selection
# --------------------------


def prefer_pulse_devices() -> None:
    """Prefer PulseAudio/PipeWire devices if available (helps with resampling)."""
    try:
        devs = sd.query_devices()
        # input
        pulse_in = next(
            (
                i
                for i, d in enumerate(devs)
                if "pulse" in d["name"].lower() and d["max_input_channels"] > 0
            ),
            None,
        )
        # output
        pulse_out = next(
            (
                i
                for i, d in enumerate(devs)
                if "pulse" in d["name"].lower() and d["max_output_channels"] > 0
            ),
            None,
        )
        current = sd.default.device
        in_idx = current[0] if isinstance(current, tuple) else None
        out_idx = current[1] if isinstance(current, tuple) else None
        if pulse_in is not None:
            in_idx = pulse_in
        if pulse_out is not None:
            out_idx = pulse_out
        if in_idx is not None or out_idx is not None:
            sd.default.device = (in_idx, out_idx)
            # print(f"Using devices (in,out): {sd.default.device}")
    except Exception:
        pass


# --------------------------
# Main loop
# --------------------------


async def await_active_conversation():
    """Continue conversation turns until timeout from last interaction."""
    print("🟢 Active mode: Speak freely (timeout after inactivity).")
    last_interaction = asyncio.get_event_loop().time()

    while True:
        interaction_occurred = await stream_audio()
        now = asyncio.get_event_loop().time()

        if interaction_occurred:
            last_interaction = now
        elif now - last_interaction > ACTIVE_TIMEOUT:
            print("⏱️ Inactive for too long. Returning to passive mode.")
            break
        else:
            # brief pause to avoid spamming the server
            await asyncio.sleep(0.5)


async def main():
    while True:
        listen_for_wake_word()
        await await_active_conversation()


if __name__ == "__main__":
    print(USER_IDENTIFIER)
    # asyncio.run(stream_audio_from_file("voice.wav"))
    asyncio.run(main())

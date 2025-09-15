import numpy as np
import wave

# Beep parameters
duration_sec = 0.3
sample_rate = 16000
frequency = 440.0
amplitude = 0.5

# Generate tone
t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
tone = amplitude * np.sin(2 * np.pi * frequency * t)

# Convert to 16-bit PCM
pcm_data = (tone * 32767).astype(np.int16)

# Save to WAV file
file_path = "beep.wav"
with wave.open(file_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_data.tobytes())

print(f"Beep saved to: {file_path}")

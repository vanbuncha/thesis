from flask import Flask, request, jsonify
import os
import wave
import tempfile
import logging
from faster_whisper import WhisperModel

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load Whisper model (use 'base' or 'small' for lower resource use)
model = WhisperModel("base.en", compute_type="int8")


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        audio_file = request.files["audio"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_temp:
            input_pcm_path = wav_temp.name
            with wave.open(input_pcm_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_file.read())

        segments, _ = model.transcribe(input_pcm_path)

        text = " ".join([seg.text.strip() for seg in segments])
        logging.debug(f"Fast Whisper STT result: {text}")

        os.remove(input_pcm_path)
        return jsonify({"text": text})

    except Exception as e:
        logging.error(f"Fast Whisper STT Error: {e}")
        return jsonify({"error": str(e)}), 500

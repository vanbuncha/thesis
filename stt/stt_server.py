from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
import wave
import os
import logging
import subprocess
# from pydub import AudioSegment


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load STT model
model = Model("/app/models/vosk-model-small-en-us-0.15")  # Path to Vosk model


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    input_path = "temp_input.ogg"
    output_path = "temp.wav"

    try:
        audio_file = request.files["audio"]
        audio_file.save(input_path)
        file_size = os.path.getsize(input_path)
        logging.debug(f"Saved input audio as {input_path} ({file_size} bytes)")

        if file_size < 1000:
            logging.error("Audio file too small or empty.")
            return jsonify({"error": "Audio file is empty or corrupted"}), 400

        # Optional: inspect header
        with open(input_path, "rb") as f:
            header = f.read(4)
            if header != b"OggS":
                logging.error(f"Invalid file header: {header}")
                return jsonify(
                    {"error": "File does not appear to be a valid OGG audio file"}
                ), 400

        # Convert to WAV using ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            logging.error("ffmpeg failed: %s", result.stderr.decode())
            return jsonify(
                {"error": f"Could not decode audio: {result.stderr.decode()}"}
            ), 400

        # Transcribe with Vosk
        wf = wave.open(output_path, "rb")
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getframerate() not in [8000, 16000]
        ):
            logging.error("Invalid WAV format")
            return jsonify(
                {"error": "WAV must be mono PCM, 16-bit, 8kHz or 16kHz"}
            ), 400

        rec = KaldiRecognizer(model, wf.getframerate())
        result_text = ""

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result_text += rec.Result()

        result_text += rec.FinalResult()
        logging.debug("Final STT result: %s", result_text)
        wf.close()

        return jsonify({"text": result_text})

    except Exception as e:
        logging.error(f"STT Error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Always clean up
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)

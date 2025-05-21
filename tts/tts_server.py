import torch
from flask import Flask, request, jsonify, send_file
from TTS.api import TTS
import traceback
import tempfile

app = Flask(__name__)

# Detect if GPU is available
use_gpu = torch.backends.mps.is_available()

# Load Coqui TTS model with GPU acceleration
tts = TTS(
    model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False
)


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


@app.route("/synthesize", methods=["POST"])
def text_to_speech():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            path = tmp_file.name
            tts.tts_to_file(text=text, file_path=path)

        return send_file(path, mimetype="audio/wav")

    except Exception as e:
        traceback.print_exc()
        app.logger.error("Exception in TTS:", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)

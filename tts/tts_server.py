import os
import torch
from flask import Flask, request, jsonify, send_file
from TTS.api import TTS
import traceback
import tempfile

os.environ["TORCH_USE_CUDA"] = "0"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

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

            try:
                return send_file(path, mimetype="audio/wav")
            finally:
                os.remove(path)

    except Exception as e:
        traceback.print_exc()
        app.logger.error("Exception in TTS:", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)

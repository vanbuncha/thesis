import torch
from flask import Flask, request, jsonify, send_file
from TTS.api import TTS

app = Flask(__name__)

# Detect if GPU is available
use_gpu = torch.backends.mps.is_available()

# Load Coqui TTS model with GPU acceleration
MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
tts = TTS(MODEL_NAME, gpu=use_gpu)


@app.route("/tts", methods=["POST"])
def text_to_speech():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        output_path = "output.wav"
        tts.tts_to_file(text=text, file_path=output_path)

        return send_file(output_path, mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)

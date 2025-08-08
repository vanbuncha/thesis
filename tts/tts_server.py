import os
from flask import Flask, request, jsonify, send_file
import torch
from TTS.api import TTS
import traceback
import tempfile

os.environ["NNPACK_DISABLE"] = "1"
os.environ["ATEN_DISABLE_NNPACK"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_USE_CUDA"] = "0"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/en/vctk/vits").to(device)


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
            speaker = data.get("speaker", tts.speakers[0])
            # language = data.get("language", "en")  # fallback to English
            # tts.tts_to_file(
            #     text=text, speaker=speaker, language=language, file_path=path
            # )

            tts.tts_to_file(text=text, speaker=speaker, file_path=path)

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

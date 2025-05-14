from flask import Flask, request, jsonify
import os
import logging
import wave
from vosk import Model, KaldiRecognizer

app = Flask(__name__)
model = Model("models/vosk-model-small-en-us-0.15")


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    input_pcm_path = "temp_input.raw"
    output_wav_path = "temp_output.wav"

    try:
        audio_file = request.files["audio"]
        audio_file.save(input_pcm_path)
        file_size = os.path.getsize(input_pcm_path)
        logging.debug(f"Saved input audio as {input_pcm_path} ({file_size} bytes)")

        if file_size < 1000:
            logging.error("Audio file too small or empty.")
            return jsonify({"error": "Audio file is empty or corrupted"}), 400

        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wf.setframerate(16000)
            with open(input_pcm_path, "rb") as pcm:
                wf.writeframes(pcm.read())

        wf = wave.open(output_wav_path, "rb")
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
        for path in [input_pcm_path, output_wav_path]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)

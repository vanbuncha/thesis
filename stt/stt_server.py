from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
import wave
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load STT model
model = Model("/app/models/vosk-model-en-us-0.42-gigaspeech")  # Path to Vosk model


@app.route("/stt", methods=["POST"])
def stt():
    try:
        # Save the uploaded audio file
        audio_file = request.files["audio"]
        audio_file.save("temp.wav")
        logging.debug("Audio file saved as temp.wav")

        # Open the audio file
        wf = wave.open("temp.wav", "rb")
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getframerate() not in [8000, 16000]
        ):
            logging.error("Invalid audio format")
            return jsonify({"error": "Audio file must be WAV format mono PCM."}), 400

        # Initialize recognizer
        rec = KaldiRecognizer(model, wf.getframerate())
        logging.debug("Recognizer initialized")

        # Process audio
        result = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result += rec.Result()
                logging.debug(f"Partial result: {result}")

        # Get final result
        final_result = rec.FinalResult()
        result += final_result
        logging.debug(f"Final result: {result}")

        # Clean up
        wf.close()
        os.remove("temp.wav")

        # Return the result
        return jsonify({"text": result})

    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


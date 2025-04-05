from flask import Flask, request, jsonify
import requests
import json
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

OLLAMA_URL = "http://ollama:11434/api/generate"  # Ollama API


@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Log the raw incoming request data
        raw_data = request.data.decode("utf-8")
        app.logger.debug("Raw Data Received: %s", raw_data)

        json_data = json.loads(raw_data)
        app.logger.debug("Parsed JSON Data: %s", json_data)

        user_input = json_data.get("prompt", "")

        # Call Ollama API with streaming enabled
        response = requests.post(
            OLLAMA_URL,
            json={"model": "mistral", "prompt": user_input, "stream": True},
            stream=True,  # Enables token-by-token response
        )

        if response.status_code == 200:
            full_response = ""

            # Read the streaming response token by token
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        decoded_chunk = json.loads(chunk.decode("utf-8"))
                        token = decoded_chunk.get("response", "")
                        full_response += token  # Append to the final response
                    except json.JSONDecodeError:
                        app.logger.error("Error decoding JSON chunk: %s", chunk)

            return jsonify({"response": full_response})  # Return full response
        else:
            app.logger.error(
                f"Error from Ollama API: {response.status_code} - {response.text}"
            )
            return jsonify({"error": "Ollama API returned an error"}), 500

    except Exception as e:
        app.logger.error("Error: %s", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

# Thesis Repository – 2025

This repository contains the implementation for the **Social Robotics System for Senior Care using LLMs and Raspberry Pi**.

---

## 🚀 Prerequisites

- [Docker CLI](https://docs.docker.com/get-started/get-docker/) (client version, not Desktop)

---

## 🛠️ Building the Project

Build and start the containers (first time may take up to 5 minutes):

```bash
docker compose up --build
```

If only code changes were made (no Dockerfile changes), you can just restart:

```bash
docker compose up
```

🤖 Installing Models (Ollama)
Once the project is running, you need to install the desired model inside the Ollama container:
```bash
docker exec -it ollama bash
ollama pull mistral
```
To list installed models:
```bash
ollama list
```
or you can also browse available models at ollama.com/library

# Client Setup
It is recommended to use a virtual environment for the client:
```python
python3 -m venv client_env
source client_env/bin/activate
```
Install requirements:
```python
pip install -r requirements.txt
```
Run the client:
```python
python wspi_client.py
```
# Model Configuration
## STT (Speech-to-Text)
Change the model in `stt_server.py`:
```
return Model("models/vosk-model-small-en-us-0.15")
```
## TTS (Text-to-Speech)
Change the model in `tts_server.py`:
```
tts = TTS(model_name="tts_models/en/vctk/vits").to(device)
```

# Database Usage
Enter the Postgres container:
```
docker exec -it elderly_care_db bash
```
Connect to Postgres:
```
psql -U supauser -d elderly_care_db
```
List tables:
```
\dt
```
# Configure IP on Client
Before starting up the client it is important to set correct server IP address:
```bash
hostname -I
```
The IP address needs to be then correctly set in .env file



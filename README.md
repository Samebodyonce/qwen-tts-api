# Qwen3-TTS Web Service

FastAPI wrapper around [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for voice-cloning text-to-speech.

## Prerequisites

- Python 3.10+
- CUDA GPU
- `sox` system package (`apt install sox libsox-fmt-all`)

## Installation

```bash
pip install numpy typing_extensions
pip install -r requirements.txt
```

(`numpy` and `typing_extensions` are installed first to avoid dependency resolution issues.)

## Running

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The model downloads on first launch (~1.5 GB) and loads onto `cuda:0`.

## API

All TTS endpoints return `audio/wav` bytes.

### Health check

```bash
curl http://localhost:8000/health
```

### One-shot TTS (JSON)

Provide a reference audio via URL or base64:

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "language": "English",
    "ref_audio_url": "https://example.com/ref.wav",
    "ref_text": "Transcript of the reference audio."
  }' \
  --output output.wav
```

Or with base64-encoded audio:

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Hello, world!\",
    \"language\": \"English\",
    \"ref_audio_base64\": \"$(base64 -w0 ref.wav)\",
    \"ref_text\": \"Transcript of the reference audio.\"
  }" \
  --output output.wav
```

### One-shot TTS (file upload)

```bash
curl -X POST http://localhost:8000/v1/tts/upload \
  -F "text=Hello, world!" \
  -F "language=English" \
  -F "ref_text=Transcript of the reference audio." \
  -F "ref_audio=@ref.wav" \
  --output output.wav
```

### Cached voice workflow

Extract a voice prompt once, then reuse it for multiple requests:

**1. Create a voice (JSON)**

```bash
curl -X POST http://localhost:8000/v1/voice \
  -H "Content-Type: application/json" \
  -d '{
    "ref_audio_url": "https://example.com/ref.wav",
    "ref_text": "Transcript of the reference audio."
  }'
# Returns: {"voice_id": "<uuid>"}
```

**1b. Or create a voice (file upload)**

```bash
curl -X POST http://localhost:8000/v1/voice/upload \
  -F "ref_text=Transcript of the reference audio." \
  -F "ref_audio=@ref.wav"
# Returns: {"voice_id": "<uuid>"}
```

**2. Synthesize with the cached voice**

```bash
curl -X POST http://localhost:8000/v1/tts/<voice_id> \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello again!", "language": "English"}' \
  --output output.wav
```

## Docker

```bash
docker build -t qwen-tts .
docker run --gpus all -p 8000:8000 qwen-tts
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MAX_CONCURRENT_INFERENCES` | `1` | Max parallel GPU inferences (semaphore size) |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed CORS origins |

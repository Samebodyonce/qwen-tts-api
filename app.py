import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from math import gcd
from typing import Optional

import httpx
import numpy as np
import soundfile as sf
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from scipy.signal import resample_poly
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger("qwen-tts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# --- Config ---
VOICES_CONFIG_PATH = os.environ.get("VOICES_CONFIG", "voices.yaml")
MODEL_PATH_0_6B = os.environ.get("MODEL_PATH_0_6B", "models/Qwen3-TTS-12Hz-0.6B-Base")
MODEL_PATH_1_6B = os.environ.get("MODEL_PATH_1_6B", "models/Qwen3-TTS-25Hz-1.5B-Base")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
MAX_CONCURRENT_INFERENCES = int(os.environ.get("MAX_CONCURRENT_INFERENCES", "1"))

# --- Global state ---
models: dict[str, Qwen3TTSModel] = {}
voice_prompts: dict[str, list] = {}
voice_config: dict = {}
gpu_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)


def _load_qwen_model(path: str) -> Qwen3TTSModel:
    try:
        m = Qwen3TTSModel.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
        )
        logger.info("Model loaded with flash_attention_2 from %s", path)
        return m
    except Exception:
        logger.info("flash_attention_2 unavailable, falling back to sdpa for %s", path)
        m = Qwen3TTSModel.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        logger.info("Model loaded with sdpa from %s", path)
        return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, voice_prompts, voice_config

    with open(VOICES_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    voice_config = raw.get("voices", {})
    logger.info("Loaded %d voices from %s", len(voice_config), VOICES_CONFIG_PATH)

    # Determine which Qwen backends are actually needed
    needed_backends = {cfg["backend"] for cfg in voice_config.values()}

    if "qwen_0_6b" in needed_backends:
        logger.info("Loading Qwen 0.6B model from %s ...", MODEL_PATH_0_6B)
        models["qwen_0_6b"] = _load_qwen_model(MODEL_PATH_0_6B)

    if "qwen_1_6b" in needed_backends:
        logger.info("Loading Qwen 1.6B model from %s ...", MODEL_PATH_1_6B)
        models["qwen_1_6b"] = _load_qwen_model(MODEL_PATH_1_6B)

    # Pre-extract voice clone prompts for all Qwen voices
    for voice_name, cfg in voice_config.items():
        backend = cfg["backend"]
        if backend in ("qwen_0_6b", "qwen_1_6b"):
            ref_audio = cfg["ref_audio"]
            ref_text = cfg.get("ref_text", "")
            logger.info("Extracting voice prompt for '%s' ...", voice_name)
            prompt = models[backend].create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text
            )
            voice_prompts[voice_name] = prompt
            logger.info("Voice prompt extracted for '%s'", voice_name)

    yield
    logger.info("Shutting down")


app = FastAPI(title="TTS Routing Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Audio conversion ---

def _to_8k_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to 8kHz mono 16-bit WAV bytes."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to 8000 Hz
    if sample_rate != 8000:
        g = gcd(8000, sample_rate)
        up = 8000 // g
        down = sample_rate // g
        audio = resample_poly(audio, up, down).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio, 8000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --- Inference helpers ---

def _qwen_synthesize(model_key: str, voice_name: str, text: str) -> bytes:
    """Run Qwen TTS inference and return 8kHz WAV bytes."""
    prompt = voice_prompts[voice_name]
    m = models[model_key]
    wav_arrays, sample_rate = m.generate_voice_clone(
        text=text,
        language="Russian",
        voice_clone_prompt=prompt,
    )
    audio = np.array(wav_arrays[0], dtype=np.float32)
    return _to_8k_wav(audio, sample_rate)


async def _elevenlabs_synthesize(voice_id: str, text: str) -> bytes:
    """Call ElevenLabs API and return 8kHz WAV bytes."""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "output_format": "pcm_16000",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"ElevenLabs API error {resp.status_code}: {resp.text[:200]}",
        )

    # pcm_16000 is raw PCM 16kHz 16-bit mono
    audio = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768.0
    return _to_8k_wav(audio, 16000)


# --- Pydantic models ---

class TTSRequest(BaseModel):
    voice: str
    text: str


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/voices")
async def list_voices():
    return {"voices": list(voice_config.keys())}


@app.post("/v1/tts")
async def tts(req: TTSRequest):
    cfg = voice_config.get(req.voice)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Voice '{req.voice}' not found")

    backend = cfg["backend"]

    if backend == "elevenlabs":
        wav_bytes = await _elevenlabs_synthesize(cfg["voice_id"], req.text)
    elif backend in ("qwen_0_6b", "qwen_1_6b"):
        async with gpu_semaphore:
            try:
                wav_bytes = await asyncio.to_thread(
                    _qwen_synthesize, backend, req.voice, req.text
                )
            except Exception as e:
                logger.exception("Qwen TTS inference failed for voice '%s'", req.voice)
                raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=500, detail=f"Unknown backend: {backend}")

    return Response(content=wav_bytes, media_type="audio/wav")

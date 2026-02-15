import asyncio
import base64
import io
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger("qwen-tts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Global state
model: Optional[Qwen3TTSModel] = None
voice_cache: dict[str, object] = {}
gpu_semaphore: asyncio.Semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT_INFERENCES", "1")))

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading Qwen3-TTS model...")
    attn = "flash_attention_2"
    try:
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation=attn,
        )
        logger.info("Model loaded with flash_attention_2")
    except Exception:
        logger.info("flash_attention_2 unavailable, falling back to sdpa")
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        logger.info("Model loaded with sdpa")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Qwen3-TTS Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---

class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    ref_audio_url: Optional[str] = None
    ref_audio_base64: Optional[str] = None
    ref_text: str = ""


class VoiceTTSRequest(BaseModel):
    text: str
    language: str = "English"


class VoiceCreateRequest(BaseModel):
    ref_audio_url: Optional[str] = None
    ref_audio_base64: Optional[str] = None
    ref_text: str = ""


# --- Helpers ---

def _resolve_ref_audio(ref_audio_url: Optional[str], ref_audio_base64: Optional[str]) -> str:
    """Return a file path or URL suitable for the model. Base64 is written to a temp file."""
    if ref_audio_url:
        return ref_audio_url
    if ref_audio_base64:
        audio_bytes = base64.b64decode(ref_audio_base64)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    raise ValueError("Either ref_audio_url or ref_audio_base64 must be provided")


def _save_upload_to_temp(data: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _synthesize(text: str, language: str, ref_audio: str, ref_text: str) -> bytes:
    """Run TTS inference and return WAV bytes."""
    wav_arrays, sample_rate = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    buf = io.BytesIO()
    sf.write(buf, wav_arrays[0], sample_rate, format="WAV")
    return buf.getvalue()


def _synthesize_with_prompt(text: str, language: str, voice_prompt: list) -> bytes:
    """Run TTS inference with a pre-extracted voice clone prompt."""
    wav_arrays, sample_rate = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_prompt,
    )
    buf = io.BytesIO()
    sf.write(buf, wav_arrays[0], sample_rate, format="WAV")
    return buf.getvalue()


def _extract_voice_prompt(ref_audio: str, ref_text: str) -> list:
    return model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/tts")
async def tts_json(req: TTSRequest):
    ref_audio = _resolve_ref_audio(req.ref_audio_url, req.ref_audio_base64)

    async with gpu_semaphore:
        try:
            wav_bytes = await asyncio.to_thread(
                _synthesize, req.text, req.language, ref_audio, req.ref_text
            )
        except Exception as e:
            logger.exception("TTS inference failed")
            raise HTTPException(status_code=500, detail=str(e))

    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/v1/tts/upload")
async def tts_upload(
    text: str = Form(...),
    language: str = Form("English"),
    ref_text: str = Form(""),
    ref_audio: UploadFile = File(...),
):
    audio_data = await ref_audio.read()
    ref_path = _save_upload_to_temp(audio_data)

    async with gpu_semaphore:
        try:
            wav_bytes = await asyncio.to_thread(
                _synthesize, text, language, ref_path, ref_text
            )
        except Exception as e:
            logger.exception("TTS inference failed")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            os.unlink(ref_path)

    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/v1/voice")
async def create_voice(req: VoiceCreateRequest):
    ref_audio = _resolve_ref_audio(req.ref_audio_url, req.ref_audio_base64)

    async with gpu_semaphore:
        try:
            prompt = await asyncio.to_thread(_extract_voice_prompt, ref_audio, req.ref_text)
        except Exception as e:
            logger.exception("Voice prompt extraction failed")
            raise HTTPException(status_code=500, detail=str(e))

    voice_id = str(uuid.uuid4())
    voice_cache[voice_id] = prompt
    logger.info("Created voice %s (cache size: %d)", voice_id, len(voice_cache))
    return {"voice_id": voice_id}


@app.post("/v1/voice/upload")
async def create_voice_upload(
    ref_text: str = Form(""),
    ref_audio: UploadFile = File(...),
):
    audio_data = await ref_audio.read()
    ref_path = _save_upload_to_temp(audio_data)

    async with gpu_semaphore:
        try:
            prompt = await asyncio.to_thread(_extract_voice_prompt, ref_path, ref_text)
        except Exception as e:
            logger.exception("Voice prompt extraction failed")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            os.unlink(ref_path)

    voice_id = str(uuid.uuid4())
    voice_cache[voice_id] = prompt
    logger.info("Created voice %s (cache size: %d)", voice_id, len(voice_cache))
    return {"voice_id": voice_id}


@app.post("/v1/tts/{voice_id}")
async def tts_with_voice(voice_id: str, req: VoiceTTSRequest):
    prompt = voice_cache.get(voice_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    async with gpu_semaphore:
        try:
            wav_bytes = await asyncio.to_thread(
                _synthesize_with_prompt, req.text, req.language, prompt
            )
        except Exception as e:
            logger.exception("TTS inference failed")
            raise HTTPException(status_code=500, detail=str(e))

    return Response(content=wav_bytes, media_type="audio/wav")

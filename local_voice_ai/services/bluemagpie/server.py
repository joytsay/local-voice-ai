"""Minimal OpenAI-compatible TTS server backed by OpenFormosa BlueMagpie-TTS."""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger("bluemagpie")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.getenv("BLUEMAGPIE_MODEL_NAME", "OpenFormosa/BlueMagpie-TTS")
MODEL_ID = os.getenv("BLUEMAGPIE_MODEL_ID", "bluemagpie-tts")
DEFAULT_VOICE = os.getenv("BLUEMAGPIE_DEFAULT_VOICE", "chinese_female")
DEFAULT_CFG_VALUE = float(os.getenv("BLUEMAGPIE_CFG_VALUE", "2.8"))
DEFAULT_INFERENCE_TIMESTEPS = int(os.getenv("BLUEMAGPIE_INFERENCE_TIMESTEPS", "9"))
DEFAULT_SEED = int(os.getenv("BLUEMAGPIE_SEED", "1729"))
FORCED_VOICE = os.getenv("BLUEMAGPIE_FORCE_VOICE", "").strip()

_model = None
_speaker_table: Optional[dict[str, object]] = None
_hung_yi_lee_centroid: Optional[torch.Tensor] = None
_model_dir: Optional[str] = None


def _complete_model_dir(path: str) -> bool:
    required = (
        "config.json",
        "tokenizer.json",
        "audiovae.pth",
        "pytorch_model.bin",
        os.path.join("checkpoints", "speaker_centroids.pt"),
    )
    return all(os.path.exists(os.path.join(path, item)) for item in required)


def _compatible_local_model_dir(path: str) -> Optional[str]:
    """Resolve an incomplete HF snapshot path to a complete cached sibling."""
    if _complete_model_dir(path):
        return path

    snapshots_dir = os.path.dirname(os.path.abspath(path))
    if os.path.basename(snapshots_dir) != "snapshots":
        return None

    candidates = sorted(
        os.listdir(snapshots_dir),
        key=lambda item: os.path.getmtime(os.path.join(snapshots_dir, item)),
        reverse=True,
    )
    for commit in candidates:
        snapshot = os.path.join(snapshots_dir, commit)
        if snapshot != os.path.abspath(path) and _complete_model_dir(snapshot):
            return snapshot
    return None


def _cached_snapshot(repo_id: str) -> Optional[str]:
    if "/" not in repo_id:
        return None

    cache_home = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or "/models"
    namespace, name = repo_id.split("/", 1)
    repo_cache = os.path.join(cache_home, "hub", f"models--{namespace}--{name}")
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    refs_main = os.path.join(repo_cache, "refs", "main")
    candidates: list[str] = []
    if os.path.exists(refs_main):
        with open(refs_main, encoding="utf-8") as f:
            candidates.append(f.read().strip())

    candidates.extend(
        sorted(
            os.listdir(snapshots_dir),
            key=lambda item: os.path.getmtime(os.path.join(snapshots_dir, item)),
            reverse=True,
        )
    )

    seen: set[str] = set()
    for commit in candidates:
        if not commit or commit in seen:
            continue
        seen.add(commit)
        snapshot = os.path.join(snapshots_dir, commit)
        if _complete_model_dir(snapshot):
            return snapshot
    return None


def _device() -> str:
    requested = os.getenv("DEVICE", "").strip().lower()
    if requested == "cpu":
        return requested
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("Requested DEVICE=cuda but CUDA is unavailable; falling back to CPU")
        return "cpu"
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("Requested DEVICE=mps but MPS is unavailable; falling back to CPU")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model() -> None:
    global _model, _speaker_table, _hung_yi_lee_centroid, _model_dir

    device = _device()
    logger.info("downloading/loading BlueMagpie model %s on %s", MODEL_NAME, device)
    from bluemagpie import BlueMagpieModel  # type: ignore[import-not-found]
    from transformers import PreTrainedTokenizerFast

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if os.path.isdir(MODEL_NAME):
        _model_dir = _compatible_local_model_dir(MODEL_NAME)
        if _model_dir is None:
            raise FileNotFoundError(
                f"BlueMagpie model directory {MODEL_NAME!r} is incomplete: "
                "expected config.json, tokenizer.json, audiovae.pth, "
                "pytorch_model.bin, and checkpoints/speaker_centroids.pt"
            )
        if os.path.abspath(_model_dir) != os.path.abspath(MODEL_NAME):
            logger.warning(
                "BlueMagpie snapshot %s is incomplete; using complete cached snapshot %s",
                MODEL_NAME,
                _model_dir,
            )
    else:
        _model_dir = _cached_snapshot(MODEL_NAME)
        if _model_dir is None:
            from huggingface_hub import snapshot_download

            _model_dir = snapshot_download(MODEL_NAME, token=token)
    logger.info("using BlueMagpie model directory %s", _model_dir)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(_model_dir, "tokenizer.json")
    )
    _model = BlueMagpieModel.from_local(
        _model_dir,
        tokenizer=tokenizer,
        training=False,
        device=device,
    )

    table_path = os.path.join(_model_dir, "checkpoints", "speaker_centroids.pt")
    if os.path.exists(table_path):
        _speaker_table = torch.load(table_path, map_location="cpu", weights_only=True)

    hung_yi_lee_path = os.path.join(
        _model_dir,
        "checkpoints",
        "hung_yi_lee_speaker_centroids.pt",
    )
    if os.path.exists(hung_yi_lee_path):
        hung_yi_lee_table = torch.load(
            hung_yi_lee_path,
            map_location="cpu",
            weights_only=True,
        )
        speaker_ids = hung_yi_lee_table.get("speaker_ids")
        centroids = hung_yi_lee_table.get("centroids")
        if isinstance(speaker_ids, list) and centroids is not None and "hung_yi_lee" in speaker_ids:
            _hung_yi_lee_centroid = centroids[speaker_ids.index("hung_yi_lee")]

    logger.info("BlueMagpie model ready on %s", getattr(_model, "device", _device()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="BlueMagpie TTS Server", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"
    cfg_value: float = DEFAULT_CFG_VALUE
    inference_timesteps: int = DEFAULT_INFERENCE_TIMESTEPS
    min_len: int = 2
    max_len: int = 2000
    retry_badcase: bool = False
    seed: int = DEFAULT_SEED


def _speaker_id(voice: str) -> str:
    normalized = (voice or DEFAULT_VOICE).strip().lower().replace("-", "_")
    aliases = {
        "chinese_female": "female_voice",
        "female": "female_voice",
        "zh_female": "female_voice",
        "mandarin_female": "female_voice",
        "taiwan_female": "female_voice",
    }
    return aliases.get(normalized, voice or DEFAULT_VOICE)


def _speaker_centroid(voice: str):
    speaker_id = _speaker_id(voice)
    if speaker_id == "hung_yi_lee" and _hung_yi_lee_centroid is not None:
        return _hung_yi_lee_centroid
    if not _speaker_table:
        return None
    speaker_ids = _speaker_table.get("speaker_ids")
    centroids = _speaker_table.get("centroids")
    if not isinstance(speaker_ids, list) or centroids is None or speaker_id not in speaker_ids:
        return None
    return centroids[speaker_ids.index(speaker_id)]


def _synthesize(
    text: str,
    voice: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    seed: int,
) -> tuple[np.ndarray, int]:
    if _model is None:
        raise RuntimeError("model not loaded")

    kwargs = {
        "target_text": text,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "min_len": min_len,
        "max_len": max_len,
        "retry_badcase": retry_badcase,
    }

    forced_voice = FORCED_VOICE or voice
    speaker_id = _speaker_id(forced_voice)
    centroid = _speaker_centroid(forced_voice)
    if centroid is not None:
        kwargs["speaker_centroid"] = centroid
        logger.info("using speaker centroid id=%s", speaker_id)
    else:
        logger.warning("speaker centroid for %s not found; using model default speaker", speaker_id)

    with torch.no_grad():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        audio = _model.generate(**kwargs)

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio_np = np.asarray(audio, dtype=np.float32).squeeze()

    return audio_np, int(getattr(_model, "sample_rate", 48000))


def _encode(audio: np.ndarray, sample_rate: int, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "mp3").lower()
    buf = io.BytesIO()

    if fmt in {"mp3", "opus", "aac", "flac"}:
        try:
            sf.write(buf, audio, sample_rate, format=fmt.upper())
            return buf.getvalue(), f"audio/{fmt}"
        except Exception:
            buf = io.BytesIO()

    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest) -> Response:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.input:
        raise HTTPException(status_code=400, detail="input is required")

    voice = req.voice or DEFAULT_VOICE
    requested_format = req.response_format or "mp3"
    logger.info(
        "tts request chars=%d voice=%s format=%s cfg=%.2f steps=%d min_len=%d "
        "max_len=%d retry=%s seed=%d",
        len(req.input),
        voice,
        requested_format,
        req.cfg_value,
        req.inference_timesteps,
        req.min_len,
        req.max_len,
        req.retry_badcase,
        req.seed,
    )
    try:
        audio, sample_rate = _synthesize(
            req.input,
            voice,
            req.cfg_value,
            req.inference_timesteps,
            req.min_len,
            req.max_len,
            req.retry_badcase,
            req.seed,
        )
    except Exception as exc:
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    data, media_type = _encode(audio, sample_rate, requested_format)
    logger.info(
        "tts response samples=%d sample_rate=%d bytes=%d media_type=%s",
        audio.size,
        sample_rate,
        len(data),
        media_type,
    )
    return Response(content=data, media_type=media_type)


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "OpenFormosa",
                }
            ],
        }
    )


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlueMagpie TTS Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)

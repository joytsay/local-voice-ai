"""Minimal OpenAI-compatible TTS server backed by OpenFormosa BlueMagpie-TTS."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


def _install_sdpa_gqa_compat() -> None:
    """Backport ``enable_gqa`` for NVIDIA's older Jetson PyTorch build."""
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    probe = torch.zeros((1, 1, 1, 1))
    try:
        original_sdpa(probe, probe, probe, enable_gqa=False)
    except TypeError as exc:
        if "enable_gqa" not in str(exc):
            raise
    else:
        return

    def scaled_dot_product_attention_compat(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args: object,
        enable_gqa: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        if enable_gqa:
            query_heads = query.size(-3)
            key_heads = key.size(-3)
            value_heads = value.size(-3)
            if key_heads != value_heads:
                raise ValueError(
                    "Grouped-query attention requires matching key/value head counts"
                )
            if query_heads % key_heads:
                raise ValueError(
                    "Grouped-query attention requires query heads to be divisible "
                    "by key/value heads"
                )
            repeats = query_heads // key_heads
            if repeats > 1:
                key = key.repeat_interleave(repeats, dim=-3)
                value = value.repeat_interleave(repeats, dim=-3)
        return original_sdpa(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = (
        scaled_dot_product_attention_compat
    )


def _install_torch_distributed_compat() -> None:
    """Supply the symbol SpeechBrain imports on non-distributed Jetson builds."""
    distributed = getattr(torch, "distributed", None)
    if distributed is None or hasattr(distributed, "ReduceOp"):
        return

    class ReduceOpCompat:
        # SpeechBrain's ECAPA inference does not call its distributed-statistics
        # path. The placeholder only lets that optional helper module import.
        SUM = None

    distributed.ReduceOp = ReduceOpCompat


_install_sdpa_gqa_compat()
_install_torch_distributed_compat()
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger("bluemagpie")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.getenv("BLUEMAGPIE_MODEL_NAME", "OpenFormosa/BlueMagpie-TTS")
MODEL_ID = os.getenv("BLUEMAGPIE_MODEL_ID", "bluemagpie-tts")
DEFAULT_VOICE = os.getenv("BLUEMAGPIE_DEFAULT_VOICE", "chinese_female")
DEFAULT_CFG_VALUE = float(os.getenv("BLUEMAGPIE_CFG_VALUE", "2.0"))
DEFAULT_INFERENCE_TIMESTEPS = int(os.getenv("BLUEMAGPIE_INFERENCE_TIMESTEPS", "10"))
DEFAULT_SEED = int(os.getenv("BLUEMAGPIE_SEED", "1729"))
FORCED_VOICE = os.getenv("BLUEMAGPIE_FORCE_VOICE", "").strip()
CLONE_DEVICE = (
    os.getenv("BLUEMAGPIE_CLONE_DEVICE", "cuda:0").strip() or "cuda:0"
)
ECAPA_MODEL = os.getenv(
    "BLUEMAGPIE_ECAPA_MODEL",
    "speechbrain/spkrec-ecapa-voxceleb",
)
MAX_REFERENCE_AUDIO_BYTES = 25 * 1024 * 1024
VOICE_CLONE_DIR = Path(os.getenv("VOICE_CLONE_DIR", "/models/voice_clones"))
SPEAKER_WINDOW_SECONDS = 6.0
SPEAKER_EMBEDDING_VERSION = "bluemagpie-ecapa-windowed-6s-v2"

_model = None
_speaker_table: Optional[dict[str, object]] = None
_hung_yi_lee_centroid: Optional[torch.Tensor] = None
_model_dir: Optional[str] = None
_speaker_encoder = None
_SPEAKER_ENCODER_LOCK = threading.Lock()
_CENTROID_LOCK = threading.Lock()


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

    if os.path.isdir(MODEL_NAME):
        logger.info("Loading BlueMagpie from local snapshot: %s", _model_dir)
    else:
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
    reference_audio: Optional[str] = None
    clone_mode: str = "speaker_centroid"


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


def _clone_profile(voice: str) -> Optional[dict[str, object]]:
    manifest_path = VOICE_CLONE_DIR / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    for entry in manifest.get("voices", []):
        if isinstance(entry, dict) and entry.get("id") == voice:
            return entry
    return None


def _clone_file(
    profile: dict[str, object],
    field: str,
    *,
    must_exist: bool,
) -> Optional[Path]:
    relative_name = str(profile.get(field, "")).strip()
    if not relative_name:
        return None

    clone_root = VOICE_CLONE_DIR.resolve()
    path = (VOICE_CLONE_DIR / relative_name).resolve()
    if not path.is_relative_to(clone_root) or path == clone_root:
        raise ValueError(f"cloned voice profile contains an unsafe {field} path")
    if must_exist and not path.is_file():
        raise ValueError(f"cloned voice {field} file is missing")
    return path


def _clone_device() -> str:
    if CLONE_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "BLUEMAGPIE_CLONE_DEVICE=%s but CUDA is unavailable; using CPU",
            CLONE_DEVICE,
        )
        return "cpu"
    return "cuda:0" if CLONE_DEVICE == "cuda" else CLONE_DEVICE


def _get_speaker_encoder():
    global _speaker_encoder

    with _SPEAKER_ENCODER_LOCK:
        if _speaker_encoder is None:
            from speechbrain.inference.speaker import EncoderClassifier

            device = _clone_device()
            savedir = VOICE_CLONE_DIR / "ecapa"
            savedir.mkdir(parents=True, exist_ok=True)
            logger.info("loading speaker encoder %s on %s", ECAPA_MODEL, device)
            _speaker_encoder = EncoderClassifier.from_hparams(
                source=ECAPA_MODEL,
                savedir=str(savedir),
                run_opts={"device": device},
            )
    return _speaker_encoder


def _extract_centroid(reference_path: Path) -> torch.Tensor:
    from bluemagpie import extract_speaker_centroid

    device = _clone_device()
    centroid = extract_speaker_centroid(
        str(reference_path),
        ecapa_model=ECAPA_MODEL,
        device=device,
        window_s=SPEAKER_WINDOW_SECONDS,
        encoder=_get_speaker_encoder(),
    )
    centroid = torch.as_tensor(centroid).detach().cpu().float().reshape(-1)
    expected_dim = int(getattr(_model, "speaker_embed_dim", 192))
    if centroid.numel() != expected_dim or not torch.isfinite(centroid).all():
        raise ValueError(
            f"invalid speaker centroid: expected {expected_dim} finite values, "
            f"got {centroid.numel()}"
        )
    return centroid


def _saved_clone_centroid(voice: str) -> Optional[torch.Tensor]:
    profile = _clone_profile(voice)
    if profile is None:
        return None

    reference_path = _clone_file(profile, "reference_wav", must_exist=True)
    if reference_path is None:
        raise ValueError("cloned voice profile has no reference audio")

    centroid_path = _clone_file(profile, "speaker_centroid", must_exist=False)
    if centroid_path is None:
        centroid_path = (VOICE_CLONE_DIR / f"{voice}.pt").resolve()
        if not centroid_path.is_relative_to(VOICE_CLONE_DIR.resolve()):
            raise ValueError("cloned voice ID produces an unsafe centroid path")

    with _CENTROID_LOCK:
        centroid = None
        if centroid_path.is_file():
            cached = torch.load(
                centroid_path,
                map_location="cpu",
                weights_only=True,
            )
            if (
                isinstance(cached, dict)
                and cached.get("version") == SPEAKER_EMBEDDING_VERSION
                and cached.get("embedding") is not None
            ):
                centroid = cached["embedding"]
            elif (
                profile.get("speaker_embedding_version")
                == SPEAKER_EMBEDDING_VERSION
            ):
                centroid = cached

        if centroid is None:
            logger.info(
                "extracting windowed speaker embedding for saved clone id=%s",
                voice,
            )
            centroid = _extract_centroid(reference_path)
            centroid_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=centroid_path.parent,
                suffix=".pt.tmp",
                delete=False,
            ) as temporary_centroid:
                temporary_centroid_path = Path(temporary_centroid.name)
            try:
                torch.save(
                    {
                        "version": SPEAKER_EMBEDDING_VERSION,
                        "embedding": centroid,
                    },
                    temporary_centroid_path,
                )
                os.replace(temporary_centroid_path, centroid_path)
            finally:
                temporary_centroid_path.unlink(missing_ok=True)

    centroid = torch.as_tensor(centroid).detach().cpu().float().reshape(-1)
    expected_dim = int(getattr(_model, "speaker_embed_dim", 192))
    if centroid.numel() != expected_dim or not torch.isfinite(centroid).all():
        raise ValueError(
            f"cached speaker centroid for {voice!r} is invalid; delete "
            f"{centroid_path.name!r} and try cloning again"
        )
    return centroid


def _decode_reference_audio(value: str) -> bytes:
    encoded = value.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("reference_audio data URL must contain base64 audio")
    try:
        audio = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("reference_audio must be valid base64") from exc
    if not audio:
        raise ValueError("reference_audio is empty")
    if len(audio) > MAX_REFERENCE_AUDIO_BYTES:
        raise ValueError("reference_audio exceeds the 25 MB limit")
    return audio


def _synthesize(
    text: str,
    voice: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    seed: int,
    reference_audio: Optional[str],
    clone_mode: str,
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
    valid_clone_modes = {"reference_wav_path", "speaker_centroid"}
    if clone_mode not in valid_clone_modes:
        raise ValueError(
            "clone_mode must be 'reference_wav_path' or 'speaker_centroid'"
        )

    reference_path: Optional[str] = None
    if reference_audio:
        reference_bytes = _decode_reference_audio(reference_audio)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as reference_file:
            reference_file.write(reference_bytes)
            reference_path = reference_file.name
        if clone_mode == "reference_wav_path":
            kwargs["reference_wav_path"] = reference_path
            logger.info(
                "using reference WAV for one-shot voice cloning bytes=%d",
                len(reference_bytes),
            )
        else:
            with _CENTROID_LOCK:
                kwargs["speaker_centroid"] = _extract_centroid(Path(reference_path))
            logger.info(
                "using extracted speaker centroid for one-shot voice cloning bytes=%d",
                len(reference_bytes),
            )
    else:
        forced_voice = FORCED_VOICE or voice
        clone_profile = _clone_profile(forced_voice)
        profile_mode = (
            str(clone_profile.get("mode", ""))
            if clone_profile is not None
            else ""
        )
        effective_clone_mode = (
            profile_mode if profile_mode in valid_clone_modes else clone_mode
        )
        if clone_profile is not None and effective_clone_mode == "reference_wav_path":
            clone_reference = _clone_file(
                clone_profile,
                "reference_wav",
                must_exist=True,
            )
            if clone_reference is None:
                raise ValueError("cloned voice profile has no reference audio")
            kwargs["reference_wav_path"] = str(clone_reference)
            logger.info("using reference WAV for saved clone id=%s", forced_voice)
        else:
            clone_centroid = _saved_clone_centroid(forced_voice)
            if clone_centroid is not None:
                kwargs["speaker_centroid"] = clone_centroid
                logger.info(
                    "using cached speaker centroid for clone id=%s",
                    forced_voice,
                )
            else:
                speaker_id = _speaker_id(forced_voice)
                centroid = _speaker_centroid(forced_voice)
                if centroid is not None:
                    kwargs["speaker_centroid"] = centroid
                    logger.info("using speaker centroid id=%s", speaker_id)
                else:
                    logger.warning(
                        "speaker centroid for %s not found; using model default speaker",
                        speaker_id,
                    )

    try:
        with torch.no_grad():
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            audio = _model.generate(**kwargs)
    finally:
        if reference_path:
            try:
                os.unlink(reference_path)
            except FileNotFoundError:
                pass

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
        "max_len=%d retry=%s seed=%d clone=%s clone_mode=%s",
        len(req.input),
        voice,
        requested_format,
        req.cfg_value,
        req.inference_timesteps,
        req.min_len,
        req.max_len,
        req.retry_badcase,
        req.seed,
        bool(req.reference_audio)
        or _clone_profile(FORCED_VOICE or voice) is not None,
        req.clone_mode,
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
            req.reference_audio,
            req.clone_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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

from __future__ import annotations

import gc
import logging
import os
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

logger = logging.getLogger("whisper")
logging.basicConfig(level=logging.INFO)

_model: Any = None
_loaded_model_name: str | None = None
_MODEL_LOCK = threading.Lock()
_diarization_pipeline: Any = None
# CTranslate2 and PyTorch share the same GPU. Serialize both inference paths so
# concurrent requests cannot create avoidable Jetson memory spikes.
_DIARIZATION_LOCK = _MODEL_LOCK

_OUTRO_TERM_GROUPS = (
    ("点赞", "點讚", "点讚", "like"),
    ("订阅", "訂閱", "subscribe"),
    ("转发", "轉發", "share"),
    ("打赏", "打賞", "donate", "tip"),
    ("栏目", "欄目", "channel"),
)
_SUBTITLE_TERMS = ("字幕", "subtitles", "captions")
_CREDIT_TERMS = (
    "志愿者",
    "志願者",
    "翻译",
    "翻譯",
    "校对",
    "校對",
    "制作",
    "製作",
    "提供",
    "amara.org",
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _denoise_enabled(requested: bool | None) -> bool:
    if requested is not None:
        return requested
    return _env_flag("STT_DENOISE_ENABLED", True)


def _command_error(label: str, result: subprocess.CompletedProcess[str]) -> HTTPException:
    detail = (result.stderr or result.stdout or "unknown error").strip()
    if len(detail) > 1200:
        detail = detail[-1200:]
    return HTTPException(status_code=500, detail=f"{label} failed: {detail}")


def _looks_like_outro_hallucination(text: str) -> bool:
    normalized = "".join(text.lower().split())
    matched_groups = sum(
        any(term in normalized for term in alternatives)
        for alternatives in _OUTRO_TERM_GROUPS
    )
    return matched_groups >= 3


def _looks_like_subtitle_credit_hallucination(text: str) -> bool:
    normalized = "".join(text.lower().split())
    return (
        any(term in normalized for term in _SUBTITLE_TERMS)
        and any(term in normalized for term in _CREDIT_TERMS)
    )


def _reliable_segment(
    segment: Any,
    *,
    minimum_average_log_probability: float,
    maximum_compression_ratio: float,
) -> bool:
    return (
        float(getattr(segment, "avg_logprob", float("-inf")))
        >= minimum_average_log_probability
        and float(getattr(segment, "compression_ratio", float("inf")))
        <= maximum_compression_ratio
        and not _looks_like_outro_hallucination(str(segment.text))
        and not _looks_like_subtitle_credit_hallucination(str(segment.text))
    )


def _denoise_audio(
    source_path: Path,
    workspace: Path,
    *,
    post_filter: bool | None = None,
    post_filter_beta: float | None = None,
    attenuation_limit_db: float | None = None,
    min_db_thresh: float | None = None,
    max_db_erb_thresh: float | None = None,
    max_db_df_thresh: float | None = None,
) -> Path:
    """Normalize audio and enhance it with the native DeepFilterNet3 runtime."""
    binary = Path(os.getenv("STT_DENOISE_BINARY", "/usr/local/bin/deep-filter"))
    model = Path(
        os.getenv(
            "STT_DENOISE_MODEL",
            "/opt/deepfilter/DeepFilterNet3_onnx.tar.gz",
        )
    )
    if not binary.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"STT denoising is enabled but the executable is missing: {binary}",
        )
    if not model.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"STT denoising is enabled but the model is missing: {model}",
        )

    timeout = float(os.getenv("STT_DENOISE_TIMEOUT", "120"))
    normalized_path = workspace / "input.wav"
    output_dir = workspace / "enhanced"
    output_dir.mkdir()

    try:
        normalize = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "48000",
                "-c:a",
                "pcm_s16le",
                str(normalized_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Audio normalization exceeded {timeout:g} seconds.",
        ) from exc
    if normalize.returncode:
        raise _command_error("Audio normalization", normalize)

    post_filter = (
        _env_flag("STT_DENOISE_POST_FILTER")
        if post_filter is None
        else post_filter
    )
    post_filter_beta = (
        float(os.getenv("STT_DENOISE_POST_FILTER_BETA", "0.02"))
        if post_filter_beta is None
        else post_filter_beta
    )
    attenuation_limit_db = (
        float(os.getenv("STT_DENOISE_ATTENUATION_LIMIT_DB", "100"))
        if attenuation_limit_db is None
        else attenuation_limit_db
    )
    min_db_thresh = (
        float(os.getenv("STT_DENOISE_MIN_DB_THRESH", "-15"))
        if min_db_thresh is None
        else min_db_thresh
    )
    max_db_erb_thresh = (
        float(os.getenv("STT_DENOISE_MAX_DB_ERB_THRESH", "35"))
        if max_db_erb_thresh is None
        else max_db_erb_thresh
    )
    max_db_df_thresh = (
        float(os.getenv("STT_DENOISE_MAX_DB_DF_THRESH", "35"))
        if max_db_df_thresh is None
        else max_db_df_thresh
    )

    command = [
        str(binary),
        "--compensate-delay",
        "--model",
        str(model),
        "--output-dir",
        str(output_dir),
        f"--atten-lim-db={attenuation_limit_db}",
        f"--min-db-thresh={min_db_thresh}",
        f"--max-db-erb-thresh={max_db_erb_thresh}",
        f"--max-db-df-thresh={max_db_df_thresh}",
    ]
    if post_filter:
        command.extend(("--pf", f"--pf-beta={post_filter_beta}"))
    command.append(str(normalized_path))

    started = time.monotonic()
    try:
        enhanced = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail=f"STT denoising exceeded {timeout:g} seconds.",
        ) from exc
    if enhanced.returncode:
        raise _command_error("DeepFilterNet3", enhanced)

    output_files = list(output_dir.glob("*.wav"))
    if len(output_files) != 1:
        raise HTTPException(
            status_code=500,
            detail="DeepFilterNet3 did not produce exactly one enhanced WAV file.",
        )
    logger.info(
        "denoised STT input with DeepFilterNet3 in %.2f seconds",
        time.monotonic() - started,
    )
    return output_files[0]


def _resolve_local_model(path: Path) -> Path | None:
    if not path.exists():
        return None
    if (path / "model.bin").is_file():
        return path.resolve()

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    refs_main = path / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        referenced_snapshot = snapshots_dir / revision
        if (referenced_snapshot / "model.bin").is_file():
            return referenced_snapshot.resolve()

    snapshots = sorted(
        (entry for entry in snapshots_dir.iterdir() if entry.is_dir()),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    return next(
        (
            snapshot.resolve()
            for snapshot in snapshots
            if (snapshot / "model.bin").is_file()
        ),
        None,
    )


def _configured_model_path() -> str | None:
    configured_path = os.getenv("VOXBOX_MODEL_PATH", "").strip()
    if not configured_path:
        return None

    path = Path(configured_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((Path("/") / path, Path("/app") / path))

    for candidate in candidates:
        resolved = _resolve_local_model(candidate)
        if resolved is not None:
            return str(resolved)

    checked = ", ".join(str(candidate) for candidate in candidates)
    message = (
        "VOXBOX_MODEL_PATH does not contain a complete faster-whisper "
        f"snapshot; checked: {checked}. A complete model directory must "
        "contain model.bin."
    )
    if _env_flag("VOXBOX_MODEL_PATH_STRICT"):
        raise FileNotFoundError(message)
    logger.warning(
        "%s Falling back to VOXBOX_HF_REPO and VOXBOX_DOWNLOAD_ROOT.",
        message,
    )
    return None


def _model_roots() -> list[Path]:
    configured = os.getenv(
        "STT_MODEL_ROOTS",
        "/models/whisper,/models/voxbox/cache/huggingface",
    )
    roots = [Path(value.strip()) for value in configured.split(",") if value.strip()]
    download_root = os.getenv("VOXBOX_DOWNLOAD_ROOT", "").strip()
    if download_root:
        roots.append(Path(download_root))
    return list(dict.fromkeys(roots))


def _cached_model(model_id: str) -> str | None:
    if "/" not in model_id:
        return None
    cache_name = "models--" + "--".join(model_id.split("/"))
    for root in _model_roots():
        resolved = _resolve_local_model(root / cache_name)
        if resolved is not None:
            return str(resolved)
    return None


def _model_name(requested_model: str | None = None) -> str:
    requested_model = (requested_model or "").strip()
    default_model = os.getenv(
        "VOXBOX_HF_REPO",
        "Systran/faster-whisper-small",
    )
    if requested_model:
        cached = _cached_model(requested_model)
        if cached is not None:
            return cached
        if requested_model == default_model:
            configured = _configured_model_path()
            if configured is not None:
                return configured
        return requested_model

    configured = _configured_model_path()
    return configured if configured is not None else default_model


def _load_model(requested_model: str | None = None) -> Any:
    global _model, _loaded_model_name

    from faster_whisper import WhisperModel

    model_name = _model_name(requested_model)
    if _model is not None and _loaded_model_name == model_name:
        return _model

    device = os.getenv("VOXBOX_DEVICE", "cpu")
    compute_type = os.getenv("VOXBOX_COMPUTE_TYPE", "").strip() or (
        "float16" if device.startswith("cuda") else "int8"
    )
    if _model is not None:
        logger.info("unloading Whisper model %s", _loaded_model_name)
        _model = None
        _loaded_model_name = None
        gc.collect()

    logger.info("loading Whisper model %s on %s", model_name, device)
    _model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=os.getenv("VOXBOX_DOWNLOAD_ROOT") or None,
    )
    _loaded_model_name = model_name
    return _model


def _ensure_cuda_runtime() -> None:
    if not os.getenv("VOXBOX_DEVICE", "cpu").startswith("cuda"):
        return
    try:
        import ctranslate2

        device_count = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        raise RuntimeError(
            "CUDA initialization failed. On Jetson, use an NVIDIA PyTorch "
            "image tagged -igpu that matches the installed JetPack release."
        ) from exc
    if device_count < 1:
        raise RuntimeError(
            "No CUDA device is available to CTranslate2. On Jetson, use an "
            "NVIDIA PyTorch image tagged -igpu that matches JetPack."
        )


def _patch_pyannote_version_check() -> None:
    """Let pyannote 3.x inspect NVIDIA's PEP 440 Torch version.

    pyannote.audio 3.3.2 uses python-semver for checkpoint compatibility
    warnings. Jetson's Torch version (for example ``2.4.0a0+...nv24.7``) is
    valid PEP 440 but not valid SemVer, so model loading otherwise fails before
    any weights are used.
    """
    import pyannote.audio.core.model as model_module
    import pyannote.audio.core.pipeline as pipeline_module
    from pyannote.audio.utils.version import check_version as original_check_version

    if getattr(model_module.check_version, "_jetson_compatible", False):
        return

    def normalize(version: str) -> str:
        from packaging.version import Version

        release = Version(str(version)).release
        return ".".join(str(part) for part in (*release, 0, 0)[:3])

    def compatible_check_version(
        library: str,
        theirs: str,
        mine: str,
        what: str = "Pipeline",
    ) -> None:
        original_check_version(
            library,
            normalize(theirs),
            normalize(mine),
            what=what,
        )

    compatible_check_version._jetson_compatible = True  # type: ignore[attr-defined]
    model_module.check_version = compatible_check_version
    pipeline_module.check_version = compatible_check_version


def _load_diarization_pipeline() -> Any:
    global _diarization_pipeline

    if _diarization_pipeline is not None:
        return _diarization_pipeline

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Speaker diarization is unavailable in this STT image.",
        ) from exc

    model_name = os.getenv(
        "STT_DIARIZATION_MODEL",
        "pyannote/speaker-diarization-3.1",
    )
    token = os.getenv("HF_TOKEN") or None
    if not token and not Path(model_name).exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "HF_TOKEN is required for pyannote speaker diarization. "
                "Accept the speaker-diarization-3.1 and segmentation-3.0 "
                "conditions on Hugging Face first."
            ),
        )

    logger.info("loading diarization pipeline %s", model_name)
    try:
        _patch_pyannote_version_check()
        _diarization_pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=token,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to load {model_name}: {exc}",
        ) from exc
    if _diarization_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to load {model_name}; check HF_TOKEN and model access.",
        )

    device = os.getenv("STT_DIARIZATION_DEVICE", "cuda:0")
    _diarization_pipeline.to(torch.device(device))
    logger.info("loaded diarization pipeline %s on %s", model_name, device)
    return _diarization_pipeline


def _speaker_turns(annotation: Any) -> list[dict[str, Any]]:
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization
    return [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _speaker_for_interval(
    start: float,
    end: float,
    turns: list[dict[str, Any]],
) -> str:
    if not turns:
        return "SPEAKER_UNKNOWN"
    overlap, selected = max(
        (
            (max(0.0, min(end, turn["end"]) - max(start, turn["start"])), turn)
            for turn in turns
        ),
        key=lambda item: item[0],
    )
    if overlap > 0.0:
        return str(selected["speaker"])
    midpoint = (start + end) / 2.0
    nearest = min(
        turns,
        key=lambda turn: min(
            abs(midpoint - turn["start"]),
            abs(midpoint - turn["end"]),
        ),
    )
    return str(nearest["speaker"])


def _diarized_transcript(
    segments: list[Any],
    turns: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    attributed: list[dict[str, Any]] = []
    maximum_gap = float(os.getenv("STT_DIARIZATION_MAX_GAP", "1.0"))
    for segment in segments:
        words = list(getattr(segment, "words", None) or [])
        if not words:
            words = [segment]
        for word in words:
            text = str(getattr(word, "word", getattr(word, "text", "")))
            start = float(getattr(word, "start", segment.start))
            end = float(getattr(word, "end", segment.end))
            speaker = _speaker_for_interval(start, end, turns)
            if (
                attributed
                and attributed[-1]["speaker"] == speaker
                and start - attributed[-1]["end"] <= maximum_gap
            ):
                attributed[-1]["text"] += text
                attributed[-1]["end"] = end
            else:
                attributed.append(
                    {"speaker": speaker, "start": start, "end": end, "text": text}
                )

    for item in attributed:
        item["text"] = item["text"].strip()
    attributed = [item for item in attributed if item["text"]]
    text = "\n".join(
        f'{item["speaker"]}: {item["text"]}' for item in attributed
    )
    return text, attributed


@asynccontextmanager
async def lifespan(_: FastAPI):
    with _MODEL_LOCK:
        _ensure_cuda_runtime()
        _load_model()
    yield


app = FastAPI(title="Local faster-whisper", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    denoise_enabled = _denoise_enabled(None)
    denoise_binary = Path(
        os.getenv("STT_DENOISE_BINARY", "/usr/local/bin/deep-filter")
    )
    denoise_model = Path(
        os.getenv(
            "STT_DENOISE_MODEL",
            "/opt/deepfilter/DeepFilterNet3_onnx.tar.gz",
        )
    )
    return {
        "status": "ok" if _model is not None else "loading",
        "denoiser": (
            "ready"
            if denoise_enabled
            and denoise_binary.is_file()
            and denoise_model.is_file()
            else "missing"
            if denoise_enabled
            else "disabled"
        ),
        "diarizer": "ready" if _diarization_pipeline is not None else "lazy",
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    name = _loaded_model_name or _model_name()
    return {
        "object": "list",
        "data": [{"id": name, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/audio/denoise", response_model=None)
async def denoise_audio(
    file: UploadFile = File(...),
    post_filter: bool = Form(default=False),
    post_filter_beta: float = Form(default=0.02),
    attenuation_limit_db: float = Form(default=100.0),
    min_db_thresh: float = Form(default=-15.0),
    max_db_erb_thresh: float = Form(default=35.0),
    max_db_df_thresh: float = Form(default=35.0),
) -> Response:
    """Return the DeepFilterNet3-enhanced 48 kHz mono WAV."""
    if not 0.0 <= post_filter_beta <= 0.05:
        raise HTTPException(
            status_code=400,
            detail="post_filter_beta must be between 0 and 0.05.",
        )
    if not 0.0 <= attenuation_limit_db <= 100.0:
        raise HTTPException(
            status_code=400,
            detail="attenuation_limit_db must be between 0 and 100.",
        )
    thresholds = (min_db_thresh, max_db_erb_thresh, max_db_df_thresh)
    if any(not -15.0 <= value <= 35.0 for value in thresholds):
        raise HTTPException(
            status_code=400,
            detail="Denoise processing thresholds must be between -15 and 35 dB.",
        )
    if min_db_thresh > min(max_db_erb_thresh, max_db_df_thresh):
        raise HTTPException(
            status_code=400,
            detail="Minimum processing threshold cannot exceed a maximum threshold.",
        )

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    temporary_path = ""
    workspace: tempfile.TemporaryDirectory[str] | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(await file.read())
            temporary_path = temporary.name
        workspace = tempfile.TemporaryDirectory(prefix="stt-denoise-preview-")
        enhanced_path = _denoise_audio(
            Path(temporary_path),
            Path(workspace.name),
            post_filter=post_filter,
            post_filter_beta=post_filter_beta,
            attenuation_limit_db=attenuation_limit_db,
            min_db_thresh=min_db_thresh,
            max_db_erb_thresh=max_db_erb_thresh,
            max_db_df_thresh=max_db_df_thresh,
        )
        return Response(
            content=enhanced_path.read_bytes(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="denoised.wav"',
            },
        )
    finally:
        if workspace is not None:
            workspace.cleanup()
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def _timestamp(seconds: float, separator: str = ",") -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
        f"{separator}{milliseconds:03d}"
    )


def _subtitle(segments: list[Any], *, vtt: bool) -> str:
    blocks: list[str] = ["WEBVTT\n"] if vtt else []
    for index, segment in enumerate(segments, start=1):
        separator = "." if vtt else ","
        timing = (
            f"{_timestamp(segment.start, separator)} --> "
            f"{_timestamp(segment.end, separator)}"
        )
        if not vtt:
            blocks.append(str(index))
        blocks.extend((timing, segment.text.strip(), ""))
    return "\n".join(blocks)


@app.post("/v1/audio/transcriptions", response_model=None)
async def transcribe(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    prompt: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    vad_filter: bool = Form(default=True),
    vad_threshold: float = Form(default=0.5),
    vad_min_speech_ms: int = Form(default=250),
    vad_min_silence_ms: int = Form(default=2000),
    vad_speech_pad_ms: int = Form(default=400),
    no_speech_threshold: float = Form(default=0.6),
    strict_no_speech_filter: bool = Form(default=False),
    minimum_average_log_probability: float = Form(default=-1.0),
    maximum_compression_ratio: float = Form(default=2.4),
    denoise: bool | None = Form(default=None),
    diarize: bool | None = Form(default=None),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
) -> Response:
    if not 0.0 <= no_speech_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="no_speech_threshold must be between 0 and 1.",
        )
    if min_speakers is not None and min_speakers < 1:
        raise HTTPException(status_code=400, detail="min_speakers must be positive.")
    if max_speakers is not None and max_speakers < 1:
        raise HTTPException(status_code=400, detail="max_speakers must be positive.")
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise HTTPException(
            status_code=400,
            detail="min_speakers cannot exceed max_speakers.",
        )
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    temporary_path = ""
    denoise_workspace: tempfile.TemporaryDirectory[str] | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(await file.read())
            temporary_path = temporary.name

        transcription_path = Path(temporary_path)
        if _denoise_enabled(denoise):
            denoise_workspace = tempfile.TemporaryDirectory(prefix="stt-denoise-")
            transcription_path = _denoise_audio(
                transcription_path,
                Path(denoise_workspace.name),
            )

        with _MODEL_LOCK:
            active_model = _load_model(model)
            segments_iter, info = active_model.transcribe(
                str(transcription_path),
                language=language or os.getenv("STT_LANGUAGE") or None,
                initial_prompt=prompt,
                temperature=temperature,
                log_prob_threshold=minimum_average_log_probability,
                compression_ratio_threshold=maximum_compression_ratio,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=False,
                repetition_penalty=1.05,
                word_timestamps=True,
                hallucination_silence_threshold=1.0,
                vad_filter=vad_filter,
                vad_parameters={
                    "threshold": vad_threshold,
                    "min_speech_duration_ms": vad_min_speech_ms,
                    "min_silence_duration_ms": vad_min_silence_ms,
                    "speech_pad_ms": vad_speech_pad_ms,
                },
            )
            decoded_segments = list(segments_iter)
            segments = [
                segment
                for segment in decoded_segments
                if _reliable_segment(
                    segment,
                    minimum_average_log_probability=(
                        minimum_average_log_probability
                    ),
                    maximum_compression_ratio=maximum_compression_ratio,
                )
            ]
            if strict_no_speech_filter:
                segments = [
                    segment
                    for segment in decoded_segments
                    if float(getattr(segment, "no_speech_prob", 0.0))
                    <= no_speech_threshold
                ]
            filtered_count = len(decoded_segments) - len(segments)
            if filtered_count:
                logger.info(
                    "removed %d unreliable Whisper segment(s); "
                    "min_logprob=%.2f max_compression=%.2f no_speech=%.2f strict=%s",
                    filtered_count,
                    minimum_average_log_probability,
                    maximum_compression_ratio,
                    no_speech_threshold,
                    strict_no_speech_filter,
                )
        diarization_turns: list[dict[str, Any]] = []
        diarized_segments: list[dict[str, Any]] = []
        use_diarization = (
            _env_flag("STT_DIARIZATION_ENABLED")
            if diarize is None
            else diarize
        )
        if use_diarization:
            diarization_options = {
                key: value
                for key, value in {
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                }.items()
                if value is not None
            }
            with _DIARIZATION_LOCK:
                diarization_pipeline = _load_diarization_pipeline()
                annotation = diarization_pipeline(
                    str(transcription_path),
                    **diarization_options,
                )
            diarization_turns = _speaker_turns(annotation)
            text, diarized_segments = _diarized_transcript(
                segments,
                diarization_turns,
            )
        else:
            text = "".join(segment.text for segment in segments).strip()

        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "srt":
            return PlainTextResponse(_subtitle(segments, vtt=False))
        if response_format == "vtt":
            return PlainTextResponse(_subtitle(segments, vtt=True))
        if response_format == "verbose_json":
            return JSONResponse(
                {
                    "task": "transcribe",
                    "language": info.language,
                    "duration": info.duration,
                    "text": text,
                    "diarization": diarization_turns,
                    "speaker_segments": diarized_segments,
                    "segments": [
                        {
                            "id": index,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                            "no_speech_probability": segment.no_speech_prob,
                            "average_log_probability": segment.avg_logprob,
                            "compression_ratio": segment.compression_ratio,
                        }
                        for index, segment in enumerate(segments)
                    ],
                }
            )
        if response_format != "json":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported response_format: {response_format}",
            )
        response: dict[str, Any] = {"text": text}
        if use_diarization:
            response["diarization"] = diarization_turns
            response["speaker_segments"] = diarized_segments
        return JSONResponse(response)
    finally:
        if denoise_workspace is not None:
            denoise_workspace.cleanup()
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("STT_BIND_PORT", "8000")),
    )


if __name__ == "__main__":
    main()

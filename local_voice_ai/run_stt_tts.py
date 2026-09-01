"""Gradio tester for local STT and TTS endpoints.

This script is intentionally a client only. Start the local service stack first,
then run this app on the host and use the browser microphone to round-trip:

    mic/file audio -> STT endpoint -> transcript -> TTS endpoint -> playback
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import gradio as gr
import httpx
import soundfile as sf


STT_PUBLIC_PORT = int(os.getenv("STT_PUBLIC_PORT", "8000"))
TTS_PUBLIC_PORT = int(os.getenv("TTS_PUBLIC_PORT", "8800"))
LLM_PUBLIC_PORT = int(os.getenv("LLM_PUBLIC_PORT", "11434"))
DEFAULT_STT_BASE_URL = os.getenv("STT_BASE_URL", f"http://127.0.0.1:{STT_PUBLIC_PORT}/v1")
DEFAULT_TTS_BASE_URL = os.getenv("TTS_BASE_URL", f"http://127.0.0.1:{TTS_PUBLIC_PORT}/v1")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", f"http://127.0.0.1:{LLM_PUBLIC_PORT}/v1")
FASTER_WHISPER_SMALL_MODEL = "Systran/faster-whisper-small"
FASTER_WHISPER_LARGE_MODEL = "Systran/faster-whisper-large-v3"
DEFAULT_STT_MODELS = list(
    dict.fromkeys(
        [
            os.getenv("VOXBOX_HF_REPO", FASTER_WHISPER_SMALL_MODEL),
            FASTER_WHISPER_SMALL_MODEL,
            FASTER_WHISPER_LARGE_MODEL,
            "nemotron-3.5-asr-streaming",
        ]
    )
)
DEFAULT_TTS_MODELS = [
    "bluemagpie-tts",
]
DEFAULT_USE_DENOISE = os.getenv("STT_DENOISE_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_USE_DIARIZATION = os.getenv(
    "STT_DIARIZATION_ENABLED",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}


def _default_llm_model() -> str | None:
    model_path = os.getenv("LLM_MODEL_PATH", "").strip()
    if model_path:
        return Path(model_path).name
    for name in ("LLM_MODEL", "LLM_HF_REPO"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


_DEFAULT_LLM_MODEL = _default_llm_model()
DEFAULT_LLM_MODELS = [_DEFAULT_LLM_MODEL] if _DEFAULT_LLM_MODEL else []
KB_PATH = Path(os.getenv("KB_PATH", "/app/knowledge/system-prompt.md"))
RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", "").strip()
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY", "").strip()
RAGFLOW_DATASET_IDS = [
    dataset_id.strip()
    for dataset_id in os.getenv("RAGFLOW_DATASET_IDS", "").split(",")
    if dataset_id.strip()
]
RAGFLOW_PAGE_SIZE = int(os.getenv("RAGFLOW_PAGE_SIZE", "8"))
RAGFLOW_SIMILARITY_THRESHOLD = float(
    os.getenv("RAGFLOW_SIMILARITY_THRESHOLD", "0.2")
)
RAGFLOW_VECTOR_WEIGHT = float(os.getenv("RAGFLOW_VECTOR_WEIGHT", "0.3"))
RAGFLOW_MAX_CONTEXT_CHARS = int(os.getenv("RAGFLOW_MAX_CONTEXT_CHARS", "12000"))
DEFAULT_VOICES = [
    "hung_yi_lee",
    "female_voice",
]
CLONE_MODE_CHOICES = [
    ("A — Reference WAV (experimental)", "reference_wav_path"),
    ("B — Speaker embedding (recommended)", "speaker_centroid"),
]
DEFAULT_CLONE_MODE = "speaker_centroid"
VOICE_CLONE_DIR = Path(os.getenv("VOICE_CLONE_DIR", "/models/voice_clones"))
STT_MODEL_ROOTS = [
    Path(value.strip())
    for value in os.getenv(
        "STT_MODEL_ROOTS",
        "/models/whisper,/models/voxbox/cache/huggingface",
    ).split(",")
    if value.strip()
]
_MANIFEST_LOCK = threading.Lock()
MIN_CLONE_REFERENCE_SECONDS = 3.0
CENTROID_WINDOW_SECONDS = 6.0
MIN_CENTROID_CHUNK_SECONDS = 1.0
SPEAKER_EMBEDDING_VERSION = "bluemagpie-ecapa-windowed-6s-v2"
_LAST_USED_LOCK = threading.Lock()
_LAST_USED_STATE: dict[str, str | None] = {
    "ip": None,
    "action": None,
    "time": None,
    "duration": None,
}
_TAIPEI_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Taipei")
_APP_CSS = """
#app-header {
    align-items: center;
    gap: 1rem;
}
#app-title h1 {
    margin: 0;
}
#app-status {
    align-items: flex-start;
    flex-wrap: nowrap;
    gap: 1rem;
}
#last-used {
    width: 100%;
    min-height: 0;
    padding: 0.25rem 0;
    text-align: right;
    color: var(--body-text-color-subdued);
    font-size: 0.85rem;
    white-space: normal;
    overflow-wrap: anywhere;
}
#input-audio-filename {
    width: 100%;
    min-height: 0;
    padding: 0;
    color: var(--body-text-color-subdued);
    font-size: 0.85rem;
    overflow-wrap: anywhere;
}
@media (max-width: 700px) {
    #last-used {
        text-align: right;
    }
}
"""


def _last_used_html() -> str:
    with _LAST_USED_LOCK:
        ip = _LAST_USED_STATE["ip"]
        action = _LAST_USED_STATE["action"]
        used_at = _LAST_USED_STATE["time"]
        duration = _LAST_USED_STATE["duration"]

    if ip is None or action is None or used_at is None or duration is None:
        return "<span>Last used: never</span>"
    return (
        f"<span>Last used by <strong>{html.escape(ip)}"
        f" · {html.escape(action)}"
        f" · {html.escape(duration)}</strong>"
        f" · {html.escape(used_at)}</span>"
    )


def _request_ip(request: gr.Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()

    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip

    client = getattr(request, "client", None)
    return str(getattr(client, "host", None) or "unknown")


def _record_last_used(
    action: str,
    request: gr.Request,
    duration_s: float,
) -> str:
    with _LAST_USED_LOCK:
        _LAST_USED_STATE["ip"] = _request_ip(request)
        _LAST_USED_STATE["action"] = action
        _LAST_USED_STATE["time"] = datetime.now(_TAIPEI_TIMEZONE).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        _LAST_USED_STATE["duration"] = f"used {duration_s:.1f} sec"
    return _last_used_html()


def _run_recorded_action(
    action: str,
    request: gr.Request,
    function: Any,
    *args: Any,
) -> Any:
    started_at = perf_counter()
    try:
        return function(*args)
    finally:
        _record_last_used(action, request, perf_counter() - started_at)


def _endpoint_defaults_for_request(request: gr.Request) -> tuple[str, str, str]:
    """Use the hostname that the browser used to reach Gradio."""
    headers = request.headers
    authority = (
        headers.get("x-forwarded-host")
        or headers.get("host")
        or "127.0.0.1"
    ).split(",", 1)[0].strip()
    hostname = urlsplit(f"//{authority}").hostname or "127.0.0.1"
    display_host = f"[{hostname}]" if ":" in hostname else hostname
    scheme = headers.get("x-forwarded-proto", "http").split(",", 1)[0].strip()

    stt_url = os.getenv(
        "STT_BASE_URL",
        f"{scheme}://{display_host}:{STT_PUBLIC_PORT}/v1",
    )
    tts_url = os.getenv(
        "TTS_BASE_URL",
        f"{scheme}://{display_host}:{TTS_PUBLIC_PORT}/v1",
    )
    llm_url = os.getenv(
        "LLM_BASE_URL",
        f"{scheme}://{display_host}:{LLM_PUBLIC_PORT}/v1",
    )
    return stt_url, tts_url, llm_url


def _csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


def _llm_model_choices(
    llm_base_url: str,
) -> list[tuple[str, str]]:
    configured = [
        model.strip()
        for model in _csv_env("LLM_MODEL_OPTIONS", DEFAULT_LLM_MODELS)
        if model and model.strip()
    ]
    model_ids: list[str] = []
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(llm_base_url.rstrip("/") + "/models")
            response.raise_for_status()
        data = response.json().get("data", [])
        model_ids = [
            str(entry["id"]).strip()
            for entry in data
            if isinstance(entry, dict) and str(entry.get("id", "")).strip()
        ]
    except (httpx.HTTPError, TypeError, ValueError):
        pass

    model_ids = list(dict.fromkeys([*model_ids, *configured]))
    return [
        (
            Path(model_id).name if model_id.startswith("/") else model_id,
            model_id,
        )
        for model_id in model_ids
    ]


def _stt_model_id(model_file: Path, root: Path) -> str:
    """Convert a Hugging Face cache directory into its repository ID."""
    for parent in model_file.parents:
        if parent == root.parent:
            break
        if parent.name.startswith("models--"):
            encoded_id = parent.name.removeprefix("models--")
            return "/".join(encoded_id.split("--"))
    return str(model_file.parent)


def _stt_model_options() -> list[str]:
    """Discover complete faster-whisper models in the shared model mount."""
    discovered: set[str] = set()
    for root in STT_MODEL_ROOTS:
        if not root.is_dir():
            continue
        try:
            discovered.update(
                _stt_model_id(model_file, root)
                for model_file in root.rglob("model.bin")
                if model_file.is_file()
            )
        except OSError:
            continue

    if discovered:
        models = sorted(discovered, key=str.casefold)
    else:
        models = _csv_env("STT_MODEL_OPTIONS", DEFAULT_STT_MODELS)
    return list(
        dict.fromkeys(
            [
                *models,
                FASTER_WHISPER_SMALL_MODEL,
                FASTER_WHISPER_LARGE_MODEL,
            ]
        )
    )


def _configured_stt_model() -> str:
    return os.getenv("VOXBOX_HF_REPO", "").strip()


def _load_clone_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(
            (VOICE_CLONE_DIR / "manifest.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"voices": []}
    return manifest if isinstance(manifest.get("voices"), list) else {"voices": []}


def _voice_options(defaults: list[str]) -> list[str]:
    clone_ids = [
        str(entry["id"])
        for entry in _load_clone_manifest()["voices"]
        if entry.get("id")
        and (VOICE_CLONE_DIR / str(entry.get("reference_wav", ""))).is_file()
    ]
    return list(dict.fromkeys([*defaults, *clone_ids]))


def _save_browser_settings(
    stt_base_url: str,
    tts_base_url: str,
    llm_base_url: str,
    stt_model: str,
    tts_model: str,
    llm_model: str,
    llm_preprompt: str,
    voice: str,
    clone_mode: str,
    language: str,
    use_llm: bool,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    response_format: str,
    timeout_s: float,
    cfg_value: float,
    inference_timesteps: int,
    seed: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    transcript: str,
    llm_transcript: str,
) -> dict[str, Any]:
    return {
        "stt_base_url": stt_base_url,
        "tts_base_url": tts_base_url,
        "llm_base_url": llm_base_url,
        "stt_model": stt_model,
        "tts_model": tts_model,
        "llm_model": llm_model,
        "llm_preprompt": llm_preprompt,
        "voice": voice,
        "clone_mode": clone_mode,
        "language": language,
        "use_llm": use_llm,
        "use_denoise": use_denoise,
        "denoise_post_filter": denoise_post_filter,
        "denoise_post_filter_beta": denoise_post_filter_beta,
        "denoise_attenuation_limit_db": denoise_attenuation_limit_db,
        "denoise_min_db_thresh": denoise_min_db_thresh,
        "denoise_max_db_erb_thresh": denoise_max_db_erb_thresh,
        "denoise_max_db_df_thresh": denoise_max_db_df_thresh,
        "use_vad": use_vad,
        "no_speech_threshold": no_speech_threshold,
        "vad_threshold": vad_threshold,
        "vad_min_speech_ms": vad_min_speech_ms,
        "vad_min_silence_ms": vad_min_silence_ms,
        "vad_speech_pad_ms": vad_speech_pad_ms,
        "response_format": response_format,
        "timeout_s": timeout_s,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "seed": seed,
        "min_len": min_len,
        "max_len": max_len,
        "retry_badcase": retry_badcase,
        "transcript": transcript,
        "llm_transcript": llm_transcript,
    }


def _restore_browser_settings(
    saved: dict[str, Any] | None,
    request: gr.Request,
) -> tuple[Any, ...]:
    settings = saved if isinstance(saved, dict) else {}
    default_stt_url, default_tts_url, default_llm_url = (
        _endpoint_defaults_for_request(request)
    )
    stt_base_url = str(settings.get("stt_base_url") or default_stt_url)
    tts_base_url = str(settings.get("tts_base_url") or default_tts_url)
    llm_base_url = str(settings.get("llm_base_url") or default_llm_url)
    stt_models = _stt_model_options()
    tts_models = _csv_env("TTS_MODEL_OPTIONS", DEFAULT_TTS_MODELS)
    llm_model_choices = _llm_model_choices(llm_base_url)
    llm_model_values = [value for _, value in llm_model_choices]
    voices = _voice_options(_csv_env("TTS_VOICE_OPTIONS", DEFAULT_VOICES))
    configured_stt_model = _configured_stt_model()

    saved_stt_model = str(settings.get("stt_model", ""))
    stt_model = next(
        (
            candidate
            for candidate in (saved_stt_model, configured_stt_model, stt_models[0])
            if candidate in stt_models
        ),
        stt_models[0],
    )
    saved_voice = str(settings.get("voice", ""))
    voice = saved_voice if saved_voice in voices else voices[0]
    saved_llm_model = str(settings.get("llm_model", ""))
    if not llm_model_values:
        llm_model = saved_llm_model or None
        if saved_llm_model:
            llm_model_choices = [(Path(saved_llm_model).name, saved_llm_model)]
    elif not saved_llm_model:
        llm_model = llm_model_values[0]
    elif saved_llm_model in llm_model_values:
        llm_model = saved_llm_model
    else:
        saved_basename = Path(saved_llm_model).name
        llm_model = next(
            (
                value
                for label, value in llm_model_choices
                if label == saved_basename
            ),
            saved_llm_model,
        )
    llm_preprompt = (
        str(settings["llm_preprompt"] or _knowledge_base_prompt())
        if "llm_preprompt" in settings
        else _knowledge_base_prompt()
    )
    clone_mode = str(settings.get("clone_mode", DEFAULT_CLONE_MODE))
    if clone_mode not in {value for _, value in CLONE_MODE_CHOICES}:
        clone_mode = DEFAULT_CLONE_MODE
    response_formats = ["wav", "mp3", "flac", "opus", "aac"]
    response_format = str(settings.get("response_format", "wav"))
    if response_format not in response_formats:
        response_format = "wav"

    return (
        stt_base_url,
        tts_base_url,
        llm_base_url,
        gr.Dropdown(choices=stt_models, value=stt_model),
        str(settings.get("tts_model") or tts_models[0]),
        gr.Dropdown(choices=llm_model_choices, value=llm_model),
        llm_preprompt,
        gr.Dropdown(choices=voices, value=voice),
        clone_mode,
        str(settings.get("language") or os.getenv("STT_LANGUAGE", "zh")),
        (
            bool(settings["use_llm"])
            if "use_llm" in settings
            else not bool(settings.get("bypass_llm", False))
        ),
        bool(settings.get("use_denoise", DEFAULT_USE_DENOISE)),
        bool(settings.get("denoise_post_filter", False)),
        settings.get("denoise_post_filter_beta", 0.02),
        settings.get("denoise_attenuation_limit_db", 100.0),
        settings.get("denoise_min_db_thresh", -15.0),
        settings.get("denoise_max_db_erb_thresh", 35.0),
        settings.get("denoise_max_db_df_thresh", 35.0),
        bool(settings.get("use_vad", True)),
        settings.get("no_speech_threshold", 0.6),
        settings.get("vad_threshold", 0.5),
        settings.get("vad_min_speech_ms", 250),
        settings.get("vad_min_silence_ms", 2000),
        settings.get("vad_speech_pad_ms", 400),
        response_format,
        settings.get("timeout_s", 180),
        settings.get("cfg_value", 2.0),
        settings.get(
            "inference_timesteps",
            int(os.getenv("BLUEMAGPIE_INFERENCE_TIMESTEPS", "10")),
        ),
        settings.get("seed", int(os.getenv("BLUEMAGPIE_SEED", "1729"))),
        settings.get("min_len", 2),
        settings.get("max_len", 2000),
        bool(settings.get("retry_badcase", False)),
        str(settings.get("transcript") or ""),
        str(settings.get("llm_transcript") or ""),
    )


def _clone_id(name: str) -> str:
    clone_id = re.sub(r"[^\w-]+", "_", (name or "").strip(), flags=re.UNICODE)
    clone_id = clone_id.strip("_").lower()[:64]
    if not clone_id:
        raise gr.Error("Enter a clone voice name.")
    if clone_id in DEFAULT_VOICES:
        raise gr.Error(f"The built-in voice name {clone_id!r} cannot be replaced.")
    return clone_id


def _save_clone_profile(
    name: str,
    audio: Any,
    clone_mode: str,
) -> tuple[str, str]:
    clone_id = _clone_id(name)
    if clone_mode not in {value for _, value in CLONE_MODE_CHOICES}:
        raise gr.Error("Select a valid voice cloning mode.")
    wav_bytes = _audio_to_wav_bytes(
        audio,
        min_duration_s=MIN_CLONE_REFERENCE_SECONDS,
    )
    VOICE_CLONE_DIR.mkdir(parents=True, exist_ok=True)
    wav_name = f"{clone_id}.wav"
    centroid_name = f"{clone_id}.pt"
    wav_path = VOICE_CLONE_DIR / wav_name
    centroid_path = VOICE_CLONE_DIR / centroid_name
    archive_path = VOICE_CLONE_DIR / f"{clone_id}.zip"

    with _MANIFEST_LOCK:
        with tempfile.NamedTemporaryFile(
            dir=VOICE_CLONE_DIR,
            suffix=".wav.tmp",
            delete=False,
        ) as temporary_wav:
            temporary_wav.write(wav_bytes)
            temporary_wav_path = Path(temporary_wav.name)
        os.replace(temporary_wav_path, wav_path)
        centroid_path.unlink(missing_ok=True)

        profile = {
            "id": clone_id,
            "name": name.strip(),
            "reference_wav": wav_name,
            "speaker_centroid": centroid_name,
            "speaker_embedding_version": SPEAKER_EMBEDDING_VERSION,
            "provider": "bluemagpie",
            "mode": clone_mode,
        }
        manifest = _load_clone_manifest()
        manifest["voices"] = [
            entry for entry in manifest["voices"] if entry.get("id") != clone_id
        ]
        manifest["voices"].append(profile)

        manifest_tmp = VOICE_CLONE_DIR / "manifest.json.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, VOICE_CLONE_DIR / "manifest.json")

        archive_tmp = VOICE_CLONE_DIR / f"{clone_id}.zip.tmp"
        with zipfile.ZipFile(archive_tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(wav_path, arcname="reference.wav")
            archive.writestr(
                "profile.json",
                json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            )
        os.replace(archive_tmp, archive_path)

    return clone_id, str(archive_path)


def delete_clone_voice(voice_id: str) -> tuple[Any, None]:
    voice_id = (voice_id or "").strip()
    if voice_id in DEFAULT_VOICES:
        raise gr.Error(f"The built-in voice {voice_id!r} cannot be deleted.")
    if not voice_id:
        raise gr.Error("Select a cloned voice to delete.")

    with _MANIFEST_LOCK:
        manifest = _load_clone_manifest()
        matching_profiles = [
            entry for entry in manifest["voices"] if entry.get("id") == voice_id
        ]
        if not matching_profiles:
            raise gr.Error(f"{voice_id!r} is not a cloned voice.")

        files_to_delete = {f"{voice_id}.zip"}
        files_to_delete.update(
            str(entry.get("reference_wav", ""))
            for entry in matching_profiles
            if entry.get("reference_wav")
        )
        files_to_delete.update(
            str(entry.get("speaker_centroid", ""))
            for entry in matching_profiles
            if entry.get("speaker_centroid")
        )
        clone_root = VOICE_CLONE_DIR.resolve()
        for relative_name in files_to_delete:
            target = (VOICE_CLONE_DIR / relative_name).resolve()
            if not target.is_relative_to(clone_root):
                raise gr.Error("The cloned voice manifest contains an unsafe path.")
            target.unlink(missing_ok=True)

        manifest["voices"] = [
            entry for entry in manifest["voices"] if entry.get("id") != voice_id
        ]
        manifest_tmp = VOICE_CLONE_DIR / "manifest.json.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, VOICE_CLONE_DIR / "manifest.json")

    choices = _voice_options(_csv_env("TTS_VOICE_OPTIONS", DEFAULT_VOICES))
    return gr.Dropdown(choices=choices, value=choices[0]), None


def _audio_to_wav_bytes(
    audio: Any,
    *,
    min_duration_s: float = 0.0,
) -> bytes:
    if audio is None:
        raise gr.Error("Record or upload audio first.")

    if isinstance(audio, str):
        try:
            data, sample_rate = sf.read(audio, always_2d=False)
        except (OSError, RuntimeError) as soundfile_error:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav") as converted:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            audio,
                            "-c:a",
                            "pcm_s16le",
                            converted.name,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    data, sample_rate = sf.read(
                        converted.name,
                        always_2d=False,
                    )
            except FileNotFoundError as exc:
                raise gr.Error(
                    "Unable to decode the input audio because FFmpeg is not installed."
                ) from exc
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or "").strip()
                message = detail or str(soundfile_error)
                raise gr.Error(
                    f"Unable to decode the input audio: {message}"
                ) from exc
            except (OSError, RuntimeError) as exc:
                raise gr.Error(f"Unable to decode the input audio: {exc}") from exc
    elif isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, data = audio
    else:
        raise gr.Error(f"Unsupported audio input: {type(audio).__name__}")

    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise gr.Error("The input audio has an invalid sample rate.")

    duration_s = len(data) / sample_rate
    if duration_s < min_duration_s:
        raise gr.Error(
            f"Voice cloning needs at least {min_duration_s:g} seconds of clean "
            f"speech; this recording is {duration_s:.1f} seconds."
        )

    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _input_audio_filename_html(audio: Any) -> str:
    if not audio:
        return "<span>Opened file: <strong>none</strong></span>"
    if isinstance(audio, str):
        filename = Path(audio).name
    else:
        filename = "microphone recording"
    return (
        "<span>Opened file: "
        f"<strong>{html.escape(filename)}</strong></span>"
    )


def _show_clone_audio_info() -> None:
    gr.Info(
        "For optimal voice cloning, use about 3 minutes of clean speech from "
        "one speaker with minimal background noise.",
        title="Clone voice reference audio",
    )


def _response_text(response: httpx.Response) -> str:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = response.json()
        return str(payload.get("text", "")).strip()
    return response.text.strip()


def _knowledge_base_prompt() -> str:
    try:
        prompt = KB_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise gr.Error(f"Unable to read knowledge base {KB_PATH}: {exc}") from exc
    if not prompt:
        raise gr.Error(f"Knowledge base {KB_PATH} is empty.")
    return prompt


def _local_wiki_context() -> str:
    """Load the wiki as a no-RAG fallback for unconfigured local development."""
    wiki_root = KB_PATH.parent
    pages = []
    for path in sorted(wiki_root.rglob("*.md")):
        if path == KB_PATH or path.name == "README.md":
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise gr.Error(f"Unable to read knowledge page {path}: {exc}") from exc
        if content:
            pages.append(f"## Source: {path.relative_to(wiki_root)}\n\n{content}")
    return "\n\n".join(pages)


def _ragflow_context(text: str, timeout_s: float) -> str:
    """Retrieve the wiki chunks relevant to a transcript from RAGFlow."""
    if not RAGFLOW_API_KEY and not RAGFLOW_DATASET_IDS:
        return _local_wiki_context()
    if not RAGFLOW_BASE_URL or not RAGFLOW_API_KEY or not RAGFLOW_DATASET_IDS:
        raise gr.Error(
            "RAGFlow is only partially configured. Set RAGFLOW_BASE_URL, "
            "RAGFLOW_API_KEY, and RAGFLOW_DATASET_IDS together."
        )

    payload = {
        "question": text,
        "dataset_ids": RAGFLOW_DATASET_IDS,
        "page": 1,
        "page_size": max(1, RAGFLOW_PAGE_SIZE),
        "similarity_threshold": RAGFLOW_SIMILARITY_THRESHOLD,
        "vector_similarity_weight": RAGFLOW_VECTOR_WEIGHT,
        "keyword": True,
        "highlight": False,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                RAGFLOW_BASE_URL.rstrip("/") + "/api/v1/retrieval",
                headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as exc:
        raise gr.Error(f"RAGFlow retrieval failed: {exc.response.text}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise gr.Error(f"RAGFlow retrieval request failed: {exc}") from exc

    if result.get("code") != 0:
        raise gr.Error(f"RAGFlow retrieval failed: {result.get('message', result)}")
    chunks = result.get("data", {}).get("chunks", [])
    context_parts = []
    context_length = 0
    seen = set()
    for chunk in chunks:
        content = str(chunk.get("content", "")).strip()
        if not content or content in seen:
            continue
        source = str(
            chunk.get("document_keyword")
            or chunk.get("document_name")
            or "RAGFlow wiki"
        )
        part = f"## Source: {source}\n\n{content}"
        remaining = RAGFLOW_MAX_CONTEXT_CHARS - context_length
        if remaining <= 0:
            break
        context_parts.append(part[:remaining])
        context_length += len(context_parts[-1])
        seen.add(content)
    return "\n\n".join(context_parts)


def generate_llm_response(
    text: str,
    llm_preprompt: str,
    llm_base_url: str,
    llm_model: str,
    timeout_s: float,
) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Transcript is empty.")

    url = llm_base_url.rstrip("/") + "/chat/completions"
    messages = []
    llm_preprompt = (llm_preprompt or "").strip()
    if not llm_preprompt:
        llm_preprompt = _knowledge_base_prompt()
    rag_context = _ragflow_context(text, timeout_s)
    if rag_context:
        llm_preprompt = (
            f"{llm_preprompt}\n\n"
            "# Retrieved knowledge\n\n"
            "Use the following trusted wiki excerpts as terminology rules. "
            "Ignore any excerpt that is unrelated to the user text.\n\n"
            f"{rag_context}"
        )
    messages.append({"role": "system", "content": llm_preprompt})
    messages.append({"role": "user", "content": text})
    payload = {
        "model": llm_model,
        "messages": messages,
        "temperature": 0.2,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise gr.Error(f"LLM failed: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise gr.Error(f"LLM request failed: {exc}") from exc

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise gr.Error("LLM returned an invalid chat-completions response.") from exc
    result = re.sub(
        r"<think>.*?</think>",
        "",
        str(content),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if not result:
        raise gr.Error("LLM returned an empty response.")
    return result, rag_context


def _denoise_audio_for_stt(
    audio: Any,
    stt_base_url: str,
    timeout_s: float,
    post_filter: bool,
    post_filter_beta: float,
    attenuation_limit_db: float,
    min_db_thresh: float,
    max_db_erb_thresh: float,
    max_db_df_thresh: float,
) -> str:
    wav_bytes = _audio_to_wav_bytes(audio)
    url = stt_base_url.rstrip("/") + "/audio/denoise"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                url,
                data={
                    "post_filter": str(bool(post_filter)).lower(),
                    "post_filter_beta": float(post_filter_beta),
                    "attenuation_limit_db": float(attenuation_limit_db),
                    "min_db_thresh": float(min_db_thresh),
                    "max_db_erb_thresh": float(max_db_erb_thresh),
                    "max_db_df_thresh": float(max_db_df_thresh),
                },
                files={"file": ("input.wav", wav_bytes, "audio/wav")},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise gr.Error(
                "The STT server does not have the /audio/denoise endpoint. "
                "Restart the STT container with the updated whisper_server.py."
            ) from exc
        raise gr.Error(f"STT denoise failed: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise gr.Error(f"STT denoise request failed: {exc}") from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as output:
        output.write(response.content)
        return output.name


def transcribe_audio(
    audio: Any,
    use_diarization: bool,
    stt_base_url: str,
    stt_model: str,
    language: str,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    timeout_s: float,
) -> tuple[str, str | None]:
    denoised_audio = None
    transcription_audio = audio
    if use_denoise:
        denoised_audio = _denoise_audio_for_stt(
            audio,
            stt_base_url,
            timeout_s,
            denoise_post_filter,
            denoise_post_filter_beta,
            denoise_attenuation_limit_db,
            denoise_min_db_thresh,
            denoise_max_db_erb_thresh,
            denoise_max_db_df_thresh,
        )
        transcription_audio = denoised_audio

    wav_bytes = _audio_to_wav_bytes(transcription_audio)
    url = stt_base_url.rstrip("/") + "/audio/transcriptions"
    data = {
        "model": stt_model,
        "response_format": "json",
        # Gradio calls the preview endpoint first so the exact enhanced WAV can
        # be displayed and reused here without running DeepFilterNet twice.
        "denoise": "false",
        "diarize": str(bool(use_diarization)).lower(),
        "vad_filter": str(bool(use_vad)).lower(),
        "no_speech_threshold": float(
            no_speech_threshold if use_denoise else 0.6
        ),
        "strict_no_speech_filter": str(bool(use_denoise)).lower(),
        "vad_threshold": float(vad_threshold),
        "vad_min_speech_ms": int(vad_min_speech_ms),
        "vad_min_silence_ms": int(vad_min_silence_ms),
        "vad_speech_pad_ms": int(vad_speech_pad_ms),
    }
    if language:
        data["language"] = language

    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                url,
                data=data,
                files={"file": ("input.wav", wav_bytes, "audio/wav")},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise gr.Error(f"STT failed: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise gr.Error(f"STT request failed: {exc}") from exc

    text = _response_text(response)
    if not text:
        return "", denoised_audio
    return text, denoised_audio


def transcribe_audio_chunks(
    audio: Any,
    stt_base_url: str,
    stt_model: str,
    language: str,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    timeout_s: float,
) -> str:
    """Transcribe short non-overlapping windows for clone preview text."""
    wav_bytes = _audio_to_wav_bytes(audio)
    data, sample_rate = sf.read(
        io.BytesIO(wav_bytes),
        always_2d=False,
    )
    window_size = max(int(CENTROID_WINDOW_SECONDS * sample_rate), 1)
    minimum_size = max(int(MIN_CENTROID_CHUNK_SECONDS * sample_rate), 1)
    chunks = [
        data[start : start + window_size]
        for start in range(0, len(data), window_size)
        if len(data[start : start + window_size]) >= minimum_size
    ]
    if not chunks and len(data):
        chunks = [data]

    transcripts = [
        transcribe_audio(
            (sample_rate, chunk),
            False,
            stt_base_url,
            stt_model,
            language,
            use_denoise,
            denoise_post_filter,
            denoise_post_filter_beta,
            denoise_attenuation_limit_db,
            denoise_min_db_thresh,
            denoise_max_db_erb_thresh,
            denoise_max_db_df_thresh,
            use_vad,
            no_speech_threshold,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
            vad_speech_pad_ms,
            timeout_s,
        )[0]
        for chunk in chunks
    ]
    return "\n".join(transcripts)


def synthesize_text(
    text: str,
    tts_base_url: str,
    tts_model: str,
    voice: str,
    clone_mode: str,
    response_format: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    seed: int,
    timeout_s: float,
    reference_audio: Any = None,
) -> str:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Transcript/text is empty.")

    url = tts_base_url.rstrip("/") + "/audio/speech"
    payload = {
        "model": tts_model,
        "input": text,
        "voice": voice,
        "clone_mode": clone_mode,
        "response_format": response_format,
        "cfg_value": cfg_value,
        "inference_timesteps": int(inference_timesteps),
        "min_len": int(min_len),
        "max_len": int(max_len),
        "retry_badcase": retry_badcase,
        "seed": int(seed),
    }
    if reference_audio is not None:
        payload["reference_audio"] = base64.b64encode(
            _audio_to_wav_bytes(reference_audio)
        ).decode("ascii")

    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise gr.Error(f"TTS failed: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise gr.Error(f"TTS request failed: {exc}") from exc

    suffix = ".wav" if response.headers.get("content-type") == "audio/wav" else f".{response_format}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as out:
        out.write(response.content)
        return out.name


def stt_llm(
    audio: Any,
    use_diarization: bool,
    stt_base_url: str,
    stt_model: str,
    language: str,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    use_llm: bool,
    llm_base_url: str,
    llm_model: str,
    llm_preprompt: str,
    timeout_s: float,
) -> tuple[str, str, str, str, str | None]:
    transcript, denoised_audio = transcribe_audio(
        audio,
        use_diarization,
        stt_base_url,
        stt_model,
        language,
        use_denoise,
        denoise_post_filter,
        denoise_post_filter_beta,
        denoise_attenuation_limit_db,
        denoise_min_db_thresh,
        denoise_max_db_erb_thresh,
        denoise_max_db_df_thresh,
        use_vad,
        no_speech_threshold,
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
        timeout_s,
    )
    if not transcript:
        raise gr.Error("STT returned an empty transcript, so LLM was skipped.")
    llm_response = ""
    rag_context = ""
    if use_llm:
        llm_response, rag_context = generate_llm_response(
            transcript,
            llm_preprompt,
            llm_base_url,
            llm_model,
            timeout_s,
        )
    return transcript, transcript, llm_response, rag_context, denoised_audio


def round_trip(
    audio: Any,
    use_diarization: bool,
    stt_base_url: str,
    stt_model: str,
    language: str,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    use_llm: bool,
    llm_base_url: str,
    llm_model: str,
    llm_preprompt: str,
    tts_base_url: str,
    tts_model: str,
    voice: str,
    clone_mode: str,
    response_format: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    seed: int,
    timeout_s: float,
) -> tuple[str, str, str, str, str, str | None]:
    transcript, denoised_audio = transcribe_audio(
        audio,
        use_diarization,
        stt_base_url,
        stt_model,
        language,
        use_denoise,
        denoise_post_filter,
        denoise_post_filter_beta,
        denoise_attenuation_limit_db,
        denoise_min_db_thresh,
        denoise_max_db_erb_thresh,
        denoise_max_db_df_thresh,
        use_vad,
        no_speech_threshold,
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
        timeout_s,
    )
    if not transcript:
        raise gr.Error("STT returned an empty transcript, so TTS was skipped.")
    llm_response = ""
    rag_context = ""
    spoken_text = transcript
    if use_llm:
        llm_response, rag_context = generate_llm_response(
            transcript,
            llm_preprompt,
            llm_base_url,
            llm_model,
            timeout_s,
        )
        spoken_text = llm_response
    output_audio = synthesize_text(
        spoken_text,
        tts_base_url,
        tts_model,
        voice,
        clone_mode,
        response_format,
        cfg_value,
        inference_timesteps,
        min_len,
        max_len,
        retry_badcase,
        seed,
        timeout_s,
    )
    return (
        transcript,
        transcript,
        llm_response,
        rag_context,
        output_audio,
        denoised_audio,
    )


def clone_voice(
    audio: Any,
    clone_name: str,
    stt_base_url: str,
    stt_model: str,
    language: str,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    tts_base_url: str,
    tts_model: str,
    voice: str,
    clone_mode: str,
    response_format: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    seed: int,
    timeout_s: float,
) -> tuple[str, str, Any, str]:
    if clone_mode == "speaker_centroid":
        transcript = transcribe_audio_chunks(
            audio,
            stt_base_url,
            stt_model,
            language,
            use_denoise,
            denoise_post_filter,
            denoise_post_filter_beta,
            denoise_attenuation_limit_db,
            denoise_min_db_thresh,
            denoise_max_db_erb_thresh,
            denoise_max_db_df_thresh,
            use_vad,
            no_speech_threshold,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
            vad_speech_pad_ms,
            timeout_s,
        )
    else:
        transcript, _ = transcribe_audio(
            audio,
            False,
            stt_base_url,
            stt_model,
            language,
            use_denoise,
            denoise_post_filter,
            denoise_post_filter_beta,
            denoise_attenuation_limit_db,
            denoise_min_db_thresh,
            denoise_max_db_erb_thresh,
            denoise_max_db_df_thresh,
            use_vad,
            no_speech_threshold,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
            vad_speech_pad_ms,
            timeout_s,
        )
    if not transcript:
        raise gr.Error("STT returned an empty transcript, so voice cloning was skipped.")
    clone_id, archive_path = _save_clone_profile(
        clone_name,
        audio,
        clone_mode,
    )
    output_audio = synthesize_text(
        " ".join(line.strip() for line in transcript.splitlines() if line.strip()),
        tts_base_url,
        tts_model,
        clone_id,
        clone_mode,
        response_format,
        cfg_value,
        inference_timesteps,
        min_len,
        max_len,
        retry_badcase,
        seed,
        timeout_s,
    )
    voice_choices = _voice_options(_csv_env("TTS_VOICE_OPTIONS", DEFAULT_VOICES))
    return (
        transcript,
        output_audio,
        gr.Dropdown(choices=voice_choices, value=clone_id),
        archive_path,
    )


def transcribe_audio_request(
    audio: Any,
    use_diarization: bool,
    stt_base_url: str,
    stt_model: str,
    language: str,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    timeout_s: float,
    request: gr.Request,
) -> tuple[str, str | None]:
    return _run_recorded_action(
        "STT",
        request,
        transcribe_audio,
        audio,
        use_diarization,
        stt_base_url,
        stt_model,
        language,
        use_denoise,
        denoise_post_filter,
        denoise_post_filter_beta,
        denoise_attenuation_limit_db,
        denoise_min_db_thresh,
        denoise_max_db_erb_thresh,
        denoise_max_db_df_thresh,
        use_vad,
        no_speech_threshold,
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
        timeout_s,
    )


def synthesize_text_request(
    text: str,
    tts_base_url: str,
    tts_model: str,
    voice: str,
    clone_mode: str,
    response_format: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    seed: int,
    timeout_s: float,
    request: gr.Request,
) -> str:
    return _run_recorded_action(
        "TTS",
        request,
        synthesize_text,
        text,
        tts_base_url,
        tts_model,
        voice,
        clone_mode,
        response_format,
        cfg_value,
        inference_timesteps,
        min_len,
        max_len,
        retry_badcase,
        seed,
        timeout_s,
    )


def stt_llm_request(
    audio: Any,
    use_diarization: bool,
    stt_base_url: str,
    stt_model: str,
    language: str,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    use_llm: bool,
    llm_base_url: str,
    llm_model: str,
    llm_preprompt: str,
    timeout_s: float,
    request: gr.Request,
) -> tuple[str, str, str, str, str | None]:
    return _run_recorded_action(
        "STT + LLM" if use_llm else "STT",
        request,
        stt_llm,
        audio,
        use_diarization,
        stt_base_url,
        stt_model,
        language,
        use_denoise,
        denoise_post_filter,
        denoise_post_filter_beta,
        denoise_attenuation_limit_db,
        denoise_min_db_thresh,
        denoise_max_db_erb_thresh,
        denoise_max_db_df_thresh,
        use_vad,
        no_speech_threshold,
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
        use_llm,
        llm_base_url,
        llm_model,
        llm_preprompt,
        timeout_s,
    )


def round_trip_request(
    audio: Any,
    use_diarization: bool,
    stt_base_url: str,
    stt_model: str,
    language: str,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    use_llm: bool,
    llm_base_url: str,
    llm_model: str,
    llm_preprompt: str,
    tts_base_url: str,
    tts_model: str,
    voice: str,
    clone_mode: str,
    response_format: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    seed: int,
    timeout_s: float,
    request: gr.Request,
) -> tuple[str, str, str, str, str, str | None]:
    return _run_recorded_action(
        "STT + LLM + TTS" if use_llm else "STT + TTS",
        request,
        round_trip,
        audio,
        use_diarization,
        stt_base_url,
        stt_model,
        language,
        use_denoise,
        denoise_post_filter,
        denoise_post_filter_beta,
        denoise_attenuation_limit_db,
        denoise_min_db_thresh,
        denoise_max_db_erb_thresh,
        denoise_max_db_df_thresh,
        use_vad,
        no_speech_threshold,
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
        use_llm,
        llm_base_url,
        llm_model,
        llm_preprompt,
        tts_base_url,
        tts_model,
        voice,
        clone_mode,
        response_format,
        cfg_value,
        inference_timesteps,
        min_len,
        max_len,
        retry_badcase,
        seed,
        timeout_s,
    )


def generate_llm_response_request(
    text: str,
    llm_preprompt: str,
    llm_base_url: str,
    llm_model: str,
    timeout_s: float,
    request: gr.Request,
) -> tuple[str, str]:
    return _run_recorded_action(
        "LLM",
        request,
        generate_llm_response,
        text,
        llm_preprompt,
        llm_base_url,
        llm_model,
        timeout_s,
    )


def clone_voice_request(
    audio: Any,
    clone_name: str,
    stt_base_url: str,
    stt_model: str,
    language: str,
    use_denoise: bool,
    denoise_post_filter: bool,
    denoise_post_filter_beta: float,
    denoise_attenuation_limit_db: float,
    denoise_min_db_thresh: float,
    denoise_max_db_erb_thresh: float,
    denoise_max_db_df_thresh: float,
    use_vad: bool,
    no_speech_threshold: float,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    tts_base_url: str,
    tts_model: str,
    voice: str,
    clone_mode: str,
    response_format: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    seed: int,
    timeout_s: float,
    request: gr.Request,
) -> tuple[str, str, Any, str]:
    return _run_recorded_action(
        "Clone voice",
        request,
        clone_voice,
        audio,
        clone_name,
        stt_base_url,
        stt_model,
        language,
        use_denoise,
        denoise_post_filter,
        denoise_post_filter_beta,
        denoise_attenuation_limit_db,
        denoise_min_db_thresh,
        denoise_max_db_erb_thresh,
        denoise_max_db_df_thresh,
        use_vad,
        no_speech_threshold,
        vad_threshold,
        vad_min_speech_ms,
        vad_min_silence_ms,
        vad_speech_pad_ms,
        tts_base_url,
        tts_model,
        voice,
        clone_mode,
        response_format,
        cfg_value,
        inference_timesteps,
        min_len,
        max_len,
        retry_badcase,
        seed,
        timeout_s,
    )


def delete_clone_voice_request(
    voice_id: str,
    request: gr.Request,
) -> tuple[Any, None]:
    return _run_recorded_action(
        "Delete voice",
        request,
        delete_clone_voice,
        voice_id,
    )


def build_demo() -> gr.Blocks:
    stt_models = _stt_model_options()
    tts_models = _csv_env("TTS_MODEL_OPTIONS", DEFAULT_TTS_MODELS)
    llm_models = _csv_env("LLM_MODEL_OPTIONS", DEFAULT_LLM_MODELS)
    voices = _voice_options(_csv_env("TTS_VOICE_OPTIONS", DEFAULT_VOICES))
    configured_stt_model = _configured_stt_model()
    selected_stt_model = (
        configured_stt_model if configured_stt_model in stt_models else stt_models[0]
    )

    with gr.Blocks(title="GeoVision STT/LLM/TTS API") as demo:
        browser_settings = gr.BrowserState(
            {},
            storage_key=os.getenv(
                "GRADIO_BROWSER_STATE_KEY",
                "local-voice-ai-settings-v1",
            ),
            secret=os.getenv(
                "GRADIO_BROWSER_STATE_SECRET",
                "local-voice-ai-browser-state-v1",
            ),
        )
        with gr.Row(equal_height=False, elem_id="app-header"):
            gr.Markdown("# GeoVision STT/LLM/TTS API", elem_id="app-title")
        with gr.Row(equal_height=False, elem_id="app-status"):
            with gr.Column(scale=2, min_width=0):
                input_audio_filename = gr.HTML(
                    _input_audio_filename_html(None),
                    elem_id="input-audio-filename",
                )
            with gr.Column(scale=3, min_width=0):
                last_used = gr.HTML(_last_used_html(), elem_id="last-used")
        last_used_timer = gr.Timer(value=5.0, active=True)

        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Input audio",
                )
                use_diarization = gr.Checkbox(
                    value=DEFAULT_USE_DIARIZATION,
                    label="Speaker diarization (pyannote 3.1)",
                    info="Label faster-whisper output as SPEAKER_00, SPEAKER_01, …",
                )
                stt_llm_button = gr.Button(
                    "Run only: STT + (LLM)",
                    variant="primary",
                )
                round_trip_button = gr.Button("Run all: STT + (LLM) + TTS", variant="secondary")
                stt_button = gr.Button("1. STT Input audio speech-to-text", variant="secondary")
                transcript = gr.Textbox(label="STT text", lines=2)
                send_stt_to_llm_button = gr.Button(
                    "2. Send text to LLM prompt",
                    variant="secondary",
                )
                direct_tts_button = gr.Button(
                    "TTS text-to-speech",
                    variant="secondary",
                )

            with gr.Column():
                with gr.Accordion("LLM instructions", open=False):
                    llm_preprompt = gr.Textbox(
                        label="System prompt (RAGFlow adds wiki context)",
                        value=_knowledge_base_prompt(),
                        lines=10,
                    )
                llm_transcript = gr.Textbox(
                    label="LLM prompt",
                    lines=2,
                )
                llm_generate_button = gr.Button(
                    "3. Send prompt to LLM",
                    variant="secondary",
                )
                llm_response = gr.Textbox(
                    label="LLM response",
                    lines=2,
                    interactive=True,
                )
                with gr.Accordion("Retrieved RAGFlow context", open=False):
                    ragflow_context = gr.Textbox(
                        label="Wiki chunks used for this LLM request",
                        lines=12,
                        interactive=False,
                        placeholder="Run an LLM request to retrieve relevant wiki chunks.",
                    )
                llm_tts_button = gr.Button(
                    "4. TTS speak LLM response",
                    variant="secondary",
                )
                output_audio = gr.Audio(
                    label="TTS output",
                    type="filepath",
                    autoplay=True,
                )

        with gr.Accordion("Dummy inputs", open=False):
            gr.Examples(
                examples=[
                    [
                        "/app/local_voice_ai/tmpm019xjlb.wav",
                        None,
                    ],
                    [
                        "/app/local_voice_ai/amazingtalkerCut.mp3",
                        None,
                    ],
                    [
                        "/app/local_voice_ai/000002.flac",
                        None,
                    ],
                    [
                        "/app/local_voice_ai/000008.flac",
                        None,
                    ],
                    [
                        "/app/local_voice_ai/中央監控站2.mp3",
                        None,
                    ],
                    [
                        "/app/local_voice_ai/中控室.mp3",
                        None,
                    ],
                    [
                        "/app/local_voice_ai/員工檢查哨.mp3",
                        None,
                    ],
                    [
                        "/app/local_voice_ai/大廳.mp3",
                        None,
                    ],
                    [
                        "/app/local_voice_ai/車輛檢查哨.mp3",
                        None,
                    ],
                    [
                        None,
                        "您好，感謝您來電奇偶科技GeoVision，我是人工智能櫃台，請問需要幫你轉接嗎？",
                    ],
                    [
                        None,
                        "幫我轉接軟體一 Louis",
                    ],
                ],
                inputs=[audio, transcript],
                label="Example",
            )

        with gr.Accordion("Settings", open=True):
            with gr.Accordion("Endpoints and models", open=True):
                with gr.Row():
                    stt_base_url = gr.Textbox(label="STT base URL", value=DEFAULT_STT_BASE_URL)
                    tts_base_url = gr.Textbox(label="TTS base URL", value=DEFAULT_TTS_BASE_URL)
                    llm_base_url = gr.Textbox(label="LLM base URL", value=DEFAULT_LLM_BASE_URL)
                with gr.Row():
                    stt_model = gr.Dropdown(
                        choices=stt_models,
                        value=selected_stt_model,
                        allow_custom_value=True,
                        label="STT model",
                    )
                    llm_model = gr.Dropdown(
                        choices=llm_models,
                        value=llm_models[0] if llm_models else None,
                        allow_custom_value=True,
                        label="LLM model",
                    )
                    tts_model = gr.Dropdown(
                        choices=tts_models,
                        value=tts_models[0],
                        allow_custom_value=True,
                        label="TTS model",
                    )
                    voice = gr.Dropdown(
                        choices=voices,
                        value=voices[0],
                        allow_custom_value=True,
                        label="Voice",
                    )
                with gr.Row():
                    language = gr.Textbox(label="STT language", value=os.getenv("STT_LANGUAGE", "zh"))
                    use_vad = gr.Checkbox(value=True, label="Use VAD")
                    use_denoise = gr.Checkbox(
                        value=DEFAULT_USE_DENOISE,
                        label="Use Denoise",
                    )
                    use_llm = gr.Checkbox(value=True, label="Use LLM")
                    response_format = gr.Dropdown(
                        choices=["wav", "mp3", "flac", "opus", "aac"],
                        value="wav",
                        label="TTS format",
                    )
                    timeout_s = gr.Slider(10, 600, value=180, step=10, label="Timeout seconds")
            with gr.Accordion("Denoise parameters", open=False):
                with gr.Row():
                    denoised_audio = gr.Audio(
                        label="Denoised STT wav",
                        type="filepath",
                        interactive=False,
                    )
                with gr.Row():
                    denoise_post_filter = gr.Checkbox(
                        value=False,
                        label="Post-filter",
                        info="Extra residual-noise cleanup; may thin speech.",
                    )
                    denoise_post_filter_beta = gr.Slider(
                        0.0,
                        0.05,
                        value=0.02,
                        step=0.005,
                        label="Post-filter beta",
                    )
                    denoise_attenuation_limit_db = gr.Slider(
                        0.0,
                        100.0,
                        value=10.0,
                        step=5.0,
                        label="Attenuation limit (dB)",
                        info="Lower values retain more original room noise.",
                    )
                with gr.Row():
                    denoise_min_db_thresh = gr.Slider(
                        -15.0,
                        35.0,
                        value=-15.0,
                        step=1.0,
                        label="Minimum processing threshold (dB)",
                    )
                    denoise_max_db_erb_thresh = gr.Slider(
                        -15.0,
                        35.0,
                        value=35.0,
                        step=1.0,
                        label="Maximum ERB threshold (dB)",
                    )
                    denoise_max_db_df_thresh = gr.Slider(
                        -15.0,
                        35.0,
                        value=35.0,
                        step=1.0,
                        label="Maximum DF threshold (dB)",
                    )
            with gr.Accordion("VAD sensitivity", open=True):
                with gr.Row():
                    no_speech_threshold = gr.Slider(
                        0.0,
                        1.0,
                        value=0.6,
                        step=0.05,
                        label="Maximum no-speech probability",
                        info="Lower values reject more Whisper segments as silence.",
                    )
                    vad_threshold = gr.Slider(
                        0.05,
                        0.95,
                        value=0.05,
                        step=0.05,
                        label="Speech threshold",
                    )
                    vad_min_speech_ms = gr.Slider(
                        0,
                        2000,
                        value=250,
                        step=50,
                        label="Minimum speech (ms)",
                    )
                    vad_min_silence_ms = gr.Slider(
                        100,
                        5000,
                        value=2000,
                        step=100,
                        label="Minimum silence (ms)",
                    )
                    vad_speech_pad_ms = gr.Slider(
                        0,
                        2000,
                        value=400,
                        step=50,
                        label="Speech padding (ms)",
                    )
            with gr.Accordion("TTS parameters", open=True):
                with gr.Row():
                    cfg_value = gr.Slider(
                        2.0,
                        2.8,
                        value=2.0,
                        step=0.1,
                        label="CFG value",
                    )
                    inference_timesteps = gr.Slider(
                        1,
                        50,
                        value=int(os.getenv("BLUEMAGPIE_INFERENCE_TIMESTEPS", "10")),
                        step=1,
                        label="Inference timesteps",
                    )
                    seed = gr.Number(
                        value=int(os.getenv("BLUEMAGPIE_SEED", "1729")),
                        precision=0,
                        label="Seed",
                    )
                with gr.Row():
                    min_len = gr.Slider(0, 100, value=2, step=1, label="Minimum length")
                    max_len = gr.Slider(100, 4000, value=2000, step=100, label="Maximum length")
                    retry_badcase = gr.Checkbox(value=False, label="Retry abnormal output")
            with gr.Accordion("Clone Voice", open=False):
                with gr.Row():
                    with gr.Column(scale=3):
                        clone_audio = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Clone voice reference audio",
                            show_label=False,
                        )
                        clone_audio_transcript = gr.Textbox(
                            label="Clone audio STT text",
                            lines=2,
                        )
                        clone_mode = gr.Radio(
                            choices=CLONE_MODE_CHOICES,
                            value=DEFAULT_CLONE_MODE,
                            label="Voice cloning mode",
                        )
                    with gr.Column(scale=1):
                        clone_name = gr.Textbox(label="Clone voice name")
                        clone_voice_button = gr.Button(
                            "Clone reference audio voice",
                            variant="secondary",
                        )
                        clone_download = gr.DownloadButton("Download cloned voice")
                        delete_voice_button = gr.Button(
                            "Delete cloned voice",
                            variant="stop",
                        )

        audio.change(
            _input_audio_filename_html,
            inputs=audio,
            outputs=input_audio_filename,
            queue=False,
        )
        stt_button.click(
            transcribe_audio_request,
            inputs=[
                audio,
                use_diarization,
                stt_base_url,
                stt_model,
                language,
                use_denoise,
                denoise_post_filter,
                denoise_post_filter_beta,
                denoise_attenuation_limit_db,
                denoise_min_db_thresh,
                denoise_max_db_erb_thresh,
                denoise_max_db_df_thresh,
                use_vad,
                no_speech_threshold,
                vad_threshold,
                vad_min_speech_ms,
                vad_min_silence_ms,
                vad_speech_pad_ms,
                timeout_s,
            ],
            outputs=[transcript, denoised_audio],
        )
        direct_tts_button.click(
            synthesize_text_request,
            inputs=[
                transcript,
                tts_base_url,
                tts_model,
                voice,
                clone_mode,
                response_format,
                cfg_value,
                inference_timesteps,
                min_len,
                max_len,
                retry_badcase,
                seed,
                timeout_s,
            ],
            outputs=output_audio,
        )
        send_stt_to_llm_button.click(
            lambda text: text,
            inputs=[transcript],
            outputs=[llm_transcript],
            queue=False,
        )
        llm_generate_button.click(
            generate_llm_response_request,
            inputs=[
                llm_transcript,
                llm_preprompt,
                llm_base_url,
                llm_model,
                timeout_s,
            ],
            outputs=[llm_response, ragflow_context],
        )
        llm_tts_button.click(
            synthesize_text_request,
            inputs=[
                llm_response,
                tts_base_url,
                tts_model,
                voice,
                clone_mode,
                response_format,
                cfg_value,
                inference_timesteps,
                min_len,
                max_len,
                retry_badcase,
                seed,
                timeout_s,
            ],
            outputs=output_audio,
        )
        stt_llm_button.click(
            stt_llm_request,
            inputs=[
                audio,
                use_diarization,
                stt_base_url,
                stt_model,
                language,
                use_denoise,
                denoise_post_filter,
                denoise_post_filter_beta,
                denoise_attenuation_limit_db,
                denoise_min_db_thresh,
                denoise_max_db_erb_thresh,
                denoise_max_db_df_thresh,
                use_vad,
                no_speech_threshold,
                vad_threshold,
                vad_min_speech_ms,
                vad_min_silence_ms,
                vad_speech_pad_ms,
                use_llm,
                llm_base_url,
                llm_model,
                llm_preprompt,
                timeout_s,
            ],
            outputs=[
                transcript,
                llm_transcript,
                llm_response,
                ragflow_context,
                denoised_audio,
            ],
        )
        round_trip_button.click(
            round_trip_request,
            inputs=[
                audio,
                use_diarization,
                stt_base_url,
                stt_model,
                language,
                use_denoise,
                denoise_post_filter,
                denoise_post_filter_beta,
                denoise_attenuation_limit_db,
                denoise_min_db_thresh,
                denoise_max_db_erb_thresh,
                denoise_max_db_df_thresh,
                use_vad,
                no_speech_threshold,
                vad_threshold,
                vad_min_speech_ms,
                vad_min_silence_ms,
                vad_speech_pad_ms,
                use_llm,
                llm_base_url,
                llm_model,
                llm_preprompt,
                tts_base_url,
                tts_model,
                voice,
                clone_mode,
                response_format,
                cfg_value,
                inference_timesteps,
                min_len,
                max_len,
                retry_badcase,
                seed,
                timeout_s,
            ],
            outputs=[
                transcript,
                llm_transcript,
                llm_response,
                ragflow_context,
                output_audio,
                denoised_audio,
            ],
        )
        clone_voice_button.click(
            clone_voice_request,
            inputs=[
                clone_audio,
                clone_name,
                stt_base_url,
                stt_model,
                language,
                use_denoise,
                denoise_post_filter,
                denoise_post_filter_beta,
                denoise_attenuation_limit_db,
                denoise_min_db_thresh,
                denoise_max_db_erb_thresh,
                denoise_max_db_df_thresh,
                use_vad,
                no_speech_threshold,
                vad_threshold,
                vad_min_speech_ms,
                vad_min_silence_ms,
                vad_speech_pad_ms,
                tts_base_url,
                tts_model,
                voice,
                clone_mode,
                response_format,
                cfg_value,
                inference_timesteps,
                min_len,
                max_len,
                retry_badcase,
                seed,
                timeout_s,
            ],
            outputs=[clone_audio_transcript, output_audio, voice, clone_download],
        )
        delete_voice_button.click(
            delete_clone_voice_request,
            inputs=voice,
            outputs=[voice, clone_download],
        )
        last_used_timer.tick(
            _last_used_html,
            outputs=last_used,
            queue=False,
        )
        setting_components = [
            stt_base_url,
            tts_base_url,
            llm_base_url,
            stt_model,
            tts_model,
            llm_model,
            llm_preprompt,
            voice,
            clone_mode,
            language,
            use_llm,
            use_denoise,
            denoise_post_filter,
            denoise_post_filter_beta,
            denoise_attenuation_limit_db,
            denoise_min_db_thresh,
            denoise_max_db_erb_thresh,
            denoise_max_db_df_thresh,
            use_vad,
            no_speech_threshold,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
            vad_speech_pad_ms,
            response_format,
            timeout_s,
            cfg_value,
            inference_timesteps,
            seed,
            min_len,
            max_len,
            retry_badcase,
            transcript,
            llm_transcript,
        ]
        demo.load(
            _restore_browser_settings,
            inputs=[browser_settings],
            outputs=setting_components,
        )
        gr.on(
            triggers=[component.change for component in setting_components],
            fn=_save_browser_settings,
            inputs=setting_components,
            outputs=browser_settings,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the STT/TTS Gradio tester.")
    parser.add_argument("--host", default=os.getenv("GRADIO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_demo().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Glass(),
        css=_APP_CSS,
        allowed_paths=[str(VOICE_CLONE_DIR)],
    )


if __name__ == "__main__":
    main()

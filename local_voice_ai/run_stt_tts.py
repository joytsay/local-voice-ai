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
DEFAULT_STT_BASE_URL = os.getenv("STT_BASE_URL", f"http://127.0.0.1:{STT_PUBLIC_PORT}/v1")
DEFAULT_TTS_BASE_URL = os.getenv("TTS_BASE_URL", f"http://127.0.0.1:{TTS_PUBLIC_PORT}/v1")
DEFAULT_STT_MODELS = list(
    dict.fromkeys(
        [
            os.getenv("VOXBOX_HF_REPO", "Systran/faster-whisper-small"),
            "Systran/faster-whisper-small",
            "Systran/faster-whisper-large-v3",
            "nemotron-3.5-asr-streaming",
        ]
    )
)
DEFAULT_TTS_MODELS = [
    "bluemagpie-tts",
]
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
    min-height: 0;
    padding: 0;
    color: var(--body-text-color-subdued);
    font-size: 0.85rem;
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


def _endpoint_defaults_for_request(request: gr.Request) -> tuple[str, str]:
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
    return stt_url, tts_url


def _csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


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
        return sorted(discovered, key=str.casefold)
    return _csv_env("STT_MODEL_OPTIONS", DEFAULT_STT_MODELS)


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
    stt_model: str,
    tts_model: str,
    voice: str,
    clone_mode: str,
    language: str,
    response_format: str,
    timeout_s: float,
    cfg_value: float,
    inference_timesteps: int,
    seed: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    transcript: str,
) -> dict[str, Any]:
    return {
        "stt_base_url": stt_base_url,
        "tts_base_url": tts_base_url,
        "stt_model": stt_model,
        "tts_model": tts_model,
        "voice": voice,
        "clone_mode": clone_mode,
        "language": language,
        "response_format": response_format,
        "timeout_s": timeout_s,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "seed": seed,
        "min_len": min_len,
        "max_len": max_len,
        "retry_badcase": retry_badcase,
        "transcript": transcript,
    }


def _restore_browser_settings(
    saved: dict[str, Any] | None,
    request: gr.Request,
) -> tuple[Any, ...]:
    settings = saved if isinstance(saved, dict) else {}
    default_stt_url, default_tts_url = _endpoint_defaults_for_request(request)
    stt_models = _stt_model_options()
    tts_models = _csv_env("TTS_MODEL_OPTIONS", DEFAULT_TTS_MODELS)
    voices = _voice_options(_csv_env("TTS_VOICE_OPTIONS", DEFAULT_VOICES))
    configured_stt_model = os.getenv("VOXBOX_HF_REPO", "").strip()

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
    clone_mode = str(settings.get("clone_mode", DEFAULT_CLONE_MODE))
    if clone_mode not in {value for _, value in CLONE_MODE_CHOICES}:
        clone_mode = DEFAULT_CLONE_MODE
    response_formats = ["wav", "mp3", "flac", "opus", "aac"]
    response_format = str(settings.get("response_format", "wav"))
    if response_format not in response_formats:
        response_format = "wav"

    return (
        str(settings.get("stt_base_url") or default_stt_url),
        str(settings.get("tts_base_url") or default_tts_url),
        gr.Dropdown(choices=stt_models, value=stt_model),
        str(settings.get("tts_model") or tts_models[0]),
        gr.Dropdown(choices=voices, value=voice),
        clone_mode,
        str(settings.get("language") or os.getenv("STT_LANGUAGE", "zh")),
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


def _response_text(response: httpx.Response) -> str:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = response.json()
        return str(payload.get("text", "")).strip()
    return response.text.strip()


def transcribe_audio(
    audio: Any,
    stt_base_url: str,
    stt_model: str,
    language: str,
    timeout_s: float,
) -> str:
    wav_bytes = _audio_to_wav_bytes(audio)
    url = stt_base_url.rstrip("/") + "/audio/transcriptions"
    data = {
        "model": stt_model,
        "response_format": "json",
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
        return ""
    return text


def transcribe_audio_chunks(
    audio: Any,
    stt_base_url: str,
    stt_model: str,
    language: str,
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
            stt_base_url,
            stt_model,
            language,
            timeout_s,
        )
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


def round_trip(
    audio: Any,
    stt_base_url: str,
    stt_model: str,
    language: str,
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
) -> tuple[str, str]:
    transcript = transcribe_audio(audio, stt_base_url, stt_model, language, timeout_s)
    if not transcript:
        raise gr.Error("STT returned an empty transcript, so TTS was skipped.")
    output_audio = synthesize_text(
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
    )
    return transcript, output_audio


def clone_voice(
    audio: Any,
    clone_name: str,
    stt_base_url: str,
    stt_model: str,
    language: str,
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
            timeout_s,
        )
    else:
        transcript = transcribe_audio(
            audio,
            stt_base_url,
            stt_model,
            language,
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
    stt_base_url: str,
    stt_model: str,
    language: str,
    timeout_s: float,
    request: gr.Request,
) -> str:
    return _run_recorded_action(
        "STT",
        request,
        transcribe_audio,
        audio,
        stt_base_url,
        stt_model,
        language,
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


def round_trip_request(
    audio: Any,
    stt_base_url: str,
    stt_model: str,
    language: str,
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
) -> tuple[str, str]:
    return _run_recorded_action(
        "STT + TTS",
        request,
        round_trip,
        audio,
        stt_base_url,
        stt_model,
        language,
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


def clone_voice_request(
    audio: Any,
    clone_name: str,
    stt_base_url: str,
    stt_model: str,
    language: str,
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
    voices = _voice_options(_csv_env("TTS_VOICE_OPTIONS", DEFAULT_VOICES))
    configured_stt_model = os.getenv("VOXBOX_HF_REPO", "").strip()
    selected_stt_model = (
        configured_stt_model if configured_stt_model in stt_models else stt_models[0]
    )

    with gr.Blocks(title="GeoVision STT/TTS API", css=_APP_CSS) as demo:
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
            with gr.Column(scale=3, min_width=300):
                gr.Markdown("# GeoVision STT/TTS API", elem_id="app-title")
            with gr.Column(scale=2, min_width=360):
                last_used = gr.HTML(_last_used_html(), elem_id="last-used")
        last_used_timer = gr.Timer(value=5.0, active=True)

        with gr.Row():
            with gr.Column():
                input_audio_filename = gr.HTML(
                    _input_audio_filename_html(None),
                    elem_id="input-audio-filename",
                )
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Input audio",
                )
                with gr.Row():
                    stt_button = gr.Button("Transcribe", variant="secondary")
                    round_trip_button = gr.Button("Transcribe + Speak", variant="primary")
                clone_name = gr.Textbox(label="Clone voice name")
                clone_voice_button = gr.Button("Clone voice", variant="secondary")
                clone_download = gr.DownloadButton("Download cloned voice")

            with gr.Column():
                transcript = gr.Textbox(label="Transcript", lines=5)
                output_audio = gr.Audio(
                    label="TTS output",
                    type="filepath",
                    autoplay=True,
                )
                tts_button = gr.Button("Speak transcript", variant="secondary")

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
                    None,
                    "您好，感謝您來電奇偶科技GeoVision，我是人工智能櫃台，請問需要幫你轉接嗎？",
                ],
            ],
            inputs=[audio, transcript],
            label="Example",
        )

        with gr.Accordion("Endpoints and models", open=True):
            with gr.Row():
                stt_base_url = gr.Textbox(label="STT base URL", value=DEFAULT_STT_BASE_URL)
                tts_base_url = gr.Textbox(label="TTS base URL", value=DEFAULT_TTS_BASE_URL)
            with gr.Row():
                stt_model = gr.Dropdown(
                    choices=stt_models,
                    value=selected_stt_model,
                    allow_custom_value=True,
                    label="STT model",
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
                clone_mode = gr.Radio(
                    choices=CLONE_MODE_CHOICES,
                    value=DEFAULT_CLONE_MODE,
                    label="Voice cloning mode",
                )
                delete_voice_button = gr.Button(
                    "Delete cloned voice",
                    variant="stop",
                )
            with gr.Row():
                language = gr.Textbox(label="STT language", value=os.getenv("STT_LANGUAGE", "zh"))
                response_format = gr.Dropdown(
                    choices=["wav", "mp3", "flac", "opus", "aac"],
                    value="wav",
                    label="TTS format",
                )
                timeout_s = gr.Slider(10, 600, value=180, step=10, label="Timeout seconds")
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

        audio.change(
            _input_audio_filename_html,
            inputs=audio,
            outputs=input_audio_filename,
            queue=False,
        )
        stt_button.click(
            transcribe_audio_request,
            inputs=[audio, stt_base_url, stt_model, language, timeout_s],
            outputs=transcript,
        )
        tts_button.click(
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
        round_trip_button.click(
            round_trip_request,
            inputs=[
                audio,
                stt_base_url,
                stt_model,
                language,
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
            outputs=[transcript, output_audio],
        )
        clone_voice_button.click(
            clone_voice_request,
            inputs=[
                audio,
                clone_name,
                stt_base_url,
                stt_model,
                language,
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
            outputs=[transcript, output_audio, voice, clone_download],
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
            stt_model,
            tts_model,
            voice,
            clone_mode,
            language,
            response_format,
            timeout_s,
            cfg_value,
            inference_timesteps,
            seed,
            min_len,
            max_len,
            retry_badcase,
            transcript,
        ]
        demo.load(
            _restore_browser_settings,
            inputs=browser_settings,
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

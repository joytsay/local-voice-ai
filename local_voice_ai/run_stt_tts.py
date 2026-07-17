"""Gradio tester for local STT and TTS endpoints.

This script is intentionally a client only. Start the local service stack first,
then run this app on the host and use the browser microphone to round-trip:

    mic/file audio -> STT endpoint -> transcript -> TTS endpoint -> playback
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import soundfile as sf


DEFAULT_STT_BASE_URL = os.getenv("STT_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://127.0.0.1:8880/v1")
DEFAULT_STT_MODELS = [
    "Systran/faster-whisper-small",
    "nemotron-3.5-asr-streaming",
]
DEFAULT_TTS_MODELS = [
    "bluemagpie-tts",
    "kokoro",
]
DEFAULT_VOICES = [
    "hung_yi_lee",
    "female_voice",
]


def _csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


def _audio_to_wav_bytes(audio: Any) -> bytes:
    if audio is None:
        raise gr.Error("Record or upload audio first.")

    if isinstance(audio, str):
        return Path(audio).read_bytes()

    if isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, data = audio
        buf = io.BytesIO()
        sf.write(buf, data, int(sample_rate), format="WAV", subtype="PCM_16")
        return buf.getvalue()

    raise gr.Error(f"Unsupported audio input: {type(audio).__name__}")


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


def synthesize_text(
    text: str,
    tts_base_url: str,
    tts_model: str,
    voice: str,
    response_format: str,
    cfg_value: float,
    inference_timesteps: int,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    seed: int,
    timeout_s: float,
) -> str:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Transcript/text is empty.")

    url = tts_base_url.rstrip("/") + "/audio/speech"
    payload = {
        "model": tts_model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "cfg_value": cfg_value,
        "inference_timesteps": int(inference_timesteps),
        "min_len": int(min_len),
        "max_len": int(max_len),
        "retry_badcase": retry_badcase,
        "seed": int(seed),
    }

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


def build_demo() -> gr.Blocks:
    stt_models = _csv_env("STT_MODEL_OPTIONS", DEFAULT_STT_MODELS)
    tts_models = _csv_env("TTS_MODEL_OPTIONS", DEFAULT_TTS_MODELS)
    voices = _csv_env("TTS_VOICE_OPTIONS", DEFAULT_VOICES)

    with gr.Blocks(title="GeoVision STT/TTS API", theme=gr.themes.Glass()) as demo:
        gr.Markdown("# GeoVision STT/TTS API")

        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Input audio",
                )
                with gr.Row():
                    stt_button = gr.Button("Transcribe", variant="secondary")
                    round_trip_button = gr.Button("Transcribe + Speak", variant="primary")

            with gr.Column():
                transcript = gr.Textbox(label="Transcript", lines=5)
                output_audio = gr.Audio(label="TTS output", type="filepath", autoplay=True)
                tts_button = gr.Button("Speak transcript", variant="secondary")

        gr.Examples(
            examples=[
                [
                    "/app/local_voice_ai/tmpm019xjlb.wav",
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
                    value=stt_models[0],
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
                    value=2.8,
                    step=0.1,
                    label="CFG value",
                )
                inference_timesteps = gr.Slider(
                    1,
                    50,
                    value=int(os.getenv("BLUEMAGPIE_INFERENCE_TIMESTEPS", "9")),
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

        stt_button.click(
            transcribe_audio,
            inputs=[audio, stt_base_url, stt_model, language, timeout_s],
            outputs=transcript,
        )
        tts_button.click(
            synthesize_text,
            inputs=[
                transcript,
                tts_base_url,
                tts_model,
                voice,
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
            round_trip,
            inputs=[
                audio,
                stt_base_url,
                stt_model,
                language,
                tts_base_url,
                tts_model,
                voice,
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

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the STT/TTS Gradio tester.")
    parser.add_argument("--host", default=os.getenv("GRADIO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_demo().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

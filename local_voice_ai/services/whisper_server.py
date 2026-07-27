from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

_model: Any = None


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


def _model_name() -> str:
    configured_path = os.getenv("VOXBOX_MODEL_PATH", "").strip()
    if configured_path:
        path = Path(configured_path).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.extend((Path("/") / path, Path("/app") / path))

        for candidate in candidates:
            resolved = _resolve_local_model(candidate)
            if resolved is not None:
                return str(resolved)

        checked = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "VOXBOX_MODEL_PATH does not contain a complete faster-whisper "
            f"snapshot; checked: {checked}. A complete model directory must "
            "contain model.bin. The local model directory is mounted at /models."
        )
    return os.getenv("VOXBOX_HF_REPO", "Systran/faster-whisper-small")


def _load_model() -> None:
    global _model

    from faster_whisper import WhisperModel

    device = os.getenv("VOXBOX_DEVICE", "cpu")
    compute_type = os.getenv("VOXBOX_COMPUTE_TYPE", "").strip() or (
        "float16" if device.startswith("cuda") else "int8"
    )
    _model = WhisperModel(
        _model_name(),
        device=device,
        compute_type=compute_type,
        download_root=os.getenv("VOXBOX_DOWNLOAD_ROOT") or None,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Local faster-whisper", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if _model is not None else "loading"}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    name = _model_name()
    return {
        "object": "list",
        "data": [{"id": name, "object": "model", "owned_by": "local"}],
    }


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
) -> Response:
    del model
    if _model is None:
        raise HTTPException(status_code=503, detail="Whisper model is not ready")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(await file.read())
            temporary_path = temporary.name

        segments_iter, info = _model.transcribe(
            temporary_path,
            language=language or os.getenv("STT_LANGUAGE") or None,
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=True,
        )
        segments = list(segments_iter)
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
                    "segments": [
                        {
                            "id": index,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
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
        return JSONResponse({"text": text})
    finally:
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

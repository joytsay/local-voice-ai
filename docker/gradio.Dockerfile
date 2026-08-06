# syntax=docker/dockerfile:1.6

ARG PYTHON_BASE=python:3.11-slim
FROM ${PYTHON_BASE}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg libsndfile1

RUN python3 -m pip install --no-cache-dir uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        gradio httpx soundfile

COPY local_voice_ai ./local_voice_ai
COPY tsmc.csv ./tsmc.csv
CMD ["python3", "-m", "local_voice_ai.run_stt_tts", "--host", "0.0.0.0", "--port", "7860"]

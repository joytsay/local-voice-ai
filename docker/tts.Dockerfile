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
        build-essential ca-certificates curl ffmpeg git libsndfile1 python3-dev

RUN python3 -m pip install --no-cache-dir uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "setuptools<80" wheel \
        "fastapi>=0.115" \
        "huggingface-hub>=0.34,<1.0" \
        "numpy>=1.26,<2" \
        soundfile \
        "uvicorn>=0.32"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-build-isolation \
        "numpy<2" \
        "transformers==4.55.4" \
        "bluemagpie-tts @ git+https://github.com/OpenFormosa/BlueMagpie-TTS.git"

# Avoid importing a mismatched torchvision binary through Transformers.
RUN uv pip uninstall --system torchvision || true
RUN /usr/bin/python3 -c \
    'import torch, transformers; from transformers import PreTrainedModel; from transformers.generation import GenerationMixin; print("TTS runtime:", torch.__version__, transformers.__version__)'

COPY local_voice_ai ./local_voice_ai
CMD ["/usr/bin/python3", "-m", "local_voice_ai.services.bluemagpie_launcher", "--host", "0.0.0.0", "--port", "8880"]

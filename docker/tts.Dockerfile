# syntax=docker/dockerfile:1.6

ARG PYTHON_BASE=python:3.11-slim
FROM ${PYTHON_BASE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG PRESERVE_BASE_TORCH=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

    RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl ffmpeg git libsndfile1 \
        ninja-build python3-dev

RUN python3 -m pip install --no-cache-dir uv
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${PRESERVE_BASE_TORCH}" = "1" ]; then \
        python3 -c 'import torch; print(torch.__version__)' >/dev/null || \
            { echo "PRESERVE_BASE_TORCH=1 requires torch in PYTHON_BASE" >&2; exit 1; }; \
        if ! python3 -c 'import torchaudio' >/dev/null 2>&1; then \
            TORCHAUDIO_TAG="$(python3 -c 'import re, torch; match = re.match(r"(\d+\.\d+\.\d+)", torch.__version__); print("v" + match.group(1) if match else "")')"; \
            test -n "${TORCHAUDIO_TAG}" || \
                { echo "Cannot derive a TorchAudio tag from the installed Torch version" >&2; exit 1; }; \
            echo "Building ${TORCHAUDIO_TAG} against preserved Torch"; \
            PYTORCH_VERSION="$(python3 -c 'from importlib.metadata import version; print(version("torch"))')" \
            USE_CUDA=0 uv pip install --system --no-deps --no-build-isolation \
                "torchaudio @ git+https://github.com/pytorch/audio.git@${TORCHAUDIO_TAG}"; \
        fi; \
        python3 -c 'from importlib.metadata import distributions; wanted = {"torch", "torchaudio"}; installed = {dist.metadata["Name"].lower(): dist.version for dist in distributions() if dist.metadata["Name"]}; print("\n".join(f"{name}=={installed[name]}" for name in sorted(wanted & installed.keys())))' \
            > /tmp/tts-torch-constraints.txt; \
    else \
        uv pip install --system \
            --index-url "${TORCH_INDEX_URL}" \
            torch torchaudio; \
        python3 -c 'from importlib.metadata import version; print("torch==%s" % version("torch")); print("torchaudio==%s" % version("torchaudio"))' \
            > /tmp/tts-torch-constraints.txt; \
    fi
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
        --constraint /tmp/tts-torch-constraints.txt \
        --extra-index-url "${TORCH_INDEX_URL}" \
        "numpy<2" \
        "transformers==4.55.4" \
        "bluemagpie-tts[clone] @ git+https://github.com/OpenFormosa/BlueMagpie-TTS.git"

# Avoid importing a mismatched torchvision binary through Transformers.
RUN uv pip uninstall --system torchvision || true
RUN /usr/bin/python3 -c \
    'import torch, torchaudio, transformers; from speechbrain.inference.speaker import EncoderClassifier; from transformers import PreTrainedModel; from transformers.generation import GenerationMixin; print("TTS runtime:", torch.__version__, torchaudio.__version__, transformers.__version__)'

COPY local_voice_ai ./local_voice_ai
CMD ["/usr/bin/python3", "-m", "local_voice_ai.services.bluemagpie_launcher", "--host", "0.0.0.0", "--port", "8880"]

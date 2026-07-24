# syntax=docker/dockerfile:1.6
#
# Single-image build for local-voice-ai.
#
# Stages:
#   frontend  → produces a Next.js static export at /app/out
#   binaries  → references upstream images for the livekit-server and llama-server binaries
#   runtime   → Python 3.11 with all deps + the binaries + the frontend
#
# Build args:
#   --build-arg LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda  (for GPU)
#   --build-arg PYTHON_BASE=python:3.11-slim                        (or nvidia/cuda...)
#   --build-arg INSTALL_JETSON_INFERENCE=1                          (for Jetson)

ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server
ARG LIVEKIT_IMAGE=livekit/livekit-server:latest
ARG PYTHON_BASE=python:3.11-slim

# ---------------- frontend ----------------
FROM node:20-slim AS frontend
WORKDIR /app
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm run build

# ---------------- binary sources ----------------
FROM ${LLAMA_IMAGE} AS llama-bin
FROM ${LIVEKIT_IMAGE} AS livekit-bin

# ---------------- runtime ----------------
FROM ${PYTHON_BASE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_JETSON_INFERENCE=0
ARG JETSON_INFERENCE_REPO=https://github.com/dusty-nv/jetson-inference.git
ARG JETSON_INFERENCE_REF=master
ARG CTRANSLATE2_VERSION=v4.5.0
ARG CTRANSLATE2_BUILD_JOBS=2

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    HF_HOME=/models \
    XDG_CACHE_HOME=/models

# Keep downloaded package archives in BuildKit's cache when an APT layer is
# invalidated. The cache mounts are not included in the resulting image.
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# System libs needed by the inference stack and the binaries
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        libgomp1 \
        python3-pip \
        tini

# On Jetson, turn the NVIDIA PyTorch image into a jetson-inference base before
# installing local-voice-ai.  Cloning recursively is required because
# jetson-inference carries jetson-utils as a git submodule.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "${INSTALL_JETSON_INFERENCE}" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            cmake \
            gstreamer1.0-libav \
            gstreamer1.0-plugins-bad \
            gstreamer1.0-plugins-base \
            gstreamer1.0-plugins-good \
            gstreamer1.0-plugins-rtp \
            gstreamer1.0-plugins-ugly \
            gstreamer1.0-tools \
            libgstreamer-plugins-bad1.0-dev \
            libgstreamer-plugins-base1.0-dev \
            libgstreamer1.0-dev \
            libopenblas-dev \
            libpython3-dev \
            python3-dev \
            python3-numpy; \
        git clone --recursive --depth=1 --branch "${JETSON_INFERENCE_REF}" \
            "${JETSON_INFERENCE_REPO}" /opt/jetson-inference; \
        cmake -S /opt/jetson-inference -B /opt/jetson-inference/build \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_INTERACTIVE=OFF; \
        cmake --build /opt/jetson-inference/build --parallel "$(nproc)"; \
        cmake --install /opt/jetson-inference/build; \
        ldconfig; \
    fi

WORKDIR /app

# Install Python deps via uv for speed and a reproducible env
RUN python3 -m pip install --no-cache-dir uv

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PRESERVE_BASE_TORCH=0

# Copy only project metadata and a minimal package skeleton first. Application
# source is copied after dependency installation so normal code edits do not
# invalidate the large ML layers.
COPY pyproject.toml ./
RUN mkdir -p local_voice_ai && touch local_voice_ai/__init__.py

# Install the ML extras in one resolution pass. Jetson PyTorch images contain
# NVIDIA-specific torch builds which must not be replaced by packages from the
# public PyTorch indexes. In that mode, constrain uv to the versions already in
# the base image and do not expose an external torch index to the resolver.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${PRESERVE_BASE_TORCH}" = "1" ]; then \
        python3 -c 'from importlib.metadata import distributions; wanted = {"torch", "torchvision", "torchaudio"}; installed = {dist.metadata["Name"].lower(): dist.version for dist in distributions() if dist.metadata["Name"]}; print("\n".join(f"{name}=={installed[name]}" for name in sorted(wanted & installed.keys())))' \
            > /tmp/base-torch-constraints.txt; \
        grep -q '^torch==' /tmp/base-torch-constraints.txt || \
            { echo "PRESERVE_BASE_TORCH=1 requires torch in PYTHON_BASE" >&2; exit 1; }; \
        uv pip install --system \
            --constraint /tmp/base-torch-constraints.txt \
            ".[ml]"; \
    else \
        uv pip install --system --index-strategy unsafe-best-match \
            --extra-index-url "${TORCH_INDEX_URL}" \
            ".[ml]"; \
    fi

# Keep Gradio separate so adding or updating the tester does not invalidate the
# much larger ML dependency layer above.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        gradio \
        "huggingface-hub>=0.34,<1.0"

# vox-box 0.0.21 pins PyAV below 13, while LiveKit requires PyAV 14 or newer.
# Install the Whisper-specific packages into an isolated import directory so
# both stacks can coexist in one image. The supervisor exposes this directory
# only to the Whisper child process.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /usr/bin/python3 "setuptools<80" wheel \
    && uv pip install --python /usr/bin/python3 --no-build-isolation --no-deps \
        --target /opt/voxbox "vox-box==0.0.21" \
    && uv pip install --python /usr/bin/python3 --upgrade \
        --target /opt/voxbox "av>=11.0,<13" \
    && uv pip install --python /usr/bin/python3 --no-deps \
        --target /opt/voxbox "faster-whisper==1.0.3" "ctranslate2==4.5.0" \
    && uv pip install --python /usr/bin/python3 --upgrade \
        --target /opt/voxbox \
        "numpy>=1.24,<2" \
        "modelscope>=1.20,<2.0" \
        "huggingface-hub>=0.34,<1.0"

# vox-box imports every backend at startup.  Its Dia TTS backend is not used by
# this image (BlueMagpie provides TTS), and Dia is unsupported on ARM, but that
# eager import otherwise loads an incompatible public torchaudio CUDA wheel.
COPY docker/voxbox_dia_stub.py /opt/voxbox/vox_box/backends/tts/dia.py

# PyPI's AArch64 CTranslate2 wheel is CPU-only.  Build the Python extension and
# shared library against the CUDA/cuDNN toolchain supplied by the Jetson NGC
# image.  CTranslate2 4.5 is the first release supporting cuDNN 9.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${PRESERVE_BASE_TORCH}" = "1" ]; then \
        git clone --recursive --depth=1 --branch "${CTRANSLATE2_VERSION}" \
            https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2; \
        cmake -S /tmp/ctranslate2 -B /tmp/ctranslate2/build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/opt/ctranslate2 \
            -DCMAKE_CUDA_ARCHITECTURES=87 \
            -DCUDA_ARCH_LIST=87 \
            -DBUILD_CLI=OFF \
            -DBUILD_SHARED_LIBS=ON \
            -DOPENMP_RUNTIME=COMP \
            -DWITH_CUDA=ON \
            -DWITH_CUDNN=ON \
            -DWITH_FLASH_ATTN=OFF \
            -DWITH_MKL=OFF \
            -DWITH_OPENBLAS=ON; \
        grep -q '^WITH_CUDA:BOOL=ON$' /tmp/ctranslate2/build/CMakeCache.txt; \
        cmake --build /tmp/ctranslate2/build --parallel "${CTRANSLATE2_BUILD_JOBS}"; \
        cmake --install /tmp/ctranslate2/build; \
        uv pip install --python /usr/bin/python3 \
            -r /tmp/ctranslate2/python/install_requirements.txt; \
        cd /tmp/ctranslate2/python; \
        CTRANSLATE2_ROOT=/opt/ctranslate2 /usr/bin/python3 setup.py bdist_wheel \
            --dist-dir /tmp/ctranslate2-wheel; \
        cd /app; \
        rm -rf /opt/voxbox/ctranslate2 /opt/voxbox/ctranslate2-*.dist-info; \
        uv pip install --python /usr/bin/python3 --no-deps --reinstall \
            /tmp/ctranslate2-wheel/*.whl; \
        PYTHONPATH=/opt/voxbox LD_LIBRARY_PATH=/opt/ctranslate2/lib:${LD_LIBRARY_PATH} \
            /usr/bin/python3 -c 'import ctranslate2; print(f"Installed CTranslate2 {ctranslate2.__version__} from {ctranslate2.__file__}"); assert "/usr/local/lib/python3.10/dist-packages/" in ctranslate2.__file__'; \
        rm -rf /tmp/ctranslate2 /tmp/ctranslate2-wheel; \
        ldconfig; \
    fi

ENV LD_LIBRARY_PATH=/opt/ctranslate2/lib:${LD_LIBRARY_PATH}

# The 24.07 Jetson/iGPU image's torchvision extension can be incompatible with
# its NVIDIA development build of torch (for example, torchvision::nms is not
# registered).  Neither Whisper nor BlueMagpie needs torchvision, but recent
# transformers versions import it opportunistically and then fail before the
# audio models can load.  Remove only torchvision in the Jetson mode while
# preserving the NVIDIA torch and torchaudio packages.
RUN if [ "${PRESERVE_BASE_TORCH}" = "1" ]; then \
        uv pip uninstall --python /usr/bin/python3 torchvision; \
    fi

# Cache LiveKit model downloads independently from normal application changes.
# agent.py reads the phone book at import time, so both inputs are copied here.
COPY local_voice_ai/agent.py ./local_voice_ai/agent.py
COPY phonebook.csv ./phonebook.csv
RUN /usr/bin/python3 -m local_voice_ai.agent download-files || true

# Copy application code only after all dependency and model-download layers.
COPY local_voice_ai ./local_voice_ai

# Drop in the binaries from upstream images.
#
# llama-server is dynamically linked against shared libraries that ship next to
# it in the upstream image's /app dir (libllama*.so, libggml*.so, libmtmd*.so),
# plus the libggml-cpu-*.so / libggml-cuda.so backends it dlopen()s at runtime.
# Its RUNPATH is the absolute build path /app/build/bin (which doesn't exist
# here) and it has no $ORIGIN entry, so copying just the binary leaves the
# loader unable to find libllama-server-impl.so. Copy the whole library set into
# a dedicated dir and register it with ldconfig so both the link-time NEEDED
# libs and the runtime-dlopen'd backends resolve. Registering via ldconfig
# (rather than LD_LIBRARY_PATH) keeps the CUDA/driver search paths the nvidia
# base image configures for the GPU build untouched.
COPY --from=llama-bin /app/ /usr/local/lib/llama/
RUN ln -s /usr/local/lib/llama/llama-server /usr/local/bin/llama-server \
    && echo /usr/local/lib/llama > /etc/ld.so.conf.d/llama.conf \
    && ldconfig
COPY --from=livekit-bin /livekit-server /usr/local/bin/livekit-server

# Drop in the static-exported frontend
COPY --from=frontend /app/out /app/frontend/out
ENV FRONTEND_DIR=/app/frontend/out

EXPOSE 7860 8080 7880 7881
VOLUME ["/models"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/bin/python3", "-m", "local_voice_ai", "serve"]

# syntax=docker/dockerfile:1.6

ARG PYTHON_BASE=python:3.11-slim
FROM ${PYTHON_BASE}

ARG INSTALL_JETSON_INFERENCE=0
ARG CTRANSLATE2_VERSION=4.5.0
ARG CUDA_ARCH_LIST=Auto
ARG DEEPFILTER_VERSION=0.5.6
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl ffmpeg git \
        libopenblas-dev libsndfile1 ninja-build python3-dev

RUN python3 -m pip install --no-cache-dir uv

# DeepFilterNet's native runtime avoids introducing a second PyTorch/torchaudio
# dependency set into the STT image. It runs on the CPU and ships native Linux
# binaries for both the amd64 development host and Jetson's arm64 architecture.
RUN case "${TARGETARCH}" in \
        amd64) DEEPFILTER_TARGET=x86_64-unknown-linux-musl ;; \
        arm64) DEEPFILTER_TARGET=aarch64-unknown-linux-gnu ;; \
        *) echo "Unsupported DeepFilterNet architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && install -d /opt/deepfilter \
    && curl -fsSL \
        "https://github.com/Rikorose/DeepFilterNet/releases/download/v${DEEPFILTER_VERSION}/deep-filter-${DEEPFILTER_VERSION}-${DEEPFILTER_TARGET}" \
        -o /usr/local/bin/deep-filter \
    && chmod 0755 /usr/local/bin/deep-filter \
    && curl -fsSL \
        "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/v${DEEPFILTER_VERSION}/models/DeepFilterNet3_onnx.tar.gz" \
        -o /opt/deepfilter/DeepFilterNet3_onnx.tar.gz
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "faster-whisper==1.0.3" \
        "fastapi>=0.115" \
        "huggingface-hub>=0.34,<1.0" \
        "numpy<2" \
        "python-multipart>=0.0.17" \
        "uvicorn>=0.32"

# PyPI's AArch64 package may not contain the Jetson CUDA backend. Only the STT
# image replaces it with a source-built CUDA library and Python wrapper.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_JETSON_INFERENCE}" = "1" ]; then \
        uv pip uninstall --system ctranslate2 \
        && git clone --recursive --depth=1 \
            --branch "v${CTRANSLATE2_VERSION}" \
            https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 \
        && cmake -S /tmp/ctranslate2 -B /tmp/ctranslate2/build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DBUILD_CLI=OFF \
            -DBUILD_TESTS=OFF \
            -DWITH_CUDA=ON \
            -DWITH_CUDNN=ON \
            -DWITH_MKL=OFF \
            -DWITH_OPENBLAS=ON \
            -DOPENMP_RUNTIME=COMP \
            -DCUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
        && cmake --build /tmp/ctranslate2/build \
        && cmake --install /tmp/ctranslate2/build \
        && ldconfig \
        && uv pip install --system \
            -r /tmp/ctranslate2/python/install_requirements.txt \
        && cd /tmp/ctranslate2/python \
        && CTRANSLATE2_ROOT=/usr/local /usr/bin/python3 setup.py bdist_wheel \
        && uv pip install --system dist/*.whl \
        && cd / \
        && rm -rf /tmp/ctranslate2; \
    fi \
    && /usr/bin/python3 -c \
        'import ctranslate2; print("CTranslate2:", ctranslate2.__version__)'

COPY local_voice_ai ./local_voice_ai
CMD ["/usr/bin/python3", "-m", "local_voice_ai.services.whisper_server"]

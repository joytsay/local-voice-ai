# syntax=docker/dockerfile:1.6

ARG PYTHON_BASE=python:3.11-slim
ARG DIARIZATION_PYTHON_BASE=nvcr.io/nvidia/pytorch:24.07-py3-igpu

FROM ${PYTHON_BASE} AS deepfilter

ARG INSTALL_JETSON_INFERENCE=0
ARG CTRANSLATE2_VERSION=4.5.0
ARG CUDA_ARCH_LIST=Auto
ARG CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
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

FROM deepfilter AS whisper

ARG INSTALL_JETSON_INFERENCE=0
ARG CTRANSLATE2_VERSION=4.5.0
ARG CUDA_ARCH_LIST=Auto

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
        test -x "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" \
        || { echo "CUDA compiler not found at ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc; use a CUDA development PYTHON_BASE" >&2; exit 1; } \
        && CT2_CUDA_ARCH_LIST="$(printf '%s\n' "${CUDA_ARCH_LIST}" \
            | sed 's/^\([0-9]\)\([0-9]\)$/\1.\2/')" \
        && uv pip uninstall --system ctranslate2 \
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
            -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}" \
            -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT_DIR}" \
            -DCMAKE_CUDA_COMPILER="${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" \
            -DCUDA_NVCC_EXECUTABLE="${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" \
            -DCUDA_ARCH_LIST="${CT2_CUDA_ARCH_LIST}" \
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

# Combined GPU STT image: faster-whisper provides transcription and pyannote
# provides anonymous speaker turns without the previous multi-model stack.
FROM ${DIARIZATION_PYTHON_BASE} AS whisper-diarization

ARG INSTALL_JETSON_INFERENCE=0
ARG CTRANSLATE2_VERSION=4.5.0
ARG CUDA_ARCH_LIST=Auto
ARG CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ARG TORCHAUDIO_VERSION=2.4.0
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

# NVIDIA's Jetson PyTorch image does not include matching torchaudio metadata.
RUN if [ "${TARGETARCH}" = "arm64" ]; then \
        git clone --recursive --depth=1 \
            --branch "v${TORCHAUDIO_VERSION}" \
            https://github.com/pytorch/audio.git /tmp/torchaudio \
        && cd /tmp/torchaudio \
        && PYTORCH_VERSION="$(python -c \
                'from importlib.metadata import version; print(version("torch"))')" \
            BUILD_SOX=0 BUILD_RIR=0 BUILD_RNNT=0 BUILD_ALIGN=0 \
            BUILD_CUDA_CTC_DECODER=0 USE_CUDA=0 USE_FFMPEG=0 \
            python setup.py bdist_wheel \
        && python -m pip install --no-deps dist/*.whl \
        && cd /app \
        && rm -rf /tmp/torchaudio; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    python -c \
        'from importlib.metadata import version; print("torch==" + version("torch")); print("torchaudio==" + version("torchaudio"))' \
        > /tmp/nvidia-pytorch-constraints.txt \
    && printf '%s\n' \
        'pyannote.core==5.0.0' \
        'pyannote.database==5.1.0' \
        'pyannote.metrics==3.2.1' \
        'pyannote.pipeline==3.0.1' \
        'lightning==2.4.0' \
        'pytorch-lightning==2.4.0' \
        'torchmetrics==1.4.2' \
        'speechbrain==1.0.3' \
        'pytorch-metric-learning==2.8.1' \
        >> /tmp/nvidia-pytorch-constraints.txt \
    && python -m pip install \
        --constraint /tmp/nvidia-pytorch-constraints.txt \
        "faster-whisper==1.0.3" \
        "fastapi>=0.115" \
        "huggingface-hub>=0.34,<1.0" \
        "numpy<2" \
        "pyannote.audio==3.3.2" \
        "python-multipart>=0.0.17" \
        "uvicorn>=0.32"

# Build CTranslate2 with Jetson CUDA support for faster-whisper.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_JETSON_INFERENCE}" = "1" ]; then \
        test -x "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" \
        || { echo "CUDA compiler not found at ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" >&2; exit 1; } \
        && CT2_CUDA_ARCH_LIST="$(printf '%s\n' "${CUDA_ARCH_LIST}" \
            | sed 's/^\([0-9]\)\([0-9]\)$/\1.\2/')" \
        && python -m pip uninstall -y ctranslate2 \
        && git clone --recursive --depth=1 \
            --branch "v${CTRANSLATE2_VERSION}" \
            https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 \
        && cmake -S /tmp/ctranslate2 -B /tmp/ctranslate2/build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DBUILD_CLI=OFF -DBUILD_TESTS=OFF \
            -DWITH_CUDA=ON -DWITH_CUDNN=ON \
            -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
            -DOPENMP_RUNTIME=COMP \
            -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}" \
            -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT_DIR}" \
            -DCMAKE_CUDA_COMPILER="${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" \
            -DCUDA_NVCC_EXECUTABLE="${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" \
            -DCUDA_ARCH_LIST="${CT2_CUDA_ARCH_LIST}" \
        && cmake --build /tmp/ctranslate2/build \
        && cmake --install /tmp/ctranslate2/build \
        && ldconfig \
        && python -m pip install \
            -r /tmp/ctranslate2/python/install_requirements.txt \
        && cd /tmp/ctranslate2/python \
        && CTRANSLATE2_ROOT=/usr/local python setup.py bdist_wheel \
        && python -m pip install dist/*.whl \
        && cd /app \
        && rm -rf /tmp/ctranslate2; \
    fi

# The optional torchvision bundled in this ARM64 image has incompatible native
# ops; pyannote does not need it, while torchmetrics otherwise tries to import it.
RUN if [ "${TARGETARCH}" = "arm64" ]; then \
        python -m pip uninstall -y torchvision; \
    fi

COPY --from=deepfilter /usr/local/bin/deep-filter /usr/local/bin/deep-filter
COPY --from=deepfilter /opt/deepfilter /opt/deepfilter
COPY local_voice_ai ./local_voice_ai

CMD ["python", "-m", "local_voice_ai.services.whisper_server"]

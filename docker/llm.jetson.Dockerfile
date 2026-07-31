# syntax=docker/dockerfile:1.6

ARG LLAMA_JETSON_BASE=nvcr.io/nvidia/pytorch:24.07-py3-igpu
FROM ${LLAMA_JETSON_BASE} AS builder

ARG LLAMA_CPP_REF=b7205
ARG LLAMA_CPP_CUDA_ARCHITECTURES=87

RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git libcurl4-openssl-dev

RUN git clone --depth=1 \
        --branch "${LLAMA_CPP_REF}" \
        https://github.com/ggml-org/llama.cpp.git \
        /tmp/llama.cpp \
    && cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CPP_CUDA_ARCHITECTURES}" \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
    && cmake --build /tmp/llama.cpp/build \
        --config Release \
        --target llama-server \
        --parallel

FROM ${LLAMA_JETSON_BASE}

RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libcurl4

COPY --from=builder /tmp/llama.cpp/build/bin/llama-server /app/llama-server
COPY docker/llm-entrypoint.sh /app/local-llm-entrypoint.sh

ENTRYPOINT ["/bin/sh", "/app/local-llm-entrypoint.sh"]

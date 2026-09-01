# syntax=docker/dockerfile:1.7

ARG RAGFLOW_REPOSITORY=https://github.com/infiniflow/ragflow.git
ARG RAGFLOW_REF=v0.26.4

FROM ubuntu:24.04 AS source
ARG RAGFLOW_REPOSITORY
ARG RAGFLOW_REF
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && git clone \
        --branch "${RAGFLOW_REF}" \
        --depth 1 \
        --single-branch \
        "${RAGFLOW_REPOSITORY}" \
        /src/ragflow

FROM ubuntu:24.04 AS downloaded-deps
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=source /src/ragflow/ /src/ragflow/
WORKDIR /src/ragflow/ragflow_deps
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3-pip \
    && python3 -m pip install --break-system-packages --no-cache-dir uv \
    && uv run --script download_deps.py \
    && rm -rf /var/lib/apt/lists/*

FROM ubuntu:24.04 AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=10 \
    UV_CONCURRENT_DOWNLOADS=4
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /ragflow

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        default-jdk \
        fonts-freefont-ttf \
        fonts-noto-cjk \
        ghostscript \
        git \
        gnupg \
        libatk-bridge2.0-0 \
        libgbm-dev \
        libgdiplus \
        libglib2.0-0 \
        libgl1 \
        libglx-mesa0 \
        libgtk-4-1 \
        libicu-dev \
        libjemalloc-dev \
        libnss3 \
        libpython3-dev \
        nginx \
        pandoc \
        pkg-config \
        postgresql-client \
        python3-pip \
        texlive \
        texlive-lang-chinese \
        texlive-latex-extra \
        texlive-xetex \
        unixodbc-dev \
        unzip \
        wget \
        xdg-utils \
    && python3 -m pip install --break-system-packages --no-cache-dir uv \
    && uv python install 3.13 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/infiniflow/resource.git /tmp/resource \
    && mkdir -p /usr/share/infinity/resource \
    && cp -a /tmp/resource/. /usr/share/infinity/resource/ \
    && rm -rf /tmp/resource

FROM base AS builder
COPY --from=source /src/ragflow/ /ragflow/

# GitHub's release-asset CDN is unreliable on some Jetson networks. Download
# this direct-URL dependency separately with curl's stronger retry controls,
# verify the hash pinned by uv.lock, and make the locked source local.
RUN mkdir -p /opt/ragflow-wheels \
    && curl \
        --location \
        --retry 20 \
        --retry-delay 5 \
        --retry-all-errors \
        --connect-timeout 60 \
        --max-time 1800 \
        --output /opt/ragflow-wheels/en_core_web_sm-3.8.0-py3-none-any.whl \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
    && echo "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85  /opt/ragflow-wheels/en_core_web_sm-3.8.0-py3-none-any.whl" \
       | sha256sum --check --strict

# RAGFlow v0.26.4 pins xgboost 1.6.0, the version required by its ARM64 build.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eux; \
    sed -i 's|gitee.com|github.com|g' uv.lock; \
    sed -i \
        's|https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl|http://127.0.0.1:8765/en_core_web_sm-3.8.0-py3-none-any.whl|g' \
        pyproject.toml uv.lock; \
    python3 -m http.server \
        8765 \
        --bind 127.0.0.1 \
        --directory /opt/ragflow-wheels \
        >/tmp/ragflow-wheel-server.log 2>&1 & server_pid=$!; \
    trap 'kill "${server_pid}" 2>/dev/null || true' EXIT; \
    until curl --fail --silent \
        http://127.0.0.1:8765/en_core_web_sm-3.8.0-py3-none-any.whl \
        --output /dev/null; do sleep 1; done; \
    UV_INSECURE_HOST=127.0.0.1:8765 \
        uv sync --python 3.13 --frozen --refresh-package litellm; \
    .venv/bin/python3 -m ensurepip --upgrade

RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    cd web \
    && NODE_OPTIONS="--max-old-space-size=8192" npm install \
    && NODE_OPTIONS="--max-old-space-size=8192" \
       VITE_BUILD_SOURCEMAP=false \
       VITE_MINIFY=esbuild \
       npm run build

RUN git describe --tags --match=v* --first-parent --always > /ragflow/VERSION \
    && rm -rf /ragflow/web/node_modules /ragflow/.git

FROM base AS production
ENV VIRTUAL_ENV=/ragflow/.venv \
    PATH=/ragflow/.venv/bin:${PATH} \
    PYTHONPATH=/ragflow/ \
    TIKA_SERVER_JAR=file:///ragflow/tika-server-standard-3.3.0.jar

COPY --from=builder /ragflow/admin /ragflow/admin
COPY --from=builder /ragflow/api /ragflow/api
COPY --from=builder /ragflow/conf /ragflow/conf
COPY --from=builder /ragflow/deepdoc /ragflow/deepdoc
COPY --from=builder /ragflow/rag /ragflow/rag
COPY --from=builder /ragflow/agent /ragflow/agent
COPY --from=builder /ragflow/mcp /ragflow/mcp
COPY --from=builder /ragflow/common /ragflow/common
COPY --from=builder /ragflow/memory /ragflow/memory
COPY --from=builder /ragflow/bin /ragflow/bin
COPY --from=builder /ragflow/tools/scripts /ragflow/tools/scripts
COPY --from=builder /ragflow/.venv /ragflow/.venv
COPY --from=builder /ragflow/pyproject.toml /ragflow/uv.lock /ragflow/VERSION /ragflow/
COPY --from=builder /ragflow/web/dist /ragflow/web/dist
COPY --from=downloaded-deps /src/ragflow/ragflow_deps/nltk_data /root/nltk_data
COPY --from=downloaded-deps \
    /src/ragflow/ragflow_deps/tika-server-standard-3.3.0.jar \
    /src/ragflow/ragflow_deps/tika-server-standard-3.3.0.jar.md5 \
    /ragflow/
COPY --from=downloaded-deps \
    /src/ragflow/ragflow_deps/cl100k_base.tiktoken \
    /ragflow/9b5ad71b2ce5302211f9c61530b329a4922fc6a4
COPY --from=builder /ragflow/docker/service_conf.yaml.template /ragflow/conf/service_conf.yaml.template
COPY --from=builder /ragflow/docker/entrypoint.sh /ragflow/entrypoint.sh
COPY --from=builder \
    /ragflow/docker/nginx/ragflow.conf.golang \
    /ragflow/docker/nginx/ragflow.conf.python \
    /ragflow/docker/nginx/ragflow.conf.hybrid \
    /ragflow/docker/nginx/nginx.conf \
    /ragflow/docker/nginx/proxy.conf \
    /etc/nginx/

RUN chmod +x /ragflow/entrypoint.sh \
    && mv /etc/nginx/ragflow.conf.golang /etc/nginx/conf.d/ragflow.conf.golang \
    && mv /etc/nginx/ragflow.conf.python /etc/nginx/conf.d/ragflow.conf.python \
    && mv /etc/nginx/ragflow.conf.hybrid /etc/nginx/conf.d/ragflow.conf.hybrid \
    && rm -f /etc/nginx/sites-enabled/default

RUN --mount=type=bind,from=downloaded-deps,source=/src/ragflow/ragflow_deps/huggingface.co,target=/huggingface.co \
    mkdir -p /ragflow/rag/res/deepdoc \
    && tar --exclude='.*' -cf - \
        /huggingface.co/InfiniFlow/text_concat_xgb_v1.0 \
        /huggingface.co/InfiniFlow/deepdoc \
       | tar -xf - --strip-components=3 -C /ragflow/rag/res/deepdoc

ENTRYPOINT ["./entrypoint.sh"]

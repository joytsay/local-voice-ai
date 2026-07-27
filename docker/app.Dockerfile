# syntax=docker/dockerfile:1.6

ARG PYTHON_BASE=python:3.11-slim
ARG NODE_IMAGE=node:20-slim

FROM ${NODE_IMAGE} AS frontend
WORKDIR /app
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm run build

FROM ${PYTHON_BASE}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg

RUN python3 -m pip install --no-cache-dir uv
COPY pyproject.toml ./
COPY local_voice_ai ./local_voice_ai
COPY phonebook.csv ./phonebook.csv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .
RUN /usr/bin/python3 -m local_voice_ai.agent download-files || true

COPY --from=frontend /app/out /app/frontend/out
CMD ["/usr/bin/python3", "-m", "local_voice_ai", "serve"]

<div align="center">
  <img src="./frontend/.github/assets/template-light.webp" alt="App Icon" width="80" />
  <h1>Local Voice AI</h1>
  <p>This project's goal is to enable anyone to easily build a powerful, private, local voice AI agent.</p>
  <p>A real-time voice AI assistant — STT, LLM, TTS — running in <strong>one container</strong>, supervised by a single Python parent process. Powered by <a href="https://docs.livekit.io/agents?utm_source=local-voice-ai">LiveKit Agents</a>.</p>
  <p>To keep up with what I'm building or request new features <a href="https://x.com/intent/follow?screen_name=ShayneParlo">send me a DM on X</a></p>
</div>

## Overview

Everything runs as managed children of one Python supervisor (`python -m local_voice_ai serve`):

- **LiveKit server** (Go binary subprocess) for WebRTC signaling — skipped if `LIVEKIT_URL` points at LiveKit Cloud.
- **llama.cpp** (`llama-server` binary subprocess) for the LLM — skipped if `LLAMA_BASE_URL` points elsewhere.
- **Nemotron STT** or **Whisper (vox-box)** — Python uvicorn child, OpenAI-compatible.
- **Kokoro TTS** — Python uvicorn child, OpenAI-compatible.
- **LiveKit Agents worker** — the orchestrator child.
- **FastAPI** in the supervisor itself, serving `POST /api/connection-details` (token minting) and the statically-exported Next.js frontend.

Children speak HTTP only over `127.0.0.1`. The image exposes three ports: `8080` (web), `7880`, `7881` (LiveKit WebRTC, only if running locally).

## Getting started

```bash
docker compose up --build
```

Open <http://localhost:8080> and click the start button.

The first build pulls upstream binaries (llama-server, livekit-server) and downloads the Nemotron + LLM weights on first request — expect tens of GB on first boot.

### GPU (NVIDIA)
```bash
LLAMA_MODEL_PATH=/models/llama.cpp/Qwen3-4B-Instruct-2507-UD-IQ1_S.gguf \
LLAMA_MODEL=qwen3-4b \
LLAMA_MODEL_ALIAS=qwen3-4b \
STT_PROVIDER=whisper \
STT_LANGUAGE=zh \
VOXBOX_DEVICE=cuda \
VOXBOX_MODEL_PATH=/models/voxbox/cache/huggingface/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120 \
BLUEMAGPIE_DEVICE=cuda \
BLUEMAGPIE_MODEL_NAME=/models/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots/78b3cbe95ed6f3097a07b5894444998c3f879075 \
LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b7205 \
PYTHON_BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
GPU_LLAMA_N_GPU_LAYERS=3 \
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose up --build
```

### NO LLM
#### GPU (NVIDIA)
```sh
ECHO_MODE=1 STT_PROVIDER=whisper STT_LANGUAGE=zh BLUEMAGPIE_DEVICE=cuda BLUEMAGPIE_MODEL_NAME=/models/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots/78b3cbe95ed6f3097a07b5894444998c3f879075  
PYTHON_BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml docker compose up --build
```

#### AGX (JETSON)

The published `server-cuda-b7205` image is amd64-only. Build llama.cpp
natively for Jetson Orin (arm64, CUDA compute capability 8.7):

```sh
LLM_DOCKERFILE=docker/llm.jetson.Dockerfile \
LLAMA_JETSON_BASE=nvcr.io/nvidia/pytorch:24.07-py3-igpu \
LLAMA_CPP_REF=b7205 \
LLAMA_CPP_CUDA_ARCHITECTURES=87 \
LLM_MODEL_PATH=/models/llama.cpp/Qwen3-4B-Instruct-2507-UD-IQ1_S.gguf \
GPU_LLAMA_N_GPU_LAYERS=99 \
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose --profile llm up --build llm
```

```sh
VOXBOX_DEVICE=cuda \
VOXBOX_MODEL_PATH= \
VOXBOX_HF_REPO=Systran/faster-whisper-small \
VOXBOX_DOWNLOAD_ROOT=/models/whisper \
PYTHON_BASE=nvcr.io/nvidia/pytorch:24.07-py3-igpu \
INSTALL_JETSON_INFERENCE=1 \
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose up stt
```

The local Whisper endpoint runs DeepFilterNet3 noise suppression before STT by
default. It uses the native CPU runtime, while Whisper remains on CUDA. Set
`STT_DENOISE_ENABLED=0` to disable it globally, or submit `denoise=false` in an
individual `/v1/audio/transcriptions` multipart request. The optional
`STT_DENOISE_POST_FILTER=1` setting is more aggressive and may remove quiet
speech, so it is disabled by default.

```sh
BLUEMAGPIE_DEVICE=cuda \
BLUEMAGPIE_CLONE_DEVICE=cuda \
BLUEMAGPIE_MODEL_NAME=/models/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots/4c2c5bcb7e87041a8eaba9df5821ec7a3e1d0c6c \
PYTHON_BASE=nvcr.io/nvidia/pytorch:24.07-py3-igpu \
PRESERVE_BASE_TORCH=1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose up --build tts
```

```sh
BLUEMAGPIE_DEVICE=cuda \
BLUEMAGPIE_CLONE_DEVICE=cuda \
BLUEMAGPIE_MODEL_NAME=/models/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots/4c2c5bcb7e87041a8eaba9df5821ec7a3e1d0c6c \
PYTHON_BASE=nvcr.io/nvidia/pytorch:24.07-py3-igpu \
PRESERVE_BASE_TORCH=1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose up --build gradio
```

### Apple Silicon

The CPU image works as-is. `llama-server` uses Metal automatically through its bundled binary.

### HTTPS reverse proxy (Caddy)

Caddy terminates HTTPS only for these two browser-facing sites:

- `https://gv.ivr.com` proxies Gradio.
- `https://gv.ivr.live.com` proxies the LiveKit web frontend. Requests under
  `/rtc` are routed to LiveKit signaling so the HTTPS frontend can connect over
  `wss://gv.ivr.live.com`.

Configure both DNS names to resolve to the Docker host, then start the `https`
profile:

```sh
CADDY_TLS=admin@example.com \
LIVEKIT_PUBLIC_URL=wss://gv.ivr.live.com \
LIVEKIT_NODE_IP=203.0.113.10 \
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose --profile https up --build
```

For a private LAN without public DNS or ACME access, use hostnames from your
local DNS and leave `CADDY_TLS=internal` (the default). Install Caddy's local
CA certificate on every browser device that must trust the sites. You can copy
it out of the named data volume through the running service:

```sh
docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt ./caddy-root.crt
```

All other published service endpoints remain on their existing HTTP ports.
LiveKit media still travels directly through `${LIVEKIT_TCP_PORT:-7881}/tcp`
and `${LIVEKIT_UDP_PORT:-7882}/udp`, so those ports must remain reachable. Set
`LIVEKIT_NODE_IP` to the LAN or public address browsers can reach.

Caddy can also run with Gradio alone. The LiveKit frontend hostname remains
configured but returns an upstream error until the `app` and `livekit` services
are started.

## Swapping in cloud providers

Each service has a single "manage" decision driven by its base URL — point it at a remote endpoint and the local subprocess is skipped:

| Goal                              | Set                                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| Use LiveKit Cloud                 | `LIVEKIT_URL=wss://your-project.livekit.cloud` (+ `LIVEKIT_API_KEY` / `…_SECRET`)   |
| Use OpenAI for the LLM            | `LLAMA_BASE_URL=https://api.openai.com/v1`, `LLAMA_MODEL=gpt-4o-mini`, `LLAMA_API_KEY=sk-…` |
| Use a remote OpenAI-compatible STT| `STT_BASE_URL=…`, `STT_MODEL=…`, `STT_API_KEY=…`                                     |
| Use a remote OpenAI-compatible TTS| `TTS_BASE_URL=…`, `TTS_API_KEY=…`                                                    |

The supervisor logs which children it manages on startup.

## Local development (no Docker)

```bash
# Python side
uv pip install -e ".[ml,dev]"
python -m local_voice_ai serve

# Frontend side, in another shell (only needed if you're editing the UI)
cd frontend && pnpm install && pnpm run dev
```

## Architecture

```
┌──────────────────────── single container ────────────────────────┐
│  python -m local_voice_ai serve                                  │
│  │                                                                │
│  ├── child: livekit-server     (skipped if LIVEKIT_URL external) │
│  ├── child: llama-server       (skipped if LLAMA_BASE_URL ext.)  │
│  ├── child: nemotron | whisper (skipped if STT_BASE_URL ext.)    │
│  ├── child: bluemagpie | kokoro (skipped if TTS_BASE_URL ext.)   │
│  ├── child: livekit-agents worker                                │
│  └── in-process: FastAPI on :8080                                 │
│        ├── POST /api/connection-details  (token minting)         │
│        └── GET  /*                       (static frontend)       │
└───────────────────────────────────────────────────────────────────┘
```

## Project structure

```
.
├─ local_voice_ai/         # Python package: supervisor + agent + services
│  ├─ __main__.py          # python -m local_voice_ai serve
│  ├─ supervisor.py        # async process supervisor
│  ├─ config.py            # env-driven config + manage-X flags
│  ├─ api.py               # FastAPI: token route + static frontend
│  ├─ agent.py             # LiveKit Agents worker
│  └─ services/
│     ├─ nemotron/server.py
│     ├─ bluemagpie/server.py
│     └─ kokoro/server.py
├─ frontend/               # Next.js (configured for static export)
├─ Dockerfile              # multi-stage build
├─ docker-compose.yml      # one service
└─ pyproject.toml          # one Python package, one venv
```

## Environment variables

See `.env` for the full list. The most important ones:

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — local-default; override for cloud.
- `LLAMA_BASE_URL`, `LLAMA_MODEL`, `LLAMA_HF_REPO`, `LLAMA_N_GPU_LAYERS`
- `STT_PROVIDER` (`nemotron`|`whisper`), `STT_BASE_URL`, `STT_MODEL`, `VOXBOX_MODEL_PATH`, `NEMOTRON_LANGUAGE`, `STT_DENOISE_ENABLED`
- `TTS_PROVIDER` (`bluemagpie`|`kokoro`), `TTS_BASE_URL`, `TTS_VOICE`, `BLUEMAGPIE_MODEL_NAME`
- `WEB_PORT` (default `8080`)
- `MANAGE_LIVEKIT`, `MANAGE_LLAMA`, `MANAGE_STT`, `MANAGE_TTS` — explicit overrides for the auto-detected "is the URL external?" logic.

## Credits

- LiveKit: <https://livekit.io/>
- LiveKit Agents: <https://docs.livekit.io/agents/>
- NVIDIA Nemotron 3.5 ASR: <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>
- OpenFormosa BlueMagpie TTS: <https://huggingface.co/OpenFormosa/BlueMagpie-TTS>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- Kokoro TTS: <https://github.com/hexgrad/kokoro>
- VoxBox (Whisper fallback): <https://pypi.org/project/vox-box/>
- DeepFilterNet3: <https://github.com/Rikorose/DeepFilterNet>

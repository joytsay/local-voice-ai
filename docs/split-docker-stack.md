# Split Docker stack

The Docker Compose stack runs each long-lived component in its own container:

| Service | Purpose | Host port |
| --- | --- | --- |
| `app` | LiveKit voice agent and application API | `8080` |
| `livekit` | LiveKit server | `7880`, `7881`, `7882/udp` |
| `gradio` | STT/TTS test UI | `7860` |
| `stt` | VoxBox/faster-whisper OpenAI-compatible API | `8000` |
| `tts` | BlueMagpie OpenAI-compatible API | `8800` |
| `llm` | llama.cpp OpenAI-compatible API | `11434` |

Each service has its own Dockerfile under `docker/`. STT owns the
faster-whisper and CUDA CTranslate2 build, TTS owns BlueMagpie, Gradio contains
only its UI/client dependencies, and the app contains the agent dependencies.
Changing one service's dependencies therefore does not invalidate the other
service images.

## Jetson AGX

Start LiveKit, the app, Gradio, Whisper, and BlueMagpie:

```sh
ECHO_MODE=1 \
PYTHON_BASE=nvcr.io/nvidia/pytorch:24.07-py3-igpu \
INSTALL_JETSON_INFERENCE=1 \
PRESERVE_BASE_TORCH=1 \
STT_PROVIDER=whisper \
STT_LANGUAGE=zh \
VOXBOX_DEVICE=cuda \
VOXBOX_MODEL_PATH=/models/voxbox/cache/huggingface/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120 \
BLUEMAGPIE_DEVICE=cuda \
BLUEMAGPIE_MODEL_NAME=/models/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots/78b3cbe95ed6f3097a07b5894444998c3f879075 \
LIVEKIT_NODE_IP=192.168.0.228 \
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose up --build
```

Set `LIVEKIT_NODE_IP` to the Jetson's LAN address. Gradio derives its displayed
STT and TTS URLs from the hostname used to open the page, so opening
`http://<jetson-ip>:7860` defaults to:

- `http://<jetson-ip>:8000/v1`
- `http://<jetson-ip>:8800/v1`

To include the isolated llama.cpp service, add its profile:

```sh
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose --profile llm up --build
```

Configure `LLM_HF_REPO` and `LLM_MODEL` when the defaults are not suitable.

## TTS-only mode

Start only BlueMagpie and its Gradio test UI:

```sh
PYTHON_BASE=nvcr.io/nvidia/pytorch:24.07-py3-igpu \
INSTALL_JETSON_INFERENCE=1 \
PRESERVE_BASE_TORCH=1 \
BLUEMAGPIE_DEVICE=cuda \
BLUEMAGPIE_MODEL_NAME=/models/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots/78b3cbe95ed6f3097a07b5894444998c3f879075 \
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose up --build tts gradio
```

Because the services no longer have cross-process supervisor dependencies,
this command does not start Whisper, Nemotron, LiveKit, or the agent.

## Rebuild behavior

Use `--build` after Dockerfile, dependency, frontend, or source changes. For
normal restarts with an already-built image, omit it:

```sh
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.local.yml \
docker compose up
```

"""FastAPI app served from the supervisor process.

Two responsibilities:
  1. ``POST /api/connection-details`` — mints a LiveKit access token. This is
     the Python port of ``frontend/app/api/connection-details/route.ts``.
  2. ``GET /*`` — serves the statically-exported Next.js frontend, when
     ``Config.frontend_dir`` is set.
"""

from __future__ import annotations

import logging
import random
from datetime import timedelta
from typing import Any, Awaitable, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit import api as lk_api

from .config import Config

logger = logging.getLogger("api")


def _mint_token(cfg: Config, agent_name: Optional[str]) -> dict[str, Any]:
    participant_name = "user"
    participant_identity = f"voice_assistant_user_{random.randint(0, 9999)}"
    room_name = f"voice_assistant_room_{random.randint(0, 9999)}"

    token = (
        lk_api.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
        .with_identity(participant_identity)
        .with_name(participant_name)
        .with_ttl(timedelta(minutes=15))
        .with_grants(
            lk_api.VideoGrants(
                room=room_name,
                room_join=True,
                can_publish=True,
                can_publish_data=True,
                can_subscribe=True,
            )
        )
    )

    if agent_name:
        token = token.with_room_config(
            lk_api.RoomConfiguration(agents=[lk_api.RoomAgentDispatch(agent_name=agent_name)])
        )

    return {
        "serverUrl": cfg.livekit_url,
        "roomName": room_name,
        "participantName": participant_name,
        "participantToken": token.to_jwt(),
    }


ReloadModels = Callable[[Optional[str], Optional[str]], Awaitable[dict[str, str]]]


def build_app(cfg: Config, reload_models: Optional[ReloadModels] = None) -> FastAPI:
    app = FastAPI(title="local-voice-ai", version="0.1.0")

    @app.post("/api/connection-details")
    async def connection_details(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}

        agent_name: Optional[str] = None
        try:
            agent_name = body.get("room_config", {}).get("agents", [{}])[0].get("agent_name")
        except (AttributeError, IndexError, TypeError):
            agent_name = None

        try:
            data = _mint_token(cfg, agent_name)
        except Exception as exc:
            logger.exception("token minting failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/models")
    async def models() -> JSONResponse:
        tts_model = cfg.bluemagpie_model_id if cfg.tts_provider == "bluemagpie" else "kokoro"
        return JSONResponse(
            {
                "stt": {
                    "current": cfg.stt_model,
                    "options": cfg.stt_model_options,
                    "managed": cfg.manage_stt,
                },
                "tts": {
                    "current": tts_model,
                    "options": cfg.tts_model_options,
                    "managed": cfg.manage_tts,
                },
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/models/reload")
    async def reload_selected_models(request: Request) -> JSONResponse:
        if reload_models is None:
            raise HTTPException(status_code=501, detail="model reload is not available")
        try:
            body = await request.json()
        except Exception:
            body = {}

        stt_model = body.get("sttModel")
        tts_model = body.get("ttsModel")
        if stt_model is not None and not isinstance(stt_model, str):
            raise HTTPException(status_code=400, detail="sttModel must be a string")
        if tts_model is not None and not isinstance(tts_model, str):
            raise HTTPException(status_code=400, detail="ttsModel must be a string")
        if not stt_model and not tts_model:
            raise HTTPException(status_code=400, detail="select at least one model")

        try:
            data = await reload_models(stt_model, tts_model)
        except Exception as exc:
            logger.exception("model reload failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    if cfg.frontend_dir:
        # SPA-style: serve static export, falling back to index.html for unknown paths.
        static = StaticFiles(directory=cfg.frontend_dir, html=True)

        @app.get("/{path:path}")
        async def spa(path: str, request: Request) -> Any:
            try:
                return await static.get_response(path or "index.html", request.scope)
            except Exception:
                return FileResponse(f"{cfg.frontend_dir}/index.html")

    return app

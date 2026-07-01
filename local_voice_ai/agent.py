"""LiveKit Agents worker.

Moved verbatim from ``livekit_agent/src/agent.py``. The only change is that the
default base URLs are loopback (``127.0.0.1``) instead of Docker service names —
the supervisor spawns the inference children on loopback ports, so this is
correct for both single-image deployment and bare-metal local runs.
"""

import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

if livekit_agent_url := os.getenv("LIVEKIT_AGENT_URL"):
    os.environ["LIVEKIT_URL"] = livekit_agent_url


PHONE_BOOK_PATH = Path(__file__).resolve().parents[1] / "phonebook.csv"


def _load_phone_book() -> str:
    return " | ".join(
        line.strip()
        for line in PHONE_BOOK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


PHONE_BOOK = _load_phone_book()

SYSTEM_INSTRUCTIONS_PREFIX = """第一句一定要先說：
您好，感謝您來電奇偶科技，我是人工智能櫃台，請問需要幫你轉接嗎？
之後的回覆要簡短、直接、清楚，一定要用繁體中文或英文回答,不要出現簡體中文,以下範例不需要念出來。
如果是找特定人員，請詢問姓名。 請查詢分機表後回答： 範例：幫您轉接{部門} {中文姓名}, {英文姓名}, 分機{分機號碼} 轉接中請稍後。 
如果有多個姓名的結果 請重新詢問完整姓名 或者部門 再去查表
以下是公司內部的分機表 欄位依序是 部門 中文姓名 英文姓名 分機號碼：
"""

SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_PREFIX + PHONE_BOOK


def _load_end_call_tool():
    for module_name in (
        "livekit.agents",
        "livekit.agents.voice",
        "livekit.agents.llm",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        tool = getattr(module, "EndCallTool", None)
        if tool is not None:
            return tool
    return None


class Assistant(Agent):
    def __init__(self) -> None:
        end_call_tool = _load_end_call_tool()
        tools = []
        if end_call_tool is not None:
            tools.append(
                end_call_tool(
                    extra_description="""轉接完成 或者 想要掛斷電話""",
                    end_instructions="""Only end the call after the caller has either received the requested information or confirmed their message and next step. Before ending, restate what will happen next.""",
                    delete_room=False,
                )
            )

        print(f"prompt: {SYSTEM_INSTRUCTIONS}")
        super().__init__(
            instructions=SYSTEM_INSTRUCTIONS,
            tools=tools,
        )

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    llama_model = os.getenv("LLAMA_MODEL", "qwen3-4b")
    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    llama_api_key = os.getenv("LLAMA_API_KEY", "no-key-needed")
    llama_timeout = float(os.getenv("LLAMA_REQUEST_TIMEOUT", "120"))

    stt_provider = os.getenv("STT_PROVIDER", "nemotron").lower()
    if stt_provider == "whisper":
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "Systran/faster-whisper-small"
    else:
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "nemotron-3.5-asr-streaming"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model = os.getenv("STT_MODEL", default_stt_model)
    stt_api_key = os.getenv("STT_API_KEY", "no-key-needed")
    stt_language = os.getenv("STT_LANGUAGE", "zh")

    tts_base_url = os.getenv("TTS_BASE_URL", "http://127.0.0.1:8880/v1")
    tts_voice = os.getenv("TTS_VOICE", "chinese_female")
    tts_api_key = os.getenv("TTS_API_KEY", "no-key-needed")

    logger.info(
        "agent session: stt=%s/%s llm=%s/%s tts=%s",
        stt_provider, stt_model, llama_base_url, llama_model, tts_base_url,
    )

    stt_kwargs = {"base_url": stt_base_url, "model": stt_model, "api_key": stt_api_key}
    if "language" in inspect.signature(openai.STT).parameters:
        stt_kwargs["language"] = stt_language

    session = AgentSession(
        stt=openai.STT(**stt_kwargs),
        llm=openai.LLM(
            base_url=llama_base_url,
            model=llama_model,
            api_key=llama_api_key,
            timeout=httpx.Timeout(llama_timeout),
        ),
        # The model name selects the wire protocol the openai TTS plugin uses:
        # only {"tts-1", "tts-1-hd"} use the raw-audio-bytes stream that the
        # Kokoro server speaks. Any other name (e.g. "kokoro") routes the plugin
        # into the gpt-4o-mini-tts SSE reader, which parses Kokoro's binary audio
        # body as text, pushes zero frames, and raises "no audio frames were
        # pushed". Kokoro ignores the model field, so "tts-1" is purely a
        # protocol selector here.
        tts=openai.TTS(base_url=tts_base_url, model="tts-1", voice=tts_voice, api_key=tts_api_key),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await ctx.connect()
    await session.start(agent=Assistant(), room=ctx.room)

    # Send the greeting immediately after the session starts so the caller
    # hears the opening line without waiting for VAD/turn detection.
    greeting = "您好，感謝您來電奇偶科技，我是人工智能櫃台，請問需要幫你轉接嗎？"
    if hasattr(session, "say"):
        await session.say(greeting)
    elif hasattr(session, "speak"):
        await session.speak(greeting)


if __name__ == "__main__":
    cli.run_app(server)

"""Dedicated Faster-Whisper/VoxBox container entrypoint."""

from __future__ import annotations

import os
import sys


def main() -> None:
    model_path = os.getenv("VOXBOX_MODEL_PATH")
    model_args = (
        ["--model", model_path]
        if model_path
        else [
            "--huggingface-repo-id",
            os.getenv("VOXBOX_HF_REPO_ID", "Systran/faster-whisper-small"),
        ]
    )
    argv = [
        sys.executable,
        "-c",
        "from vox_box.main import main; raise SystemExit(main())",
        "start",
        *model_args,
        "--data-dir",
        os.getenv("VOXBOX_DATA_DIR", "/data"),
        "--device",
        os.getenv("VOXBOX_DEVICE", "cuda"),
        "--host",
        "0.0.0.0",
        "--port",
        os.getenv("STT_BIND_PORT", "8000"),
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()

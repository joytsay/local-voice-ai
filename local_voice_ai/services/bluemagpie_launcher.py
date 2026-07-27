from __future__ import annotations

import os
import runpy
from pathlib import Path


def _patch_scaled_dot_product_attention() -> None:
    import torch.nn.functional as functional

    original = functional.scaled_dot_product_attention
    if getattr(original, "_bluemagpie_gqa_compat", False):
        return

    def scaled_dot_product_attention(
        query,
        key,
        value,
        *args,
        enable_gqa: bool = False,
        **kwargs,
    ):
        if enable_gqa and query.shape[-3] != key.shape[-3]:
            query_heads = query.shape[-3]
            key_heads = key.shape[-3]
            if key_heads <= 0 or query_heads % key_heads:
                raise ValueError(
                    "GQA requires query heads to be divisible by key/value heads"
                )
            groups = query_heads // key_heads
            key = key.repeat_interleave(groups, dim=-3)
            value = value.repeat_interleave(groups, dim=-3)
        return original(query, key, value, *args, **kwargs)

    scaled_dot_product_attention._bluemagpie_gqa_compat = True
    functional.scaled_dot_product_attention = scaled_dot_product_attention


def _normalize_model_path() -> None:
    configured = os.getenv("BLUEMAGPIE_MODEL_NAME", "").strip()
    if not configured:
        return

    path = Path(configured).expanduser()
    if path.is_absolute():
        return

    if path.parts and path.parts[0] == "models":
        os.environ["BLUEMAGPIE_MODEL_NAME"] = str(Path("/") / path)
        return

    for candidate in (Path("/") / path, Path("/app") / path):
        if candidate.exists():
            os.environ["BLUEMAGPIE_MODEL_NAME"] = str(candidate.resolve())
            return


def main() -> None:
    _normalize_model_path()
    _patch_scaled_dot_product_attention()
    runpy.run_module(
        "local_voice_ai.services.bluemagpie.server",
        run_name="__main__",
        alter_sys=True,
    )


if __name__ == "__main__":
    main()

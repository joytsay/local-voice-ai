"""Disabled Dia backend for the ARM64 Whisper-only VoxBox installation."""


class Dia:
    """Keep VoxBox's backend registry importable without loading torchaudio."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "The VoxBox Dia TTS backend is disabled in this image; use BlueMagpie TTS"
        )

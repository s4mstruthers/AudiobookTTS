"""Engine registry. Engines are imported lazily: heavy ML deps load only when used."""

from __future__ import annotations

from audiobooktts.engines.base import TTSEngine, Voice

_ENGINES = {}


def get_engine(name: str = "kokoro") -> TTSEngine:
    if name in _ENGINES:
        return _ENGINES[name]
    if name == "kokoro":
        from audiobooktts.engines.kokoro import KokoroEngine

        engine = KokoroEngine()
    else:
        raise ValueError(f"Unknown TTS engine: {name!r} (available: kokoro)")
    _ENGINES[name] = engine
    return engine


__all__ = ["TTSEngine", "Voice", "get_engine"]

"""Engine registry. Engines are imported lazily: heavy ML deps load only when used.

Every model call is funnelled onto one dedicated thread. MLX GPU streams are
thread-local, so a model loaded on one thread and used from another aborts the
process with "There is no Stream(gpu, N) in current thread" — and the web
server does exactly that, serving requests from a rotating threadpool while
rendering jobs on their own worker thread. Serialising the calls also matches
how the models are used in practice: one book at a time, sequentially.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from audiobooktts.engines.base import TTSEngine, Voice

# Larger cloning models, for voices Chatterbox struggles with.
BIG_MODELS = {
    "higgs": "mlx-community/higgs-audio-v2-3B-mlx-q8",
}
ENGINE_NAMES = ("chatterbox", *BIG_MODELS)

_ENGINES: dict[str, TTSEngine] = {}
_LOCK = threading.Lock()
# A single worker owns all MLX state for the life of the process.
_MLX = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def _run(fn, *args, **kwargs):
    """Execute fn on the MLX thread, propagating the result or exception."""
    if threading.current_thread().name.startswith("mlx"):
        return fn(*args, **kwargs)  # already there; don't deadlock
    return _MLX.submit(fn, *args, **kwargs).result()


class _SerializedEngine(TTSEngine):
    """Thread-confining proxy around a real engine."""

    def __init__(self, inner: TTSEngine):
        self._inner = inner
        self.name = inner.name
        # Copy explicitly: class attributes defined on TTSEngine resolve on the
        # proxy itself, so __getattr__ never sees them and the inner engine's
        # value would be silently ignored.
        self.deterministic = inner.deterministic

    @property
    def sample_rate(self) -> int:
        # Set correctly only once the model has loaded, so read it live.
        return self._inner.sample_rate

    def list_voices(self) -> list[Voice]:
        return _run(self._inner.list_voices)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        return _run(self._inner.synthesize, text, voice, speed)

    def __getattr__(self, item):
        # Anything engine-specific (e.g. prepare_conditionals) falls through.
        return getattr(self._inner, item)


def _build(name: str) -> TTSEngine:
    if name == "chatterbox":
        from audiobooktts.engines.chatterbox import ChatterboxEngine

        return ChatterboxEngine()
    if name in BIG_MODELS:
        from audiobooktts.engines.generic_clone import GenericCloneEngine

        return GenericCloneEngine(name, BIG_MODELS[name])
    raise ValueError(
        f"Unknown TTS engine: {name!r} (available: {', '.join(ENGINE_NAMES)})"
    )


def get_engine(name: str = "chatterbox") -> TTSEngine:
    with _LOCK:
        if name not in _ENGINES:
            _ENGINES[name] = _SerializedEngine(_build(name))
        return _ENGINES[name]


__all__ = ["ENGINE_NAMES", "TTSEngine", "Voice", "get_engine"]

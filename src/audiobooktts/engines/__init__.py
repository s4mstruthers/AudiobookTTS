"""Engine registry. Engines are imported lazily: heavy ML deps load only when used.

Every model call is funnelled onto one dedicated worker thread. MLX GPU streams
are thread-local, so a model loaded on one thread and used from another aborts
the process with "There is no Stream(gpu, N) in current thread" — and the web
server does exactly that, serving requests from a rotating threadpool while
rendering jobs on their own worker thread. Serialising the calls also keeps a
single model from being driven concurrently on any backend, which matches how
it is used in practice: one book at a time, sequentially.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from audiobooktts.engines.base import EngineUnavailableError, TTSEngine, Voice

# Larger cloning models (MLX only), for voices Chatterbox struggles with.
BIG_MODELS = {
    "higgs": "mlx-community/higgs-audio-v2-3B-mlx-q8",
}
ENGINE_NAMES = ("chatterbox", *BIG_MODELS)

_WORKER_PREFIX = "tts-worker"
_ENGINES: dict[str, TTSEngine] = {}
_LOCK = threading.Lock()
# A single worker owns all model state for the life of the process.
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix=_WORKER_PREFIX)


def _run(fn, *args, **kwargs):
    """Execute fn on the worker thread, propagating the result or exception."""
    if threading.current_thread().name.startswith(_WORKER_PREFIX):
        return fn(*args, **kwargs)  # already there; don't deadlock
    return _WORKER.submit(fn, *args, **kwargs).result()


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

    @property
    def inner(self) -> TTSEngine:
        return self._inner

    def load(self) -> None:
        _run(self._inner.load)

    def list_voices(self) -> list[Voice]:
        return self._inner.list_voices()  # reads the voice folder; no model needed

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        return _run(self._inner.synthesize, text, voice, speed)

    def conditioning_windows(self) -> dict[str, float] | None:
        return self._inner.conditioning_windows()

    def __getattr__(self, item):
        # Anything engine-specific falls through to the real engine.
        return getattr(self._inner, item)


def _build(name: str) -> TTSEngine:
    if name == "chatterbox":
        from audiobooktts.engines.chatterbox import ChatterboxEngine

        return ChatterboxEngine()
    if name in BIG_MODELS:
        from audiobooktts.engines.generic_clone import GenericCloneEngine

        return GenericCloneEngine(name, BIG_MODELS[name])
    raise ValueError(f"Unknown TTS engine: {name!r} (available: {', '.join(ENGINE_NAMES)})")


def get_engine(name: str = "chatterbox") -> TTSEngine:
    """The shared engine instance for ``name``. Constructing it loads no model."""
    with _LOCK:
        if name not in _ENGINES:
            _ENGINES[name] = _SerializedEngine(_build(name))
        return _ENGINES[name]


def reset_engines() -> None:
    """Forget constructed engines, e.g. after the configuration changes."""
    with _LOCK:
        _ENGINES.clear()


__all__ = [
    "ENGINE_NAMES",
    "EngineUnavailableError",
    "TTSEngine",
    "Voice",
    "get_engine",
    "reset_engines",
]

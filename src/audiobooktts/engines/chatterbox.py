"""Chatterbox (Resemble AI, MIT): voice cloning from a short reference clip.

There is no fixed voice roster: a "voice" is a reference recording in the voice
library, and 10-15 seconds of clean, consistent speech is enough. The model
preserves accent and timbre.

Two interchangeable backends run the same model:

* **MLX** (``mlx-audio``) on Apple Silicon Macs — the fastest option there.
* **PyTorch** (``chatterbox-tts``) everywhere else: NVIDIA GPUs via CUDA on
  Windows and Linux, Apple GPUs via MPS, or the CPU.

``chatterbox_backend = "auto"`` picks MLX on Apple Silicon when it is
installed, otherwise PyTorch.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import platform
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from audiobooktts.engines.base import CloningEngine, EngineUnavailableError

logger = logging.getLogger(__name__)

BACKENDS = ("mlx", "torch")
DEVICES = ("auto", "cuda", "mps", "cpu")
# Conditioning prepared per clip is small; keep enough for a multi-narrator book.
_MAX_CACHED_VOICES = 8
# Formats the PyTorch loader (librosa/soundfile) reads directly.
_TORCH_NATIVE_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def installed_backends() -> dict[str, bool]:
    return {"mlx": _installed("mlx_audio"), "torch": _installed("chatterbox")}


def install_hint() -> str:
    if is_apple_silicon():
        return 'Install a backend with: pip install "audiobooktts[mlx]"'
    return (
        'Install a backend with: pip install "audiobooktts[torch]"  '
        "(for an NVIDIA GPU, install the CUDA build of PyTorch first; see the README)"
    )


def resolve_backend(preference: str = "auto") -> str:
    """Choose the backend to run, raising EngineUnavailableError if none can."""
    preference = (preference or "auto").lower()
    have = installed_backends()
    if preference in BACKENDS:
        if not have[preference]:
            package = "mlx-audio" if preference == "mlx" else "chatterbox-tts"
            raise EngineUnavailableError(
                f"The {preference} backend was requested but {package} is not installed. "
                + install_hint()
            )
        return preference
    if preference != "auto":
        raise EngineUnavailableError(
            f"Unknown chatterbox_backend {preference!r}; use auto, mlx or torch"
        )
    if have["mlx"] and is_apple_silicon():
        return "mlx"
    if have["torch"]:
        return "torch"
    if have["mlx"]:
        return "mlx"  # e.g. MLX's Linux build
    raise EngineUnavailableError("No Chatterbox backend is installed. " + install_hint())


def resolve_device(preference: str = "auto") -> str:
    """The PyTorch device to use: CUDA, then Apple MPS, then CPU."""
    import torch

    preference = (preference or "auto").lower()
    if preference not in DEVICES:
        raise EngineUnavailableError(
            f"Unknown device {preference!r}; use one of {', '.join(DEVICES)}"
        )
    if preference == "cuda" and not torch.cuda.is_available():
        raise EngineUnavailableError(
            "device is set to cuda but PyTorch cannot see a CUDA GPU. "
            "Install the CUDA build of PyTorch, or set device to auto."
        )
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class _ConditioningCache:
    """Voice conditioning keyed by clip contents, least recently used out."""

    def __init__(self, size: int = _MAX_CACHED_VOICES):
        self._size = size
        self._items: OrderedDict = OrderedDict()

    def get(self, ref: Path, build):
        from audiobooktts.voices import clip_key

        key = clip_key(ref)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        value = build(ref)
        self._items[key] = value
        while len(self._items) > self._size:
            self._items.popitem(last=False)
        return value


class _MLXBackend:
    """mlx-audio's Chatterbox, across both its base and turbo variants."""

    name = "mlx"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = "Apple GPU (MLX)"
        self.sample_rate = 24000
        self._model = None
        self._builtin = None
        self._cache = _ConditioningCache()
        self._generate_takes_conds = False
        self._prepare_takes_rate = False

    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_audio.tts.utils import load_model

        logger.info("Loading %s with MLX", self.model_id)
        model = load_model(self.model_id)
        rate = getattr(model, "sample_rate", None)
        if isinstance(rate, int):
            self.sample_rate = rate
        # Keep the built-in voice's conditioning so we can switch back to it
        # without reloading the whole model.
        self._builtin = getattr(model, "_conds", None)
        self._generate_takes_conds = "conds" in inspect.signature(model.generate).parameters
        self._prepare_takes_rate = (
            "ref_sr" in inspect.signature(model.prepare_conditionals).parameters
        )
        self._model = model

    def _prepare(self, ref: Path):
        """Derive the voice conditioning.

        The turbo model takes a path and stores the result on itself. The base
        model wants samples plus a rate and *returns* the conditioning — ignore
        the return value and generation silently falls back to the model's own
        built-in voice, which is a different person entirely.
        """
        model = self._model
        if self._prepare_takes_rate:
            import mlx.core as mx

            from audiobooktts.voices import load_clip

            samples, rate = load_clip(ref)
            return model.prepare_conditionals(mx.array(samples), rate)
        model.prepare_conditionals(str(ref))
        return getattr(model, "_conds", None)

    def generate(self, text: str, ref: Path | None) -> np.ndarray:
        self.load()
        model = self._model
        kwargs: dict = {"text": text, "verbose": False}
        if ref is None:
            model._conds = self._builtin
        else:
            # Passing ref_audio on every call re-derives the conditioning, which
            # measured ~10x slower than preparing once per voice and reusing it.
            conds = self._cache.get(ref, self._prepare)
            if conds is not None:
                model._conds = conds
                # The base model takes the conditioning as an argument; passing
                # it explicitly guarantees the reference voice is the one used.
                if self._generate_takes_conds:
                    kwargs["conds"] = conds
        segments = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in model.generate(**kwargs)
        ]
        return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)


class _TorchBackend:
    """The official PyTorch Chatterbox package (chatterbox-tts)."""

    name = "torch"

    def __init__(self, turbo: bool, device: str):
        self.turbo = turbo
        self.requested_device = device
        self.device = device
        self.sample_rate = 24000
        self._model = None
        self._builtin = None
        self._cache = _ConditioningCache()

    def load(self) -> None:
        if self._model is not None:
            return
        self.device = resolve_device(self.requested_device)
        if self.turbo:
            from chatterbox.tts_turbo import ChatterboxTurboTTS as model_cls
        else:
            from chatterbox.tts import ChatterboxTTS as model_cls
        logger.info("Loading %s on %s with PyTorch", model_cls.__name__, self.device)
        model = model_cls.from_pretrained(device=self.device)
        self.sample_rate = int(getattr(model, "sr", self.sample_rate))
        self._builtin = getattr(model, "conds", None)
        self._model = model

    def _prepare(self, ref: Path):
        model = self._model
        path = ref
        tmp = None
        if ref.suffix.lower() not in _TORCH_NATIVE_SUFFIXES:
            # e.g. m4a: decode with ffmpeg into a wav the loader can read.
            import soundfile as sf

            from audiobooktts.voices import load_clip

            samples, rate = load_clip(ref)
            fd, tmp_name = tempfile.mkstemp(suffix=".wav")
            os.close(fd)  # Windows cannot reopen a file that is still held open
            tmp = Path(tmp_name)
            sf.write(tmp, samples, rate)
            path = tmp
        try:
            model.prepare_conditionals(str(path))
        except AssertionError as e:  # the model asserts on clips of 5s or less
            raise ValueError(f"Reference clip {ref.name} is unusable: {e}") from None
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        return model.conds

    def generate(self, text: str, ref: Path | None) -> np.ndarray:
        import torch

        self.load()
        model = self._model
        if ref is None:
            if self._builtin is None:
                raise EngineUnavailableError(
                    "This model has no built-in voice; choose a cloned voice instead"
                )
            model.conds = self._builtin
        else:
            model.conds = self._cache.get(ref, self._prepare)
        with torch.inference_mode():
            wav = model.generate(text)
        return wav.detach().cpu().numpy().astype(np.float32).reshape(-1)


class ChatterboxEngine(CloningEngine):
    name = "chatterbox"
    deterministic = False  # autoregressive: pace wanders between runs

    def __init__(
        self,
        model_id: str | None = None,
        backend: str | None = None,
        device: str | None = None,
    ):
        from audiobooktts.config import Config

        cfg = Config.load()
        self.model_id = model_id or cfg.chatterbox_model
        self.backend_preference = backend or cfg.chatterbox_backend
        self.device_preference = device or cfg.device
        # The backend is chosen on first use, so listing voices and managing
        # the library work even before any ML package is installed.
        self._backend: _MLXBackend | _TorchBackend | None = None

    @property
    def turbo(self) -> bool:
        return "turbo" in self.model_id.lower()

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self._backend.sample_rate if self._backend else 24000

    @property
    def backend(self):
        if self._backend is None:
            chosen = resolve_backend(self.backend_preference)
            if chosen == "mlx":
                self._backend = _MLXBackend(self.model_id)
            else:
                self._backend = _TorchBackend(self.turbo, self.device_preference)
        return self._backend

    def load(self) -> None:
        self.backend.load()

    def describe(self) -> str:
        """Human-readable summary of the backend and device, loading nothing."""
        backend = self.backend
        variant = "Chatterbox-Turbo" if self.turbo else "Chatterbox"
        return f"{variant} on {backend.name} ({backend.device})"

    def conditioning_windows(self) -> dict[str, float]:
        # Seconds of the reference each model reads, from its ENC_COND_LEN and
        # DEC_COND_LEN: anything beyond the longer window is discarded, and the
        # speaker window fixes identity, so a clip's opening matters most.
        speaker = 15.0 if self.turbo else 6.0
        decoder = 10.0
        return {"speaker_s": speaker, "decoder_s": decoder, "useful_s": max(speaker, decoder)}

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """Speech for ``text``. ``speed`` is ignored: pacing is a separate time-stretch."""
        from audiobooktts.voices import reference_path

        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.float32)
        return self.backend.generate(text, reference_path(voice))

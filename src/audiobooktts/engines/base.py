"""TTS engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class EngineUnavailableError(RuntimeError):
    """The engine's ML backend is not installed or cannot run on this machine."""


@dataclass
class Voice:
    id: str
    label: str
    language: str = "en"
    kind: str = ""  # "builtin" or "clone"


class TTSEngine(ABC):
    name: str
    sample_rate: int = 24000
    #: Whether the same text and voice always produce the same audio.
    #: Autoregressive engines sample, so their pace varies run to run and a
    #: per-voice speaking rate cannot be measured reliably.
    deterministic: bool = True

    def load(self) -> None:  # noqa: B027 - optional hook, deliberately empty
        """Load model weights now. Optional: synthesize() loads lazily.

        Call it before relying on ``sample_rate``, which some models only
        report once loaded.
        """

    @abstractmethod
    def list_voices(self) -> list[Voice]: ...

    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """Return mono float32 audio at ``self.sample_rate``."""

    def conditioning_windows(self) -> dict[str, float] | None:
        """How many seconds of a reference clip the model reads, if it clones."""
        return None


class CloningEngine(TTSEngine):
    """An engine whose voices are reference clips from the voice library."""

    def list_voices(self) -> list[Voice]:
        from audiobooktts.voices import DEFAULT_VOICE, clip_paths

        voices = [Voice(DEFAULT_VOICE, "Default — the model's own voice", "en", "builtin")]
        for name, path in sorted(clip_paths().items(), key=lambda kv: kv[0].lower()):
            voices.append(Voice(name, f"{name} — cloned from {path.name}", "en", "clone"))
        return voices

"""TTS engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Voice:
    id: str
    label: str
    language: str = "en"


class TTSEngine(ABC):
    name: str
    sample_rate: int

    @abstractmethod
    def list_voices(self) -> list[Voice]: ...

    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """Return mono float32 audio at self.sample_rate."""

"""Larger cloning models, behind the same interface as Chatterbox.

Chatterbox is 0.5B and reproduces timbre well but imposes its own articulation,
so a non-native accent comes out wrong. Bigger models saw a wider range of
speakers and may hold an accent better. They all take a reference clip; the
only real differences are what the argument is called and whether they want a
path or samples, which `_call_kwargs` works out by inspection.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from audiobooktts.engines.base import TTSEngine, Voice
from audiobooktts.engines.chatterbox import (
    DEFAULT_VOICE,
    VOICES_DIR,
    _AUDIO_SUFFIXES,
)


class GenericCloneEngine(TTSEngine):
    """Wraps any mlx-audio TTS model that clones from a reference clip."""

    deterministic = False
    sample_rate = 24000

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(self.model_id)
            rate = getattr(self._model, "sample_rate", None)
            if isinstance(rate, int):
                self.sample_rate = rate
        return self._model

    @staticmethod
    def reference_path(voice: str) -> Path | None:
        if not voice or voice == DEFAULT_VOICE:
            return None
        candidate = Path(voice)
        if candidate.is_file():
            return candidate
        for suffix in _AUDIO_SUFFIXES:
            p = VOICES_DIR / f"{voice}{suffix}"
            if p.is_file():
                return p
        return None

    def list_voices(self) -> list[Voice]:
        voices = [Voice(DEFAULT_VOICE, "Default — the model's own voice", "en", "")]
        if VOICES_DIR.is_dir():
            for p in sorted(VOICES_DIR.iterdir()):
                if p.suffix.lower() in _AUDIO_SUFFIXES:
                    voices.append(
                        Voice(p.stem, f"{p.stem} — cloned from {p.name}", "en", "clone")
                    )
        return voices

    def _call_kwargs(self, model, text: str, ref: Path | None) -> dict:
        params = inspect.signature(model.generate).parameters
        kwargs: dict = {"text": text}
        if "verbose" in params:
            kwargs["verbose"] = False
        if ref is None:
            return kwargs
        # Some models take a path, others want samples. Higgs, for one, tries
        # float() on whatever it is given and fails loudly on a path.
        import mlx.core as mx
        import soundfile as sf

        wav, rate = sf.read(str(ref), dtype="float32")
        samples = mx.array(np.asarray(wav).reshape(-1))
        for key in ("ref_audio", "audio_prompt", "speaker_audio"):
            if key in params:
                kwargs[key] = samples
                break
        else:
            if "voice" in params:
                kwargs["voice"] = str(ref)
        for rate_key in ("sample_rate", "ref_sr", "audio_prompt_sr"):
            if rate_key in params:
                kwargs[rate_key] = rate
                break
        return kwargs

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        model = self._ensure_model()
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.float32)
        ref = self.reference_path(voice)
        segments = []
        for result in model.generate(**self._call_kwargs(model, text, ref)):
            segments.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(segments)

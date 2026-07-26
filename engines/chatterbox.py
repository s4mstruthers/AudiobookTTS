"""Chatterbox (Resemble AI, MIT) via mlx-audio — voice cloning from a reference clip.

Unlike Kokoro there is no fixed voice roster: a "voice" here is a reference
recording. Drop clips into ~/.audiobooktts/voices/ and each becomes selectable
by filename. 5–10 seconds of clean, consistent speech is enough, and the model
preserves accent — which is the way past Kokoro's B- ceiling for British.

Much heavier than Kokoro (~0.5B vs 82M parameters), so renders are slower;
`abtts voices --engine chatterbox` and the benchmark in the README give the
current numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audiobooktts.config import APP_DIR
from audiobooktts.engines.base import TTSEngine, Voice

MODEL_ID = "mlx-community/chatterbox-turbo-8bit"
VOICES_DIR = APP_DIR / "voices"
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# The model's own voice, used when no reference clip is selected.
DEFAULT_VOICE = "default"


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    sample_rate = 24000  # corrected from the model once loaded
    deterministic = False  # autoregressive: pace wanders between runs

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._prepared_voice: str | None = None
        self._builtin_conds = None

    def _ensure_model(self):
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(self.model_id)
            rate = getattr(self._model, "sample_rate", None)
            if isinstance(rate, int):
                self.sample_rate = rate
            # Keep the built-in voice's conditioning so we can switch back to it
            # without reloading the whole model.
            self._builtin_conds = getattr(self._model, "_conds", None)
        return self._model

    # --- voices -------------------------------------------------------------

    @staticmethod
    def reference_path(voice: str) -> Path | None:
        """Resolve a voice name to a reference clip, if one exists."""
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
        voices = [
            Voice(DEFAULT_VOICE, "Default — the model's own voice", "en", "")
        ]
        if VOICES_DIR.is_dir():
            for p in sorted(VOICES_DIR.iterdir()):
                if p.suffix.lower() in _AUDIO_SUFFIXES:
                    voices.append(
                        Voice(p.stem, f"{p.stem} — cloned from {p.name}", "en", "clone")
                    )
        return voices

    # --- synthesis ----------------------------------------------------------

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        model = self._ensure_model()
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.float32)

        # Passing ref_audio re-derives the voice conditioning on *every* call,
        # which measured ~10x slower than reusing it. Prepare once per voice and
        # then generate against the cached conditionals.
        ref = self.reference_path(voice)
        if ref is not None and self._prepared_voice != voice:
            model.prepare_conditionals(str(ref))
            self._prepared_voice = voice

        if ref is None and self._prepared_voice is not None:
            model._conds = self._builtin_conds  # back to the built-in voice
            self._prepared_voice = None

        kwargs: dict = {"text": text, "verbose": False}
        if speed and speed != 1.0:
            kwargs["speed"] = speed

        segments = []
        for result in model.generate(**kwargs):
            segments.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(segments)

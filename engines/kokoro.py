"""Kokoro-82M via mlx-audio (Apple Silicon)."""

from __future__ import annotations

import numpy as np

from audiobooktts.engines.base import TTSEngine, Voice

MODEL_ID = "mlx-community/Kokoro-82M-bf16"

# Curated English voices (Kokoro v1.0 ships 54 across 8 languages;
# unknown ids are passed straight through to the model).
_VOICES = [
    Voice("af_heart", "Heart — US female (best overall)"),
    Voice("af_bella", "Bella — US female, warm"),
    Voice("af_nicole", "Nicole — US female, soft"),
    Voice("af_sarah", "Sarah — US female, clear"),
    Voice("af_sky", "Sky — US female, bright"),
    Voice("am_adam", "Adam — US male, deep"),
    Voice("am_michael", "Michael — US male, warm"),
    Voice("am_fenrir", "Fenrir — US male, energetic"),
    Voice("bf_emma", "Emma — UK female"),
    Voice("bf_isabella", "Isabella — UK female"),
    Voice("bm_george", "George — UK male"),
    Voice("bm_lewis", "Lewis — UK male"),
]


class KokoroEngine(TTSEngine):
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(self.model_id)
        return self._model

    def list_voices(self) -> list[Voice]:
        return list(_VOICES)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        model = self._ensure_model()
        lang_code = voice[0] if voice[:1] in ("a", "b") else "a"
        segments = []
        for result in model.generate(
            text=text, voice=voice, speed=speed, lang_code=lang_code
        ):
            segments.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(segments)

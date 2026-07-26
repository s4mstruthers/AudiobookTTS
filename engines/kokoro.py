"""Kokoro-82M via mlx-audio (Apple Silicon)."""

from __future__ import annotations

import numpy as np

from audiobooktts.engines.base import TTSEngine, Voice

MODEL_ID = "mlx-community/Kokoro-82M-bf16"

# All 28 English voices shipped with Kokoro v1.0, ordered UK first and then by
# the model's own published quality grade (see the upstream VOICES.md). The
# grades matter: the D-tier voices were trained on minutes of audio rather than
# hours and sound noticeably rougher over a full book.
# Kokoro also ships Spanish, French, Hindi, Italian, Japanese, Portuguese and
# Chinese voices; any id is passed straight through, so those work too, though
# Japanese and Chinese need the extra `misaki[ja]` / `misaki[zh]` packages.
_VOICES = [
    # --- British ---
    Voice("bf_emma", "Emma — UK female, warm (best British)", "en-GB", "B-"),
    Voice("bf_isabella", "Isabella — UK female, bright", "en-GB", "C"),
    Voice("bm_fable", "Fable — UK male, storytelling", "en-GB", "C"),
    Voice("bm_george", "George — UK male, warm", "en-GB", "C"),
    Voice("bm_lewis", "Lewis — UK male, deeper", "en-GB", "D+"),
    Voice("bf_alice", "Alice — UK female, light", "en-GB", "D"),
    Voice("bf_lily", "Lily — UK female, soft", "en-GB", "D"),
    Voice("bm_daniel", "Daniel — UK male, measured", "en-GB", "D"),
    # --- American ---
    Voice("af_heart", "Heart — US female (best overall)", "en-US", "A"),
    Voice("af_bella", "Bella — US female, warm", "en-US", "A-"),
    Voice("af_nicole", "Nicole — US female, soft/intimate", "en-US", "B-"),
    Voice("af_aoede", "Aoede — US female", "en-US", "C+"),
    Voice("af_kore", "Kore — US female", "en-US", "C+"),
    Voice("af_sarah", "Sarah — US female, clear", "en-US", "C+"),
    Voice("am_fenrir", "Fenrir — US male, energetic", "en-US", "C+"),
    Voice("am_michael", "Michael — US male, warm", "en-US", "C+"),
    Voice("am_puck", "Puck — US male, lively", "en-US", "C+"),
    Voice("af_alloy", "Alloy — US female", "en-US", "C"),
    Voice("af_nova", "Nova — US female", "en-US", "C"),
    Voice("af_sky", "Sky — US female, bright", "en-US", "C-"),
    Voice("af_jessica", "Jessica — US female", "en-US", "D"),
    Voice("af_river", "River — US female", "en-US", "D"),
    Voice("am_echo", "Echo — US male", "en-US", "D"),
    Voice("am_eric", "Eric — US male", "en-US", "D"),
    Voice("am_liam", "Liam — US male", "en-US", "D"),
    Voice("am_onyx", "Onyx — US male, deep", "en-US", "D"),
    Voice("am_santa", "Santa — US male, character", "en-US", "D-"),
    Voice("am_adam", "Adam — US male, deep", "en-US", "F+"),
]

# Kokoro's language prefixes: a=American, b=British, e=Spanish, f=French,
# h=Hindi, i=Italian, j=Japanese, p=Portuguese, z=Chinese.
_LANG_PREFIXES = set("abefhijpz")


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
        lang_code = voice[0] if voice[:1] in _LANG_PREFIXES else "a"
        # Kokoro splits on newlines and voices each fragment separately, which
        # drops a falling cadence mid-sentence. One call must be one utterance.
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.float32)
        segments = []
        for result in model.generate(
            text=text, voice=voice, speed=speed, lang_code=lang_code
        ):
            segments.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(segments)

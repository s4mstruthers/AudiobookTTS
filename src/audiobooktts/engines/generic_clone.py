"""Larger cloning models (MLX only), behind the same interface as Chatterbox.

Chatterbox is 0.5B and reproduces timbre well but imposes its own articulation,
so a non-native accent comes out wrong. Bigger models saw a wider range of
speakers and may hold an accent better. They all take a reference clip; the
only real differences are what the argument is called and whether they want a
path or samples, which ``_call_kwargs`` works out by inspection.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np

from audiobooktts.engines.base import CloningEngine, EngineUnavailableError

logger = logging.getLogger(__name__)


class GenericCloneEngine(CloningEngine):
    """Wraps any mlx-audio TTS model that clones from a reference clip."""

    deterministic = False
    sample_rate = 24000

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self._model = None
        self._params: dict = {}

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError:
            raise EngineUnavailableError(
                f'The {self.name} engine needs MLX (Apple Silicon): pip install "audiobooktts[mlx]"'
            ) from None
        logger.info("Loading %s with MLX", self.model_id)
        self._model = load_model(self.model_id)
        rate = getattr(self._model, "sample_rate", None)
        if isinstance(rate, int):
            self.sample_rate = rate
        self._params = dict(inspect.signature(self._model.generate).parameters)

    def _call_kwargs(self, text: str, ref: Path | None) -> dict:
        params = self._params
        kwargs: dict = {"text": text}
        if "verbose" in params:
            kwargs["verbose"] = False
        if ref is None:
            return kwargs
        # Some models take a path, others want samples. Higgs, for one, tries
        # float() on whatever it is given and fails loudly on a path.
        import mlx.core as mx

        from audiobooktts.voices import load_clip

        samples, rate = load_clip(ref)
        for key in ("ref_audio", "audio_prompt", "speaker_audio"):
            if key in params:
                kwargs[key] = mx.array(samples)
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
        from audiobooktts.voices import reference_path

        self.load()
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.float32)
        kwargs = self._call_kwargs(text, reference_path(voice))
        segments = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in self._model.generate(**kwargs)
        ]
        return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)

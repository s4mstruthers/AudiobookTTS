"""Match every voice to the same narration pace.

Cloned voices inherit their reference speaker's speaking rate, so they differ
noticeably: measured across one set of voices the spread was 158.6 to 180.0
words per minute, a 13% difference. Commercial audiobooks sit around 150-160
wpm, so most TTS output is also a shade fast.

Each voice is measured once on a fixed passage and the result cached. The
correction is applied as a pitch-preserving time-stretch, which is the only
option that works across backends: Chatterbox's generate() has no working
speed parameter.
"""

from __future__ import annotations

import logging
import threading

from audiobooktts import config
from audiobooktts.audio import trim_silence
from audiobooktts.fsutil import read_json, write_json

logger = logging.getLogger(__name__)

# Deliberately long. Chatterbox is autoregressive and its pace wanders: the
# same voice measured 157.9 to 182.2 wpm across five runs of a 30-word line,
# a wider spread than the difference between voices. Rate over a longer passage
# averages that out, and better matches a whole chapter anyway.
CALIBRATION_TEXT = (
    "The evening train was late again, and the platform had almost emptied by "
    "the time it arrived. She pulled her coat tighter and waited, counting the "
    "seconds without meaning to. Somewhere behind the hedge a bird began, "
    "stopped, and began again. The lamp above the door turned slowly, patient "
    "and enormous, indifferent to the weather and to her. She had come this way "
    "every evening for eleven years, and could have walked the length of the "
    "platform with her eyes closed. Morning would bring letters, and with the "
    "letters news, and with the news the ordinary business of deciding what to "
    "do about it. For now there was only the cold, the quiet, and the distant "
    "sound of an engine that might or might not be the one she wanted."
)
CALIBRATION_WORDS = len(CALIBRATION_TEXT.split())
# Averaged over several passes. Two was not enough: the same voice still
# measured 180.7-189.5 wpm across four runs of the long passage, and two
# Chatterbox voices whose true rates are within 2 wpm of each other came out
# 8 wpm apart, giving one a visibly heavier stretch than the other.
CALIBRATION_RUNS = 4

# atempo degrades audibly outside roughly ±25%, so refuse to correct beyond it.
MIN_TEMPO = 0.75
MAX_TEMPO = 1.35
# Bounds on the final stretch, user speed included.
MIN_SPEED = 0.5
MAX_SPEED = 2.0

# Serialises read-modify-write of the cache, and stops two callers (a preview
# and a job) from calibrating the same voice twice at once.
_lock = threading.RLock()


def _key(engine_name: str, voice: str) -> str:
    return f"{engine_name}/{voice or 'default'}"


def _load_cache() -> dict[str, float]:
    data = read_json(config.PACING_CACHE_PATH, default={})
    if not isinstance(data, dict):
        return {}
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def _save_cache(cache: dict[str, float]) -> None:
    write_json(config.PACING_CACHE_PATH, dict(sorted(cache.items())))


def cached_wpm(engine_name: str, voice: str) -> float | None:
    """The stored rate for a voice, or None if it has never been measured."""
    return _load_cache().get(_key(engine_name, voice))


def is_calibrated(engine, voice: str) -> bool:
    return cached_wpm(engine.name, voice) is not None


def measure_wpm(engine, voice: str, use_cache: bool = True) -> float:
    """Words per minute this voice narrates at. Cached after the first call.

    The passage is synthesised in the same chunks a real render uses, with the
    same silence trimming, so the rate measured is the rate you get.
    """
    from audiobooktts.textproc import chunk_paragraphs

    key = _key(engine.name, voice)
    with _lock:
        cache = _load_cache()
        if use_cache and key in cache:
            return cache[key]

        chunks = [text for text, _ in chunk_paragraphs(CALIBRATION_TEXT)]
        rates = []
        for run in range(CALIBRATION_RUNS):
            logger.info("Calibrating pace of %s (pass %d/%d)", key, run + 1, CALIBRATION_RUNS)
            samples = sum(
                trim_silence(engine.synthesize(text, voice), engine.sample_rate).size
                for text in chunks
            )
            seconds = samples / engine.sample_rate
            if seconds > 0:
                rates.append(CALIBRATION_WORDS / seconds * 60)
        wpm = round(sum(rates) / len(rates), 2) if rates else 0.0

        cache = _load_cache()  # re-read: another voice may have been added meanwhile
        cache[key] = wpm
        _save_cache(cache)
        return wpm


def reference_wpm(engine, voice: str) -> float:
    """The rate to correct against for this engine and voice.

    Deterministic engines get a per-voice figure. Autoregressive ones do not:
    their pace wanders by about ±5 wpm between runs, which is as large as the
    real differences between their voices — measured twice, two voices 2 wpm
    apart came out 8 wpm apart and drew visibly different stretches. Pooling
    every measured voice for such an engine gives all of them the same
    correction, which is the point of syncing pace in the first place, and
    keeps the stretch smaller.
    """
    own = measure_wpm(engine, voice)
    if getattr(engine, "deterministic", True):
        return own
    prefix = f"{engine.name}/"
    rates = [v for k, v in _load_cache().items() if k.startswith(prefix) and v > 0]
    return sum(rates) / len(rates) if rates else own


def tempo_for(engine, voice: str, target_wpm: float, user_speed: float = 1.0) -> float:
    """The time-stretch factor bringing this voice to target_wpm, times user_speed.

    Pacing is skipped (only user_speed applies) when target_wpm <= 0 or the
    voice cannot be measured, so callers can apply the result unconditionally.
    Only the automatic correction is limited; the user's own speed is honoured.
    """
    correction = 1.0
    if target_wpm and target_wpm > 0:
        try:
            natural = reference_wpm(engine, voice)
        except Exception:
            logger.warning(
                "Could not measure the pace of %s; pacing disabled for it", voice, exc_info=True
            )
            natural = 0.0
        if natural > 0:
            correction = max(MIN_TEMPO, min(MAX_TEMPO, target_wpm / natural))
    tempo = float(user_speed or 1.0) * correction
    return max(MIN_SPEED, min(MAX_SPEED, tempo))


def forget(voice: str | None = None) -> None:
    """Drop cached measurements, for a voice (across engines) or entirely."""
    with _lock:
        if voice is None:
            cache: dict[str, float] = {}
        else:
            cache = {k: v for k, v in _load_cache().items() if k.split("/", 1)[-1] != voice}
        _save_cache(cache)

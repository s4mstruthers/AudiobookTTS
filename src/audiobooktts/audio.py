"""In-memory signal helpers for mono float32 audio."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

SILENCE_THRESHOLD = 0.01
# Keep a good slice of the natural decay either side. Trimming hard against the
# speech and splicing in pure digital silence is what makes narration sound
# breathless — the voice stops dead instead of releasing.
SILENCE_KEEP_S = 0.14
# The engine leaves roughly 0.13s between sentences inside one utterance, and
# less than that at commas. Anything at or above this is treated as a sentence
# break worth lengthening; shorter dips are left alone.
INTERNAL_PAUSE_MIN_S = 0.09


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(sample_rate * seconds)), dtype=np.float32)


def as_mono_float32(audio) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Strip the model's leading/trailing padding, keeping a short margin."""
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > SILENCE_THRESHOLD)
    if loud.size == 0:
        return np.zeros(0, dtype=np.float32)
    keep = int(sample_rate * SILENCE_KEEP_S)
    start = max(0, int(loud[0]) - keep)
    end = min(audio.size, int(loud[-1]) + keep)
    return audio[start:end]


def stretch_internal_pauses(
    audio: np.ndarray, sample_rate: int, target_s: float, max_gaps: int
) -> np.ndarray:
    """Lengthen the gaps the engine placed between sentences — those only.

    Reading a paragraph in one call keeps its natural flow, but the engine's
    own sentence gaps are far too brisk for narration and cannot be
    configured. The silences are therefore found and padded afterwards.

    max_gaps is what makes this safe. Speech is full of quiet moments — soft
    consonants, breaths, the dip between words — and padding those produces
    audio that seems to cut out mid-phrase: one paragraph of two sentences
    offered fifteen candidate silences. Only the max_gaps longest are
    stretched, because a full stop is held longer than anything inside a
    sentence, so with one gap per full stop the rest are left alone.
    """
    if audio.size == 0 or target_s <= 0 or max_gaps <= 0:
        return audio
    quiet = np.abs(audio) <= SILENCE_THRESHOLD
    if not quiet.any():
        return audio

    # Runs of silence, as (start, end) index pairs.
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    bounds = np.concatenate(([0], edges + 1, [audio.size]))
    min_len = int(sample_rate * INTERNAL_PAUSE_MIN_S)
    target_len = int(sample_rate * target_s)

    runs = [
        (start, end)
        for start, end in pairwise(bounds)
        if quiet[start] and start > 0 and end < audio.size and (end - start) >= min_len
    ]
    # The longest silences are the sentence ends.
    chosen = {
        start for start, _ in sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:max_gaps]
    }
    if not chosen:
        return audio

    pieces: list[np.ndarray] = []
    for start, end in pairwise(bounds):
        segment = audio[start:end]
        pieces.append(segment)
        if start in chosen and target_len > segment.size:
            pieces.append(np.zeros(target_len - segment.size, dtype=np.float32))
    return np.concatenate(pieces)

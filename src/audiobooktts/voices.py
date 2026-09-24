"""The voice library: reference clips for voice cloning, and the recordings they came from.

Cloning quality tracks reference quality closely: a noisy or inconsistent clip
gives a noisy clone. This module normalises whatever you feed it — a LibriVox
mp3, a phone recording — into the short, loudness-matched mono wav the cloning
engines expect, and keeps the original so the clip can be re-cut later.

Every audio file in the voices folder is a selectable voice, named by its file
name without the extension.
"""

from __future__ import annotations

import contextlib
import re
import shutil
from pathlib import Path

import numpy as np

from audiobooktts import config, ffmpeg

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
#: The model's own built-in voice, used when no reference clip is selected.
DEFAULT_VOICE = "default"

TARGET_SR = 24000
# Chatterbox slices the reference: the base model reads 6s for the speaker
# embedding and 10s for the decoder prompt, the turbo variant 15s and 10s.
# 15s therefore fills every window either model has, and the surplus is simply
# discarded. Both reject clips of 5 seconds or less.
DEFAULT_DURATION = 15.0
# LibriVox files open with a spoken boilerplate credit; starting inside it gives
# the clone a stilted "This is a LibriVox recording" cadence.
LIBRIVOX_INTRO_SKIP = 45.0

MIN_USEFUL_S = 6.0  # Chatterbox rejects anything at or under 5 seconds
MAX_USEFUL_S = 30.0  # beyond 15s the model discards the surplus anyway
MAX_NAME_LEN = 64


class VoiceClipError(RuntimeError):
    pass


def sanitize_name(name: str) -> str:
    """A voice name that is safe as a file name on every platform."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:MAX_NAME_LEN]
    if not safe:
        raise VoiceClipError("Please give the voice a name (letters, digits, - or _)")
    if safe.lower() == DEFAULT_VOICE:
        raise VoiceClipError(f'"{DEFAULT_VOICE}" is reserved for the model\'s own voice')
    return safe


def clip_paths() -> dict[str, Path]:
    """Every reference clip, keyed by voice name. A .wav wins over other formats."""
    found: dict[str, Path] = {}
    if not config.VOICES_DIR.is_dir():
        return found
    rank = {s: i for i, s in enumerate(AUDIO_SUFFIXES)}
    files = sorted(
        (
            p
            for p in config.VOICES_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in rank and not p.name.startswith(".")
        ),
        key=lambda p: (p.stem.lower(), rank[p.suffix.lower()]),
    )
    for p in files:
        found.setdefault(p.stem, p)
    return found


def reference_path(voice: str) -> Path | None:
    """Resolve a voice to its reference clip; None means the built-in voice.

    A library name is looked up first; any other existing audio file path is
    accepted too, so ``--voice path/to/clip.wav`` works from the command line.
    """
    if not voice or voice == DEFAULT_VOICE:
        return None
    clip = clip_paths().get(voice)
    if clip is not None:
        return clip
    candidate = Path(voice).expanduser()
    if candidate.is_file() and candidate.suffix.lower() in AUDIO_SUFFIXES:
        return candidate
    raise VoiceClipError(
        f"Unknown voice {voice!r}. Add one with `abtts add-voice`, "
        f"or put a clip in {config.VOICES_DIR}"
    )


def clip_key(path: Path) -> tuple[str, int, int]:
    """Identity of a clip's current contents, so a re-cut clip is re-read."""
    st = path.stat()
    return str(path.resolve()), st.st_mtime_ns, st.st_size


def load_clip(path: Path) -> tuple[np.ndarray, int]:
    """Decode a clip to mono float32 samples and their sample rate.

    Mixed down rather than flattened: reshaping a stereo array to 1-D
    interleaves the channels into double-length noise.
    """
    import soundfile as sf

    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return data.mean(axis=1), int(rate)
    except (RuntimeError, sf.LibsndfileError):
        pass  # e.g. m4a, which libsndfile cannot read; ffmpeg can
    proc = ffmpeg.run(
        ["-v", "error", "-i", str(path), "-ac", "1", "-ar", str(TARGET_SR), "-f", "f32le", "pipe:1"]
    )
    if proc.returncode != 0 or not proc.stdout:
        raise VoiceClipError(f"Could not read audio from {path.name}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), TARGET_SR


def _duration(path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        try:
            return ffmpeg.probe_duration(path)
        except ValueError as e:
            raise VoiceClipError(str(e)) from None


def add_voice(
    source: str | Path,
    name: str,
    start: float = LIBRIVOX_INTRO_SKIP,
    duration: float = DEFAULT_DURATION,
    keep_source: bool = True,
) -> Path:
    """Extract a segment, normalise it, and save it as a selectable voice.

    Returns the path of the saved clip.
    """
    name = sanitize_name(name)
    source = Path(source).expanduser()
    if not source.is_file():
        raise VoiceClipError(f"No such file: {source}")
    if not MIN_USEFUL_S <= duration <= MAX_USEFUL_S:
        raise VoiceClipError(
            f"Clip length must be {MIN_USEFUL_S:.0f}–{MAX_USEFUL_S:.0f}s "
            f"(about 15s is ideal); got {duration:g}s"
        )

    total = _duration(source)
    if total < duration:
        raise VoiceClipError(
            f"Source is only {total:.1f}s long, shorter than the {duration:g}s clip"
        )
    start = max(0.0, start)
    if start + duration > total:
        # Fall back to the middle of the file, which is usually clean narration.
        start = (total - duration) / 2

    config.VOICES_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.VOICES_DIR / f"{name}.wav"
    tmp = config.VOICES_DIR / f".{name}.tmp.wav"
    result = ffmpeg.run(
        [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{duration:.2f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SR),
            # Match broadcast loudness so clips from different sources behave alike.
            "-af",
            "loudnorm=I=-19:TP=-2:LRA=11",
            "-f",
            "wav",
            str(tmp),
        ],
        text=True,
    )
    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise VoiceClipError(f"ffmpeg failed: {result.stderr[-500:]}")
    from audiobooktts.fsutil import replace_file

    replace_file(tmp, dest)
    # One clip per name: a manually added name.mp3 would otherwise shadow it.
    for other in config.VOICES_DIR.glob(f"{name}.*"):
        if other != dest and other.stem == name and other.suffix.lower() in AUDIO_SUFFIXES:
            other.unlink(missing_ok=True)

    # The clip changed, so any measured pace is stale.
    from audiobooktts.pacing import forget

    forget(name)
    if keep_source:
        # Keep the original so the clip can be re-cut from a different point.
        with contextlib.suppress(OSError):
            store_source(source, name)
    return dest


def list_clips() -> list[dict]:
    """Every saved reference clip with the details the UI needs."""
    from audiobooktts.pacing import cached_wpm

    rows = []
    for name, path in sorted(clip_paths().items(), key=lambda kv: kv[0].lower()):
        try:
            seconds = _duration(path)
        except VoiceClipError:
            seconds = 0.0
        rows.append(
            {
                "name": name,
                "seconds": round(seconds, 1),
                "bytes": path.stat().st_size,
                "wpm": cached_wpm("chatterbox", name),
                "has_source": source_path(name) is not None,
            }
        )
    return rows


def delete_voice(name: str) -> bool:
    """Remove a voice's clip, its source recording and its calibration."""
    from audiobooktts.pacing import forget

    clip = clip_paths().get(name)
    if clip is None:
        return False
    for path in config.VOICES_DIR.glob(f"{glob_escape(name)}.*"):
        if path.stem == name and path.suffix.lower() in AUDIO_SUFFIXES:
            path.unlink(missing_ok=True)
    source = source_path(name)
    if source is not None:
        source.unlink(missing_ok=True)
    forget(name)
    return True


def glob_escape(name: str) -> str:
    """Escape glob metacharacters so a name only ever matches itself."""
    return re.sub(r"([*?\[])", r"[\1]", name)


def source_path(name: str) -> Path | None:
    """The original recording a voice was cut from, if it was kept."""
    if not config.VOICE_SOURCES_DIR.is_dir():
        return None
    for p in sorted(config.VOICE_SOURCES_DIR.glob(f"{glob_escape(name)}.*")):
        if p.is_file() and p.stem == name:
            return p
    return None


def store_source(src: str | Path, name: str) -> Path:
    """Keep the original recording so a clip can be re-cut later."""
    name = sanitize_name(name)
    src = Path(src)
    config.VOICE_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.VOICE_SOURCES_DIR / f"{name}{src.suffix.lower() or '.audio'}"
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    for old in config.VOICE_SOURCES_DIR.glob(f"{name}.*"):
        if old != dest and old.stem == name:
            old.unlink(missing_ok=True)
    return dest


def waveform_peaks(path: str | Path, buckets: int = 900) -> dict:
    """Peak amplitude per bucket, for drawing a waveform.

    Decoded to mono 8kHz first: plenty for a visual envelope and far quicker
    than reading a two-minute file at full rate.
    """
    path = Path(path)
    proc = ffmpeg.run(
        ["-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000", "-f", "f32le", "pipe:1"]
    )
    if proc.returncode != 0 or not proc.stdout:
        raise VoiceClipError(f"Could not read audio from {path.name}")
    samples = np.abs(np.frombuffer(proc.stdout, dtype=np.float32))
    if samples.size == 0:
        raise VoiceClipError("Recording appears to be empty")
    buckets = max(1, min(int(buckets), samples.size))
    edges = np.linspace(0, samples.size, buckets + 1, dtype=np.int64)[:-1]
    peaks = np.maximum.reduceat(samples, edges)
    top = float(peaks.max()) or 1.0
    return {
        "duration": round(samples.size / 8000, 2),
        "peaks": [round(float(p) / top, 4) for p in peaks],
    }


def cut_segment(path: str | Path, start: float, duration: float) -> bytes:
    """A plain wav of one stretch, for auditioning before committing to it."""
    proc = ffmpeg.run(
        [
            "-v",
            "error",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{max(0.1, min(duration, MAX_USEFUL_S)):.3f}",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SR),
            "-f",
            "wav",
            "pipe:1",
        ]
    )
    if proc.returncode != 0 or not proc.stdout:
        raise VoiceClipError("Could not cut that segment")
    return proc.stdout

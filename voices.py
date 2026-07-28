"""Turn any recording into a clean reference clip for voice cloning.

Cloning quality tracks reference quality closely: a noisy or inconsistent clip
gives a noisy clone. This normalises whatever you feed it — a LibriVox mp3, a
phone recording — into the short, loudness-matched mono wav the cloning engines
expect.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from audiobooktts.config import APP_DIR
from audiobooktts.engines.chatterbox import VOICES_DIR

TARGET_SR = 24000
# Chatterbox slices the reference: the base model reads 6s for the speaker
# embedding and 10s for the decoder prompt, the turbo variant 15s and 10s.
# 15s therefore fills every window either model has, and the surplus is simply
# discarded. Both reject clips of 5 seconds or less.
DEFAULT_DURATION = 15.0
# LibriVox files open with a spoken boilerplate credit; starting inside it gives
# the clone a stilted "This is a LibriVox recording" cadence.
LIBRIVOX_INTRO_SKIP = 45.0

MIN_USEFUL_S = 6.0   # Chatterbox rejects anything at or under 5 seconds
MAX_USEFUL_S = 30.0  # beyond 15s the model discards the surplus anyway


class VoiceClipError(RuntimeError):
    pass


def _ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise VoiceClipError(f"Could not read audio from {path}")


def add_voice(
    source: str | Path,
    name: str,
    start: float = LIBRIVOX_INTRO_SKIP,
    duration: float = DEFAULT_DURATION,
    voices_dir: Path | None = None,
) -> Path:
    """Extract a segment, normalise it, and save it as a selectable voice.

    Returns the path of the saved clip.
    """
    if shutil.which("ffmpeg") is None:
        raise VoiceClipError("ffmpeg is required (brew install ffmpeg)")
    source = Path(source).expanduser()
    if not source.is_file():
        raise VoiceClipError(f"No such file: {source}")

    if not (MIN_USEFUL_S <= duration <= MAX_USEFUL_S):
        raise VoiceClipError(
            f"Clip length must be {MIN_USEFUL_S:.0f}–{MAX_USEFUL_S:.0f}s "
            f"(5–10s is ideal); got {duration:g}s"
        )

    total = _ffprobe_duration(source)
    if start + duration > total:
        # Fall back to the middle of the file, which is usually clean narration.
        start = max(0.0, (total - duration) / 2)
    if total < duration:
        raise VoiceClipError(
            f"Source is only {total:.1f}s long, shorter than the {duration:g}s clip"
        )

    out_dir = voices_dir or VOICES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.wav"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-t", f"{duration:.2f}", "-i", str(source),
        "-ac", "1", "-ar", str(TARGET_SR),
        # Match broadcast loudness so clips from different sources behave alike
        "-af", "loudnorm=I=-19:TP=-2:LRA=11",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dest.exists():
        raise VoiceClipError(f"ffmpeg failed: {result.stderr[-500:]}")
    # Keep the original so the clip can be re-cut from a different point.
    try:
        store_source(source, name)
    except OSError:
        pass
    return dest


def list_voices(voices_dir: Path | None = None) -> list[dict]:
    """Every saved reference clip with the details the UI needs."""
    from audiobooktts.pacing import _load_cache

    out_dir = voices_dir or VOICES_DIR
    cache = _load_cache()
    rows = []
    if out_dir.is_dir():
        for p in sorted(out_dir.iterdir()):
            if p.suffix.lower() != ".wav":
                continue
            try:
                seconds = _ffprobe_duration(p)
            except VoiceClipError:
                seconds = 0.0
            rows.append({
                "name": p.stem,
                "seconds": round(seconds, 1),
                "bytes": p.stat().st_size,
                "wpm": cache.get(f"chatterbox/{p.stem}"),
            })
    return rows


def delete_voice(name: str, voices_dir: Path | None = None) -> bool:
    """Remove a reference clip and forget its calibration."""
    from audiobooktts.pacing import forget

    out_dir = voices_dir or VOICES_DIR
    path = out_dir / f"{name}.wav"
    if not path.is_file():
        return False
    path.unlink()
    forget(name)
    return True


def conditioning_windows() -> dict[str, float]:
    """How much of a reference clip the active model actually reads.

    Anything past the longer window is sliced off before the model sees it, and
    the speaker window is the one that fixes voice identity — so the *start* of
    a clip matters more than its length.
    """
    from audiobooktts.config import Config

    model_id = getattr(Config.load(), "chatterbox_model", "")
    try:
        if "turbo" in model_id.lower():
            from mlx_audio.tts.models.chatterbox_turbo import chatterbox_turbo as m

            cls = m.ChatterboxTurboTTS
        else:
            from mlx_audio.tts.models.chatterbox import chatterbox as m

            cls = m.Model
        return {
            "speaker_s": cls.ENC_COND_LEN / 16000,
            "decoder_s": cls.DEC_COND_LEN / 24000,
            "useful_s": max(cls.ENC_COND_LEN / 16000, cls.DEC_COND_LEN / 24000),
        }
    except Exception:
        return {"speaker_s": 15.0, "decoder_s": 10.0, "useful_s": 15.0}


SOURCE_DIR = APP_DIR / "voice_sources"


def source_path(name: str) -> Path | None:
    """The original recording a voice was cut from, if it was kept."""
    if not SOURCE_DIR.is_dir():
        return None
    for p in sorted(SOURCE_DIR.glob(f"{name}.*")):
        if p.is_file():
            return p
    return None


def store_source(src: str | Path, name: str) -> Path:
    """Keep the original recording so a clip can be re-cut later."""
    src = Path(src)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    dest = SOURCE_DIR / f"{name}{src.suffix.lower() or '.audio'}"
    for old in SOURCE_DIR.glob(f"{name}.*"):
        if old != dest:
            old.unlink(missing_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    return dest


def waveform_peaks(path: str | Path, buckets: int = 900) -> dict:
    """Peak amplitude per bucket, for drawing a waveform.

    Decoded to mono 8kHz first: plenty for a visual envelope and far quicker
    than reading a two-minute file at full rate.
    """
    import numpy as np

    path = Path(path)
    duration = _ffprobe_duration(path)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", "8000", "-f", "f32le", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise VoiceClipError(f"Could not read audio from {path.name}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        raise VoiceClipError("Recording appears to be silent")
    buckets = max(1, min(buckets, samples.size))
    edges = np.linspace(0, samples.size, buckets + 1, dtype=int)
    peaks = [
        float(np.abs(samples[a:b]).max()) if b > a else 0.0
        for a, b in zip(edges[:-1], edges[1:])
    ]
    top = max(peaks) or 1.0
    return {
        "duration": round(duration, 2),
        "peaks": [round(p / top, 4) for p in peaks],
    }


def cut_segment(path: str | Path, start: float, duration: float) -> bytes:
    """A plain wav of one stretch, for auditioning before committing to it."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}",
         "-t", f"{max(0.1, duration):.3f}", "-i", str(path),
         "-ac", "1", "-ar", str(TARGET_SR), "-f", "wav", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise VoiceClipError("Could not cut that segment")
    return proc.stdout

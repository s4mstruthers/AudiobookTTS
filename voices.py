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

from audiobooktts.engines.chatterbox import VOICES_DIR

TARGET_SR = 24000
# Chatterbox reads at most 15s into the reference for the speaker embedding and
# 10s for the decoder prompt, slicing away the rest, so 15s fills every window
# it has. It also asserts the clip is longer than 5 seconds.
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
    return dest

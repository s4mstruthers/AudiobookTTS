"""Locating and running the ffmpeg command-line tools.

ffmpeg does the decoding, time-stretching, loudness mastering and M4B muxing.
It is found on PATH, or wherever ``AUDIOBOOKTTS_FFMPEG`` points (useful on
Windows, where it is often unpacked into a folder that is not on PATH).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg or ffprobe is not installed or cannot be found."""


def install_hint() -> str:
    if sys.platform == "win32":
        return (
            "Install it with `winget install Gyan.FFmpeg` (or `choco install ffmpeg`), "
            "then open a new terminal."
        )
    if sys.platform == "darwin":
        return "Install it with `brew install ffmpeg`."
    return (
        "Install it with your package manager, "
        "e.g. `sudo apt install ffmpeg` or `sudo dnf install ffmpeg`."
    )


@cache
def find_tool(name: str) -> str:
    """Absolute path of ``ffmpeg`` or ``ffprobe``; raises if it is missing."""
    override = os.environ.get("AUDIOBOOKTTS_FFMPEG")
    if override:
        # Accept either the ffmpeg executable itself or the folder holding it.
        base = Path(override)
        folder = base if base.is_dir() else base.parent
        for candidate in (folder / name, folder / f"{name}.exe"):
            if candidate.is_file():
                return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FFmpegNotFoundError(f"{name} was not found. {install_hint()}")


def ffmpeg() -> str:
    return find_tool("ffmpeg")


def ffprobe() -> str:
    return find_tool("ffprobe")


def run(
    args: list[str], *, input: bytes | None = None, text: bool = False
) -> subprocess.CompletedProcess:
    """Run ffmpeg with ``args``, capturing output (never raises on exit status)."""
    return subprocess.run(
        [ffmpeg(), *args],
        input=input,
        capture_output=True,
        text=text,
        # Decode logs as UTF-8 whatever the console code page: book titles in
        # metadata errors would otherwise raise on Windows.
        **({"encoding": "utf-8", "errors": "replace"} if text else {}),
    )


def version() -> str | None:
    """The first line of ``ffmpeg -version``, or None if ffmpeg is unavailable."""
    try:
        result = run(["-version"], text=True)
    except (FFmpegNotFoundError, OSError):
        return None
    return result.stdout.splitlines()[0] if result.stdout else None


def probe_duration(path: str | Path) -> float:
    """Duration of a media file in seconds. Raises ValueError if unreadable."""
    result = subprocess.run(
        [ffprobe(), "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise ValueError(f"Could not read audio from {Path(path).name}") from None

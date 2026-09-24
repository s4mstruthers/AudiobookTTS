"""Play a WAV file through the speakers, on any operating system."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _players(path: Path) -> list[list[str]]:
    candidates = []
    # ffplay ships with most ffmpeg builds, which this program needs anyway.
    ffplay = shutil.which("ffplay")
    if ffplay:
        candidates.append([ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])
    if sys.platform == "darwin" and shutil.which("afplay"):
        candidates.append(["afplay", str(path)])
    for tool in ("paplay", "pw-play", "aplay"):
        if shutil.which(tool):
            candidates.append([tool, str(path)])
    return candidates


def play_wav(path: str | Path) -> bool:
    """Play ``path`` and block until it finishes. False if no player was found."""
    path = Path(path)
    for cmd in _players(path):
        try:
            if subprocess.run(cmd, check=False).returncode == 0:
                return True
        except OSError:
            continue
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return True
    return False

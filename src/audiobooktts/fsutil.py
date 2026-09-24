"""Small, cross-platform filesystem helpers.

Everything here exists because the obvious one-liner misbehaves somewhere:

* ``Path.write_text`` uses the locale encoding, which is cp1252 on most
  Windows installs, so book titles, respellings and chapter names outside that
  code page either raise or are silently mangled. All text IO here is UTF-8.
* ``os.replace`` onto a file another thread has open fails on Windows with
  ``PermissionError`` rather than waiting, so atomic writes retry briefly.
* Windows forbids more characters in file names than POSIX does, plus
  trailing dots and spaces and a handful of reserved device names.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

_REPLACE_ATTEMPTS = 10
_REPLACE_BACKOFF_S = 0.05

# Characters Windows rejects in file names, plus ASCII control characters.
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _replace_with_retry(src: Path, dst: Path) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            # Windows: the destination is momentarily open elsewhere (a reader
            # listing jobs, an antivirus scan). It clears within milliseconds.
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write a file so readers only ever see the old or the new contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique temporary name per writer, so concurrent writers never collide.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        _replace_with_retry(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def read_json(path: str | Path, default: Any = None) -> Any:
    """Parse a UTF-8 JSON file, returning ``default`` if it is missing or corrupt."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: str | Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def replace_file(src: str | Path, dst: str | Path) -> None:
    """``os.replace`` that tolerates Windows' transient sharing violations."""
    _replace_with_retry(Path(src), Path(dst))


def safe_filename(name: str, max_len: int = 150, fallback: str = "untitled") -> str:
    """A file name that is valid on Windows, macOS and Linux alike.

    Kept well under Windows' 260-character path limit so that the output
    folder has room to be nested somewhere reasonable.
    """
    cleaned = _UNSAFE_FILENAME_RE.sub("", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned[:max_len].rstrip(" .")
    if not cleaned:
        return fallback
    if cleaned.split(".")[0].upper() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned


def is_within(path: str | Path, root: str | Path) -> bool:
    """Whether ``path`` resolves to somewhere inside ``root``."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False
    return True

"""Job storage: one directory per conversion job, with a manifest for resume."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from audiobooktts import config
from audiobooktts.fsutil import atomic_write_text

#: Job states. "running" jobs record the owning process id so another process
#: can tell a live job from one orphaned by a crash.
STATUSES = ("pending", "running", "done", "failed", "cancelled", "interrupted")

_JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


class JobNotFoundError(KeyError):
    """No job with this id exists."""


@dataclass
class ChapterState:
    index: int
    title: str
    # Empty means "use the job's voice". Set per chapter so a book with more
    # than one narrator — an epilogue in another character's voice — works.
    voice: str = ""
    n_chunks: int = 0
    done: bool = False
    duration_s: float = 0.0


@dataclass
class Manifest:
    job_id: str
    epub_path: str
    epub_sha256: str
    book_title: str
    book_author: str
    engine: str
    voice: str
    speed: float
    status: str = "pending"
    error: str = ""
    created_at: float = field(default_factory=time.time)
    # Where the finished m4b goes. Persisted so `resume` writes to the same
    # place the job was created with, not whatever the current config says.
    output_dir: str = ""
    output_path: str = ""
    # A custom cover image, used instead of the epub's own when set.
    cover_path: str = ""
    # Process currently rendering this job, while status is "running".
    pid: int = 0
    # The time-stretch chosen for each voice on the first run. Reusing it on
    # resume keeps every chapter of one book at the same pace, even if voice
    # calibrations have changed in between.
    tempo_by_voice: dict[str, float] = field(default_factory=dict)
    chapters: list[ChapterState] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        d = dict(d)
        chapter_keys = {f.name for f in fields(ChapterState)}
        chapters = [
            ChapterState(**{k: v for k, v in c.items() if k in chapter_keys})
            for c in d.pop("chapters", [])
        ]
        known = {f.name for f in fields(cls)}
        return cls(chapters=chapters, **{k: v for k, v in d.items() if k in known})

    @property
    def chapters_done(self) -> int:
        return sum(1 for c in self.chapters if c.done)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "book"


def is_valid_job_id(job_id: str) -> bool:
    return bool(_JOB_ID_RE.match(job_id))


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is running (works on Windows too)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        error_access_denied = 5
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            # Access denied means it exists but belongs to someone else.
            return ctypes.get_last_error() == error_access_denied
        try:
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == still_active
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        # Signal 0 checks for existence without delivering anything. (Never on
        # Windows, where os.kill would terminate the process.)
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class JobStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else config.JOBS_DIR

    def new_job_id(self, book_title: str) -> str:
        return f"{slugify(book_title)}-{uuid.uuid4().hex[:6]}"

    def job_dir(self, job_id: str) -> Path:
        # Ids reach here from URLs; refuse anything that could escape the root.
        if not is_valid_job_id(job_id):
            raise JobNotFoundError(job_id)
        return self.root / job_id

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def chapter_audio_path(self, job_id: str, index: int) -> Path:
        return self.job_dir(job_id) / f"ch_{index:03d}.flac"

    def exists(self, job_id: str) -> bool:
        return is_valid_job_id(job_id) and self.manifest_path(job_id).is_file()

    def save(self, manifest: Manifest) -> None:
        atomic_write_text(
            self.manifest_path(manifest.job_id), json.dumps(manifest.to_dict(), indent=2)
        )

    def load(self, job_id: str) -> Manifest:
        try:
            text = self.manifest_path(job_id).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise JobNotFoundError(job_id) from None
        return Manifest.from_dict(json.loads(text))

    def list_jobs(self) -> list[Manifest]:
        out = []
        if not self.root.is_dir():
            return out
        for p in self.root.glob("*/manifest.json"):
            try:
                out.append(Manifest.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError):
                continue  # a half-deleted or corrupt job must not hide the rest
        return sorted(out, key=lambda m: m.created_at, reverse=True)

    def reconcile_running(self) -> list[str]:
        """Mark orphaned jobs as interrupted, returning the ids changed.

        A job still marked "running" whose process has died was killed with
        it. Left alone it stays "running" forever: it can never be cancelled
        (no thread to signal) and never offers Resume. Jobs whose process is
        still alive — a web server rendering while the CLI lists jobs — are
        left untouched.
        """
        changed = []
        for manifest in self.list_jobs():
            if manifest.status != "running":
                continue
            if manifest.pid == os.getpid() or pid_alive(manifest.pid):
                continue
            manifest.status = "interrupted"
            manifest.error = "Interrupted: the process rendering it stopped"
            manifest.pid = 0
            self.save(manifest)
            changed.append(manifest.job_id)
        return changed

    def delete(self, job_id: str) -> None:
        d = self.job_dir(job_id)
        if d.exists():
            shutil.rmtree(d)

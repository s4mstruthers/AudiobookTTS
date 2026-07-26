"""Job storage: one directory per conversion job, with a manifest for resume."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from audiobooktts.config import JOBS_DIR


@dataclass
class ChapterState:
    index: int
    title: str
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
    status: str = "pending"  # pending | running | done | failed | cancelled
    error: str = ""
    created_at: float = field(default_factory=time.time)
    # Where the finished m4b goes. Persisted so `resume` writes to the same
    # place the job was created with, not whatever the current config says.
    output_dir: str = ""
    output_path: str = ""
    chapters: list[ChapterState] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        chapters = [ChapterState(**c) for c in d.pop("chapters", [])]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(chapters=chapters, **known)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "book"


class JobStore:
    def __init__(self, root: Path = JOBS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def new_job_id(self, book_title: str) -> str:
        return f"{slugify(book_title)}-{uuid.uuid4().hex[:6]}"

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def chapter_audio_path(self, job_id: str, index: int) -> Path:
        return self.job_dir(job_id) / f"ch_{index:03d}.flac"

    def save(self, manifest: Manifest) -> None:
        d = self.job_dir(manifest.job_id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path(manifest.job_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest.to_dict(), indent=2))
        tmp.replace(self.manifest_path(manifest.job_id))

    def load(self, job_id: str) -> Manifest:
        return Manifest.from_dict(json.loads(self.manifest_path(job_id).read_text()))

    def list_jobs(self) -> list[Manifest]:
        out = []
        for p in sorted(self.root.glob("*/manifest.json")):
            try:
                out.append(Manifest.from_dict(json.loads(p.read_text())))
            except Exception:
                continue
        return sorted(out, key=lambda m: m.created_at, reverse=True)

    def delete(self, job_id: str) -> None:
        import shutil

        d = self.job_dir(job_id)
        if d.exists():
            shutil.rmtree(d)

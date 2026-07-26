"""Conversion pipeline: epub -> per-chapter audio -> packaged m4b.

Progress flows through a callback so the CLI progress bar and the web UI's
SSE stream share one code path. Chapters are checkpointed: a finished
chapter's FLAC plus its manifest entry mean resume never re-synthesizes it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from audiobooktts.config import Config
from audiobooktts.engines import get_engine
from audiobooktts.epub import Book, parse_epub
from audiobooktts.packaging import package_m4b
from audiobooktts.store import ChapterState, JobStore, Manifest, file_sha256
from audiobooktts.textproc import chunk_text, clean_text, normalize_chapter_title


@dataclass
class Progress:
    job_id: str
    stage: str  # parsing | synthesizing | packaging | done | failed
    chapter_index: int = 0
    n_chapters: int = 0
    chunk_index: int = 0
    n_chunks: int = 0
    chapter_title: str = ""
    message: str = ""


ProgressFn = Callable[[Progress], None]


class CancelledError(Exception):
    pass


def create_job(
    epub_path: str | Path,
    config: Config,
    store: JobStore | None = None,
    llm_cleanup: bool = False,
) -> Manifest:
    store = store or JobStore()
    book = parse_epub(epub_path)
    manifest = Manifest(
        job_id=store.new_job_id(book.title),
        epub_path=str(Path(epub_path).resolve()),
        epub_sha256=file_sha256(epub_path),
        book_title=book.title,
        book_author=book.author,
        engine=config.engine,
        voice=config.voice,
        speed=config.speed,
        output_dir=config.output_dir,
        chapters=[
            ChapterState(index=i, title=normalize_chapter_title(ch.title))
            for i, ch in enumerate(book.chapters)
        ],
    )
    if llm_cleanup:
        manifest.status = "pending"
    store.save(manifest)
    return manifest


def run_job(
    job_id: str,
    config: Config,
    store: JobStore | None = None,
    on_progress: ProgressFn | None = None,
    cancel_event: threading.Event | None = None,
    llm_cleanup: bool = False,
) -> Manifest:
    store = store or JobStore()
    manifest = store.load(job_id)
    notify = on_progress or (lambda p: None)

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            manifest.status = "cancelled"
            store.save(manifest)
            raise CancelledError(job_id)

    try:
        notify(Progress(job_id, "parsing", message="Re-parsing epub"))
        if file_sha256(manifest.epub_path) != manifest.epub_sha256:
            raise RuntimeError("Source epub changed since job was created")
        book: Book = parse_epub(manifest.epub_path)
        if len(book.chapters) != len(manifest.chapters):
            raise RuntimeError("Chapter count mismatch on resume")

        cleaner = None
        if llm_cleanup:
            from audiobooktts.llm_cleanup import get_cleaner

            cleaner = get_cleaner(config.llm)

        engine = get_engine(manifest.engine)
        manifest.status = "running"
        store.save(manifest)
        n_chapters = len(book.chapters)

        for i, chapter in enumerate(book.chapters):
            state = manifest.chapters[i]
            audio_path = store.chapter_audio_path(job_id, i)
            if state.done and audio_path.exists():
                continue
            check_cancel()

            text = clean_text(chapter.text)
            if cleaner is not None:
                text = cleaner.clean(text)
            # Announce the chapter title before its body
            title = normalize_chapter_title(chapter.title)
            chunks = chunk_text(text)
            if title:
                chunks = [title + "."] + chunks
            state.n_chunks = len(chunks)

            pieces: list[np.ndarray] = []
            pause = np.zeros(int(engine.sample_rate * 0.4), dtype=np.float32)
            for j, chunk in enumerate(chunks):
                check_cancel()
                notify(
                    Progress(
                        job_id, "synthesizing", i, n_chapters, j, len(chunks), title
                    )
                )
                audio = engine.synthesize(chunk, manifest.voice, manifest.speed)
                if audio.size:
                    pieces.append(audio)
                    pieces.append(pause)

            full = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            tmp = audio_path.with_suffix(".tmp.flac")
            sf.write(tmp, full, engine.sample_rate, format="FLAC")
            tmp.replace(audio_path)
            state.duration_s = float(len(full)) / engine.sample_rate
            state.done = True
            store.save(manifest)

        notify(Progress(job_id, "packaging", n_chapters, n_chapters, message="Packaging m4b"))
        out_dir = Path(manifest.output_dir or config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = package_m4b(
            book=book,
            manifest=manifest,
            store=store,
            out_dir=out_dir,
            bitrate=config.bitrate,
        )
        manifest.output_path = str(output)
        manifest.status = "done"
        store.save(manifest)
        notify(Progress(job_id, "done", n_chapters, n_chapters, message=str(output)))
        return manifest
    except CancelledError:
        raise
    except Exception as e:
        manifest.status = "failed"
        manifest.error = str(e)
        store.save(manifest)
        notify(Progress(job_id, "failed", message=str(e)))
        raise

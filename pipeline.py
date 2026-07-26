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
from audiobooktts.textproc import (
    CLAUSE,
    PARAGRAPH,
    SENTENCE,
    chunk_paragraphs,
    clean_text,
    normalize_chapter_title,
)

# Fallback pause lengths; Config overrides these. The engine pads each
# utterance with ~0.3s leading and ~0.5s trailing silence, which is trimmed
# off, so these values alone set the pacing.
GAP_CLAUSE_S = 0.2      # mid-sentence, where a long sentence had to be split
GAP_SENTENCE_S = 0.65   # after a full stop
GAP_PARAGRAPH_S = 1.1   # between paragraphs
GAP_TITLE_S = 1.8       # after the announced chapter title

_SILENCE_THRESHOLD = 0.01
_SILENCE_KEEP_S = 0.03


def _trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Strip the model's leading/trailing padding, keeping a short margin."""
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > _SILENCE_THRESHOLD)
    if loud.size == 0:
        return np.zeros(0, dtype=np.float32)
    keep = int(sample_rate * _SILENCE_KEEP_S)
    start = max(0, int(loud[0]) - keep)
    end = min(audio.size, int(loud[-1]) + keep)
    return audio[start:end]


def render_chunks(
    engine,
    chunks: list[tuple[str, bool]],
    voice: str,
    speed: float = 1.0,
    config: Config | None = None,
    announces_title: bool = False,
    on_chunk=None,
) -> np.ndarray:
    """Synthesize chunks and join them with the configured pauses.

    Shared by the renderer and the preview so that what you audition is what
    you get. Previewing through a separate path meant hearing the engine's own
    ~0.13s sentence gap instead of the configured pacing.
    """
    cfg = config or Config()
    sr = engine.sample_rate
    gaps = {
        CLAUSE: getattr(cfg, "gap_clause", GAP_CLAUSE_S),
        SENTENCE: getattr(cfg, "gap_sentence", GAP_SENTENCE_S),
        PARAGRAPH: getattr(cfg, "gap_paragraph", GAP_PARAGRAPH_S),
    }
    g_title = getattr(cfg, "gap_title", GAP_TITLE_S)

    # Voices differ in natural pace by more than 10%, and Chatterbox ignores the
    # speed argument entirely, so the correction is a time-stretch applied here.
    from audiobooktts.mastering import apply_filters, atempo_chain
    from audiobooktts.pacing import tempo_for

    tempo = tempo_for(engine, voice, getattr(cfg, "target_wpm", 0.0), speed)
    # Kokoro honours speed natively; let it, and leave the stretch to pacing.
    native_speed = speed if engine.name == "kokoro" else 1.0
    if engine.name == "kokoro" and tempo != 1.0:
        tempo /= speed or 1.0

    pieces: list[np.ndarray] = []
    for j, (chunk, kind) in enumerate(chunks):
        if on_chunk is not None:
            on_chunk(j, len(chunks))
        audio = _trim_silence(engine.synthesize(chunk, voice, native_speed), sr)
        if not audio.size:
            continue
        pieces.append(audio)
        gap = g_title if (j == 0 and announces_title) else gaps.get(kind, gaps[SENTENCE])
        pieces.append(np.zeros(int(sr * gap), dtype=np.float32))

    if not pieces:
        return np.zeros(0, dtype=np.float32)
    full = np.concatenate(pieces)
    if abs(tempo - 1.0) > 0.01:
        # Stretch the assembled chapter, not each chunk: the gaps scale with the
        # speech, so pacing stays proportional.
        full = apply_filters(full, sr, atempo_chain(tempo))
    return full


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
    chapter_voices: dict[int, str] | None = None,
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
            ChapterState(
                index=i,
                title=normalize_chapter_title(ch.title),
                voice=(chapter_voices or {}).get(i, ""),
            )
            for i, ch in enumerate(book.chapters)
        ],
    )
    store.save(manifest)
    return manifest


def run_job(
    job_id: str,
    config: Config,
    store: JobStore | None = None,
    on_progress: ProgressFn | None = None,
    cancel_event: threading.Event | None = None,
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
            # Announce the chapter title before its body
            title = normalize_chapter_title(chapter.title)
            chunks = chunk_paragraphs(text)
            if title:
                chunks = [(title + ".", PARAGRAPH)] + chunks
            state.n_chunks = len(chunks)

            def progress(j: int, total: int, _i=i, _title=title):
                check_cancel()
                notify(
                    Progress(job_id, "synthesizing", _i, n_chapters, j, total, _title)
                )

            full = render_chunks(
                engine, chunks, state.voice or manifest.voice, manifest.speed,
                config=config, announces_title=bool(title), on_chunk=progress,
            )
            if not full.size:
                full = np.zeros(1, dtype=np.float32)
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
            master=getattr(config, "master_audio", True),
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

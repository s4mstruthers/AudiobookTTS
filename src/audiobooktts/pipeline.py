"""Conversion pipeline: epub -> per-chapter audio -> packaged m4b.

Progress flows through a callback so the CLI progress bar and the web UI's
event stream share one code path. Chapters are checkpointed: a finished
chapter's FLAC plus its manifest entry mean resume never re-synthesizes it.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from audiobooktts import pacing
from audiobooktts.audio import as_mono_float32, silence, stretch_internal_pauses, trim_silence
from audiobooktts.config import Config
from audiobooktts.engines import TTSEngine, get_engine
from audiobooktts.epub import Book, parse_epub
from audiobooktts.fsutil import replace_file
from audiobooktts.mastering import (
    MasteringError,
    apply_filters,
    atempo_chain,
    filter_chain,
    time_stretch_file,
)
from audiobooktts.packager import package_m4b
from audiobooktts.store import ChapterState, JobStore, Manifest, file_sha256
from audiobooktts.textproc import (
    CLAUSE,
    PARAGRAPH,
    SENTENCE,
    chunk_paragraphs,
    clean_text,
    normalize_chapter_title,
    preview_chunks,
    preview_sample,
    split_sentences,
)

logger = logging.getLogger(__name__)

Chunks = list[tuple[str, str]]
# Tempo differences smaller than this are inaudible; skip the stretch.
_TEMPO_EPSILON = 0.01


@dataclass
class Progress:
    job_id: str
    # parsing | loading | calibrating | synthesizing | packaging | done | failed
    stage: str
    chapter_index: int = 0
    n_chapters: int = 0
    chunk_index: int = 0
    n_chunks: int = 0
    chapter_title: str = ""
    message: str = ""


ProgressFn = Callable[[Progress], None]


class JobCancelledError(Exception):
    """The job was cancelled through its cancel event."""


def chapter_chunks(title: str, text: str, unit: str = "paragraph") -> Chunks:
    """The chunks for one chapter: its announced title, then its body."""
    chunks = chunk_paragraphs(clean_text(text), unit=unit)
    title = normalize_chapter_title(title)
    return [(title + ".", PARAGRAPH), *chunks] if title else chunks


def _segments(
    engine: TTSEngine,
    chunks: Chunks,
    voice: str,
    config: Config,
    announces_title: bool,
    on_chunk: Callable[[int, int], None] | None,
) -> Iterator[np.ndarray]:
    """Speech and the configured pause after it, chunk by chunk."""
    sr = engine.sample_rate
    gaps = {
        CLAUSE: config.gap_clause,
        SENTENCE: config.gap_sentence,
        PARAGRAPH: config.gap_paragraph,
    }
    # Only when asked: the engine's own spacing between sentences reads
    # naturally, and overriding it makes the narration laboured.
    stretch = config.stretch_sentence_pauses and config.synthesis_unit == "paragraph"
    total = len(chunks)
    for j, (chunk, kind) in enumerate(chunks):
        if on_chunk is not None:
            on_chunk(j, total)
        audio = trim_silence(as_mono_float32(engine.synthesize(chunk, voice)), sr)
        if not audio.size:
            continue
        if stretch:
            audio = stretch_internal_pauses(
                audio, sr, gaps[SENTENCE], max_gaps=len(split_sentences(chunk)) - 1
            )
        yield audio
        gap = config.gap_title if (j == 0 and announces_title) else gaps.get(kind, gaps[SENTENCE])
        yield silence(gap, sr)
    if on_chunk is not None:
        on_chunk(total, total)


def render_chunks(
    engine: TTSEngine,
    chunks: Chunks,
    voice: str,
    config: Config | None = None,
    *,
    tempo: float = 1.0,
    announces_title: bool = False,
    on_chunk: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Synthesize chunks in memory and join them with the configured pauses.

    Used for previews, so that what you audition is what you get.
    """
    config = config or Config()
    pieces = list(_segments(engine, chunks, voice, config, announces_title, on_chunk))
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    full = np.concatenate(pieces)
    if abs(tempo - 1.0) > _TEMPO_EPSILON:
        # Stretch the assembled audio, not each chunk: the gaps scale with the
        # speech, so pacing stays proportional.
        full = apply_filters(full, engine.sample_rate, atempo_chain(tempo))
    return full


def render_chapter(
    engine: TTSEngine,
    chunks: Chunks,
    voice: str,
    config: Config,
    dest: Path,
    *,
    tempo: float = 1.0,
    announces_title: bool = False,
    on_chunk: Callable[[int, int], None] | None = None,
) -> float:
    """Render a chapter into the FLAC file ``dest``; returns its duration in seconds.

    Audio streams to disk as it is synthesised rather than accumulating in
    memory, which for an hour-long chapter would be ~350 MB of samples before
    the copies made while stretching it.
    """
    sr = engine.sample_rate
    stretch = abs(tempo - 1.0) > _TEMPO_EPSILON
    tmp_flac = dest.with_name(dest.stem + ".tmp.flac")
    # Before stretching, keep full float precision in a scratch WAV.
    first = dest.with_name(dest.stem + ".raw.wav") if stretch else tmp_flac
    fmt, subtype = ("WAV", "FLOAT") if stretch else ("FLAC", "PCM_16")

    written = 0
    try:
        with sf.SoundFile(
            first, "w", samplerate=sr, channels=1, format=fmt, subtype=subtype
        ) as out:
            for segment in _segments(engine, chunks, voice, config, announces_title, on_chunk):
                out.write(segment)
                written += segment.size
            if not written:
                out.write(silence(0.5, sr))  # keep the chapter marker valid
        if stretch:
            try:
                time_stretch_file(first, tmp_flac, tempo)
            except MasteringError:
                logger.warning("Pace correction failed for %s; keeping the natural pace", dest.name)
                with sf.SoundFile(
                    tmp_flac, "w", samplerate=sr, channels=1, format="FLAC", subtype="PCM_16"
                ) as out:
                    for block in sf.blocks(str(first), blocksize=sr * 60, dtype="float32"):
                        out.write(block)
        replace_file(tmp_flac, dest)
    finally:
        tmp_flac.unlink(missing_ok=True)
        if stretch:
            first.unlink(missing_ok=True)
    return float(sf.info(str(dest)).duration)


def render_preview(
    engine: TTSEngine,
    book: Book,
    chapter_index: int,
    voice: str,
    config: Config,
    speed: float = 1.0,
    max_chars: int = 500,
) -> tuple[np.ndarray, str]:
    """A short sample of one chapter, rendered and mastered like the real thing.

    Returns the audio and the excerpt, in the book's own spelling for display.
    """
    if not book.chapters:
        raise ValueError("This book has no chapters")
    chapter = book.chapters[max(0, min(chapter_index, len(book.chapters) - 1))]
    sample = preview_sample(clean_text(chapter.text, respell=False), max_chars)
    chunks = preview_chunks(clean_text(chapter.text), max_chars, unit=config.synthesis_unit)
    chunks = chunks or [(sample, PARAGRAPH)]

    engine.load()
    tempo = pacing.tempo_for(engine, voice, config.target_wpm, speed)
    audio = render_chunks(engine, chunks, voice, config, tempo=tempo)
    if config.master_audio and audio.size:
        audio = apply_filters(audio, engine.sample_rate, filter_chain())
    return audio, sample


def create_job(
    epub_path: str | Path,
    config: Config,
    store: JobStore | None = None,
    chapter_voices: dict[int, str] | None = None,
    cover_path: str | Path | None = None,
) -> Manifest:
    """Register a conversion job without rendering anything yet."""
    store = store or JobStore()
    book = parse_epub(epub_path)
    if not book.chapters:
        raise ValueError("No readable chapters found in this epub")
    chapter_voices = chapter_voices or {}
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
                voice=chapter_voices.get(i, ""),
            )
            for i, ch in enumerate(book.chapters)
        ],
    )
    if cover_path:
        # The job keeps its own copy, so replacing or deleting the uploaded
        # image later cannot change a book that is still being rendered.
        src = Path(cover_path)
        if not src.is_file():
            raise FileNotFoundError(f"Cover image not found: {src}")
        dest = store.job_dir(manifest.job_id) / f"cover{src.suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        manifest.cover_path = str(dest)
    store.save(manifest)
    return manifest


def run_job(
    job_id: str,
    config: Config,
    store: JobStore | None = None,
    on_progress: ProgressFn | None = None,
    cancel_event: threading.Event | None = None,
) -> Manifest:
    """Render every unfinished chapter, then package the book.

    Raises JobCancelledError if ``cancel_event`` is set, and re-raises any
    failure after recording it in the manifest.
    """
    store = store or JobStore()
    manifest = store.load(job_id)
    notify = on_progress or (lambda p: None)

    def check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            manifest.status = "cancelled"
            manifest.pid = 0
            store.save(manifest)
            raise JobCancelledError(job_id)

    try:
        notify(Progress(job_id, "parsing", message="Reading the epub"))
        if not Path(manifest.epub_path).is_file():
            raise FileNotFoundError(f"Source epub is missing: {manifest.epub_path}")
        if file_sha256(manifest.epub_path) != manifest.epub_sha256:
            raise RuntimeError("Source epub changed since the job was created")
        book: Book = parse_epub(manifest.epub_path)
        if len(book.chapters) != len(manifest.chapters):
            raise RuntimeError("Chapter count mismatch on resume")

        manifest.status = "running"
        manifest.error = ""
        manifest.pid = os.getpid()
        store.save(manifest)
        n_chapters = len(book.chapters)

        pending = [
            i
            for i, state in enumerate(manifest.chapters)
            if not (state.done and store.chapter_audio_path(job_id, i).exists())
        ]
        voices = sorted({manifest.chapters[i].voice or manifest.voice for i in pending})
        # Fail in seconds, not hours in, if a voice has gone missing.
        from audiobooktts.voices import reference_path

        for voice in voices:
            reference_path(voice)

        engine = get_engine(manifest.engine)
        if pending:
            check_cancel()
            notify(
                Progress(
                    job_id, "loading", n_chapters=n_chapters, message="Loading the voice model"
                )
            )
            engine.load()

        for voice in voices:
            if voice in manifest.tempo_by_voice:
                continue
            check_cancel()
            if config.target_wpm > 0 and not pacing.is_calibrated(engine, voice):
                notify(
                    Progress(
                        job_id,
                        "calibrating",
                        n_chapters=n_chapters,
                        message=f"Measuring the pace of voice '{voice or 'default'}' (one-off)",
                    )
                )
            manifest.tempo_by_voice[voice] = round(
                pacing.tempo_for(engine, voice, config.target_wpm, manifest.speed), 4
            )
            store.save(manifest)

        for i in pending:
            check_cancel()
            chapter, state = book.chapters[i], manifest.chapters[i]
            voice = state.voice or manifest.voice
            chunks = chapter_chunks(chapter.title, chapter.text, unit=config.synthesis_unit)
            state.n_chunks = len(chunks)
            title = normalize_chapter_title(chapter.title)

            def progress(j: int, total: int, _i=i, _title=title) -> None:
                check_cancel()
                notify(Progress(job_id, "synthesizing", _i, n_chapters, j, total, _title))

            state.duration_s = render_chapter(
                engine,
                chunks,
                voice,
                config,
                store.chapter_audio_path(job_id, i),
                tempo=manifest.tempo_by_voice.get(voice, 1.0),
                announces_title=bool(title),
                on_chunk=progress,
            )
            state.done = True
            store.save(manifest)

        notify(Progress(job_id, "packaging", n_chapters, n_chapters, message="Packaging the m4b"))
        output = package_m4b(
            book=book,
            manifest=manifest,
            store=store,
            out_dir=Path(manifest.output_dir or config.output_dir).expanduser(),
            bitrate=config.bitrate,
            master=config.master_audio,
        )
        manifest.output_path = str(output)
        manifest.status = "done"
        manifest.pid = 0
        store.save(manifest)
        notify(Progress(job_id, "done", n_chapters, n_chapters, message=str(output)))
        return manifest
    except JobCancelledError:
        raise
    except KeyboardInterrupt:
        manifest.status = "interrupted"
        manifest.error = "Interrupted"
        manifest.pid = 0
        store.save(manifest)
        raise
    except Exception as e:
        logger.debug("Job %s failed", job_id, exc_info=True)
        manifest.status = "failed"
        manifest.error = str(e) or type(e).__name__
        manifest.pid = 0
        store.save(manifest)
        notify(Progress(job_id, "failed", message=manifest.error))
        raise

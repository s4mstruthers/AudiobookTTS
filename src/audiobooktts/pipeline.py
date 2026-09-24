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
from audiobooktts.packager import package_m4b
from audiobooktts.store import ChapterState, JobStore, Manifest, file_sha256
from audiobooktts.textproc import (
    CLAUSE,
    PARAGRAPH,
    SENTENCE,
    chunk_paragraphs,
    clean_text,
    normalize_chapter_title,
    split_sentences,
)

# Fallback pause lengths; Config overrides these. The engine pads each
# utterance with ~0.3s leading and ~0.5s trailing silence, which is trimmed
# off, so these values alone set the pacing.
GAP_CLAUSE_S = 0.25     # mid-sentence, where a long sentence had to be split
GAP_SENTENCE_S = 0.3    # only at a join within a split paragraph
GAP_PARAGRAPH_S = 1.9   # between paragraphs
GAP_TITLE_S = 2.6       # after the announced chapter title

_SILENCE_THRESHOLD = 0.01
# Keep a good slice of the natural decay either side. Trimming hard against the
# speech and splicing in pure digital silence is what makes narration sound
# breathless — the voice stops dead instead of releasing.
_SILENCE_KEEP_S = 0.14


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


# The engine leaves roughly 0.13s between sentences inside one utterance, and
# less than that at commas. Anything at or above this is treated as a sentence
# break worth lengthening; shorter dips are left alone.
_INTERNAL_PAUSE_MIN_S = 0.09


def stretch_internal_pauses(
    audio: np.ndarray, sample_rate: int, target_s: float, max_gaps: int
) -> np.ndarray:
    """Lengthen the gaps the engine placed between sentences — those only.

    Reading a paragraph in one call keeps its natural flow, but the engine's
    own sentence gaps are far too brisk for narration and cannot be
    configured. The silences are therefore found and padded afterwards.

    max_gaps is what makes this safe. Speech is full of quiet moments — soft
    consonants, breaths, the dip between words — and padding those produces
    audio that seems to cut out mid-phrase: one paragraph of two sentences
    offered fifteen candidate silences. Only the max_gaps longest are
    stretched, because a full stop is held longer than anything inside a
    sentence, so with one gap per full stop the rest are left alone.
    """
    if audio.size == 0 or target_s <= 0 or max_gaps <= 0:
        return audio
    quiet = np.abs(audio) <= _SILENCE_THRESHOLD
    if not quiet.any():
        return audio

    # Runs of silence, as (start, end) index pairs.
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    bounds = np.concatenate(([0], edges + 1, [audio.size]))
    min_len = int(sample_rate * _INTERNAL_PAUSE_MIN_S)
    target_len = int(sample_rate * target_s)

    runs = [
        (start, end)
        for start, end in zip(bounds[:-1], bounds[1:])
        if quiet[start] and start > 0 and end < audio.size
        and (end - start) >= min_len
    ]
    # The longest silences are the sentence ends.
    chosen = {
        start for start, _ in
        sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:max_gaps]
    }

    pieces: list[np.ndarray] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        segment = audio[start:end]
        pieces.append(segment)
        if start in chosen and target_len > segment.size:
            pieces.append(np.zeros(target_len - segment.size, dtype=np.float32))
    return np.concatenate(pieces) if pieces else audio


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

    # Chatterbox has no speed parameter, so both the user's speed and the pace
    # correction are applied as one time-stretch.
    tempo = tempo_for(engine, voice, getattr(cfg, "target_wpm", 0.0), speed)
    native_speed = 1.0

    pieces: list[np.ndarray] = []
    for j, (chunk, kind) in enumerate(chunks):
        if on_chunk is not None:
            on_chunk(j, len(chunks))
        audio = _trim_silence(engine.synthesize(chunk, voice, native_speed), sr)
        if not audio.size:
            continue
        # Only when asked: the engine's own spacing between sentences reads
        # naturally, and overriding it makes the narration laboured.
        if (
            getattr(cfg, "stretch_sentence_pauses", False)
            and getattr(cfg, "synthesis_unit", "paragraph") == "paragraph"
        ):
            audio = stretch_internal_pauses(
                audio, sr, gaps[SENTENCE], max_gaps=len(split_sentences(chunk)) - 1
            )
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
            chunks = chunk_paragraphs(
                text, unit=getattr(config, "synthesis_unit", "paragraph")
            )
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

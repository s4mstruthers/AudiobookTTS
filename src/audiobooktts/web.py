"""Local FastAPI web UI: upload an epub, preview voices, run and watch a job.

The server is meant for one person on their own machine. By default it only
listens on 127.0.0.1 and rejects requests addressed to any other host name,
which defeats DNS-rebinding attacks from web pages open in the same browser;
state-changing requests from another origin are refused as well.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import shutil
import string
import sys
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlsplit

import soundfile as sf
from fastapi import APIRouter, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from audiobooktts import __version__, config
from audiobooktts.config import Config
from audiobooktts.engines import ENGINE_NAMES, EngineUnavailableError, get_engine
from audiobooktts.epub import Book, EpubError, parse_epub
from audiobooktts.ffmpeg import FFmpegNotFoundError
from audiobooktts.fsutil import is_within, replace_file, safe_filename
from audiobooktts.pipeline import JobCancelledError, Progress, create_job, render_preview, run_job
from audiobooktts.store import JobNotFoundError, JobStore, pid_alive
from audiobooktts.voices import VoiceClipError

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
COVER_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
LOOPBACK_HOSTS = ["127.0.0.1", "localhost", "[::1]", "::1"]
# How often an event stream checks for new progress.
_EVENT_POLL_S = 0.25
_DRIVES = "::drives"  # pseudo-path listing the drives on Windows

store = JobStore()


# ---------------------------------------------------------------------------
# Background runs
# ---------------------------------------------------------------------------


class JobRunner:
    """One job rendering on a background thread, observable by any number of clients.

    Progress is kept as "latest state" rather than a queue of events: the UI
    only ever shows the latest, and a queue can be consumed by one listener
    only, so a second tab or a reconnect used to steal events and hang.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._state: dict = {"stage": "starting", "message": "Starting"}
        self._version = 0
        self.finished = False

    def publish(self, state: dict) -> None:
        with self._lock:
            self._state = state
            self._version += 1

    def snapshot(self) -> tuple[int, dict, bool]:
        with self._lock:
            return self._version, dict(self._state), self.finished


_runners: dict[str, JobRunner] = {}
_runners_lock = threading.Lock()


def _live_runner(job_id: str) -> JobRunner | None:
    with _runners_lock:
        return _runners.get(job_id)


def _progress_dict(p: Progress) -> dict:
    return {
        "stage": p.stage,
        "chapter_index": p.chapter_index,
        "n_chapters": p.n_chapters,
        "chunk_index": p.chunk_index,
        "n_chunks": p.n_chunks,
        "chapter_title": p.chapter_title,
        "message": p.message,
    }


def _start_runner(job_id: str, cfg: Config) -> JobRunner:
    with _runners_lock:
        if job_id in _runners:
            raise HTTPException(409, "This job is already running")
        runner = JobRunner(job_id)
        _runners[job_id] = runner

    def work() -> None:
        try:
            run_job(job_id, cfg, store, lambda p: runner.publish(_progress_dict(p)), runner.cancel)
        except JobCancelledError:
            pass
        except Exception:
            logger.exception("Job %s failed", job_id)  # recorded in its manifest
        finally:
            with runner._lock:
                runner.finished = True
                runner._version += 1
            # Finished jobs are no longer live: they can be downloaded,
            # resumed or deleted, and must stop showing Watch/Cancel.
            with _runners_lock:
                _runners.pop(job_id, None)

    threading.Thread(target=work, name=f"job-{job_id}", daemon=True).start()
    return runner


def _final_state(job_id: str) -> dict:
    m = store.load(job_id)
    return {"stage": m.status, "message": m.output_path if m.status == "done" else m.error}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_manifest(job_id: str):
    try:
        return store.load(job_id)
    except JobNotFoundError:
        raise HTTPException(404, "No such job") from None


@lru_cache(maxsize=4)
def _parse_cached(path: str, _mtime_ns: int, _size: int) -> Book:
    return parse_epub(path)


def _book(path: str | Path) -> Book:
    """Parse an uploaded epub, reusing the result while the file is unchanged.

    The preview, the cover and the name scanner each need the parsed book, and
    parsing a long epub takes seconds.
    """
    p = Path(path)
    if not p.is_file() or not is_within(p, config.UPLOAD_DIR):
        raise HTTPException(400, "Unknown book: upload it first")
    st = p.stat()
    try:
        return _parse_cached(str(p.resolve()), st.st_mtime_ns, st.st_size)
    except EpubError as e:
        raise HTTPException(400, str(e)) from None


def _engine(name: str):
    if name not in ENGINE_NAMES:
        raise HTTPException(400, f"Unknown engine {name!r}")
    return get_engine(name)


def _wav_response(audio, sample_rate: int, headers: dict | None = None) -> Response:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav", headers=headers)


def _clip_or_404(name: str) -> Path:
    from audiobooktts.voices import clip_paths

    clip = clip_paths().get(name)
    if clip is None:
        raise HTTPException(404, "No such voice")
    return clip


def _source_or_404(name: str) -> Path:
    from audiobooktts.voices import source_path

    src = source_path(name)
    if src is None:
        raise HTTPException(404, "No stored recording for this voice — upload it again")
    return src


async def _save_upload(upload: UploadFile, directory: Path, suffix: str) -> tuple[Path, str]:
    """Stream an upload to a temporary file in ``directory``; returns (path, sha256)."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    tmp = directory / f".upload-{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}{suffix}"
    with open(tmp, "wb") as f:
        while chunk := await upload.read(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    return tmp, digest.hexdigest()


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class StartJobRequest(BaseModel):
    path: str
    engine: str = "chatterbox"
    voice: str = ""
    speed: float = Field(1.0, ge=0.5, le=2.0)
    output_dir: str | None = None
    chapter_voices: dict[int, str] = Field(default_factory=dict)
    cover_id: str | None = None


class PronunciationEntry(BaseModel):
    word: str = Field(min_length=1, max_length=100)
    say_as: str = Field(min_length=1, max_length=200)


class RecutRequest(BaseModel):
    start: float = Field(45.0, ge=0)
    duration: float = Field(15.0, gt=0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")


@router.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@router.get("/api/voices")
def voices(engine: str = "chatterbox"):
    e = _engine(engine)
    return {"engine": e.name, "voices": [v.__dict__ for v in e.list_voices()]}


@router.get("/api/config")
def get_config():
    cfg = Config.load()
    return {
        "engine": cfg.engine,
        "voice": cfg.voice_for(),
        "speed": cfg.speed,
        "bitrate": cfg.bitrate,
        "output_dir": cfg.output_dir,
    }


# --- voice library ---------------------------------------------------------


@router.get("/api/voice-library")
def voice_library():
    """Saved reference clips, with duration and measured pace."""
    from audiobooktts.voices import list_clips

    return {"voices": list_clips(), "windows": get_engine("chatterbox").conditioning_windows()}


@router.get("/api/voice-library/{name}/audio")
def voice_audio(name: str):
    return FileResponse(_clip_or_404(name))


@router.post("/api/voice-library")
async def create_voice(
    file: UploadFile,
    name: str = Form(...),
    start: float = Form(45.0),
    duration: float = Form(15.0),
):
    """Build a reference clip from an uploaded recording in one step."""
    from audiobooktts.voices import add_voice, sanitize_name

    try:
        safe = sanitize_name(name)
    except VoiceClipError as e:
        raise HTTPException(400, str(e)) from None
    tmp, _ = await _save_upload(
        file, config.VOICE_SOURCES_DIR, Path(file.filename or "").suffix or ".audio"
    )
    try:
        await asyncio.to_thread(add_voice, tmp, safe, start, duration)
    except VoiceClipError as e:
        raise HTTPException(400, str(e)) from None
    finally:
        tmp.unlink(missing_ok=True)
    return {"name": safe}


@router.post("/api/voice-library/{name}/recut")
def recut_voice(name: str, body: RecutRequest):
    """Re-cut an existing voice from a different point in its source."""
    from audiobooktts.voices import add_voice

    src = _source_or_404(name)
    try:
        add_voice(src, name, start=body.start, duration=body.duration, keep_source=False)
    except VoiceClipError as e:
        raise HTTPException(400, str(e)) from None
    return {"name": name, "recut": True}


@router.post("/api/voice-library/{name}/recalibrate")
def recalibrate_voice(name: str):
    from audiobooktts.pacing import forget, measure_wpm

    _clip_or_404(name)
    forget(name)
    engine = get_engine("chatterbox")
    engine.load()
    return {"name": name, "wpm": round(measure_wpm(engine, name), 1)}


@router.delete("/api/voice-library/{name}")
def remove_voice(name: str):
    from audiobooktts.voices import delete_voice

    if not delete_voice(name):
        raise HTTPException(404, "No such voice")
    return {"deleted": name}


@router.get("/api/voice-source/{name}/peaks")
def voice_source_peaks(name: str, buckets: int = 900):
    """Waveform envelope of the recording a voice was cut from."""
    from audiobooktts.voices import waveform_peaks

    src = _source_or_404(name)
    try:
        data = waveform_peaks(src, buckets=max(10, min(buckets, 4000)))
    except VoiceClipError as e:
        raise HTTPException(400, str(e)) from None
    return {**data, "name": name, "source": src.name}


@router.get("/api/voice-source/{name}/segment")
def voice_source_segment(name: str, start: float = 0.0, duration: float = 15.0):
    """A stretch of the source recording, to audition before committing."""
    from audiobooktts.voices import cut_segment

    try:
        return Response(cut_segment(_source_or_404(name), start, duration), media_type="audio/wav")
    except VoiceClipError as e:
        raise HTTPException(400, str(e)) from None


@router.post("/api/voice-source")
async def upload_voice_source(file: UploadFile, name: str = Form(...)):
    """Store a recording without cutting it yet, so the clip can be chosen by ear."""
    from audiobooktts.voices import sanitize_name, store_source, waveform_peaks

    try:
        safe = sanitize_name(name)
    except VoiceClipError as e:
        raise HTTPException(400, str(e)) from None
    tmp, _ = await _save_upload(
        file, config.VOICE_SOURCES_DIR, Path(file.filename or "").suffix or ".audio"
    )
    try:
        stored = store_source(tmp, safe)
        data = await asyncio.to_thread(waveform_peaks, stored)
    except VoiceClipError as e:
        raise HTTPException(400, str(e)) from None
    finally:
        tmp.unlink(missing_ok=True)
    return {**data, "name": safe}


# --- covers ----------------------------------------------------------------


@router.post("/api/cover")
async def upload_cover(file: UploadFile):
    """A custom cover image, embedded in place of the epub's own."""
    suffix = Path(file.filename or "").suffix.lower() or ".jpg"
    if suffix not in COVER_SUFFIXES:
        raise HTTPException(400, "Cover must be a jpg, png or webp image")
    tmp, digest = await _save_upload(file, config.COVER_DIR, suffix)
    # Named by content, so covers chosen for different jobs never overwrite
    # each other while those jobs are still rendering.
    cover_id = f"{digest[:20]}{suffix}"
    replace_file(tmp, config.COVER_DIR / cover_id)
    return {"id": cover_id, "url": f"/api/covers/{cover_id}"}


def _cover_path(cover_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{20}\.(jpg|jpeg|png|webp)", cover_id):
        raise HTTPException(404, "No such cover")
    path = config.COVER_DIR / cover_id
    if not path.is_file():
        raise HTTPException(404, "No such cover")
    return path


@router.get("/api/covers/{cover_id}")
def cover_file(cover_id: str):
    return FileResponse(_cover_path(cover_id))


@router.get("/api/cover")
def book_cover(path: str):
    book = _book(path)
    if not book.cover:
        raise HTTPException(404, "No cover")
    return Response(book.cover, media_type=book.cover_media_type or "image/jpeg")


# --- pronunciation ---------------------------------------------------------


@router.get("/api/pronunciation")
def get_pronunciation():
    from audiobooktts.pronunciation import load_lexicon

    lex = load_lexicon()
    return {"entries": [{"word": k, "say_as": lex[k]} for k in sorted(lex, key=str.lower)]}


@router.post("/api/pronunciation")
def set_pronunciation(entry: PronunciationEntry):
    from audiobooktts.pronunciation import add

    try:
        add(entry.word, entry.say_as)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"word": entry.word.strip(), "say_as": entry.say_as.strip()}


@router.delete("/api/pronunciation/{word}")
def delete_pronunciation(word: str):
    from audiobooktts.pronunciation import remove

    remove(word)
    return {"deleted": word}


@router.get("/api/pronunciation/try")
def try_pronunciation(word: str, voice: str, say_as: str = "", engine: str = "chatterbox"):
    """Hear one word — as written, or as a candidate respelling.

    Spoken inside a short carrier sentence, because a word alone gets an
    unnatural reading that tells you little about how it will sound in the book.
    """
    spoken = (say_as or word).strip()
    if not spoken:
        raise HTTPException(400, "Nothing to say")
    e = _engine(engine)
    audio = e.synthesize(f"They sailed on towards {spoken}, arriving at dusk.", voice)
    return _wav_response(audio, e.sample_rate)


@router.get("/api/pronunciation/suggest")
def suggest_pronunciation(path: str, limit: int = 40):
    from audiobooktts.pronunciation import suggest

    _book(path)  # validates the path
    return {
        "words": [{"word": w, "count": n} for w, n in suggest(path, limit=max(1, min(limit, 200)))]
    }


# --- folders ---------------------------------------------------------------


def _windows_drives() -> list[str]:
    if hasattr(os, "listdrives"):  # Python 3.12+
        return list(os.listdrives())
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]


@router.get("/api/browse")
def browse(path: str = ""):
    """List sub-folders so the UI can offer a folder picker.

    The browser's own directory picker yields an opaque handle, not a path the
    server can write to, so the listing has to come from this side.
    """
    home = Path.home().resolve()
    if sys.platform == "win32" and path == _DRIVES:
        return {
            "path": "This PC",
            "parent": None,
            "dirs": [{"name": d, "path": d} for d in _windows_drives()],
            "home": str(home),
        }
    base = Path(path).expanduser() if path else home
    try:
        base = base.resolve()
        if not base.is_dir():
            base = home
        dirs = sorted(
            (p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name.lower(),
        )
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {base}") from None
    parent = (
        str(base.parent) if base.parent != base else (_DRIVES if sys.platform == "win32" else None)
    )
    return {
        "path": str(base),
        "parent": parent,
        "dirs": [{"name": p.name, "path": str(p)} for p in dirs],
        "home": str(home),
    }


# --- books and previews ----------------------------------------------------


@router.post("/api/upload")
async def upload(file: UploadFile):
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".epub"):
        raise HTTPException(400, "Please upload a .epub file")
    tmp, digest = await _save_upload(file, config.UPLOAD_DIR, ".epub")
    # One folder per distinct file: re-uploading a different book with the
    # same name must not overwrite the source of an existing job.
    dest = config.UPLOAD_DIR / digest[:16] / safe_filename(filename, fallback="book.epub")
    dest.parent.mkdir(parents=True, exist_ok=True)
    replace_file(tmp, dest)
    try:
        book = _book(dest)
        if not book.chapters:
            raise HTTPException(400, "No readable chapters found in this epub")
    except HTTPException:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise
    return {
        "path": str(dest),
        "title": book.title,
        "author": book.author,
        "has_cover": book.cover is not None,
        "chapters": [
            {"index": i, "title": c.title, "chars": len(c.text)}
            for i, c in enumerate(book.chapters)
        ],
        "total_chars": sum(len(c.text) for c in book.chapters),
    }


@router.get("/api/preview")
def preview(
    path: str, voice: str, engine: str = "chatterbox", speed: float = 1.0, chapter: int = 0
):
    """Synthesize a short sample of this book's own text as a wav."""
    book = _book(path)
    e = _engine(engine)
    audio, sample = render_preview(e, book, chapter, voice, Config.load(), speed=speed)
    # Percent-encode: real book text carries newlines and non-Latin-1 characters,
    # which are illegal in an HTTP header and abort the response. The client
    # decodes with decodeURIComponent.
    return _wav_response(
        audio, e.sample_rate, headers={"X-Sample-Text": quote(sample[:600], safe="")}
    )


# --- jobs --------------------------------------------------------------------


@router.post("/api/jobs")
def start_job(body: StartJobRequest):
    epub = Path(body.path)
    _book(epub)  # validates it is an uploaded, readable epub
    engine = _engine(body.engine)
    cover = _cover_path(body.cover_id) if body.cover_id else None
    available = {v.id for v in engine.list_voices()}
    for v in {body.voice, *body.chapter_voices.values()}:
        if v and v not in available:
            raise HTTPException(400, f"Unknown voice {v!r}")

    cfg = Config.load()
    cfg.remember_voice(body.engine, body.voice)
    cfg.speed = body.speed
    if body.output_dir:
        cfg.output_dir = str(Path(body.output_dir).expanduser())
    cfg.save()  # the settings you just used become the defaults next time

    chapter_voices = {k: v for k, v in body.chapter_voices.items() if v}
    manifest = create_job(epub, cfg, store, chapter_voices=chapter_voices, cover_path=cover)
    _start_runner(manifest.job_id, cfg)
    return {"job_id": manifest.job_id, "n_chapters": len(manifest.chapters)}


@router.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    manifest = _load_manifest(job_id)
    if manifest.status == "running" and manifest.pid != os.getpid() and pid_alive(manifest.pid):
        raise HTTPException(409, f"This job is running in another process (pid {manifest.pid})")
    cfg = Config.load()
    cfg.engine, cfg.voice, cfg.speed = manifest.engine, manifest.voice, manifest.speed
    _start_runner(job_id, cfg)
    return {"job_id": job_id, "resumed": True}


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    runner = _live_runner(job_id)
    if runner is None:
        raise HTTPException(404, "Job is not running")
    runner.cancel.set()
    return {"cancelled": True}


@router.get("/api/jobs")
def list_jobs():
    with _runners_lock:
        live_ids = set(_runners)
    return {
        "jobs": [
            {
                "job_id": m.job_id,
                "title": m.book_title,
                "author": m.book_author,
                "status": m.status,
                "voice": m.voice,
                "done": m.chapters_done,
                "total": len(m.chapters),
                "output_path": m.output_path,
                "error": m.error,
                # Only a job running in *this* process can be cancelled or watched.
                "live": m.job_id in live_ids,
            }
            for m in store.list_jobs()
        ]
    }


@router.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    manifest = _load_manifest(job_id)
    if _live_runner(job_id) is not None or (
        manifest.status == "running" and pid_alive(manifest.pid)
    ):
        raise HTTPException(409, "Cancel the job before deleting it")
    store.delete(job_id)
    return {"deleted": job_id}


@router.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    """Server-sent events: the job's progress until it finishes.

    A job that is no longer running yields its final state straight away, so
    a client that attaches late still learns the outcome.
    """
    _load_manifest(job_id)
    runner = _live_runner(job_id)

    def event(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    async def stream():
        if runner is None:
            yield event(_final_state(job_id))
            return
        seen = -1
        last_stage = None
        while not await request.is_disconnected():
            version, state, finished = runner.snapshot()
            if finished:
                final = _final_state(job_id)
                if final["stage"] != last_stage:  # the runner may have sent it already
                    yield event(final)
                return
            if version != seen:
                seen, last_stage = version, state.get("stage")
                yield event(state)
            await asyncio.sleep(_EVENT_POLL_S)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    m = _load_manifest(job_id)
    if not m.output_path or not Path(m.output_path).is_file():
        raise HTTPException(404, "Audiobook not ready")
    return FileResponse(m.output_path, media_type="audio/mp4", filename=Path(m.output_path).name)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Anything still marked "running" by a process that has since died is
    # flipped to interrupted, so it becomes resumable instead of stuck.
    for job_id in store.reconcile_running():
        logger.info("Marked orphaned job %s as interrupted", job_id)
    yield


def create_app(restrict_hosts: bool = True) -> FastAPI:
    """Build the web application.

    ``restrict_hosts`` accepts only requests addressed to localhost; turn it
    off only when deliberately serving on another interface.
    """
    app = FastAPI(title="AudioBookTTS", version=__version__, lifespan=_lifespan)
    app.include_router(router)

    @app.middleware("http")
    async def refuse_cross_origin_writes(request: Request, call_next):
        # A page on another site can make the browser send simple POSTs here
        # (form uploads, cancel). Browsers always label them with an Origin.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if (
                origin
                and origin != "null"
                and urlsplit(origin).netloc != request.headers.get("host")
            ):
                return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(EngineUnavailableError)
    @app.exception_handler(FFmpegNotFoundError)
    async def _unavailable(_request: Request, exc: Exception):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(VoiceClipError)
    @app.exception_handler(JobNotFoundError)
    async def _bad_request(_request: Request, exc: Exception):
        status = 404 if isinstance(exc, JobNotFoundError) else 400
        return JSONResponse({"detail": str(exc).strip("'")}, status_code=status)

    if restrict_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOOPBACK_HOSTS)
    return app


#: Default application, for `uvicorn audiobooktts.web:app`.
app = create_app()

"""Local FastAPI web UI: upload an epub, preview voices, run and watch a job."""

from __future__ import annotations

import asyncio
import io
import json
import queue
import shutil
import threading
from pathlib import Path
from urllib.parse import quote

import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from audiobooktts.config import APP_DIR, Config
from audiobooktts.engines import get_engine
from audiobooktts.epub import parse_epub
from audiobooktts.mastering import apply_filters, filter_chain
from audiobooktts.pipeline import (
    CancelledError,
    create_job,
    render_chunks,
    run_job,
)
from audiobooktts.store import JobStore
from audiobooktts.textproc import clean_text, preview_chunks, preview_sample

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = APP_DIR / "uploads"

app = FastAPI(title="AudioBookTTS")
store = JobStore()

# job_id -> live run state
_runners: dict[str, dict] = {}
_lock = threading.Lock()

# Runs live in this process's memory, so anything still marked "running" at
# startup died with a previous server. Flip it to interrupted so it becomes
# resumable instead of being stuck with no available action.
store.reconcile_running()


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/api/voices")
def voices(engine: str = "kokoro"):
    e = get_engine(engine)
    return {"engine": e.name, "voices": [v.__dict__ for v in e.list_voices()]}


@app.get("/api/config")
def get_config():
    cfg = Config.load()
    return {
        "engine": cfg.engine,
        "voice": cfg.voice_for(),
        "speed": cfg.speed,
        "bitrate": cfg.bitrate,
        "output_dir": cfg.output_dir,
    }


@app.get("/api/browse")
def browse(path: str = ""):
    """List sub-directories so the UI can offer a folder picker.

    The browser's own directory picker yields an opaque handle, not a path the
    server can write to, so the listing has to come from this side.
    """
    base = Path(path).expanduser() if path else Path.home()
    try:
        base = base.resolve()
        if not base.is_dir():
            base = Path.home().resolve()
        dirs = sorted(
            (p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name.lower(),
        )
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {base}")
    return {
        "path": str(base),
        "parent": str(base.parent) if base.parent != base else None,
        "dirs": [{"name": p.name, "path": str(p)} for p in dirs],
        "home": str(Path.home()),
    }


@app.post("/api/upload")
async def upload(file: UploadFile):
    if not (file.filename or "").lower().endswith(".epub"):
        raise HTTPException(400, "Please upload a .epub file")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / Path(file.filename).name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        book = parse_epub(dest)
    except Exception as e:
        raise HTTPException(400, f"Could not parse epub: {e}")
    if not book.chapters:
        raise HTTPException(400, "No readable chapters found in this epub")
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


@app.get("/api/cover")
def cover(path: str):
    book = parse_epub(path)
    if not book.cover:
        raise HTTPException(404, "No cover")
    return Response(book.cover, media_type=book.cover_media_type or "image/jpeg")


@app.get("/api/preview")
def preview(path: str, voice: str, engine: str = "kokoro", speed: float = 1.0,
            chapter: int = 0):
    """Synthesize a short sample of this book's own text as a wav."""
    book = parse_epub(path)
    if not book.chapters:
        raise HTTPException(400, "No chapters")
    ch = book.chapters[min(chapter, len(book.chapters) - 1)]
    cfg = Config.load()
    text = clean_text(ch.text)
    sample = preview_sample(text)
    # Render through the same path as the real thing, so the pacing, pauses and
    # loudness you audition are the ones you get.
    chunks = preview_chunks(text) or [(sample, True)]

    e = get_engine(engine)
    audio = render_chunks(e, chunks, voice, speed, config=cfg)
    if getattr(cfg, "master_audio", True) and audio.size:
        audio = apply_filters(audio, e.sample_rate, filter_chain())
    buf = io.BytesIO()
    sf.write(buf, audio, e.sample_rate, format="WAV")
    buf.seek(0)
    # Percent-encode: real book text carries newlines and non-Latin-1 characters,
    # which are illegal in an HTTP header and abort the response. The client
    # decodes with decodeURIComponent.
    return Response(
        buf.read(),
        media_type="audio/wav",
        headers={"X-Sample-Text": quote(sample[:600], safe="")},
    )


def _start_runner(job_id: str, cfg: Config):
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def on_progress(p):
        q.put({
            "stage": p.stage,
            "chapter_index": p.chapter_index,
            "n_chapters": p.n_chapters,
            "chunk_index": p.chunk_index,
            "n_chunks": p.n_chunks,
            "chapter_title": p.chapter_title,
            "message": p.message,
        })

    def work():
        try:
            run_job(job_id, cfg, store, on_progress, cancel_event=cancel)
        except CancelledError:
            q.put({"stage": "cancelled", "message": "Cancelled"})
        except Exception as e:
            q.put({"stage": "failed", "message": str(e)})
        finally:
            q.put(None)  # sentinel: stream complete

    t = threading.Thread(target=work, daemon=True)
    with _lock:
        _runners[job_id] = {"queue": q, "cancel": cancel, "thread": t}
    t.start()


@app.post("/api/jobs")
def start_job(payload: dict):
    path = payload.get("path")
    if not path or not Path(path).exists():
        raise HTTPException(400, "Missing or unknown epub path")
    cfg = Config.load()
    cfg.remember_voice(
        payload.get("engine", cfg.engine), payload.get("voice", cfg.voice)
    )
    cfg.speed = float(payload.get("speed", cfg.speed))
    if payload.get("output_dir"):
        cfg.output_dir = payload["output_dir"]
    cfg.save()  # the settings you just used become the defaults next time

    raw = payload.get("chapter_voices") or {}
    chapter_voices = {int(k): v for k, v in raw.items() if v}
    manifest = create_job(path, cfg, store, chapter_voices=chapter_voices)
    _start_runner(manifest.job_id, cfg)
    return {"job_id": manifest.job_id, "n_chapters": len(manifest.chapters)}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    manifest = store.load(job_id)
    cfg = Config.load()
    cfg.engine, cfg.voice, cfg.speed = manifest.engine, manifest.voice, manifest.speed
    _start_runner(job_id, cfg)
    return {"job_id": job_id, "resumed": True}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with _lock:
        runner = _runners.get(job_id)
    if not runner:
        raise HTTPException(404, "Job is not running")
    runner["cancel"].set()
    return {"cancelled": True}


@app.get("/api/jobs")
def list_jobs():
    with _lock:
        live_ids = set(_runners)
    out = []
    for m in store.list_jobs():
        out.append({
            "job_id": m.job_id,
            "title": m.book_title,
            "author": m.book_author,
            "status": m.status,
            "voice": m.voice,
            "done": sum(1 for c in m.chapters if c.done),
            "total": len(m.chapters),
            "output_path": m.output_path,
            "error": m.error,
            # Only a job running in *this* process can be cancelled or watched.
            "live": m.job_id in live_ids,
        })
    return {"jobs": out}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with _lock:
        if job_id in _runners:
            raise HTTPException(409, "Cancel the job before deleting it")
    store.delete(job_id)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """Server-sent events stream of progress for a running job."""
    with _lock:
        runner = _runners.get(job_id)
    if not runner:
        raise HTTPException(404, "Job is not running")
    q: queue.Queue = runner["queue"]

    async def gen():
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                m = store.load(job_id)
                final = {"stage": m.status, "message": m.output_path or m.error}
                yield f"data: {json.dumps(final)}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    m = store.load(job_id)
    if not m.output_path or not Path(m.output_path).exists():
        raise HTTPException(404, "Audiobook not ready")
    return FileResponse(
        m.output_path,
        media_type="audio/mp4",
        filename=Path(m.output_path).name,
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

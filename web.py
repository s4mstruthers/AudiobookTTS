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
import re

from fastapi import FastAPI, Form, HTTPException, UploadFile
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
from audiobooktts.textproc import (
    PARAGRAPH,
    clean_text,
    preview_chunks,
    preview_sample,
)

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = APP_DIR / "uploads"
COVER_DIR = APP_DIR / "covers"
SOURCE_DIR = APP_DIR / "voice_sources"

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
def voices(engine: str = "chatterbox"):
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


@app.get("/api/voice-library")
def voice_library():
    """Saved reference clips, with duration and measured pace."""
    from audiobooktts.voices import list_voices

    from audiobooktts.voices import conditioning_windows

    return {"voices": list_voices(), "windows": conditioning_windows()}


@app.get("/api/voice-library/{name}/audio")
def voice_audio(name: str):
    from audiobooktts.engines.chatterbox import VOICES_DIR

    path = VOICES_DIR / f"{name}.wav"
    if not path.is_file():
        raise HTTPException(404, "No such voice")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/voice-source/{name}/peaks")
def voice_source_peaks(name: str, buckets: int = 900):
    """Waveform envelope of the recording a voice was cut from."""
    from audiobooktts.voices import VoiceClipError, source_path, waveform_peaks

    src = source_path(name)
    if src is None:
        raise HTTPException(404, "No stored recording for this voice")
    try:
        data = waveform_peaks(src, buckets=buckets)
    except VoiceClipError as e:
        raise HTTPException(400, str(e))
    data["name"] = name
    data["source"] = src.name
    return data


@app.get("/api/voice-source/{name}/segment")
def voice_source_segment(name: str, start: float = 0.0, duration: float = 15.0):
    """A stretch of the source recording, to audition before committing."""
    from audiobooktts.voices import VoiceClipError, cut_segment, source_path

    src = source_path(name)
    if src is None:
        raise HTTPException(404, "No stored recording for this voice")
    try:
        return Response(cut_segment(src, start, duration), media_type="audio/wav")
    except VoiceClipError as e:
        raise HTTPException(400, str(e))


@app.post("/api/voice-source")
async def upload_voice_source(file: UploadFile, name: str = Form(...)):
    """Store a recording without cutting it yet, so it can be edited first."""
    from audiobooktts.voices import VoiceClipError, store_source, waveform_peaks

    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not safe:
        raise HTTPException(400, "Please give the voice a name")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SOURCE_DIR / f"__upload{Path(file.filename or '').suffix or '.audio'}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        stored = store_source(tmp, safe)
        data = waveform_peaks(stored)
    except VoiceClipError as e:
        raise HTTPException(400, str(e))
    finally:
        tmp.unlink(missing_ok=True)
    data["name"] = safe
    return data


@app.post("/api/voice-library")
async def create_voice(
    file: UploadFile,
    name: str = Form(...),
    start: float = Form(45.0),
    duration: float = Form(15.0),
):
    """Build a reference clip from an uploaded recording."""
    from audiobooktts.voices import VoiceClipError, add_voice

    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not safe:
        raise HTTPException(400, "Please give the voice a name")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    # Keep the source so the clip can be re-cut from a different point later.
    src = SOURCE_DIR / f"{safe}{Path(file.filename or '').suffix or '.audio'}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        add_voice(src, safe, start=start, duration=duration)
    except VoiceClipError as e:
        raise HTTPException(400, str(e))
    return {"name": safe, "source": str(src)}


@app.post("/api/voice-library/{name}/recut")
def recut_voice(name: str, payload: dict):
    """Re-cut an existing voice from a different point in its source."""
    from audiobooktts.pacing import forget
    from audiobooktts.voices import VoiceClipError, add_voice

    matches = list(SOURCE_DIR.glob(f"{name}.*")) if SOURCE_DIR.is_dir() else []
    if not matches:
        raise HTTPException(
            404, "No stored source for this voice — re-upload the recording"
        )
    try:
        add_voice(
            matches[0], name,
            start=float(payload.get("start", 45.0)),
            duration=float(payload.get("duration", 15.0)),
        )
    except VoiceClipError as e:
        raise HTTPException(400, str(e))
    forget(name)  # the clip changed, so its measured pace is stale
    return {"name": name, "recut": True}


@app.post("/api/voice-library/{name}/recalibrate")
def recalibrate_voice(name: str):
    from audiobooktts.pacing import forget, measure_wpm

    forget(name)
    wpm = measure_wpm(get_engine("chatterbox"), name)
    return {"name": name, "wpm": round(wpm, 1)}


@app.delete("/api/voice-library/{name}")
def remove_voice(name: str):
    from audiobooktts.voices import delete_voice

    if not delete_voice(name):
        raise HTTPException(404, "No such voice")
    for stale in SOURCE_DIR.glob(f"{name}.*") if SOURCE_DIR.is_dir() else []:
        stale.unlink(missing_ok=True)
    return {"deleted": name}


@app.post("/api/cover")
async def upload_cover(file: UploadFile):
    """A custom cover image, embedded in place of the epub's own."""
    COVER_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(400, "Cover must be a jpg, png or webp image")
    dest = COVER_DIR / f"cover{suffix}"
    for old in COVER_DIR.glob("cover.*"):
        old.unlink(missing_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest)}


@app.get("/api/cover-file")
def cover_file(path: str):
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "No cover")
    return FileResponse(p)


@app.get("/api/pronunciation")
def get_pronunciation():
    from audiobooktts.pronunciation import load_lexicon

    lex = load_lexicon()
    return {"entries": [{"word": k, "say_as": lex[k]} for k in sorted(lex)]}


@app.post("/api/pronunciation")
def set_pronunciation(payload: dict):
    from audiobooktts.pronunciation import add

    word = (payload.get("word") or "").strip()
    say_as = (payload.get("say_as") or "").strip()
    if not word or not say_as:
        raise HTTPException(400, "Both the word and how to say it are needed")
    add(word, say_as)
    return {"word": word, "say_as": say_as}


@app.delete("/api/pronunciation/{word}")
def delete_pronunciation(word: str):
    from audiobooktts.pronunciation import remove

    remove(word)
    return {"deleted": word}


@app.get("/api/pronunciation/try")
def try_pronunciation(word: str, voice: str, say_as: str = "", engine: str = "chatterbox"):
    """Hear one word — as written, or as a candidate respelling.

    Spoken inside a short carrier sentence, because a word alone gets an
    unnatural reading that tells you little about how it will sound in the book.
    """
    spoken = (say_as or word).strip()
    if not spoken:
        raise HTTPException(400, "Nothing to say")
    e = get_engine(engine)
    audio = e.synthesize(f"They sailed on towards {spoken}, arriving at dusk.", voice)
    buf = io.BytesIO()
    sf.write(buf, audio, e.sample_rate, format="WAV")
    buf.seek(0)
    return Response(buf.read(), media_type="audio/wav")


@app.get("/api/pronunciation/suggest")
def suggest_pronunciation(path: str, limit: int = 40):
    from audiobooktts.pronunciation import suggest

    if not Path(path).is_file():
        raise HTTPException(400, "Unknown epub")
    return {"words": [{"word": w, "count": n} for w, n in suggest(path, limit=limit)]}


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
def preview(path: str, voice: str, engine: str = "chatterbox", speed: float = 1.0,
            chapter: int = 0):
    """Synthesize a short sample of this book's own text as a wav."""
    book = parse_epub(path)
    if not book.chapters:
        raise HTTPException(400, "No chapters")
    ch = book.chapters[min(chapter, len(book.chapters) - 1)]
    cfg = Config.load()
    text = clean_text(ch.text)
    # Show the book's wording, not the respellings fed to the engine.
    sample = preview_sample(clean_text(ch.text, respell=False))
    # Render through the same path as the real thing, so the pacing, pauses and
    # loudness you audition are the ones you get.
    chunks = preview_chunks(
        text, unit=getattr(cfg, "synthesis_unit", "paragraph")
    ) or [(sample, PARAGRAPH)]

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
    if payload.get("cover_path"):
        manifest.cover_path = payload["cover_path"]
        store.save(manifest)
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

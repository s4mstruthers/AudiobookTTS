"""abtts: convert EPUB e-books into chaptered, mastered M4B audiobooks."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, TextColumn, TimeElapsedColumn
from rich.progress import Progress as RichProgress
from rich.table import Table

from audiobooktts import __version__
from audiobooktts.config import APP_DIR, Config
from audiobooktts.store import JobStore

console = Console()
logger = logging.getLogger("audiobooktts")

DEFAULT_PORT = 8765
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _known_errors() -> tuple[type[BaseException], ...]:
    """Failures with a clear message of their own, shown without a traceback."""
    from audiobooktts.engines import EngineUnavailableError
    from audiobooktts.epub import EpubError
    from audiobooktts.ffmpeg import FFmpegNotFoundError
    from audiobooktts.mastering import MasteringError
    from audiobooktts.packager import PackagingError
    from audiobooktts.store import JobNotFoundError
    from audiobooktts.voices import VoiceClipError

    return (
        EngineUnavailableError,
        EpubError,
        FFmpegNotFoundError,
        MasteringError,
        PackagingError,
        JobNotFoundError,
        VoiceClipError,
        FileNotFoundError,
        ValueError,
    )


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    for attr in ("engine", "speed", "bitrate"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(cfg, attr, value)
    if getattr(args, "output_dir", None):
        cfg.output_dir = str(Path(args.output_dir).expanduser().resolve())
    # Resolve last: an explicit --voice wins, otherwise fall back to one the
    # selected engine actually has.
    cfg.voice = getattr(args, "voice", None) or cfg.voice_for(cfg.engine)
    return cfg


def _run_with_progress(job_id: str, cfg: Config) -> None:
    from audiobooktts.pipeline import run_job

    store = JobStore()
    with RichProgress(
        TextColumn("[bold blue]{task.fields[chapter]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
        TimeElapsedColumn(),
        console=console,
    ) as rp:
        task = rp.add_task("synth", total=1, chapter="starting…")

        def on_progress(p) -> None:
            if p.stage == "synthesizing":
                rp.update(
                    task,
                    total=max(p.n_chunks, 1),
                    completed=p.chunk_index,
                    chapter=f"[{p.chapter_index + 1}/{p.n_chapters}] {p.chapter_title[:40]}",
                )
            elif p.stage in ("parsing", "loading", "calibrating", "packaging"):
                rp.update(task, completed=0, chapter=p.message or p.stage)

        manifest = run_job(job_id, cfg, store, on_progress)
    console.print(f"[green bold]Done:[/] {manifest.output_path}")


def cmd_convert(args: argparse.Namespace) -> None:
    from audiobooktts.pipeline import create_job

    cfg = _apply_overrides(Config.load(), args)
    manifest = create_job(args.epub, cfg, JobStore(), cover_path=args.cover)
    console.print(
        f"[bold]{manifest.book_title}[/] by {manifest.book_author} — "
        f"{len(manifest.chapters)} chapters, voice [cyan]{cfg.voice}[/] "
        f"(job {manifest.job_id})"
    )
    _run_with_progress(manifest.job_id, cfg)


def cmd_resume(args: argparse.Namespace) -> None:
    from audiobooktts.store import pid_alive

    store = JobStore()
    store.reconcile_running()
    manifest = store.load(args.job_id)
    if manifest.status == "running" and pid_alive(manifest.pid):
        raise ValueError(f"Job {args.job_id} is already running (process {manifest.pid})")
    # The job's own engine, voice and speed apply, so every chapter matches.
    cfg = Config.load()
    cfg.engine, cfg.voice, cfg.speed = manifest.engine, manifest.voice, manifest.speed
    if args.bitrate:
        cfg.bitrate = args.bitrate
    console.print(
        f"Resuming [bold]{manifest.book_title}[/]: "
        f"{manifest.chapters_done}/{len(manifest.chapters)} chapters already finished"
    )
    _run_with_progress(args.job_id, cfg)


def cmd_voices(args: argparse.Namespace) -> None:
    from audiobooktts.engines import get_engine

    engine = get_engine(args.engine or Config.load().engine)
    table = Table(title=f"{engine.name} voices")
    table.add_column("voice", style="cyan")
    table.add_column("type")
    table.add_column("description")
    for v in engine.list_voices():
        table.add_row(v.id, v.kind, v.label)
    console.print(table)


def cmd_preview(args: argparse.Namespace) -> None:
    import soundfile as sf

    from audiobooktts.engines import get_engine
    from audiobooktts.epub import parse_epub
    from audiobooktts.pipeline import render_preview
    from audiobooktts.playback import play_wav

    cfg = _apply_overrides(Config.load(), args)
    book = parse_epub(args.epub)
    if not book.chapters:
        raise ValueError("No chapters found in this epub")
    index = max(0, min(args.chapter - 1, len(book.chapters) - 1))
    engine = get_engine(cfg.engine)
    console.print(
        f"Previewing voice [cyan]{cfg.voice}[/] on chapter {index + 1} "
        f"([dim]{book.chapters[index].title}[/])…"
    )
    audio, sample = render_preview(engine, book, index, cfg.voice, cfg, speed=cfg.speed)
    console.print(f"[dim]{sample[:160]}…[/]")

    if args.save:
        sf.write(args.save, audio, engine.sample_rate)
        console.print(f"[green]Saved[/] {args.save}")
        return
    fd, name = tempfile.mkstemp(suffix=".wav", prefix="abtts-preview-")
    os.close(fd)  # Windows cannot write a file that is still held open
    try:
        sf.write(name, audio, engine.sample_rate)
        console.print("[dim]Playing… (Ctrl-C to stop)[/]")
        if not play_wav(name):
            keep = Path.cwd() / "preview.wav"
            sf.write(keep, audio, engine.sample_rate)
            console.print(f"[yellow]No audio player found.[/] Saved the preview to {keep}")
    finally:
        Path(name).unlink(missing_ok=True)


def cmd_add_voice(args: argparse.Namespace) -> None:
    from audiobooktts.voices import add_voice

    dest = add_voice(args.source, args.name, args.start, args.duration)
    console.print(
        f"[green]Added voice[/] [cyan]{dest.stem}[/] → {dest}\n"
        f"Try it: abtts preview yourbook.epub --voice {dest.stem}"
    )


def cmd_remove_voice(args: argparse.Namespace) -> None:
    from audiobooktts.voices import delete_voice

    if not delete_voice(args.name):
        raise ValueError(f"No voice named {args.name!r}")
    console.print(f"Removed voice [cyan]{args.name}[/]")


def cmd_say(args: argparse.Namespace) -> None:
    from audiobooktts import pronunciation

    if args.suggest:
        rows = pronunciation.suggest(args.suggest)
        table = Table(title="Names the dictionary doesn't know (most frequent first)")
        table.add_column("word", style="cyan")
        table.add_column("times", justify="right")
        for w, n in rows:
            table.add_row(w, str(n))
        console.print(table)
        console.print('Add one with: [cyan]abtts say Kizhi "Kee-zhee"[/]')
        return
    if args.remove:
        pronunciation.remove(args.remove)
        console.print(f"Removed [cyan]{args.remove}[/]")
        return
    if args.word and args.say_as:
        pronunciation.add(args.word, args.say_as)
        console.print(f"[green]{args.word}[/] will be said as [cyan]{args.say_as}[/]")
        return
    if args.word:
        raise ValueError(f'Also give the respelling, e.g. abtts say {args.word} "..."')
    lexicon = pronunciation.load_lexicon()
    if not lexicon:
        console.print("No entries yet.")
        return
    table = Table(title="Pronunciation lexicon")
    table.add_column("word", style="cyan")
    table.add_column("said as")
    for k in sorted(lexicon, key=str.lower):
        table.add_row(k, lexicon[k])
    console.print(table)


def cmd_jobs(args: argparse.Namespace) -> None:
    store = JobStore()
    # Jobs left "running" by a crashed process are orphaned; surface them as
    # interrupted so they can be resumed.
    for job_id in store.reconcile_running():
        console.print(f"[yellow]Marked orphaned job as interrupted:[/] {job_id}")
    jobs = store.list_jobs()
    if not jobs:
        console.print("No jobs yet.")
        return
    table = Table(title="Jobs")
    for col in ("job id", "book", "status", "progress", "output"):
        table.add_column(col)
    for m in jobs:
        table.add_row(
            m.job_id,
            m.book_title,
            m.status,
            f"{m.chapters_done}/{len(m.chapters)}",
            m.output_path or m.error or "-",
        )
    console.print(table)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from audiobooktts.web import create_app

    loopback = args.host in _LOOPBACK_HOSTS
    # Host-header checking needs a plain host name; IPv6 literals are loopback
    # anyway when they are ::1.
    app = create_app(restrict_hosts=args.host in ("127.0.0.1", "localhost"))
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    console.print(f"[bold]AudioBookTTS[/] web UI: {url}   (Ctrl-C to stop)")
    if not loopback:
        console.print(
            "[yellow]Warning:[/] the web UI has no login. Anyone who can reach "
            f"{args.host}:{args.port} can use it and browse this computer's folders."
        )
    if args.open:
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Report everything that decides whether conversion can run here."""
    from importlib import metadata

    from audiobooktts import ffmpeg
    from audiobooktts.engines.chatterbox import install_hint, installed_backends, resolve_backend

    def pkg_version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    ok, bad = "[green]✓[/]", "[red]✗[/]"
    table = Table(title="AudioBookTTS environment", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Version", __version__)
    table.add_row("Python", f"{platform.python_version()} ({sys.executable})")
    table.add_row("Platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    table.add_row("Data folder", str(APP_DIR))

    problems = 0
    for tool in ("ffmpeg", "ffprobe"):
        try:
            table.add_row(tool, f"{ok} {ffmpeg.find_tool(tool)}")
        except ffmpeg.FFmpegNotFoundError:
            table.add_row(tool, f"{bad} not found — {ffmpeg.install_hint()}")
            problems += 1

    have = installed_backends()
    table.add_row(
        "mlx-audio", f"{ok} {pkg_version('mlx-audio')}" if have["mlx"] else "not installed"
    )
    table.add_row(
        "chatterbox-tts",
        f"{ok} {pkg_version('chatterbox-tts')}" if have["torch"] else "not installed",
    )
    cfg = Config.load()
    try:
        chosen = resolve_backend(cfg.chatterbox_backend)
        table.add_row("Backend", f"{ok} {chosen} (setting: {cfg.chatterbox_backend})")
    except Exception as e:
        table.add_row("Backend", f"{bad} {e}")
        problems += 1
        chosen = None

    if have["torch"] and chosen == "torch":
        import torch

        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None)
        mps_ok = bool(mps and mps.is_available())
        device = (
            "CUDA " + torch.cuda.get_device_name(0)
            if cuda
            else ("Apple MPS" if mps_ok else "CPU only")
        )
        table.add_row("PyTorch", f"{torch.__version__} — {device}")
        if not cuda and not mps_ok:
            table.add_row(
                "", "[yellow]CPU synthesis works but is slow; a GPU is strongly recommended[/]"
            )
    console.print(table)
    if problems:
        console.print(f"[red]{problems} problem(s) found.[/] " + ("" if chosen else install_hint()))
        sys.exit(1)
    console.print("[green]Ready to convert.[/]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abtts",
        description="Convert EPUB e-books into chaptered, mastered M4B audiobooks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show detailed logs and tracebacks"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add_voice_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--engine", default=None, help="TTS engine (default: from config)")
        p.add_argument("--voice", default=None, help="voice name or path to a reference clip")
        p.add_argument("--speed", type=float, default=None, help="speed multiplier, e.g. 1.1")

    p = sub.add_parser("convert", help="convert an epub to an m4b audiobook")
    p.add_argument("epub", help="the .epub file")
    add_voice_options(p)
    p.add_argument("--bitrate", default=None, help="AAC bitrate (default 80k)")
    p.add_argument("--output-dir", default=None, help="where to save the m4b")
    p.add_argument("--cover", default=None, help="cover image to use instead of the epub's own")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("resume", help="resume an interrupted or failed job")
    p.add_argument("job_id", help="id from `abtts jobs`")
    p.add_argument("--bitrate", default=None, help="AAC bitrate for the final encode")
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("jobs", help="list conversion jobs")
    p.set_defaults(fn=cmd_jobs)

    p = sub.add_parser("voices", help="list available voices")
    p.add_argument("--engine", default=None)
    p.set_defaults(fn=cmd_voices)

    p = sub.add_parser("preview", help="hear a voice read part of your book")
    p.add_argument("epub")
    p.add_argument("--chapter", type=int, default=1, help="chapter number to sample (default 1)")
    add_voice_options(p)
    p.add_argument(
        "--save", metavar="WAV", default=None, help="save the preview instead of playing it"
    )
    p.set_defaults(fn=cmd_preview)

    p = sub.add_parser("add-voice", help="make a cloning reference clip from any recording")
    p.add_argument("source", help="audio or video file (mp3, wav, m4b, …)")
    p.add_argument("--name", required=True, help="name to select the voice by")
    p.add_argument(
        "--start",
        type=float,
        default=45.0,
        help="seconds into the file to start (default skips LibriVox intros)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="clip length in seconds; 15 fills every window the model reads",
    )
    p.set_defaults(fn=cmd_add_voice)

    p = sub.add_parser("remove-voice", help="delete a voice from the library")
    p.add_argument("name")
    p.set_defaults(fn=cmd_remove_voice)

    p = sub.add_parser("say", help="teach it how to pronounce a name")
    p.add_argument("word", nargs="?", help="the word as written in the book")
    p.add_argument("say_as", nargs="?", help='respelling, e.g. "Kee-zhee"')
    p.add_argument(
        "--suggest", metavar="EPUB", help="list names in an epub that likely need a respelling"
    )
    p.add_argument("--remove", metavar="WORD", help="delete an entry")
    p.set_defaults(fn=cmd_say)

    p = sub.add_parser("serve", help="run the local web UI")
    p.add_argument("--host", default="127.0.0.1", help="interface to listen on (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})")
    p.add_argument("--open", action="store_true", help="open the UI in your browser")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("doctor", help="check ffmpeg, backends and GPU support")
    p.set_defaults(fn=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )
    if not args.verbose:
        logger.setLevel(logging.INFO)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — resume with:[/] abtts jobs")
        sys.exit(130)
    except _known_errors() as e:
        if args.verbose:
            console.print_exception()
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

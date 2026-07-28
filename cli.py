"""abtts command-line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress as RichProgress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from audiobooktts.config import Config
from audiobooktts.store import JobStore

console = Console()


def _apply_overrides(cfg: Config, args) -> Config:
    for attr in ("engine", "speed", "bitrate"):
        v = getattr(args, attr, None)
        if v is not None:
            setattr(cfg, attr, v)
    if getattr(args, "output_dir", None):
        cfg.output_dir = args.output_dir
    # Resolve last: an explicit --voice wins, otherwise fall back to one the
    # selected engine actually has.
    cfg.voice = getattr(args, "voice", None) or cfg.voice_for(cfg.engine)
    return cfg


def _run_with_progress(job_id: str, cfg: Config):
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

        def on_progress(p):
            if p.stage == "synthesizing":
                rp.update(
                    task,
                    total=max(p.n_chunks, 1),
                    completed=p.chunk_index,
                    chapter=f"[{p.chapter_index + 1}/{p.n_chapters}] {p.chapter_title[:40]}",
                )
            elif p.stage == "packaging":
                rp.update(task, chapter="packaging m4b…")

        manifest = run_job(job_id, cfg, store, on_progress)
    console.print(f"[green bold]Done:[/] {manifest.output_path}")


def cmd_convert(args):
    from audiobooktts.pipeline import create_job

    cfg = _apply_overrides(Config.load(), args)
    store = JobStore()
    manifest = create_job(args.epub, cfg, store)
    console.print(
        f"[bold]{manifest.book_title}[/] by {manifest.book_author} — "
        f"{len(manifest.chapters)} chapters, voice [cyan]{cfg.voice}[/] "
        f"(job {manifest.job_id})"
    )
    _run_with_progress(manifest.job_id, cfg)


def cmd_resume(args):
    cfg = _apply_overrides(Config.load(), args)
    store = JobStore()
    store.reconcile_running()
    manifest = store.load(args.job_id)
    done = sum(1 for c in manifest.chapters if c.done)
    console.print(
        f"Resuming [bold]{manifest.book_title}[/]: "
        f"{done}/{len(manifest.chapters)} chapters already finished"
    )
    _run_with_progress(args.job_id, cfg)


def cmd_voices(args):
    from audiobooktts.engines import get_engine

    engine = get_engine(args.engine or "chatterbox")
    table = Table(title=f"{engine.name} voices")
    table.add_column("id", style="cyan")
    table.add_column("accent")
    table.add_column("grade")
    table.add_column("description")
    for v in engine.list_voices():
        table.add_row(v.id, v.language, v.grade, v.label)
    console.print(table)


def cmd_preview(args):
    from audiobooktts.engines import get_engine
    from audiobooktts.epub import parse_epub
    from audiobooktts.textproc import clean_text, preview_sample

    import soundfile as sf

    cfg = _apply_overrides(Config.load(), args)
    book = parse_epub(args.epub)
    if not book.chapters:
        console.print("[red]No chapters found in epub[/]")
        sys.exit(1)
    chapter = book.chapters[min(args.chapter - 1, len(book.chapters) - 1)]
    sample = preview_sample(clean_text(chapter.text), max_chars=600)

    console.print(f"Previewing voice [cyan]{cfg.voice}[/] on: [dim]{sample[:100]}…[/]")
    engine = get_engine(cfg.engine)
    audio = engine.synthesize(sample, cfg.voice, cfg.speed)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, engine.sample_rate)
        console.print("[dim]Playing… (ctrl-c to stop)[/]")
        subprocess.run(["afplay", f.name])


def cmd_add_voice(args):
    from audiobooktts.voices import VoiceClipError, add_voice

    try:
        dest = add_voice(args.source, args.name, args.start, args.duration)
    except VoiceClipError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)
    console.print(
        f"[green]Added voice[/] [cyan]{args.name}[/] → {dest}\n"
        f"Try it: abtts preview yourbook.epub --engine chatterbox --voice {args.name}"
    )


def cmd_say(args):
    from audiobooktts import pronunciation

    if args.suggest:
        rows = pronunciation.suggest(args.suggest)
        table = Table(title="Names the dictionary doesn't know (most frequent first)")
        table.add_column("word", style="cyan")
        table.add_column("times", justify="right")
        for w, n in rows:
            table.add_row(w, str(n))
        console.print(table)
        console.print("Add one with: [cyan]abtts say Kizhi \"Kee-zhee\"[/]")
        return
    if args.remove:
        pronunciation.remove(args.remove)
        console.print(f"Removed [cyan]{args.remove}[/]")
        return
    if args.word and args.say_as:
        pronunciation.add(args.word, args.say_as)
        console.print(f"[green]{args.word}[/] will be said as [cyan]{args.say_as}[/]")
        return
    lexicon = pronunciation.load_lexicon()
    table = Table(title="Pronunciation lexicon")
    table.add_column("word", style="cyan")
    table.add_column("said as")
    for k in sorted(lexicon):
        table.add_row(k, lexicon[k])
    console.print(table if lexicon else "No entries yet.")


def cmd_jobs(args):
    store = JobStore()
    # Jobs left "running" by a killed server are orphaned; surface them as
    # interrupted so they can be resumed.
    for job_id in store.reconcile_running():
        console.print(f"[yellow]Marked orphaned job as interrupted:[/] {job_id}")
    table = Table(title="Jobs")
    for col in ("job id", "book", "status", "progress", "output"):
        table.add_column(col)
    for m in store.list_jobs():
        done = sum(1 for c in m.chapters if c.done)
        table.add_row(
            m.job_id, m.book_title, m.status,
            f"{done}/{len(m.chapters)}", m.output_path or "-",
        )
    console.print(table)


def cmd_serve(args):
    import uvicorn

    from audiobooktts.web import app

    console.print(f"[bold]AudioBookTTS[/] web UI: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="abtts", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--engine", default=None)
        p.add_argument("--voice", default=None)
        p.add_argument("--speed", type=float, default=None)
        p.add_argument("--bitrate", default=None)
        p.add_argument("--output-dir", default=None)

    p = sub.add_parser("convert", help="Convert an epub to an m4b audiobook")
    p.add_argument("epub")
    add_common(p)
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("resume", help="Resume an interrupted job")
    p.add_argument("job_id")
    add_common(p)
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("voices", help="List available voices")
    p.add_argument("--engine", default=None)
    p.set_defaults(fn=cmd_voices)

    p = sub.add_parser("preview", help="Hear a voice on your book's text")
    p.add_argument("epub")
    p.add_argument("--chapter", type=int, default=1)
    add_common(p)
    p.set_defaults(fn=cmd_preview)

    p = sub.add_parser(
        "add-voice", help="Make a cloning reference clip from any recording"
    )
    p.add_argument("source", help="Audio file (mp3, wav, m4b, …)")
    p.add_argument("--name", required=True, help="Name to select it by")
    p.add_argument("--start", type=float, default=45.0,
                   help="Seconds into the file to start (default skips LibriVox intros)")
    p.add_argument("--duration", type=float, default=15.0,
                   help="Clip length; 15s fills every window the model reads")
    p.set_defaults(fn=cmd_add_voice)

    p = sub.add_parser("say", help="Teach it how to pronounce a name")
    p.add_argument("word", nargs="?", help="The word as written in the book")
    p.add_argument("say_as", nargs="?", help='Respelling, e.g. "Kee-zhee"')
    p.add_argument("--suggest", metavar="EPUB",
                   help="List names in an epub that likely need a respelling")
    p.add_argument("--remove", metavar="WORD", help="Delete an entry")
    p.set_defaults(fn=cmd_say)

    p = sub.add_parser("jobs", help="List conversion jobs")
    p.set_defaults(fn=cmd_jobs)

    p = sub.add_parser("serve", help="Run the local web UI")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — resume with:[/] abtts jobs")
        sys.exit(130)


if __name__ == "__main__":
    main()

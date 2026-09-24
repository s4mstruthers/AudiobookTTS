"""Package per-chapter FLAC files into a chaptered, tagged .m4b via ffmpeg."""

from __future__ import annotations

import tempfile
from pathlib import Path

from audiobooktts import ffmpeg
from audiobooktts.epub import Book
from audiobooktts.fsutil import replace_file, safe_filename
from audiobooktts.store import JobStore, Manifest


class PackagingError(RuntimeError):
    pass


def _escape_metadata(value: str) -> str:
    # FFMETADATA escapes these with a backslash; newlines would end the value.
    for c in ("\\", "=", ";", "#", "\n"):
        value = value.replace(c, "\\" + c)
    return value


def ffmetadata(manifest: Manifest, book: Book) -> str:
    """Global tags plus one [CHAPTER] per chapter, timed from rendered durations."""
    lines = [
        ";FFMETADATA1",
        f"title={_escape_metadata(book.title)}",
        f"artist={_escape_metadata(book.author)}",
        f"album_artist={_escape_metadata(book.author)}",
        f"album={_escape_metadata(book.title)}",
        "genre=Audiobook",
    ]
    t = 0.0
    for ch in manifest.chapters:
        start_ms = round(t * 1000)
        t += ch.duration_s
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={round(t * 1000)}",
            f"title={_escape_metadata(ch.title)}",
        ]
    return "\n".join(lines) + "\n"


def _concat_list(paths: list[Path]) -> str:
    # Forward slashes work for ffmpeg on every platform and need no escaping;
    # single quotes in a path are closed, escaped and reopened.
    def quote(p: Path) -> str:
        return "'" + p.resolve().as_posix().replace("'", "'\\''") + "'"

    return "ffconcat version 1.0\n" + "".join(f"file {quote(p)}\n" for p in paths)


def output_path(book: Book, out_dir: Path) -> Path:
    title = safe_filename(book.title, max_len=120, fallback="Audiobook")
    author = safe_filename(book.author, max_len=60, fallback="")
    name = f"{title} - {author}" if author and author.lower() != "unknown" else title
    return out_dir / f"{name}.m4b"


def _write_cover(tmp: Path, manifest: Manifest, book: Book) -> Path | None:
    custom = Path(manifest.cover_path) if manifest.cover_path else None
    if custom is not None and custom.is_file():
        dest = tmp / f"cover{custom.suffix.lower()}"
        dest.write_bytes(custom.read_bytes())
        return dest
    if book.cover:
        ext = "png" if (book.cover_media_type or "").endswith("png") else "jpg"
        dest = tmp / f"cover.{ext}"
        dest.write_bytes(book.cover)
        return dest
    return None


def package_m4b(
    book: Book,
    manifest: Manifest,
    store: JobStore,
    out_dir: Path,
    bitrate: str = "80k",
    master: bool = True,
) -> Path:
    """Encode the finished chapters into ``out_dir`` and return the file path."""
    flacs = [store.chapter_audio_path(manifest.job_id, ch.index) for ch in manifest.chapters]
    missing = [p.name for p in flacs if not p.exists()]
    if missing:
        raise PackagingError(f"Missing chapter audio: {', '.join(missing[:3])}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path(book, out_dir)
    # Encode beside the target and swap it in at the end, so a failed or
    # interrupted encode never destroys an audiobook that was already there.
    partial = out_path.with_name(out_path.name + ".part")

    with tempfile.TemporaryDirectory(prefix="abtts-") as td:
        tmp = Path(td)
        concat_file = tmp / "concat.txt"
        concat_file.write_text(_concat_list(flacs), encoding="utf-8")
        meta_file = tmp / "meta.txt"
        meta_file.write_text(ffmetadata(manifest, book), encoding="utf-8")
        cover_file = _write_cover(tmp, manifest, book)

        concat_input = ["-f", "concat", "-safe", "0"]
        cmd = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *concat_input,
            "-i",
            str(concat_file),
            "-i",
            str(meta_file),
        ]
        if cover_file:
            cmd += ["-i", str(cover_file)]
        cmd += ["-map_metadata", "1", "-map_chapters", "1", "-map", "0:a"]
        if cover_file:
            cmd += [
                "-map",
                "2:v",
                "-c:v",
                "png" if cover_file.suffix == ".png" else "mjpeg",
                "-disposition:v",
                "attached_pic",
            ]
        if master:
            # Two-pass: measure the assembled book, then correct it. Raw TTS
            # lands near -26 dB RMS with peaks around -2 dB; ACX wants -23..-18
            # dB RMS and peaks under -3 dB.
            from audiobooktts.mastering import filter_chain, measure

            cmd += ["-af", filter_chain(measure(concat_file, concat_input))]
        cmd += [
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            "-ac",
            "1",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(partial),
        ]
        result = ffmpeg.run(cmd, text=True)
        if result.returncode != 0:
            partial.unlink(missing_ok=True)
            raise PackagingError(f"ffmpeg failed: {result.stderr[-2000:]}")
    replace_file(partial, out_path)
    return out_path

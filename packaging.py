"""Package per-chapter FLAC files into a chaptered .m4b via ffmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from audiobooktts.epub import Book
from audiobooktts.store import JobStore, Manifest


def _ffmetadata(manifest: Manifest, book: Book) -> str:
    def esc(s: str) -> str:
        for c in ("\\", "=", ";", "#", "\n"):
            s = s.replace(c, "\\" + c)
        return s

    lines = [
        ";FFMETADATA1",
        f"title={esc(book.title)}",
        f"artist={esc(book.author)}",
        f"album={esc(book.title)}",
        "genre=Audiobook",
    ]
    t = 0.0
    for ch in manifest.chapters:
        start_ms = int(t * 1000)
        t += ch.duration_s
        end_ms = int(t * 1000)
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={esc(ch.title)}",
        ]
    return "\n".join(lines) + "\n"


def package_m4b(
    book: Book,
    manifest: Manifest,
    store: JobStore,
    out_dir: Path,
    bitrate: str = "80k",
    master: bool = True,
) -> Path:
    job_id = manifest.job_id
    flacs = [store.chapter_audio_path(job_id, ch.index) for ch in manifest.chapters]
    missing = [p for p in flacs if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing chapter audio: {missing[:3]}")

    safe_title = "".join(c for c in book.title if c not in '/\\:*?"<>|').strip() or "book"
    safe_author = "".join(c for c in book.author if c not in '/\\:*?"<>|').strip()
    out_path = out_dir / (f"{safe_title} - {safe_author}.m4b" if safe_author else f"{safe_title}.m4b")

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        concat_file = tdp / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{str(p).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for p in flacs)
        )
        meta_file = tdp / "meta.txt"
        meta_file.write_text(_ffmetadata(manifest, book))

        cover_file = None
        if book.cover:
            ext = "png" if (book.cover_media_type or "").endswith("png") else "jpg"
            cover_file = tdp / f"cover.{ext}"
            cover_file.write_bytes(book.cover)

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(meta_file),
        ]
        if cover_file:
            cmd += ["-i", str(cover_file)]
        cmd += ["-map_metadata", "1", "-map", "0:a"]
        if cover_file:
            cmd += [
                "-map", "2:v", "-c:v", "mjpeg" if cover_file.suffix == ".jpg" else "png",
                "-disposition:v", "attached_pic",
            ]
        if master:
            # Two-pass: measure the assembled book, then correct it. Raw TTS
            # lands near -26 dB RMS with peaks around -2 dB; ACX wants -23..-18
            # dB RMS and peaks under -3 dB.
            from audiobooktts.mastering import filter_chain, measure

            measured = measure(concat_file, ["-f", "concat", "-safe", "0"])
            cmd += ["-af", filter_chain(measured)]
        cmd += [
            "-c:a", "aac", "-b:a", bitrate, "-ac", "1",
            "-movflags", "+faststart",
            "-f", "mp4", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")
    return out_path

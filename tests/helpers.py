"""Builders for test books and audio."""

from __future__ import annotations

import shutil
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")


def tiny_png(w: int = 8, h: int = 8) -> bytes:
    raw = b"".join(b"\x00" + bytes([200, 30, 30]) * w for _ in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


PARAGRAPH = (
    "Mr. Holmes arrived on the 2nd July, 1895, at 6 p.m. with 10,000 roubles. "
    "“Where?” she asked. The CENTRAL LONDON HATCHERY stood silent. "
    "The boat Fereale waited. "
)


def build_epub(
    path: Path,
    *,
    title: str = "The Test Book",
    author: str = "Jane Tester",
    chapters: int = 3,
    cover: str | None = "epub3",
    front_matter: bool = True,
) -> Path:
    """Write an epub with ebooklib: numbered chapters, optional cover and copyright page."""
    from ebooklib import epub

    b = epub.EpubBook()
    b.set_identifier("test-book")
    b.set_title(title)
    b.set_language("en")
    b.add_author(author)
    if cover == "epub3":
        b.set_cover("cover.png", tiny_png())
    docs = []
    numerals = ["I", "II", "III", "IV", "V", "VI"]
    for i in range(chapters):
        name = f"Chapter {numerals[i]}"
        doc = epub.EpubHtml(title=name, file_name=f"ch{i + 1}.xhtml", lang="en")
        body = "".join(f"<p>{PARAGRAPH * (i + 1)}Paragraph {j}.</p>" for j in range(3))
        doc.content = f"<h1>{name}</h1>{body}"
        b.add_item(doc)
        docs.append(doc)
    toc = [epub.Link(d.file_name, d.title, d.file_name) for d in docs]
    spine: list = ["nav"]
    if front_matter:
        cr = epub.EpubHtml(title="Copyright", file_name="copyright.xhtml")
        cr.content = "<p>" + "All rights reserved. " * 20 + "</p>"
        b.add_item(cr)
        toc.insert(0, epub.Link("copyright.xhtml", "Copyright", "copyright"))
        spine.append(cr)
    b.toc = toc
    b.add_item(epub.EpubNcx())
    b.add_item(epub.EpubNav())
    b.spine = spine + docs
    epub.write_epub(str(path), b)
    return path


def build_raw_epub(path: Path, files: dict[str, str | bytes], opf_path: str = "content.opf") -> Path:
    """Write an epub zip by hand, for layouts ebooklib's writer cannot produce."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            f'<rootfile full-path="{opf_path}" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
        for name, data in files.items():
            z.writestr(name, data)
    return path




def write_tone(path: Path, seconds: float, sample_rate: int = 24000, channels: int = 1) -> Path:
    import soundfile as sf

    t = np.arange(int(seconds * sample_rate)) / sample_rate
    mono = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate)
    return path

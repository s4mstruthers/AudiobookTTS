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
    cover: bool = True,
    front_matter: bool = True,
    toc: str = "both",
) -> Path:
    """Write an EPUB 3 book: numbered chapters, optional cover and copyright page.

    toc chooses the table of contents: "ncx" (EPUB 2), "nav" (EPUB 3) or "both",
    as most real books carry.
    """
    numerals = ["I", "II", "III", "IV", "V", "VI"]
    docs = [(f"ch{i + 1}.xhtml", f"Chapter {numerals[i]}", i) for i in range(chapters)]
    files: dict[str, str | bytes] = {}
    for name, heading, i in docs:
        body = "".join(f"<p>{PARAGRAPH * (i + 1)}Paragraph {j}.</p>" for j in range(3))
        files[f"EPUB/{name}"] = _xhtml(heading, f"<h1>{heading}</h1>{body}")
    toc_entries = [(name, heading) for name, heading, _ in docs]
    spine = ["nav"] + [f"c{i}" for i in range(chapters)]
    manifest = [
        f'<item id="c{i}" href="{name}" media-type="application/xhtml+xml"/>' for name, _, i in docs
    ]
    if front_matter:
        files["EPUB/copyright.xhtml"] = _xhtml(
            "Copyright", "<p>" + "All rights reserved. " * 20 + "</p>"
        )
        manifest.append(
            '<item id="copy" href="copyright.xhtml" media-type="application/xhtml+xml"/>'
        )
        spine.insert(1, "copy")
        toc_entries.insert(0, ("copyright.xhtml", "Copyright"))
    if cover:
        files["EPUB/cover.png"] = tiny_png()
        manifest.append(
            '<item id="cover-img" href="cover.png" media-type="image/png" '
            'properties="cover-image"/>'
        )

    nav_links = "".join(f'<li><a href="{h}">{t}</a></li>' for h, t in toc_entries)
    files["EPUB/nav.xhtml"] = _xhtml(
        "Contents", f'<nav epub:type="toc" id="toc"><h1>Contents</h1><ol>{nav_links}</ol></nav>'
    )
    manifest.append(
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml"'
        + (' properties="nav"' if toc in ("nav", "both") else "")
        + "/>"
    )
    spine_toc = ""
    if toc in ("ncx", "both"):
        points = "".join(
            f'<navPoint id="p{i}"><navLabel><text>{t}</text></navLabel>'
            f'<content src="{h}"/></navPoint>'
            for i, (h, t) in enumerate(toc_entries)
        )
        files["EPUB/toc.ncx"] = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
            f"<head/><docTitle><text>{title}</text></docTitle><navMap>{points}</navMap></ncx>"
        )
        manifest.append('<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
        spine_toc = ' toc="ncx"'

    itemrefs = "".join(f'<itemref idref="{i}"/>' for i in spine)
    files["EPUB/content.opf"] = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">test-book</dc:identifier>
    <dc:title>{title}</dc:title>
    <dc:creator>{author}</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>{"".join(manifest)}</manifest>
  <spine{spine_toc}>{itemrefs}</spine>
</package>"""
    return build_raw_epub(path, files, opf_path="EPUB/content.opf")


def _xhtml(title: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        f"<head><title>{title}</title></head><body>{body}</body></html>"
    )


def build_raw_epub(
    path: Path, files: dict[str, str | bytes], opf_path: str = "content.opf"
) -> Path:
    """Write an epub zip by hand, with complete control over its layout."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
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

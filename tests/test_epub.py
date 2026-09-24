"""EPUB parsing: chapters, titles, covers and awkward real-world layouts."""

from __future__ import annotations

import pytest

from audiobooktts.epub import EpubError, _repair_metadata, parse_epub
from helpers import build_epub, build_raw_epub, tiny_png

BODY = "<p>" + "It was a quiet evening and nothing stirred at all. " * 8 + "</p>"


def _opf(manifest: str, spine: str, metadata_extra: str = "", toc_id: str = "ncx") -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">raw-book</dc:identifier>
    <dc:title>Raw Book</dc:title>
    <dc:creator>A. Writer</dc:creator>
    <dc:language>en</dc:language>
    {metadata_extra}
  </metadata>
  <manifest>{manifest}</manifest>
  <spine toc="{toc_id}">{spine}</spine>
</package>"""


def _ncx(points: list[tuple[str, str]]) -> str:
    nav = "".join(
        f'<navPoint id="p{i}" playOrder="{i + 1}"><navLabel><text>{title}</text></navLabel>'
        f'<content src="{src}"/></navPoint>'
        for i, (src, title) in enumerate(points)
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        '<head><meta name="dtb:uid" content="raw-book"/></head>'
        f"<docTitle><text>Raw Book</text></docTitle><navMap>{nav}</navMap></ncx>"
    )


def _xhtml(heading: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>x</title></head><body><h1>{heading}</h1>{BODY}</body></html>"
    )


def test_chapters_titles_and_front_matter(epub_path):
    book = parse_epub(epub_path)
    assert (book.title, book.author) == ("The Test Book", "Jane Tester")
    # The copyright page is skipped; the heading repeating the title is dropped.
    assert [c.title for c in book.chapters] == ["Chapter I", "Chapter II", "Chapter III"]
    assert not book.chapters[0].text.startswith("Chapter I")


def test_epub3_cover_image_is_found(epub_path):
    # Regression: the EPUB 3 cover-image property was missed, losing the cover
    # of most modern books.
    book = parse_epub(epub_path)
    assert book.cover == tiny_png()
    assert book.cover_media_type == "image/png"


def test_epub2_meta_cover_is_found(tmp_path):
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="img1" href="images/front.png" media-type="image/png"/>'
    )
    path = build_raw_epub(
        tmp_path / "e2.epub",
        {
            "content.opf": _opf(
                manifest, '<itemref idref="c1"/>', '<meta name="cover" content="img1"/>'
            ),
            "toc.ncx": _ncx([("c1.xhtml", "Opening")]),
            "c1.xhtml": _xhtml("Opening"),
            "images/front.png": tiny_png(),
        },
    )
    assert parse_epub(path).cover == tiny_png()


def test_ncx_links_relative_to_ncx_folder_and_percent_encoded(tmp_path):
    # Regression: NCX links are relative to the NCX file and may be
    # percent-encoded; both must be resolved to match the spine.
    manifest = (
        '<item id="ncx" href="OEBPS/toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="c1" href="OEBPS/Text/one.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c2" href="OEBPS/Text/chapter%20two.xhtml" media-type="application/xhtml+xml"/>'
    )
    path = build_raw_epub(
        tmp_path / "sub.epub",
        {
            "content.opf": _opf(manifest, '<itemref idref="c1"/><itemref idref="c2"/>'),
            "OEBPS/toc.ncx": _ncx(
                [("Text/one.xhtml", "The Arrival"), ("Text/chapter%20two.xhtml", "The Departure")]
            ),
            "OEBPS/Text/one.xhtml": _xhtml("One"),
            "OEBPS/Text/chapter two.xhtml": _xhtml("Two"),
        },
    )
    assert [c.title for c in parse_epub(path).chapters] == ["The Arrival", "The Departure"]


def test_single_file_book_split_at_anchors(tmp_path):
    body = "".join(f'<h2 id="c{i}">Part {i}</h2>{BODY}' for i in range(1, 4))
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="all" href="all.xhtml" media-type="application/xhtml+xml"/>'
    )
    path = build_raw_epub(
        tmp_path / "single.epub",
        {
            "content.opf": _opf(manifest, '<itemref idref="all"/>'),
            "toc.ncx": _ncx([(f"all.xhtml#c{i}", f"Part {i}") for i in range(1, 4)]),
            "all.xhtml": '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            + body
            + "</body></html>",
        },
    )
    chapters = parse_epub(path).chapters
    assert [c.title for c in chapters] == ["Part 1", "Part 2", "Part 3"]
    assert all(c.text.startswith("It was a quiet evening") for c in chapters)


def test_empty_navmap_does_not_crash(tmp_path):
    # Regression: an empty navMap crashed the previous parser.
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
    )
    path = build_raw_epub(
        tmp_path / "empty-toc.epub",
        {
            "content.opf": _opf(manifest, '<itemref idref="c1"/>'),
            "toc.ncx": _ncx([]),
            "c1.xhtml": _xhtml("Only Chapter"),
        },
    )
    assert [c.title for c in parse_epub(path).chapters] == ["Only Chapter"]


def test_not_an_epub_raises(tmp_path):
    bad = tmp_path / "bad.epub"
    bad.write_bytes(b"definitely not a zip")
    with pytest.raises(EpubError):
        parse_epub(bad)


def test_book_without_cover(tmp_path):
    book = parse_epub(build_epub(tmp_path / "nocover.epub", cover=False))
    assert book.cover is None


@pytest.mark.parametrize(
    ("title", "author", "expected"),
    [
        ("Emma by Jane Austen", "Unknown", ("Emma", "Jane Austen")),
        ("Emma by Jane Austen", "", ("Emma", "Jane Austen")),
        # A real title containing "by" with a proper author is left alone.
        ("Stand by Me", "Stephen King", ("Stand by Me", "Stephen King")),
    ],
)
def test_repair_metadata(title, author, expected):
    assert _repair_metadata(title, author) == expected


@pytest.mark.parametrize("toc", ["ncx", "nav", "both"])
def test_either_table_of_contents_gives_the_titles(tmp_path, toc):
    book = parse_epub(build_epub(tmp_path / f"{toc}.epub", toc=toc))
    assert [c.title for c in book.chapters] == ["Chapter I", "Chapter II", "Chapter III"]


def test_xml_external_entities_are_not_resolved(tmp_path):
    # A crafted epub must not be able to read files from this computer (XXE).
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    manifest = '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
    opf = _opf(manifest, '<itemref idref="c1"/>', toc_id="").replace(
        "<dc:title>Raw Book</dc:title>", "<dc:title>&xxe;</dc:title>"
    )
    opf = opf.replace(
        '<?xml version="1.0" encoding="utf-8"?>',
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<!DOCTYPE package [<!ENTITY xxe SYSTEM "{secret.as_uri()}">]>',
    )
    path = build_raw_epub(tmp_path / "xxe.epub", {"content.opf": opf, "c1.xhtml": _xhtml("One")})
    assert "TOP-SECRET" not in parse_epub(path).title


def test_links_with_the_wrong_case_still_resolve(tmp_path):
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="c1" href="Text/One.xhtml" media-type="application/xhtml+xml"/>'
    )
    path = build_raw_epub(
        tmp_path / "case.epub",
        {
            "content.opf": _opf(manifest, '<itemref idref="c1"/>'),
            "toc.ncx": _ncx([("Text/One.xhtml", "Opening")]),
            "text/one.xhtml": _xhtml("One"),  # stored with different case
        },
    )
    assert [c.title for c in parse_epub(path).chapters] == ["Opening"]


def test_missing_chapter_file_is_skipped(tmp_path):
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c2" href="gone.xhtml" media-type="application/xhtml+xml"/>'
    )
    path = build_raw_epub(
        tmp_path / "missing.epub",
        {
            "content.opf": _opf(manifest, '<itemref idref="c1"/><itemref idref="c2"/>'),
            "toc.ncx": _ncx([("c1.xhtml", "Present")]),
            "c1.xhtml": _xhtml("Present"),
        },
    )
    assert [c.title for c in parse_epub(path).chapters] == ["Present"]


def test_missing_package_file_raises(tmp_path):
    path = build_raw_epub(tmp_path / "noopf.epub", {"c1.xhtml": _xhtml("One")})
    with pytest.raises(EpubError, match="missing"):
        parse_epub(path)

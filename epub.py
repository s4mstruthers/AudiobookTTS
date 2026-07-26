"""Parse an epub into an ordered list of chapters of plain text."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import ITEM_DOCUMENT, ITEM_IMAGE, epub as ebepub

# Epub content is XHTML, but the lenient HTML parser is the deliberate choice
# here: real-world epubs are frequently malformed and a strict XML parse fails.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Spine documents shorter than this (in characters) are treated as
# decorative (cover pages, blank separators) and dropped.
MIN_CHAPTER_CHARS = 200

_SKIP_TITLE_RE = re.compile(
    r"^(cover|title\s*page|copyright|contents|table of contents|dedication|"
    r"colophon|also by|about the author|other books)",
    re.IGNORECASE,
)


@dataclass
class Chapter:
    title: str
    text: str


@dataclass
class Book:
    title: str
    author: str
    language: str
    chapters: list[Chapter] = field(default_factory=list)
    cover: bytes | None = None
    cover_media_type: str | None = None


def _flatten_toc(toc) -> list[tuple[str, str]]:
    """Flatten ebooklib's nested TOC into (href-without-fragment, title) pairs."""
    out: list[tuple[str, str]] = []
    for node in toc:
        if isinstance(node, tuple):
            section, children = node
            if getattr(section, "href", None):
                out.append((section.href.split("#")[0], section.title or ""))
            out.extend(_flatten_toc(children))
        elif getattr(node, "href", None):
            out.append((node.href.split("#")[0], node.title or ""))
    return out


def _html_to_text(html: bytes | str) -> tuple[str, str]:
    """Return (heading, body text) for one spine document."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "sup", "table", "figure", "nav"]):
        tag.decompose()

    heading = ""
    h = soup.find(["h1", "h2", "h3"])
    if h:
        heading = h.get_text(" ", strip=True)

    blocks: list[str] = []
    body = soup.body or soup
    for el in body.find_all(["p", "h1", "h2", "h3", "h4", "blockquote", "li"]):
        txt = el.get_text(" ", strip=True)
        if txt:
            blocks.append(txt)
    if not blocks:  # fallback: whole-document text
        txt = body.get_text(" ", strip=True)
        if txt:
            blocks.append(txt)
    return heading, "\n\n".join(blocks)


def _find_cover(book: ebepub.EpubBook) -> tuple[bytes | None, str | None]:
    for item in book.get_items_of_type(ITEM_IMAGE):
        name = (item.get_name() or "").lower()
        item_id = (item.get_id() or "").lower()
        if "cover" in name or "cover" in item_id:
            return item.get_content(), item.media_type
    return None, None


def parse_epub(path: str | Path) -> Book:
    eb = ebepub.read_epub(str(path), options={"ignore_ncx": False})

    def meta(name: str, default: str = "") -> str:
        try:
            values = eb.get_metadata("DC", name)
            return values[0][0] if values else default
        except Exception:
            return default

    toc_titles = dict(_flatten_toc(eb.toc))
    docs = {item.get_name(): item for item in eb.get_items_of_type(ITEM_DOCUMENT)}

    chapters: list[Chapter] = []
    n = 0
    for spine_id, _linear in eb.spine:
        item = eb.get_item_with_id(spine_id)
        if item is None or item.get_name() not in docs:
            continue
        href = item.get_name()
        heading, text = _html_to_text(item.get_content())
        title = toc_titles.get(href) or heading
        if len(text) < MIN_CHAPTER_CHARS:
            continue
        if title and _SKIP_TITLE_RE.match(title.strip()):
            continue
        n += 1
        chapters.append(Chapter(title=title or f"Chapter {n}", text=text))

    cover, cover_mt = _find_cover(eb)
    return Book(
        title=meta("title", Path(path).stem),
        author=meta("creator", "Unknown"),
        language=meta("language", "en"),
        chapters=chapters,
        cover=cover,
        cover_media_type=cover_mt,
    )

"""Parse an epub into an ordered list of chapters of plain text."""

from __future__ import annotations

import bisect
import posixpath
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE, ITEM_NAVIGATION
from ebooklib import epub as ebepub

# Epub content is XHTML, but the lenient HTML parser is the deliberate choice
# here: real-world epubs are frequently malformed and a strict XML parse fails.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Spine documents shorter than this (in characters) are treated as
# decorative (cover pages, blank separators) and dropped.
MIN_CHAPTER_CHARS = 200

_BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "li"]

# Front and back matter nobody wants narrated.
_SKIP_TITLE_RE = re.compile(
    r"^(cover|title\s*page|copyright|contents|table of contents|dedication|"
    r"colophon|also by|about the author|other books)",
    re.IGNORECASE,
)

# Calibre-converted books often stuff "Title by Author" into dc:title and put
# something useless in dc:creator, so the pattern is worth recognising.
_TITLE_BY_AUTHOR_RE = re.compile(r"^(.{2,}?)\s+by\s+(.{2,})$", re.IGNORECASE)


class EpubError(ValueError):
    """The file could not be read as an epub."""


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


def _norm_href(href: str) -> str:
    return posixpath.normpath(unquote(href)).lstrip("/")


def _flatten_toc(toc) -> list[tuple[str, str, str]]:
    """Flatten the nested TOC into ordered (href, fragment, title) triples."""
    out: list[tuple[str, str, str]] = []

    def add(node) -> None:
        href = getattr(node, "href", None)
        if href:
            base, _, frag = href.partition("#")
            out.append((base, frag, getattr(node, "title", "") or ""))

    def walk(nodes) -> None:
        # ebooklib returns a single Link, not a list, for an empty navMap.
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]
        for node in nodes:
            if isinstance(node, tuple) and len(node) == 2:
                section, children = node
                add(section)
                walk(children)
            elif isinstance(node, list):
                walk(node)
            else:
                add(node)

    walk(toc)
    return out


def _href_resolver(book: ebepub.EpubBook, doc_names: set[str]):
    """Map a TOC href onto a spine document name.

    ebooklib resolves nav-document links against the nav file's folder, but
    NCX links are passed through as written: relative to the NCX file, and
    still percent-encoded. Chapters were silently lost from books laid out
    that way, so every plausible reading is tried.
    """
    by_norm = {_norm_href(n): n for n in doc_names}
    by_base: dict[str, list[str]] = {}
    for n in doc_names:
        by_base.setdefault(posixpath.basename(_norm_href(n)), []).append(n)
    toc_dirs = {
        posixpath.dirname(_norm_href(item.get_name()))
        for item in book.get_items_of_type(ITEM_NAVIGATION)
    }

    def resolve(href: str) -> str | None:
        norm = _norm_href(href)
        if norm in by_norm:
            return by_norm[norm]
        for d in toc_dirs:
            joined = _norm_href(posixpath.join(d, href))
            if joined in by_norm:
                return by_norm[joined]
        same_base = by_base.get(posixpath.basename(norm), [])
        return same_base[0] if len(same_base) == 1 else None

    return resolve


def _leaf_blocks(root) -> list:
    """Innermost block elements only.

    A block that contains another block is skipped because its children cover
    the same text. Taking the outermost blocks instead would collapse the whole
    document whenever one <blockquote> or <div>-like wrapper encloses the book;
    taking every match would narrate nested paragraphs twice.
    """
    return [el for el in root.find_all(_BLOCK_TAGS) if el.find(_BLOCK_TAGS) is None]


def _clean_soup(html: bytes | str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "sup", "table", "figure", "nav"]):
        tag.decompose()
    # <br> carries no text, so without this the words either side would run
    # together once inline separators are removed.
    for br in soup.find_all("br"):
        br.replace_with(" ")
    return soup


def _text_of(el) -> str:
    """Block text with the document's own spacing preserved.

    Inline markup must not introduce whitespace. Drop caps are written
    `<span>A</span>bout` with no space — inserting one there yields "A bout".
    Where a space is genuinely intended the source has it, as in
    `<span>A</span> SQUAT`, so the markup alone decides the word boundaries.
    """
    return re.sub(r"\s+", " ", el.get_text()).strip()


def _block_texts(root) -> list[str]:
    return [txt for el in _leaf_blocks(root) if (txt := _text_of(el))]


def _element_order(body) -> dict[int, int]:
    return {id(el): i for i, el in enumerate(body.descendants) if getattr(el, "name", None)}


def _anchor_position(body, frag: str, order: dict[int, int]) -> int | None:
    el = body.find(id=frag) or body.find("a", attrs={"name": frag})
    if el is None:
        return None
    # An inline <a id="…"> usually sits inside the chapter heading, so anchor on
    # that heading instead — otherwise the heading falls into the section above.
    if el.name not in _BLOCK_TAGS:
        parent = el.parent
        if parent is not None and parent.name in _BLOCK_TAGS and id(parent) in order:
            el = parent
    return order.get(id(el))


def _split_at_fragments(
    soup: BeautifulSoup, fragments: list[tuple[str, str]]
) -> tuple[list[str], list[Chapter]] | None:
    """Split one document into chapters at the given TOC anchor fragments.

    Many epubs (especially calibre single-file conversions) hold the entire book
    in one document and mark chapters only with #anchors. Returns the text found
    before the first anchor plus one Chapter per anchor, or None if the anchors
    could not be located.
    """
    body = soup.body or soup
    order = _element_order(body)

    anchors: list[tuple[int, str]] = []
    for frag, title in fragments:
        pos = _anchor_position(body, frag, order)
        if pos is not None:
            anchors.append((pos, title))
    if len(anchors) < 2:
        return None
    anchors.sort(key=lambda a: a[0])

    positions = [pos for pos, _ in anchors]
    sections: list[list[str]] = [[] for _ in anchors]
    preamble: list[str] = []

    for el in _leaf_blocks(body):
        txt = _text_of(el)
        if not txt:
            continue
        # The last anchor at or before this block owns it.
        idx = bisect.bisect_right(positions, order.get(id(el), -1)) - 1
        (preamble if idx < 0 else sections[idx]).append(txt)

    chapters = [
        Chapter(title=title, text="\n\n".join(texts))
        for (_, title), texts in zip(anchors, sections, strict=True)
    ]
    return preamble, chapters


def _strip_leading_title(text: str, title: str) -> str:
    """Drop a heading block that just repeats the chapter title.

    The heading is part of the document, so it lands in the chapter text; the
    pipeline also announces the title before the body. Left alone the listener
    hears "Chapter 2. Chapter 2. About half way…".
    """
    if not title or not text:
        return text
    blocks = text.split("\n\n")
    key = re.sub(r"[^a-z0-9]+", "", title.lower())
    if key and re.sub(r"[^a-z0-9]+", "", blocks[0].lower()) == key:
        return "\n\n".join(blocks[1:]).lstrip()
    return text


def _find_cover(book: ebepub.EpubBook) -> tuple[bytes | None, str | None]:
    """The cover image, found the way readers find it.

    In order: an EPUB 3 ``cover-image`` item (which ebooklib types as a cover,
    not an image, so searching images alone misses it), the EPUB 2
    ``<meta name="cover">`` reference, then any image named like a cover.
    """
    for item in book.get_items_of_type(ITEM_COVER):
        if item.get_content():
            return item.get_content(), item.media_type

    try:
        metas = book.get_metadata("OPF", "meta")
    except (KeyError, AttributeError):
        metas = []
    for _value, attrs in metas:
        attrs = attrs or {}
        if attrs.get("name") != "cover":
            continue
        item = book.get_item_with_id(attrs.get("content", ""))
        if item is not None and (item.media_type or "").startswith("image/"):
            return item.get_content(), item.media_type

    for item in book.get_items_of_type(ITEM_IMAGE):
        name = (item.get_name() or "").lower()
        item_id = (item.get_id() or "").lower()
        if "cover" in name or "cover" in item_id:
            return item.get_content(), item.media_type
    return None, None


def _repair_metadata(title: str, author: str) -> tuple[str, str]:
    """Recover title/author when dc:title holds 'Title by Author'.

    Only overrides the author when the file's own value is clearly unusable —
    empty, "Unknown", or a copy of the title — so correctly tagged books whose
    titles legitimately contain " by " are left alone.
    """
    m = _TITLE_BY_AUTHOR_RE.match(title.strip())
    if not m:
        return title, author
    real_title, real_author = m.group(1).strip(), m.group(2).strip()
    a = (author or "").strip().lower()
    if a in ("", "unknown") or a == real_title.lower():
        return real_title, real_author
    return title, author


def _is_front_matter(title: str) -> bool:
    return bool(title) and bool(_SKIP_TITLE_RE.match(title.strip()))


def parse_epub(path: str | Path) -> Book:
    """Read an epub into narratable chapters. Raises EpubError if unreadable."""
    try:
        eb = ebepub.read_epub(str(path), options={"ignore_ncx": False})
    except Exception as e:  # ebooklib raises a zoo of types for bad input
        raise EpubError(f"Not a readable epub: {e}") from e

    def meta(name: str, default: str = "") -> str:
        try:
            values = eb.get_metadata("DC", name)
            return (values[0][0] or default).strip() if values else default
        except (KeyError, IndexError, AttributeError):
            return default

    docs = {item.get_name() for item in eb.get_items_of_type(ITEM_DOCUMENT)}
    resolve = _href_resolver(eb, docs)

    # Fragments per document, in TOC order, plus a plain title for whole-file entries.
    frags_by_href: dict[str, list[tuple[str, str]]] = {}
    title_by_href: dict[str, str] = {}
    for href, frag, title in _flatten_toc(eb.toc):
        name = resolve(href)
        if name is None:
            continue
        if frag:
            frags_by_href.setdefault(name, []).append((frag, title))
        else:
            title_by_href.setdefault(name, title)

    chapters: list[Chapter] = []
    for spine_id, _linear in eb.spine:
        item = eb.get_item_with_id(spine_id)
        if item is None or item.get_name() not in docs:
            continue
        href = item.get_name()
        soup = _clean_soup(item.get_content())

        fragments = frags_by_href.get(href, [])
        if len(fragments) >= 2:
            split = _split_at_fragments(soup, fragments)
            if split is not None:
                preamble, sub_chapters = split
                pre_title = title_by_href.get(href, "Introduction")
                pre_text = "\n\n".join(preamble)
                if len(pre_text) >= MIN_CHAPTER_CHARS and not _is_front_matter(pre_title):
                    chapters.append(Chapter(title=pre_title, text=pre_text))
                chapters.extend(
                    c
                    for c in sub_chapters
                    if len(c.text) >= MIN_CHAPTER_CHARS and not _is_front_matter(c.title)
                )
                continue

        body = soup.body or soup
        blocks = _block_texts(body)
        if not blocks:
            txt = _text_of(body)
            blocks = [txt] if txt else []
        text = "\n\n".join(blocks)
        if len(text) < MIN_CHAPTER_CHARS:
            continue

        heading = ""
        h = soup.find(["h1", "h2", "h3"])
        if h:
            heading = _text_of(h)
        title = title_by_href.get(href) or heading
        if _is_front_matter(title):
            continue
        chapters.append(Chapter(title=title or f"Chapter {len(chapters) + 1}", text=text))

    title, author = _repair_metadata(meta("title", Path(path).stem), meta("creator", "Unknown"))
    for ch in chapters:
        ch.text = _strip_leading_title(ch.text, ch.title)
    chapters = [c for c in chapters if len(c.text) >= MIN_CHAPTER_CHARS]

    cover, cover_mt = _find_cover(eb)
    return Book(
        title=title or Path(path).stem,
        author=author or "Unknown",
        language=meta("language", "en"),
        chapters=chapters,
        cover=cover,
        cover_media_type=cover_mt,
    )

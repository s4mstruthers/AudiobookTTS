"""A minimal, read-only EPUB container reader.

An epub is a zip archive: ``META-INF/container.xml`` points at an OPF package
file, which lists the book's metadata, every file in it (the manifest), the
reading order (the spine) and where the table of contents lives. That is all
this module reads. It needs only the standard library and lxml, which keeps
the project free of copyleft dependencies.

Every path handed out is a full path inside the zip, already resolved against
the file that referred to it and percent-decoded, so callers can compare them
directly.
"""

from __future__ import annotations

import posixpath
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from lxml import etree

# Hostile or corrupt files must not be able to exhaust memory by claiming a
# huge uncompressed size (a "zip bomb"). Real chapters are well under this.
MAX_ENTRY_BYTES = 256 * 1024 * 1024

DOCUMENT_TYPES = {"application/xhtml+xml", "text/html", "application/html"}
NCX_TYPE = "application/x-dtbncx+xml"
_OPS_NS = "http://www.idpf.org/2007/ops"

# Resolving entities would let a crafted file read local files (XXE); recover
# tolerates the malformed XML that real-world epubs are full of.
_XML_PARSER = etree.XMLParser(
    resolve_entities=False, no_network=True, recover=True, huge_tree=False
)


class EpubFormatError(ValueError):
    """The file is not a readable epub."""


@dataclass
class ManifestItem:
    id: str
    path: str  # full path inside the zip
    media_type: str
    properties: set[str] = field(default_factory=set)

    @property
    def is_document(self) -> bool:
        return self.media_type in DOCUMENT_TYPES


@dataclass
class TocEntry:
    path: str  # full path inside the zip, without any #fragment
    fragment: str
    title: str


def _local(tag) -> str:
    """Element name without its namespace, so namespace variations don't matter."""
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _children(el, name: str) -> list:
    return [c for c in el if _local(c.tag) == name]


def _first(el, name: str):
    return next((c for c in el.iter() if _local(c.tag) == name), None)


def _text(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def resolve(base_file: str, href: str) -> tuple[str, str]:
    """Resolve ``href`` against the zip path of the file that contains it.

    Returns (path, fragment), with the path normalised and percent-decoded.
    """
    href, _, fragment = href.partition("#")
    if not href:
        return base_file, fragment
    joined = posixpath.join(posixpath.dirname(base_file), unquote(href))
    return posixpath.normpath(joined).lstrip("/"), fragment


class EpubFile:
    """An open epub. Use as a context manager, or call close()."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self._zip = zipfile.ZipFile(self.path)
        except (zipfile.BadZipFile, OSError) as e:
            raise EpubFormatError(f"Not a readable epub: {e}") from e
        # Some epubs are authored on case-insensitive filesystems and refer to
        # files with the wrong case; fall back to a case-insensitive match.
        self._names = {name: name for name in self._zip.namelist()}
        self._folded = {name.lower(): name for name in self._names}
        try:
            self._load()
        except EpubFormatError:
            self.close()
            raise
        except Exception as e:
            self.close()
            raise EpubFormatError(f"Not a readable epub: {e}") from e

    # --- zip access ------------------------------------------------------------

    def exists(self, path: str) -> bool:
        return path in self._names or path.lower() in self._folded

    def read(self, path: str) -> bytes:
        name = self._names.get(path) or self._folded.get(path.lower())
        if name is None:
            raise KeyError(path)
        info = self._zip.getinfo(name)
        if info.file_size > MAX_ENTRY_BYTES:
            raise EpubFormatError(f"{name} is implausibly large ({info.file_size} bytes)")
        return self._zip.read(info)

    def _xml(self, path: str):
        root = etree.fromstring(self.read(path), _XML_PARSER)
        if root is None:
            raise EpubFormatError(f"{path} is not valid XML")
        return root

    # --- package document --------------------------------------------------------

    def _load(self) -> None:
        try:
            container = self._xml("META-INF/container.xml")
        except KeyError:
            raise EpubFormatError("Not an epub: META-INF/container.xml is missing") from None
        rootfile = _first(container, "rootfile")
        if rootfile is None or not rootfile.get("full-path"):
            raise EpubFormatError("Not an epub: container.xml names no package file")
        self.opf_path = posixpath.normpath(unquote(rootfile.get("full-path"))).lstrip("/")
        try:
            opf = self._xml(self.opf_path)
        except KeyError:
            raise EpubFormatError(f"Package file {self.opf_path} is missing") from None

        self.metadata: dict[str, list[str]] = {}
        self._meta_attrs: list[dict[str, str]] = []
        metadata = _first(opf, "metadata")
        if metadata is not None:
            for el in metadata.iter():
                if not isinstance(el.tag, str):
                    continue  # comments and processing instructions
                name = _local(el.tag)
                if name == "meta":
                    self._meta_attrs.append(dict(el.attrib))
                elif etree.QName(el.tag).namespace == "http://purl.org/dc/elements/1.1/":
                    value = _text(el)
                    if value:
                        self.metadata.setdefault(name, []).append(value)

        self.manifest: dict[str, ManifestItem] = {}
        manifest = _first(opf, "manifest")
        for el in _children(manifest, "item") if manifest is not None else []:
            item_id, href = el.get("id"), el.get("href")
            if not item_id or not href:
                continue
            path, _ = resolve(self.opf_path, href)
            self.manifest[item_id] = ManifestItem(
                id=item_id,
                path=path,
                media_type=(el.get("media-type") or "").strip().lower(),
                properties=set((el.get("properties") or "").split()),
            )
        self._by_path = {item.path: item for item in self.manifest.values()}

        spine = _first(opf, "spine")
        self.spine: list[ManifestItem] = []
        self._ncx_id = spine.get("toc", "") if spine is not None else ""
        for ref in _children(spine, "itemref") if spine is not None else []:
            item = self.manifest.get(ref.get("idref", ""))
            if item is not None:
                self.spine.append(item)

    def meta(self, name: str, default: str = "") -> str:
        """The first Dublin Core value for ``name`` (title, creator, language...)."""
        values = self.metadata.get(name)
        return values[0] if values else default

    def item_at(self, path: str) -> ManifestItem | None:
        return self._by_path.get(path)

    # --- table of contents -------------------------------------------------------

    def toc(self) -> list[TocEntry]:
        """The table of contents, flattened in reading order.

        The EPUB 2 NCX is preferred when present, as it was with the previous
        parser, so chapter titles do not change; the EPUB 3 navigation
        document is used otherwise, or when the NCX lists nothing.
        """
        return self._ncx_toc() or self._nav_toc()

    def _ncx_toc(self) -> list[TocEntry]:
        item = self.manifest.get(self._ncx_id) or next(
            (i for i in self.manifest.values() if i.media_type == NCX_TYPE), None
        )
        if item is None or not self.exists(item.path):
            return []
        nav_map = _first(self._xml(item.path), "navMap")
        out: list[TocEntry] = []

        def walk(parent) -> None:
            for point in _children(parent, "navPoint"):
                content = _first(point, "content")
                label = _first(point, "navLabel")
                if content is not None and content.get("src"):
                    path, frag = resolve(item.path, content.get("src"))
                    out.append(TocEntry(path, frag, _text(label)))
                walk(point)

        if nav_map is not None:
            walk(nav_map)
        return out

    def _nav_toc(self) -> list[TocEntry]:
        item = next((i for i in self.manifest.values() if "nav" in i.properties), None)
        if item is None or not self.exists(item.path):
            return []
        root = self._xml(item.path)
        navs = [el for el in root.iter() if _local(el.tag) == "nav"]
        toc_nav = next(
            (
                n
                for n in navs
                if "toc" in (n.get(f"{{{_OPS_NS}}}type") or n.get("type") or "").split()
            ),
            navs[0] if navs else None,
        )
        out: list[TocEntry] = []

        def walk(ol) -> None:
            for li in _children(ol, "li"):
                link = next((c for c in li if _local(c.tag) == "a"), None)
                if link is not None and link.get("href"):
                    path, frag = resolve(item.path, link.get("href"))
                    out.append(TocEntry(path, frag, _text(link)))
                for sub in _children(li, "ol"):
                    walk(sub)

        if toc_nav is not None:
            for ol in _children(toc_nav, "ol"):
                walk(ol)
        return out

    # --- cover -------------------------------------------------------------------

    def cover(self) -> tuple[bytes | None, str | None]:
        """The cover image, found the way reading apps find it.

        In order: the EPUB 3 ``cover-image`` manifest property, the EPUB 2
        ``<meta name="cover">`` reference, then any image named like a cover.
        """
        candidates = [i for i in self.manifest.values() if "cover-image" in i.properties]
        for attrs in self._meta_attrs:
            if attrs.get("name") == "cover" and attrs.get("content") in self.manifest:
                candidates.append(self.manifest[attrs["content"]])
        candidates += [
            i
            for i in self.manifest.values()
            if "cover" in i.path.lower().rsplit("/", 1)[-1] or "cover" in i.id.lower()
        ]
        for item in candidates:
            if item.media_type.startswith("image/") and self.exists(item.path):
                return self.read(item.path), item.media_type
        return None, None

    # --- lifecycle -----------------------------------------------------------------

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> EpubFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

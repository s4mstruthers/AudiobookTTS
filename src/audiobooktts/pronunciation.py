"""A user-editable pronunciation lexicon.

Chatterbox is text-only, so the one approach that works is respelling a word
the way it should sound before it reaches the model: "Kizhi" becomes
"Kee-zhee".

Entries live in ``pronunciation.json`` in the data directory and apply to every
render and preview, so a boat named Fereale is said the same way all 135 times.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from pathlib import Path

from audiobooktts.config import LEXICON_PATH
from audiobooktts.fsutil import read_json, write_json

_lock = threading.Lock()
# (file mtime, lexicon, compiled pattern), reloaded when the file changes.
_cache: tuple[int, dict[str, str], re.Pattern | None] | None = None

# Word lists shipped by most Linux and macOS systems. Windows has none, which
# is why suggest() also learns common words from the book itself.
_SYSTEM_WORDLISTS = (Path("/usr/share/dict/words"), Path("/usr/dict/words"))
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z'’-]{2,}\b")
_LOWER_WORD_RE = re.compile(r"\b[a-z][a-z'’-]{2,}\b")
_POSSESSIVE_RE = re.compile(r"['’]s$")


def _compile(lexicon: dict[str, str]) -> re.Pattern | None:
    if not lexicon:
        return None
    # Longest first, so "Fereale's" wins over "Fereale" if both are listed.
    words = sorted(lexicon, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


def _load() -> tuple[dict[str, str], re.Pattern | None]:
    global _cache
    try:
        stamp = LEXICON_PATH.stat().st_mtime_ns
    except OSError:
        with _lock:
            _cache = None
        return {}, None
    with _lock:
        if _cache is not None and _cache[0] == stamp:
            return _cache[1], _cache[2]
    data = read_json(LEXICON_PATH, default={})
    lexicon = {str(k): str(v) for k, v in data.items() if k and v} if isinstance(data, dict) else {}
    pattern = _compile(lexicon)
    with _lock:
        _cache = (stamp, lexicon, pattern)
    return lexicon, pattern


def load_lexicon() -> dict[str, str]:
    """The saved lexicon (a copy; edit it through add() and remove())."""
    return dict(_load()[0])


def apply(text: str, lexicon: dict[str, str] | None = None) -> str:
    """Replace every listed word with its respelling."""
    if lexicon is None:
        lexicon, pattern = _load()
    else:
        pattern = _compile(lexicon)
    if not lexicon or pattern is None:
        return text
    lookup = {k.lower(): v for k, v in lexicon.items()}
    return pattern.sub(lambda m: lookup.get(m.group(0).lower(), m.group(0)), text)


def save_lexicon(lexicon: dict[str, str]) -> None:
    global _cache
    write_json(LEXICON_PATH, dict(sorted(lexicon.items(), key=lambda kv: kv[0].lower())))
    with _lock:
        _cache = None


def add(word: str, say_as: str) -> dict[str, str]:
    word, say_as = word.strip(), say_as.strip()
    if not word or not say_as:
        raise ValueError("Both the word and its respelling are required")
    # Matching is case-insensitive, so replace any entry differing only in case.
    lexicon = {k: v for k, v in load_lexicon().items() if k.lower() != word.lower()}
    lexicon[word] = say_as
    save_lexicon(lexicon)
    return lexicon


def remove(word: str) -> dict[str, str]:
    lexicon = {k: v for k, v in load_lexicon().items() if k.lower() != word.lower()}
    save_lexicon(lexicon)
    return lexicon


def _system_words() -> set[str]:
    for path in _SYSTEM_WORDLISTS:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        return {w.strip().lower() for w in text.split()}
    return set()


def suggest(epub_path: str | Path, limit: int = 40) -> list[tuple[str, int]]:
    """Capitalised words that look like names, most frequent first.

    Proper nouns are where TTS goes wrong, and a book's own names repeat often
    enough that fixing the top of this list covers most of the damage.
    """
    from audiobooktts.epub import parse_epub
    from audiobooktts.textproc import clean_text

    known = _system_words()
    counts: Counter[str] = Counter()
    for chapter in parse_epub(epub_path).chapters:
        # The book's own spelling: respelled text would suggest the respellings.
        text = clean_text(chapter.text, respell=False)
        counts.update(_CAP_WORD_RE.findall(text))
        # A word the book also uses in lower case ("The", "When") is ordinary
        # vocabulary, not a name. This works with no dictionary installed.
        known.update(w.lower() for w in _LOWER_WORD_RE.findall(text))

    listed = {k.lower() for k in load_lexicon()}
    out = []
    for word, n in counts.most_common():
        key = word.lower()
        stem = _POSSESSIVE_RE.sub("", key)
        if key in known or stem in known or key in listed or stem in listed:
            continue
        out.append((word, n))
        if len(out) >= limit:
            break
    return out

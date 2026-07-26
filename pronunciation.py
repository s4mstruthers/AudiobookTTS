"""A user-editable pronunciation lexicon.

Neither engine takes phonemes: Kokoro's front end could, but Chatterbox is
text-only, so the one approach that works for both is respelling the word the
way it should sound before it reaches the model. "Kizhi" becomes "Kee-zhee".

Entries live in ~/.audiobooktts/pronunciation.json and apply to every render
and preview, so a boat named Fereale is said the same way all 135 times.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from audiobooktts.config import APP_DIR

LEXICON_PATH = APP_DIR / "pronunciation.json"

_cache: tuple[float, dict[str, str], re.Pattern | None] | None = None


def _compile(lexicon: dict[str, str]) -> re.Pattern | None:
    if not lexicon:
        return None
    # Longest first, so "Fereale's" wins over "Fereale" if both are listed.
    words = sorted(lexicon, key=len, reverse=True)
    return re.compile(
        r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE
    )


def load_lexicon() -> dict[str, str]:
    """The saved lexicon, reloaded when the file changes."""
    global _cache
    try:
        stamp = LEXICON_PATH.stat().st_mtime
    except OSError:
        _cache = None
        return {}
    if _cache is not None and _cache[0] == stamp:
        return _cache[1]
    try:
        data = json.loads(LEXICON_PATH.read_text())
        lexicon = {str(k): str(v) for k, v in data.items() if k and v}
    except (OSError, ValueError):
        lexicon = {}
    _cache = (stamp, lexicon, _compile(lexicon))
    return lexicon


def _pattern() -> re.Pattern | None:
    load_lexicon()
    return _cache[2] if _cache else None


def apply(text: str, lexicon: dict[str, str] | None = None) -> str:
    """Replace every listed word with its respelling."""
    if lexicon is None:
        lexicon = load_lexicon()
        pattern = _pattern()
    else:
        pattern = _compile(lexicon)
    if not lexicon or pattern is None:
        return text
    lookup = {k.lower(): v for k, v in lexicon.items()}
    return pattern.sub(lambda m: lookup.get(m.group(0).lower(), m.group(0)), text)


def save_lexicon(lexicon: dict[str, str]) -> None:
    global _cache
    APP_DIR.mkdir(parents=True, exist_ok=True)
    LEXICON_PATH.write_text(json.dumps(lexicon, indent=2, sort_keys=True))
    _cache = None


def add(word: str, say_as: str) -> dict[str, str]:
    lexicon = load_lexicon()
    lexicon[word] = say_as
    save_lexicon(lexicon)
    return lexicon


def remove(word: str) -> dict[str, str]:
    lexicon = {k: v for k, v in load_lexicon().items() if k.lower() != word.lower()}
    save_lexicon(lexicon)
    return lexicon


_SYSTEM_WORDS = Path("/usr/share/dict/words")
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z'’-]{2,}\b")


def suggest(epub_path: str | Path, limit: int = 40) -> list[tuple[str, int]]:
    """Capitalised words the dictionary doesn't know, by frequency.

    Proper nouns are where TTS goes wrong, and a book's own names repeat often
    enough that fixing the top of this list covers most of the damage.
    """
    from collections import Counter

    from audiobooktts.epub import parse_epub
    from audiobooktts.textproc import clean_text

    try:
        known = {w.strip().lower() for w in
                 _SYSTEM_WORDS.read_text(encoding="utf-8", errors="ignore").split()}
    except OSError:
        known = set()

    counts: Counter[str] = Counter()
    for chapter in parse_epub(epub_path).chapters:
        for word in _CAP_WORD_RE.findall(clean_text(chapter.text)):
            counts[word] += 1

    lexicon = {k.lower() for k in load_lexicon()}
    out = [
        (w, n) for w, n in counts.most_common()
        if w.lower() not in known and w.lower() not in lexicon
        and not w.lower().rstrip("'s") in known
    ]
    return out[:limit]

"""Rule-based text normalization for narration, plus sentence-aware chunking."""

from __future__ import annotations

import re

# Kokoro quality degrades on long inputs; keep chunks comfortably short.
MAX_CHUNK_CHARS = 450

_ABBREVIATIONS = {
    r"\bMr\.": "Mister",
    r"\bMrs\.": "Missus",
    r"\bMs\.": "Miz",
    r"\bDr\.": "Doctor",
    r"\bProf\.": "Professor",
    r"\bSt\.": "Saint",
    r"\bvs\.": "versus",
    r"\betc\.": "et cetera",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\bapprox\.": "approximately",
}

_ROMAN_RE = re.compile(r"^(?:chapter\s+)?([ivxlcdm]+)\.?$", re.IGNORECASE)
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

# A sentence boundary is terminal punctuation, any closing quotes/brackets, then
# whitespace — but only when what follows starts a new sentence. Requiring an
# uppercase/digit start keeps dialogue with a speech tag intact: in
# '"Where?" she asked.' the lowercase "she" means the sentence continues.
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?…])[\"'”’)\]]*\s+(?=[\"'“‘(\[]*[A-Z0-9])"
)


def roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(c not in _ROMAN_VALUES for c in s):
        return None
    total = 0
    for i, c in enumerate(s):
        v = _ROMAN_VALUES[c]
        if i + 1 < len(s) and _ROMAN_VALUES[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


def normalize_chapter_title(title: str) -> str:
    """'Chapter XII' -> 'Chapter 12'; bare 'IV' -> 'Chapter 4'."""
    m = _ROMAN_RE.match(title.strip())
    if m:
        n = roman_to_int(m.group(1))
        if n:
            return f"Chapter {n}"
    return title.strip()


def clean_text(text: str) -> str:
    # Drop citation/footnote markers like [12] or [note 3]
    text = re.sub(r"\[\s*(?:note\s*)?\d+\s*\]", "", text, flags=re.IGNORECASE)
    # Normalize unicode punctuation that some TTS front-ends mangle
    text = text.replace(" ", " ")
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = text.replace("—", " — ").replace("–", " – ")
    text = text.replace("…", "...")
    for pattern, repl in _ABBREVIATIONS.items():
        text = re.sub(pattern, repl, text)
    # Collapse whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split into sentences without dropping any characters.

    Slicing at match ends rather than re.split keeps closing quotation marks
    attached to the sentence they belong to.
    """
    out: list[str] = []
    prev = 0
    for m in _SENTENCE_END_RE.finditer(text):
        piece = text[prev : m.end()].strip()
        if piece:
            out.append(piece)
        prev = m.end()
    tail = text[prev:].strip()
    if tail:
        out.append(tail)
    return out


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks of at most max_chars, never mid-sentence.

    Paragraph breaks are preferred split points; a single sentence longer
    than max_chars is split at commas/semicolons as a last resort.
    """
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in split_sentences(paragraph):
            if len(sentence) > max_chars:
                flush()
                # Hard case: break an overlong sentence at clause boundaries
                piece = ""
                for clause in re.split(r"(?<=[,;:])\s+", sentence):
                    if piece and len(piece) + len(clause) + 1 > max_chars:
                        chunks.append(piece.strip())
                        piece = clause
                    else:
                        piece = f"{piece} {clause}".strip()
                if piece.strip():
                    chunks.append(piece.strip())
                continue
            if current and len(current) + len(sentence) + 1 > max_chars:
                flush()
            current = f"{current} {sentence}".strip()
        # Prefer chunk boundaries at paragraph ends: flush if fairly full
        if len(current) > max_chars * 0.6:
            flush()
    flush()
    return chunks

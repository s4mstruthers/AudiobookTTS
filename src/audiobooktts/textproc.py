"""Rule-based text normalisation for narration, plus sentence-aware chunking."""

from __future__ import annotations

import re
import textwrap

from audiobooktts import pronunciation
from audiobooktts.numerals import expand as expand_numbers

# Chatterbox destabilises on long inputs — at 900 characters it collapsed to a
# third of the expected length — so no chunk may exceed this.
MAX_CHUNK_CHARS = 450

# What follows a chunk, and therefore how long the pause after it should be.
CLAUSE, SENTENCE, PARAGRAPH = "clause", "sentence", "paragraph"
SYNTHESIS_UNITS = ("paragraph", "sentence")

_ABBREVIATIONS = [
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"\bMr\.", "Mister"),
        (r"\bMrs\.", "Missus"),
        (r"\bMs\.", "Miz"),
        (r"\bDr\.", "Doctor"),
        (r"\bProf\.", "Professor"),
        (r"\bSt\.", "Saint"),
        (r"\bvs\.", "versus"),
        (r"\betc\.", "et cetera"),
        (r"\be\.g\.", "for example"),
        (r"\bi\.e\.", "that is"),
        (r"\bapprox\.", "approximately"),
    )
]
_FOOTNOTE_RE = re.compile(r"\[\s*(?:note\s*)?\d+\s*\]", re.IGNORECASE)

# Novels render small caps as ALL CAPS, and autoregressive engines like
# Chatterbox destabilise on them — measured 26% longer and 2.5x more variable
# in duration on the same sentence.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’]*")
MIN_CAPS_WORD = 5  # shorter all-caps tokens are usually acronyms (BBC, NASA)
# Kept as-is even when long, since they are read as letters or as themselves.
_KEEP_UPPER = {"OK", "TV", "UK", "US", "USA", "BBC", "DVD", "NASA", "AM", "PM"}
# Inside a run of caps (usually a proper name) these read better lowercased.
_RUN_STOPWORDS = {"and", "of", "the", "in", "for", "to", "a", "an", "on", "at"}

_CHAPTER_NUMERAL_RE = re.compile(r"^(chapter\s+)?([ivxlcdm]+)\.?$", re.IGNORECASE)
# Only well-formed numerals count, so titles such as "Civil", "Vivid" or "Did",
# which happen to be spelt with numeral letters, are left as words.
_CANONICAL_ROMAN_RE = re.compile(r"^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")
# A bare numeral heading beyond this is more likely a word ("MIX") than a chapter.
_MAX_BARE_CHAPTER = 200
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

# A sentence boundary is terminal punctuation, any closing quotes/brackets, then
# whitespace — but only when what follows starts a new sentence. Requiring an
# uppercase/digit start keeps dialogue with a speech tag intact: in
# '"Where?" she asked.' the lowercase "she" means the sentence continues.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+(?=[\"'“‘(\[]*[A-Z0-9])")
_CLAUSE_BREAK_RE = re.compile(r"(?<=[,;:])\s+")


def roman_to_int(s: str) -> int | None:
    """Value of a well-formed Roman numeral (any case), else None."""
    if not s or not _CANONICAL_ROMAN_RE.match(s.upper()):
        return None
    s = s.lower()
    total = 0
    for i, c in enumerate(s):
        v = _ROMAN_VALUES[c]
        if i + 1 < len(s) and _ROMAN_VALUES[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total or None


def normalize_chapter_title(title: str) -> str:
    """'Chapter XII' -> 'Chapter 12'; a bare 'IV' -> 'Chapter 4'."""
    title = title.strip()
    m = _CHAPTER_NUMERAL_RE.match(title)
    if m:
        prefix, numeral = m.groups()
        n = roman_to_int(numeral)
        # A bare numeral must be written in one case: "Mix" is a word, "MIX" or
        # "mix" could be a numeral, but only a plausible chapter number counts.
        uniform = numeral.isupper() or numeral.islower()
        if n and (prefix or (uniform and n <= _MAX_BARE_CHAPTER)):
            return f"Chapter {n}"
    return title


def normalize_caps(text: str, min_len: int = MIN_CAPS_WORD) -> str:
    """Convert small-caps styling to ordinary case, leaving acronyms alone.

    A word is converted when it sits in a run of two or more consecutive
    all-caps words (a proper name like CENTRAL LONDON HATCHERY), or when it is
    long enough on its own to be styling rather than an acronym.
    """
    words = list(_WORD_RE.finditer(text))
    upper = {i for i, m in enumerate(words) if len(m.group()) > 1 and m.group().isupper()}

    # Indices belonging to a run of 2+ adjacent all-caps words.
    in_run: set[int] = set()
    run: list[int] = []
    for i in range(len(words) + 1):
        if i in upper:
            run.append(i)
            continue
        if len(run) >= 2:
            in_run.update(run)
        run = []

    pieces: list[str] = []
    last = 0
    for i, m in enumerate(words):
        w = m.group()
        if i not in upper or w in _KEEP_UPPER:
            continue
        if i not in in_run and len(w) < min_len:
            continue
        if i in in_run and w.lower() in _RUN_STOPWORDS:
            replacement = w.lower()
        else:
            replacement = w[0] + w[1:].lower()
        pieces.append(text[last : m.start()])
        pieces.append(replacement)
        last = m.end()
    pieces.append(text[last:])
    return "".join(pieces)


def clean_text(text: str, respell: bool = True) -> str:
    """Normalise for narration.

    respell=False keeps the book's own spelling, for anything shown on screen:
    the pronunciation lexicon rewrites "Inge" to "Inga" for the engine's
    benefit, which is alarming to read back as if the text had changed.
    """
    text = _FOOTNOTE_RE.sub("", text)
    # Normalise unicode punctuation that some TTS front-ends mangle.
    text = text.replace(" ", " ")
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = text.replace("—", " — ").replace("–", " – ")
    text = text.replace("…", "...")
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = normalize_caps(text)
    # Numerals before names: Chatterbox is text-only and mangles compact forms,
    # rendering "2nd July" as "tussen july".
    text = expand_numbers(text)
    # Respell names the engine mispronounces. Applied here so the preview and
    # the render — which share this function — can never disagree.
    if respell:
        text = pronunciation.apply(text)
    # Collapse whitespace, keeping paragraph breaks but removing newlines from
    # *within* a paragraph. Many epubs are hard-wrapped mid-sentence, and a
    # stray newline puts a falling cadence and a gap in the middle of a sentence.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    paragraphs = re.split(r"\n{2,}", text)
    text = "\n\n".join(p.replace("\n", " ") for p in paragraphs)
    text = re.sub(r"[ \t]+", " ", text)
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


def _split_long_sentence(sentence: str, max_chars: int) -> list[list[str]]:
    """Break one over-long sentence into [text, kind] pieces within max_chars.

    Splits at clause punctuation first, so the pauses fall where a reader would
    breathe. A clause that is still too long (no commas at all) is wrapped at
    word boundaries as a last resort, which guarantees the limit is kept.
    """
    parts: list[str] = []
    for clause in _CLAUSE_BREAK_RE.split(sentence):
        if len(clause) <= max_chars:
            parts.append(clause)
        else:
            parts.extend(
                textwrap.wrap(clause, max_chars, break_long_words=True, break_on_hyphens=False)
            )

    pieces: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + 1 + len(part) > max_chars:
            pieces.append(current)
            current = part
        else:
            current = f"{current} {part}" if current else part
    if current:
        pieces.append(current)
    # Mid-sentence joins get the short clause pause; the end is a sentence end.
    return [[p, CLAUSE] for p in pieces[:-1]] + [[pieces[-1], SENTENCE]]


def chunk_paragraphs(
    text: str, max_chars: int = MAX_CHUNK_CHARS, unit: str = "paragraph"
) -> list[tuple[str, str]]:
    """Split text into (chunk, break_kind) pairs no longer than max_chars.

    unit="paragraph" packs whole sentences into each chunk, up to a paragraph:
    read in one call a paragraph keeps its own flow, whereas sentences
    generated separately start cold and sound disconnected.

    unit="sentence" gives every sentence its own call, so the caller controls
    every pause exactly.

    The break kind matters: an overlong sentence has to be split at a comma or
    semicolon, and pausing there as long as at a full stop breaks the sentence
    in half to the ear.
    """
    if unit not in SYNTHESIS_UNITS:
        raise ValueError(f"unit must be one of {SYNTHESIS_UNITS}, not {unit!r}")
    out: list[list[str]] = []

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        first = len(out)

        if unit == "sentence":
            for sentence in split_sentences(paragraph):
                if len(sentence) <= max_chars:
                    out.append([sentence, SENTENCE])
                else:
                    out.extend(_split_long_sentence(sentence, max_chars))
        else:
            group = ""
            for sentence in split_sentences(paragraph):
                if len(sentence) > max_chars:
                    if group:
                        out.append([group, SENTENCE])
                    pieces = _split_long_sentence(sentence, max_chars)
                    out.extend(pieces[:-1])
                    # Later sentences may still share a call with the tail.
                    group = pieces[-1][0]
                elif group and len(group) + 1 + len(sentence) > max_chars:
                    out.append([group, SENTENCE])
                    group = sentence
                else:
                    group = f"{group} {sentence}" if group else sentence
            if group:
                out.append([group, SENTENCE])

        if len(out) > first:
            out[-1][1] = PARAGRAPH
    return [(t, k) for t, k in out]


def preview_chunks(
    text: str, max_chars: int = 500, unit: str = "paragraph"
) -> list[tuple[str, str]]:
    """The chunks a preview covers, so it can render exactly like the real run."""
    picked: list[tuple[str, str]] = []
    total = 0
    # Chunk to the sample budget, not the synthesis budget, or a single 450-char
    # chunk would blow past a smaller max_chars.
    for chunk, kind in chunk_paragraphs(text, max_chars=min(MAX_CHUNK_CHARS, max_chars), unit=unit):
        if picked and total + len(chunk) + 1 > max_chars:
            break
        picked.append((chunk, kind))
        total += len(chunk) + 1
    return picked


def preview_sample(text: str, max_chars: int = 500) -> str:
    """A short excerpt for auditioning a voice, ending on a sentence.

    Slicing the raw text instead would cut mid-word and make every voice sound
    like it had been interrupted.
    """
    return " ".join(c for c, _ in preview_chunks(text, max_chars)) or text[:max_chars]

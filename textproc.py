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

# Novels render small caps as ALL CAPS. Kokoro phonemises those identically to
# title case, but autoregressive engines like Chatterbox destabilise on them —
# measured 26% longer and 2.5x more variable in duration on the same sentence.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’]*")
MIN_CAPS_WORD = 5  # shorter all-caps tokens are usually acronyms (BBC, NASA)
# Kept as-is even when long, since they are read as letters or as themselves.
_KEEP_UPPER = {"OK", "TV", "UK", "US", "USA", "BBC", "DVD", "NASA", "AM", "PM"}
# Inside a run of caps (usually a proper name) these read better lowercased.
_RUN_STOPWORDS = {"and", "of", "the", "in", "for", "to", "a", "an", "on", "at"}

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


def normalize_caps(text: str, min_len: int = MIN_CAPS_WORD) -> str:
    """Convert small-caps styling to ordinary case, leaving acronyms alone.

    A word is converted when it sits in a run of two or more consecutive
    all-caps words (a proper name like CENTRAL LONDON HATCHERY), or when it is
    long enough on its own to be styling rather than an acronym.
    """
    words = list(_WORD_RE.finditer(text))
    upper = [
        i for i, m in enumerate(words)
        if len(m.group()) > 1 and m.group().isupper()
    ]
    upper_set = set(upper)

    # Indices belonging to a run of 2+ adjacent all-caps words.
    in_run: set[int] = set()
    run: list[int] = []
    for i in range(len(words)):
        if i in upper_set:
            run.append(i)
        else:
            if len(run) >= 2:
                in_run.update(run)
            run = []
    if len(run) >= 2:
        in_run.update(run)

    pieces: list[str] = []
    last = 0
    for i, m in enumerate(words):
        w = m.group()
        if i not in upper_set or w in _KEEP_UPPER:
            continue
        if i not in in_run and len(w) < min_len:
            continue
        if i in in_run and w.lower() in _RUN_STOPWORDS:
            replacement = w.lower()
        else:
            replacement = w[0] + w[1:].lower()
        pieces.append(text[last:m.start()])
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
    text = normalize_caps(text)
    # Numerals before names: Chatterbox is text-only and mangles compact forms,
    # rendering "2nd July" as "tussen july".
    from audiobooktts.numbers import expand as expand_numbers

    text = expand_numbers(text)
    # Respell names the engines mispronounce. Applied here so the preview and
    # the render — which share this function — can never disagree.
    if respell:
        from audiobooktts import pronunciation

        text = pronunciation.apply(text)
    # Collapse whitespace, keeping paragraph breaks but removing newlines from
    # *within* a paragraph. Many epubs are hard-wrapped mid-sentence, and Kokoro
    # splits its input on newlines and renders each fragment as its own
    # utterance — so a stray newline puts a falling cadence and a gap in the
    # middle of a sentence.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\x00", text)  # protect paragraph breaks
    text = text.replace("\n", " ")
    text = text.replace("\x00", "\n\n")
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


# What follows a chunk, and therefore how long the pause after it should be.
CLAUSE, SENTENCE, PARAGRAPH = "clause", "sentence", "paragraph"


def chunk_paragraphs(
    text: str, max_chars: int = MAX_CHUNK_CHARS, unit: str = "paragraph"
) -> list[tuple[str, str]]:
    """Split into (chunk, break_kind) pairs, one sentence per chunk.

    Sentences are synthesised separately so the caller controls every pause.
    Left inside one call the engine supplies its own sentence gap, measured at
    about 0.13s — far too brisk for narration, and not adjustable.

    The break kind matters: an overlong sentence has to be split at a comma or
    semicolon, and pausing there as long as at a full stop breaks the sentence
    in half to the ear.
    """
    out: list[list] = []

    if unit == "paragraph":
        # A paragraph read in one go keeps its own flow; sentences generated
        # separately start cold and sound disconnected. Long paragraphs are
        # still broken up, because the engine destabilises past roughly 450
        # characters — at 900 it collapsed to a third of the expected length.
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            group = ""
            for sentence in split_sentences(paragraph):
                if group and len(group) + len(sentence) + 1 > max_chars:
                    out.append([group.strip(), SENTENCE])
                    group = sentence
                elif len(sentence) > max_chars:
                    if group:
                        out.append([group.strip(), SENTENCE])
                        group = ""
                    piece = ""
                    for clause in re.split(r"(?<=[,;:])\s+", sentence):
                        if piece and len(piece) + len(clause) + 1 > max_chars:
                            out.append([piece.strip(), CLAUSE])
                            piece = clause
                        else:
                            piece = f"{piece} {clause}".strip()
                    group = piece
                else:
                    group = f"{group} {sentence}".strip()
            if group.strip():
                out.append([group.strip(), SENTENCE])
            if out:
                out[-1][1] = PARAGRAPH
        return [(t, k) for t, k in out]

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in split_sentences(paragraph):
            if len(sentence) <= max_chars:
                out.append([sentence, SENTENCE])
                continue
            piece = ""
            for clause in re.split(r"(?<=[,;:])\s+", sentence):
                if piece and len(piece) + len(clause) + 1 > max_chars:
                    out.append([piece.strip(), CLAUSE])  # mid-sentence
                    piece = clause
                else:
                    piece = f"{piece} {clause}".strip()
            if piece.strip():
                out.append([piece.strip(), SENTENCE])  # the sentence ends here
        if out:
            out[-1][1] = PARAGRAPH
    return [(t, k) for t, k in out]


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Chunk texts only, for callers that don't care about paragraph breaks."""
    return [c for c, _ in chunk_paragraphs(text, max_chars)]


def preview_sample(text: str, max_chars: int = 500) -> str:
    """A short excerpt for auditioning a voice, ending on a sentence.

    Slicing the raw text instead would cut mid-word and make every voice sound
    like it had been interrupted.
    """
    return " ".join(c for c, _ in preview_chunks(text, max_chars)) or text[:max_chars]


def preview_chunks(
    text: str, max_chars: int = 500, unit: str = "paragraph"
) -> list[tuple[str, str]]:
    """The chunks a preview covers, so it can render exactly like the real run."""
    picked: list[tuple[str, bool]] = []
    total = 0
    # Chunk to the sample budget, not the synthesis budget, or a single 450-char
    # chunk would blow past a smaller max_chars.
    for chunk, ends in chunk_paragraphs(
        text, max_chars=min(MAX_CHUNK_CHARS, max_chars), unit=unit
    ):
        if picked and total + len(chunk) + 1 > max_chars:
            break
        picked.append((chunk, ends))
        total += len(chunk) + 1
    return picked

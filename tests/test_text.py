"""Text normalisation, numerals, chunking and the pronunciation lexicon."""

from __future__ import annotations

import pytest

from audiobooktts import pronunciation
from audiobooktts.numerals import expand
from audiobooktts.textproc import (
    CLAUSE,
    MAX_CHUNK_CHARS,
    PARAGRAPH,
    SENTENCE,
    chunk_paragraphs,
    clean_text,
    normalize_caps,
    normalize_chapter_title,
    preview_chunks,
    roman_to_int,
    split_sentences,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("They marched 10,000 men.", "They marched ten thousand men."),
        ("It held 1,500,000 coins.", "It held one million five hundred thousand coins."),
        ("We met at 6 p.m. The end.", "We met at six p.m. The end."),
        ("We met at 6pm. Then left.", "We met at six p.m. Then left."),
        ("At 6.05am and 6:30 PM.", "At six oh five a.m. and six thirty p.m."),
        ("They had to march 3 miles.", "They had to march three miles."),
        ("I may 2 times.", "I may two times."),
        ("On the 2nd of July 1895.", "On the second of July eighteen ninety-five."),
        ("On 2nd July, and July 4th.", "On second of July, and July the fourth."),
        ("July 2, 2000.", "July the second, two thousand."),
        ("From 1989-1990.", "From nineteen eighty-nine to nineteen ninety."),
        ("In 1992 a crew of 1400.", "In nineteen ninety-two a crew of one thousand four hundred."),
        ("Ref 00412.", "Ref zero zero four one two."),
        ("It cost $5.99.", "It cost five dollars ninety-nine."),
        ("Only £1 and $0.50.", "Only one pound and fifty cents."),
        ("A $1.5 million deal.", "A one point five million dollars deal."),
        ("Up 50% and 12.5%.", "Up fifty percent and twelve point five percent."),
        ("Pi is 3.14.", "Pi is three point one four."),
        ("The 24th man.", "The twenty-fourth man."),
    ],
)
def test_expand_numerals(text, expected):
    assert expand(text) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Chapter XII", "Chapter 12"),
        ("IV", "Chapter 4"),
        ("iv.", "Chapter 4"),
        ("LIV", "Chapter 54"),
        # Words spelt with numeral letters are not numerals.
        ("Mix", "Mix"),
        ("MIX", "MIX"),
        ("Civil", "Civil"),
        ("Vivid", "Vivid"),
        ("Did", "Did"),
        ("Prologue", "Prologue"),
    ],
)
def test_normalize_chapter_title(title, expected):
    assert normalize_chapter_title(title) == expected


def test_roman_to_int_rejects_malformed():
    assert roman_to_int("XIV") == 14
    assert roman_to_int("IIII") is None
    assert roman_to_int("IC") is None
    assert roman_to_int("") is None


def test_normalize_caps_keeps_acronyms():
    assert normalize_caps("The CENTRAL LONDON HATCHERY") == "The Central London Hatchery"
    assert normalize_caps("She worked for NASA and the BBC") == "She worked for NASA and the BBC"
    assert normalize_caps("It was ENORMOUS") == "It was Enormous"


def test_clean_text_normalises_for_narration():
    text = "Mr. Smith said “hi”[12] —\nand left…\n\n\nNew para."
    assert clean_text(text) == 'Mister Smith said "hi" — and left...\n\nNew para.'


def test_split_sentences_keeps_speech_tags_together():
    assert split_sentences('"Where?" she asked. He left.') == ['"Where?" she asked.', "He left."]


def _long_sentence(words: int = 100) -> str:
    return "This " + ", ".join(["clause is getting longer"] * (words // 4)) + "."


@pytest.mark.parametrize("unit", ["paragraph", "sentence"])
def test_no_chunk_exceeds_the_limit(unit):
    # Regression: a long sentence after a short one was emitted unsplit.
    text = f"Short one. {_long_sentence()} After. " + "Word " * 150 + "end.\n\nSecond. Two."
    chunks = chunk_paragraphs(text, unit=unit)
    assert max(len(c) for c, _ in chunks) <= MAX_CHUNK_CHARS
    # Nothing is lost or duplicated.
    assert " ".join(c for c, _ in chunks).split() == text.split()


def test_chunk_break_kinds():
    text = f"{_long_sentence()} Then more.\n\nNext paragraph."
    chunks = chunk_paragraphs(text, unit="sentence")
    kinds = [k for _, k in chunks]
    assert kinds[0] == CLAUSE  # a split mid-sentence gets the short pause
    assert kinds[-1] == PARAGRAPH
    assert PARAGRAPH in kinds[:-1] and SENTENCE in kinds


def test_paragraph_unit_groups_sentences():
    chunks = chunk_paragraphs("One. Two. Three.\n\nFour.", unit="paragraph")
    assert chunks == [("One. Two. Three.", PARAGRAPH), ("Four.", PARAGRAPH)]


def test_chunk_rejects_unknown_unit():
    with pytest.raises(ValueError):
        chunk_paragraphs("Hi.", unit="word")


def test_preview_chunks_respect_budget():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    picked = preview_chunks(text, max_chars=200)
    assert sum(len(c) + 1 for c, _ in picked) <= 200 + 1


def test_pronunciation_lexicon_round_trip():
    pronunciation.add("Kizhi", "Kee-zhee")
    pronunciation.add("kizhi", "Kizzy")  # same word, different case: replaced
    assert pronunciation.load_lexicon() == {"kizhi": "Kizzy"}
    assert pronunciation.apply("To Kizhi and KIZHI.") == "To Kizzy and Kizzy."
    assert "Kizzy" in clean_text("Off to Kizhi.")
    assert "Kizhi" in clean_text("Off to Kizhi.", respell=False)
    pronunciation.remove("KIZHI")
    assert pronunciation.load_lexicon() == {}


def test_pronunciation_file_is_utf8(app_home):
    pronunciation.add("Zoë", "Zo-ee")
    raw = (app_home / "pronunciation.json").read_bytes()
    assert "Zoë".encode() in raw


def test_suggest_finds_names_not_common_words(epub_path):
    pronunciation.add("Holmes", "Homes")
    words = dict(pronunciation.suggest(epub_path))
    assert "Holmes" not in words  # already in the lexicon
    # "The" also appears in lower case in the book, so it is ordinary vocabulary
    # even where no system dictionary exists (as on Windows).
    assert "The" not in words
    assert "Fereale" in words  # an invented name
    # Respellings are never suggested back.
    assert "Homes" not in words

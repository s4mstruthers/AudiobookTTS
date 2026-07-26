"""Write numbers out in words before they reach the engine.

Chatterbox is text-only and has to guess at compact forms: given "2nd July" it
produced "tussen july". Kokoro's front end copes better, but expanding here
means both engines read the same thing.

Dates use British order — "2nd July" becomes "second of July" — since that is
how the text is written.
"""

from __future__ import annotations

import re

from num2words import num2words

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

# A bare four-digit number in prose is nearly always a year. Below this,
# quantities are more likely than dates.
YEAR_MIN, YEAR_MAX = 1100, 2099
# Longer runs are reference numbers, not quantities: read them digit by digit.
MAX_CARDINAL_DIGITS = 4


def _ordinal(n: int) -> str:
    return num2words(n, to="ordinal")


def _year(n: int) -> str:
    return num2words(n, to="year")


def _cardinal(n: int) -> str:
    # num2words puts commas in large numbers, which read as pauses.
    return num2words(n).replace(",", "")


def _digits(s: str) -> str:
    return " ".join(_cardinal(int(d)) for d in s)


# Four digits can be a year or a quantity: "born in 1992" against "1500 roubles".
# Only the words that can precede a date decide it.
# Bare "of" is excluded on purpose: "summer of 1992" is a date but "a crew of
# 1400" is a quantity, so the cue has to be the word before it.
_YEAR_CUE_RE = re.compile(
    r"(?:\b(?:in|since|by|from|until|till|during|around|about|circa|after|"
    r"before|summer|winter|spring|autumn|year|early|late)\b(?:\s+of)?"
    r"|[,–—-])\s*$",
    re.IGNORECASE,
)


def _looks_like_a_year(text: str, start: int) -> bool:
    return bool(_YEAR_CUE_RE.search(text[max(0, start - 24):start]))


def _time(match: re.Match) -> str:
    hour, minute, meridiem = match.group(1), match.group(2), match.group(3).lower()
    said = _cardinal(int(hour))
    if minute:
        m = int(minute)
        # "6.05" is "six oh five"; "6.30" is "six thirty".
        said += " oh " + _cardinal(m) if 0 < m < 10 else (
            "" if m == 0 else " " + _cardinal(m)
        )
    return f"{said} {meridiem[0]}.{meridiem[1]}."


def expand(text: str) -> str:
    """Rewrite numerals, dates and times as words."""
    # Times first: 6am, 6.30pm, 6:30 pm
    text = re.sub(
        r"\b(\d{1,2})(?:[:.](\d{2}))?\s?([ap]\.?m\.?)\b",
        _time, text, flags=re.IGNORECASE,
    )

    # Year ranges: 1989-1990
    text = re.sub(
        rf"\b({YEAR_MIN//1*1}\d{{0,3}}|\d{{4}})\s?[–—-]\s?(\d{{4}})\b",
        lambda m: (f"{_year(int(m.group(1)))} to {_year(int(m.group(2)))}"
                   if YEAR_MIN <= int(m.group(1)) <= YEAR_MAX
                   and YEAR_MIN <= int(m.group(2)) <= YEAR_MAX
                   else m.group(0)),
        text,
    )

    # "2nd July" -> "second of July"
    text = re.sub(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)\s+({MONTHS})\b",
        lambda m: f"{_ordinal(int(m.group(1)))} of {m.group(2)}",
        text, flags=re.IGNORECASE,
    )
    # "July 2nd" -> "July the second"
    text = re.sub(
        rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)\b",
        lambda m: f"{m.group(1)} the {_ordinal(int(m.group(2)))}",
        text, flags=re.IGNORECASE,
    )
    # "July 2, 2000" -> "July the second, two thousand"
    text = re.sub(
        rf"\b({MONTHS})\s+(\d{{1,2}})\b(?!\s*(?:st|nd|rd|th))",
        lambda m: f"{m.group(1)} the {_ordinal(int(m.group(2)))}",
        text, flags=re.IGNORECASE,
    )

    # Any remaining ordinal: 24th -> twenty-fourth
    text = re.sub(
        r"\b(\d{1,3})(?:st|nd|rd|th)\b",
        lambda m: _ordinal(int(m.group(1))), text, flags=re.IGNORECASE,
    )

    # Decimals: 9.6 -> nine point six
    text = re.sub(
        r"\b(\d+)\.(\d+)\b",
        lambda m: f"{_cardinal(int(m.group(1)))} point {_digits(m.group(2))}",
        text,
    )

    def number(match: re.Match) -> str:
        raw = match.group(0)
        value = int(raw)
        if len(raw) > MAX_CARDINAL_DIGITS or raw.startswith("0"):
            return _digits(raw)
        if (
            len(raw) == 4
            and YEAR_MIN <= value <= YEAR_MAX
            and _looks_like_a_year(text, match.start())
        ):
            return _year(value)
        return _cardinal(value)

    text = re.sub(r"\b\d+\b", number, text)
    # "11pm." would otherwise end up as "eleven p.m..". Scoped to the meridiem
    # so an ellipsis is left intact.
    return re.sub(r"([ap]\.m)\.\.", r"\1.", text)

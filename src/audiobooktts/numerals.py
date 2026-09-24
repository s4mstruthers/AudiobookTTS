"""Write numbers out in words before they reach the engine.

Chatterbox is text-only and has to guess at compact forms: given "2nd July" it
produced "tussen july". Expanding numerals here means the model only ever sees
words, and the preview and the render read identically.

Dates use British order — "2nd July" becomes "second of July" — since that is
how the text is written.
"""

from __future__ import annotations

import re

from num2words import num2words

# Month names must be capitalised: in prose "march" and "may" are far more
# often a verb than a date ("they had to march 3 miles").
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_ORDINAL_SUFFIX = r"(?:st|nd|rd|th|ST|ND|RD|TH)"

# A bare four-digit number in prose is nearly always a year. Below this,
# quantities are more likely than dates.
YEAR_MIN, YEAR_MAX = 1100, 2099
# Longer runs are reference numbers, not quantities: read them digit by digit.
MAX_CARDINAL_DIGITS = 4

_CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
}
_SCALES = r"(?:thousand|million|billion|trillion)"

# Four digits can be a year or a quantity: "born in 1992" against "1500 roubles".
# Only the words that can precede a date decide it.
# Bare "of" is excluded on purpose: "summer of 1992" is a date but "a crew of
# 1400" is a quantity, so the cue has to be the word before it.
_YEAR_CUE_RE = re.compile(
    r"(?:\b(?:in|since|by|from|until|till|during|around|about|circa|after|"
    r"before|summer|winter|spring|autumn|year|early|late)\b(?:\s+of)?"
    r"|\b(?:" + MONTHS + r")\b"  # "2nd July 1895", "July 1895"
    r"|[,–—-])\s*$",
    re.IGNORECASE,
)

_TIME_RE = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s?([ap])\.?m\b\.?", re.IGNORECASE)
_CURRENCY_RE = re.compile(
    r"([$£€])\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?:\s+(" + _SCALES + r"))?\b"
)
_PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s?%")
_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_YEAR_RANGE_RE = re.compile(r"\b(\d{4})\s?[–—-]\s?(\d{4})\b")
_DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}}){_ORDINAL_SUFFIX}\s+(?:of\s+)?({MONTHS})\b")
_MONTH_DAY_RE = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}})(?:{_ORDINAL_SUFFIX})?\b")
_ORDINAL_RE = re.compile(rf"\b(\d{{1,3}}){_ORDINAL_SUFFIX}\b")
_DECIMAL_RE = re.compile(r"\b(\d+)\.(\d+)\b")
_INTEGER_RE = re.compile(r"\b\d+\b")


def _ordinal(n: int) -> str:
    return num2words(n, to="ordinal")


def _year(n: int) -> str:
    return num2words(n, to="year")


def _cardinal(n: int) -> str:
    # num2words puts commas in large numbers, which read as pauses.
    return num2words(n).replace(",", "")


def _digits(s: str) -> str:
    return " ".join(_cardinal(int(d)) for d in s)


def _is_year(n: int) -> bool:
    return YEAR_MIN <= n <= YEAR_MAX


def _looks_like_a_year(text: str, start: int) -> bool:
    return bool(_YEAR_CUE_RE.search(text[max(0, start - 24) : start]))


def _time(match: re.Match) -> str:
    hour, minute, meridiem = match.group(1), match.group(2), match.group(3).lower()
    said = _cardinal(int(hour))
    if minute:
        m = int(minute)
        # "6.05" is "six oh five"; "6.30" is "six thirty"; "6.00" is just "six".
        if 0 < m < 10:
            said += " oh " + _cardinal(m)
        elif m:
            said += " " + _cardinal(m)
    return f"{said} {meridiem}.m."


def _currency(match: re.Match) -> str:
    symbol, whole, fraction, scale = match.groups()
    one, many, sub_one, sub_many = _CURRENCIES[symbol]
    amount = int(whole.replace(",", ""))
    if scale:
        # "$5 million" is five million dollars; "$2.5 million" two point five.
        number = _cardinal(amount)
        if fraction:
            number += " point " + _digits(fraction)
        return f"{number} {scale} {many}"
    said = f"{_cardinal(amount)} {one if amount == 1 else many}"
    if fraction:
        cents = int(fraction.ljust(2, "0"))
        if amount == 0:
            return f"{_cardinal(cents)} {sub_one if cents == 1 else sub_many}"
        if cents:
            said += f" {_cardinal(cents)}"
    return said


def _decimal(match: re.Match) -> str:
    return f"{_cardinal(int(match.group(1)))} point {_digits(match.group(2))}"


def _percent(match: re.Match) -> str:
    raw = match.group(1)
    if "." in raw:
        whole, frac = raw.split(".", 1)
        return f"{_cardinal(int(whole))} point {_digits(frac)} percent"
    return f"{_cardinal(int(raw))} percent"


def _year_range(match: re.Match) -> str:
    a, b = int(match.group(1)), int(match.group(2))
    # The start may be early ("1066-1087"); the order is what marks a range.
    if 1000 <= a < b <= YEAR_MAX:
        return f"{_year(a)} to {_year(b)}"
    return match.group(0)


def expand(text: str) -> str:
    """Rewrite numerals, dates, times, money and percentages as words."""
    # Times first, before "6.30" can be read as a decimal: 6am, 6.30pm, 6:30 p.m.
    text = _TIME_RE.sub(_time, text)
    # Money and percentages before the separators and decimals they contain.
    text = _CURRENCY_RE.sub(_currency, text)
    text = _PERCENT_RE.sub(_percent, text)
    # "10,000" is one number. Left alone it became "ten,zero zero zero".
    text = _THOUSANDS_RE.sub(lambda m: _cardinal(int(m.group(0).replace(",", ""))), text)

    # Year ranges: 1989-1990 -> "nineteen eighty-nine to nineteen ninety"
    text = _YEAR_RANGE_RE.sub(_year_range, text)
    # "2nd July" / "2nd of July" -> "second of July"
    text = _DAY_MONTH_RE.sub(lambda m: f"{_ordinal(int(m.group(1)))} of {m.group(2)}", text)
    # "July 2nd" / "July 2" -> "July the second"
    text = _MONTH_DAY_RE.sub(lambda m: f"{m.group(1)} the {_ordinal(int(m.group(2)))}", text)
    # Any remaining ordinal: 24th -> twenty-fourth
    text = _ORDINAL_RE.sub(lambda m: _ordinal(int(m.group(1))), text)
    # Decimals: 9.6 -> nine point six
    text = _DECIMAL_RE.sub(_decimal, text)

    def number(match: re.Match) -> str:
        raw = match.group(0)
        value = int(raw)
        if len(raw) > MAX_CARDINAL_DIGITS or (raw.startswith("0") and len(raw) > 1):
            return _digits(raw)
        if len(raw) == 4 and _is_year(value) and _looks_like_a_year(text, match.start()):
            return _year(value)
        return _cardinal(value)

    return _INTEGER_RE.sub(number, text)

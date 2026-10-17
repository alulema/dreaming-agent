"""Deterministic grounding checks: are the names and numbers in a generated text present in
its source? Used by the Writer (source = the user's message) and the Dream Validator
(source = the target entries). Catches prompt-example leakage and invented details."""

import re
import unicodedata
from difflib import SequenceMatcher

# Capitalised words (not sentence-initial) and numbers: the details a model can invent.
NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-ZÁÉÍÓÚÑ][\wáéíóúñü]+)")
NUMBER = re.compile(r"\d+")
STOP = {"the", "user", "user's", "users", "they", "their", "he", "she", "i"}
# Words the Writer capitalises in English but which the source may spell differently.
CALENDAR = {"january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december", "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday", "english", "spanish"}
NUMBER_WORDS = {
    "uno": 1, "una": 1, "un": 1, "one": 1, "dos": 2, "two": 2, "tres": 3, "three": 3,
    "cuatro": 4, "four": 4, "cinco": 5, "five": 5, "seis": 6, "six": 6, "siete": 7, "seven": 7,
    "ocho": 8, "eight": 8, "nueve": 9, "nine": 9, "diez": 10, "ten": 10, "once": 11,
    "eleven": 11, "doce": 12, "twelve": 12, "veinte": 20, "twenty": 20, "treinta": 30,
}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _name_in(name: str, source: str, fuzzy: bool) -> bool:
    name = fold(name)
    if name in source:
        return True
    if not fuzzy:
        return False
    # Cross-language spellings: Lisbon/Lisboa 0.83, Japan/Japon 0.8, Porto/Oporto 0.91.
    return any(SequenceMatcher(None, name, tok).ratio() >= 0.75
               for tok in re.findall(r"\w{3,}", source))


def _number_in(num: str, source: str) -> bool:
    if re.search(rf"(?<!\d){num}(?!\d)", source):
        return True
    return any(v == int(num) for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", source))


def ungrounded(text: str, source: str, *, fuzzy: bool = False) -> list[str]:
    """Names and numbers in `text` that do not appear in `source`."""
    src = fold(source)
    missing = []
    for name in dict.fromkeys(NAME.findall(text)):
        f = fold(name)
        if f in STOP or f in CALENDAR or _name_in(name, src, fuzzy):
            continue
        missing.append(name)
    missing += [n for n in dict.fromkeys(NUMBER.findall(text)) if not _number_in(n, src)]
    return missing

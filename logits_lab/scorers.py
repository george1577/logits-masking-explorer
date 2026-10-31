"""Scorers that judge a piece of text. Sequence selection uses them to decide which beams survive."""

import re
from functools import lru_cache


@lru_cache(maxsize=None)
def _phrase_pattern(phrase: str) -> re.Pattern:
    # Whole-word, case-insensitive. The reference notebook used plain substring matching,
    # which counts "add" inside "address"; whole-word matching avoids those false hits.
    return re.compile(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)")


def find_phrases(text: str, phrases) -> list[str]:
    """Return the phrases that appear in `text`, in the order they were given."""
    lowered = text.lower()
    return [p for p in phrases if _phrase_pattern(p).search(lowered)]


def brand_score(text: str, desired, banned) -> int:
    """Count of desired phrases present minus count of banned phrases present."""
    return len(find_phrases(text, desired)) - len(find_phrases(text, banned))


def alliteration_score(text: str, letter: str) -> int:
    """Count of words that begin with `letter` (case-insensitive)."""
    letter = letter.lower()
    words = re.findall(r"[A-Za-z']+", text)
    return sum(1 for w in words if w[0].lower() == letter)


def load_phrases(path) -> list[str]:
    """Read one phrase per line, lowercased, de-duplicated, order preserved."""
    with open(path, encoding="utf-8") as f:
        seen = dict.fromkeys(line.strip().lower() for line in f if line.strip())
    return list(seen)

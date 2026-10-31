"""Token-level masking: forbid the next token whenever it would complete a banned phrase.

A banned phrase is usually several tokens long, and the same words tokenize differently
depending on case and on whether a space comes before them ("cure" vs " Cure"). So each
phrase is expanded into its surface variants, each variant is tokenized, and the resulting
token sequences are indexed by their prefix:

    (" cure",)            -> {" cancer"}      ban " cancer" right after " cure"
    ()                    -> {" ADHD", ...}   single-token phrases are banned everywhere

At each step we look at the tail of what has been generated so far, and ban every token
that would finish one of the sequences.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch


def phrase_variants(phrase: str) -> list[str]:
    """Surface forms a phrase can take inside running text."""
    forms = [phrase.lower(), phrase.capitalize(), phrase.title(), phrase.upper()]
    variants = []
    for form in forms:
        for v in (form, " " + form):
            if v not in variants:
                variants.append(v)
    return variants


@dataclass
class TokenBanList:
    """Index of banned token sequences, keyed by the prefix that must come first."""

    by_prefix: dict[tuple[int, ...], set[int]] = field(default_factory=dict)
    # phrase -> list of (variant text, token ids); kept so the UI can show the tokenizations
    tokenizations: dict[str, list[tuple[str, list[int]]]] = field(default_factory=dict)

    @classmethod
    def from_phrases(cls, phrases: Iterable[str], encode: Callable[[str], list[int]]) -> "TokenBanList":
        by_prefix: dict[tuple[int, ...], set[int]] = defaultdict(set)
        tokenizations = {}
        for phrase in phrases:
            entries = []
            for variant in phrase_variants(phrase):
                ids = encode(variant)
                if not ids:
                    continue
                entries.append((variant, ids))
                by_prefix[tuple(ids[:-1])].add(ids[-1])
            tokenizations[phrase] = entries
        return cls(by_prefix=dict(by_prefix), tokenizations=tokenizations)

    @property
    def prefix_lengths(self) -> list[int]:
        return sorted({len(p) for p in self.by_prefix})

    def banned_next(self, generated: list[int]) -> set[int]:
        """Token ids that must not come next, given the tokens generated so far."""
        banned: set[int] = set()
        for n in self.prefix_lengths:
            if n > len(generated):
                break
            tail = tuple(generated[len(generated) - n:]) if n else ()
            banned |= self.by_prefix.get(tail, set())
        return banned


def apply_token_mask(logits: torch.Tensor, banned_ids: Iterable[int]) -> torch.Tensor:
    """Return a copy of 1-D `logits` with banned ids pushed to -inf (probability exactly 0)."""
    masked = logits.clone()
    ids = [i for i in banned_ids if 0 <= i < masked.shape[-1]]
    if ids:
        masked[ids] = float("-inf")
    return masked

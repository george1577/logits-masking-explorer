"""Sequence regeneration for acrostic poems.

Generate a few short segments, throw away the ones whose line initials stop spelling the
target word, and continue from the most probable segment that still fits. Repeat until the
poem is complete or no segment fits.
"""

from dataclasses import dataclass, field
from typing import Callable

LEADING_JUNK = " \t*-#\"'“‘•>0123456789.)"


def line_initial(line: str) -> str:
    stripped = line.lstrip(LEADING_JUNK)
    return stripped[0].lower() if stripped else ""


@dataclass
class AcrosticCheck:
    lines: list[str]        # non-empty lines, truncated to the target word's length
    initials: str
    fits: bool              # initials so far are a prefix of the target word
    complete: bool          # every letter has a finished line


def check_acrostic(text: str, target: str, finished: bool) -> AcrosticCheck:
    """Judge a poem-so-far. `finished` means generation stopped (EOS), so the last line is whole."""
    target = target.lower()
    raw = text.split("\n")
    lines = [l.strip() for l in raw if l.strip()][: len(target)]
    initials = "".join(line_initial(l) for l in lines)
    fits = bool(initials) and target.startswith(initials)

    # a line counts as finished once a newline follows it, or generation ended
    whole = raw if finished else raw[:-1]
    whole_lines = [l.strip() for l in whole if l.strip()]
    complete = fits and initials == target and len(whole_lines) >= len(target)
    return AcrosticCheck(lines, initials, fits, complete)


@dataclass
class Segment:
    text: str               # the newly generated continuation
    logprob: float
    finished: bool


@dataclass
class SegmentVerdict:
    segment: Segment
    check: AcrosticCheck
    chosen: bool = False


@dataclass
class Round:
    number: int
    poem_before: str
    verdicts: list[SegmentVerdict]

    @property
    def chosen(self):
        return next((v for v in self.verdicts if v.chosen), None)


@dataclass
class RegenerationResult:
    target: str
    rounds: list[Round] = field(default_factory=list)
    poem_lines: list[str] = field(default_factory=list)
    status: str = "running"   # "complete" | "stuck" | "out of rounds"


# generate_segments(poem so far) -> candidate continuations
SegmentFn = Callable[[str], list[Segment]]


def regenerate(generate_segments: SegmentFn, target: str, max_rounds: int = 8) -> RegenerationResult:
    result = RegenerationResult(target=target.lower())
    poem = ""

    for number in range(1, max_rounds + 1):
        verdicts = []
        for seg in generate_segments(poem):
            verdicts.append(SegmentVerdict(seg, check_acrostic(poem + seg.text, target, seg.finished)))
        rnd = Round(number, poem, verdicts)
        result.rounds.append(rnd)

        fitting = [v for v in verdicts if v.check.fits]
        if not fitting:
            result.status = "stuck"
            result.poem_lines = check_acrostic(poem, target, True).lines if poem else []
            return result

        best = max(fitting, key=lambda v: v.segment.logprob)
        best.chosen = True
        if best.check.complete:
            result.status = "complete"
            result.poem_lines = best.check.lines
            return result
        # Continue from the chosen segment. End with a newline only if its last line was
        # finished; otherwise the next segment carries on writing that same line.
        full = poem + best.segment.text
        poem = "\n".join(best.check.lines)
        if best.segment.finished or full.rstrip(" \t").endswith("\n"):
            poem += "\n"

    result.status = "out of rounds"
    result.poem_lines = check_acrostic(poem, target, True).lines
    return result

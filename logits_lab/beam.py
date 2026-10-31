"""A small, transparent beam search loop with a pluggable selector (see docs/adr/0001).

Each step:
  1. decode every beam's text so far
  2. the selector decides which beams survive
  3. the model gives next-token log-probabilities for every beam
  4. every non-surviving beam's whole row is replaced by MASK_VALUE, the same move the
     reference notebooks make inside their LogitsProcessor
  5. the best `num_beams` (beam, next token) pairs, by total log-probability, become the new beams

A step trace is recorded so the UI can show which beams were alive and which were masked.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

# The notebooks use -10000 rather than -inf ("torch doesn't like it to be -np.inf").
MASK_VALUE = -10000.0

# step_fn: given each unfinished beam's generated token ids, return log-probabilities shaped [num_beams, vocab]
StepFn = Callable[[list[list[int]]], torch.Tensor]


@dataclass
class Selection:
    keep: list[bool]
    scores: Optional[list[float]] = None


# selector: given each beam's decoded text, decide which beams survive
Selector = Callable[[list[str]], Selection]


def keep_all(texts: list[str]) -> Selection:
    """Plain beam search: nothing is masked."""
    return Selection(keep=[True] * len(texts))


def keep_top_scorers(score_fn: Callable[[str], float]) -> Selector:
    """Notebook rule: beams tied for the highest score survive; all others are masked."""

    def select(texts: list[str]) -> Selection:
        scores = [score_fn(t) for t in texts]
        best = max(scores)
        return Selection(keep=[s == best for s in scores], scores=scores)

    return select


@dataclass
class Beam:
    token_ids: list[int]
    logprob: float = 0.0
    finished: bool = False


@dataclass
class BeamSnapshot:
    text: str
    logprob: float
    score: Optional[float]
    kept: bool
    finished: bool


@dataclass
class StepTrace:
    step: int
    beams: list[BeamSnapshot]

    @property
    def num_kept(self) -> int:
        return sum(b.kept for b in self.beams)


@dataclass
class BeamResult:
    beams: list[Beam]               # final beams, best first
    texts: list[str]
    scores: Optional[list[float]]
    trace: list[StepTrace] = field(default_factory=list)

    @property
    def best_text(self) -> str:
        return self.texts[0]


def beam_search(
    step_fn: StepFn,
    decode: Callable[[list[int]], str],
    eos_ids: set[int],
    num_beams: int = 4,
    max_new_tokens: int = 60,
    selector: Selector = keep_all,
) -> BeamResult:
    beams = [Beam(token_ids=[])]
    trace: list[StepTrace] = []

    for step in range(max_new_tokens):
        if all(b.finished for b in beams):
            break

        texts = [decode(b.token_ids) for b in beams]
        selection = selector(texts)
        trace.append(StepTrace(step, [
            BeamSnapshot(
                text=t,
                logprob=b.logprob,
                score=None if selection.scores is None else selection.scores[i],
                kept=selection.keep[i],
                finished=b.finished,
            )
            for i, (b, t) in enumerate(zip(beams, texts))
        ]))

        # Only unfinished beams go to the model; they all have the same length.
        live = [i for i, b in enumerate(beams) if not b.finished]
        live_logprobs = step_fn([beams[i].token_ids for i in live])
        row_of = {beam_idx: live_logprobs[r] for r, beam_idx in enumerate(live)}

        # Candidates are (total log-prob, parent beam index, next token or None for "stay finished")
        candidates: list[tuple[float, int, Optional[int]]] = []
        for i, beam in enumerate(beams):
            if beam.finished:
                # a finished beam can only stay as it is
                stay = beam.logprob if selection.keep[i] else beam.logprob + MASK_VALUE
                candidates.append((stay, i, None))
                continue
            row = row_of[i]
            if not selection.keep[i]:
                row = torch.full_like(row, MASK_VALUE)
            top = torch.topk(row, k=min(num_beams, row.shape[-1]))
            for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                candidates.append((beam.logprob + lp, i, tok))

        candidates.sort(key=lambda c: c[0], reverse=True)
        new_beams = []
        for total, parent, tok in candidates[:num_beams]:
            src = beams[parent]
            if tok is None:
                new_beams.append(Beam(list(src.token_ids), total, finished=True))
            else:
                new_beams.append(Beam(src.token_ids + [tok], total, finished=tok in eos_ids))
        beams = new_beams

    # Final pick honours the selector too: surviving beams first, then by log-probability.
    texts = [decode(b.token_ids) for b in beams]
    final = selector(texts)
    order = sorted(range(len(beams)), key=lambda i: (final.keep[i], beams[i].logprob), reverse=True)
    return BeamResult(
        beams=[beams[i] for i in order],
        texts=[texts[i] for i in order],
        scores=None if final.scores is None else [final.scores[i] for i in order],
        trace=trace,
    )

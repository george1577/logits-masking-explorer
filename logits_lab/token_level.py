"""Greedy generation that records the next-token distribution before and after token-level masking."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .masking import TokenBanList, apply_token_mask

# next_logits(previous token or None for the first call) -> 1-D logits over the vocabulary
NextLogitsFn = Callable[[Optional[int]], torch.Tensor]


@dataclass
class Candidate:
    token_id: int
    prob: float
    masked: bool


@dataclass
class TokenStep:
    step: int
    chosen_id: int
    before: list[Candidate]      # top-k of the raw distribution, masked ones flagged
    after: list[Candidate]       # top-k after masking and renormalising
    num_banned: int              # how many vocabulary entries were masked at this step

    @property
    def mask_changed_choice(self) -> bool:
        """True when the token greedy decoding would have picked was banned."""
        return bool(self.before) and self.before[0].masked


@dataclass
class TokenRun:
    token_ids: list[int]
    steps: list[TokenStep]


def greedy_generate(
    next_logits: NextLogitsFn,
    eos_ids: set[int],
    max_new_tokens: int = 100,
    ban_list: Optional[TokenBanList] = None,
    top_k: int = 10,
) -> TokenRun:
    generated: list[int] = []
    steps: list[TokenStep] = []
    prev: Optional[int] = None

    for step in range(max_new_tokens):
        logits = next_logits(prev).float()
        banned = ban_list.banned_next(generated) if ban_list else set()
        masked_logits = apply_token_mask(logits, banned)

        before_p = torch.softmax(logits, dim=-1)
        after_p = torch.softmax(masked_logits, dim=-1)
        top_before = torch.topk(before_p, top_k)
        top_after = torch.topk(after_p, top_k)

        chosen = int(torch.argmax(masked_logits))
        steps.append(TokenStep(
            step=step,
            chosen_id=chosen,
            before=[Candidate(t, p, t in banned) for p, t in zip(top_before.values.tolist(), top_before.indices.tolist())],
            after=[Candidate(t, p, False) for p, t in zip(top_after.values.tolist(), top_after.indices.tolist())],
            num_banned=len(banned),
        ))
        generated.append(chosen)
        prev = chosen
        if chosen in eos_ids:
            break

    return TokenRun(token_ids=generated, steps=steps)

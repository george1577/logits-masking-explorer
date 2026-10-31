import math

import torch

from logits_lab.masking import TokenBanList, apply_token_mask, phrase_variants
from logits_lab.token_level import greedy_generate

# A toy word-level tokenizer: each space-separated word (with its leading space) is one token.
VOCAB = ["<eos>", " cure", " cancer", " the", "cure", " Cure", " helps", " Cancer", " ADHD", "ADHD"]
IDS = {w: i for i, w in enumerate(VOCAB)}


def toy_encode(text: str) -> list[int]:
    out = []
    pieces = text.replace(" ", "\0 ").split("\0")
    for piece in pieces:
        if piece and piece not in IDS:
            return []  # untokenizable in the toy vocab: skip this variant
        if piece:
            out.append(IDS[piece])
    return out


def test_variants_cover_case_and_leading_space():
    v = phrase_variants("cure cancer")
    assert "cure cancer" in v and " cure cancer" in v and " Cure Cancer" in v


def test_multi_token_phrase_only_bans_completing_token():
    bans = TokenBanList.from_phrases(["cure cancer"], toy_encode)
    assert IDS[" cancer"] not in bans.banned_next([IDS[" the"]])
    assert IDS[" cancer"] in bans.banned_next([IDS[" the"], IDS[" cure"]])
    assert IDS[" Cancer"] in bans.banned_next([IDS[" Cure"]])


def test_single_token_phrase_is_banned_everywhere():
    bans = TokenBanList.from_phrases(["adhd"], toy_encode)
    assert {IDS[" ADHD"], IDS["ADHD"]} <= bans.banned_next([])
    assert {IDS[" ADHD"], IDS["ADHD"]} <= bans.banned_next([IDS[" the"], IDS[" helps"]])


def test_apply_token_mask_zeroes_probability_and_renormalises():
    logits = torch.tensor([2.0, 1.0, 0.0])
    masked = apply_token_mask(logits, {0})
    probs = torch.softmax(masked, dim=-1)
    assert probs[0] == 0
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-6)
    assert logits[0] == 2.0  # original untouched


def test_greedy_generate_avoids_banned_completion():
    # The fake model always wants " cure" then " cancer", then end.
    script = [IDS[" cure"], IDS[" cancer"], IDS["<eos>"]]
    state = {"step": 0}

    def next_logits(prev):
        logits = torch.zeros(len(VOCAB))
        logits[script[min(state["step"], 2)]] = 5.0
        logits[IDS[" helps"]] = 4.0  # runner-up
        state["step"] += 1
        return logits

    bans = TokenBanList.from_phrases(["cure cancer"], toy_encode)
    run = greedy_generate(next_logits, eos_ids={IDS["<eos>"]}, max_new_tokens=5, ban_list=bans, top_k=3)
    assert run.token_ids[:2] == [IDS[" cure"], IDS[" helps"]]
    assert run.steps[1].mask_changed_choice
    assert not run.steps[0].mask_changed_choice

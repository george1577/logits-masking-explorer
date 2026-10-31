"""Wire the language model to each technique. The Streamlit app calls only these functions."""

from dataclasses import dataclass

from .autocomplete import autocomplete_messages, grounding_lines
from .beam import BeamResult, beam_search, keep_all, keep_top_scorers
from .lm import LanguageModel
from .masking import TokenBanList
from .prompts import acrostic_messages, nursery_rhyme_messages, product_description_messages
from .regeneration import RegenerationResult, Segment, check_acrostic, regenerate
from .scorers import alliteration_score, brand_score
from .token_level import TokenRun, greedy_generate


# ---- ① token-level masking ------------------------------------------------------------

@dataclass
class TokenLevelComparison:
    ban_list: TokenBanList
    unmasked: TokenRun
    masked: TokenRun


def run_token_level(lm: LanguageModel, item: str, banned: list[str], max_new_tokens: int, top_k: int = 10):
    prompt = lm.chat_prompt_ids(product_description_messages(item))
    ban_list = TokenBanList.from_phrases(banned, lm.encode)
    unmasked = greedy_generate(lm.greedy_stepper(prompt), lm.eos_ids, max_new_tokens, None, top_k)
    masked = greedy_generate(lm.greedy_stepper(prompt), lm.eos_ids, max_new_tokens, ban_list, top_k)
    return TokenLevelComparison(ban_list, unmasked, masked)


# ---- ② sequence selection -------------------------------------------------------------

@dataclass
class SelectionComparison:
    baseline: BeamResult
    selected: BeamResult
    score_fn: object


def run_sequence_selection(lm: LanguageModel, scorer: str, subject: str, desired: list[str],
                           banned: list[str], num_beams: int, max_new_tokens: int):
    if scorer == "brand":
        messages = product_description_messages(subject)
        score_fn = lambda text: brand_score(text, desired, banned)
    else:
        messages = nursery_rhyme_messages(subject)
        letter = subject.strip()[0]
        score_fn = lambda text: alliteration_score(text, letter)

    prompt = lm.chat_prompt_ids(messages)
    baseline = beam_search(lm.beam_step_fn(prompt), lm.decode, lm.eos_ids, num_beams, max_new_tokens, keep_all)
    selected = beam_search(lm.beam_step_fn(prompt), lm.decode, lm.eos_ids, num_beams, max_new_tokens,
                           keep_top_scorers(score_fn))
    return SelectionComparison(baseline, selected, score_fn)


# ---- ③ sequence regeneration ----------------------------------------------------------

def run_regeneration(lm: LanguageModel, animal: str, target: str, num_segments: int = 6,
                     segment_tokens: int = 16, max_rounds: int = 10) -> RegenerationResult:
    # A segment ends at the end of a line (or at segment_tokens), so each round adds one line.
    stop_ids = lm.eos_ids | lm.newline_ids

    def generate_segments(poem_so_far: str) -> list[Segment]:
        # Hint the letter the next line needs. The prompt only asks; the regeneration check enforces.
        done = check_acrostic(poem_so_far, target, finished=True) if poem_so_far else None
        needed = target[len(done.lines)] if done and len(done.lines) < len(target) else target[0]
        if poem_so_far and not poem_so_far.endswith("\n"):
            needed = ""  # mid-line: the current line already has its letter
        prompt = lm.chat_prompt_ids(acrostic_messages(animal, target, needed), prefill=poem_so_far)
        result = beam_search(lm.beam_step_fn(prompt), lm.decode, stop_ids, num_segments, segment_tokens)
        return [
            Segment(text, beam.logprob, finished=bool(beam.token_ids) and beam.token_ids[-1] in lm.eos_ids)
            for text, beam in zip(result.texts, result.beams)
        ]

    return regenerate(generate_segments, target, max_rounds)


# ---- ④ autocomplete -------------------------------------------------------------------

@dataclass
class Suggestions:
    examples: list[str]
    options: list[str]
    logprobs: list[float]


def suggest_continuations(lm: LanguageModel, document: list[str], typed_prefix: str, accepted: str,
                          num_beams: int = 5, chunk_tokens: int = 8) -> Suggestions:
    examples = grounding_lines(document, typed_prefix)
    prompt = lm.chat_prompt_ids(autocomplete_messages(examples, typed_prefix), prefill=accepted)
    result = beam_search(lm.beam_step_fn(prompt), lm.decode, lm.eos_ids, num_beams, chunk_tokens)
    # Beams often differ only by trailing punctuation; show each distinct continuation once.
    options, logprobs, seen = [], [], set()
    for text, beam in zip(result.texts, result.beams):
        key = text.strip().rstrip(".,;:!?").lower()
        if key in seen:
            continue
        seen.add(key)
        options.append(text)
        logprobs.append(beam.logprob)
    return Suggestions(examples, options, logprobs)


def full_autocomplete_text(typed_prefix: str, accepted: str) -> str:
    """The model often repeats the typed prefix at the start of its completion; don't show it twice."""
    if not accepted.strip():
        return typed_prefix
    if accepted.strip().lower().startswith(typed_prefix.strip().lower()):
        return accepted.strip()
    return f"{typed_prefix.rstrip()} {accepted.lstrip()}"

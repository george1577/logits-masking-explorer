"""The only module that touches the real model. Everything else works on plain token ids and logits."""

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


def pick_device() -> str:
    # Apple's MPS backend is opt-in: recent torch builds crash on older macOS versions,
    # and a 0.5B model is fast enough on CPU. Set LME_DEVICE=mps (or cuda) to override.
    if os.environ.get("LME_DEVICE"):
        return os.environ["LME_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


class LanguageModel:
    def __init__(self, model_id: str = MODELS[0]):
        self.model_id = model_id
        self.device = pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()

        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self._newline_ids = None

    @property
    def newline_ids(self) -> set[int]:
        """Every token whose text contains a line break (used to end a segment at the end of a line)."""
        if self._newline_ids is None:
            vocab = self.tokenizer.get_vocab()
            self._newline_ids = {i for i in vocab.values() if "\n" in self.tokenizer.decode([i])}
        return self._newline_ids

    # ---- text <-> ids ---------------------------------------------------------------

    def chat_prompt_ids(self, messages: list[dict], prefill: str = "") -> list[int]:
        """Token ids for a chat prompt, optionally with the start of the assistant's reply already written."""
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.encode(text + prefill)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_label(self, token_id: int) -> str:
        """A token's text with whitespace made visible."""
        if token_id in self.eos_ids:
            return "⟨end⟩"
        text = self.tokenizer.decode([token_id])
        return text.replace(" ", "␣").replace("\n", "⏎")

    # ---- model calls ----------------------------------------------------------------

    def greedy_stepper(self, prompt_ids: list[int]):
        """A next_logits(prev_token) function that reuses the KV cache between steps."""
        state = {"past": None}

        @torch.no_grad()
        def next_logits(prev: Optional[int]) -> torch.Tensor:
            ids = prompt_ids if prev is None else [prev]
            out = self.model(
                input_ids=torch.tensor([ids], device=self.device),
                past_key_values=state["past"],
                use_cache=True,
            )
            state["past"] = out.past_key_values
            return out.logits[0, -1].float().cpu()

        return next_logits

    def beam_step_fn(self, prompt_ids: list[int]):
        """A step_fn for beam_search: log-probabilities of the next token for every beam.

        Keeps a KV cache across steps. Each new beam is its parent plus one token, so the
        cache rows are reordered to follow the parents and only the newest token is fed in.
        """
        state = {"beams": None, "past": None}

        @torch.no_grad()
        def step(beams: list[list[int]]) -> torch.Tensor:
            parents = None
            if state["beams"] is not None:
                index = {tuple(b): i for i, b in enumerate(state["beams"])}
                parents = [index.get(tuple(b[:-1])) for b in beams]
            if parents is None or None in parents or not all(beams):
                # first step (or untracked beams): run the full sequences
                inputs, past = [prompt_ids + b for b in beams], None
            else:
                past = state["past"]
                past.reorder_cache(torch.tensor(parents, device=self.device))
                inputs = [[b[-1]] for b in beams]
            out = self.model(
                input_ids=torch.tensor(inputs, device=self.device),
                past_key_values=past,
                use_cache=True,
            )
            state["beams"], state["past"] = [list(b) for b in beams], out.past_key_values
            return torch.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu()

        return step

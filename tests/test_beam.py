import torch

from logits_lab.beam import MASK_VALUE, beam_search, keep_all, keep_top_scorers

# Toy vocab: token id -> text
VOCAB = ["<eos>", " good", " bad", " ok"]
EOS = 0


def decode(ids):
    return "".join(VOCAB[i] for i in ids)


def step_fn(beams):
    # The model prefers " bad" > " ok" > " good" > eos, regardless of history.
    row = torch.log_softmax(torch.tensor([0.0, 1.0, 3.0, 2.0]), dim=-1)
    return row.repeat(len(beams), 1)


def test_plain_beam_search_follows_model_preference():
    result = beam_search(step_fn, decode, {EOS}, num_beams=2, max_new_tokens=2, selector=keep_all)
    assert result.best_text == " bad bad"


def test_selector_masks_losing_beams():
    # Score: number of " good" minus number of " bad"
    score = lambda t: t.count("good") - t.count("bad")
    result = beam_search(step_fn, decode, {EOS}, num_beams=3, max_new_tokens=3,
                         selector=keep_top_scorers(score))
    # After step 0 the beams are " bad", " ok", " good"; only " good" scores highest and survives,
    # so all three beams at step 2 must descend from it.
    step1 = result.trace[1]
    kept = [b.text for b in step1.beams if b.kept]
    assert kept == [" good"]
    assert all(t.startswith(" good") for t in result.texts)
    assert result.scores[0] == max(result.scores)


def test_ties_all_survive():
    score = lambda t: 0
    result = beam_search(step_fn, decode, {EOS}, num_beams=3, max_new_tokens=2,
                         selector=keep_top_scorers(score))
    assert all(b.kept for b in result.trace[1].beams)


def test_finished_beams_stop_growing():
    def eos_first(beams):
        return torch.log_softmax(torch.tensor([5.0, 0.0, 0.0, 0.0]), dim=-1).repeat(len(beams), 1)

    result = beam_search(eos_first, decode, {EOS}, num_beams=2, max_new_tokens=5)
    assert result.beams[0].token_ids == [EOS]
    assert result.beams[0].finished


def test_mask_value_matches_notebook():
    assert MASK_VALUE == -10000.0

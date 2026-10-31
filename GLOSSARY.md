# Logits Masking Explorer

A learning tool for seeing how logits masking steers a language model's output, token by token and beam by beam.

## Language

### Model output

**Logits**:
The raw, unnormalized scores the model assigns to every token in its vocabulary for the next position.
_Avoid_: Scores (ambiguous with Brand Score), probabilities

**Next-token distribution**:
The probabilities obtained by applying softmax to the logits at one generation step.
_Avoid_: Logits (they are pre-softmax)

**Log-probability**:
The summed log of the probabilities of every token in a sequence; how likely the model considers the whole sequence.
_Avoid_: Logprob score, confidence

**Beam**:
One candidate partial sequence kept alive during beam search.
_Avoid_: Hypothesis

### Techniques

**Logits masking**:
The overall pattern of steering generation by suppressing continuations that don't meet a goal, either within a next-token distribution or across whole sequences.
_Avoid_: Filtering, censoring

**Token-level masking**:
Logits masking that pushes specific tokens' logits to negative infinity within a single next-token distribution.
_Avoid_: Token banning

**Sequence selection**:
Logits masking during beam search where a selector picks which beams survive each step and every other beam's logits are suppressed. The selector may be a scorer or a person.
_Avoid_: Sequence-level masking, beam filtering, beam pruning

**Sequence regeneration**:
Generating in fixed-length segments, discarding segments that violate a constraint, and continuing from the most probable segment that satisfies it.
_Avoid_: Retry, resampling

**Segment**:
The fixed-length continuation produced by one round of sequence regeneration.
_Avoid_: Chunk, candidate

**Logit biasing**:
Adding to or subtracting from logits to make tokens more or less likely without making them impossible. Related to masking but distinct from it.
_Avoid_: Soft masking, boosting

### Scenarios

**Banned phrase**:
A phrase the output must not contain.
_Avoid_: Forbidden word, negative

**Desired phrase**:
A phrase the output should contain as often as possible.
_Avoid_: Brand word, positive

**Brand score**:
The count of desired phrases present in a text minus the count of banned phrases present.
_Avoid_: Score, evaluation

**Alliteration score**:
The count of words in a text that begin with a chosen letter.
_Avoid_: Letter count

**Target word**:
The word an acrostic poem's line initials must spell, in order.
_Avoid_: Start word, acrostic word

**Typed prefix**:
The text a person has entered so far in autocomplete, which grounds and seeds the continuations offered.
_Avoid_: Query, search term

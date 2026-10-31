# How Logits Masking Works: a Beginner's Guide

This guide explains **logits masking** from the ground up and walks through every part of the
Logits Masking Explorer app. No machine-learning background is needed. If a word is unfamiliar,
check the [glossary](../GLOSSARY.md).

> **The 30-second version.** A language model writes one small piece of text (a *token*) at a time.
> Before choosing each token it gives every possible token a score. **Logits masking** means
> editing those scores *before* the choice is made, so that tokens (or whole drafts) we don't want
> can never be picked. The model still does the writing; we just take some options off the table.

---

## 1. How a language model picks the next word

A language model doesn't write sentences in one go. It runs a loop:

```mermaid
flowchart LR
    A["Text so far<br/>'Experience the'"] --> B["Model"]
    B --> C["A score (logit) for every<br/>token in the vocabulary<br/>(about 150,000 of them)"]
    C --> D["Softmax: turn scores<br/>into probabilities"]
    D --> E["Pick a token<br/>(e.g. the most likely)"]
    E --> F["Append it to the text"]
    F --> A
```

- A **token** is a chunk of text: often a whole word with its leading space (`␣power`), sometimes a
  piece of a word (`Ult` + `imate`), sometimes punctuation or a line break.
- **Logits** are the model's raw scores, one per token. Higher means "I think this fits better".
  They can be any number, negative or positive.
- **Softmax** turns logits into **probabilities** that add up to 100%. A token whose logit is
  −∞ (minus infinity) gets a probability of exactly 0%.
- **Greedy decoding** simply picks the most probable token each time.

That last point is the whole trick: **if you can change the logits, you can control what the model
is allowed to say.**

---

## 2. Token-level masking (tab ①)

### The idea

Before the model picks a token, we find the tokens we don't want and set their logits to −∞.
Their probability drops to 0, and the probabilities of everything else are scaled up so they
still add up to 100% (that's called *renormalising*).

### A real example from the app

The app asks the model (Qwen2.5-0.5B) to write a product description for a "protein drink", with a
list of phrases Amazon restricts in listings ("ultimate", "perfect", "natural", …). After writing
`Experience the`, here is what the model wanted next:

| Token | Before masking | Banned? | After masking |
|---|---|---|---|
| `␣ultimate` | 25.0% | 🚫 yes | 0% |
| `␣perfect` | 18.3% | 🚫 yes | 0% |
| `␣power` | 14.0% | | **26.4%** ← picked |
| `␣energy` | 4.4% | | 8.4% |
| `␣boost` | under 3.1% | | 4.9% |

Without the mask the model would have written "Experience the **ultimate** …". With it, it wrote
"Experience the **power** of protein…". The finished unmasked description contained five banned
phrases ("natural", "perfect", "perfect for", "quality", "ultimate"); the masked one contained none.

### The catch: phrases are made of several tokens

You can't just ban "the token for *all natural*", because there isn't one:

| Phrase as written | Tokens |
|---|---|
| `all natural` | `all` · `␣natural` |
| ` All natural` | `␣All` · `␣natural` |
| `Ultimate` | `Ult` · `imate` |
| ` ultimate` | `␣ultimate` |

Two things to notice: the **same words split differently** depending on capital letters and a
leading space, and one phrase can be **several tokens long**.

So the app does **prefix-aware** masking:

1. For each banned phrase, list its spelling variants (lowercase, Capitalised, Title Case, UPPER,
   each with and without a leading space) and tokenize each one.
2. At every step, look at the last few tokens generated so far.
3. If they match the beginning of a banned token sequence, ban the token that would **finish** it.
   One-token phrases are banned everywhere.

```mermaid
flowchart TD
    A["Tokens generated so far<br/>... ␣all"] --> B{"Do the last tokens match<br/>the start of a banned sequence?"}
    B -- "yes: '␣all' starts '␣all ␣natural'" --> C["Ban '␣natural' for this step"]
    B -- "no" --> D["Ban only one-token phrases"]
    C --> E["Set banned logits to -inf"]
    D --> E
    E --> F["Softmax + pick the top token"]
```

Banning `␣natural` *only right after* `␣all` keeps the mask narrow, so "natural" isn't blocked
everywhere unless it's banned by itself (in this list it is).

**Limitation:** the model could still spell a phrase in a way we didn't list (for example
`nat` + `ural`). Token masking only blocks the token sequences you give it.

---

## 3. Sequence selection (tab ②)

### From one draft to several: beam search

Instead of keeping one draft, **beam search** keeps several drafts alive at the same time. Each
draft is called a **beam**. At each step every beam proposes its best next tokens, and the overall
best few (by total probability) become the new beams.

### The idea

**Sequence selection** adds a judge. At every step, a **scorer** reads each beam's text so far.
Beams tied for the best score survive; every other beam gets **all** of its logits replaced with
`-10000`. A beam with logits like that can't win a place in the next round, so only the survivors'
continuations go forward.

```mermaid
flowchart TD
    A["4 beams (drafts)"] --> B["Scorer rates each draft<br/>e.g. brand score"]
    B --> C{"Tied for the best score?"}
    C -- "yes" --> D["Keep its logits"]
    C -- "no" --> E["Replace ALL its logits<br/>with -10000"]
    D --> F["Model proposes next tokens"]
    E --> F
    F --> G["Keep the best 4 (beam, token) pairs<br/>by total log-probability"]
    G --> A
```

> Why `-10000` and not −∞? The reference notebooks use `-10000` because some PyTorch operations
> misbehave with infinity. It's so low that it acts like −∞.

### A real example from the app

Scorer: **brand score** = (desired phrases present) − (banned phrases present).

| Step | Beam text so far | Score | Status |
|---|---|---|---|
| 3 | `Boost your energy` | 0 | ✅ survives |
| 3 | `Experience the ultimate` | −1 | 🚫 masked |
| 3 | `Discover the ultimate` | −1 | 🚫 masked |
| 3 | `Experience the perfect` | −1 | 🚫 masked |

The three beams that used a banned word are cut off. All four beams at the next step grow out of
"Boost your energy".

Want to know exactly how the next four beams get picked, number by number? See
[Appendix A: how beam search picks the next beams](#appendix-a-how-beam-search-picks-the-next-beams).

### Two scorers, one mechanism

The app also offers notebook 1's **alliteration score** (count words starting with a letter).
Same beam loop, different judge. Try it with "donkey": the selected output may collapse into
"Donkey donkey donkey…". That's the scorer being **gamed**: the selector rewards *only* the
score. Masking enforces a goal; it doesn't understand quality. The notebook counters this by also
asking for readability in the prompt.

---

## 4. Sequence regeneration (tab ③)

### The idea

Some rules can only be checked on a chunk of text, not on a single token. Example: an
**acrostic poem**, where the first letters of the lines spell a word (B-O-L-D). So instead of
masking every step:

1. Generate a few short **segments** (here: candidate next lines).
2. Throw away every segment that breaks the rule.
3. Keep the **most probable** segment that still fits, and continue from there.

```mermaid
flowchart TD
    A["Poem so far"] --> B["Generate 6 candidate segments<br/>(each ends at a line break)"]
    B --> C["Check each: do the line initials<br/>still spell the start of the word?"]
    C --> D{"Any segment fits?"}
    D -- "no" --> E["Stuck: report it"]
    D -- "yes" --> F["Choose the fitting segment<br/>with the highest log-probability"]
    F --> G{"Word complete?"}
    G -- "no" --> A
    G -- "yes" --> H["Done ✅"]
```

**Log-probability** is how likely the model thinks a whole segment is. It's a negative number;
closer to 0 means more likely.

### A real example from the app (target WILD, animal tiger)

| Round | Chosen line | Initials |
|---|---|---|
| 1 | Wild, the tiger roars, | W |
| 2 | In the forest's embrace, | WI |
| 3 | Lions watch, | WIL |
| 4 | Dogs bark, | WILD ✅ |

Not great poetry from a 0.5B model, but every line passed the check. When **no** segment fits, the
technique is stuck: it can only choose among what the model offers. That's its real limitation.

**Masking or not?** Nothing is set to −∞ here. Rejecting a whole segment is the same idea
("this continuation is not allowed") applied after the fact, to a chunk instead of a token.

---

## 5. Autocomplete (tab ④)

### The idea

Same as sequence selection, but **you** are the selector. You type a few words; the model, given
lines from *Julius Caesar* that contain them, offers several beams. Clicking one keeps that beam
and is equivalent to setting every other beam's logits to `-10000`.

```mermaid
sequenceDiagram
    participant You
    participant App
    participant Model
    You->>App: type "Lend"
    App->>App: find lines in the play containing "Lend"
    App->>Model: prompt with those lines + "Lend ____"
    Model-->>App: 5 beams, e.g. "Lend me your hand", "friends, Romans, countrymen, lend"
    App-->>You: show them as buttons
    You->>App: click "Lend me your hand"
    Note over App: chosen beam kept, others -10000
    App->>Model: continue from the chosen beam
```

The notebook picked one beam **per token** (at random, to simulate a person). The app lets you
pick per **chunk** of about 8 tokens, which feels like real autocomplete and uses the same idea.

---

## 6. How the app is built

```mermaid
flowchart TD
    UI["app.py<br/>Streamlit UI: 4 tabs"] --> P["pipelines.py<br/>wires the model to each technique"]

    P --> T["token_level.py<br/>greedy loop, records before/after"]
    P --> B["beam.py<br/>our own beam search + selectors"]
    P --> R["regeneration.py<br/>segment / check / choose loop"]
    P --> A["autocomplete.py<br/>grounding lines + prompt"]

    T --> M["masking.py<br/>prefix-aware ban list, -inf mask"]
    B --> S["scorers.py<br/>brand score, alliteration score"]
    R --> B
    A --> B

    P --> LM["lm.py<br/>the only code that touches the model"]
    LM --> HF["Qwen2.5 via Hugging Face transformers"]

    D[("data/<br/>banned_phrases.txt<br/>desired_phrases.txt<br/>julius_caesar.txt")] --> UI
```

**Design choices worth knowing:**

- **`lm.py` is the only place that talks to the model.** Every other module works on plain token ids
  and numbers, so it can be tested with tiny fake models (see `tests/`).
- **Beam search is hand-written** ([ADR 0001](adr/0001-custom-beam-search.md)). The library's
  built-in beam search hides which beams existed at each step; ours records it so you can see it.
- **KV cache.** To stay fast on a laptop, the model remembers its work on earlier tokens instead of
  re-reading the whole text every step. In beam search the cache is reshuffled to follow each beam's
  parent.

---

## 7. The three techniques side by side

| | Token-level masking | Sequence selection | Sequence regeneration |
|---|---|---|---|
| What gets blocked | Individual tokens | Whole beams (drafts) | Whole segments |
| When | Every step, before choosing | Every step, before choosing | After a segment is written |
| Rule must be checkable on… | the next token | the text so far | a finished chunk |
| Guarantee | Listed token sequences never appear | Survivors always have the best score | Every kept segment passes the check |
| Failure mode | Misses unlisted spellings | Scorer gets gamed | Gets stuck if nothing fits |
| In the app | Tab ① | Tabs ② and ④ | Tab ③ |

---

## 8. Where the app differs from the notebooks

| Notebook | The app | Why |
|---|---|---|
| Llama 3.2 3B on a GPU | Qwen2.5 0.5B / 1.5B on a laptop CPU | Runs on 8 GB RAM, no access token needed |
| Beam *sampling* (`do_sample=True`) | Deterministic beam search | Same result every run, easier to study |
| Phrases matched as plain substrings | Whole-word matching | Substring matching counts "add" inside "address" |
| Regeneration segments of 16 tokens | Segments end at a line break | A 16-token segment often spans two lines and fails on the second |
| LLM picks the target word and a seed line | You type the target word | The extra steps are prompt engineering, not the lesson |
| Regeneration prompt doesn't mention the next letter | Prompt hints the next letter | The small model rarely finds it otherwise; the check still enforces it |
| Autocomplete picks per token, at random | You pick per ~8-token chunk | Real autocomplete feel, same mechanism |

---

## Appendix A: how beam search picks the next beams

This walks through steps 3 → 6 of tab ② with the **default settings** (brand score,
"protein drink", 4 beams). Every number below is real, taken from the app's model
(Qwen2.5-0.5B on CPU, which is deterministic, so you'll see the same values).

### A.1 Where a token's probability comes from

For each beam, the model reads **the prompt + that beam's text so far** and outputs one logit for
every token in its vocabulary (~150,000). Softmax turns those into probabilities that add up to 1:

```
p(token | text so far) = e^(logit of token) / sum over all tokens of e^(logit)
```

This is a *conditional* probability: "how likely is this token, **given** this exact text before
it". The same token gets a different probability after "Boost your energy and" than after
"Boost your energy levels", because the model is reading a different text.

The app works with the **log** of the probability (`log p`), always ≤ 0:

| p | log p |
|---|---|
| 1.0 (certain) | 0 |
| 0.5 | −0.69 |
| 0.1 | −2.30 |
| 0.01 | −4.61 |

### A.2 A beam's total: multiply probabilities = add logs

The probability of a whole text is the product of the probabilities of its tokens, each one given
the tokens before it:

```
P("Boost your energy and") = P("Boost") × P(" your" | "Boost") × P(" energy" | "Boost your") × P(" and" | "Boost your energy")
```

Multiplying many small numbers quickly gives numbers too tiny for a computer to store accurately,
so we **add logs** instead (log(a × b) = log a + log b). That running sum is the beam's
**total log-probability**:

```
total(child) = total(parent) + log p(new token | parent's text)
```

Check it on real numbers: `Boost your energy` has total −2.952 (p = 0.0522). The model gives
` and` a probability of 0.536 after it (log p = −0.624). So
`Boost your energy and` = −2.952 + (−0.624) = **−3.576**, and e^−3.576 = 0.028 = 0.0522 × 0.536. ✓

### A.3 The rule for picking the next 4 beams

At every step:

1. Each surviving beam proposes its **top 4** next tokens. Masked beams' tokens all have log p = −10000.
2. That gives up to 4 × 4 = **16 candidates**, each with `total = parent total + log p(token)`.
3. Sort all 16 by total. The **best 4 overall** become the next beams, no matter which parent they
   came from. One parent can supply all 4 children, or none.

Why only the top 4 per beam? Because no beam can ever place more than 4 children. If its 4th best
token doesn't make the cut, its 5th best (which has an even lower total) can't either.

### A.4 Step 3 → 4: one survivor fills every slot

| Beam at step 3 | Total | Brand score | Status |
|---|---|---|---|
| `Boost your energy` | −2.952 | 0 | ✅ survives |
| `Experience the ultimate` | −3.134 | −1 | 🚫 masked |
| `Discover the ultimate` | −3.232 | −1 | 🚫 masked |
| `Experience the perfect` | −3.447 | −1 | 🚫 masked |

Candidates (top 8 of 16):

| Rank | From beam | Next token | p(token) | log p | Total | |
|---|---|---|---|---|---|---|
| 1 | Boost your energy | ` and` | 0.536 | −0.624 | **−3.576** | kept |
| 2 | Boost your energy | ` with` | 0.293 | −1.226 | **−4.178** | kept |
| 3 | Boost your energy | ` levels` | 0.060 | −2.813 | **−5.765** | kept |
| 4 | Boost your energy | `,` | 0.058 | −2.853 | **−5.805** | kept |
| 5–16 | the three masked beams | anything | 0 | −10000 | ≈ −10003 | dropped |

Because the masked beams' candidates are all around −10003, the 4 slots go to the survivor's top 4
tokens **even though ` levels` and `,` are weak** (6% each). Masking forced weak continuations in
simply because there was nothing else.

### A.5 Step 4 → 5: the weak beams lose (your question)

All four beams now score 0 on the brand score, so nothing is masked; this is plain beam search.

| Beam at step 4 | Total | p of whole text |
|---|---|---|
| `Boost your energy and` | −3.576 | 0.0280 |
| `Boost your energy with` | −4.178 | 0.0153 |
| `Boost your energy levels` | −5.765 | 0.0031 |
| `Boost your energy,` | −5.805 | 0.0030 |

All 16 candidates:

| Rank | From beam | Next token | p(token) | log p | Total | |
|---|---|---|---|---|---|---|
| 1 | …energy **with** | ` our` | 0.680 | −0.386 | **−4.565** | kept |
| 2 | …energy **and** | ` support` | 0.101 | −2.291 | **−5.867** | kept |
| 3 | …energy **and** | ` boost` | 0.089 | −2.421 | **−5.997** | kept |
| 4 | …energy **and** | ` muscle` | 0.086 | −2.455 | **−6.031** | kept |
| 5 | …energy with | ` this` | 0.147 | −1.916 | −6.094 | |
| 6 | …energy and | ` fuel` | 0.075 | −2.592 | −6.168 | |
| 7 | …energy with | ` a` | 0.133 | −2.016 | −6.194 | |
| 8 | …energy **levels** | ` and` | **0.504** | −0.685 | −6.450 | |
| 9 | …energy levels | ` with` | 0.443 | −0.815 | −6.579 | |
| 10 | …energy **,** | ` boost` | 0.172 | −1.762 | −7.567 | |
| 11 | …energy , | ` support` | 0.108 | −2.224 | −8.029 | |
| 12 | …energy , | ` enhance` | 0.101 | −2.296 | −8.102 | |
| 13 | …energy , | ` fuel` | 0.071 | −2.647 | −8.453 | |
| 14 | …energy with | ` the` | 0.013 | −4.340 | −8.518 | |
| 15 | …energy levels | `,` | 0.027 | −3.604 | −9.369 | |
| 16 | …energy levels | ` for` | 0.006 | −5.088 | −10.853 | |

**So, to answer the question:** yes. Every possible continuation of `…levels` and `…,` ends up
with a lower **total** than the 4 winners. Look at rank 8: ` and` after `levels` is a very likely
next token (50%, far better than ` support` at 10%). But the `levels` beam started this step far
behind:

```
"…energy and support"  = −3.576 + (−2.291) = −5.867   ✅ kept
"…energy levels and"   = −5.765 + (−0.685) = −6.450   ❌ dropped
```

The gap between the parents (−3.576 vs −5.765, about 9× in probability) is bigger than anything
one good next token can make up. And ` and` was `levels`' **best** option, so every other token
after `levels` did even worse. Same story for `,`: its best candidate, ` boost`, only reaches
−7.567.

In plain words: beam search asks **"which whole sentence-so-far is most likely?"**, not "which
next word is most likely?" A beam that took an unlikely turn earlier carries that cost with it.

### A.6 Step 5 → 6: the same rule again

| Rank | From beam (total) | Next token | p(token) | Total | |
|---|---|---|---|---|---|
| 1 | …and boost (−5.997) | ` your` | 0.826 | **−6.189** | kept |
| 2 | …with our (−4.565) | ` delicious` | 0.188 | **−6.236** | kept |
| 3 | …with our (−4.565) | ` high` | 0.156 | **−6.424** | kept |
| 4 | …with our (−4.565) | ` protein` | 0.127 | **−6.628** | kept |
| 5 | …and support (−5.867) | ` your` | 0.437 | −6.694 | |
| 7 | …and muscle (−6.031) | ` with` | 0.243 | −7.447 | |

Two things to notice:

- The strong parent `…with our` supplies **three** children.
- `…and boost` was only 3rd of 4 at step 5, but its next token is almost certain (` your`, 83%),
  which costs it almost nothing (−0.19). So it jumps to 1st place. The ranking can change at every
  step: a beam is kept if its **running total** stays in the top 4.

### A.7 Summary

| Question | Answer |
|---|---|
| How is a token's probability calculated? | The model reads the prompt + that beam's text, gives a logit per token, and softmax turns the logits into probabilities. |
| What is a beam's total? | The sum of the log-probabilities of all its tokens = log of the probability of the whole text. |
| How are the next 4 beams chosen? | Every surviving beam proposes its top 4 tokens; the 4 candidates with the highest totals win, from any parent. |
| Why did `levels` and `,` lose every child? | Their own totals were already low; even their best next token couldn't lift them above the children of `and` and `with`. |
| What does masking change? | A masked beam's tokens all get −10000, so its children rank last and never win a slot. Masking only decides **who may compete**; the totals decide who wins. |

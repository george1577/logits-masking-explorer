"""Logits Masking Explorer: a Streamlit app for seeing how logits masking steers an LLM.

Run with:  uv run streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from logits_lab.autocomplete import load_document
from logits_lab.beam import MASK_VALUE
from logits_lab.lm import MODELS, LanguageModel
from logits_lab.pipelines import (
    full_autocomplete_text,
    run_regeneration,
    run_sequence_selection,
    run_token_level,
    suggest_continuations,
)
from logits_lab.regeneration import line_initial
from logits_lab.scorers import find_phrases, load_phrases

DATA = Path(__file__).parent / "data"
DOC_LINK = "docs/how-logits-masking-works.md"

st.set_page_config(page_title="Logits Masking Explorer", page_icon="🎭", layout="wide")


@st.cache_resource(show_spinner="Loading model (first run downloads it)…")
def get_model(model_id: str) -> LanguageModel:
    return LanguageModel(model_id)


@st.cache_data
def get_phrases(name: str) -> list[str]:
    return load_phrases(DATA / name)


@st.cache_data
def get_play() -> list[str]:
    return load_document(DATA / "julius_caesar.txt")


def lines_of(text: str) -> list[str]:
    return [l.strip().lower() for l in text.splitlines() if l.strip()]


def visible(text: str) -> str:
    return text.replace("\n", "⏎ ")


# ---- sidebar ----------------------------------------------------------------------------

with st.sidebar:
    st.title("🎭 Logits Masking Explorer")
    model_id = st.selectbox("Model", MODELS, help="0.5B is fast on a laptop CPU. 1.5B writes better but is ~3x slower.")
    lm = get_model(model_id)
    st.caption(f"Running on **{lm.device}**")
    st.markdown(
        "**Learning path**\n"
        "1. Token-level masking\n"
        "2. Sequence selection\n"
        "3. Sequence regeneration\n"
        "4. Autocomplete\n\n"
        f"Read the companion guide: `{DOC_LINK}`"
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "① Token-level masking", "② Sequence selection", "③ Sequence regeneration", "④ Autocomplete",
])

# ---- ① token-level masking --------------------------------------------------------------

with tab1:
    st.header("① Token-level masking")
    st.markdown(
        "At every step the model gives a score (**logit**) to every token in its vocabulary. "
        "Token-level masking sets the logits of forbidden tokens to **−∞** before the next token is "
        "chosen, so their probability becomes exactly 0 and the rest is renormalised. "
        "Here a token is forbidden only if it would **complete a banned phrase** given what has been "
        "written so far. Both runs below use greedy decoding (always pick the top token)."
    )
    c1, c2 = st.columns([2, 1])
    t1_item = c1.text_input("Product to describe", "protein drink", key="t1_item")
    t1_max = c2.slider("Max new tokens", 20, 200, 100, step=10, key="t1_max")
    with st.expander("Banned phrases (editable, one per line)"):
        t1_banned = st.text_area("Banned phrases", "\n".join(get_phrases("banned_phrases.txt")),
                                 height=200, key="t1_banned", label_visibility="collapsed")

    if st.button("Generate both runs", type="primary", key="t1_go"):
        with st.spinner("Generating without and with the mask…"):
            st.session_state.t1 = run_token_level(lm, t1_item, lines_of(t1_banned), t1_max)

    res = st.session_state.get("t1")
    if res:
        banned = list(res.ban_list.tokenizations)
        left, right = st.columns(2)
        for col, title, run in ((left, "Without mask", res.unmasked), (right, "With mask", res.masked)):
            text = lm.decode(run.token_ids)
            hits = find_phrases(text, banned)
            col.subheader(title)
            col.write(text)
            col.caption(f"Banned phrases in output: {', '.join(hits) if hits else 'none ✅'}")

        changed = [s.step for s in res.masked.steps if s.mask_changed_choice]
        st.markdown(
            f"**The mask changed the model's top choice at {len(changed)} step(s):** "
            + (", ".join(map(str, changed)) if changed else "none. Try adding a phrase the model used.")
        )

        steps = res.masked.steps
        default = changed[0] if changed else 0
        step = st.slider("Inspect a step of the masked run", 0, len(steps) - 1, default, key="t1_step")
        s = steps[step]
        so_far = lm.decode(res.masked.token_ids[:step])
        st.markdown(f"Text so far: `…{visible(so_far[-120:])}` → next token **`{lm.token_label(s.chosen_id)}`**")
        st.caption(f"{s.num_banned} vocabulary entries were masked at this step.")

        b, a = st.columns(2)
        b.markdown("**Before masking** (raw model probabilities)")
        b.dataframe(pd.DataFrame({
            "token": [lm.token_label(c.token_id) for c in s.before],
            "probability": [c.prob for c in s.before],
            "masked?": ["🚫 −∞" if c.masked else "" for c in s.before],
        }), column_config={"probability": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)},
            hide_index=True, width="stretch")
        a.markdown("**After masking** (renormalised)")
        a.dataframe(pd.DataFrame({
            "token": [lm.token_label(c.token_id) for c in s.after],
            "probability": [c.prob for c in s.after],
        }), column_config={"probability": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)},
            hide_index=True, width="stretch")

        with st.expander("How the banned phrases tokenize"):
            st.markdown(
                "A phrase is usually **several tokens**, and the same words tokenize differently depending on "
                "capitalisation and a leading space (`␣`). Each variant below is a separate token sequence; "
                "the mask bans the **last** token of a sequence once the earlier ones have been generated."
            )
            needle = st.text_input("Filter phrases", "", key="t1_filter").lower()
            rows = [
                {"phrase": p, "variant": visible(v).replace(" ", "␣"),
                 "tokens": " | ".join(lm.token_label(i) for i in ids), "count": len(ids)}
                for p, entries in res.ban_list.tokenizations.items() if needle in p
                for v, ids in entries
            ]
            st.dataframe(pd.DataFrame(rows[:400]), hide_index=True, width="stretch")

# ---- ② sequence selection ---------------------------------------------------------------

with tab2:
    st.header("② Sequence selection")
    st.markdown(
        "Beam search keeps several candidate texts (**beams**) alive at once. At every step a **scorer** "
        "rates each beam's text so far; beams tied for the best score survive, and every other beam has "
        f"**all** of its logits replaced with **{MASK_VALUE:.0f}**, so none of its continuations can make "
        "the cut. The scorer is the swappable part."
    )
    scorer = st.radio("Scorer", ["brand", "alliteration"], horizontal=True, key="t2_scorer",
                      format_func=lambda s: {"brand": "Brand score (notebook 0)",
                                             "alliteration": "Alliteration score (notebook 1)"}[s])
    c1, c2, c3 = st.columns([2, 1, 1])
    if scorer == "brand":
        subject = c1.text_input("Product to describe", "protein drink", key="t2_item")
        with st.expander("Desired and banned phrases (editable)"):
            d, bcol = st.columns(2)
            t2_desired = d.text_area("Desired phrases", "\n".join(get_phrases("desired_phrases.txt")), height=200)
            t2_banned = bcol.text_area("Banned phrases", "\n".join(get_phrases("banned_phrases.txt")), height=200)
        st.caption("Brand score = number of desired phrases present − number of banned phrases present.")
    else:
        subject = c1.text_input("Animal", "donkey", key="t2_animal")
        t2_desired = t2_banned = ""
        st.caption(f"Alliteration score = number of words starting with **{subject.strip()[:1].upper()}**.")
    num_beams = c2.slider("Beams", 2, 6, 4, key="t2_beams")
    t2_max = c3.slider("Max new tokens", 20, 120, 60, step=10, key="t2_max")

    if st.button("Run beam search with and without selection", type="primary", key="t2_go"):
        with st.spinner("Running two beam searches…"):
            st.session_state.t2 = run_sequence_selection(
                lm, scorer, subject, lines_of(t2_desired), lines_of(t2_banned), num_beams, t2_max)
            st.session_state.t2_kind = scorer

    res = st.session_state.get("t2")
    if res:
        left, right = st.columns(2)
        for col, title, r in ((left, "Plain beam search (no masking)", res.baseline),
                              (right, "With sequence selection", res.selected)):
            col.subheader(title)
            col.write(r.best_text)
            col.caption(f"Score: **{res.score_fn(r.best_text)}**")

        if st.session_state.get("t2_kind") == "alliteration":
            st.info(
                "If the selected output turns into a repeated word, that is the scorer being **gamed**: the "
                "selector rewards nothing but the score. The notebook fights this by also asking for readability "
                "in the prompt and by sampling. Masking enforces a goal; it doesn't know about quality."
            )

        trace = res.selected.trace
        st.markdown("**Beams that survived at each step**")
        st.bar_chart(pd.DataFrame({"surviving beams": [t.num_kept for t in trace]}), height=160)

        step = st.slider("Inspect a step", 0, len(trace) - 1, min(10, len(trace) - 1), key="t2_step")
        st.dataframe(pd.DataFrame([{
            "beam": i,
            "text so far": "…" + visible(b.text[-90:]),
            "log-probability": round(b.logprob, 2),
            "score": b.score,
            "status": "✅ survives" if b.kept else f"🚫 masked ({MASK_VALUE:.0f})",
        } for i, b in enumerate(trace[step].beams)]), hide_index=True, width="stretch")

# ---- ③ sequence regeneration ------------------------------------------------------------

with tab3:
    st.header("③ Sequence regeneration")
    st.markdown(
        "Instead of masking inside every step, generate a short **segment** (here: one line), check it, and "
        "throw away segments that break the rule. Continue from the **most probable** segment that still "
        "fits. The rule here is an acrostic: the first letters of the lines must spell the **target word**."
    )
    c1, c2, c3, c4 = st.columns(4)
    animal = c1.text_input("Animal", "tiger", key="t3_animal")
    target = c2.text_input("Target word", "bold", key="t3_target").strip().lower()
    num_segments = c3.slider("Segments per round", 3, 8, 6, key="t3_segments")
    max_rounds = c4.slider("Max rounds", 4, 12, 10, key="t3_rounds")

    if st.button("Write the acrostic", type="primary", key="t3_go", disabled=not target.isalpha()):
        with st.spinner("Generating segments round by round…"):
            st.session_state.t3 = run_regeneration(lm, animal, target, num_segments, max_rounds=max_rounds)

    res = st.session_state.get("t3")
    if res:
        badge = {"complete": "✅ Complete", "stuck": "⚠️ Stuck: no segment fit",
                 "out of rounds": "⏱️ Out of rounds"}[res.status]
        st.subheader(f"{badge} · target **{res.target.upper()}**")
        if res.poem_lines:
            st.markdown("  \n".join(
                f"**{l[0]}**{l[1:]}" if l else l for l in res.poem_lines))
        if res.status == "stuck":
            st.caption("This is the technique's real failure mode: it can only choose among the segments the "
                       "model offers. Try another target word, more segments, or the 1.5B model.")

        for rnd in res.rounds:
            chosen = rnd.chosen
            label = f"Round {rnd.number}: " + (f"chose “{visible(chosen.segment.text).strip()}”" if chosen
                                                else "no segment fit")
            with st.expander(label, expanded=rnd is res.rounds[-1]):
                st.caption(f"Poem before this round: `{visible(rnd.poem_before) or '(empty)'}`")
                st.dataframe(pd.DataFrame([{
                    "segment": visible(v.segment.text),
                    "log-probability": round(v.segment.logprob, 2),
                    "initials so far": v.check.initials.upper(),
                    "fits?": "✅" if v.check.fits else "❌",
                    "chosen": "⭐" if v.chosen else "",
                } for v in rnd.verdicts]), hide_index=True, width="stretch")

# ---- ④ autocomplete ---------------------------------------------------------------------

with tab4:
    st.header("④ Autocomplete")
    st.markdown(
        "Sequence selection where **you** are the selector. The model, grounded on lines from "
        "*Julius Caesar* that contain what you typed, offers several beams. Clicking one is the same as "
        f"keeping that beam and setting every other beam's logits to **{MASK_VALUE:.0f}**."
    )
    ac = st.session_state.setdefault("ac", {"prefix": "", "accepted": "", "sugg": None, "history": []})

    c1, c2 = st.columns([3, 1])
    typed = c1.text_input("Start typing a phrase", "Lend", key="t4_typed")
    if c2.button("Start / reset", type="primary", key="t4_start"):
        ac.update(prefix=typed.strip(), accepted="", sugg=None, history=[])

    def accept(i: int):
        option = ac["sugg"].options[i]
        ac["history"].append({"options": ac["sugg"].options, "chosen": i})
        ac["accepted"] += option
        ac["sugg"] = None

    def undo():
        if ac["history"]:
            last = ac["history"].pop()
            ac["accepted"] = ac["accepted"][: len(ac["accepted"]) - len(last["options"][last["chosen"]])]
            ac["sugg"] = None

    if ac["prefix"]:
        if ac["sugg"] is None:
            with st.spinner("Asking the model for continuations…"):
                ac["sugg"] = suggest_continuations(lm, get_play(), ac["prefix"], ac["accepted"])
        sugg = ac["sugg"]

        main, side = st.columns([3, 2])
        with main:
            st.markdown(f"### {full_autocomplete_text(ac['prefix'], ac['accepted'])}▌")
            st.caption("Pick a continuation:")
            for i, option in enumerate(sugg.options):
                label = visible(option).strip() or "⟨end of sentence⟩"
                st.button(label, key=f"t4_opt_{len(ac['history'])}_{i}", on_click=accept, args=(i,),
                          width="stretch", disabled=not option.strip())
            st.button("↩ Undo last pick", on_click=undo, disabled=not ac["history"], key="t4_undo")
            with st.expander(f"Grounding: {len(sugg.examples)} line(s) of the play contain “{ac['prefix']}”"):
                st.text("\n".join(sugg.examples) or "(none: the model is on its own)")
        with side:
            st.markdown("**What your last click did to the beams**")
            if ac["history"]:
                last = ac["history"][-1]
                st.dataframe(pd.DataFrame([{
                    "beam": visible(o).strip() or "⟨end⟩",
                    "logits": "kept ✅" if j == last["chosen"] else f"{MASK_VALUE:.0f} 🚫",
                } for j, o in enumerate(last["options"])]), hide_index=True, width="stretch")
            else:
                st.caption("Nothing yet: pick a continuation.")
            st.markdown("**Model's ranking of the current options**")
            st.dataframe(pd.DataFrame({
                "option": [visible(o).strip() or "⟨end⟩" for o in sugg.options],
                "log-probability": [round(lp, 2) for lp in sugg.logprobs],
            }), hide_index=True, width="stretch")
    else:
        st.caption("Type a word or two that appears in the play (e.g. “Lend”, “Beware”, “Et tu”) and press Start.")

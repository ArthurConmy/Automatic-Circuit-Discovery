import streamlit as st
from streamlit.components.v1 import html
from pathlib import Path
from transformer_lens import HookedTransformer
from transformer_lens.rs.callum.streamlit.my_styling import styling

styling()

from transformer_lens.rs.callum.demo_page_backend import (
    generate_4_html_plots,
    parse_str_tok_for_printing,
)

if "model" not in st.session_state:
    with st.spinner("Loading model (this only needs to happen once) ..."):
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device="cpu"
        )
        model.set_use_attn_result(True)
    
    st.session_state["model"] = model

input_text = st.text_area("Input text", "All's fair in love and war.")

def generate():
    assert input_text is not None
    assert input_text != ""
    with st.spinner("Generating data from prompt..."):
        st.session_state["input_text"] = input_text
        input_toks = st.session_state["model"].to_tokens(st.session_state["input_text"])
        input_str_toks = st.session_state["model"].to_str_tokens(st.session_state["input_text"])
        input_str_toks_parsed = [list(map(parse_str_tok_for_printing, input_str_toks))]
        LOSS_DIFF, LOGITS_ORIG, LOGITS_ABLATED, LOGITS_DIRECT, ATTN = generate_4_html_plots(
            model=st.session_state["model"],
            data_toks=input_toks,
            data_str_toks_parsed=input_str_toks_parsed
        )
        st.session_state["LOSS_DIFF"] = LOSS_DIFF.strip().split("\n</div>\n\n")
        st.session_state["LOGITS_ORIG"] = LOGITS_ORIG.strip().split("\n</div>\n\n")
        st.session_state["LOGITS_ABLATED"] = LOGITS_ABLATED.strip().split("\n</div>\n\n")
        st.session_state["LOGITS_DIRECT"] = LOGITS_DIRECT.strip().split("\n</div>\n\n")
        st.session_state["ATTN"] = ATTN.strip().split("\n</div>\n\n")

button = st.button("Generate", on_click=generate)

NEG_HEADS = ["10.7", "11.10"]
ABLATION_TYPES = ["mean, direct", "zero, direct", "mean, patched", "zero, patched"]
ATTENTION_TYPES = ["info-weighted", "standard"]
ATTN_VIS_TYPES = ["large", "small"]

if st.session_state.get("LOSS_DIFF", None) is not None:

    LOSS_DIFF = st.session_state["LOSS_DIFF"]
    LOGITS_ORIG = st.session_state["LOGITS_ORIG"]
    LOGITS_ABLATED = st.session_state["LOGITS_ABLATED"]
    LOGITS_DIRECT = st.session_state["LOGITS_DIRECT"]
    ATTN = st.session_state["ATTN"]

    BATCH_SIZE = len(LOSS_DIFF) // (len(NEG_HEADS) * len(ABLATION_TYPES))
    assert BATCH_SIZE == 1

    batch_idx = 0

    loss_diff_keys = [
        (batch_idx, head, ablation_type)
        for batch_idx in range(BATCH_SIZE)
        for head in NEG_HEADS
        for ablation_type in ABLATION_TYPES
    ]
    logits_direct_keys = [
        (batch_idx, head)
        for batch_idx in range(BATCH_SIZE)
        for head in NEG_HEADS
    ]
    logits_orig_keys = [
        batch_idx
        for batch_idx in range(BATCH_SIZE)
    ]
    attn_keys = [
        (batch_idx, head, attn_vis_type, attn_type)
        for batch_idx in range(BATCH_SIZE)
        for head in NEG_HEADS
        for attn_vis_type in ATTN_VIS_TYPES
        for attn_type in ATTENTION_TYPES[::-1]
    ]
    LOSS_DIFF = {k: v for k, v in zip(loss_diff_keys, LOSS_DIFF)}
    LOGITS_ABLATED = {k: v for k, v in zip(loss_diff_keys, LOGITS_ABLATED)}
    LOGITS_ORIG = {k: v for k, v in zip(logits_orig_keys, LOGITS_ORIG)}
    LOGITS_DIRECT = {k: v for k, v in zip(logits_direct_keys, LOGITS_DIRECT)}
    ATTN = {k: v for k, v in zip(attn_keys, ATTN)}

    head = st.sidebar.radio("Pick a head", NEG_HEADS + ["both"])
    assert head != "both", "Both not implemented yet."
    ablation_type = st.sidebar.radio("Pick a type of ablation", ABLATION_TYPES)
    attention_type = st.sidebar.radio("Pick a type of attention", ATTENTION_TYPES)
    attention_vis_type = st.sidebar.radio("Pick a type of attention view", ATTN_VIS_TYPES)

    head_name = head.replace(".", "")

    st.markdown(
r"""Navigate through the four tabs below to see the different model visualisations.

### What is this page for?

### How does each visualisation relate to copy-suppression?

### A few good example prompts

<details>
<summary>All's fair in love and war.</summary>

The model will predict `"... love and love"` when you ablate 10.7!

This is an example of 10.7 suppressing **naive copying**.

</details>

<details>
<summary>I picked up the first box. I picked up the second box. I picked up the third and final box.</summary>

This is a great example of situations where copy-suppression is good/bad respectively. The model will copy-suppress `" box"` after the tokens `" second"` and `" final"` (which is bad because `" box"` was actually correct here), but it will also heavily copy suppress `" box"` after `" third"`, which is good because `" box"` was incorrect here.

This is an example of 10.7 suppressing **naive induction** (specifically, naive fuzzy induction). More generally, it's an example of **breaking the pattern**; reducing the model's overconfidence.

There's also some copy-suppression for `I -> picked`. There isn't copy-suppression for other words in the induction pattern e.g. `picked -> up`, `up -> the`, or `. -> I` are not copy-suppressed, because these are function words.

</details>

<br><br>
""", unsafe_allow_html=True)

    tabs = st.tabs([
        "Loss difference from ablating negative head",
        "Direct effect on logits",
        "Attention patterns",
        "Prediction-attention?",
    ])

    with tabs[0]:
        st.markdown(
r"""
# Loss difference from ablating negative head
""", unsafe_allow_html=True)

        html(LOSS_DIFF[(batch_idx, head, ablation_type)], height=200)

    with tabs[1]:
        st.markdown(
r"""
# Direct effect on logits
""", unsafe_allow_html=True)

        html(LOGITS_DIRECT[(batch_idx, head)], height=300)

        st.markdown(
r"""
## Model predictions, before / after ablation
""", unsafe_allow_html=True)

        cols = st.columns(2)

        with cols[0]:
            st.markdown("### Original")
            html(LOGITS_ORIG[batch_idx], height=400)
        with cols[1]:
            st.markdown("### Ablated")
            html(LOGITS_ABLATED[(batch_idx, head, ablation_type)], height=400)

    with tabs[2]:
        st.markdown(
r"""
# Attention patterns
""", unsafe_allow_html=True)

        html(ATTN[(batch_idx, head, attention_vis_type, attention_type)], height=800)

    with tabs[3]:
        st.markdown(
r"""
# How much of the unembedding is in the residual stream?
""")

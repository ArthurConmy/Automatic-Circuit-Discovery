import streamlit as st
from streamlit.components.v1 import html
from pathlib import Path

from transformer_lens.rs.callum.demo_page_backend import (
    ModelResults
)

PATH = Path("/home/ubuntu/Transformerlens/transformer_lens/rs/callum/streamlit/explore_prompts")
LOSS_DIFF_PATH = PATH / "loss_difference_from_ablating.html"
LOGITS_ORIG_PATH = PATH / "logits_orig.html"
LOGITS_ABLATED_PATH = PATH / "logits_ablated.html"
LOGITS_DIRECT_PATH = PATH / "logits_direct.html"
ATTN_PATH = PATH / "attn_patterns.html"
NEG_HEADS = ["10.7", "11.10"]
ABLATION_TYPES = ["mean, direct", "zero, direct", "mean, patched", "zero, patched"]
ATTENTION_TYPES = ["info-weighted", "standard"]
ATTN_VIS_TYPES = ["large", "small"]

with open(LOSS_DIFF_PATH) as f:
    LOSS_DIFF = f.read().split("</div>\n")
with open(LOGITS_ORIG_PATH) as f:
    LOGITS_ORIG = f.read().split("</div>\n")
with open(LOGITS_ABLATED_PATH) as f:
    LOGITS_ABLATED = f.read().split("</div>\n")
with open(LOGITS_DIRECT_PATH) as f:
    LOGITS_DIRECT = f.read().split("</div>\n")
with open(ATTN_PATH) as f:
    ATTN = f.read().split("</div>\n")
BATCH_SIZE = len(LOSS_DIFF) // (len(NEG_HEADS) * len(ABLATION_TYPES))

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

first_idx = 36 if (36 < BATCH_SIZE) else 0
batch_idx = st.sidebar.slider("Pick a sequence", 0, BATCH_SIZE, first_idx)
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
""")

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

This visualisation shows the loss difference from ablating head 10.7 (for various different kinds of ablation).

If we want to answer "what is a particular head near the end of the model doing, and why is it useful?" then it's natural to look at the cases where ablating it has a large effect on the model's loss.

Our theory is that **negative heads detect tokens which are being predicted (query-side), attend back to previous instances of that token (key-side), and negatively copy them, thereby suppressing the logits on that token.** The 2 plots after this one will provide evidence for this.

We'll use as an example the string `"...whether Bourdain Market will open at the pier"`, specifically the `"at the pier"` part. But we think these results hold up pretty well in most cases where ablating a head has a large effect on loss (i.e. this result isn't cherry-picked).
""", unsafe_allow_html=True)

    html(LOSS_DIFF[(batch_idx, head, ablation_type)], height=200)

with tabs[1]:
    st.markdown(
r"""
# Direct effect on logits

Hover over token T to see what the direct effect of 10.7 is on the logits for T (as a prediction).

The notable observation - **for most of the examples where ablating 10.7 has a large effect on loss, you can see from here that they're important because they push down a token by a lot which appeared earlier in context.**

Take our `at the pier` example. The head is pushing down the prediction for `pier` by a lot, both following `at` and following `the`. In the first case this is helpful, because `pier` actually didn't come next. But in the second case it's unhelpful, because `pier` did come next.

To complete this picture, we still want to look at the attention patterns, and verify that the head is attending to the token it's pushing down on. Note that, to make the logits more interpretable, I've subtracted their mean (so they're "logits with mean zero" not "logprobs").

""", unsafe_allow_html=True)

    html(LOGITS_DIRECT[(batch_idx, head)], height=300)

    st.markdown(
r"""In the columns below, you can also compare whole model's predictions before / after ablation. You can see, for instance, that the probability the model assigns to ` Pier` following both the `at` and `the` tokens increases by a lot when we ablate head 10.7. For `(at, the)`, the `Pier`-probability goes from 63.80% to 93.52% when we ablate, which pushes the `the`-probability down (which in this case is bad for the model), and for `(the, pier)`, the `pier`-probability goes from 0.86% to 2.41% (which in this case is good for the model).

Note that the negative head is suppressing both `pier` and `Pier`, because they have similar embeddings. As we'll see below, it's actually suppressing `Pier` directly, and suppressing `pier` is just a consequence of this.

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

We'll conclude with a look at attention patterns. We expect to see both `at` and `the` attending back to `Pier` - this is indeed what we find.

Note that the other two clear examples of a nontrivial attention pattern in this example also seem like they could be examples of "unembedding of token T attending to previous embedding of token T":

* `questioning` and `whether` attend to `Anthony` and `Bour`
    * It seems reasonable that `Anthony` and `Bour` are being predicted at these points. From the first plot, we can see that this is bad in the `questioning` case, because `Bour` was actually correct.
* `Eater` attends to `NY`
    * This is probably induction-suppression (i.e. `NY` was predicted because of induction, and this forms 10.7's attn pattern). In this case, it's good, because `NY` didn't come first.
""", unsafe_allow_html=True)

    html(ATTN[(batch_idx, head, attention_vis_type, attention_type)], height=800)

with tabs[3]:
    st.markdown(
r"""
# How much of the unembedding is in the residual stream?

To conclude this section, we'll take a look at all the nontrivial attention patterns in the model, and see how many of them involve the unembedding of a token it attends to having a high dot product with the residual stream, relative to the average source token unembedding in that sequence.
"""
)

# %%

from transformer_lens.cautils.notebook import *

import markdown


from transformer_lens.rs.callum.demo_page_backend import (
    parse_str,
    parse_str_tok_for_printing,
    HeadResults,
    LogitResults,
    ModelResults,
    get_data_dict,
    topk_of_Nd_tensor,
)

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
model.set_use_split_qkv_input(False)
model.set_use_attn_result(True)

def to_string(toks):
    s = model.to_string(toks)
    s = s.replace("\n", "\\n")
    return s

def parse_str(s: str):
    doubles = "“”"
    singles = "‘’"
    for char in doubles: s = s.replace(char, '"')
    for char in singles: s = s.replace(char, "'")
    return s

def parse_str_tok_for_printing(s: str):
    s = s.replace("\n", "\\n")
    return s

# %%

BATCH_SIZE = 100
SEQ_LEN = 150 # 1024

DATA_STR = get_webtext(seed=6)[:BATCH_SIZE]
DATA_STR = [parse_str(s) for s in DATA_STR]

DATA_TOKS = model.to_tokens(DATA_STR)
DATA_STR_TOKS = model.to_str_tokens(DATA_STR)

if SEQ_LEN < 1024:
    DATA_TOKS = DATA_TOKS[:, :SEQ_LEN]
    DATA_STR_TOKS = [str_toks[:SEQ_LEN] for str_toks in DATA_STR_TOKS]

DATA_STR_TOKS_PARSED = [[parse_str_tok_for_printing(tok) for tok in toks] for toks in DATA_STR_TOKS]

print(DATA_TOKS.shape, "\n")

print(DATA_STR_TOKS[0])


# %% [markdown]

# # Data

# Here's where I gather data for the other visualisations.

# %%

MODEL_RESULTS = get_data_dict(model, DATA_TOKS, negative_heads = [(10, 7), (11, 10)])

# %% [markdown]

# # Activations

# Here's where I get the activation plots, where each value actually shows the effect on logits of ablating.

# We show `(ablated loss) - (original loss)`, so blue (positivity) indicates that this head is helpful, because loss goes up when it gets ablated. 

# %%

batch_idx = 36

assert MODEL_RESULTS.loss_orig.shape == MODEL_RESULTS.loss.mean_patched[(10, 7)].shape == (BATCH_SIZE, SEQ_LEN - 1)

loss_diffs = t.stack([
    t.stack(list(MODEL_RESULTS.loss.mean_direct.data.values())),
    t.stack(list(MODEL_RESULTS.loss.zero_direct.data.values())),
    t.stack(list(MODEL_RESULTS.loss.mean_patched.data.values())),
    t.stack(list(MODEL_RESULTS.loss.zero_patched.data.values())),
]) - MODEL_RESULTS.loss_orig
loss_diffs_padded = t.concat([loss_diffs, t.zeros((4, 2, BATCH_SIZE, 1))], dim=-1)
loss_diffs_padded = einops.rearrange(
    loss_diffs_padded, "loss_type head batch seq -> batch seq loss_type head"
)

html = cv.activations.text_neuron_activations(
    tokens = DATA_STR_TOKS_PARSED[batch_idx],
    activations = [loss_diffs_padded[batch_idx]],
    first_dimension_name = "loss_type",
    first_dimension_labels = ["mean, direct", "zero, direct", "mean, patched", "zero, patched"],
    second_dimension_name = "head",
    second_dimension_labels = ["10.7", "11.10"],
)
html = markdown.markdown(
f"""# Loss difference from ablating head 10.7

This shows the loss difference from ablating head 10.7 (for various different kinds of ablation).

This is the first plot you should look at. Once you look at this plot, there are 2 more to look at:

* Direct logit effect plots - in these cases, what is 10.7 pushing up/down on? (is it like we think, pushing down on a token which was predicted & exists earlier in context?)
* Attention plots - what is 10.7 attending to? (is it like we think, attending to the thing it's negatively copying?)

We'll be zeroing on the `at the pier` text about 1/3 of the way through, because it's an elegant example. Most others also hold up if you investigate them.
""") + "<br><hr><br>" + str(html)

with open("loss_difference_from_ablating.html", "w") as file:
    file.write(html)

# %% [markdown]

# I also want to be able to print out what the biggest ones are.

# Do the top 5 results here hold up to sanity checks, i.e. do they look like copy suppression?

# %%

def to_string(toks):
    s = model.to_string(toks)
    s = s.replace("\n", "\\n")
    return s

display(cv.logits.token_log_probs(
    DATA_TOKS[batch_idx].cpu(),
    MODEL_RESULTS.logits_orig[batch_idx].log_softmax(-1),
    to_string = to_string
))

display(cv.logits.token_log_probs(
    DATA_TOKS[batch_idx].cpu(),
    MODEL_RESULTS.logits.mean_direct[10, 7][batch_idx].log_softmax(-1),
    to_string = to_string
))

html = cv.logits.token_log_probs(
    DATA_TOKS[batch_idx].cpu(),
    MODEL_RESULTS.direct_effect[10, 7][batch_idx].log_softmax(-1),
    to_string = to_string,
    top_k = 10,
    negative = True,
)

html = markdown.markdown(
f"""# Direct logit effect from head 10.7

Hover over token T to see what the direct effect of 10.7 is on the logits for T (as a prediction).

The notable observation - **for most of the examples where ablating 10.7 has a large effect on loss, you can see from here that they're important because they push down a token by a lot which appeared earlier in context.**

* **For examples where 10.7 is unhelpful, it's accidentally pushing down a token which was correct.**
    * Example - `[' the', ' pier']` near the start. 
    * Presumably because `' pier'` was already predicted, so we attend back to `' Pier'` earlier in context (interesting that it's not exactly the same word, but it's close enough!) and negatively copy it.
    * And this ended up being bad, because `' pier'` was actually correct.
* **For examples where 10.7 is helpful, it's pushing down a token which wasn't correct.**
    * Example - `[' at', ' the', ' pier']` near the start.
    * Presumably the model is predicting `[' at', ' Pier']` because it copies this from earlier in context.
    * So head 10.7 attends back to & negatively copies `' pier'`. And this ended up being good, because `' pier'` was actually incorrect - the correct answer was `' the'`.

To confirm, this, you can see the attention patterns.""") + "<br><hr><br>" + str(html)

with open("direct_logit_effect_from_107.html", "w") as file:
    file.write(html)

# %%

loss_diffs_mean_direct_107 = loss_diffs[2, 0]

most_useful_positions = topk_of_Nd_tensor(loss_diffs_mean_direct_107, 5)

for batch_idx, seq_pos in most_useful_positions:
    print("\n".join([
        f"Batch = {batch_idx}",
        f"Seq pos = {seq_pos}",
        f"Loss increase from ablation = {loss_diffs_mean_direct_107[batch_idx, seq_pos]}",
        f"Text = {''.join(DATA_STR_TOKS_PARSED[batch_idx][seq_pos-10: seq_pos+1])}",
        ""
    ]))

# %% [markdown]

# # Attention Patterns

# %%

batch_idx = 36

# TODO - value-weighted!

weighted_attn_107 = einops.einsum(
    MODEL_RESULTS.pattern[10, 7][batch_idx],
    MODEL_RESULTS.out_norm[10, 7][batch_idx] / MODEL_RESULTS.out_norm[10, 7][batch_idx].max(),
    "seqQ seqK, seqK -> seqQ seqK"
)
weighted_attn_1110 = einops.einsum(
    MODEL_RESULTS.pattern[11, 10][batch_idx],
    MODEL_RESULTS.out_norm[11, 10][batch_idx] / MODEL_RESULTS.out_norm[11, 10][batch_idx].max(),
    "seqQ seqK, seqK -> seqQ seqK"
)

html = cv.attention.attention_heads(
    attention = t.stack([weighted_attn_107, weighted_attn_1110])[:, :45, :45], # (heads=2, seqQ, seqK)
    tokens = DATA_STR_TOKS_PARSED[batch_idx][:45], # list of length seqQ
    attention_head_names = ["10.7", "11.10"],
)

html = markdown.markdown(
f"""# Attention patterns

We conclude by looking at attention patterns. As expected, we see both `' at'` and `' the'` attending back to `' Pier'`.
""") + "<br><hr><br>" + str(html)

with open("attention_patterns.html", "w") as file:
    file.write(html)

# %%
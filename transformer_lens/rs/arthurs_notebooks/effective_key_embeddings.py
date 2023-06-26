# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_split_qkv_normalized_input(True)
DEVICE = "cuda"
LAYER_IDX, HEAD_IDX = NEG_HEADS[model.cfg.model_name]
INCLUDE_ORTHOGONAL = True

# %%

# goal: see which key component is best
# setup: do IOI

N = 200
warnings.warn("Auto IOI")
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=True,
    seed=35795,
    device=DEVICE,
)
update_word_lists = {" " + sent.split()[-1]: sent for sent in ioi_dataset.sentences}
assert len(update_word_lists) == len(set(update_word_lists.keys())), "Non-uniqueness!"

# %%

# (...continued)
#
# Let's edit the keys. Let's compare the components in the directions of
# and the normalized attention I guess?
#
# Baselines:
# - Full
# - MLP0
# - My `<BOS>The word` trick
# - Callum locked attentions
# ???

# %%

# Cache The Inputs
_, cache = model.run_with_cache(
    ioi_dataset.toks,
    names_filter=lambda name: name
    in [f"blocks.{LAYER_IDX}.hook_q_input", f"blocks.{LAYER_IDX}.hook_k_input", f"blocks.{LAYER_IDX}.attn.hook_attn_scores", "blocks.0.hook_mlp_out"],
)

# batch, pos, n_heads, d_model
cached_query_input = cache["blocks.{}.hook_q_input".format(LAYER_IDX)][
    torch.arange(N), ioi_dataset.word_idx["end"], HEAD_IDX, :
]
cached_key_input = cache["blocks.{}.hook_k_input".format(LAYER_IDX)][
    torch.arange(N), ioi_dataset.word_idx["IO"], HEAD_IDX, :
]
cached_mlp0 = cache["blocks.0.hook_mlp_out"][torch.arange(N), ioi_dataset.word_idx["IO"], :]
cached_attention_scores = cache["blocks.{}.attn.hook_attn_scores".format(LAYER_IDX)][torch.arange(N), HEAD_IDX, ioi_dataset.word_idx["end"], ioi_dataset.word_idx["IO"]]
assert list(cached_attention_scores.shape) == [N], cached_attention_scores.shape

assert list(cached_query_input.shape) == [
    N,
    model.cfg.d_model,
], cached_query_input.shape

#%%

unnormalized_query_input = cached_query_input

# %%

W_Q = model.W_Q[LAYER_IDX, HEAD_IDX]
W_K = model.W_K[LAYER_IDX, HEAD_IDX]

def dot_with_query(
    unnormalized_keys,
    unnormalized_queries = unnormalized_query_input,
):
# unnormalized_queries = unnormalized_query_input
# unnormalized_keys = cached_key_input
# if True:
    queries_normalized = torch.stack(
        [query / (query.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)
        for query in unnormalized_queries],
        dim=0,
    )
    keys_normalized = torch.stack(
        [key / (key.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)
        for key in unnormalized_keys],
        dim=0,
    )

    q_and_k_vectors = list(zip(queries_normalized, keys_normalized))

    results = []
    for q_vector, k_vector in tqdm(q_and_k_vectors): # TODO easy to batch, mate...
        query_side_vector = einops.einsum(
            q_vector,
            W_Q,
            "d_model, d_model d_head -> d_head",
        ) + model.b_Q[LAYER_IDX, HEAD_IDX]
        
        # TODO to do this addition maximally safe, assert some shapes and/or einops.repeat the bias
        key_side_vector = einops.einsum(
            k_vector,
            W_K,
            "d_model, d_model d_head -> d_head",
        ) + model.b_K[LAYER_IDX, HEAD_IDX]

        assert list(query_side_vector.shape) == [
            model.cfg.d_head,
        ], query_side_vector.shape
        assert list(key_side_vector.shape) == [
            model.cfg.d_head,
        ], key_side_vector.shape

        attention_scores = einops.einsum(
            query_side_vector,
            key_side_vector,
            "d_head, d_head ->",
        ) / np.sqrt(model.cfg.d_head)
        results.append(attention_scores.item())
        # assert False
    return torch.tensor(results)

results = dot_with_query(cached_key_input)
assert torch.allclose(results.cpu(), cached_attention_scores.cpu(), atol=1e-2, rtol=1e-2), (results.norm().item(), cached_attention_scores.norm().item(), "dude the assertion is 1e-2 this sure should work!")

#%%

the_inputs = model.to_tokens(
    "The " + sentence.split()[-1] for sentence in ioi_dataset.sentences
)
assert list(the_inputs.shape) == [N, 3], the_inputs.shape # "<bos>The IO"
_, cache = model.run_with_cache(
    the_inputs,
    names_filter=lambda name: f"blocks.{LAYER_IDX}.hook_resid_pre" == name,
)
the_residuals = cache["blocks.{}.hook_resid_pre".format(LAYER_IDX)][:, -1]

_, cache = model.run_with_cache(
    ioi_dataset.toks,
    names_filter = lambda name: name.endswith("attn.hook_pattern"),
) 

def attn_lock(z, hook, ioi_dataset, scaled=False):
    old_z = z.clone()
    z[:] *= 0.0

    if scaled:
        for idx in range(N):
            att_io = old_z[idx, :, ioi_dataset.word_idx["IO"][idx], ioi_dataset.word_idx["IO"][idx]]
            att_bos = old_z[idx, :, ioi_dataset.word_idx["IO"][idx], 0]

            z[idx, :, ioi_dataset.word_idx["IO"][idx], ioi_dataset.word_idx["IO"][idx]] = att_io/(att_io + att_bos)
            z[idx, :, ioi_dataset.word_idx["IO"][idx], 0] = att_bos/(att_io + att_bos)

    else:
        # this thing is vectorized but difficult to extend
        all_key_positions = [[0 for _ in range(N)], ioi_dataset.word_idx["IO"]]
        for key_positions in all_key_positions:
            z[torch.arange(N), :, ioi_dataset.word_idx["IO"], key_positions] = old_z[torch.arange(N), :, ioi_dataset.word_idx["IO"], key_positions] 

    return z

callums_baselines = []

for scaled in [False, True]:
    model.reset_hooks()
    for layer in range(LAYER_IDX):
        model.add_hook(
            "blocks.{}.attn.hook_pattern".format(layer),
            partial(attn_lock, scaled=scaled, ioi_dataset=ioi_dataset),
            level=1,
        )
    _, cache = model.run_with_cache(
        ioi_dataset.toks,
        names_filter = lambda name: name==f"blocks.{LAYER_IDX}.hook_resid_pre",
    )
    model.reset_hooks()
    callums_baseline = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"][torch.arange(N), ioi_dataset.word_idx["IO"]]
    callums_baselines.append(callums_baseline)

callums_baseline, callums_baseline_scaled = callums_baselines

#%%

# Baselines:
# - Full
# - MLP0
# - My `<BOS>The word` trick
# - Callum locked attentions
# ???

baselines = {
    "Actual key inputs": cached_key_input,
    "MLP0": cached_mlp0,
    "`The` baseline": the_residuals,
    "Callum's baseline": callums_baseline,
    # "Callum's baseline (scaled)": callums_baseline_scaled,
}

#%%

# orthogonal components 

OTHER=False

if INCLUDE_ORTHOGONAL:
    old_baseline_keys = list(baselines.keys())
    for baseline_name in old_baseline_keys:
        if baseline_name.startswith("Actual"):
            continue
        normalized_baseline = baselines[baseline_name] / baselines[baseline_name].norm(dim=-1, keepdim=True)
        normalized_keys = cached_key_input / cached_key_input.norm(dim=-1, keepdim=True)

        if OTHER:
            other_orthogonal_complement = normalized_keys - normalized_baseline
            baselines["Other complement to " + baseline_name] = other_orthogonal_complement

        else:
            orthogonal_complement = normalized_keys - einops.einsum(
                normalized_baseline,
                normalized_keys,
                "batch d_model, batch d_model -> batch",
            ).unsqueeze(-1) * normalized_baseline

            orthogonal_complement = orthogonal_complement / orthogonal_complement.norm(dim=-1, keepdim=True)

            assert einops.einsum(
                orthogonal_complement,
                normalized_baseline,
                "batch d_model, batch d_model -> batch",
            ).abs().max().item() < 1e-5, "Orthogonal complement is not orthogonal to the baseline"    

            assert torch.allclose(
                orthogonal_complement.norm(dim=-1),
                normalized_keys.norm(dim=-1),
                atol=1e-5,
                rtol=1e-5,
            ), "Orthogonal complement is not the same norm as the keys"

            baselines["Orthogonal complement of " + baseline_name] = orthogonal_complement

#%%

histone = {
    key: einops.einsum(
        cached_key_input / cached_key_input.norm(dim=-1, keepdim=True),
        value / value.norm(dim=-1, keepdim=True),
        "batch d_model, batch d_model -> batch",
    ) for key, value in baselines.items()
}

#%%

hist(
    list(histone.values()),
    title="Cosine sim of baselines with the actual keys",
    names=list(baselines.keys()),
    width=800,
    height=600,
    opacity=0.7,
    marginal="box",
    template="simple_white",
    nbins=50,
    # static=True,
)

hist(
    [
        dot_with_query(
            baseline,
        ) for baseline in baselines.values()
    ],
    title="Attention Scores of Different Keys (these keys scaled to norm 1)",
    names=list(baselines.keys()),
    width=800,
    height=600,
    opacity=0.7,
    marginal="box",
    template="simple_white",
    # static=True,
)

# %%

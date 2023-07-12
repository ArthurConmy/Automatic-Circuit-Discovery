#%% [markdown] [1]:

"""
Mostly cribbed from transformer_lens/rs/callum/orthogonal_query_investigation_2.ipynb
(but I prefer .py investigations)
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthur_utils import dot_with_query
from transformer_lens.rs.callum.keys_fixed import (
    project,
    get_effective_embedding_2,
)
from transformer_lens.rs.callum.orthogonal_query_investigation import (
    decompose_attn_scores_full,
    create_fucking_massive_plot_1,
    create_fucking_massive_plot_2,
    token_to_qperp_projection,
    FakeIOIDataset,
)
clear_output()
USE_IOI = False
LAYER_IDX, HEAD_IDX=10,7

#%% [markdown] [2]:

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
# model.set_use_split_qkv_normalized_input(True)
clear_output()

#%%

s = "Then Bob had a ring that was perfect for Alice and he knew no would like it more than Alice"
logits = model(s)
top_logits = torch.topk(logits[0, -2, :], k=10).indices
print(model.to_str_tokens(top_logits))

#%%

filtered_examples = Path("../arthur/json_data/filtered_for_high_attention.json") # where these come from is documented in git history... it's somewhat selective but BASICALLY the easiest ~40% of the Top 5% crap
with open(filtered_examples, "r") as f:
    filtered_examples = json.load(f)

#%%

TRIMMED_SIZE = 20 # for real
filtered_examples = {
    k: v for i, (k, v) in enumerate(filtered_examples.items()) if i < TRIMMED_SIZE
} 

#%%

filtered_dataset = IOIDataset(
    N = 30,
    prompt_type="mixed",
)
    
#%%

filtered_dataset = FakeIOIDataset(
    sentences = list(filtered_examples.values()),
    io_tokens=list(filtered_examples.keys()),
    key_increment=0,
    model=model,
)

# %%

results, ioi_cache = decompose_attn_scores_full(
    ioi_dataset=filtered_dataset,
    batch_size=filtered_dataset.N,
    seed=0,
    nnmh= (10, 7),
    model= model,
    use_effective_embedding = False,
    use_layer0_heads = False,
    subtract_S1_attn_scores = True,
    include_S1_in_unembed_projection = False,
    project_onto_comms_space = "W_EE0A",
    return_cache=True,
)
ioi_cache = ioi_cache.to("cpu")
gc.collect()
torch.cuda.empty_cache()

# %%

# Breaking up these attention scores into comparison to attention scores to surrounding token, I think
# 1. Try the manual computation of attention scores from blocks.10.hook_resid_pre

default_attention_scores = dot_with_query(
    unnormalized_keys = ioi_cache[get_act_name("resid_pre", 10)][0].to("cuda"),
    unnormalized_queries = ioi_cache[get_act_name("resid_pre", 10)][0].to("cuda"),
    model = model,
    layer_idx = 10,
    head_idx = 7,
)

# oh lol that's some diagonal crap

# %%

att_scores = ioi_cache[get_act_name("attn_scores", 10)][0, 7]

# %%

diag = att_scores[torch.arange(att_scores.shape[0]), torch.arange(att_scores.shape[0])]

# %%

assert torch.allclose(diag, default_attention_scores, atol=1e-2, rtol=1e-2)
# TODO 1e-2 is super high, what's up???

# %%

# A) do this for exactly one example (no baseline)
# B) do this with a baseline subtracted
# C) do this for whole batch

#%%

all_residual_stream = {}
for hook_name in ["hook_embed", "hook_pos_embed"] + [f"blocks.{layer_idx}.hook_mlp_out" for layer_idx in range(LAYER_IDX)] + [f"blocks.{layer_idx}.attn.hook_result" for layer_idx in range(LAYER_IDX)]:
    if "attn" in hook_name:
        for head_idx in range(model.cfg.n_heads):
            all_residual_stream[f"{hook_name}_{head_idx}"] = ioi_cache[hook_name][torch.arange(filtered_dataset.N), filtered_dataset.word_idx["end"], head_idx, :]
    else:
        all_residual_stream[hook_name] = ioi_cache[hook_name][torch.arange(filtered_dataset.N), filtered_dataset.word_idx["end"], :]

#%%

comps = torch.zeros((2, filtered_dataset.N, len(all_residual_stream)))

for batch_idx in range(len(range(filtered_dataset.N))):
    results = {}
    for mode in [-2, -1, 1, 2, "parallel", "perp"]:
        gc.collect()
        t.cuda.empty_cache()
 
        if isinstance(mode, str):
            unnormalized_queries = [project(tens[batch_idx].cuda(), model.W_U[:, filtered_dataset.io_tokenIDs[batch_idx]])[int(mode=="parallel")] for tens in list(all_residual_stream.values())]
            inc = 0
        else:
            unnormalized_queries = [tens[batch_idx].cuda() for tens in list(all_residual_stream.values())]
            inc = mode

        attention_scores = dot_with_query(
            unnormalized_keys=einops.repeat(ioi_cache[get_act_name("resid_pre", 10)][batch_idx, filtered_dataset.word_idx["IO"][batch_idx] + inc].to("cuda"), "d_model -> components d_model", components=len(all_residual_stream)).clone(),
            normalize_keys=True,
            add_key_bias=True,
            unnormalized_queries=unnormalized_queries,
            normalize_queries=False,
            add_query_bias=False,
            model=model,
            layer_idx=10,
            head_idx=7,
            use_tqdm=False,
        )
        results[mode] = attention_scores.cpu()

    mean_others_numerator = sum([value for key, value in results.items() if isinstance(key, int)])
    mean_others_denominator = len([key for key in results.keys() if isinstance(key, int)])
    mean_others = mean_others_numerator / mean_others_denominator
    warnings.warn("remove")
    mean_others *=0

    for mode in ["parallel", "perp"]:
        comps[int(mode=="parallel"), batch_idx, :] = results[mode].cpu() - (mean_others/2)

    if batch_idx % 10 == 0:
        fig = go.Figure()
        for mode in ["parallel", "perp"]:    
            fig.add_trace(
                go.Bar(
                    x = list(all_residual_stream.keys()),
                    y = results[mode].cpu() - (mean_others/2),
                    name = mode,
                )
            )
        try:
            fig.update_layout(
                title = str(batch_idx) + "..." + list(filtered_examples.values())[batch_idx][:40]
            )
        except:
            pass
        fig.show()

#%%

relevant_indices = [i for i, key in enumerate(all_residual_stream) if key.startswith(tuple([f"blocks.{i}" for i in range(7, 12)]))]

fig = go.Figure()
for mode in ["parallel", "perp"]:    
    fig.add_trace(
        go.Bar(
            x = [key for key in list(all_residual_stream.keys()) if key.startswith(tuple([f"blocks.{i}" for i in range(7, 12)]))],
            y = comps[int(mode=="parallel"), :, relevant_indices].mean(dim=0).cpu(),
            name = mode,
        )
    )
try:
    fig.update_layout(
        title= "Baseline subtracted attention score contribution for each model component, on some top 5% examples"
    )
except:
    pass
fig.show()

# %%

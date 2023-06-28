# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project, get_effective_embedding_2
from transformer_lens.rs.arthurs_notebooks.arthur_utils import *
import argparse

#%%

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)

# %%

BATCH_SIZE=30
batched_tokens, targets = get_filtered_webtext(model, batch_size=BATCH_SIZE, seed=1729, device="cuda", max_seq_len=1024)
effective_embeddings = get_effective_embedding_2(model)

# %%

# Find the top 5% of things by importance
# Do this crap
# See change in loss

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
# NEGATIVE_HEAD_IDX, NEGATIVE_LAYER_IDX = 9, 9

# for NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX in itertools.product(range(11, -1, -1), range(12)):
    # NEG_HEADS[model.cfg.model_name]:

END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
names_filter1 = (
    lambda name: name == END_STATE_HOOK
    or name==f"blocks.{model.cfg.n_layers}.hook_resid_post"
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.hook_resid_pre"
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_result"
)
model = model.to("cuda:1")
logits, cache = model.run_with_cache(
    batched_tokens.to("cuda:1"),
    names_filter=names_filter1,
    device="cpu",
)
model = model.to("cuda:0")
cache.to("cuda:0")
print("Done")
cpu_logits = logits.cpu()
del logits
gc.collect()
torch.cuda.empty_cache()

# %%

batched_tokens_loss = get_loss_from_end_state(
    model=model,
    end_state=cache[get_act_name("resid_post", model.cfg.n_layers-1)],
    targets=targets,
)

#%%

head_output = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]
assert head_output.shape == (BATCH_SIZE, model.cfg.n_ctx, model.cfg.d_model)

#%%

mean_head_output = einops.reduce(head_output, "b s d -> d", reduction="mean")

#%%

mean_ablated_end_states = cache[get_act_name("resid_post", model.cfg.n_layers-1)] - head_output + einops.repeat(mean_head_output, "d -> b s d", b=BATCH_SIZE, s=model.cfg.n_ctx)
mean_ablated_loss = get_loss_from_end_state(
    model=model,
    end_state=mean_ablated_end_states,
    targets=targets,
)

# %%

max_importance_examples = sorted(
    [
        (
            batch_idx,
            seq_idx,
            (mean_ablated_loss-batched_tokens_loss)[batch_idx, seq_idx].item(),
        )
        for batch_idx, seq_idx in itertools.product(
            range(BATCH_SIZE), range(model.cfg.n_ctx)
        )
    ],
    key=lambda x: x[2],
    reverse=True,
)

# %%

# Get the top 5% of things by importance
all_top_5_percent = max_importance_examples[: len(max_importance_examples)//100]

# shuffle them
np.random.seed(799)
np.random.shuffle(all_top_5_percent)
top_5_percent = all_top_5_percent[: BATCH_SIZE]

top5p_batch_indices = [x[0] for x in top_5_percent]
top5p_seq_indices = [x[1] for x in top_5_percent]

#%%

top5p_tokens = batched_tokens[top5p_batch_indices]
top5p_targets = torch.LongTensor([targets[top5p_batch_idx, top5p_seq_idx] for top5p_batch_idx, top5p_seq_idx in zip(top5p_batch_indices, top5p_seq_indices)])

#%%

top5p_losses = batched_tokens_loss[top5p_batch_indices, top5p_seq_indices]

# %%

# Do the key-side thing where we project onto W_U
# selected_unembeddings = cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][torch.tensor(top5p_batch_indices), torch.tensor(top5p_seq_indices)]

keyside_projections = t.zeros((BATCH_SIZE, model.cfg.n_ctx, model.cfg.d_model))
keyside_orthogonals = t.zeros((BATCH_SIZE, model.cfg.n_ctx, model.cfg.d_model))

for batch_idx, seq_idx in tqdm(list(itertools.product(range(BATCH_SIZE), range(model.cfg.n_ctx)))):
    keyside_vector, keyside_orthogonal = project(
        cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][batch_idx, seq_idx],
        effective_embeddings["W_E (only MLPs)"][batched_tokens[batch_idx, seq_idx]],
    )
    keyside_projections[batch_idx, seq_idx] = keyside_vector
    keyside_orthogonals[batch_idx, seq_idx] = keyside_orthogonal

queryside_vectors = t.zeros((BATCH_SIZE, model.cfg.d_model)).cuda()

# just do this part for the individual queries that we need
for batch_batch_idx, (batch_idx, seq_idx) in enumerate(list(zip(top5p_batch_indices, 
top5p_seq_indices))):
    queryside_vector, queryside_orthogonal = project(
        cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][batch_idx, seq_idx],
        dir=[model.W_U.T[top5p_tokens[batch_idx, earlier_seq_idx]] for earlier_seq_idx in range(seq_idx+1)],
    )
    queryside_vectors[batch_batch_idx] = queryside_vector

#%%

new_k_input = t.zeros((BATCH_SIZE, model.cfg.n_ctx, model.cfg.d_model))

for batch_batch_idx, batch_idx in enumerate(top5p_batch_indices):
    new_k_input[batch_batch_idx] = torch.stack([
        keyside_projections[batch_idx, seq_idx] for seq_idx in range(model.cfg.n_ctx)
    ])

#%%

model.set_use_split_qkv_input(True)

model.reset_hooks()
model.add_hook(
    get_act_name("k_input", NEGATIVE_LAYER_IDX),
    partial(set_to_value, head_idx=NEGATIVE_HEAD_IDX, new_value=new_k_input.to("cuda:1")),
    level=1,
)
model.add_hook(
    get_act_name("q_input", NEGATIVE_LAYER_IDX),
    partial(set_to_value, head_idx=NEGATIVE_HEAD_IDX, seq_indices = top5p_seq_indices, new_value=queryside_vectors.to("cuda:1")),
    level=1,
)
model.to("cuda:1")
logits, cache = model.run_with_cache(
    top5p_tokens.to("cuda:1"),
    names_filter = lambda name: name in [get_act_name("result", NEGATIVE_LAYER_IDX), get_act_name("resid_post", model.cfg.n_layers-1)],
    device="cuda:0"
)
model.reset_hooks()
model.to("cuda:0")
new_head_out = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][torch.arange(len(top5p_tokens)), top5p_seq_indices, NEGATIVE_HEAD_IDX].unsqueeze(0)

end_state = cache[get_act_name("resid_post", model.cfg.n_layers-1)][torch.arange(BATCH_SIZE), top5p_seq_indices][None]

relevant_head_outs = []
for batch_idx, seq_idx in zip(top5p_batch_indices, top5p_seq_indices):
    relevant_head_outs.append(
        head_output[batch_idx, seq_idx]
    )
relevant_head_outs = t.stack(relevant_head_outs).unsqueeze(0)

assert end_state.shape == relevant_head_outs.shape == new_head_out.shape, (end_state.shape, relevant_head_outs.shape, new_head_out.shape)

end_state -= relevant_head_outs
end_state += new_head_out

loss = get_loss_from_end_state(
    end_state=end_state,
    model=model,
    logits=None,
    targets=top5p_targets.unsqueeze(0),
)

#%%

px.scatter(
    x=top5p_losses.cpu().tolist(),
    y=loss.cpu().tolist(),
    labels={
        "x": "Original Loss",
        "y": "New Loss",
    },
    # title=f"{NEGATIVE_LAYER_IDX}.{NEGATIVE_HEAD_IDX} Losses for sample of Top 5% Direct Effect positions w/ both projections",
    # text = [f"Batch {batch_idx}, Seq {seq_idx}" for batch_idx, seq_idx in zip(top5p_batch_indices, top5p_seq_indices)],
)

# %%

error = (loss.cpu() - top5p_losses.cpu()).abs().mean().item()
top5pdata = top5p_losses.cpu().tolist()
lossdata = loss.cpu().tolist()

# read the existing data
with open("../arthur/json_data/approximations_with_key_projection_and_query.json", "r") as f:
    cur_json = json.load(f)
# cur_json = {}

# update the data
cur_json[str((NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX))] = {
    "layer_idx": NEGATIVE_LAYER_IDX,
    "head_idx": NEGATIVE_HEAD_IDX,
    "top5p_losses": top5pdata,
    "losses": lossdata,
    "error": error,
    "mean_ablated_loss": [mean_ablated_loss[batch_idx, seq_idx].item() for batch_idx, seq_idx in zip(top5p_batch_indices, top5p_seq_indices)],
    "time": ctime()+"_remember_sometimes_this_is_an_hour_too_early",
}    

# write the data
# write the updated data (don't overwrite!)
# with open("../arthur/json_data/approximations_with_key_projection_and_query.json", "w") as f:
    # f.write(json.dumps(cur_json, indent=4))

# %%

# now also try the q projection
# %%

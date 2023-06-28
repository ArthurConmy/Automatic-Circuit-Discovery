# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
from transformer_lens.rs.arthurs_notebooks.arthur_utils import *
import argparse

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

# %%

# Find the top 5% of things by importance
# Do this crap
# See change in loss

#%%

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
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
    model=model.cpu(),
    end_state=cache[get_act_name("resid_post", model.cfg.n_layers-1)].cpu(),
    targets=targets.cpu(),
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
all_top_5_percent = max_importance_examples[: len(max_importance_examples)//20]

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
        model.W_U.T[batched_tokens[batch_idx, seq_idx]],
    )
    keyside_projections[batch_idx, seq_idx] = keyside_vector
    keyside_orthogonals[batch_idx, seq_idx] = keyside_orthogonal

#%%

new_k_input = t.zeros((len(keyside_projections), model.cfg.n_ctx, model.cfg.d_model))

for batch_idx in top5p_batch_indices:
    new_k_input[batch_idx] = torch.stack([
        keyside_projections[batch_idx, seq_idx] for seq_idx in range(model.cfg.n_ctx)
    ])

#%%

model.set_use_split_qkv_input(True)

logits = model.run_with_hooks(
    top5p_tokens,
    fwd_hooks=[
        (
            get_act_name("k_input", NEGATIVE_LAYER_IDX),
            partial(set_to_unembedding, head_idx=NEGATIVE_HEAD_IDX, new_value=new_k_input),
        ),
    ]
)[torch.arange(len(top5p_tokens)), top5p_seq_indices]

loss = get_loss_from_end_state(
    end_state=None,
    model=model,
    logits=logits,
    targets=top5p_targets,
)

#%%

px.scatter(
    x=top5p_losses.cpu(),
    y=loss.cpu(),
    labels={
        "x": "Original Loss",
        "y": "New Loss",
    },
    title="Losses for sample of Top 5% Direct Effect positions w/ key projection",
    # text = [f"Batch {batch_idx}, Seq {seq_idx}" for batch_idx, seq_idx in zip(top5p_batch_indices, top5p_seq_indices)],
)

# %%

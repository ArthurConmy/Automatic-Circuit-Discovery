# %% [markdown] [4]:

"""
Cribbed from key_and_query_projection.py
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

BATCH_SIZE = 30
batched_tokens, targets = get_filtered_webtext(model, batch_size=BATCH_SIZE, seed=1729, device="cuda", max_seq_len=1024)
effective_embeddings = get_effective_embedding_2(model)

# %%

# Find the top 5% of things by importance
# Do this crap
# See change in loss

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
# NEGATIVE_HEAD_IDX, NEGATIVE_LAYER_IDX = 9, 9
# for NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX in [(10, 0), (10, 7), (9, 9), (11, 10)] + list(itertools.product(range(11, -1, -1), range(12))):

END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
names_filter1 = (
    lambda name: name == END_STATE_HOOK
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

original_end_state = cache[get_act_name("resid_post", model.cfg.n_layers-1)]

batched_tokens_loss = get_loss_from_end_state(
    model=model,
    end_state=original_end_state,
    targets=targets,
)

#%%

head_output = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]
assert head_output.shape == (BATCH_SIZE, model.cfg.n_ctx, model.cfg.d_model)

#%%

if ipython is not None:
    unembed = einops.einsum(
        head_output, 
        model.W_U,
        "b s d_model, d_model d_vocab -> b s d_vocab",
    )

#%% 

if ipython is not None:
    the_topk = torch.topk(
        -unembed,
        k=10,
        dim=-1,
    ).indices


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

top5p_batch_indices = [x[0] for x in all_top_5_percent]
top5p_seq_indices = [x[1] for x in all_top_5_percent]

#%%

my_dict = {}

if ipython is not None:
    for idx in range(30):
        # print("-"*50)
        # print(f"Batch {top5p_batch_indices[idx]}, Seq {top5p_seq_indices[idx]}")

        current_tokens = batched_tokens[top5p_batch_indices[idx], :top5p_seq_indices[idx]+1].tolist()
        # print("PROMPT:", model.to_string(batched_tokens[top5p_batch_indices[idx], :top5p_seq_indices[idx]+2]))

        top_negs = top5p_topks[idx].tolist()

        is_in_top_negs = {
            i: int(top_negs[i] in current_tokens) for i in range(3)
        }
        if sum(list(is_in_top_negs.values()))==1:
            the_tokens = current_tokens + [top_negs[i] for i in range(3) if is_in_top_negs[i]]
            assert len(the_tokens) == len(current_tokens) + 1
            print(model.to_string(the_tokens))
            my_dict[model.to_string(the_tokens[-1:])] = model.to_string(the_tokens[1:])
        else:
            print("FAIL")
        print("-"*50)
        # top_negs = top_negs[1:]
        # print("Top negs", model.to_string(top5p_topks[idx].tolist()))
        # print("More", print("PROMPT:", model.to_string(batched_tokens[top5p_batch_indices[idx], top5p_seq_indices[idx]:top5p_seq_indices[idx]+7])))

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
        effective_embeddings["W_E (including MLPs)"][batched_tokens[batch_idx, seq_idx]],
    )
    keyside_projections[batch_idx, seq_idx] = keyside_vector
    keyside_orthogonals[batch_idx, seq_idx] = keyside_orthogonal

queryside_vectors = t.zeros((BATCH_SIZE, model.cfg.d_model)).cuda()

# just do this part for the individual queries that we need
for batch_batch_idx, (batch_idx, seq_idx) in enumerate(list(zip(top5p_batch_indices, 
top5p_seq_indices))):

    queryside_vector, queryside_orthogonal = project(
        cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][batch_idx, seq_idx],
        dir=[model.W_U.T[batched_tokens[batch_idx, earlier_seq_idx]] for earlier_seq_idx in range(seq_idx+1)],
    )
    queryside_vectors[batch_batch_idx] = queryside_vector

    # warnings.warn("Another lock on")
    # queryside_vectors[batch_batch_idx] = model.W_U.T[top5p_tokens[batch_idx, seq_idx]]
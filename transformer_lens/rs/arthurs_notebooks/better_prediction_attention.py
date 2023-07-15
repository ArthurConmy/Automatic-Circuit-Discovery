# %% [markdown] [4]:

"""
Cribbed from key_and_query_projection.py
"""

import ast
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

MAX_SEQ_LEN = 512 # half of 1024 as
BATCH_SIZE = 30
batched_tokens, targets = get_filtered_webtext(model, batch_size=BATCH_SIZE, seed=1727, device="cuda", max_seq_len=MAX_SEQ_LEN)
effective_embeddings = get_effective_embedding_2(model)

# %%

# Find the top 5% of things by importance
# Do this crap
# See change in loss

# NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
# NEGATIVE_HEAD_IDX, NEGATIVE_LAYER_IDX = 9, 9
for NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX in [(10, 0), (11, 10)] + list(itertools.product(range(11, -1, -1), range(12))):

    END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
    names_filter1 = (
        lambda name: name == END_STATE_HOOK
        or name==f"blocks.{NEGATIVE_LAYER_IDX}.hook_resid_pre"
        or name==f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_result"
        or name==f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_attn_scores"
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

    batched_tokens_loss = get_metric_from_end_state(
        model=model,
        end_state=original_end_state,
        targets=targets,
    )

    #%%

    head_output = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]
    assert head_output.shape == (BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model)

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

    mean_ablated_end_states = cache[get_act_name("resid_post", model.cfg.n_layers-1)] - head_output + einops.repeat(mean_head_output, "d -> b s d", b=BATCH_SIZE, s=MAX_SEQ_LEN)
    mean_ablated_loss = get_metric_from_end_state(
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
                range(BATCH_SIZE), range(MAX_SEQ_LEN)
            )
        ],
        key=lambda x: x[2],
        reverse=True,
    )

    # %%

    # Get the top 5% of things by importance

    TOP5P_BATCH_SIZE = len(max_importance_examples) // 20
    all_top_5_percent = max_importance_examples[:TOP5P_BATCH_SIZE]

    np.random.seed(1827)
    np.random.shuffle(all_top_5_percent)

    top5p_batch_indices = [x[0] for x in all_top_5_percent]
    top5p_seq_indices = [x[1] for x in all_top_5_percent]

    #%%

    top5p_tokens = batched_tokens[top5p_batch_indices]
    top5p_targets = torch.LongTensor([targets[top5p_batch_idx, top5p_seq_idx] for top5p_batch_idx, top5p_seq_idx in zip(top5p_batch_indices, top5p_seq_indices)])

    #%%

    top5p_losses = batched_tokens_loss[top5p_batch_indices, top5p_seq_indices]

    # %%

    # Do the key-side thing where we project onto W_U

    keyside_projections = t.zeros((BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model))
    keyside_orthogonals = t.zeros((BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model))

    for batch_idx, seq_idx in tqdm(list(itertools.product(range(BATCH_SIZE), range(MAX_SEQ_LEN)))):
        keyside_vector, keyside_orthogonal = project(
            cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][batch_idx, seq_idx],
            effective_embeddings["W_E (including MLPs)"][batched_tokens[batch_idx, seq_idx]],
        )
        keyside_projections[batch_idx, seq_idx] = keyside_vector
        keyside_orthogonals[batch_idx, seq_idx] = keyside_orthogonal

    #%% 

    queryside_vectors = t.ones((TOP5P_BATCH_SIZE, model.cfg.d_model)).cuda() * (-420)
    all_queryside_components = [] # t.ones((TOP5P_BATCH_SIZE, MAX_SEQ_LEN+1)) * (-420)
    NORMY=False
    for batch_batch_idx, (top5p_batch_idx, top5p_seq_idx) in tqdm(list(enumerate(list(zip(top5p_batch_indices, top5p_seq_indices))))):
        t.cuda.empty_cache()

        my_direction_indices = list([batched_tokens[top5p_batch_idx, earlier_seq_idx].item() for earlier_seq_idx in range(top5p_seq_idx+1)])
        assert len(my_direction_indices) == top5p_seq_idx+1
        my_directions = torch.stack([model.W_U.T[my_direction_idx] for my_direction_idx in my_direction_indices], dim=0)

        all_queryside_components.append(einops.einsum(
            cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][top5p_batch_idx, top5p_seq_idx],
            my_directions,
            "d_model, s d_model -> s",
        ))

        # queryside_vector, queryside_orthogonal, queryside_component = project(
        #     cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][top5p_batch_idx, top5p_seq_idx],
        #     dir=my_directions,
        #     return_component=True,
        # )
        # queryside_vectors[batch_batch_idx] = queryside_vector
        # assert len(queryside_component) == len(my_direction_indices) # number of distinct tokens

        # if NORMY:
        #     queryside_norms = [model.W_U.T[batched_tokens[top5p_batch_idx, earlier_seq_idx]].norm(dim=0).item() for earlier_seq_idx in range(top5p_seq_idx+1)]
        #     queryside_norms = torch.tensor(queryside_norms)
        #     assert queryside_component.shape == queryside_norms.shape
        #     queryside_components.append(queryside_component * queryside_norms.cuda())
        # else:
        #     queryside_components.append([queryside_component[my_directions_lookup[idx]] for idx in range(top5p_seq_idx+1)])

        # # warnings.warn("Another lock on")
        # # queryside_vectors[batch_batch_idx] = model.W_U.T[top5p_tokens[batch_idx, seq_idx]]

    #%%

    # all_queryside_norms = []
    # for batch_batch_idx, (top5p_batch_idx, top5p_seq_idx) in tqdm(list(enumerate(list(zip(top5p_batch_indices, top5p_seq_indices))))):
    #     queryside_norms = [model.W_U.T[batched_tokens[top5p_batch_idx, earlier_seq_idx]].norm(dim=0).item() for earlier_seq_idx in range(top5p_seq_idx+1)]
    #     queryside_components[batch_batch_idx] = torch.tensor(queryside_components[batch_batch_idx]) 
    #     if NORMY:
    #         queryside_components[batch_batch_idx] /= torch.tensor(queryside_norms)

    #     # queryside_norms = torch.tensor(queryside_norms)
    #     # assert queryside_component.shape == queryside_norms.shape
    #     # queryside_components.append(queryside_component * queryside_norms.cuda())

    #%%

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly 
    colors = colors * 1000

    def topn(x, y, n=1):
        """Returns whether the top x is in the top n y points"""
        y = sorted(list(enumerate(y)), reverse=True, key=lambda x: x[1])
        x = sorted(list(enumerate(x)), reverse=True, key=lambda x: x[1])
        yinds = [yind for yind, _ in y]
        return x[0][0] in yinds[:n]

    cnt = 0
    for i in range(100):
        x = (cache[get_act_name("attn_scores", NEGATIVE_LAYER_IDX)][top5p_batch_indices[i], NEGATIVE_HEAD_IDX, top5p_seq_indices[i], :top5p_seq_indices[i]+1]).cpu()
        y = torch.tensor(all_queryside_components[i]).cpu()
        if topn(y, x, n=5):
            cnt+=1

        if False:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    text=[(j, model.to_str_tokens(batched_tokens[top5p_batch_indices[i], max(0,j-1):j+2])) for j in range(top5p_seq_indices[i]+1)],
                    marker=dict(
                        color=colors[i],
                    ),
                )
            )

    # # read the existing data
    MY_FNAME = "../arthur/json_data/more_corr.json"
    with open(MY_FNAME, "r") as f:
        cur_json = json.load(f)

    mean_ablation_bad =  (mean_ablated_loss[top5p_batch_indices, top5p_seq_indices] - top5p_losses).tolist()

    # update the data
    cur_json[str((NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX))] = {
        "layer_idx": NEGATIVE_LAYER_IDX,
        "head_idx": NEGATIVE_HEAD_IDX,
        "cnt": cnt,
    }

    # write the data
    # write the updated data (don't overwrite!)
    with open(MY_FNAME, "w") as f:
        f.write(json.dumps(cur_json, indent=4))

    print(cnt/100)
    if False:
        fig.show()

# %%

# # read the existing data
MY_FNAME = "../arthur/json_data/more_corr.json"
with open(MY_FNAME, "r") as f:
    cur_json = json.load(f)

# %%

percents = t.zeros(12, 12)

# %%

data=[]

for k in cur_json.keys():
    layer_idx, head_idx = ast.literal_eval(k)
    if layer_idx>6:
        data.append(((layer_idx, head_idx), cur_json[k]["cnt"] / 100))

# %%

data = sorted(data, key=lambda x: x[1], reverse=True)
# %%

px.bar(
    x=[f"{layer_idx}, {head_idx}" for (layer_idx, head_idx), _ in data],
    y=[cnt for _, cnt in data],
    title="Percentage of the important prompts where the top in-context unembedding is one of the top 5 attention scores",
    text=[f"{cnt*100:.2f}%" for _, cnt in data],
    color = ["red" if i<2 else "blue" for i, cnt in enumerate(data)],
    labels={
        "x": "Layer, Head",
        "y": "Percentage",
    },
    height=500,
    # width=1000,
).update_layout(
    # font=dict(
    #     size=20,
    # ),
    xaxis=dict(
        tickangle=45,
    ),
    # yaxis=dict(
    #     tickformat="%",
    # ),
).show()


# %%

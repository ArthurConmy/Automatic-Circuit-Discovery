# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *
import argparse

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)
DEVICE = "cuda"
SHOW_PLOT = False
DATASET_SIZE = 500
BATCH_SIZE = 30  # seems to be about the limit of what this box can handle

# %%

dataset = get_webtext(seed=1729)
max_seq_len = model.tokenizer.model_max_length

# %%

filtered_tokens = []
targets = []  # targets for prediction

print("Not rapid, but not THAT slow :-) ")
_idx = -1
while len(filtered_tokens) < DATASET_SIZE:
    _idx += 1
    cur_tokens = model.to_tokens(dataset[_idx], truncate=False).tolist()[0]
    if (
        len(cur_tokens) >= max_seq_len
    ):  # so we're not biasing towards early sequence positions...
        filtered_tokens.append(cur_tokens[:max_seq_len])
        targets.append(cur_tokens[1 : max_seq_len + 1])

mybatch = torch.LongTensor(filtered_tokens[:BATCH_SIZE])
mytargets = torch.LongTensor(targets[:BATCH_SIZE])

# %%

END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"

names_filter = lambda name: name == END_STATE_HOOK or name.endswith("hook_result")

logits, cache = model.run_with_cache(
    mybatch.to(DEVICE),
    names_filter=names_filter,
    device="cpu",
)
end_state = cache[END_STATE_HOOK].cpu()  # shape (batch_size, seq_len, hidden_size)
full_logits = logits.cpu()

del logits
gc.collect()
torch.cuda.empty_cache()

# %%

def get_loss_from_end_state(
    end_state,
    targets,
    return_logits=False,
):
    # end state has shape batch, seq_len, hidden_size
    # targets has shape batch, seq_len

    assert list(end_state.shape) == list(targets.shape) + [
        model.cfg.d_model
    ], f"end_state.shape: {end_state.shape}, targets.shape: {targets.shape}"

    assert len(end_state.shape)==3, "We stricter now"

    post_layer_norm = model.ln_final(end_state)
    logits = model.unembed(post_layer_norm)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -log_probs[
        torch.arange(targets.shape[0]).unsqueeze(1),
        torch.arange(targets.shape[1]).unsqueeze(0),
        targets,
    ]
    if return_logits:
        return loss, logits
    return loss


# %%

my_loss = get_loss_from_end_state(end_state.to(DEVICE), mytargets).cpu()

# %%

their_loss = model(
    mybatch.to(DEVICE),
    return_type="loss",
    loss_per_token=True,
).cpu()

# %%

assert list(their_loss.shape) == [
    my_loss.shape[0],
    my_loss.shape[1] - 1,
], f"their_loss.shape: {their_loss.shape}, my_loss.shape: {my_loss.shape}"

torch.testing.assert_close(
    their_loss,
    my_loss[:, :-1],
    atol=1e-2,
    rtol=1e-2,
)  # yey

# %%

results_log = {}

# %%

for layer_idx, head_idx in itertools.product(
    range(model.cfg.n_layers-1, -1, -1), range(model.cfg.n_heads)
):
    head_output_hook = f"blocks.{layer_idx}.attn.hook_result"
    head_output = cache[head_output_hook][
        :, :, head_idx
    ].cpu()  # shape (batch_size, seq_len, hidden_size)
    mean_output = einops.reduce(
        head_output,
        "batch seq_len hidden_size -> hidden_size",
        reduction="mean",
    )
    mean_ablation_loss = get_loss_from_end_state(
        end_state=(end_state - head_output + mean_output[None, None]).to(DEVICE),
        targets=mytargets,
        return_logits=False,
    ).cpu()

    loss_changes = (mean_ablation_loss - my_loss).cpu()
    flattened_loss_changes = einops.rearrange(
        loss_changes, "batch seq_len -> (batch seq_len)"
    )

    if SHOW_PLOT:
        hist(
            [
                einops.rearrange(
                    mean_ablation_loss - my_loss, "batch seq_len -> (batch seq_len)"
                )
            ],
            nbins=500,
            title=f"Change in loss when mean ablating {layer_idx}.{head_idx}",
        )

    results_log[(layer_idx, head_idx)] = {
        "mean_change_in_loss": flattened_loss_changes.mean().item(),
        "std": flattened_loss_changes.std().item(),
        "abs_mean": flattened_loss_changes.abs().mean().item(),
        "loss_changes": loss_changes.cpu(),
    }
    print(list(results_log.items())[-1])

# %%

# The global plot is weird 11.0 with crazy importance, 10.7 variance low ... ?

px.bar(
    x=[str(x) for x in list(results_log.keys())],
    y=[x["mean_change_in_loss"] for x in results_log.values()],
    error_y=[x["std"] for x in results_log.values()],
).show()

# %%

# Even accounting all the cases where heads are actively harmful, it still seems like we don't really get negative heads...

px.bar(
    x=[str(x) for x in list(results_log.keys())],
    y=[
        (torch.nn.functional.relu(x["loss_changes"]) > 0).double().mean()
        for x in results_log.values()
    ],
    title="Proportions of tokens in OWT where mean ablating the direct effect of a head is helpful",
).show()

# %%

# In the max importance examples, which token does the head have the most effect on?

datab = {}

model.set_use_split_qkv_input(True)

for layer_idx, head_idx in itertools.product(
    range(11, 8, -1), range(model.cfg.n_heads)
):

# for layer_idx, head_idx in [(10, 7)]:
    max_importance_examples = sorted(
        [
            (
                batch_idx,
                seq_idx,
                results_log[(layer_idx, head_idx)]["loss_changes"][batch_idx, seq_idx].item(),
            )
            for batch_idx, seq_idx in itertools.product(
                range(BATCH_SIZE), range(max_seq_len)
            )
        ],
        key=lambda x: x[2],
        reverse=True,
    )

    head_output_hook = f"blocks.{layer_idx}.attn.hook_result"
    head_output = cache[head_output_hook][
        :, :, head_idx
    ].cpu()  # shape (batch_size, seq_len, hidden_size)
    mean_output = einops.reduce(
        head_output,
        "batch seq_len hidden_size -> hidden_size",
        reduction="mean",
    )
    mean_ablation_loss, mean_ablation_logits = get_loss_from_end_state(
        end_state=(end_state - head_output + mean_output[None, None]).to(DEVICE),
        targets=mytargets,
        return_logits=True,
    )
    mean_ablation_loss = mean_ablation_loss.to("cpu")
    mean_ablation_logits = mean_ablation_logits.to("cpu")

    cnt = 0
    copy_suppress_log = []

    for batch_idx, seq_idx, change_in_loss in tqdm(max_importance_examples[:300]):
        change_in_logits = full_logits[batch_idx, seq_idx] - mean_ablation_logits[batch_idx, seq_idx]
        prompt_words = list(set(list(mybatch[batch_idx,:seq_idx+1].tolist())))
        copy_suppress = change_in_logits[prompt_words].min()
        copy_suppress_log.append(copy_suppress.item())
        
    del mean_ablation_loss
    del mean_ablation_logits
    del head_output
    gc.collect()
    torch.cuda.empty_cache()

    print(layer_idx, head_idx, np.mean(copy_suppress_log), np.var(copy_suppress_log))
    # print("avg loss", datab[(layer_idx, head_idx)]["avg_loss"])
    # print("avg loss change", datab[(layer_idx, head_idx)]["avg_loss_change"])
    # print("avg error", datab[(layer_idx, head_idx)]["avg_error"])

# %%

def parse_data_to_dict(data):
    lines = data.split('\n')
    result_dict = {}

    for line_idx in range(0, len(lines), 3):
        layer_idx, head_idx, copy_suppress_mean, copy_suppress_var = lines[line_idx+2].split()
        layer_idx = int(layer_idx)
        head_idx = int(head_idx)
        copy_suppress_mean = float(copy_suppress_mean)
        copy_suppress_var = float(copy_suppress_var)

        result_dict[(layer_idx, head_idx)] = {
            "copy_suppress_mean": copy_suppress_mean,
            "copy_suppress_var": copy_suppress_var,
        }
            
    return result_dict

# generated from the copy suppression script
data = """100%
300/300 [00:00<00:00, 3506.77it/s]
11 0 -0.9016816154122352 0.5201871295325757
100%
300/300 [00:00<00:00, 3144.80it/s]
11 1 -0.4044969256718953 0.031844296624758996
100%
300/300 [00:00<00:00, 3172.04it/s]
11 2 -0.7388574319084485 0.1072463993261014
100%
300/300 [00:00<00:00, 3321.18it/s]
11 3 -0.43112966855367024 0.03152586384032467
100%
300/300 [00:00<00:00, 3073.56it/s]
11 4 -0.47693766315778097 0.07003433177171575
100%
300/300 [00:00<00:00, 3185.79it/s]
11 5 -0.4015259406963984 0.03355677279733544
100%
300/300 [00:00<00:00, 3172.29it/s]
11 6 -0.36947757452726365 0.026246634875785695
100%
300/300 [00:00<00:00, 3200.98it/s]
11 7 -0.33805570036172866 0.017020611048886594
100%
300/300 [00:00<00:00, 3365.80it/s]
11 8 -0.25031042834122974 1.6237580302933803
100%
300/300 [00:00<00:00, 2609.99it/s]
11 9 -0.4280617571870486 0.03441768881449543
100%
300/300 [00:00<00:00, 2729.19it/s]
11 10 -1.1136790512005488 0.29488302626701873
100%
300/300 [00:00<00:00, 2657.37it/s]
11 11 -0.9186084069311619 0.670046600938536
100%
300/300 [00:00<00:00, 2758.59it/s]
10 0 -0.5324919090668361 0.05695003976559484
100%
300/300 [00:00<00:00, 2501.65it/s]
10 1 -0.4772719904780388 0.04517875384375263
100%
300/300 [00:00<00:00, 2477.50it/s]
10 2 -0.511786490380764 0.03389539861635245
100%
300/300 [00:00<00:00, 2724.08it/s]
10 3 -0.38995554715394976 0.04442058760369615
100%
300/300 [00:00<00:00, 2653.49it/s]
10 4 -0.340000065912803 0.019087140112775115
100%
300/300 [00:00<00:00, 2712.52it/s]
10 5 -0.6252672380208969 0.0691287308996828
100%
300/300 [00:00<00:00, 2639.37it/s]
10 6 -0.4483438401420911 0.02757532322252427
100%
300/300 [00:00<00:00, 2657.01it/s]
10 7 -1.7386940280596415 0.29417134614089735
100%
300/300 [00:00<00:00, 2450.21it/s]
10 8 -0.27631867786248526 0.018791682778451068
100%
300/300 [00:00<00:00, 2543.87it/s]
10 9 -0.351839574277401 0.0258264367121916
100%
300/300 [00:00<00:00, 2704.63it/s]
10 10 -0.5870076884826024 0.07468998995874163
100%
300/300 [00:00<00:00, 2599.13it/s]
10 11 -0.4399674787123998 0.0316214673151143
100%
300/300 [00:00<00:00, 3375.63it/s]
9 0 -0.3342344471812248 0.02692906721012261
100%
300/300 [00:00<00:00, 2830.80it/s]
9 1 -0.369691844334205 0.043611652938372956
100%
300/300 [00:00<00:00, 3367.39it/s]
9 2 -0.3142923931777477 0.013720869772415286
100%
300/300 [00:00<00:00, 3420.14it/s]
9 3 -0.38847506006558735 0.02535700541648442
100%
300/300 [00:00<00:00, 2632.87it/s]
9 4 -0.26610014503200846 0.015068695777554977
100%
300/300 [00:00<00:00, 3256.16it/s]
9 5 -0.8419594989220301 0.11326557327638707
100%
300/300 [00:00<00:00, 3173.06it/s]
9 6 -0.5419738794366519 0.05079622240332223
100%
300/300 [00:00<00:00, 3149.75it/s]
9 7 -0.32550418059031166 0.01631604746415235
100%
300/300 [00:00<00:00, 3367.45it/s]
9 8 -0.307948471903801 0.03385204378668494
100%
300/300 [00:00<00:00, 2608.21it/s]
9 9 -0.5532821393013001 0.0500570998217686
100%
300/300 [00:00<00:00, 3300.43it/s]
9 10 -0.41829231098294256 0.020102954189287616
100%
300/300 [00:00<00:00, 3079.44it/s]
9 11 -0.4879091795285543 0.02666743623329828"""

parsed_dict = parse_data_to_dict(data)
# print(parsed_dict)

#%%

fig = px.bar(
    x = [f"{layer_idx}.{head_idx}" for layer_idx, head_idx in itertools.product(range(9, 12), range(12))],
    y = [parsed_dict[(layer_idx, head_idx)]["copy_suppress_mean"] for layer_idx, head_idx in itertools.product(range(9, 12), range(12))],
)

# add error bars
fig.add_scatter(
    x = [f"{layer_idx}.{head_idx}" for layer_idx, head_idx in itertools.product(range(9, 12), range(12))],
    y = [parsed_dict[(layer_idx, head_idx)]["copy_suppress_mean"]+np.sqrt(parsed_dict[(layer_idx, head_idx)]["copy_suppress_var"]) for layer_idx, head_idx in itertools.product(range(9, 12), range(12))],
    mode="lines",
    line=dict(color="black", width=1, dash="dash"),
    name="+1 std",
)
fig.add_scatter(
    x = [f"{layer_idx}.{head_idx}" for layer_idx, head_idx in itertools.product(range(9, 12), range(12))],
    y = [parsed_dict[(layer_idx, head_idx)]["copy_suppress_mean"]-np.sqrt(parsed_dict[(layer_idx, head_idx)]["copy_suppress_var"]) for layer_idx, head_idx in itertools.product(range(9, 12), range(12))],
    mode="lines",
    line=dict(color="black", width=1, dash="dash"),
    name="-1 std",
)

# add title 
fig.update_layout(
    title="Copy suppression",
    xaxis_title="Layer.Head",
    yaxis_title="Copy suppression",
)

#%%

fig = go.Figure()

fig.add_scatter(
    x=[x["avg_loss"] for x in parsed_dict.values()],
    y=[x["avg_error"]+x["avg_loss"] for x in parsed_dict.values()],
    error_y=dict(
        type='data',
        symmetric=False,
        array=[max(0, x["avg_loss_change"]-x["avg_error"]) for x in parsed_dict.values()],
        arrayminus=[max(0, x["avg_error"]-x["avg_loss_change"]) for x in parsed_dict.values()]
    ),
    text=[str(x) for x in parsed_dict],
    mode="markers",
)

# add x=y line
fig.add_scatter(
    x=[0, 10],
    y=[0, 10],
    mode="lines",
    line=dict(color="black", width=1, dash="dash"),
)

# add labels
if False:
    for i, atxt in enumerate(parsed_dict.keys()):
        txt=str(atxt)
        fig.add_annotation(
            x=parsed_dict[atxt]["avg_loss"],
            y=parsed_dict[atxt]["avg_error"]+parsed_dict[atxt]["avg_loss"],
            text=txt + " " + str(parsed_dict[atxt]["avg_loss_change"]),
            showarrow=False,
            yshift=10,
        )


fig.update_layout(
    title="Dot is average loss w/ the approximation. Line is mean ablating head",
    xaxis_title="Average model loss for top prompts where this head is useful",
    yaxis_title="New loss",
)
# %%

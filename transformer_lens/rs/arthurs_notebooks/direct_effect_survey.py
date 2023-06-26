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
CACHE_ALL_HEAD = True

# %% [markdown]
# Probably deprecated way of running all this as a script...

parser = argparse.ArgumentParser()
parser.add_argument("--layer_idx", type=int, default=10)
parser.add_argument("--head_idx", type=int, default=7)

if ipython is None:
    args = parser.parse_args()
    LAYER_IDX = args.layer_idx
    HEAD_IDX = args.head_idx
else:  # we are in a notebook
    args = parser.parse_args([])
    LAYER_IDX, HEAD_IDX = NEG_HEADS[model.cfg.model_name]

# %%

dataset = get_webtext(seed=1729)
max_seq_len = model.tokenizer.model_max_length

# %%

filtered_tokens = []
targets = []  # targets for prediction

print("Not rapid, but not THAT slow : ) ")
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

HEAD_OUTPUT_HOOK = f"blocks.{LAYER_IDX}.attn.hook_result"
END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"

if CACHE_ALL_HEAD:
    names_filter = lambda name: name == END_STATE_HOOK or name.endswith("hook_result")
else:
    names_filter = lambda name: name in [END_STATE_HOOK, HEAD_OUTPUT_HOOK]

logits, cache = model.run_with_cache(
    mybatch.to(DEVICE),
    names_filter=names_filter,
    device="cpu",
)
head_output = cache[HEAD_OUTPUT_HOOK][
    :, :, HEAD_IDX
].cpu()  # shape (batch_size, seq_len, hidden_size)
end_state = cache[END_STATE_HOOK].cpu()  # shape (batch_size, seq_len, hidden_size)

del logits
gc.collect()
torch.cuda.empty_cache()

# %%

mean_output = einops.reduce(
    head_output,
    "batch seq_len hidden_size -> hidden_size",
    reduction="mean",
)

# %%


def get_loss_from_end_state(
    end_state,
    targets,
):
    # end state has shape batch, seq_len, hidden_size
    # targets has shape batch, seq_len

    assert list(end_state.shape) == list(targets.shape) + [
        model.cfg.d_model
    ], f"end_state.shape: {end_state.shape}, targets.shape: {targets.shape}"

    post_layer_norm = model.ln_final(end_state)
    logits = model.unembed(post_layer_norm)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -log_probs[
        torch.arange(targets.shape[0]).unsqueeze(1),
        torch.arange(targets.shape[1]).unsqueeze(0),
        targets,
    ]
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

mean_ablation_loss = get_loss_from_end_state(
    end_state=(end_state - head_output + mean_output[None, None]).to(DEVICE),
    targets=mytargets,
).cpu()

# %%

results_log = {}

# %%

for layer_idx, head_idx in itertools.product(
    range(model.cfg.n_layers), range(model.cfg.n_heads)
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
            title="Change in loss when mean ablating",
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

for LAYER_IDX, HEAD_IDX in itertools.product(
    range(11, 8, -1), range(model.cfg.n_heads)
):
# for LAYER_IDX, HEAD_IDX in [(10, 7)]:
    max_importance_examples = sorted(
        [
            (
                batch_idx,
                seq_idx,
                results_log[(LAYER_IDX, HEAD_IDX)]["loss_changes"][batch_idx, seq_idx],
            )
            for batch_idx, seq_idx in itertools.product(
                range(BATCH_SIZE), range(max_seq_len)
            )
        ],
        key=lambda x: x[2].item(),
        reverse=True,
    )

    head_output_hook = f"blocks.{LAYER_IDX}.attn.hook_result"
    head_output = cache[HEAD_OUTPUT_HOOK][
        :, :, HEAD_IDX
    ].cpu()  # shape (batch_size, seq_len, hidden_size)
    mean_output = einops.reduce(
        head_output,
        "batch seq_len hidden_size -> hidden_size",
        reduction="mean",
    )
    mean_ablation_loss = get_loss_from_end_state(
        end_state=(end_state - head_output + mean_output[None, None]).to(DEVICE),
        targets=mytargets,
    ).cpu()

    cnt = 0

    for batch_idx, seq_idx, change_in_loss in tqdm(max_importance_examples[:300]):
        change_in_state = head_output[batch_idx, seq_idx] - mean_output
        logits = einops.einsum(
            change_in_state.to(DEVICE),
            model.W_U,
            "d_model, d_model d_vocab -> d_vocab",
        ).cpu()

        botk = torch.topk(-logits, 10)

        # replace attention of 10.7 with attention only to these items...

        def repper_hook(z, hook, indices, tokens):
            old_z = z.clone()
            z[:, HEAD_IDX] = 0.0

            for i in indices:
                for j in range(tokens.shape[-1]):
                    if tokens[j].item() == i:
                        z[0, HEAD_IDX, seq_idx, j] = old_z[0, HEAD_IDX, seq_idx, j]

            return z

        endcache = {}
        def cacher(z, hook, endcache):
            endcache[0] = z.clone()
            return z

        model.reset_hooks()
        logits = model.run_with_hooks(
            mybatch[batch_idx : batch_idx + 1, : seq_idx + 1],
            fwd_hooks=[
                (
                    f"blocks.{LAYER_IDX}.attn.hook_pattern",
                    partial(
                        repper_hook,
                        indices=[0] + botk.indices.tolist(),
                        tokens=mybatch[batch_idx, : seq_idx + 1],
                    ),
                ),
                (
                    f"blocks.{model.cfg.n_layers-1}.hook_resid_post",
                    partial(cacher, endcache=endcache),
                ),
            ],
        )
        # probs = torch.nn.functional.softmax(logits, dim=-1)
        # loss = -torch.log(probs[0, -1, mytargets[batch_idx, seq_idx].item()]).item()

        loss=get_loss_from_end_state(
            endcache[0][0, seq_idx].to(DEVICE).unsqueeze(0).unsqueeze(0),
            mytargets[batch_idx, seq_idx:seq_idx+1].unsqueeze(0).to(DEVICE),
        )

        if my_loss[batch_idx, seq_idx].item()>loss.item():
            cnt+=1
            # print("yes")
            # , mean_ablation_loss[batch_idx, seq_idx])

        # else:
            # print("no")


        # if (
        #     len(
        #         set(mybatch[batch_idx, : seq_idx + 1].tolist()).intersection(
        #             set(botk.indices.tolist())
        #         )
        #     )
        #     != 0
        # ):
        #     # print("no")
        #     # print("yes")
        #     cnt += 1

    print(LAYER_IDX, HEAD_IDX, cnt/300)

# %%

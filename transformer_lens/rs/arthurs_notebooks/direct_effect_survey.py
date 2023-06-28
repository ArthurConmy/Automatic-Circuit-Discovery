# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
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
NUM_THINGS = 300
USE_RANDOM_SAMPLE=True

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

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
names_filter1 = (
    lambda name: name == END_STATE_HOOK
    or name.endswith("hook_result")
    or name.endswith(".hook_resid_pre")
)

model = model.to("cuda:1")
logits, cache = model.run_with_cache(
    mybatch.to("cuda:1"),
    names_filter=names_filter1,
    device="cpu",
)
model = model.to("cuda:0")
print("Done")
end_state = cache[END_STATE_HOOK].to("cuda")  # shape (batch_size, seq_len, hidden_size)
full_logits = logits

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

    assert len(end_state.shape) == 3, "We stricter now"

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

tl_path = Path(__file__)
assert "/TransformerLens/" in str(tl_path), "This is a hacky way to get the path"

while tl_path.stem!="TransformerLens" and str(tl_path.parent)!=str(tl_path):
    tl_path = tl_path.parent

if (tl_path / "results_log.pt").exists():
    results_log = torch.load(tl_path / "results_log.pt")

else:
    results_log={}
    for layer_idx, head_idx in itertools.product(
        range(model.cfg.n_layers - 1, -1, -1), range(model.cfg.n_heads)
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
            "mean_ablation_loss": mean_ablation_loss.cpu(),
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
    title="Proportion of token predictions in OWT where mean ablating the direct effect of a head is helpful",
).show()

# %%

def simulate_effective_embedding(
    model: HookedTransformer,
) -> Float[Tensor, "d_vocab d_model"]:
    """Cribbed from `transformer_lens/rs/callums_notebooks/subtract_embedding.ipynb`"""
    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    embeds = W_E.unsqueeze(0)
    pre_attention = model.blocks[0].ln1(embeds)
    # !!! b_O is not zero. Seems like b_V is, but we'll add it to be safe rather than sorry
    assert model.b_V[0].norm().item() < 1e-4
    assert model.b_O[0].norm().item() > 1e-4
    vout = (
        einops.einsum(  # equivalent to locking attention to 1
            pre_attention,
            model.W_V[0],
            "b s d_model, num_heads d_model d_head -> b s num_heads d_head",
        )
        + model.b_V[0]
    )
    post_attention = (
        einops.einsum(
            vout,
            model.W_O[0],
            "b s num_heads d_head, num_heads d_head d_model_out -> b s d_model_out",
        )
        + model.b_O[0]
    )
    resid_mid = post_attention + embeds
    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()
    return {
        "W_U (or W_E, no MLPs)": W_U.T,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE,
    }
embeddings_dict = simulate_effective_embedding(model)

# %%

# Test that effective embedding is the same as lock attention and zero pos embed
model.reset_hooks()
model.add_hook(
    name="hook_pos_embed",
    hook=lambda z, hook: z * 0.0,
)
model.add_hook(
    name="blocks.0.attn.hook_pattern",
    hook=lock_attn,
)
mlp_out_hook = "blocks.0.hook_mlp_out"
hook_resid_pre = "blocks.1.hook_resid_pre"
_, cache_test = model.run_with_cache(
    torch.arange(model.tokenizer.model_max_length).unsqueeze(0).to(DEVICE),
    names_filter=lambda name: name in [mlp_out_hook, hook_resid_pre],
)
torch.testing.assert_close(
    cache_test[mlp_out_hook][0],
    embeddings_dict["W_E (only MLPs)"][: model.tokenizer.model_max_length],
    atol=1e-3,
    rtol=1e-3,
)
torch.testing.assert_close(
    cache_test[hook_resid_pre][0],
    embeddings_dict["W_E (including MLPs)"][: model.tokenizer.model_max_length],
    atol=1e-3,
    rtol=1e-3,
)

# %%

gc.collect()
torch.cuda.empty_cache()

# In the max importance examples, which token does the head have the most effect on?
datab = {}
model.set_use_split_qkv_input(True)
for layer_idx, head_idx in [(10, 7)] + list(
    itertools.product(range(11, 8, -1), range(model.cfg.n_heads))
):
    print("-"*50)
    print(layer_idx, head_idx)

    # for layer_idx, head_idx in [(10, 7)]:
    max_importance_examples = sorted(
        [
            (
                batch_idx,
                seq_idx,
                results_log[(layer_idx, head_idx)]["loss_changes"][
                    batch_idx, seq_idx
                ].item(),
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
    ].to("cuda")  # shape (batch_size, seq_len, hidden_size)
    mean_output = einops.reduce(
        head_output,
        "batch seq_len hidden_size -> hidden_size",
        reduction="mean",
    )
    mean_ablation_loss = results_log[(layer_idx, head_idx)]["mean_ablation_loss"]

    mals = []
    new_losses = []
    orig_losses = []

    random_indices = np.random.choice(len(max_importance_examples), NUM_THINGS, replace=False).tolist()
    progress_bar = tqdm(random_indices) if USE_RANDOM_SAMPLE else tqdm(max_importance_examples[:NUM_THINGS])

    for current_iter_element in progress_bar:
        if USE_RANDOM_SAMPLE:
            batch_idx, seq_idx, change_in_loss = max_importance_examples[current_iter_element]
        else:
            batch_idx, seq_idx, change_in_loss = current_iter_element

        cur_output = (head_output-mean_output)[batch_idx, seq_idx]

        # unembed = einops.einsum(
        #     cur_output,
        #     model.W_U,
        #     "d_model_out, d_model_out d_vocab -> d_vocab",
        # )
        # topk = torch.topk(unembed, k=10).indices
        # print([model.to_string([tk]) for tk in topk])
        # print([model.to_string([j]) for j in mybatch[batch_idx, max(0, seq_idx-100):seq_idx+1]])
        # print(model.to_string([mybatch[batch_idx, seq_idx+1]]))

        names_filter2 = lambda name: name.endswith("hook_v") or name.endswith(
            "hook_pattern"
        )
        model.reset_hooks()
        _, cache2 = model.run_with_cache(
            mybatch[batch_idx : batch_idx + 1, : seq_idx + 1],
            names_filter=names_filter2,
        )

        vout = cache2[f"blocks.{layer_idx}.attn.hook_v"][
            0, :, head_idx
        ]  # (seq_len, d_head)
        att_pattern = cache2[f"blocks.{layer_idx}.attn.hook_pattern"][
            0, head_idx, seq_idx
        ]  # shape (seq_len)
        ovout = einops.einsum(
            vout,
            model.W_O[layer_idx, head_idx],
            "s d_head, d_head d_model_out -> s d_model_out",
        )
        gc.collect()
        torch.cuda.empty_cache()

        for ovout_idx in range(len(ovout)):
            projected, _ = project(ovout[ovout_idx], model.W_U[:, mybatch[batch_idx, seq_idx]])
            if einops.einsum(
                ovout[ovout_idx],
                projected,
                "d_model_out, d_model_out -> ",
            ).item()>0: # only include negative components
                ovout[ovout_idx] = projected

        att_out = einops.einsum(
            att_pattern,
            ovout,
            "s, s d_model_out -> d_model_out",
        )

        # add in orthogonal component
        parallel_component, orthogonal_component = project(
            att_out, 
            mean_output.to(DEVICE),
        )
        att_out += orthogonal_component

        # print([model.to_string([tk]) for tk in topk])
        # print(
        #     [
        #         model.to_string([j])
        #         for j in mybatch[batch_idx, max(0, seq_idx - 10) : seq_idx + 1]
        #     ]
        # )
        # torch.testing.assert_close(
        #     att_out.cpu(),
        #     cache[f"blocks.{layer_idx}.attn.hook_result"][batch_idx, seq_idx, head_idx],
        #     atol=1e-3,
        #     rtol=1e-3,
        # )

        new_loss = get_loss_from_end_state(
            end_state=(
                end_state[batch_idx : batch_idx + 1, seq_idx : seq_idx + 1]
                - head_output[
                    batch_idx : batch_idx + 1, seq_idx : seq_idx + 1, head_idx
                ]
                + att_out[None, None]
            ),
            targets=mytargets[batch_idx : batch_idx + 1, seq_idx : seq_idx + 1],
        ).item()
        mal = mean_ablation_loss[batch_idx, seq_idx]
        orig_loss = my_loss[batch_idx, seq_idx]

        mals.append(mal.item())
        new_losses.append(new_loss)
        orig_losses.append(orig_loss.item())

        # print(f"{mal.item():.2f} {orig_loss.item():.2f} {new_loss:.2f}")

    datab[(layer_idx, head_idx)] = {
        "mals_mean": np.mean(mals),
        "mals_std": np.std(mals),
        "new_losses_mean": np.mean(new_losses),
        "new_losses_std": np.std(new_losses),
        "orig_losses_mean": np.mean(orig_losses),
        "orig_losses_std": np.std(orig_losses),
        "mals": mals,
        "new_losses": new_losses,
        "orig_losses": orig_losses,
    }

    for k in datab[(layer_idx, head_idx)]:
        if k.endswith("mean") or k.endswith("std"):
            print(k, datab[(layer_idx, head_idx)][k], end="/")
    print()

# %%

# TODO speed this up and do Callum's proposed experiment with one component in unembed direction 

# %%


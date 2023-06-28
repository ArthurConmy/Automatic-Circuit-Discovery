# %% [markdown] [4]:

from transformer_lens.cautils.notebook import *
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
TEST=False
BATCH_SIZE = 30  # seems to be about the limit of what this box can handle
LAYER_IDX,HEAD_IDX=(9,9)

# %%

dataset = get_webtext(seed=1729)
max_seq_len = model.tokenizer.model_max_length

# %%

print(f"Generating {BATCH_SIZE=} documents that are all longer than the context length") # so we always fill context

filtered_tokens = []
targets = []  # targets for prediction

_idx = -1
while len(filtered_tokens) < BATCH_SIZE:
    _idx += 1
    cur_tokens = model.to_tokens(dataset[_idx], truncate=False).tolist()[0]
    if (
        len(cur_tokens) > max_seq_len
    ):  # so we're not biasing towards early sequence positions...
        filtered_tokens.append(cur_tokens[:max_seq_len])
        targets.append(cur_tokens[1 : max_seq_len + 1])

batch_of_prompts = torch.LongTensor(filtered_tokens)
batch_of_targets = torch.LongTensor(targets)

# %%

print("Cache some useful things")
END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
names_filter1 = (
    lambda name: name == END_STATE_HOOK
    or name.endswith("hook_result")
    or name.endswith(".hook_resid_pre")
)
logits, cache = model.run_with_cache(
    batch_of_prompts.to("cuda"),
    names_filter=names_filter1,
    device="cpu",
)
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

batch_of_losses = get_loss_from_end_state(end_state.to(DEVICE), batch_of_targets).cpu()

# %%

if TEST: # test that the end state function works
    their_loss = model(
        batch_of_prompts.to(DEVICE),
        return_type="loss",
        loss_per_token=True,
    ).cpu()

    assert list(their_loss.shape) == [
        batch_of_losses.shape[0],
        batch_of_losses.shape[1] - 1,
    ], f"their_loss.shape: {their_loss.shape}, my_loss.shape: {batch_of_losses.shape}"

    torch.testing.assert_close(
        their_loss,
        batch_of_losses[:, :-1],
        atol=1e-2,
        rtol=1e-2,
    )  # yey

# %%

gc.collect()
torch.cuda.empty_cache()
head_output_hook = f"blocks.{LAYER_IDX}.attn.hook_result"
head_output = cache[head_output_hook][
    :, :, HEAD_IDX
].cpu()  # shape (batch_size, seq_len, hidden_size)
mean_output = einops.reduce(
    head_output,
    "batch seq_len hidden_size -> hidden_size",
    reduction="mean",
).cpu()
mean_ablation_loss = get_loss_from_end_state(
    end_state=(end_state.cpu() - head_output + mean_output[None, None]).to(DEVICE),
    targets=batch_of_targets,
    return_logits=False,
)

loss_changes = mean_ablation_loss.cpu() - batch_of_losses.cpu()
flattened_loss_changes = einops.rearrange(
    loss_changes, "batch seq_len -> (batch seq_len)"
)

if SHOW_PLOT:
    hist(
        [
            einops.rearrange(
                mean_ablation_loss - batch_of_losses, "batch seq_len -> (batch seq_len)"
            )
        ],
        nbins=500,
        title=f"Change in loss when mean ablating {LAYER_IDX}.{HEAD_IDX}",
    )

results_log={
    "mean_change_in_loss": flattened_loss_changes.mean().item(),
    "std": flattened_loss_changes.std().item(),
    "abs_mean": flattened_loss_changes.abs().mean().item(),
    "loss_changes": loss_changes.cpu(),
    "mean_ablation_loss": mean_ablation_loss.cpu(),
}

#%%

max_importance_examples = sorted(
    [
        (
            batch_idx,
            seq_idx,
            results_log["loss_changes"][
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

head_output_hook = f"blocks.{LAYER_IDX}.attn.hook_result"
head_output = cache[head_output_hook][
    :, :, HEAD_IDX
].to("cuda")  # shape (batch_size, seq_len, hidden_size)
mean_output = einops.reduce(
    head_output,
    "batch seq_len hidden_size -> hidden_size",
    reduction="mean",
)
mean_ablation_loss = results_log["mean_ablation_loss"]

mals = []
new_losses = []
orig_losses = []

for batch_idx, seq_idx, change_in_loss in tqdm(max_importance_examples[:10]):
    cur_output = (head_output-mean_output)[batch_idx, seq_idx]

    unembed = einops.einsum(
        cur_output,
        model.W_U,
        "d_model_out, d_model_out d_vocab -> d_vocab",
    )
    topk = torch.topk(unembed, k=10).indices
    print("-"*50)
    print(batch_idx, seq_idx)
    print("Top completions:", [model.to_string([tk]) for tk in topk])
    print("Context:", [model.to_string([j]) for j in batch_of_prompts[batch_idx, max(0, seq_idx-100):seq_idx+1]])
    print("Correct token:", (model.to_string([batch_of_prompts[batch_idx, seq_idx+1]])))
    continue

# %%

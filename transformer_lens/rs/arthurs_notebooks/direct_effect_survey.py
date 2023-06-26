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

END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"

names_filter = lambda name: name == END_STATE_HOOK or name.endswith("hook_result")

logits, cache = model.run_with_cache(
    mybatch.to(DEVICE),
    names_filter=names_filter,
    device="cpu",
)
end_state = cache[END_STATE_HOOK].cpu()  # shape (batch_size, seq_len, hidden_size)

del logits
gc.collect()
torch.cuda.empty_cache()

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

    assert len(end_state.shape)==3, "We stricter now"

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
# for layer_idx, HEAD_IDX in [(10, 7)]:
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

    # Do "the" baseline???

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

    cnt = 0
    avg_loss = []
    avg_loss_change = []
    avg_error = []

    for batch_idx, seq_idx, change_in_loss in tqdm(max_importance_examples[:300]):
        change_in_state = head_output[batch_idx, seq_idx] - mean_output
        the_tokens = torch.zeros((seq_idx+1, 3)).long()
        the_tokens[:, -1] = mybatch[batch_idx, : seq_idx + 1]
        the_tokens[:, 1] = model.to_tokens("The", prepend_bos=False).item()
        the_tokens[:, 0] = model.tokenizer.bos_token_id
        _, another_cache = model.run_with_cache(
            the_tokens,
            names_filter=lambda name: name==f"blocks.{layer_idx}.hook_resid_pre",
        )
        key_inputs = another_cache[f"blocks.{layer_idx}.hook_resid_pre"][:, -1]
        def set_to_value(z, hook, value, head_idx):
            assert z.shape[2]==model.cfg.n_heads, z.shape
            assert z[0, :, head_idx].shape==value.shape, (z[:, :, head_idx].shape, value.shape)
            z[0, :, head_idx] = value
        endcache={}
        def cacher(z, hook, endcache):
            endcache[0] = z.cpu()
            return z

        model.reset_hooks()
        logits = model.run_with_hooks(
            mybatch[batch_idx : batch_idx + 1, : seq_idx + 1],
            fwd_hooks=[
                (
                    f"blocks.{layer_idx}.hook_k_input",
                    partial(
                        set_to_value,
                        value=key_inputs,
                        head_idx=head_idx,
                    ),
                ),
                (
                    f"blocks.{layer_idx}.attn.hook_result",
                    partial(cacher, endcache=endcache),
                ),
            ],
        )
        # probs = torch.nn.functional.softmax(logits, dim=-1)
        # loss = -torch.log(probs[0, -1, mytargets[batch_idx, seq_idx].item()]).item()

        end_result = endcache[0][0, seq_idx, head_idx]
        assert list(end_result.shape) == [model.cfg.d_model], endcache[0][0, seq_idx, head_idx].shape

        loss=get_loss_from_end_state(
            (end_state[batch_idx, seq_idx] - head_output[batch_idx, seq_idx] + end_result)[None, None].to(DEVICE),
            mytargets[batch_idx:batch_idx+1, seq_idx:seq_idx+1].to(DEVICE),
        )
        avg_loss.append(my_loss[batch_idx, seq_idx].item())
        avg_loss_change.append((mean_ablation_loss[batch_idx, seq_idx]-my_loss[batch_idx, seq_idx]).item())

        assert abs(avg_loss_change[-1]-change_in_loss)<1e-5, (avg_loss_change[-1], change_in_loss)
        # TODO fix, consistently failing assertion???


        avg_error.append(loss.item()-my_loss[batch_idx, seq_idx].item())

        if my_loss[batch_idx, seq_idx].item()>loss.item():
            cnt+=1

    datab[(layer_idx, head_idx)] = {
        "avg_loss": np.mean(avg_loss),
        "avg_loss_change": np.mean(avg_loss_change),
        "avg_error": np.mean(avg_error),
    }

    print(layer_idx, head_idx, cnt/300)
    print("avg loss", datab[(layer_idx, head_idx)]["avg_loss"])
    print("avg loss change", datab[(layer_idx, head_idx)]["avg_loss_change"])
    print("avg error", datab[(layer_idx, head_idx)]["avg_error"])

# %%

def parse_data_to_dict(data):
    lines = data.split('\n')
    result_dict = {}

    for line_idx in range(0, len(lines), 6):
        layer_idx, head_idx, _ = lines[line_idx].split()

        # Check if line contains key-value pair
        assert lines[line_idx + 1].startswith("avg loss")
        assert lines[line_idx + 2].startswith("avg loss change")
        assert lines[line_idx + 3].startswith("avg error")

        result_dict[(layer_idx, head_idx)] = {
            "avg_loss": float(lines[line_idx + 1].split()[-1]),
            "avg_loss_change": float(lines[line_idx + 2].split()[-1]),
            "avg_error": float(lines[line_idx + 3].split()[-1]),
        }
            
    return result_dict

data = """11 0 0.013333333333333334
avg loss 2.3835536268632858
avg loss change 0.7386037816603979
avg error 0.6539135828831544
100%
300/300 [00:34<00:00, 8.89it/s]
11 1 0.13
avg loss 3.864693093250195
avg loss change 0.7659014473358791
avg error 0.3382451789081097
100%
300/300 [00:34<00:00, 8.54it/s]
11 2 0.24333333333333335
avg loss 3.4978087133169176
avg loss change 2.652760692834854
avg error 0.5238392366965612
100%
300/300 [00:32<00:00, 10.32it/s]
11 3 0.49666666666666665
avg loss 4.21456408187747
avg loss change 1.3247506026426952
avg error 0.03020534579952558
100%
300/300 [00:35<00:00, 9.41it/s]
11 4 0.06333333333333334
avg loss 4.63112070629994
avg loss change 0.5176826830705007
avg error 0.39782712231079737
100%
300/300 [00:34<00:00, 8.34it/s]
11 5 0.03
avg loss 3.3653839365641276
avg loss change 0.5493491295973459
avg error 0.42650381724039715
100%
300/300 [00:34<00:00, 9.08it/s]
11 6 0.04666666666666667
avg loss 4.21505222722888
avg loss change 0.6185185609261195
avg error 0.5150644576052824
100%
300/300 [00:34<00:00, 8.23it/s]
11 7 0.07666666666666666
avg loss 2.794633297820886
avg loss change 0.6033858813842138
avg error 0.3840236317118009
100%
300/300 [00:34<00:00, 9.04it/s]
11 8 0.07666666666666666
avg loss 3.741642370223999
avg loss change 0.88694431245327
avg error 0.458658616344134
100%
300/300 [00:34<00:00, 9.44it/s]
11 9 0.09666666666666666
avg loss 3.741892358313004
avg loss change 0.6671927403410276
avg error 0.382763326416413
100%
300/300 [00:33<00:00, 8.20it/s]
11 10 0.29333333333333333
avg loss 4.992569005936384
avg loss change 0.5695823103189468
avg error 0.12431741530696551
100%
300/300 [00:33<00:00, 7.81it/s]
11 11 0.14333333333333334
avg loss 6.023248256916801
avg loss change 0.5564728027582169
avg error 0.33571807148555916
100%
300/300 [00:32<00:00, 9.51it/s]
10 0 0.37333333333333335
avg loss 2.534186307216684
avg loss change 1.9031446488698323
avg error 0.3274131795515617
100%
300/300 [00:35<00:00, 9.70it/s]
10 1 0.016666666666666666
avg loss 3.0247301920006673
avg loss change 0.8361247881253561
avg error 0.6841816642135382
100%
300/300 [00:35<00:00, 8.10it/s]
10 2 0.05333333333333334
avg loss 2.9887292234102887
avg loss change 1.7848887467384338
avg error 0.9455112421512604
100%
300/300 [00:33<00:00, 7.94it/s]
10 3 0.07666666666666666
avg loss 3.07864759683609
avg loss change 1.0100470993916193
avg error 0.5578694013754527
100%
300/300 [00:33<00:00, 7.39it/s]
10 4 0.056666666666666664
avg loss 2.873488065674901
avg loss change 0.6942227291067441
avg error 0.49977676404019195
100%
300/300 [00:33<00:00, 9.51it/s]
10 5 0.0033333333333333335
avg loss 2.6171088931709527
avg loss change 1.0065552939971287
avg error 0.9571821940193573
100%
300/300 [00:34<00:00, 7.35it/s]
10 6 0.09666666666666666
avg loss 3.3077179816613596
avg loss change 1.1842566108703614
avg error 0.4771018896748622
100%
300/300 [00:33<00:00, 10.40it/s]
10 7 0.21
avg loss 5.080783624947071
avg loss change 1.0295758157968522
avg error 0.2629986247420311
100%
300/300 [00:36<00:00, 9.57it/s]
10 8 0.0033333333333333335
avg loss 2.12368817307055
avg loss change 0.5812099544207255
avg error 0.5338043490300576
100%
300/300 [00:35<00:00, 7.95it/s]
10 9 0.023333333333333334
avg loss 1.9291445140664776
avg loss change 0.9361788300673167
avg error 0.7159277528896928
100%
300/300 [00:35<00:00, 7.86it/s]
10 10 0.04666666666666667
avg loss 2.7494985501157743
avg loss change 1.4382764037450155
avg error 1.0388174733333289
100%
300/300 [00:33<00:00, 7.48it/s]
10 11 0.04
avg loss 3.2883034194012484
avg loss change 0.7354107892513275
avg error 0.4603416486084461
100%
300/300 [00:32<00:00, 8.26it/s]
9 0 0.17333333333333334
avg loss 3.4325320261220136
avg loss change 0.5358555382490158
avg error 0.1949662038187186
100%
300/300 [00:34<00:00, 7.41it/s]
9 1 0.02
avg loss 2.4590366874386866
avg loss change 0.433472909728686
avg error 0.35887142397463323
100%
300/300 [00:32<00:00, 10.53it/s]
9 2 0.3233333333333333
avg loss 3.657995838522911
avg loss change 0.7993131331602732
avg error 0.1421295601129532
100%
300/300 [00:32<00:00, 9.74it/s]
9 3 0.0033333333333333335
avg loss 3.8196983632445334
avg loss change 0.5313643322388331
avg error 0.4152933729688327
100%
300/300 [00:35<00:00, 7.76it/s]
9 4 0.17
avg loss 2.6795193911592166
avg loss change 0.39626698623100914
avg error 0.1695009661714236
100%
300/300 [00:33<00:00, 9.49it/s]
9 5 0.44
avg loss 5.235816111962
avg loss change 0.371458647052447
avg error 0.03031807541847229
100%
300/300 [00:33<00:00, 8.99it/s]
9 6 0.07
avg loss 2.069377131834626
avg loss change 1.1677824054161707
avg error 0.68795611264805
100%
300/300 [00:34<00:00, 9.22it/s]
9 7 0.023333333333333334
avg loss 1.9160348110894363
avg loss change 0.841538261671861
avg error 0.6344573659201463
100%
300/300 [00:31<00:00, 9.19it/s]
9 8 0.46
avg loss 3.7813717314600943
avg loss change 0.8586706201235453
avg error 0.06401054916282495
100%
300/300 [00:35<00:00, 8.35it/s]
9 9 0.023333333333333334
avg loss 2.153989301795761
avg loss change 0.984823618332545
avg error 0.7984988825768232
100%
300/300 [00:32<00:00, 10.62it/s]
9 10 0.0033333333333333335
avg loss 2.7616556242605053
avg loss change 0.44298332224289577
avg error 0.40676913832624756
100%
300/300 [00:35<00:00, 7.96it/s]
9 11 0.01
avg loss 2.5527770445744196
avg loss change 2.0903026541074117
avg error 1.6865818623701732"""

parsed_dict = parse_data_to_dict(data)
print(parsed_dict)

# %%

fig = go.Figure()

fig.add_scatter(
    x=[x["avg_loss"] for x in parsed_dict.values()],
    y=[x["avg_error"]+x["avg_loss"] for x in parsed_dict.values()],
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
for i, atxt in enumerate(parsed_dict.keys()):
    txt=str(atxt)
    fig.add_annotation(
        x=parsed_dict[atxt]["avg_loss"],
        y=parsed_dict[atxt]["avg_error"]+parsed_dict[atxt]["avg_loss"],
        text=txt,
        showarrow=False,
        yshift=10,
    )


fig.update_layout(
    title="Loss vs Error",
    xaxis_title="Average model loss for top prompts where this head is useful",
    yaxis_title="Average model loss applying key 'the' approximation for this head",
)

# %%

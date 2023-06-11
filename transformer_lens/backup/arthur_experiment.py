# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
# os.getcwd(C:\Users\calsm\Documents\AI Alignment\SERIMATS_23\TransformerLens\transformer_lens)
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

from datasets import load_dataset
import torch as t
import torch
import einops
import plotly.express as px
import numpy as np
from tqdm import tqdm
from jaxtyping import Float
from torch import Tensor
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.utils import to_numpy
from transformer_lens.hackathon.sweep import sweep_train_model
from transformer_lens.hackathon.model import AndModel, Config, get_all_data, get_all_outputs
from transformer_lens.hackathon.train import TrainingConfig, train_model

# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
# os.getcwd(C:\Users\calsm\Documents\AI Alignment\SERIMATS_23\TransformerLens\transformer_lens)
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import torch as t
from transformer_lens.backup.ioi_dataset import IOIDataset, NAMES
import transformer_lens.backup.ioi_circuit_extraction
import einops
import plotly.express as px
import numpy as np
from tqdm import tqdm
import transformer_lens
from transformer_lens.utils import to_numpy
from transformer_lens.hackathon.sweep import sweep_train_model
from transformer_lens.hackathon.model import AndModel, Config, get_all_data, get_all_outputs
from transformer_lens.hackathon.train import TrainingConfig, train_model

# %%

def to_tensor(
    tensor,
):
    return t.from_numpy(to_numpy(tensor))

def imshow(
    tensor, 
    **kwargs,
):
    tensor = to_tensor(tensor)
    zmax = tensor.abs().max().item()

    if "zmin" not in kwargs:
        kwargs["zmin"] = -zmax
    if "zmax" not in kwargs:
        kwargs["zmax"] = zmax
    if "color_continuous_scale" not in kwargs:
        kwargs["color_continuous_scale"] = "RdBu"

    fig = px.imshow(
        to_numpy(tensor),
        **kwargs,
    )
    fig.show()
def to_tensor(
    tensor,
):
    return t.from_numpy(to_numpy(tensor))

def imshow(
    tensor, 
    **kwargs,
):
    tensor = to_tensor(tensor)
    zmax = tensor.abs().max().item()

    if "zmin" not in kwargs:
        kwargs["zmin"] = -zmax
    if "zmax" not in kwargs:
        kwargs["zmax"] = zmax
    if "color_continuous_scale" not in kwargs:
        kwargs["color_continuous_scale"] = "RdBu"

    fig = px.imshow(
        to_numpy(tensor),
        **kwargs,
    )
    fig.show()
# %%

model = transformer_lens.HookedTransformer.from_pretrained("gpt2")

# %%

N = 25
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=True,
    seed=1,
    device=DEVICE,
)

#%%

model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)

# %%

# reproduce name mover heatmap

logits, cache = model.run_with_cache(
    ioi_dataset.toks,
    names_filter = lambda name: name.endswith("hook_result"),
)

# # %%

# per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
# per_head_residual = einops.rearrange(
#     per_head_residual, 
#     "(layer head) ... -> layer head ...", 
#     layer=model.cfg.n_layers
# )

# #%%

# def residual_stack_to_logit_diff(
#     residual_stack: Float[Tensor, "... batch d_model"], 
#     cache: ActivationCache,
#     logit_diff_directions: Float[Tensor, "batch d_model"],
# ) -> Float[Tensor, "..."]:
#     '''
#     Gets the avg logit difference between the correct and incorrect answer for a given 
#     stack of components in the residual stream.
#     '''
#     # SOLUTION
#     batch_size = residual_stack.size(-2)
#     # scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
#     return einops.einsum(
#         residual_stack, logit_diff_directions,
#         "... batch d_model, batch d_model -> ..."
#     ) / batch_size

# # %%

# answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(t.tensor([ioi_dataset.io_tokenIDs, ioi_dataset.s_tokenIDs]).T)
# print("Answer residual directions shape:", answer_residual_directions.shape)

# correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
# logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions


# %%

unembedding = model.W_U.clone()
logit_attribution = t.zeros((12, 12))
for i in range(12):
    end_logits = cache["result", i][t.arange(N), ioi_dataset.word_idx["end"], :, :]
    out_dir = unembedding[:, ioi_dataset.io_tokenIDs] - unembedding[:, ioi_dataset.s_tokenIDs]

    layer_attribution_old = einops.einsum(
        end_logits,
        out_dir,
        "b n d, d b -> b n",
    )

    for j in range(12):
        logit_attribution[i, j] = layer_attribution_old[:, j].mean()

# %%

imshow(
    logit_attribution,
)

# %% [markdown]
# <p> 10.7 is a fascinating head because it REVERSES direction in the backup step</p>
# <p> What is it doing on the IOI distribution? </p>

#%%

LAYER_IDX = 10
HEAD_IDX = 7
HOOK_NAME = f"blocks.{LAYER_IDX}.attn.hook_result"

head_output = cache["result", LAYER_IDX][t.arange(N), ioi_dataset.word_idx["end"], HEAD_IDX, :]

head_logits = einops.einsum(
    head_output,
    unembedding,
    "b d, d V -> b V",
)

for b in range(10, 12):
    print("PROMPT:")
    print(ioi_dataset.tokenized_prompts[b])

    for outps_type, outps in [
        ("TOP TOKENS", t.topk(head_logits[b], 10).indices),
        ("BOTTOM_TOKENS", t.topk(-head_logits[b], 10).indices),
    ]:
        print(outps_type)
        for i in outps:
            print(ioi_dataset.tokenizer.decode(i))
    print()

print("So it seems like the bottom tokens (checked on more prompts, seems legit) are JUST the correct answer, and the top tokens are not interpretable")

# %% [markdown]
# <p> Okay let's look more generally at OWT...</p>

#%%

# Let's see some WEBTEXT
raw_dataset = load_dataset("stas/openwebtext-10k")
train_dataset = raw_dataset["train"]
dataset = [train_dataset[i]["text"] for i in range(len(train_dataset))]

# %%

# Let's see if 10.7 can be approximated as having zero bias, or whether it consistently pushes for/against some tokens
# ... let's do it one document at a time for ease ...

contributions = []

for i in tqdm(list(range(2)) + [5]):
    tokens = model.tokenizer(
        dataset[i], 
        return_tensors="pt", 
        truncation=True, 
        padding=True
    )["input_ids"].to(DEVICE)
    
    if tokens.shape[1] < 256: # lotsa short docs here
        print("SKIPPING short document", tokens.shape)
        continue
    tokens = tokens[0:1, :256]

    model.reset_hooks()
    logits, cache = model.run_with_cache(
        tokens,
        names_filter = lambda name: name in [HOOK_NAME, "ln_final.hook_scale"],
    )
    output = cache[HOOK_NAME][0, :, HEAD_IDX] / cache["ln_final.hook_scale"][0, :, 0].unsqueeze(dim=-1) # account for layer norm scaling
    
    contribution = einops.einsum(
        output,
        unembedding,
        "s d, d V -> s V",
    )
    contributions.append(contribution.clone())

    for j in range(256):
        if contribution[j].norm().item() > 80:
            print(model.to_str_tokens(tokens[0, j-30: j+1]))
            print(model.tokenizer.decode(tokens[0, j+1]))
            print()

            top_tokens = t.topk(contribution[j], 10).indices
            bottom_tokens = t.topk(-contribution[j], 10).indices

            print("TOP TOKENS")
            for i in top_tokens:
                print(model.tokenizer.decode(i))
            print()
            print("BOTTOM TOKENS")
            for i in bottom_tokens:
                print(model.tokenizer.decode(i))

full_contributions = t.cat(contributions, dim=0)

#%%

# Some dataset statistics suggest that probably 
# i) 11.10 has a persistent bias towards/against some tokens
# ii) There are certain token positions where it pushes for *somethings* way more than other positsions

# Evidence for i) 
print(full_contributions.mean(dim=0).norm().item()) # fairly large
print(full_contributions.norm(dim=0).mean().item()) # almost as large # TODO nail what I'm trying to measure here

px.histogram(to_tensor(full_contributions.norm(dim=0)), title="Distributions of 11.10 contribution-to-each-logit norms").show()
px.histogram(to_tensor(full_contributions.mean(dim=0)), title="Distributions of 11.10 mean-contribution-to-each-logit").show()
px.histogram(to_tensor(full_contributions[4]), title="Distribution of logit contributions for a single token pos").show() # so similar to Previous!!!

#%%

mean_contributions_to_tokens = full_contributions.mean(dim=0)
top_tokens = t.topk(mean_contributions_to_tokens, 10).indices
bottom_tokens = t.topk(-mean_contributions_to_tokens, 10).indices

print("TOP TOKENS")
for i in top_tokens:
    print(model.tokenizer.decode(i))
print()
print("BOTTOM TOKENS")
for i in bottom_tokens:
    print(model.tokenizer.decode(i))

# %%

# That didn't help. Maybe isolating the heavy tail of logit norms???
px.histogram(to_tensor(full_contributions.norm(dim=1)), title="Distributions of 11.10 contribution-to-each-logit norms").show()

# %%

# OBSERVATION: the top outputs are almost never interpretable and the bottom outputs almost always are... --- they seem to repress copying:
#
# GOOD: [',', ' which', ' is', ' in', ' favor', ' of', ' legalization', ',', ' blacks', ' are', ' arrested', ' for', ' marijuana', ' possession', ' between', ' four', ' and', ' twelve', ' times', ' more', ' than'] -> 11.10 blocks " blacks"
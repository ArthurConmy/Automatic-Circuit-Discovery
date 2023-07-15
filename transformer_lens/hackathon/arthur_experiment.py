#%%

naive_induction = {
    """\#\!/usr/bin/python # -*- coding: UTF-8 -*- import sys import"""
}

# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
from typeguard import typechecked
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")
from transformer_lens.cautils.notebook import *

import torch as t
import torch
import einops
import itertools
import plotly.express as px
import numpy as np
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from jaxtyping import Float, Int, jaxtyped
from typing import Union, List, Dict, Tuple, Callable, Optional
from torch import Tensor
import gc
import transformer_lens
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens import utils
from transformer_lens.utils import to_numpy
t.set_grad_enabled(False)

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

# %%

# MODEL_NAME = "solu-10l"
# MODEL_NAME = "solu-10l"
# MODEL_NAME = "gpt2-large"
# MODEL_NAME = "gpt2-medium"
MODEL_NAME = "stanford-gpt2-small-e"

dataset = get_webtext(dataset=("NeelNanda/c4-code-20k" if "solu" in MODEL_NAME.lower() else "stas/openwebtext-10k"))

model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME)
from transformer_lens.hackathon.ioi_dataset import IOIDataset, NAMES

LAYER_IDX, HEAD_IDX = 3, 0
# for LAYER_IDX, HEAD_IDX in itertools.product(range(model.cfg.n_layers-1, -1, -1), model.cfg.n_heads):
# %%

N = 100
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

ioi_dataset = IOIDataset(
    prompt_type="mixed" if model.cfg.tokenizer_name == "gpt2" else "BABA", # hacky fix for solu-10l IOi tokenization
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=True,
    nb_templates=None if model.cfg.tokenizer_name == "gpt2" else 1,
    seed=1,
    device=DEVICE,
)

#%%

model.set_use_attn_result(False)
model.set_use_split_qkv_input(True)

# %%

# reproduce name mover heatmap

logits, cache = model.run_with_cache(
    ioi_dataset.toks,
    names_filter = lambda name: name.endswith("z"),
)

# %%

logit_attribution = t.zeros((model.cfg.n_layers, model.cfg.n_heads))
for i in range(model.cfg.n_layers):
    attn_result = einops.einsum(
        cache["z", i][t.arange(N), ioi_dataset.word_idx["end"]], # (batch, head_idx, d_head)
        model.W_O[i], # (head_idx, d_head, d_model)
        "batch head_idx d_head, head_idx d_head d_model -> batch head_idx d_model",
    )
    logit_dir = model.W_U.clone()[:, ioi_dataset.io_tokenIDs] - model.W_U.clone()[:, ioi_dataset.s_tokenIDs]

    layer_attribution_old = einops.einsum(
        attn_result,
        logit_dir,
        "batch head_idx d_model, d_model batch -> batch head_idx",
    )

    for j in range(model.cfg.n_heads):
        logit_attribution[i, j] = layer_attribution_old[:, j].mean()

# %%

imshow(
    logit_attribution,
    title="GPT-2 Small head direct logit attribution",
)

# %%

# LAYER_IDX, HEAD_IDX = {
#     "SoLU_10L1280W_C4_Code": (9, 18), # (9, 18) is somewhat cheaty
#     "gpt2": (10, 7),
# }[model.cfg.model_name]

W_U = model.W_U
W_Q_negative = model.W_Q[LAYER_IDX, HEAD_IDX]
W_K_negative = model.W_K[LAYER_IDX, HEAD_IDX]

W_E = model.W_E

# ! question - what's the approximation of GPT2-small's embedding?
# lock attn to 1 at current position
# lock attn to average
# don't include attention

#%%

from transformer_lens import FactoredMatrix

full_QK_circuit = FactoredMatrix(W_U.T @ W_Q_negative, W_K_negative.T @ W_E.T)

indices = t.randint(0, model.cfg.d_vocab, (250,))
full_QK_circuit_sample = full_QK_circuit.A[indices, :] @ full_QK_circuit.B[:, indices]

full_QK_circuit_sample_centered = full_QK_circuit_sample - full_QK_circuit_sample.mean(dim=1, keepdim=True)

imshow(
    full_QK_circuit_sample_centered,
    labels={"x": "Source / key token (embedding)", "y": "Destination / query token (unembedding)"},
    title="Full QK circuit for negative name mover head",
    width=700,
)

# %%

def top_1_acc_iteration(full_QK_circuit: FactoredMatrix, batch_size: int = 100) -> float: 
    '''
    This should take the argmax of each row (i.e. over dim=1) and return the fraction of the time that's equal to the correct logit
    '''
    # SOLUTION
    A, B = full_QK_circuit.A, full_QK_circuit.B
    nrows = full_QK_circuit.shape[0]
    nrows_max_on_diagonal = 0

    for i in tqdm(range(0, nrows + batch_size, batch_size)):
        rng = range(i, min(i + batch_size, nrows))
        if rng:
            submatrix = A[rng, :] @ B
            diag_indices = t.tensor(rng, device=submatrix.device)
            nrows_max_on_diagonal += (submatrix.argmax(-1) == diag_indices).float().sum().item()
    
    return nrows_max_on_diagonal / nrows

print(f"Top-1 accuracy of full QK circuit: {top_1_acc_iteration(full_QK_circuit):.2%}")

# %%

def top_5_acc_iteration(full_OV_circuit: FactoredMatrix, batch_size: int = 100) -> float:
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    A, B = full_OV_circuit.A, full_OV_circuit.B
    nrows = full_OV_circuit.shape[0]
    nrows_top5_on_diagonal = 0

    for i in tqdm(range(0, nrows + batch_size, batch_size)):
        rng = range(i, min(i + batch_size, nrows))
        if rng:
            submatrix = A[rng, :] @ B
            diag_indices = t.tensor(rng, device=submatrix.device).unsqueeze(-1)
            top5 = t.topk(submatrix, k=5).indices
            nrows_top5_on_diagonal += (diag_indices == top5).sum().item()

    return nrows_top5_on_diagonal / nrows

print(f"Top-5 accuracy of full QK circuit: {top_5_acc_iteration(full_QK_circuit):.2%}")

# %%

def lock_attn(
    attn_patterns: Float[t.Tensor, "batch head_idx dest_pos src_pos"],
    hook: HookPoint,
    ablate: bool = False,
    bos: bool=False,
) -> Float[t.Tensor, "batch head_idx dest_pos src_pos"]:
    
    assert isinstance(attn_patterns, Float[t.Tensor, "batch head_idx dest_pos src_pos"])
    assert hook.layer() == 0

    batch, n_heads, seq_len = attn_patterns.shape[:3]
    if bos:
        attn_new = torch.zeros_like(attn_patterns)
        attn_new[:, :, :, 0] = 1.0
    else:
        attn_new = einops.repeat(t.eye(seq_len), "dest src -> batch head_idx dest src", batch=batch, head_idx=n_heads).clone().to(attn_patterns.device)
    if ablate:
        attn_new = attn_new * 0
    return attn_new

def fwd_pass_lock_attn0_to_self(
    model: HookedTransformer,
    input: Union[List[str], Int[t.Tensor, "batch seq_pos"]],
    ablate: bool = False,
    bos: bool = False,
) -> Float[t.Tensor, "batch seq_pos d_vocab"]:

    model.reset_hooks()
    
    loss = model.run_with_hooks(
        input,
        return_type="loss",
        fwd_hooks=[(utils.get_act_name("pattern", 0), partial(lock_attn, ablate=ablate, bos=bos))],
    )

    return loss

# %%

for i, s in enumerate(dataset):
    loss_hooked = fwd_pass_lock_attn0_to_self(model, s)
    print(f"Loss with attn locked to self: {loss_hooked:.2f}")
    loss_hooked_0 = fwd_pass_lock_attn0_to_self(model, s, ablate=True)
    print(f"Loss with attn locked to zero: {loss_hooked_0:.2f}")
    loss_orig = model(s, return_type="loss")
    print(f"Loss with attn free: {loss_orig:.2f}\n")
    loss_bos = fwd_pass_lock_attn0_to_self(model, s, bos=True)
    print(f"Loss with attn locked to bos: {loss_bos:.2f}\n")

    # gc.collect()

    if i == 5:
        break

# %%

if "gpt" in model.cfg.model_name: # sigh, tied embeddings
    # Calculate W_{EE} edit
    batch_size = 1000
    nrows = model.cfg.d_vocab
    W_EE = t.zeros((nrows, model.cfg.d_model)).to(DEVICE)

    for i in tqdm(range(0, nrows + batch_size, batch_size)):
        cur_range = t.tensor(range(i, min(i + batch_size, nrows)))
        if len(cur_range)>0:
            embeds = W_E[cur_range].unsqueeze(0)
            pre_attention = model.blocks[0].ln1(embeds)
            post_attention = einops.einsum(
                pre_attention, 
                model.W_V[0],
                model.W_O[0],
                "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
            )
            normalized_resid_mid = model.blocks[0].ln2(post_attention + embeds)
            resid_post = model.blocks[0].mlp(normalized_resid_mid) # TODO not resid post!!!
            W_EE[cur_range.to(DEVICE)] = resid_post

else: 
    W_EE = W_E # untied embeddings so no need to calculate!

# %%

if "gpt" in model.cfg.model_name: # sigh, tied embeddings
    # sanity check this is the same 

    def remove_pos_embed(z, hook):
        return 0.0 * z

    # setup a forward pass that 
    model.reset_hooks()
    model.add_hook(
        name="hook_pos_embed",
        hook=remove_pos_embed,
        level=1, # ???
    ) 
    model.add_hook(
        name=utils.get_act_name("pattern", 0),
        hook=lock_attn,
    )
    logits, cache = model.run_with_cache(
        torch.arange(1000).to(DEVICE).unsqueeze(0),
        names_filter=lambda name: name=="blocks.1.hook_resid_pre",
        return_type="logits",
    )


    W_EE_test = cache["blocks.1.hook_resid_pre"].squeeze(0)
    W_EE_prefix = W_EE_test[:1000]

    assert torch.allclose(
        W_EE_prefix,
        W_EE_test,
        atol=1e-4,
        rtol=1e-4,
    )   

# %%

def get_EE_QK_circuit(
    layer_idx,
    head_idx,
    random_seeds: Optional[int] = 5,
    num_samples: Optional[int] = 500,
    bags_of_words: Optional[List[List[int]]] = None, # each List is a List of unique tokens
    mean_version: bool = True,
    show_plot: bool = False,
):
    assert (random_seeds is None and num_samples is None) != (bags_of_words is None), (random_seeds is None, num_samples is None, bags_of_words is None, "Must specify either random_seeds and num_samples or bag_of_words_version")

    if bags_of_words is not None:
        random_seeds = len(bags_of_words) # eh not quite random seeds but whatever
        assert all([len(bag_of_words) == len(bags_of_words[0])] for bag_of_words in bags_of_words), "Must have same number of words in each bag of words"
        num_samples = len(bags_of_words[0])

    W_Q_head = model.W_Q[layer_idx, head_idx]
    W_K_head = model.W_K[layer_idx, head_idx]

    EE_QK_circuit = FactoredMatrix(W_U.T @ W_Q_head, W_K_head.T @ W_EE.T)
    EE_QK_circuit_result = t.zeros((num_samples, num_samples))

    for random_seed in range(random_seeds):
        if bags_of_words is None:
            indices = t.randint(0, model.cfg.d_vocab, (num_samples,))
        else:
            indices = t.tensor(bags_of_words[random_seed])

        n_layers, n_heads, d_model, d_head = model.W_Q.shape

        # assert False, "TODO: add Q and K and V biases???"
        EE_QK_circuit_sample = einops.einsum(
            EE_QK_circuit.A[indices, :],
            EE_QK_circuit.B[:, indices],
            "num_query_samples d_head, d_head num_key_samples -> num_query_samples num_key_samples"
        ) / np.sqrt(d_head)

        if mean_version:
            # we're going to take a softmax so the constant factor is arbitrary 
            # and it's a good idea to centre all these results so adding them up is reasonable
            EE_QK_mean = EE_QK_circuit_sample.mean(dim=1, keepdim=True)
            EE_QK_circuit_sample_centered = EE_QK_circuit_sample - EE_QK_mean 
            EE_QK_circuit_result += EE_QK_circuit_sample_centered.cpu()

        else:
            EE_QK_softmax = t.nn.functional.softmax(EE_QK_circuit_sample, dim=-1)
            EE_QK_circuit_result += EE_QK_softmax.cpu()

    EE_QK_circuit_result /= random_seeds

    if show_plot:
        imshow(
            EE_QK_circuit_result,
            labels={"x": "Source/Key Token (embedding)", "y": "Destination/Query Token (unembedding)"},
            title=f"EE QK circuit for head {layer_idx}.{head_idx}",
            width=700,
        )

    return EE_QK_circuit_result

#%%

def get_single_example_plot(
    layer, 
    head,
    sentence="Tony Abbott under fire from Cabinet colleagues over decision",
):
    tokens = model.tokenizer.encode(sentence)
    pattern = get_EE_QK_circuit(
        layer,
        head,
        random_seeds=None,
        num_samples=None,
        show_plot=True,
        bags_of_words=[tokens],
        mean_version=False,
    )
    imshow(
        pattern, 
        x=sentence.split(" "), 
        y=sentence.split(" "),
        title=f"Unembedding Attention Score for Head {layer}.{head}",
        labels = {"y": "Query (W_U)", "x": "Key (W_EE)"},
    )

# %%
        
# Prep some bags of words...
# OVERLY LONG because it really helps to have the bags of words the same length

bags_of_words = []

OUTER_LEN = 100
INNER_LEN = 10

idx = -1
while len(bags_of_words) < OUTER_LEN:
    idx+=1
    cur_tokens = model.tokenizer.encode(dataset[idx])
    cur_bag = []
    
    for i in range(len(cur_tokens)):
        if len(cur_bag) == INNER_LEN:
            break
        if cur_tokens[i] not in cur_bag:
            cur_bag.append(cur_tokens[i])

    if len(cur_bag) == INNER_LEN:
        bags_of_words.append(cur_bag)

#%%

if False:
    for idx in range(OUTER_LEN):
        print(model.tokenizer.decode(bags_of_words[idx]), "ye")
        softmaxed_attn = get_EE_QK_circuit(
            LAYER_IDX,
            HEAD_IDX,
            show_plot=True,
            num_samples=None,
            random_seeds=None,
            bags_of_words=bags_of_words[idx:idx+1],
            mean_version=False,
        )

#%% [markdown]
# <p> Observe that a large value of num_samples gives better results </p>

# WARNING: ! below here is with random words

for num_samples, random_seeds in [
    (2**i, 2**(10-i)) for i in range(4, 11)
]:
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer, head in tqdm(list(itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):

        bags_of_words = None
        mean_version = False

        softmaxed_attn = get_EE_QK_circuit(
            layer,
            head,
            show_plot=False,
            num_samples=num_samples,
            random_seeds=random_seeds,
            bags_of_words=bags_of_words,
            mean_version=mean_version,
        )
        if mean_version:
            softmaxed_attn = t.nn.functional.softmax(softmaxed_attn, dim=-1)
        trace = einops.einsum(
            softmaxed_attn,
            "i i -> ",
        )
        results[layer, head] = trace / softmaxed_attn.shape[0] # average attention on "diagonal"
    
    imshow(results, title=f"num_samples={num_samples}, random_seeds={random_seeds}")

#%% [markdown]
# <p> Most of the experiments from here are Arthur's early experiments on 11.10 on the full distribution </p>

logits, cache = model.run_with_cache(
ioi_dataset.toks,
names_filter = lambda name: name.endswith("hook_result"),
)

# %%

unembedding = model.W_U.clone()

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

# In this cell I look at the sequence positions where the
# NORM of the 10.7 output (divided by the layer norm scale)
# is very large across several documents
# 
# we find that 
# i) just like in IOI, the top tokens are not interpretable and the bottom tokens repress certain tokens in prompt
# ii) unlike in IOI it seems that it is helpfully blocks the wrong tokens from prompt from being activated - example:
# 
# ' blacks', ' are', ' arrested', ' for', ' marijuana', ' possession', ' between', ' four', ' and', ' twelve', ' times', ' more', ' than'] -> 10.7 REPRESSES " blacks"

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

# %%

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

model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
from transformer_lens.hackathon.ioi_dataset import IOIDataset, NAMES

# %%

N = 10
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

W_U = model.W_U
W_Q_ten_seven = model.W_Q[10, 7]
W_K_ten_seven = model.W_K[10, 7]
W_E = model.W_E

# ! question - what's the approximation of GPT2-small's embedding?
# lock attn to 1 at current position
# lock attn to average
# don't include attention

#%%

from transformer_lens import FactoredMatrix

full_QK_circuit = FactoredMatrix(W_U.T @ W_Q_ten_seven, W_K_ten_seven.T @ W_E.T)

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
) -> Float[t.Tensor, "batch head_idx dest_pos src_pos"]:
    
    assert isinstance(attn_patterns, Float[t.Tensor, "batch head_idx dest_pos src_pos"])
    assert hook.layer() == 0

    batch, n_heads, seq_len = attn_patterns.shape[:3]
    attn_new = einops.repeat(t.eye(seq_len), "dest src -> batch head_idx dest src", batch=batch, head_idx=n_heads).clone().to(attn_patterns.device)
    if ablate:
        attn_new = attn_new * 0
    return attn_new

W_pos = model.W_pos
assert model.W_pos.shape == (model.cfg.n_ctx, model.cfg.d_model)

def add_back_pos_embed(z, hook):
    sliced_embed = W_pos[:z.shape[1], :]
    reshaped_embed = sliced_embed.unsqueeze(0)
    assert z.shape[1:] == reshaped_embed.shape[1:], f"{z.shape[1:]} != {reshaped_embed.shape[1:]}"
    return z + reshaped_embed

def fwd_pass_lock_attn0_to_self(
    model: HookedTransformer,
    input: Union[List[str], Int[t.Tensor, "batch seq_pos"]],
    ablate: bool = False,
) -> Float[t.Tensor, "batch seq_pos d_vocab"]:
    model.reset_hooks()
    loss = model.run_with_hooks(
        input,
        return_type="loss",
        fwd_hooks=[
            (utils.get_act_name("pattern", 0), partial(lock_attn, ablate=ablate)),
            ("hook_pos_embed", lambda z, hook: z*0.0),
            ("blocks.1.hook_resid_pre", add_back_pos_embed),
        ],
    )

    return loss

# %%

raw_dataset = load_dataset("stas/openwebtext-10k")
train_dataset = raw_dataset["train"]
dataset = [train_dataset[i]["text"] for i in range(len(train_dataset))]

# %%

print("NOTE: now this also zeros positional embeddings...")

for i, s in enumerate(dataset):
    toks = model.to_tokens(s)
    loss_hooked = fwd_pass_lock_attn0_to_self(model, toks)
    print(f"Loss with attn locked to self: {loss_hooked:.2f}")
    loss_hooked_0 = fwd_pass_lock_attn0_to_self(model, toks, ablate=True)
    print(f"Loss with attn locked to zero: {loss_hooked_0:.2f}")
    loss_orig = model(toks, return_type="loss")
    print(f"Loss with attn free: {loss_orig:.2f}\n")

    # gc.collect()

    if i == 5:
        break

# %%

# Calculate W_{TE} edit

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
        resid_post = model.blocks[0].mlp(normalized_resid_mid)
        W_EE[cur_range.to(DEVICE)] = resid_post

# %%

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

#%%

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

for layer, head in [
    (9, 9),
    (8, 10),
    (10, 7), 
    (11, 10),
]:
    get_single_example_plot(layer, head)

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

for idx in range(OUTER_LEN):
    print(model.tokenizer.decode(bags_of_words[idx]), "ye")
    softmaxed_attn = get_EE_QK_circuit(
        10,
        7,
        show_plot=True,
        num_samples=None,
        random_seeds=None,
        bags_of_words=bags_of_words[idx:idx+1],
        mean_version=False,
    )

#%%

# observe that a large value of num_samples gives better results

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
    
    imshow(results - results.mean(), title=f"num_samples={num_samples}, random_seeds={random_seeds}")

# %%

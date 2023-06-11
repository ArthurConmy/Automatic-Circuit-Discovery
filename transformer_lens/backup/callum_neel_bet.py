# %%

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import torch as t
from torch import Tensor
from typing import List, Tuple, Optional, Callable
from jaxtyping import Float, Int
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, ActivationCache, HookedTransformer
from transformer_lens.backup.ioi_dataset import IOIDataset, NAMES
from functools import partial
from tqdm import tqdm
import itertools
import plotly.express as px

def imshow(
    tensor, 
    **kwargs,
):
    tensor = t.from_numpy(utils.to_numpy(tensor))
    zmax = tensor.abs().max().item()

    if "zmin" not in kwargs:
        kwargs["zmin"] = -zmax
    if "zmax" not in kwargs:
        kwargs["zmax"] = zmax
    if "color_continuous_scale" not in kwargs:
        kwargs["color_continuous_scale"] = "RdBu"

    fig = px.imshow(
        utils.to_numpy(tensor),
        **kwargs,
    )
    fig.show()

# %%

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
model.set_use_split_qkv_input(True)

# %%

N = 1
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=True,
    seed=1,
    device=DEVICE,
)
abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

# %%

def logits_to_ave_logit_diff_2(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset = ioi_dataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    
    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


model.reset_hooks(including_permanent=True)

ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)

ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()

def ioi_metric_2(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float = ioi_average_logit_diff,
    corrupted_logit_diff: float = abc_average_logit_diff,
    ioi_dataset: IOIDataset = ioi_dataset,
) -> float:
    '''
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset), 
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
    return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

# %%

batch_idx = 0
nnmh = (10, 7)

CIRCUIT = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative name mover": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}

model.reset_hooks(including_permanent=True)

import einops

result_diff = []
for nmh in CIRCUIT["name mover"]:
    z_ioi = ioi_cache["z", nmh[0]][batch_idx, :, nmh[1]]
    z_abc = abc_cache["z", nmh[0]][batch_idx, :, nmh[1]]
    result_diff.append(einops.einsum(z_abc - z_ioi, model.W_O[nmh[0], nmh[1]], "s dk, dk dm -> s dm"))

def hook_fn(q_input: Float[Tensor, "batch seq d_model"], hook: HookPoint):
    q_input[batch_idx, :, nnmh[1]] += sum(result_diff)
    return q_input

model.add_hook(name=utils.get_act_name("q_input", nnmh[0]), hook=hook_fn, is_permanent=True)

_, patched_cache = model.run_with_cache(ioi_dataset.toks, return_type=None)

model.reset_hooks(including_permanent=True)

# %%

import circuitsvis as cv

html = str(cv.attention.attention_heads(
    attention = ioi_cache["pattern", nnmh[0]][batch_idx, nnmh[1]].unsqueeze(0),
    tokens = model.to_str_tokens(ioi_dataset.sentences[batch_idx], prepend_bos=True),
))
with open("attention_orig.html", "w") as f:
    f.write(html)

html = str(cv.attention.attention_heads(
    attention = patched_cache["pattern", nnmh[0]][batch_idx, nnmh[1]].unsqueeze(0),
    tokens = model.to_str_tokens(ioi_dataset.sentences[batch_idx], prepend_bos=True),
))
with open("attention_patched.html", "w") as f:
    f.write(html)

# %%

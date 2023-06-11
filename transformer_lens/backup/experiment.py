# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
# os.getcwd(C:\Users\calsm\Documents\AI Alignment\SERIMATS_23\TransformerLens\transformer_lens)
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import torch as t
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

# %%

per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_residual = einops.rearrange(
    per_head_residual, 
    "(layer head) ... -> layer head ...", 
    layer=model.cfg.n_layers
)

#%%

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given 
    stack of components in the residual stream.
    '''
    # SOLUTION
    batch_size = residual_stack.size(-2)
    # scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size

# %%

answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(t.tensor([ioi_dataset.io_tokenIDs, ioi_dataset.s_tokenIDs]).T)
print("Answer residual directions shape:", answer_residual_directions.shape)

correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions


# %%

logit_attribution = t.zeros((12, 12))
for i in range(12):
    layer_attribution = einops.einsum(
        cache["result", i][t.arange(N), ioi_dataset.word_idx["end"], :, :],
        logit_diff_directions, # model.W_U, # TODO Arthur memorise these...
        "b n d, b d -> b n",
    )
    
    unembedding = model.W_U.clone()
    unembedding = model.unembed.W_U.clone()

    # layer_attribution_old = einops.einsum(

    # )

    for j in range(12):
        logit_attribution[i, j] = layer_attribution[:, j].mean()

# %%

imshow(
    logit_attribution,
)
# %%

# %%

import os

from typeguard import typechecked

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
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
from datasets import load_dataset
from tqdm import tqdm
from jaxtyping import Float, Int, jaxtyped
from typing import Union, List, Dict, Tuple, Callable, Optional
from torch import Tensor
import gc
import transformer_lens
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens import utils
from transformer_lens.utils import to_numpy
from transformer_lens.hackathon.sweep import sweep_train_model
from transformer_lens.hackathon.model import AndModel, Config, get_all_data, get_all_outputs
from transformer_lens.hackathon.train import TrainingConfig, train_model

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

from transformer_lens.backup.ioi_dataset import IOIDataset, NAMES

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

logit_attribution = t.zeros((12, 12))
for i in range(12):
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

    for j in range(12):
        logit_attribution[i, j] = layer_attribution_old[:, j].mean()

# %%

imshow(
    logit_attribution,
)
# %%

W_U = model.W_U
W_Q = model.W_Q[10, 7]
W_K = model.W_K[10, 7]
W_E = model.W_E

# ! question - what's the approximation of GPT2-small's embedding?
# lock attn to 1 at current position
# lock attn to average
# don't include attention

from transformer_lens import FactoredMatrix

full_QK_circuit = FactoredMatrix(W_U.T @ W_Q, W_K.T @ W_E.T)

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

def fwd_pass_lock_attn0_to_self(
    model: HookedTransformer,
    input: Union[List[str], Int[t.Tensor, "batch seq_pos"]],
    ablate: bool = False
) -> Float[t.Tensor, "batch seq_pos d_vocab"]:

    model.reset_hooks()

    def hook_attn(
        attn_patterns: Float[t.Tensor, "batch head_idx dest_pos src_pos"],
        hook: HookPoint
    ) -> Float[t.Tensor, "batch head_idx dest_pos src_pos"]:
        
        assert isinstance(attn_patterns, Float[t.Tensor, "batch head_idx dest_pos src_pos"])
        assert hook.layer() == 0

        batch, n_heads, seq_len = attn_patterns.shape[:3]
        attn_new = einops.repeat(t.eye(seq_len), "dest src -> batch head_idx dest src", batch=batch, head_idx=n_heads).clone().to(attn_patterns.device)
        if ablate:
            attn_new = attn_new * 0
        return attn_new
    
    loss = model.run_with_hooks(
        input,
        return_type="loss",
        fwd_hooks=[(utils.get_act_name("pattern", 0), hook_attn)],
    )

    return loss


# %%

raw_dataset = load_dataset("stas/openwebtext-10k")
train_dataset = raw_dataset["train"]
dataset = [train_dataset[i]["text"] for i in range(len(train_dataset))]

# %%

for i, s in enumerate(dataset):
    loss_hooked = fwd_pass_lock_attn0_to_self(model, s)
    print(f"Loss with attn locked to self: {loss_hooked:.2f}")
    loss_hooked_0 = fwd_pass_lock_attn0_to_self(model, s, ablate=True)
    print(f"Loss with attn locked to zero: {loss_hooked_0:.2f}")
    loss_orig = model(s, return_type="loss")
    print(f"Loss with attn free: {loss_orig:.2f}\n")

    # gc.collect()

    if i == 5:
        break

# %%

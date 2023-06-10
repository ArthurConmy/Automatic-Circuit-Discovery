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
from tqdm import tqdm
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

#%%


cfg = Config(
    N=10,
    M=10,
    d_model=40,
    d_mlp=20,
    relu_at_end=False,
)

train_cfg = TrainingConfig(num_epochs=1000, weight_decay=1e-2)

# %%

and_model, loss_list = train_model(cfg, train_cfg)
px.line(loss_list)

# %%

all_outputs, cache = get_all_outputs(and_model, return_cache=True) # all_outputs N*M, N, M

# %%

px.imshow(
    cache["mlp.hook_post"],
    animation_frame=-1,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
)

# %%

N_range = [3]
d_mlp_range = [2]
d_model_range = [3] # list(range(3, 11))
seed_range = [3]
num_epochs = 3000

cfg_list = [
    Config(N=N, M=N, d_model=d_model, d_mlp=d_mlp, relu_at_end=True)
    for N in N_range
    for d_mlp in d_mlp_range
    for d_model in d_model_range
    for seed in seed_range
]
train_cfg_list = [
    TrainingConfig(num_epochs=num_epochs, weight_decay=0.0, batch_size=N*N, seed=seed)
    for N in N_range
    for d_mlp in d_mlp_range
    for d_model in d_model_range
    for seed in seed_range
]

loss_tensor = sweep_train_model(cfg_list, train_cfg_list, save_models=True, verbose=True, show_plot=True)

# %%

and_model, loss_list = train_model(cfg_list[1], train_cfg_list[1])
px.line(loss_list)

#%% 

and_model = AndModel(cfg_list[0])
and_model.load_state_dict(t.load("repo_models/and_model_Config_N=3_M=3_d_model=3_d_mlp=2_relu_at_end=True_correct=6.pth"))

# %%

outputs, cache = get_all_outputs(and_model, return_cache=True)

# %%

imshow(
    einops.rearrange(outputs, "n m ... -> (n m) ..."),
    animation_frame=0,
)

# %%

imshow(
    cache["mlp.hook_post"],
    facet_col=-1,
    title="Neuron activations post",
)
# %%

neuron_outs = einops.einsum(
    and_model.mlp.W_out,
    and_model.unembed.W_U,
    "d_mlp d_model, d_model N M -> d_mlp N M",
)

imshow(
    neuron_outs,
    facet_col=0,
    title="Neuron contribtions",
)
# %%

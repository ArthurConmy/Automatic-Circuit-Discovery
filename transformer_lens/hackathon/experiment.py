# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
# os.getcwd(C:\Users\calsm\Documents\AI Alignment\SERIMATS_23\TransformerLens\transformer_lens)
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import torch
import einops
import plotly.express as px
from tqdm import tqdm
from transformer_lens.hackathon.sweep import sweep_train_model
from transformer_lens.hackathon.model import AndModel, Config, get_all_data, get_all_outputs
from transformer_lens.hackathon.train import TrainingConfig, train_model

# %%

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

# plotly code to vizualize output of the model
px.imshow(
    all_outputs,
    animation_frame=0,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
)

# %%

px.imshow(
    cache["mlp.hook_post"],
    animation_frame=-1,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
)

# %%

N_range = [3, 5, 10, 15, 20, 30, 40, 50]
d_mlp_range = [3]
d_model_range = [5, 10, 20, 40]

cfg_list = [
    Config(N=N, M=N, d_model=d_model, d_mlp=d_mlp, relu_at_end=True)
    for N in N_range
    for d_mlp in d_mlp_range
    for d_model in d_model_range
]
train_cfg_list = [
    TrainingConfig(num_epochs=1500, weight_decay=0.0, batch_size=N*N)
    for N in N_range
    for d_mlp in d_mlp_range
    for d_model in d_model_range
]

loss_tensor = sweep_train_model(cfg_list, train_cfg_list, save_models=True, verbose=True)

# %%

and_model, loss_list = train_model(cfg_list[1], train_cfg_list[1])
px.line(loss_list)
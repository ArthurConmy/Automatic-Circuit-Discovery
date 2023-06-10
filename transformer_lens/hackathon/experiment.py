# %%

# import os
# os.getcwd(C:\Users\calsm\Documents\AI Alignment\SERIMATS_23\TransformerLens\transformer_lens)
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import plotly.express as px
from transformer_lens.hackathon.sweep import sweep_train_model
from transformer_lens.hackathon.model import AndModel, Config, get_all_data, get_all_outputs
from transformer_lens.hackathon.train import TrainingConfig, train_model

# %%

cfg = Config(
    N=10,
    M=10,
    d_model=40,
    d_mlp=20,
    relu_at_end=True,
)

train_cfg = TrainingConfig(num_epochs=500, weight_decay=0.0)

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

cfg_list = [
    Config(N=N, M=N, d_model=40, d_mlp=d_mlp, relu_at_end=True)
    for N in range(1, 11)
    for d_mlp in range(5, 25)
]

loss_tensor = sweep_train_model(cfg_list, train_cfg)

# %%
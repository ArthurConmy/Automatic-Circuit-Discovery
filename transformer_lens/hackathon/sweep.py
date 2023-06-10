# %%

from typing import List, Dict, Optional, Callable, Any, Union
import torch
from dataclasses import dataclass
import einops
from tqdm import tqdm
from transformer_lens.utils import to_numpy
import matplotlib.pyplot as plt
import plotly.express as px

from transformer_lens.hackathon.model import AndModel, Config, get_all_data
from transformer_lens.hackathon.train import train_model, TrainingConfig

# %%

def sweep_train_model(
    cfg_list: List[Config], 
    train_cfg: Union[TrainingConfig, List[TrainingConfig]],
    save_models: bool = True,
    verbose: bool = False,
    show_plot: bool = False,
):

    train_cfg_list = train_cfg if not isinstance(train_cfg, TrainingConfig) else [train_cfg] * len(cfg_list)

    loss_tensor = torch.zeros(len(cfg_list))

    for idx, (cfg, train_cfg) in tqdm(list(enumerate(zip(cfg_list, train_cfg_list)))):

        and_model, loss_list = train_model(cfg, train_cfg, use_progress_bar=False)
        # todo - early stopping

        if show_plot:
            px.line(loss_list)

        if save_models:
            torch.save(and_model.state_dict(), f"saved_models/and_model_{str(cfg)}.pth")

        if verbose:
            loss_scaled = torch.tensor(loss_list[-3:]) * cfg.N * cfg.M
            loss_scaled = ", ".join([f"{x:.4f}" for x in loss_scaled])
            n_superposed = cfg.N * cfg.M * (1 - loss_list[-1])
            printout = f"cfg: {cfg.get_str()}\nloss: {loss_scaled}\nn_superposed: {n_superposed:.4f}\n\n"
            print(printout)

        loss_tensor[idx] = loss_list[-1] * cfg.N * cfg.M

    return loss_tensor



# %%

# 24%|██▍       | 6/25 [00:27<01:27,  4.61s/it]
# cfg: Config(N=4, M=4, d_model=20, d_mlp=3, relu_at_end=True)
# loss: 10.0000, 10.0000, 10.0000
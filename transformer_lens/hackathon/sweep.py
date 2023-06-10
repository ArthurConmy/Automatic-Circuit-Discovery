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
            printout = f"cfg: {str(cfg)}\nloss: {loss_scaled}\n\n"
            print(printout)

        loss_tensor[idx] = loss_list[-1] * cfg.N * cfg.M

    return loss_tensor



# %%


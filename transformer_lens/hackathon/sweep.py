# %%

from typing import List, Dict, Optional, Callable, Any, Union
import torch
from dataclasses import dataclass
import einops
from tqdm import tqdm
from transformer_lens.utils import to_numpy

from transformer_lens.hackathon.model import AndModel, Config, get_all_data
from transformer_lens.hackathon.train import train_model, TrainingConfig

# %%

def sweep_train_model(cfg_list: List[Config], train_cfg: Union[TrainingConfig, List[TrainingConfig]]):

    train_cfg_list = train_cfg if not isinstance(train_cfg, TrainingConfig) else [train_cfg] * len(cfg_list)

    loss_tensor = torch.zeros(len(cfg_list))

    for idx, (cfg, train_cfg) in enumerate(zip(cfg_list, train_cfg_list)):

        and_model, loss_list = train_model(cfg, train_cfg, use_progress_bar=False)
        # todo - early stopping

        loss_tensor[idx] = loss_list[-1] * cfg.N * cfg.M

    return loss_tensor



# %%


#%%

import torch
from dataclasses import dataclass
import einops
from tqdm import tqdm
from transformer_lens.utils import to_numpy

from transformer_lens.hackathon.model import AndModel, Config, get_all_data

# %%


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    weight_decay: float = 1.0e-2
    batch_size: int = 100
    num_epochs: int = 10
    print_every: int = 10
    seed: int = 42

# %%

def train_model(cfg: Config, train_cfg: TrainingConfig, use_progress_bar: bool = True):

    loss_list = []
    assert (cfg.N * cfg.M) % train_cfg.batch_size == 0
    torch.manual_seed(train_cfg.seed)
    and_model = AndModel(cfg).to(cfg.device)

    data, labels = get_all_data(cfg.N, cfg.M)

    optimizer = torch.optim.AdamW(and_model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    
    progress_bar = tqdm(range(train_cfg.num_epochs)) if use_progress_bar else range(train_cfg.num_epochs)
    for epoch in progress_bar:

        torch.cuda.empty_cache()

        indices = torch.randperm(cfg.N * cfg.M)

        curr_data = einops.rearrange(
            data[indices], 
            "(n_batches batch_size) two -> n_batches batch_size two",
            batch_size=train_cfg.batch_size
        )
        curr_labels = einops.rearrange(
            labels[indices], 
            "(n_batches batch_size) ... -> n_batches batch_size ...",
            batch_size=train_cfg.batch_size
        )

        n_batches, batch_size = curr_data.shape[:2]

        # print(curr_data.shape)
        # print(curr_labels.shape)

        for batch_idx, (batch_data, batch_labels) in list(enumerate(zip(curr_data, curr_labels))):
            batch_data = batch_data.to(cfg.device)
            batch_labels = batch_labels.to(cfg.device)
            logits = and_model(batch_data)
            loss = (logits - batch_labels.float()).pow(2).sum() / batch_size
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

        if use_progress_bar:
            progress_bar.set_description(f"Epoch {epoch}, loss {loss.item():.3e}")

    return and_model, loss_list

#%%



# %%
#%%

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import torch.nn as nn
from transformer_lens.HookedTransformer import HookedRootModule, HookPoint
import torch
import einops
import torch.nn as nn
from jaxtyping import Float, Int
from typing import Dict, Union
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

# Embed & Unembed
class Embed(torch.nn.Module):
    """Copied from TransformerLens (but doesn't rely on HookedTransformerConfig...)"""

    def __init__(self, d_vocab, d_model):
        super().__init__()

        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(d_vocab, d_model)
        )

    def forward(
        self, tokens: Int[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch d_model"]:
        return self.W_E[tokens, :]

class Unembed(torch.nn.Module):
    def __init__(self, d_model, d_vocab1, d_vocab2):
        super().__init__()

        self.d_model = d_model
        self.d_vocab1 = d_vocab1
        self.d_vocab2 = d_vocab2

        self.W_U: Float[torch.Tensor, "d_model d_vocab1 d_vocab2"] = nn.Parameter(
            torch.empty(self.d_model, self.d_vocab1, self.d_vocab2)
        )
        self.b_U: Float[torch.Tensor, "d_vocab1 d_vocab2"] = nn.Parameter(
            torch.zeros(self.d_vocab1, self.d_vocab2)
        )

    def forward(
        self, residual: Float[torch.Tensor, "batch d_model"]
    ) -> Float[torch.Tensor, "batch d_vocab1 d_vocab2"]:
        return (
            einops.einsum(
                residual,
                self.W_U,
                "batch d_model, d_model d_vocab1 d_vocab2 -> batch d_vocab1 d_vocab2",
            )
            + self.b_U
        )

#%%

# MLP Layers
class MLP(HookPoint):
    def __init__(
        self,
        d_model, 
        d_mlp,
        act_fn="relu",
    ):
        """Copied from TransformerLens (but doesn't rely on HookedTransformerConfig...)"""

        self.d_model = d_model
        self.d_mlp = d_mlp
        self.act_fn = act_fn

        super().__init__()
        self.W_in = nn.Parameter(torch.empty(self.d_model, self.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(self.d_mlp))
        self.W_out = nn.Parameter(torch.empty(self.d_mlp, self.d_model))
        self.b_out = nn.Parameter(torch.zeros(self.d_model))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.act_fn == "relu":
            self.act_fn = F.relu
        elif self.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.act_fn == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Invalid activation function name: {self.act_fn}")

    def forward(
        self, x: Float[torch.Tensor, "batch d_model"]
    ) -> Float[torch.Tensor, "batch d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(
            einops.einsum(
                x, 
                self.W_in,
                "batch d_model, d_model d_mlp -> batch d_mlp", 
            )
            + self.b_in
        )  # [batch, pos, d_mlp]

        post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]

        return (
            einops.einsum(
                post_act,
                self.W_out,
                "batch d_mlp, d_mlp d_model -> batch d_model",
            )
            + self.b_out
        )

#%%


@dataclass
class Config:
    N: int = 10
    M: int = 10
    d_model: int = 40
    d_mlp: int = 20
    relu_at_end: bool = False

class AndModel(HookedRootModule):

    def __init__(self, cfg: Config):
        super().__init__()

        self.embed1 = Embed(d_model=cfg.d_model, d_vocab=cfg.N)
        self.hook_embed1 = HookPoint()
        self.embed2 = Embed(d_model=cfg.d_model, d_vocab=cfg.M)
        self.hook_embed2 = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.mlp = MLP(d_model=cfg.d_model, d_mlp=cfg.d_mlp)
        self.hook_resid_post = HookPoint()   
        self.unembed = Unembed(d_model=cfg.d_model, d_vocab1=cfg.N, d_vocab2=cfg.M)
        self.relu_at_end = cfg.relu_at_end

        if self.relu_at_end:
            self.hook_relu_out = HookPoint()

        super().setup()
        self.init_weights()

    def init_weights(self):
        weight_names = [name for name, param in self.named_parameters() if "W_" in name]
        assert sorted(weight_names) == ['embed1.W_E', 'embed2.W_E', 'mlp.W_in', 'mlp.W_out', 'unembed.W_U'], sorted(weight_names)
        for name, param in self.named_parameters():
            if "W_" in name: # W_in, W_out, W_E * 2, W_U
                nn.init.normal_(param, std=(1/param.shape[0])**0.5)
                # TODO - maybe return to do smth better?

    def forward(self, x: Int[torch.Tensor, "batch 2"]):
        e1 = self.hook_embed1(self.embed1(x[:, 0]))
        e2 = self.hook_embed2(self.embed2(x[:, 1]))
        mlp_pre = self.hook_resid_pre(e1 + e2)
        mlp = self.hook_resid_post(self.mlp(mlp_pre))
        unembed = self.unembed(mlp)
        if self.relu_at_end:
            return self.hook_relu_out(F.relu(unembed))
        else: 
            return unembed

# %%

# now onto training

def get_all_data(N, M):
    all_data_inputs = torch.zeros((N*M, 2)).long()
    all_data_labels = torch.zeros((N*M, N, M)).long()
    for i in range(N):
        for j in range(M):
            all_data_labels[i*M + j, i, j] = 1
            all_data_inputs[i*M + j, 0] = i
            all_data_inputs[i*M + j, 1] = j
    return all_data_inputs, all_data_labels

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

cfg = Config(relu_at_end=True)
train_cfg = TrainingConfig()

input_data, input_labels = get_all_data(cfg.N, cfg.M)

# and_model.reset_hooks()
# logits, cache = and_model.run_with_cache(
#     input_data[0].unsqueeze(0),
# )
# print(cache.keys())

# %%

def train_model(cfg: Config, train_cfg: TrainingConfig):

    loss_list = []
    assert (cfg.N * cfg.M) % train_cfg.batch_size == 0
    torch.manual_seed(train_cfg.seed)
    and_model = AndModel(cfg).to(device)

    data, labels = get_all_data(cfg.N, cfg.M)

    optimizer = torch.optim.AdamW(and_model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    
    progress_bar = tqdm(range(train_cfg.num_epochs))
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
        # print(curr_data.shape)
        # print(curr_labels.shape)

        for batch_idx, (batch_data, batch_labels) in list(enumerate(zip(curr_data, curr_labels))):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            logits = and_model(batch_data)
            loss = (logits - batch_labels.float()).pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

        progress_bar.set_description(f"Epoch {epoch}, loss {loss.item():.3e}")

    return and_model, loss_list


cfg = Config(d_mlp=100, d_model=100, relu_at_end=True)
train_cfg = TrainingConfig(num_epochs=1000, weight_decay=0.0)
and_model, loss_list = train_model(cfg, train_cfg)

import plotly.express as px
px.line(loss_list)

# %%

# plotly code to vizualize output

px.imshow(
    # 3D tensor,
    animation_frame=0,
)
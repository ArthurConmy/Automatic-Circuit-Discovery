# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import torch.nn as nn
from transformer_lens.HookedTransformer import HookedRootModule, HookPoint
from transformer_lens.utils import to_numpy
import torch
import einops
import torch.nn as nn
from jaxtyping import Float, Int
from typing import Dict, Union
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm

MAIN = __name__ == "__main__"


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
    N: int
    M: int
    d_model: int
    d_mlp: int
    relu_at_end: bool
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self) -> str:
        return f"N={self.N}_M={self.M}_d_model={self.d_model}_d_mlp={self.d_mlp}_relu={self.relu_at_end}"

    def __repr__(self) -> str:
        return f"N={self.N}_M={self.M}_d_model={self.d_model}_d_mlp={self.d_mlp}_relu={self.relu_at_end}"

class AndModel(HookedRootModule):

    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
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

    def to(self, device):
        self.cfg.device = device
        return super().to(self.cfg.device)

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

def get_all_outputs(model: AndModel, return_cache=False):
    all_data, all_labels = get_all_data(N=model.cfg.N, M=model.cfg.M)
    
    if return_cache:
        all_outputs, all_cache = model.run_with_cache(all_data)
        all_outputs = to_numpy(all_outputs)
    else:
        all_outputs = to_numpy(model(all_data))
        all_cache = None

    all_outputs = einops.rearrange(
        all_outputs,
        "(n m) ... -> n m ...",
        n=model.cfg.N,
    )

    cache_keys = list(all_cache.keys())
    for key in cache_keys:
        all_cache[key] = einops.rearrange(
            to_numpy(all_cache[key]),
            "(n m) ... -> n m ...",
            n=model.cfg.N,
        )

    return all_outputs, all_cache
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

from transformer_lens.components import Embed, Unembed, MLP
from transformer_lens.HookedTransformer import HookedRootModule, HookPoint
import torch
import einops
import torch.nn as nn
from jaxtyping import Float, Int
from typing import Dict, Union
import torch.nn.functional as F

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

class AndModel(HookedRootModule):

    # TODO; move args to config

    def __init__(
        self,
        N, 
        M,
        d_model,
        d_mlp,
        relu_at_end=False,
    ):
        super().__init__()

        self.embed1 = Embed(d_model=d_model, d_vocab=N)
        self.hook_embed1 = HookPoint()
        self.embed2 = Embed(d_model=d_model, d_vocab=M)
        self.hook_embed2 = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.mlp = MLP(d_model=d_model, d_mlp=d_mlp)
        self.hook_resid_post = HookPoint()   
        self.unembed = Unembed(d_model=d_model, d_vocab1=N, d_vocab2=M)
        self.relu_at_end = relu_at_end

        if self.relu_at_end:
            self.hook_relu_out = HookPoint()

        super().setup()

    def forward(self, x1, x2):
        e1 = self.hook_embed1(self.embed1(x1))
        e2 = self.hook_embed2(self.embed2(x2))
        mlp_pre = self.hook_resid_pre(e1 + e2)
        mlp = self.hook_resid_post(self.mlp(mlp_pre))
        unembed = self.unembed(mlp)
        if self.relu_at_end:
            return self.hook_relu_out(F.relu(unembed))
        else: 
            return unembed

N = 100
M = 100
d_model = 50
d_mlp = 20
batch_size = 100
and_model = AndModel(N=N, M=M, d_model=d_model, d_mlp=d_mlp)

#%%

input_data1 = torch.randint(N, (batch_size,))
input_data2 = torch.randint(M, (batch_size,))

# %%

and_model.reset_hooks()
logits, cache = and_model.run_with_cache(
    input_data1,
    input_data2, 
    # return_type="logits",
)

# %%
 
print(cache.keys())

# %%

# now onto training
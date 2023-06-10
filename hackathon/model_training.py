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
import torch.nn as nn
from jaxtyping import Float, Int
from typing import Dict, Union

#%%

# Embed & Unembed
class Embed(HookPoint):
    def __init__(self, d_vocab, d_model):
        super().__init__()

        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(d_vocab, d_model)
        )

    def forward(
        self, tokens: Int[torch.Tensor, "batch pos"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return self.W_E[tokens, :]

class AndModel(HookedRootModule):
    def __init__(
        self,
        N, 
        M,
        d_model,
        d_mlp,
    ):
        super().__init__()

        self.embed1 = Embed(d_model=d_model, d_vocab=N)
        self.embed2 = Embed(d_model=d_model, d_vocab=M)

        super.setup()
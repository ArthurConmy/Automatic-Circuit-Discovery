import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from torchtyping import TensorType as TT
from transformer_lens.torchtyping_helper import T


@dataclass
class HookedTransformerKeyValueCacheEntry:
    past_keys: TT[T.batch, T.pos_so_far, T.n_heads, T.d_head]
    past_values: TT[T.batch, T.pos_so_far, T.n_heads, T.d_head]

    @classmethod
    def init_cache_entry(
        cls,
        cfg: HookedTransformerConfig,
        device: torch.device,
        batch_size: int = 1,
    ):
        return cls(
            past_keys=torch.empty(
                (batch_size, 0, cfg.n_heads, cfg.d_head), device=device
            ),
            past_values=torch.empty(
                (batch_size, 0, cfg.n_heads, cfg.d_head), device=device
            ),
        )

    def append(
        self,
        new_keys: TT[T.batch, T.new_tokens, T.n_heads, T.d_head],
        new_values: TT[T.batch, T.new_tokens, T.n_heads, T.d_head],
    ):
        updated_keys: TT[
            "batch", "pos_so_far + new_tokens", "n_heads", "d_head"
        ] = torch.cat([self.past_keys, new_keys], dim=1)
        updated_values: TT[
            "batch", "pos_so_far + new_tokens", "n_heads", "d_head"
        ] = torch.cat([self.past_values, new_values], dim=1)
        self.past_keys = updated_keys
        self.past_values = updated_values
        return updated_keys, updated_values


@dataclass
class HookedTransformerKeyValueCache:
    """
    A cache for storing past keys and values for the Transformer. This is important for generating text - we can cache a lot of past computation and avoid repeating ourselves!

    This cache is a list of HookedTransformerKeyValueCacheEntry objects, one for each layer in the Transformer. Each object stores a [batch, pos_so_far, n_heads, d_head] tensor for both keys and values, and each entry has an append method to add a single new key and value.

    Generation is assumed to be done by initializing with some prompt and then continuing iteratively one token at a time. So append only works for adding a single token's worth of keys and values, and but the cache can be initialized with many.

    """

    entries: List[HookedTransformerKeyValueCacheEntry]

    @classmethod
    def init_cache(
        cls, cfg: HookedTransformerConfig, device: torch.device, batch_size: int = 1
    ):
        return cls(
            entries=[
                HookedTransformerKeyValueCacheEntry.init_cache_entry(
                    cfg, device, batch_size
                )
                for _ in range(cfg.n_layers)
            ]
        )

    def __getitem__(self, idx):
        return self.entries[idx]

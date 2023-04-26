#%%

from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore

import acdc
from acdc import HookedTransformer
from acdc.induction.utils import get_all_induction_things
import torch

# %%

num_examples = 40
seq_len = 300
    
(
    tl_model,
    toks_int_values,
    toks_int_values_other,
    metric,
    mask_rep,
) = get_all_induction_things(kl_return_tensor=True, num_examples=num_examples, seq_len=seq_len, device="cuda", return_mask_rep=True)

assert tl_model.cfg.use_attn_result, "set htis to Ttue,"

#%%

_, corrupted_cache = tl_model.run_with_cache(
    toks_int_values_other,
)

#%%

clean_cache = tl_model.add_caching_hooks(
    # toks_int_values,
    incl_bwd=True,
)

clean_logits = tl_model(toks_int_values)
kl_result = metric(clean_logits)
kl_result.backward()

#%%

shap = list(clean_cache["blocks.0.attn.hook_result"].shape)
assert len(shap) == 4, shap
assert shap[2] == 8, shap # not num_heads ???

#%%

results = {}

for layer_idx in range(2):

    fwd_hook_name = f"blocks.{layer_idx}.attn.hook_result"
    bwd_hook_name = f"blocks.{layer_idx}.attn.hook_result_grad"

    cur_results = torch.abs(torch.einsum(
        "bshd,bshd->bsh",
        clean_cache[f"blocks.{layer_idx}.attn.hook_result_grad"], # gradient
        clean_cache[fwd_hook_name] - corrupted_cache[fwd_hook_name],
    ))

    for head_idx in range(8):
        results[(layer_idx, head_idx)] = cur_results[:, :, head_idx].sum() / mask_rep.int().sum().item()
        print(layer_idx, head_idx, results[(layer_idx, head_idx)].item())

    # seems reasonable

# %%

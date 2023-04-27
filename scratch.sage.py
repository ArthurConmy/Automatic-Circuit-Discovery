#%%

from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore

from tqdm import tqdm
import warnings
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
) = get_all_induction_things(
    kl_return_tensor=True, 
    num_examples=num_examples, 
    seq_len=seq_len, 
    device="cuda", 
    return_mask_rep=True,
    return_one_element=False
)

assert tl_model.cfg.use_attn_result, "Set this to True"

#%%

with torch.no_grad():
    _, corrupted_cache = tl_model.run_with_cache(
        toks_int_values_other,
    )
tl_model.zero_grad() # extra hopeful....
tl_model.global_cache.second_cache = corrupted_cache

#%%

clean_cache = tl_model.add_caching_hooks(
    # toks_int_values,
    incl_bwd=True,
)

clean_logits = tl_model(toks_int_values) 
kl_result = metric(clean_logits)
assert list(kl_result.shape) == [num_examples, seq_len], kl_result.shape
kl_result = (kl_result * mask_rep).sum() / mask_rep.int().sum().item()
kl_result.backward(retain_graph=True)

#%%

shap = list(clean_cache["blocks.0.attn.hook_result"].shape)
assert len(shap) == 4, shap
assert shap[2] == 8, shap # not num_heads ???

#%%

keys = []
for layer_idx in range(2):
    for head_idx in range(8):
        keys.append((layer_idx, head_idx))

results = {(layer_idx, head_idx): torch.zeros(size=(num_examples, seq_len)) for layer_idx, head_idx in keys}

for i in tqdm(range(num_examples)):
    for j in tqdm(range(seq_len)):

        if mask_rep[i, j] == 0:
            continue

        tl_model.zero_grad()
        tl_model.reset_hooks()
        clean_cache = tl_model.add_caching_hooks(incl_bwd=True)
        clean_logits = tl_model(toks_int_values)
        kl_result = metric(clean_logits)[i, j]
        kl_result.backward(retain_graph=True)

        for layer_idx in range(2):
            fwd_hook_name = f"blocks.{layer_idx}.attn.hook_result"
            bwd_hook_name = f"blocks.{layer_idx}.attn.hook_result_grad"

            cur_results = torch.abs(torch.einsum(
                "bshd,bshd->bsh",
                clean_cache[bwd_hook_name], # gradient
                clean_cache[fwd_hook_name] - corrupted_cache[fwd_hook_name],
            ))

            for head_idx in range(8):
                results[(layer_idx, head_idx)][(i, j)] = cur_results[i, j, head_idx].item()

#%%

torch.save(results, "bad_myf1.pt")

# %%

kls = {(layer_idx, head_idx): torch.zeros(size=(num_examples, seq_len)) for layer_idx, head_idx in results.keys()}

from tqdm import tqdm

for i in tqdm(range(num_examples)):
    for j in tqdm(range(seq_len)):

        if mask_rep[i, j] == 0:
            continue # lolololol

        tl_model.zero_grad()
        warnings.warn("Untested reset...")
        tl_model.reset_hooks()
        clean_cache = tl_model.add_caching_hooks(incl_bwd=True)
        clean_logits = tl_model(toks_int_values)
        kl_result = metric(clean_logits)[i, j]
        print(f"{kl_result=}")
        kl_result.backward(retain_graph=True)

        for layer_idx in range(2):
            fwd_hook_name = f"blocks.{layer_idx}.attn.hook_result"

            for head_idx in range(8):
                g = tl_model.hook_dict[fwd_hook_name].xi.grad[0, 0, head_idx, 0].norm().item()
                kls[(layer_idx, head_idx)][i, j] = g

#%%

torch.save(kls, "myf.pt")

#%% 

results2 = torch.load("myf.pt")
results2 = {
    key: value.sum() / mask_rep.int().sum().item() for key, value in results2.items()
}

# compare results and results2

#%%

results2 = {
    (layer_idx, head_idx): kls[(layer_idx, head_idx)].sum() / mask_rep.int().sum().item() for layer_idx, head_idx in results.keys()
}

#%%

for k in results:
    print(k, results[k].norm().item(), kls[k].norm().item())

# %%

#%%

from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import torch
from collections import OrderedDict
import warnings
from acdc import HookedTransformer, HookedTransformerConfig # TODO why don't we have to do HookedTransformer.HookedTransformer???
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import TorchIndex
from acdc.graphics import show_pp
from acdc.induction.utils import get_all_induction_things, one_item_per_batch

NUM_EXAMPLES = 5
SEQ_LEN = 300
USE_BATCH = True
ZERO_WQ = True

#%%

try:
    induction_tuple = get_all_induction_things(NUM_EXAMPLES, SEQ_LEN, "cuda", return_mask_rep=True, kl_return_tensor=True, return_base_model_probs=True, kl_take_mean=False)
    model, toks_int_values, toks_int_values_other, metric, mask_rep, base_model_probs = induction_tuple

    if USE_BATCH:
        toks_int_values_batch, toks_int_values_other_batch, end_positions, metric = one_item_per_batch(toks_int_values, toks_int_values_other, mask_rep, base_model_probs, kl_take_mean=False)
    else:
        toks_int_values_batch, toks_int_values_other_batch = toks_int_values, toks_int_values_other

except Exception as e:
    warnings.warn(f"Failed to load induction; error: {e}")
    cfg = { # HookedTransformerConfig(
        "n_layers": 2,
        "d_model": 128,
        "n_heads": 7,
        "n_ctx": 1024,
        "d_head": 128,
        "attn_only": True,
        "d_vocab": 50257,
        "use_global_cache": True,
    }
    model = HookedTransformer(cfg, is_masked=False)
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)

    torch.random.manual_seed(42)
    toks_int_values = torch.randint(0, 10_000, (NUM_EXAMPLES, SEQ_LEN)).long()
    toks_int_values_other = torch.randint(0, 10_000, (NUM_EXAMPLES, SEQ_LEN)).long()

#%%

model.reset_hooks()
model.global_cache.clear()

exp = TLACDCExperiment(
    model=model,
    threshold=0.075,
    using_wandb=True,
    zero_ablation=False,
    ds=toks_int_values_batch,
    ref_ds=toks_int_values_other_batch,
    metric=lambda x: 0.0,
    second_metric=None,
    verbose=True,
    second_cache_cpu=False,
    hook_verbose=True,
    first_cache_cpu=False,
    add_sender_hooks=True,
    add_receiver_hooks=False,
    remove_redundant=False,
)

while True: exp.step()

#%%

model.reset_hooks()
exp.setup_model_hooks(
    add_sender_hooks=True,
    add_receiver_hooks=False, # more. More. MORE
)

#%%

attn_out_name = "blocks.{layer}.attn.hook_result"
exp.model.hook_dict["blocks.0.attn.hook_result"].remove_hooks()
exp.model.hook_dict["blocks.1.attn.hook_result"].remove_hooks()
exp.model.hook_dict["blocks.1.hook_resid_post"].remove_hooks()

for layer_idx in range(2):
    for head_idx in range(8):
        exp.add_sender_hook(
            node=exp.corr.graph[attn_out_name.format(layer=layer_idx)][TorchIndex([None, None, head_idx])],
            add_backwards_hook=True,
        )

exp.add_receiver_hook(
    node=exp.corr.graph["blocks.1.hook_resid_post"][TorchIndex([None])],
)

#%%

# torch.random.manual_seed(41)
# wu_device = model.unembed.W_U.device
# model.unembed.W_U = torch.nn.Parameter(torch.randn(model.unembed.W_U.shape).to(wu_device))

if True:
    # remove one pointless connection: necessary since gradients are 0 for KL ...

    removed_heads = [
        ("blocks.0.attn.hook_result", TorchIndex([None, None, 5])),
        # ("blocks.1.attn.hook_result", TorchIndex([None, None, 6])),
    ]

    receivers = exp.corr.edges["blocks.1.hook_resid_post"][TorchIndex([None])]
    for receiver_name in receivers:
        for receiver_index in receivers[receiver_name]:
            print(receiver_name, receiver_index)
            edge = receivers[receiver_name][receiver_index]
            if (receiver_name, receiver_index) in removed_heads:
                edge.present = False
            else:
                edge.present = True

# #%%

# def back(a,b,c):
#     try:
#         print(a.name, b[0].norm().item(), c[0].norm().item(), b[0].requires_grad, c[0].requires_grad)
#     except Exception as e:
#         print(a.name, "error", str(e)[:50])
# tot=0
# for module in model.modules():
#     module.register_backward_hook(back)

#%%

model.global_cache.gradient_cache = OrderedDict()

if True:
    model.zero_grad()
    kls = metric(model(toks_int_values_batch.clone()))
    loss = kls.mean()
    print("Backwards passing...")
    loss.backward(retain_graph=True)
    print("Done.")

#%%

for layer_idx in range(1, -1, -1):
    for head_idx in range(8):
        gradient = model.global_cache.gradient_cache[f"blocks.{layer_idx}.attn.hook_result"][:, :, head_idx]
        linear_walk = model.global_cache.second_cache[f"blocks.{layer_idx}.attn.hook_result"][:, :, head_idx] - model.global_cache.cache[f"blocks.{layer_idx}.attn.hook_result"][:, :, head_idx]
        # ... I guess that we dot over the sequence dimension, too

        val = torch.einsum(
            "bsd,bsd->b",
            gradient.cpu(),
            linear_walk.cpu(),
        )

        print(layer_idx, head_idx, val.abs().mean().item())

#%%

show_pp(
    gradient.norm(dim=-1).detach(),
)
# assert False # things are wrong as there should not be gradients from the future...

#%%

zers = torch.zeros_like(val).detach()
for i in range(len(end_positions)):
    zers[i, end_positions[i]] = 1.0
show_pp(zers)

#%%

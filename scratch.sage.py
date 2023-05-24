#%%

from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import torch
from acdc import HookedTransformer, HookedTransformerConfig # TODO why don't we have to do HookedTransformer.HookedTransformer???
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import TorchIndex

#%%

cfg = {# HookedTransformerConfig(
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

# %%

class M1(torch.nn.Module):
    def forward(self, x):
        return x * 2.0
    
class M2(torch.nn.Module):
    def forward(self, x):
        return x - 1.0

class M3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = M1()
        self.m2 = M2()

    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(x1)
        return x2

#%%

def back(a,b,c):
    try:
        print(a.name, b[0].norm().item(), c[0].norm().item())
    except:
        print(a.name)

tot=0
for module in model.modules():
    module.register_backward_hook(back)

model.hook_dict["blocks.1.hook_resid_pre"].add_hook(
    lambda z, hook: z + torch.randn(z.shape), # should mean no gradients on Layer 0
)

tens = torch.Tensor([[1.0, -1.0, 3.0]]).long()

# tens.requires_grad = True
# -> 2, -2, 6
# -> 1, -3, 5
# norm -> 35

loss = model(tens).norm()**2
# %%

loss.backward()

# %%

torch.random.manual_seed(42)
toks_int_values = torch.randint(0, 10_000, (40, 300)).long()
toks_int_values_other = torch.randint(0, 10_000, (40, 300)).long()

model.reset_hooks()
model.global_cache.clear()
exp = TLACDCExperiment(
    model=model,
    threshold=100_000.0,
    using_wandb=False,
    zero_ablation=False,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=lambda x: 0.0,
    second_metric=None,
    verbose=True,
    second_cache_cpu=True,
    hook_verbose=True,
    first_cache_cpu=True,
    add_sender_hooks=True,
    add_receiver_hooks=False,
    remove_redundant=False,
)

model.reset_hooks()
exp.setup_model_hooks(
    add_sender_hooks=True,
    add_receiver_hooks=True, # more. More. MORE
)

# %%

induction_heads = [
    ("blocks.1.attn.hook_result", TorchIndex([None, None, 5])),
    ("blocks.1.attn.hook_result", TorchIndex([None, None, 6])),
]

receivers = exp.corr.edges["blocks.1.hook_resid_post"][TorchIndex([None])]
for receiver_name in receivers:
    for receiver_index in receivers[receiver_name]:
        print(receiver_name, receiver_index)
        edge = receivers[receiver_name][receiver_index]
        if (receiver_name, receiver_index) in induction_heads:
            edge.present = True
        else:
            edge.present = False

#%%

def back(a,b,c):
    try:
        print(a.name, b[0].norm().item(), c[0].norm().item())
    except:
        print(a.name)
tot=0
for module in model.modules():
    module.register_backward_hook(back) # lol somehow this gets doubles...

# %%

loss = model(toks_int_values).norm()
# %%

loss.backward(retain_graph=True)
# %%

#%%

import transformer_lens
from transformer_lens.ioi_dataset import IOIDataset
from functools import *
from utils import logit_diff
import torch

import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic('load_ext', 'autoreload')
    IPython.get_ipython().run_line_magic('autoreload', '2')
    print("Autoreload enabled")

#%%

def get_model():
    return transformer_lens.HookedTransformer.from_pretrained("gpt2")

model = get_model()

if "is_correct_branch" not in dir(model):
    raise Exception("You're probably importing a wrong version of transformer_lens... pip uninstall transformer_lens?")

#%%

CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
        # (7, 1),
    ],  # unclear exactly what (7,1) does
    "previous token": [
        (2, 2),
        # (2, 9),
        (4, 11),
        # (4, 3),
        # (4, 7),
        # (5, 6),
        # (3, 3),
        # (3, 7),
        # (3, 6),
    ],
}

circuit_heads = [head for head_list in CIRCUIT.values() for head in head_list]
heads_by_layer = [[] for _ in range(12)]
for layer, head in circuit_heads:
    heads_by_layer[layer].append(head)

inv_heads_by_layer = [list(range(12)) for _ in range(12)]
for layer, head in circuit_heads:
    inv_heads_by_layer[layer].remove(head)


#%%

dataset = IOIDataset(N=100, prompt_type="mixed")

#%%

model.set_use_attn_result(True)

#%%

for n, p in model.named_parameters():
    print(n, p.shape, p.requires_grad)

#%%

# how to drop out all the other components?
# TODO check if there's a better way 
# ... for now do resid_pdrop on both MLPs and attention (no attn_pdrop)
# 

#%%

class DropoutManager:
    def __init__(self):
        self.p = 0.1

    def __call__(self, z):
        """Dropout a proportion p of the inputs..."""
        return torch.nn.functional.dropout(z, p=self.p)


D = DropoutManager()

#%%

def zero_out(z, hook, D: DropoutManager, idxs=None):
    if idxs is None:    
        return D(z)

    else:
        zp = z[idxs]
        zp = D(zp)
        z[idxs] = zp
        return z

fwd_hooks = []

model.reset_hooks()
for hook_name in model.hook_dict:

    # all MLPs and all irrelevant heads ...
    if hook_name.endswith("attn.hook_result"): # remember to zero out at position 2 ...
        print("Hooking", hook_name)
        layer = int(hook_name.split(".")[1])
        fwd_hooks.append((hook_name, partial(zero_out, D=D, idxs=inv_heads_by_layer[layer])))

    if hook_name.endswith("mlp_out"):
        print("Hooking", hook_name)
        fwd_hooks.append((hook_name, partial(zero_out, D=D)))

#%%

D.p = 0.0
model.reset_hooks()
logits = model.run_with_hooks(
    dataset.toks.long(),
    fwd_hooks=fwd_hooks,
)
print(logit_diff(logits, dataset))

#%%

parameters = []
for block in model.blocks:
    parameters.append(block.attn.W_Q)
    parameters.append(block.attn.W_K)
    parameters.append(block.attn.W_V) # 12 768 64 shape
    parameters.append(block.attn.W_O) # weirdly 12 64 768 shape

opt = torch.optim.Adam(parameters, lr=1e-4)

def custom_step(opt):
    print("Custom step")
    for i, p in enumerate(parameters): # multiply by binary mask
        p.grad[inv_heads_by_layer[i // 4]] = 0.0
    opt.step()

#%%

def get_loss(model, dataset, d_samples=10, return_var = False): 

    loss_list = []

    for _ in range(d_samples):
        logits = model.run_with_hooks(
            dataset.toks.long(),
            fwd_hooks=fwd_hooks,
        )
        loss_list.append(logit_diff(logits, dataset, item=False))

    ans = sum(loss_list) / d_samples

    if return_var:
        return ans, torch.var(torch.stack(loss_list))

    return ans    

model.reset_hooks()
d_samples = 10

while D.p <= 1.0:
    log = {}
    opt.zero_grad()
    loss, var = get_loss(model, dataset, d_samples=d_samples, return_var=True)
    loss.backward()
    custom_step(opt)

    log["old_loss"] = loss.item()
    log["old_loss_var"] = var.item()
    log["new_loss"] = get_loss(model, dataset).item()
    log["newer_loss"] = get_loss(model, dataset).item()

    if log["newer_loss"] < -3.0:
        D.p += 0.1
        print("-"*10, D.p, "-"*10)

    print(log)

#%%

l = logit_diff(model, dataset, mean=False, item=False)

#%%

def save_model(
    model,
    path,
):
    torch.save(model.state_dict(), path)

save_model(model, "initial_dropout_full.pt")
#%%
from transformer_lens import HookedTransformer
import torch
from functools import partial
torch.set_grad_enabled(False)

from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#%%
m3 = HookedTransformer.from_pretrained("attn-only-demo")

#%%
model = HookedTransformer.from_pretrained("gpt2")
m2 = HookedTransformer.from_pretrained("NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr")
m2.set_use_split_qkv_input(True)

#%%

text = "Hello world!"
cache = {}
m3.reset_hooks()
m3.cache_all(cache)
# m3(text)
loss = m3(text, return_type="loss")

#%%

import os
for key in cache:
    fname = "pts/" + key.replace(".", "_") + "_extra.pt"
    torch.save(cache[key].cpu(), os.path.join(os.getcwd(), fname))

#%%

for key in cache:
    fname_broke = "pts/" + key.replace(".", "_") + ".pt"
    fname = "pts/" + key.replace(".", "_") + "_extra.pt"

    t1 = torch.load(os.path.join(os.getcwd(), fname_broke))
    t2 = torch.load(os.path.join(os.getcwd(), fname))

    try:
        print(key, torch.allclose(t1, t2))
    except:
        print(key, "failed", t1.shape, t2.shape)


#%%

def joker_hook(z, hook):
    model.set_use_attn_result(True)
    return z

#%%
assert not torch.allclose(
    model.W_pos[0], torch.zeros_like(model.W_pos[0])
)
#%%

# rn
import timeit

def measure_forward_pass_time(model):
    def forward_pass():
        tokens = torch.randint(0, 50000, (100, 1024)).cuda()
        def hoker(z, hook):
            print("hooked")
            return z
        model.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.0.attn.hook_result", hoker)],
        )
    return timeit.timeit(forward_pass, number=1)

# usage
torch.cuda.empty_cache()
time = measure_forward_pass_time(m2)
print(f'Time taken to do a forward pass: {time} seconds')
# m# %%

#%%

savez = None

def hooker(z, hook):
    print("hooked!", hook.name, z.shape, z.stride())

    global savez 
    savez = z

    return z

#%%

m2.reset_hooks()
m2.add_hook("blocks.0.attn.hook_q_input", joker_hook)
m2.set_use_attn_result(True)
# m2.add_hook("blocks.0.attn.hook_result", joker_hook)

#%%

tokens = torch.randint(0, 50000, (100, 1024)).cuda()

#%%

a=m2(tokens)

#%%

b = model(tokens)
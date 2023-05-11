from transformer_lens.HookedTransformer import HookedTransformer
from functools import partial
import torch

class PatchingManager:
    def __init__(self):
        self.cache = {}

    def save_hook(self, z, hook):
        self.cache[z.name] = z.clone()
        return z 
    
    def double_direct_output_hook(self, z, hook, hook_name):
        z += self.cache[z.name]
        return z
    
model = HookedTransformer.from_pretrained("redwood_attn_2l", center_writing_weights=False, center_unembed=False, fold_ln=False)

p = PatchingManager()

model.add_hook(
    "blocks.0.hook_attn_out", 
    p.save_hook,
)

# make the effect of attention layer 0 on the logits DOUBLED 
model.add_hook(
    'blocks.1.hook_resid_post',
    partial(p.double_direct_output_hook, hook_name='blocks.0.hook_attn_out'),
)

logits = model(torch.arange(5).unsqueeze(0))

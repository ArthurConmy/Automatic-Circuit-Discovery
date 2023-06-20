#%% [markdown]

from transformer_lens.cautils.notebook import * # use from transformer_lens.cautils.utils import * instead for the same effect without autoreload
import gc 
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

#%%

model = HookedTransformer.from_pretrained("solu-10l", device=DEVICE)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)

#%%

data = get_webtext()
toks = model.to_tokens(
    data[0],
)[0]

#%%

for power in range(0, 20):
    length = 2 ** power
    if length > len(toks):
        break
    print(length)
    logits = model(toks[:length])
    del logits
    gc.collect()
    torch.cuda.empty_cache()

# %%

for batch_size in range(1, 10):
    print(batch_size)
    batchy = [toks for _ in range(batch_size)]
    logits = model(batchy)
    print(batch_size)
    del logits
    gc.collect()
    torch.cuda.empty_cache()

# %%


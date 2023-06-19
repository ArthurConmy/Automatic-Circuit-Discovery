#%% [markdown]
# <h1> Example notebook for the cautils import * statement </h1>

from transformer_lens.cautils.notebook import *

#%%

# load a model
model = HookedTransformer.from_pretrained("solu-1l")

#%%

# get direct logit attribution
MAX_SIZE = 50
tokens = [model.tokenizer.decode([i]) for i in range(MAX_SIZE)]
first_hundred_direct_path = einops.einsum(
    model.W_E[:MAX_SIZE, :], 
    model.W_U[:, :MAX_SIZE],
    "v1 d, d v2 -> v1 v2",
)

# plot it
old_imshow(
    first_hundred_direct_path,
    x = tokens,
    y = tokens,
    labels = {"y": "Embedding token", "x": "Unembedding token"},
    title = "Direct path logit attribution",
)

# %%

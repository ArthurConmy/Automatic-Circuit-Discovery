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

#%%

# do prediction-attention score
# then do neg head projection...
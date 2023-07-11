#%% [markdown] [1]:

"""
Mostly cribbed from transformer_lens/rs/callum/orthogonal_query_investigation_2.ipynb
(but I prefer .py investigations)
"""


from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import (
    project,
    get_effective_embedding_2,
)

from transformer_lens.rs.callum.orthogonal_query_investigation import (
    decompose_attn_scores_full,
    create_fucking_massive_plot_1,
    create_fucking_massive_plot_2,
    token_to_qperp_projection,
    FakeIOIDataset,
)

clear_output()
USE_IOI = False
#%% [markdown] [2]:

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
# model.set_use_split_qkv_normalized_input(True)
clear_output()

#%%

filtered_examples = Path("../arthur/json_data/filtered_for_high_attention.json") # where these come from is documented in git history... it's somewhat selective but BASICALLY the easiest ~40% of the Top 5% crap
with open(filtered_examples, "r") as f:
    filtered_examples = json.load(f)

#%%

TRIMMED_SIZE = 20 # for real
filtered_examples = {
    k: v for i, (k, v) in enumerate(filtered_examples.items()) if i < TRIMMED_SIZE
} 

#%%

filtered_dataset = FakeIOIDataset(
    sentences = list(filtered_examples.values()),
    io_tokens=list(filtered_examples.keys()),
    key_increment=0,
    model=model,
)

# %%

results, ioi_cache = decompose_attn_scores_full(
    ioi_dataset=filtered_dataset,
    batch_size=filtered_dataset.N,
    seed=0,
    nnmh= (10, 7),
    model= model,
    use_effective_embedding = False,
    use_layer0_heads = False,
    subtract_S1_attn_scores = True,
    include_S1_in_unembed_projection = False,
    project_onto_comms_space = "W_EE0A",
    return_cache=True,
)
ioi_cache = ioi_cache.to("cpu")
gc.collect()
torch.cuda.empty_cache()

# %%

# Breaking up these attention scores into comparison to attention scores to surrounding token, I think
# 1. Try the manual computation of attention scores from blocks.10.hook_resid_pre


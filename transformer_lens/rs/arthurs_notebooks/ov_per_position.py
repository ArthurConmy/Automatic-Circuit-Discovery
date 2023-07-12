#%% [markdown] [1]:

"""
Cribbed from key_and_query_projection but let's do OV projection now!!!
Mostly cribbed from transformer_lens/rs/callum/orthogonal_query_investigation_2.ipynb
(but I prefer .py investigations)
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthur_utils import dot_with_query
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
DO_LOGIT_LENS = False
LAYER_IDX, HEAD_IDX = 10, 7

# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project, get_effective_embedding_2
from transformer_lens.rs.arthurs_notebooks.arthur_utils import *
import argparse

#%%

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)

# %%

MAX_SEQ_LEN = 512
BATCH_SIZE = 50
batched_tokens, targets = get_filtered_webtext(model, batch_size=BATCH_SIZE, seed=1717, device="cuda", max_seq_len=MAX_SEQ_LEN)
effective_embeddings = get_effective_embedding_2(model)

# %%

# Find the top 5% of things by importance
# Do this crap
# See change in loss

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = 10, 7
# for NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX in [(10, 0), (10, 7), (9, 9), (11, 10)] + list(itertools.product(range(11, -1, -1), range(12))):

END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
# warnings.warn("Changed to scores for a diff comparison")
# attention_pattern_hook_name = get_act_name("attn_scores", NEGATIVE_LAYER_IDX)
attention_pattern_hook_name = get_act_name("pattern", NEGATIVE_LAYER_IDX)
names_filter1 = (
    lambda name: name == END_STATE_HOOK
    or name==get_act_name("resid_pre", 1)
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.hook_resid_pre"
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_result"
    or name==attention_pattern_hook_name
    or name==get_act_name("v", LAYER_IDX)
)
logits, cache = model.run_with_cache(
    batched_tokens,
    names_filter=names_filter1,
)
gc.collect()
torch.cuda.empty_cache()

# %%

original_end_state = cache[END_STATE_HOOK]

batched_tokens_loss = get_loss_from_end_state(
    model=model,
    end_state=original_end_state,
    targets=targets,
)

#%%

head_output = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]
assert head_output.shape == (BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model)

#%%

if ipython is not None and DO_LOGIT_LENS:
    unembed = einops.einsum(
        head_output, 
        model.W_U,
        "b s d_model, d_model d_vocab -> b s d_vocab",
    )

#%% 

if ipython is not None and DO_LOGIT_LENS:
    the_topk = torch.topk(
        -unembed.cpu(),
        k=10,
        dim=-1,
    ).indices


#%%

mean_head_output = einops.reduce(head_output, "b s d -> d", reduction="mean")

#%%

mean_ablated_end_states = cache[get_act_name("resid_post", model.cfg.n_layers-1)] - head_output + einops.repeat(mean_head_output, "d -> b s d", b=BATCH_SIZE, s=MAX_SEQ_LEN)
mean_ablated_loss = get_loss_from_end_state(
    model=model,
    end_state=mean_ablated_end_states,
    targets=targets,
)

# %%

max_importance_examples = sorted(
    [
        (
            batch_idx,
            seq_idx,
            (mean_ablated_loss-batched_tokens_loss)[batch_idx, seq_idx].item(),
        )
        for batch_idx, seq_idx in itertools.product(
            range(BATCH_SIZE), range(MAX_SEQ_LEN)
        )
    ],
    key=lambda x: x[2],
    reverse=True,
)

# %%

# Get the top 5% of things by importance
all_top_5_percent = max_importance_examples[: len(max_importance_examples)//20]

np.random.seed(799)
# warnings.warn("No shuffle!!!")
np.random.shuffle(all_top_5_percent)
top_5_percent = all_top_5_percent[: BATCH_SIZE]

top5p_batch_indices = [x[0] for x in top_5_percent]
top5p_seq_indices = [x[1] for x in top_5_percent]

#%%

if ipython is not None and DO_LOGIT_LENS:
    top5p_topks = the_topk[top5p_batch_indices, top5p_seq_indices]

#%%

top5p_tokens = batched_tokens[top5p_batch_indices]
top5p_targets = torch.LongTensor([targets[top5p_batch_idx, top5p_seq_idx] for top5p_batch_idx, top5p_seq_idx in zip(top5p_batch_indices, top5p_seq_indices)])

#%%

top5p_losses = batched_tokens_loss[top5p_batch_indices, top5p_seq_indices]

#%%

# 1. Make thing that calculates OV independently for each position
# 2. Check it agrees with normal stuff!
# 3. Get Neel's results : ) 
# 4. Generalize

W_E = model.W_E.clone()
W_U = model.W_U.clone()
W_O = model.W_O.clone()[LAYER_IDX, HEAD_IDX] # d_head d_model

#%%

resid_pre = cache[get_act_name("resid_pre", LAYER_IDX)]
head_pre = model.blocks[LAYER_IDX].ln1(resid_pre)
head_v = cache[get_act_name("v", LAYER_IDX)][:, :, HEAD_IDX, :]
head_pattern = cache[get_act_name("pattern", LAYER_IDX)][:, HEAD_IDX, :, :]
positionwise_z = einops.einsum(
    head_v,
    head_pattern,
    "batch key_pos d_head, \
    batch query_pos key_pos -> \
    batch query_pos key_pos d_head", # contributions from each source position
)

#%%

top5p_positionwise_z = positionwise_z[top5p_batch_indices, top5p_seq_indices]
del positionwise_z
gc.collect()
torch.cuda.empty_cache()

#%%

top5p_positionwise_out = einops.einsum(
    top5p_positionwise_z,
    W_O,
    "batch key_pos d_head, \
    d_head d_model -> \
    batch key_pos d_model",
)

#%%

top_unembeds_per_position = einops.einsum(
    top5p_positionwise_out,
    W_U,
    "batch key_pos d_model, \
    d_model d_vocab -> \
    batch key_pos d_vocab",
)

#%%

total_unembed = einops.reduce(
    top_unembeds_per_position,
    "batch key_pos d_vocab -> batch d_vocab",
    reduction="sum",
)

#%%

average_unembed = einops.einsum(
    mean_head_output,
    W_U,
    "d_model, d_model d_vocab -> d_vocab",
)

#%%

def to_string(toks):
    s = model.to_string(toks)
    s = s.replace("\n", "\\n")
    return s

if DO_LOGIT_LENS:
    for batch_idx in range(len(top_unembeds_per_position)):
        the_logits = -top_unembeds_per_position[batch_idx][1:top5p_seq_indices[batch_idx]+2]
        max_logits = the_logits[:, 1:-1].max().item()
        my_obj = cv.logits.token_log_probs( # I am using this in a very cursed way: 
            top5p_tokens[batch_idx][:top5p_seq_indices[batch_idx]+1],
            the_logits - max_logits,
            to_string = to_string
        )

        print("True completion:"+model.to_string(top5p_tokens[batch_idx][top5p_seq_indices[batch_idx]+1]))
        print("Top negs:")
        print(model.to_str_tokens(torch.topk(-total_unembed[batch_idx]+average_unembed, dim=-1, k=10).indices))
        display(my_obj)

#%%

tot_cont = top5p_positionwise_out.sum(dim=1)[0] # batch key_pos d_model
tot_cont_cache = cache[get_act_name("result", LAYER_IDX)][top5p_batch_indices[0], top5p_seq_indices[0], HEAD_IDX, :]
assert tot_cont.shape == tot_cont_cache.shape, (tot_cont.shape, tot_cont_cache.shape)
assert torch.allclose(
    tot_cont,
    tot_cont_cache,
    atol=1e-4,
    rtol = 1e-4,
), (tot_cont.norm(), tot_cont_cache.norm())

#%%

# See how much loss is recovered when we project these OVs onto the unembedding for given tokens?

projection_directions = einops.rearrange(
    W_U[:, batched_tokens],
    "d_model batch vocab -> batch vocab d_model",
)[top5p_batch_indices]
model_vectors = top5p_positionwise_out # batch key_pos d_model

projected_vectors, perpendicular_vectors = project(
    x = model_vectors,
    dir = projection_directions,
    return_type = "projections",
)

#%%

print(head_pattern[:2, :2, :2])

#%%

relevant_head_pattern = head_pattern[top5p_batch_indices, top5p_seq_indices]

#%%

print(relevant_head_pattern[:2, :2])

#%%

ov_projected_head_out = einops.einsum(
    projected_vectors,
    relevant_head_pattern,
    "batch key_pos d_model, \
    batch key_pos -> \
    batch key_pos d_model",
)

#%%



#%%

for ovout_idx in range(len(ovout)):
    projected, _ = project(ovout[ovout_idx], model.W_U[:, mybatch[batch_idx, seq_idx]])
    if einops.einsum(
        ovout[ovout_idx],
        projected,
        "d_model_out, d_model_out -> ",
    ).item()>0: # only include negative components
        ovout[ovout_idx] = projected

#%%

all_contributions = [sorted([
    (j, contribution) for j, contribution in top_unembeds_per_position[i]
],
key=lambda x: x[1], reverse=True) for i in range(len(top_unembeds_per_position))]

#%%

for i in range(len(top5p_batch_indices)):
    pass

#%%

embeds = W_E.unsqueeze(0)
pre_attention = model.blocks[0].ln1(embeds)

# !!! b_O is not zero. Seems like b_V is, but we'll add it to be safe rather than sorry 
assert model.b_V[0].norm().item() < 1e-4
assert model.b_O[0].norm().item() > 1e-4

vout = einops.einsum(
    pre_attention,
    model.W_V[0],
    "b s d_model, num_heads d_model d_head -> b s num_heads d_head",
) + model.b_V[0]
post_attention = einops.einsum(
    vout,
    model.W_O[0],
    "b s num_heads d_head, num_heads d_head d_model_out -> b s d_model_out",
) + model.b_O[0]

resid_mid = post_attention + embeds
normalized_resid_mid = model.blocks[0].ln2(resid_mid)
mlp_out = model.blocks[0].mlp(normalized_resid_mid)

#%%

"""
The second one is probably what I want : ) 
"""

from matplotlib import pyplot as plt
import numpy as np
from matplotlib_venn import venn3, venn3_circles
plt.figure(figsize=(4,4))
v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
v.get_patch_by_id('100').set_alpha(1.0)
v.get_patch_by_id('100').set_color('white')
v.get_label_by_id('100').set_text('Unknown')
v.get_label_by_id('A').set_text('Set "A"')
c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
c[0].set_lw(1.0)
c[0].set_ls('dotted')
plt.title("Sample Venn diagram")
plt.annotate('Unknown set', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
                ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
plt.show()

# %%

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles

# Define the set sizes and the intersections
A = 100
B = 80
C = 60
AB = 30
AC = 20
BC = 15
ABC = 5

venn_labels = {'100': A, '010': B, '001': C, '110': AB, '101': AC, '011': BC, '111': ABC}

# Create the venn diagram
venn = venn3(subsets=venn_labels, set_labels=('A', 'B', 'C'))

# Create the venn circles to add edge color
venn_circles = venn3_circles(subsets=venn_labels)

# Define the colors for each region (7 regions for 3-set venn diagram)
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black']

# Apply the colors to each region
for region_color, region in zip(colors, venn_labels.keys(), strict=True):
    venn.get_patch_by_id(region).set_color(region_color)
    venn.get_patch_by_id(region).set_alpha(0.4)  # For better visualization

# Optional: if you want to color the text as well, you can use:
for text_color, text in zip(colors, venn.set_labels):
    text.set_color(text_color)

plt.show()

# %%

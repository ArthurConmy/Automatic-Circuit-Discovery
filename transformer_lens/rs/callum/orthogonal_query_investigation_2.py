#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


from transformer_lens.cautils.notebook import *

from transformer_lens.rs.callum.keys_fixed import (
    # attn_scores_as_linear_func_of_keys,
    # attn_scores_as_linear_func_of_queries,
    # get_attn_scores_as_linear_func_of_queries_for_histogram,
    # get_attn_scores_as_linear_func_of_keys_for_histogram,
    # decompose_attn_scores,
    # plot_contribution_to_attn_scores,
    project,
    get_effective_embedding_2,
)

from transformer_lens.rs.callum.orthogonal_query_investigation import (
    decompose_attn_scores_full,
    create_fucking_massive_plot_1,
    create_fucking_massive_plot_2,
    token_to_qperp_projection
)

# effective_embeddings = get_effective_embedding(model) 

# W_U = effective_embeddings["W_U (or W_E, no MLPs)"]
# W_EE = effective_embeddings["W_E (including MLPs)"]
# W_EE_subE = effective_embeddings["W_E (only MLPs)"]

clear_output()


# In[2]:


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

# sudo pkill -9 python


# In[3]:


effective_embeddings = get_effective_embedding_2(model)

W_EE = effective_embeddings['W_E (including MLPs)']
W_EE0 = effective_embeddings['W_E (only MLPs)']
W_E = model.W_E

# Define an easier-to-use dict!
effective_embeddings = {"W_EE": W_EE, "W_EE0": W_EE0, "W_E": W_E}


# # Step 1

# In[4]:


NAME_MOVERS = [(9, 9), (9, 6), (8, 10)]


def hook_value_input(
    v_input: Float[Tensor, "batch seq n_heads d_model"],
    hook: HookPoint,
    head: int,
    ioi_dataset: IOIDataset,
    ioi_cache: ActivationCache,
    patch_type: Literal["mlp_out", "resid_post", "W_EE", "W_EE0", "W_E"] = "mlp_out",
):
    N = len(ioi_dataset)
    assert patch_type in ["mlp_out", "resid_post", "W_EE", "W_EE0", "W_E"]

    if patch_type in ["mlp_out", "resid_post"]:
        patching_values = ioi_cache[patch_type, 0]
        patching_values_IO = patching_values[range(N), ioi_dataset.word_idx["IO"]]
        patching_values_S1 = patching_values[range(N), ioi_dataset.word_idx["S1"]]
    elif patch_type in ["W_EE", "W_EE0", "W_E"]:
        effective_embedding = effective_embeddings[patch_type]
        patching_values_IO = effective_embedding[ioi_dataset.io_tokenIDs]
        patching_values_S1 = effective_embedding[ioi_dataset.s_tokenIDs]

    v_input[range(N), ioi_dataset.word_idx["IO"], head, :] = patching_values_IO
    v_input[range(N), ioi_dataset.word_idx["S1"], head, :] = patching_values_S1
    return v_input



def hook_calc_attn_diff_IO_vs_S(
    attn_scores: Float[Tensor, "batch n_heads seqQ seqK"],
    hook: HookPoint,
    head: int,
    ioi_dataset: IOIDataset,
):
    N = len(ioi_dataset)
    attn_scores_to_IO = attn_scores[range(N), head, ioi_dataset.word_idx["end"], ioi_dataset.word_idx["IO"]]
    attn_scores_to_S = attn_scores[range(N), head, ioi_dataset.word_idx["end"], ioi_dataset.word_idx["S1"]]
    hook.ctx["results"] = (attn_scores_to_IO - attn_scores_to_S).mean().item()



def run_examples(N: int, patch_type: str):
    ioi_dataset, ioi_cache = generate_data_and_caches(N, model, seed=42, prepend_bos=True, only_ioi=True, symmetric=True)

    model.reset_hooks()

    # Define forward hook for recording the attn diff (from END->IO vs END->S1)
    fwd_hooks = [(utils.get_act_name("attn_scores", 10), partial(hook_calc_attn_diff_IO_vs_S, head=7, ioi_dataset=ioi_dataset))]

    # Add the other forward hooks, for patching the value input of all name movers (replacing it with the "purer signal" of just the extended embedding, i.e. MLP output)
    for layer, head in NAME_MOVERS:
        fwd_hooks.append((utils.get_act_name("v_input", layer), partial(hook_value_input, head=head, ioi_dataset=ioi_dataset, ioi_cache=ioi_cache, patch_type=patch_type)))

    # Run the model with these hooks (results are stored in the hook context for 10.7's attention scores)
    model.run_with_hooks(ioi_dataset.toks, return_type = None, fwd_hooks = fwd_hooks)

    # Read results off the hook context
    results = model.hook_dict[utils.get_act_name("attn_scores", 10)].ctx["results"]

    print(f"Patching with {patch_type:<10}: {results:.4f}")


# In[5]:


for patch_type in ["mlp_out", "resid_post", "W_EE", "W_EE0", "W_E"]:
    run_examples(N=20, patch_type=patch_type)


# ### Conclusion
# 
# This means I can legitimately define a linear map using $W_{EE}$, it gets nearly as good performance as the optimal version (which just uses `mlp_out`). **That's great!**

# # Step 2
# 
# First, define a function which gets me the directions I want to project onto, after I've got my q-perp bits. This is the function `token_to_qperp_projection`.
# 
# Then, integrate that into the `decompose_attn_scores_full` function, and get the fucking massive plots.

# In[7]:


batch_size = 40
seed = 0
ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True, prepend_bos=True)


contribution_to_attn_scores = decompose_attn_scores_full(
    batch_size = 40,
    seed = 0,
    nnmh = (10, 7),
    model = model,
    use_effective_embedding = False,
    use_layer0_heads = False,
    subtract_S1_attn_scores = True,
    include_S1_in_unembed_projection = False,
    project_onto_comms_space = "W_EE0A",
)
# ! Why is the performance bad if I use W_EE0 instead of W_EE? It shouldn't be, since the main component comes from MLP0.
# ! Can we improve W_EE by adding the mean positional unembedding vector?


# In[9]:


create_fucking_massive_plot_1(contribution_to_attn_scores)


# In[10]:


create_fucking_massive_plot_2(contribution_to_attn_scores)


# # Step 3
# 
# Now that this all looks good, let's try defining a full matrix using our bilinear form thing.

# In[14]:


NAME_MOVER = (9, 9)

NEG_NAME_MOVER = (10, 7)

# ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True, prepend_bos=True)
# token_ids_IO = ioi_dataset.io_tokenIDs

token_ids = model.to_tokens(NAMES, prepend_bos=False).squeeze()

q_input = token_to_qperp_projection(
    token_ids = token_ids,
    model = model,
    effective_embedding = "W_EE0A",
    name_mover = NAME_MOVER,
)
k_input = W_EE[token_ids]

W_Q = model.W_Q[NEG_NAME_MOVER[0], NEG_NAME_MOVER[1]]
W_K = model.W_K[NEG_NAME_MOVER[0], NEG_NAME_MOVER[1]]

attn_scores = (q_input @ W_Q @ W_K.T @ k_input.T) / (model.cfg.d_head ** 0.5)
attn_scores = attn_scores - attn_scores.mean(dim=-1, keepdims=True)

sm = attn_scores.softmax(dim=-1)

imshow(attn_scores)


# In[13]:


results = t.zeros(10, 12)

for layer in tqdm(range(10)):
    for head in range(12):
        NAME_MOVER = (layer, head)

        NEG_NAME_MOVER = (10, 7)

        # ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True, prepend_bos=True)
        # token_ids_IO = ioi_dataset.io_tokenIDs

        token_ids = model.to_tokens(NAMES, prepend_bos=False).squeeze()

        q_input = token_to_qperp_projection(
            token_ids = token_ids,
            model = model,
            effective_embedding = 'W_EE0A',
            name_mover = NAME_MOVER,
        )
        k_input = W_EE[token_ids]

        W_Q = model.W_Q[NEG_NAME_MOVER[0], NEG_NAME_MOVER[1]]
        W_K = model.W_K[NEG_NAME_MOVER[0], NEG_NAME_MOVER[1]]

        attn_scores = (q_input @ W_Q @ W_K.T @ k_input.T) / (model.cfg.d_head ** 0.5)
        attn_scores = attn_scores - attn_scores.mean(dim=-1, keepdims=True)

        sm = attn_scores.softmax(dim=-1)

        results[layer, head] = sm.diag().mean().item()

imshow(results)


# In[15]:


results2 = t.zeros(10, 12)

token_ids = model.to_tokens(NAMES, prepend_bos=False).squeeze()
unembeddings = model.W_U.T[token_ids]

for layer in tqdm(range(10)):
    for head in range(12):
        NAME_MOVER = (layer, head)

        NEG_NAME_MOVER = (10, 7)

        # ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True, prepend_bos=True)
        # token_ids_IO = ioi_dataset.io_tokenIDs

        token_ids = model.to_tokens(NAMES, prepend_bos=False).squeeze()

        W_V = model.W_V[layer, head]
        W_O = model.W_O[layer, head]

        q_input = project(einops.einsum(
            W_EE[token_ids], W_V, W_O,
            "batch d_model_in, d_model_in d_head, d_head d_model_out -> batch d_model_out"
        ), unembeddings)[0]
        k_input = W_EE[token_ids]

        W_Q = model.W_Q[NEG_NAME_MOVER[0], NEG_NAME_MOVER[1]]
        W_K = model.W_K[NEG_NAME_MOVER[0], NEG_NAME_MOVER[1]]

        attn_scores = (q_input @ W_Q @ W_K.T @ k_input.T) / (model.cfg.d_head ** 0.5)
        attn_scores = attn_scores - attn_scores.mean(dim=-1, keepdims=True)

        sm = attn_scores.softmax(dim=-1)

        results2[layer, head] = sm.diag().mean().item()

imshow(results2)


# In[16]:


imshow(results)


# # Want to find an example of signal suppression rather than copying suppression
# 
# How will I present these results? In the form of a schematic which basically captures the circuit I'm forming to reproduce functionality of the negative heads. 
# 
# There's 2 different tracks: copying suppression and signal suppression. Copying suppression is the query-side signal which is parallel to the unembedding. Signal suppression is the query-side signal which is perpendicular to the unembedding, but in the direction of the `(token -> effective embedding -> 9.9 OV)` map.
# 
# Maybe this answers the question of "how can the negative head be useful, if it only pushes down things which were already predicted?" - because it doesn't only do this, it also pushes things down if there's a signal to push them down. 
# 
# I want to find an example where we have signal suppression but not copying suppression. To do this, I will:
# 
# * For each token that's predicted, find token `X` which is maximally attended to (assuming this isn't BOS, is more than 20%, and the context window is at least 10), and then plot a scatter of `(logprobs attention to X)` (or maybe log-odds) vs `(component of unembedding of X in the residual stream)`
# * I expect these to be positively correlated (that'd be a nice scatter!), but not perfectly correlated. In particular, there will be some examples where the logprobs are high but the unembedding is close to zero. This will be signal suppression!
# 
# # All's fair in love and love
# 
# Choose either this, the `Mr John and Mr Smith spoke to Mr` examples, or this notebook's examples: https://colab.research.google.com/drive/12UtMwPh124dYTCos1JGgMM9c7OheumJl?usp=sharing
# 
# One of them should allow you to analyse the copy-suppression behaviour, or signal-suppression behaviour, or both.
# 
# Can do some things like injecting the unembedding for `" John"` into the query-side input for the negative name mover, and showing that this causes suppression. On the x-axis is the component of unembedding injected in, on the y-axis is the change in logits for `" John"` (as direct output from head 10.7), and the attention probs. We expect to see the probs head to one, and the logits decrease & hit some lower bound, as the unembedding component is increased (this partly or fully offsets the direct effect that injecting these logits has). (In fact, maybe we just inject it into resid pre at layer 10, and see how much the heads change - I'm guessing 10.7 will easily change most!). And we expect to see exactly the same behaviour when you inject in the signal suppression direction, rather than the copy suppression direction.
# 
# # Other models - preliminary stuff?
# 
# Would be nice to see if these kinda hold up in other models.-

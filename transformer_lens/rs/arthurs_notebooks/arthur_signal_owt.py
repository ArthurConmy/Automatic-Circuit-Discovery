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
    token_to_qperp_projection
)

clear_output()

USE_IOI = True

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

#%%

head_9_9_max_activating = {
    # " James": ' Later I concluded that, because the Russians kept secret their developments in military weapons, they thought it improper to ask us about ours.\n\nJames F. Byrnes, Speaking Frankly (New York: Harper and Brothers, 1947) p. 263.\n\nSecretary of State', # weird cos James no space
    " du": """Jamaica Inn by Daphne du Maurier

Broken Rule(s): non-protagonist points of view and numerous redundancies.

Most writers compulsively check and double check for echoes and redundancies and remove them. Contrast that with Daphne""",
    " Sweeney": "Mandatory safety training was part of Sweeney's centerpiece bill, passed by both houses of the legislature last year but conditionally vetoed by the governor. The bill would have changed the way the state issues firearms licenses, made background checks instant and included private sales in the law. It also would have required proof of safety training prior to the issuance of a gun license. Training was among the elements altered by the governor's veto. After the conditional veto,",
    " Mark": """it will end up in a bright good place.

The Adventures of Huckleberry Finn by Mark Twain

Broken Rule: multiple regional dialects that are challenging to understand.

No discussion of writerly rule-breaking would be complete without mentioning""",
    " Lal": "Tyrone Troutman's girlfriend, Donna Lalor, told investigators that Joseph Troutman became enraged after his son grabbed a radio from the table and smashed it.\n\nJoseph Troutman then grabbed a butcher knife, came at his son and stabbed him,",
    " 1906": """ a Neapolitan street game, the "morra".[3][5] (In this game, two persons wave their hands simultaneously, while a crowd of surrounding gamblers guess, in chorus, at the total number of fingers exposed by the principal players.)[6] This activity was prohibited by the local government, and some people started making the players pay for being "protected" against the passing police.[3][7][8]

Camorristi in Naples, 1906 in Naples,""",
    " Mush": """ Omar Mushaweh, a Turkey-based leader of the Syrian Muslim Brotherhood, told IRIN in an online interview.

But, like other opposition sympathisers interviewed,""",
}

if USE_IOI:
    N = 200
    warnings.warn("Auto IOI")
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=True,
        seed=35795,
    )
    update_word_lists = {" " + sent.split()[-1]: sent for sent in ioi_dataset.sentences}
    # lol will reduce in size cos of duplicate IOs
    old_update_word_lists=deepcopy(update_word_lists)
    for k, v in old_update_word_lists.items():
        assert v.endswith(k), (k, v)
        update_word_lists[k] = v[:-len(k)]
    assert len(update_word_lists) == len(set(update_word_lists.keys())), "Non-uniqueness!"
    head_9_9_max_activating = update_word_lists

for k, v in list(head_9_9_max_activating.items()):
    assert v.count(k)==1, k

tokens = model.to_tokens(list(head_9_9_max_activating.values()), truncate=False)
key_positions = []
query_positions = []
for i in range(len(tokens)):
    if tokens[i, -1].item()!=model.tokenizer.pad_token_id:
        query_positions.append(tokens.shape[-1]-1)
    else:
        for j in range(len(tokens[i])-1, -1, -1):
            if tokens[i, j].item()!=model.tokenizer.pad_token_id:
                query_positions.append(j)
                break

    key_token = model.to_tokens([list(head_9_9_max_activating.keys())[i]], prepend_bos=False).item()
    assert tokens[i].tolist().count(key_token)==1
    key_positions.append(tokens[i].tolist().index(key_token))

assert len(key_positions)==len(query_positions)==len(tokens), ("Missing things probably", len(key_positions), len(query_positions), len(tokens))

#%% [markdown] [3]:

effective_embeddings = get_effective_embedding_2(model)
W_EE = effective_embeddings['W_E (including MLPs)']
W_EE0 = effective_embeddings['W_E (only MLPs)']
W_E = model.W_E
# Define an easier-to-use dict!
effective_embeddings = {"W_EE": W_EE, "W_EE0": W_EE0, "W_E": W_E}

#%% [markdown] [4]:

def owt_decompose_attn_scores_full(
    tokens: Int[torch.Tensor, "batch_size seq_len"],
    key_positions: List[int],
    query_positions: List[int],
    key_tokens,
    nnmh: Tuple[int, int],
    model: HookedTransformer,
    cache: Optional[ActivationCache] = None,
    use_effective_embedding: bool = False,
    use_layer0_heads: bool = False,
    project_onto_comms_space: Optional[Literal["W_EE", "W_EE0", "W_E", "W_EE0A"]] = None,
):
    """
    Built from `transformer_lens/rs/callum/orthogonal_query_investigation.py`
    
    and made to be for general prompts
    """

    t.cuda.empty_cache()

    if cache is None:
        _, cache = model.run_with_cache(
            tokens,
        )
    batch_size = len(tokens)
    assert len(tokens)==len(query_positions)==len(key_positions)==len(key_tokens), (len(tokens), len(query_positions), len(key_positions))

    if project_onto_comms_space is not None: 
        assert project_onto_comms_space in ["W_EE", "W_EE0", "W_E", "W_EE0A"]
        W_EE_dict = get_effective_embedding_2(model)
        W_EE = W_EE_dict['W_E (including MLPs)']
        W_EE0 = W_EE_dict['W_E (only MLPs)']
        W_E = model.W_E
        W_EE_dict = {"W_EE": W_EE, "W_EE0": W_EE0, "W_EE0A": W_EE - W_E, "W_E": W_E}
        effective_embeddings_matrix = W_EE_dict[project_onto_comms_space]
        comms_space_projections = {
            f"{layer}.{head}" : token_to_qperp_projection(
                key_tokens,
                model = model,
                effective_embedding = effective_embeddings_matrix,
                name_mover = (layer, head)
            )
            for layer in range(nnmh[0]) for head in range(model.cfg.n_heads)
        }

    # TODO decide if we can/should include this...
    # IO_seq_pos_indices = ioi_dataset.word_idx["IO"]
    # end_seq_pos_indices = ioi_dataset.word_idx["end"]
    ln_scale_key = cache["scale", nnmh[0], "ln1"][range(batch_size), key_positions, nnmh[1]] # TODO shape???
    ln_scale_query = cache["scale", nnmh[0], "ln1"][range(batch_size), query_positions, nnmh[1]]

    # * Get the MLP0 output (note that we need to be careful here if we're subtracting the S1 baseline, because we actually need the 2 different MLP0s)
    if use_effective_embedding:
        W_EE_dict = get_effective_embedding_2(model)
        W_EE = (W_EE_dict["W_E (including MLPs)"] - W_EE_dict["W_E (no MLPs)"]) if use_layer0_heads else W_EE_dict["W_E (only MLPs)"]
        MLP0_output = W_EE[key_tokens]
    else:
        if use_layer0_heads:
            MLP0_output = cache["mlp_out", 0][range(batch_size), key_positions] + cache["attn_out", 0][range(batch_size), key_positions]
        else:
            MLP0_output = cache["mlp_out", 0][range(batch_size), key_positions]

    # * Get the unembeddings
    unembeddings = model.W_U.T[key_tokens]

    t.cuda.empty_cache()

    num_key_decomps = 2
    num_query_decomps = 3 if project_onto_comms_space else 2
    contribution_to_attn_scores = t.zeros(
        num_key_decomps * num_query_decomps, # this is for the key & query options: (∥ / ⟂) MLP0 on key side, same for unembed on IO side plus the comms space
        3 + (nnmh[0] * (1 + model.cfg.n_heads)), # this is for the query-side
        3 + (nnmh[0] * (1 + model.cfg.n_heads)), # this is for the key-side
    )

    keyside_components = []
    queryside_components = []

    # TODO - calculate product directly after filling these in, in case it's too large? Or maybe it's fine cause they are on CPU.
    keys_decomposed = t.zeros(num_key_decomps, 3 + (nnmh[0] * (1 + model.cfg.n_heads)), batch_size, model.cfg.d_head)
    queries_decomposed = t.zeros(num_query_decomps, 3 + (nnmh[0] * (1 + model.cfg.n_heads)), batch_size, model.cfg.d_head)

    def get_component(component_name, layer=None, keyside=False):
        '''
        Gets component (key or query side).

        If we need to subtract the baseline, it returns both the component for IO and the component for S1 (so we can project then subtract scores from each other).
        '''
        full_component = cache[component_name, layer]
        if keyside:
            component_IO = full_component[range(batch_size), key_positions] / (ln_scale_key.unsqueeze(1) if (component_name == "result") else ln_scale_key)
            return component_IO
        else:
            component_END = full_component[range(batch_size), query_positions] / (ln_scale_query.unsqueeze(1) if (component_name == "result") else ln_scale_query)
            return component_END

    b_K = model.b_K[nnmh[0], nnmh[1]]
    b_K = einops.repeat(b_K, "d_head -> batch d_head", batch=batch_size)
    b_Q = einops.repeat(model.b_Q[nnmh[0], nnmh[1]], "d_head -> batch d_head", batch=batch_size)

    # First, get the biases and direct terms
    keyside_components.extend([
        ("b_K", b_K),
        ("embed", get_component("embed", keyside=True)),
        ("pos_embed", get_component("pos_embed", keyside=True)),
    ])
    queryside_components.extend([
        ("b_Q", b_Q),
        ("embed", get_component("embed", keyside=False)),
        ("pos_embed", get_component("pos_embed", keyside=False)),
    ])

    # Next, get all the MLP terms
    for layer in range(nnmh[0]):
        keyside_components.append((f"mlp_out_{layer}", get_component("mlp_out", layer=layer, keyside=True)))
        queryside_components.append((f"mlp_out_{layer}", get_component("mlp_out", layer=layer, keyside=False)))

    # Lastly, all the heads
    for layer in range(nnmh[0]):
        keyside_heads = get_component("result", layer=layer, keyside=True)
        queryside_heads = get_component("result", layer=layer, keyside=False)
        for head in range(model.cfg.n_heads):
            keyside_components.append((f"{layer}.{head}", keyside_heads[:, head, :]))
            queryside_components.append((f"{layer}.{head}", queryside_heads[:, head, :]))

    # Now, we do the projection thing...
    # ... for keys ....
    keys_decomposed[1, 0] = keyside_components[0][1]
    for i, (keyside_name, keyside_component) in enumerate(keyside_components[1:], 1):
        projections = torch.stack(project(keyside_component, MLP0_output))

        keys_decomposed[:, i] = einops.einsum(projections.cpu(), model.W_K[nnmh[0], nnmh[1]].cpu(), "projection batch d_model, d_model d_head -> projection batch d_head")

    # ... and for queries ...
    queries_decomposed[1, 0] = queryside_components[0][1]
    for i, (queryside_name, queryside_component) in enumerate(queryside_components[1:], 1):
        queryside_par, queryside_perp = project(queryside_component, unembeddings)
        queries_decomposed[0, i] = einops.einsum(queryside_par.cpu(), model.W_Q[nnmh[0], nnmh[1]].cpu(), "batch d_model, d_model d_head -> batch d_head")
        if project_onto_comms_space and ("." in queryside_name):
            # * Project the perpendicular part of the component into the direction of the communication channel
            queryside_comms, queryside_comms_perp = project(queryside_perp, comms_space_projections[queryside_name])
            queries_decomposed[1, i] = einops.einsum(queryside_comms.cpu(), model.W_Q[nnmh[0], nnmh[1]].cpu(), "batch d_model, d_model d_head -> batch d_head")
            queries_decomposed[2, i] = einops.einsum(queryside_comms_perp.cpu(), model.W_Q[nnmh[0], nnmh[1]].cpu(), "batch d_model, d_model d_head -> batch d_head")
        else:
            queries_decomposed[-1, i] = einops.einsum(queryside_perp.cpu(), model.W_Q[nnmh[0], nnmh[1]].cpu(), "batch d_model, d_model d_head -> batch d_head")
    

    # Finally, we do the outer product thing
    for (key_idx, keyside_component) in enumerate(keys_decomposed.unbind(dim=1)):
        for (query_idx, queryside_component) in enumerate(queries_decomposed.unbind(dim=1)):
            contribution_to_attn_scores[:, query_idx, key_idx] = einops.einsum(
                queryside_component,
                keyside_component,
                "q_projection batch d_head, k_projection batch d_head -> q_projection k_projection batch"
            ).mean(-1).flatten() / (model.cfg.d_head ** 0.5)


    return contribution_to_attn_scores

#%%

res = []
for key_bump in [2, 0]:
    new_key_positions = torch.LongTensor(key_positions) + key_bump
    res.append(owt_decompose_attn_scores_full(
        tokens=tokens,
        key_positions=new_key_positions.tolist(),
        key_tokens = tokens[torch.arange(len(tokens)), key_positions],
        query_positions=query_positions,
        nnmh=(10, 7),
        model=model,
        cache=None, # TODO make fast
        use_effective_embedding=False,
        use_layer0_heads = True,
        project_onto_comms_space="W_EE0A",
    ))
create_fucking_massive_plot_1(res[1]-(sum(res)-res[1])/(len(res)-1))

# %%

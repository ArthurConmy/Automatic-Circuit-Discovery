from transformer_lens.cautils.utils import *
from transformer_lens.rs.callum.keys_fixed import project, get_effective_embedding_2


def token_to_qperp_projection(
    token_ids: Int[Tensor, "batch"],
    model: HookedTransformer,
    effective_embedding: Union[Float[Tensor, "d_vocab d_model"], Literal["W_EE", "W_EE0", "W_E", "W_EE0A"]],
    name_mover: Tuple[int, int] = (9, 9),
    return_both: bool = False,
):
    '''
    Given token_ids, performs the following map:

        (1) Returns the effective embedding of that token (according to the effective_embedding_type argument)
        (2) Maps it through the name mover's OV circuit (9.9 by default)
        (3) Takes the component perpendicular to the unembedding
    '''
    # Get our effective embeddings dictionary
    if isinstance(effective_embedding, str):
        W_EE_dict = get_effective_embedding_2(model)
        W_EE = W_EE_dict['W_E (including MLPs)']
        W_EE0 = W_EE_dict['W_E (only MLPs)']
        W_E = model.W_E
        W_EE_dict = {"W_EE": W_EE, "W_EE0": W_EE0, "W_EE0A": W_EE - W_E, "W_E": W_E}
        effective_embeddings_matrix = W_EE_dict[effective_embedding]
    else:
        effective_embeddings_matrix = effective_embedding

    # * (1)
    embeddings = effective_embeddings_matrix[token_ids]

    # * (2)
    layer, head = name_mover
    W_V = model.W_V[layer, head]
    W_O = model.W_O[layer, head]
    name_mover_output = einops.einsum(
        embeddings, W_V, W_O,
        "batch d_model_in, d_model_in d_head, d_head d_model_out -> batch d_model_out",
    )

    # * (3) 
    unembeddings = model.W_U.T[token_ids]
    output_par, output_perp = project(name_mover_output, unembeddings)

    if return_both:
        return output_par, output_perp
    else:
        return output_perp




def decompose_attn_scores_full(
    batch_size: int,
    seed: int,
    nnmh: Tuple[int, int],
    model: HookedTransformer,
    use_effective_embedding: bool = False,
    use_layer0_heads: bool = False,
    subtract_S1_attn_scores: bool = False,
    include_S1_in_unembed_projection: bool = False,
    project_onto_comms_space: Optional[Literal["W_EE", "W_EE0", "W_E", "W_EE0A"]] = None,
    ioi_dataset = None, # pass this if you want to use a custom FakeIOIDataset
    return_cache: bool = False,
    
):
    t.cuda.empty_cache()
    # if (ioi_dataset is None) or (ioi_cache is None):
    #     ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True, prepend_bos=True)
    # else:
    #    c assert isinstance(ioi_dataset, IOIDataset) and isinstance(ioi_cache, ActivationCache)
    
    if ioi_dataset is None:
        ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True, prepend_bos=True)
    else:
        _, ioi_cache = model.run_with_cache(ioi_dataset.toks)

    seq_len = ioi_dataset.toks.shape[1]

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
                ioi_dataset.io_tokenIDs,
                model = model,
                effective_embedding = effective_embeddings_matrix,
                name_mover = (layer, head)
            )
            for layer in range(nnmh[0]) for head in range(model.cfg.n_heads)
        }

    S1_seq_pos_indices = ioi_dataset.word_idx["S1"]
    IO_seq_pos_indices = ioi_dataset.word_idx["IO"]
    end_seq_pos_indices = ioi_dataset.word_idx["end"]

    ln_scale_S1 = ioi_cache["scale", nnmh[0], "ln1"][range(batch_size), S1_seq_pos_indices, nnmh[1]]
    ln_scale_IO = ioi_cache["scale", nnmh[0], "ln1"][range(batch_size), IO_seq_pos_indices, nnmh[1]]
    ln_scale_end = ioi_cache["scale", nnmh[0], "ln1"][range(batch_size), end_seq_pos_indices, nnmh[1]]

    # * Get the MLP0 output (note that we need to be careful here if we're subtracting the S1 baseline, because we actually need the 2 different MLP0s)
    if use_effective_embedding:
        W_EE_dict = get_effective_embedding_2(model)
        W_EE = (W_EE_dict["W_E (including MLPs)"] - W_EE_dict["W_E (no MLPs)"]) if use_layer0_heads else W_EE_dict["W_E (only MLPs)"]
        MLP0_output = W_EE[ioi_dataset.io_tokenIDs]
        MLP0_output_S1 = W_EE[ioi_dataset.s_tokenIDs]
    else:
        if use_layer0_heads:
            MLP0_output = ioi_cache["mlp_out", 0][range(batch_size), IO_seq_pos_indices] + ioi_cache["attn_out", 0][range(batch_size), IO_seq_pos_indices]
            MLP0_output_S1 = ioi_cache["mlp_out", 0][range(batch_size), S1_seq_pos_indices] + ioi_cache["attn_out", 0][range(batch_size), S1_seq_pos_indices]
        else:
            MLP0_output = ioi_cache["mlp_out", 0][range(batch_size), IO_seq_pos_indices]
            MLP0_output_S1 = ioi_cache["mlp_out", 0][range(batch_size), S1_seq_pos_indices]

    # * Get the unembeddings
    unembeddings = model.W_U.T[ioi_dataset.io_tokenIDs]
    unembeddings_S1 = model.W_U.T[ioi_dataset.s_tokenIDs]

    t.cuda.empty_cache()

    num_key_decomps = 2
    num_query_decomps = 3 if project_onto_comms_space else 2
    contribution_to_attn_scores = t.zeros(
        num_key_decomps * num_query_decomps, # this is for the key & query options: (∥ / ⟂) MLP0 on key side, same for unembed on IO side plus the comms space
        4 + (nnmh[0] * (1 + model.cfg.n_heads)), # this is for the query-side
        4 + (nnmh[0] * (1 + model.cfg.n_heads)), # this is for the key-side
    )

    keyside_components = []
    queryside_components = []

    # Number of components = 4 (embed, pos_embed, b_Q or b_K, other accumulated biases) plus one for every head & MLP in each layer
    keys_decomposed = t.zeros(num_key_decomps, 4 + (nnmh[0] * (1 + model.cfg.n_heads)), batch_size, model.cfg.d_head)
    queries_decomposed = t.zeros(num_query_decomps, 4 + (nnmh[0] * (1 + model.cfg.n_heads)), batch_size, model.cfg.d_head)

    def get_component(component, layer=None, keyside=False):
        '''
        Gets component (key or query side).

        If we need to subtract the baseline, it returns both the component for IO and the component for S1 (so we can project then subtract scores from each other).
        '''
        full_component = ioi_cache[component, layer] if isinstance(component, str) else component
        has_head_dim = (full_component.ndim == 4)
        if keyside:
            component_IO = full_component[range(batch_size), IO_seq_pos_indices] / (ln_scale_IO.unsqueeze(1) if has_head_dim else ln_scale_IO)
            component_S1 = full_component[range(batch_size), S1_seq_pos_indices] / (ln_scale_S1.unsqueeze(1) if has_head_dim else ln_scale_S1)
            return (component_IO, component_S1) if subtract_S1_attn_scores else component_IO
        else:
            component_END = full_component[range(batch_size), end_seq_pos_indices] / (ln_scale_end.unsqueeze(1) if has_head_dim else ln_scale_end)
            return component_END

    # Get biases: keyside, queryside, and also the biases from attention layers and MLP layers
    # Note that b_K and b_Q are already d_head (they go into the perpendicular buckets), but b_O is in the residual stream (so it does get decomposed)
    b_K = einops.repeat(model.b_K[nnmh[0], nnmh[1]] * (0.0 if subtract_S1_attn_scores else 1.0), "d_head -> batch d_head", batch=batch_size)
    b_Q = einops.repeat(model.b_Q[nnmh[0], nnmh[1]], "d_head -> batch d_head", batch=batch_size)
    b_O = einops.repeat(model.b_O[:nnmh[0]].sum(0) + model.b_out[:nnmh[0]].sum(0), "d_model -> batch seq_len d_model", batch=batch_size, seq_len=seq_len)

    # First, get the biases and direct terms
    keyside_components.extend([
        ("b_K", b_K),
        ("b_O", get_component(b_O, keyside=True)),
        ("embed", get_component("embed", keyside=True)),
        ("pos_embed", get_component("pos_embed", keyside=True)),
    ])
    queryside_components.extend([
        ("b_Q", b_Q),
        ("b_O", get_component(b_O, keyside=False)),
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
            if subtract_S1_attn_scores:
                keyside_components.append((f"{layer}.{head}", (keyside_heads[0][:, head, :], keyside_heads[1][:, head, :])))
            else:
                keyside_components.append((f"{layer}.{head}", keyside_heads[:, head, :]))
            queryside_components.append((f"{layer}.{head}", queryside_heads[:, head, :]))

    # Now, we do the projection thing...
    
    # * ... for keys ....
    # adding attention head bias & all residual stream biases
    keys_decomposed[-1, 0] = keyside_components[0][1]
    # then iterating through all the others and adding them too (possibly splitting them into projections)
    for i, (keyside_name, keyside_component) in tqdm(list(enumerate(keyside_components[1:], 1))):
        if subtract_S1_attn_scores:
            keyside_component_IO, keyside_component_S1 = keyside_component
            projections = project(keyside_component_IO, MLP0_output), project(keyside_component_S1, MLP0_output_S1)
            projections = t.stack([projections[0][0] - projections[1][0], projections[0][1] - projections[1][1]])
        else:
            projections = project(keyside_component, MLP0_output)
        keys_decomposed[:, i] = einops.einsum(projections.cpu(), model.W_K[nnmh[0], nnmh[1]].cpu(), "projection batch d_model, d_model d_head -> projection batch d_head")
    
    # * ... and for queries ...
    # adding attention head bias & all residual stream biases
    queries_decomposed[-1, 0] = queryside_components[0][1]
    # then iterating through all the others and adding them too (possibly splitting them into projections)
    for i, (queryside_name, queryside_component) in enumerate(queryside_components[1:], 1):
        queryside_par, queryside_perp = project(queryside_component, unembeddings if not(include_S1_in_unembed_projection) else [unembeddings, unembeddings_S1])
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


    if return_cache: 
        return contribution_to_attn_scores, ioi_cache

    return contribution_to_attn_scores





def create_fucking_massive_plot_1(contribution_to_attn_scores):

    full_labels = ["attn_bias", "other_biases", "W<sub>E</sub>", "W<sub>pos</sub>"]
    full_labels += [f"MLP<sub>{L}</sub>" for L in range(10)]
    full_labels += [f"{L}.{H}" for L in range(10) for H in range(12)]

    if contribution_to_attn_scores.shape[0] == 4:
        projection_labels = [
            "q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>",
            "q ∥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>"
        ]
    else:
        projection_labels = [
            "q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>",
            "q ∥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO] & ∥ comms, k ∥ MLP<sub>0</sub>",
            "q ⊥ W<sub>U</sub>[IO] & ∥ comms, k ⊥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO] & ⟂ comms, k ∥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO] & ⟂ comms, k ⊥ MLP<sub>0</sub>"
        ]

    zmax = contribution_to_attn_scores.abs().max().item()

    imshow(
        contribution_to_attn_scores,
        animation_frame = 0,
        animation_labels = projection_labels,
        x = full_labels,
        y = full_labels,
        height = 1600,
        zmin = -zmax,
        zmax = zmax,
    )



def create_fucking_massive_plot_2(contribution_to_attn_scores):

    full_labels = ["attn_bias", "other_biases", "W<sub>E</sub>", "W<sub>pos</sub>"]
    full_labels += [f"MLP<sub>{L}</sub>" for L in range(10)]
    full_labels += [f"{L}.{H}" for L in range(10) for H in range(12)]

    if contribution_to_attn_scores.shape[0] == 4:
        projection_labels = [
            "q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>",
            "q ∥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>"
        ]
    else:
        projection_labels = [
            "q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>",
            "q ∥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO] & ∥ comms, k ∥ MLP<sub>0</sub>",
            "q ⊥ W<sub>U</sub>[IO] & ∥ comms, k ⊥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO] & ⟂ comms, k ∥ MLP<sub>0</sub>", 
            "q ⊥ W<sub>U</sub>[IO] & ⟂ comms, k ⊥ MLP<sub>0</sub>"
        ]


    contribution_to_attn_scores_normalized = contribution_to_attn_scores / contribution_to_attn_scores.sum()

    all_labels = [f"Q = {q}, K = {k}, {proj}" for proj in projection_labels for q in full_labels for k in full_labels]
    all_values = contribution_to_attn_scores_normalized.flatten()

    topk = all_values.abs().topk(40)

    top_labels = [all_labels[i] for i in topk.indices]
    top_values = [all_values[i] for i in topk.indices]

    unembedding_channel = [
        f'Q = {q}, K = MLP<sub>{L}</sub>, q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>'
        for q in ["9.6", "9.9", "8.10", "7.9"]
        for L in [0] # range(10)
    ]
    comms_channel = [
        f'Q = {q}, K = MLP<sub>{L}</sub>, q ⊥ W<sub>U</sub>[IO] & ∥ comms, k ∥ MLP<sub>0</sub>'
        for q in ["9.6", "9.9", "8.10", "7.9"]
        for L in [0] # range(10)
    ]
    color_indices = [int(l in comms_channel) * 2 + int(l in unembedding_channel) for l in top_labels[::-1]]


    fig = px.bar(
        y = top_labels[::-1],
        x = top_values[::-1],
        color = [["#8f4424", "#1560d4", "#1eb02d"][i] for i in color_indices],
        orientation = "h",
        color_discrete_map = "identity",
        height = 100 + 25 * len(top_labels),
        labels = {"x": "Contribution to attention scores", "y": ""},
        title = "Largest single contributions to attention scores",
    )
    fig.update_yaxes(
        categoryorder = "array",
        categoryarray = top_labels[::-1],
    )
    fig.update_traces(showlegend=True)
    fig.update_layout(legend_traceorder="reversed")

    newnames = ["Not expected", "Main comms channel", "Main unembeddings channel"]
    fig.for_each_trace(lambda t: t.update(name = newnames.pop(0)))

    fig.show()

class FakeIOIDataset:
    """Used for normal webtext things where we imitate the IOI dataset methods
    
    so that we can use the great above `decompose_full_attn_scores` function"""

    def __init__(
        self,
        sentences,
        io_tokens,
        key_increment,
        model,
    ):
        self.N=len(sentences)
        sentences_trimmed = []
        for k, v in list(zip(io_tokens, sentences, strict=True)):
            assert v.endswith(k), (k, v)
            sentences_trimmed.append(v[:-len(k)])

        self.toks = model.to_tokens(sentences_trimmed)
        self.word_idx={}
        self.word_idx["IO"] = []
        self.word_idx["end"] = []
        self.word_idx["S1"] = []

        for i in range(len(self.toks)):
            if self.toks[i, -1].item()!=model.tokenizer.pad_token_id:
                self.word_idx["end"].append(self.toks.shape[-1]-1)
            else:
                for j in range(len(self.toks[i])-1, -1, -1):
                    if self.toks[i, j].item()!=model.tokenizer.pad_token_id:
                        self.word_idx["end"].append(j)
                        token_counter=self.toks[i].tolist().count(self.toks[i, j].item())
                        break

            key_token = model.to_tokens([io_tokens[i]], prepend_bos=False).item()
            assert self.toks[i].tolist().count(key_token)==1, (io_tokens[i], sentences_trimmed[i])
            self.word_idx["IO"].append(self.toks[i].tolist().index(key_token))

        self.io_tokenIDs = self.toks[torch.arange(self.N), self.word_idx["IO"]]

        self.word_idx["S1"] = (torch.LongTensor(self.word_idx["IO"]) + key_increment)
        assert self.N==len(self.word_idx["IO"])==len(self.word_idx["S1"]), ("Missing things probably", len(self.word_idx["IO"]), len(self.word_idx["S1"]), self.N)

        assert 0 <= self.word_idx["S1"].min().item()
        assert self.toks.shape[1] > self.word_idx["S1"].max().item()
        self.word_idx["S1"] = self.word_idx["S1"].tolist()
        self.s_tokenIDs = self.toks[torch.arange(self.N), self.word_idx["S1"]]

    def __len__(self):
        return self.N
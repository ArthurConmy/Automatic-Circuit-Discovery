from torch import native_dropout
from transformer_lens.cautils.utils import *
# from transformer_lens.rs.callum.generate_bag_of_words_quad_plot import get_effective_embedding


def get_effective_embedding_2(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:

    # TODO - make this consistent (i.e. change the func in `generate_bag_of_words_quad_plot` to also return W_U and W_E separately)

    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    # t.testing.assert_close(W_E[:10, :10], W_U[:10, :10].T)  NOT TRUE, because of the center unembed part!

    resid_pre = W_E.unsqueeze(0)
    pre_attention = model.blocks[0].ln1(resid_pre)
    attn_out = einops.einsum(
        pre_attention, 
        model.W_V[0],
        model.W_O[0],
        "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
    )
    resid_mid = attn_out + resid_pre
    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()

    t.cuda.empty_cache()

    return {
        "W_E (no MLPs)": W_E,
        "W_U": W_U.T,
        # "W_E (raw, no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE
    }



def attn_scores_as_linear_func_of_queries(
    batch_idx: Optional[Union[int, List[int], Int[Tensor, "batch"]]],
    head: Tuple[int, int],
    model: HookedTransformer,
    ioi_cache: ActivationCache,
    ioi_dataset: IOIDataset,
    subtract_S1_attn_scores: bool = False,
) -> Float[Tensor, "d_model"]:
    '''
    If you hold keys fixed, then attention scores are a linear function of the queries.
    I want to fix the keys of head 10.7, and get a linear function mapping queries -> attention scores.
    I can then see if (for example) the unembedding vector for the IO token has a really big image in this linear fn.

    Here, if `subtract_S1_attn_scores` is True, this means we should change the linear map, from key_IO_linear_map to
    (key_IO_linear_map - key_S1_linear_map). Same for the bias term.
    '''
    layer, head_idx = head
    if isinstance(batch_idx, int):
        batch_idx = [batch_idx]
    if batch_idx is None:
        batch_idx = range(len(ioi_cache["q", 0]))

    keys_all = ioi_cache["k", layer][:, :, head_idx] # shape (all_batch, seq_K, d_head)
    if subtract_S1_attn_scores:
        keys = keys_all[batch_idx, ioi_dataset.word_idx["IO"][batch_idx]] - keys_all[batch_idx, ioi_dataset.word_idx["S1"][batch_idx]]
    else:
        keys = keys_all[batch_idx, ioi_dataset.word_idx["IO"][batch_idx]]
    
    W_Q = model.W_Q[layer, head_idx].clone() # shape (d_model, d_head)
    b_Q = model.b_Q[layer, head_idx].clone() # shape (d_head,)

    linear_map = einops.einsum(W_Q, keys, "d_model d_head, batch d_head -> batch d_model") / (model.cfg.d_head ** 0.5)
    bias_term = einops.einsum(b_Q, keys, "d_head, batch d_head -> batch") / (model.cfg.d_head ** 0.5)

    if isinstance(batch_idx, int):
        linear_map = linear_map[0]
        bias_term = bias_term[0]

    return linear_map, bias_term


def attn_scores_as_linear_func_of_keys(
    batch_idx: Optional[Union[int, List[int], Int[Tensor, "batch"]]],
    head: Tuple[int, int],
    model: HookedTransformer,
    ioi_cache: ActivationCache,
    ioi_dataset: IOIDataset,
    subtract_S1_attn_scores: bool = False
) -> Float[Tensor, "d_model"]:
    '''
    If you hold queries fixed, then attention scores are a linear function of the keys.
    I want to fix the queries of head 10.7, and get a linear function mapping keys -> attention scores.
    I can then see if (for example) the embedding vector for the IO token has a really big image in this linear fn.

    Here, if `subtract_S1_attn_scores` is True, this implies that we'll be passing (key_IO - key_S1) to this linear
    map. So we want to make the bias zero, but not change the linear map.
    '''
    layer, head_idx = head
    if isinstance(batch_idx, int):
        batch_idx = [batch_idx]
    if batch_idx is None:
        batch_idx = range(len(ioi_cache["q", 0]))

    queries = ioi_cache["q", layer][:, :, head_idx] # shape (all_batch, seq_K, d_head)
    queries_at_END = queries[batch_idx, ioi_dataset.word_idx["end"][batch_idx]] # shape (batch, d_head)
    
    W_K = model.W_K[layer, head_idx].clone() # shape (d_model, d_head)
    b_K = model.b_K[layer, head_idx].clone() # shape (d_head,)

    linear_map = einops.einsum(W_K, queries_at_END, "d_model d_head, batch d_head -> batch d_model") / (model.cfg.d_head ** 0.5)
    bias_term = einops.einsum(b_K, queries_at_END, "d_head, batch d_head -> batch") / (model.cfg.d_head ** 0.5)

    if isinstance(batch_idx, int):
        linear_map = linear_map[0]
        bias_term = bias_term[0]

    if subtract_S1_attn_scores:
        # In this case, we assume the key-side vector supplied will be the difference between the key vectors for the IO and S1 tokens.
        # We don't change the linear map, but we do change the bias term (because it'll be added then subtracted, i.e. it should be zero!)
        bias_term *= 0

    return linear_map, bias_term









def get_attn_scores_as_linear_func_of_queries_for_histogram(
    NNMH: Tuple[int, int],
    num_batches: int,
    batch_size: int,
    model: HookedTransformer,
    name_tokens: List[int],
    subtract_S1_attn_scores: bool = False,
):
    names = ["W_U[IO]", "NMH 9.9 output", "NMH 9.9, ⟂ W_U[IO]", "W_U[S]", "W_U[random name]", "W_U[random]", "No patching"]

    attn_scores = {k: t.empty((0,)).to(device) for k in names}
    attn_probs = {k: t.empty((0,)).to(device) for k in names}

    for seed in tqdm(range(num_batches)):

        ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True)

        linear_map, bias_term = attn_scores_as_linear_func_of_queries(batch_idx=None, head=NNMH, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)
        assert linear_map.shape == (batch_size, model.cfg.d_model)

        # Has to be manual, because apparently `apply_ln_to_stack` doesn't allow it to be applied at different sequence positions
        nmh_99_output = einops.einsum(ioi_cache["z", 9][range(batch_size), ioi_dataset.word_idx["end"], 9], model.W_O[9, 9], "batch d_head, d_head d_model -> batch d_model")
        W_U_IO = model.W_U.T[t.tensor(ioi_dataset.io_tokenIDs)]
        nmh_99_par, nmh_99_perp = project(nmh_99_output, W_U_IO, return_type="projections")
        resid_vectors = {
            "W_U[IO]": W_U_IO,
            "W_U[S]": model.W_U.T[t.tensor(ioi_dataset.s_tokenIDs)],
            "W_U[random]": model.W_U.T[t.randint(size=(batch_size,), low=0, high=model.cfg.d_vocab)],
            "W_U[random name]": model.W_U.T[np.random.choice(name_tokens, size=(batch_size,))],
            "NMH 9.9 output": nmh_99_output,
            "NMH 9.9, ⟂ W_U[IO]": nmh_99_perp,
            "No patching": ioi_cache["resid_pre", NNMH[0]][range(batch_size), ioi_dataset.word_idx["end"]]
        }

        normalized_resid_vectors = {
            name: (q_side_vector - q_side_vector.mean(dim=-1, keepdim=True)) / q_side_vector.var(dim=-1, keepdim=True).pow(0.5)
            for name, q_side_vector in resid_vectors.items()
        }

        new_attn_scores = {
            name: einops.einsum(linear_map, q_side_vector, "batch d_model, batch d_model -> batch") + bias_term
            for name, q_side_vector in normalized_resid_vectors.items()
        }

        attn_scores = {
            name: t.cat([attn_scores[name], new_attn_scores[name]])
            for name in attn_scores.keys()
        }
    
        # Get the attention scores from END to all other tokens (so I can get new attn probs from patching)
        other_attn_scores_at_this_posn = ioi_cache["attn_scores", NNMH[0]][range(batch_size), NNMH[1], ioi_dataset.word_idx["end"]]

        t.cuda.empty_cache()

        for k, v in new_attn_scores.items():
            all_attn_scores = other_attn_scores_at_this_posn.clone()
            # Set the attention scores from END -> IO to their new values
            all_attn_scores[range(batch_size), ioi_dataset.word_idx["IO"]] = v
            # Take softmax over keys, get new attn probs from END -> IO
            all_probs = all_attn_scores.softmax(dim=-1)[range(batch_size), ioi_dataset.word_idx["IO"]]
            attn_probs[k] = t.cat([attn_probs[k], all_probs])

    return attn_scores, attn_probs




def get_attn_scores_as_linear_func_of_keys_for_histogram(
    NNMH: Tuple[int, int],
    num_batches: int,
    batch_size: int,
    model: HookedTransformer,
    subtract_S1_attn_scores: bool = False,
):
    effective_embeddings = get_effective_embedding_2(model) 

    W_E = effective_embeddings["W_E (no MLPs)"]
    W_EE = effective_embeddings["W_E (including MLPs)"]
    W_EE_subE = effective_embeddings["W_E (only MLPs)"]

    keyside_names = ["W_E[IO]", "W_EE[IO]", "W_EE_subE[IO]", "No patching", "MLP0_out"]
    attn_scores = {k: t.empty((0,)).to(device) for k in keyside_names}
    attn_probs = {k: t.empty((0,)).to(device) for k in keyside_names}

    for seed in tqdm(range(num_batches)):

        ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True)

        linear_map, bias_term = attn_scores_as_linear_func_of_keys(batch_idx=None, head=NNMH, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)
        assert linear_map.shape == (batch_size, model.cfg.d_model)

        # Has to be manual, because apparently `apply_ln_to_stack` doesn't allow it to be applied at different sequence positions
        # ! Note - I don't actually have to do this if I'm computing cosine similarity! Maybe I should be doing this instead?
        resid_vectors = {
            "W_E[IO]": W_E[t.tensor(ioi_dataset.io_tokenIDs)],
            "W_EE[IO]": W_EE[t.tensor(ioi_dataset.io_tokenIDs)],
            "W_EE_subE[IO]": W_EE_subE[t.tensor(ioi_dataset.io_tokenIDs)],
            "No patching": ioi_cache["resid_pre", NNMH[0]][range(batch_size), ioi_dataset.word_idx["IO"]],
            "MLP0_out": ioi_cache["mlp_out", 0][range(batch_size), ioi_dataset.word_idx["IO"]],
        }
        normalized_resid_vectors = {
            name: (k_side_vector - k_side_vector.mean(dim=-1, keepdim=True)) / k_side_vector.var(dim=-1, keepdim=True).pow(0.5)
            for name, k_side_vector in resid_vectors.items()
        }
        
        if subtract_S1_attn_scores:
            resid_vectors_baseline = {
                "W_E[IO]": W_E[t.tensor(ioi_dataset.s_tokenIDs)],
                "W_EE[IO]": W_EE[t.tensor(ioi_dataset.s_tokenIDs)],
                "W_EE_subE[IO]": W_EE_subE[t.tensor(ioi_dataset.s_tokenIDs)],
                "No patching": ioi_cache["resid_pre", NNMH[0]][range(batch_size), ioi_dataset.word_idx["S1"]],
                "MLP0_out": ioi_cache["mlp_out", 0][range(batch_size), ioi_dataset.word_idx["S1"]],
            }
            normalized_resid_vectors_baseline = {
                name: (k_side_vector - k_side_vector.mean(dim=-1, keepdim=True)) / k_side_vector.var(dim=-1, keepdim=True).pow(0.5)
                for name, k_side_vector in resid_vectors_baseline.items()
            }
            normalized_resid_vectors = {name: normalized_resid_vectors[name] - normalized_resid_vectors_baseline[name] for name in resid_vectors.keys()}


        new_attn_scores = {
            name: einops.einsum(linear_map, k_side_vector, "batch d_model, batch d_model -> batch") + bias_term
            for name, k_side_vector in normalized_resid_vectors.items()
        }

        attn_scores = {
            name: t.cat([attn_scores[name], new_attn_scores[name]])
            for name in attn_scores.keys()
        }
    
        # Get the attention scores from END to all other tokens (so I can get new attn probs from patching)
        other_attn_scores_at_this_posn = ioi_cache["attn_scores", NNMH[0]][range(batch_size), NNMH[1], ioi_dataset.word_idx["end"]]

        t.cuda.empty_cache()

        for k, v in new_attn_scores.items():
            all_attn_scores = other_attn_scores_at_this_posn.clone()
            # Set the attention scores from END -> IO to their new values
            all_attn_scores[range(batch_size), ioi_dataset.word_idx["IO"]] = v
            # Take softmax over keys, get new attn probs from END -> IO
            all_probs = all_attn_scores.softmax(dim=-1)[range(batch_size), ioi_dataset.word_idx["IO"]]
            attn_probs[k] = t.cat([attn_probs[k], all_probs])

    if subtract_S1_attn_scores:
        keyside_names = {
            "W_E[IO]": "W_E[IO] - W_E[S1]", 
            "W_EE[IO]": "W_EE[IO] - W_EE[S1]", 
            "W_EE_subE[IO]": "W_EE_subE[IO] - W_EE_subE[S1]", 
            "No patching": "No patching (IO - S1)",
            "MLP0_out": "MLP0_out (IO - S1)",
        }
        attn_scores = {keyside_names[k]: v for (k, v) in attn_scores.items()}

    return attn_scores, attn_probs





def decompose_attn_scores(
    batch_size: int,
    seed: int,
    nnmh: Tuple[int, int],
    model: HookedTransformer,
    decompose_by: Literal["keys", "queries"],
    show_plot: bool = False,
    intervene_on_query: Literal["sub_W_U_IO", "project_to_W_U_IO", None] = None,
    intervene_on_key: Literal["sub_MLP0", "project_to_MLP0", None] = None,
    use_effective_embedding: bool = False,
    use_layer0_heads: bool = False,
    subtract_S1_attn_scores: bool = False,
    include_S1_in_unembed_projection: bool = False,
    static: bool = False, # determines if plot is static
):
    '''
    Creates heatmaps of attention score decompositions.

    decompose_by:
        Can be "keys" or "queries", indicating what I want to take the decomposition over.
        For instance, if "keys", then we treat queries as fixed, and decompose attn score contributions by component writing on the key-side.

    intervene_on_query:
        If None, we leave the query vector unchanged.
        If "sub_W_U_IO", we substitute the query vector with the embedding of the IO token (because we think this is the "purest signal" for copy-suppression on query-side).
        If "project_to_W_U_IO", we get 2 different query vectors: the one from residual stream being projected onto the IO unembedding direction, and the residual.

    intervene_on_key:
        If None, we leave the key vector unchanged.
        If "sub_MLP0", we substitute the key vector with the output of the first MLP (because we think this is the "purest signal" for copy-suppression on key-side).
        If "project_to_MLP0", we get 2 different key vectors: the one from residual stream being projected onto the MLP0 direction, and the residual.

    
    Note that the `intervene_on_query` and `intervene_on_key` args do different things depending on the value of `decompose_by`.
    If `decompose_by == "keys"`, then:
        `intervene_on_query` is used to overwrite/decompose the query vector at head 10.7 (i.e. to define a new linear function)
            see (1A)
        `intervene_on_key` is used to overwrite/decompose the key vectors by component, before head 10.7
            see (1B)
    If `decompose_by == "queries"`, then the reverse is true (see (2A) and (2B)).

    Some other important arguments:

    use_effective_embedding:
        If True, we use the effective embedding (i.e. from fixing self-attn to be 1 in attn layer 0) rather than the actual output of MLP0. These shouldn't really be that 
        different (if our W_EE is principled), but unfortunately they are.

    use_layer0_heads:
        If True, then rather than using MLP0 output, we use MLP0 output plus the layer0 attention heads.

    subtract_S1_attn_scores:
        If "S1", we subtract the attention score from "END" to "S1"
        This seems like it might help clear up some annoying noise we see in the plots, and make the core pattern a bit cleaner.

        To be clear: 
            if decompose_by == "keys", then for each keyside component, we want to see if (END -> component_IO) is higher than (END -> component_S1)
                which means we'll need the component for IO and for S1, when we get to the ioi_cache indexing stage
                see (3A)
            if decompose_by == "queries", then for each queryside component, we want to see if (component_END -> IO) is higher than (component_END -> S1)
                which means we'll need to subtract the keyside linear map for S1 from the keyside linear map for IO
                see (3B)
    '''
    t.cuda.empty_cache()
    ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True, prepend_bos=True)

    S1_seq_pos_indices = ioi_dataset.word_idx["S1"]
    IO_seq_pos_indices = ioi_dataset.word_idx["IO"]
    end_seq_pos_indices = ioi_dataset.word_idx["end"]

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
    MLP0_output_scaled = (MLP0_output - MLP0_output.mean(-1, keepdim=True)) / MLP0_output.var(dim=-1, keepdim=True).pow(0.5)

    # * Get the unembeddings
    unembeddings = model.W_U.T[ioi_dataset.io_tokenIDs]
    unembeddings_S1 = model.W_U.T[ioi_dataset.s_tokenIDs]
    unembeddings_scaled = (unembeddings - unembeddings.mean(-1, keepdim=True)) / unembeddings.var(dim=-1, keepdim=True).pow(0.5)

    # * Get residual stream pre-heads
    resid_pre = ioi_cache["resid_pre", nnmh[0]]
    resid_pre_normalised = (resid_pre - resid_pre.mean(-1, keepdim=True)) / resid_pre.var(dim=-1, keepdim=True).pow(0.5)
    resid_pre_normalised_slice_IO = resid_pre_normalised[range(batch_size), IO_seq_pos_indices]
    resid_pre_normalised_slice_S1 = resid_pre_normalised[range(batch_size), S1_seq_pos_indices]
    resid_pre_normalised_slice_end = resid_pre_normalised[range(batch_size), end_seq_pos_indices]

    if decompose_by == "keys":

        assert intervene_on_query in [None, "sub_W_U_IO", "project_to_W_U_IO"]

        decomp_seq_pos_indices = IO_seq_pos_indices
        lin_map_seq_pos_indices = end_seq_pos_indices

        W_Q = model.W_Q[nnmh[0], nnmh[1]]
        b_Q = model.b_Q[nnmh[0], nnmh[1]]
        q_name = utils.get_act_name("q", nnmh[0])
        q_raw = ioi_cache[q_name].clone()
        
        # ! (1A)
        # * Get 2 linear functions from keys -> attn scores, corresponding to the 2 different components of query vectors: (∥ / ⟂) to W_U[IO]
        if intervene_on_query == "project_to_W_U_IO":
            if include_S1_in_unembed_projection:
                resid_pre_in_unembed_dir, resid_pre_in_unembed_perpdir = project(resid_pre_normalised_slice_end, [unembeddings, unembeddings_S1])
            else:
                resid_pre_in_unembed_dir, resid_pre_in_unembed_perpdir = project(resid_pre_normalised_slice_end, unembeddings)

            # Overwrite the query-side vector in the cache with the projection in the unembedding direction
            q_new = einops.einsum(resid_pre_in_unembed_dir, W_Q, "batch d_model, d_model d_head -> batch d_head")
            q_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = q_new
            ioi_cache_dict_io_dir = {**ioi_cache.cache_dict, **{q_name: q_raw.clone()}}
            ioi_cache_io_dir = ActivationCache(cache_dict=ioi_cache_dict_io_dir, model=model)
            linear_map_io_dir, bias_term_io_dir = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_io_dir, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)

            # Overwrite the query-side vector with the bit that's perpendicular to the IO unembedding (plus the bias term)
            q_new = einops.einsum(resid_pre_in_unembed_perpdir, W_Q, "batch d_model, d_model d_head -> batch d_head") + b_Q
            q_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = q_new
            ioi_cache_dict_io_perpdir = {**ioi_cache.cache_dict, **{q_name: q_raw.clone()}}
            ioi_cache_io_perpdir = ActivationCache(cache_dict=ioi_cache_dict_io_perpdir, model=model)
            linear_map_io_perpdir, bias_term_io_perpdir = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_io_perpdir, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)
            
            linear_map_dict = {"IO_dir": (linear_map_io_dir, bias_term_io_dir), "IO_perp": (linear_map_io_perpdir, bias_term_io_perpdir)}

        # ! (1A)
        # * Get new linear function from keys -> attn scores, corresponding to subbing in W_U[IO] as queryside vector
        # * TODO - replace `sub`, because it implies `subtract` rather than `substitute`
        elif intervene_on_query == "sub_W_U_IO":
            # Overwrite the query-side vector by replacing it with the (normalized) W_U[IO] unembeddings
            q_new = einops.einsum(unembeddings_scaled, W_Q, "batch d_model, d_model d_head -> batch d_head") + b_Q
            q_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = q_new
            ioi_cache_dict_io_subbed = {**ioi_cache.cache_dict, **{q_name: q_raw.clone()}}
            ioi_cache_io_subbed = ActivationCache(cache_dict=ioi_cache_dict_io_subbed, model=model)
            linear_map_io_subbed, bias_term_io_subbed = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_io_subbed, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)
            
            linear_map_dict = {"IO_sub": (linear_map_io_subbed, bias_term_io_subbed)}

        # * Get linear function from keys -> attn scores (no intervention on query)
        else:
            linear_map, bias_term = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)
            linear_map_dict = {"unchanged": (linear_map, bias_term)}


    
    elif decompose_by == "queries":

        assert intervene_on_key in [None, "sub_MLP0", "project_to_MLP0"]

        decomp_seq_pos_indices = end_seq_pos_indices
        lin_map_seq_pos_indices = end_seq_pos_indices

        W_K = model.W_K[nnmh[0], nnmh[1]]
        b_K = model.b_K[nnmh[0], nnmh[1]]
        k_name = utils.get_act_name("k", nnmh[0])
        k_raw = ioi_cache[k_name].clone()
        
        # ! (2B)
        # * Get 2 linear functions from queries -> attn scores, corresponding to the 2 different components of key vectors: (∥ / ⟂) to MLP0_out
        if intervene_on_key == "project_to_MLP0":
            resid_pre_in_mlp0_dir, resid_pre_in_mlp0_perpdir = project(resid_pre_normalised_slice_IO, MLP0_output)
            resid_pre_in_mlp0_dir_S1, resid_pre_in_mlp0_perpdir_S1 = project(resid_pre_normalised_slice_S1, MLP0_output_S1)

            # Overwrite the key-side vector in the cache with the projection in the MLP0_output direction
            # Do the same with the S1 baseline (note that we might not actually use it, but it's good to have it there)
            k_new = einops.einsum(resid_pre_in_mlp0_dir, W_K, "batch d_model, d_model d_head -> batch d_head")
            k_raw[range(batch_size), IO_seq_pos_indices, nnmh[1]] = k_new
            k_new_S1 = einops.einsum(resid_pre_in_mlp0_dir_S1, W_K, "batch d_model, d_model d_head -> batch d_head")
            k_raw[range(batch_size), S1_seq_pos_indices, nnmh[1]] = k_new_S1
            ioi_cache_dict_mlp0_dir = {**ioi_cache.cache_dict, **{k_name: k_raw.clone()}}
            ioi_cache_mlp0_dir = ActivationCache(cache_dict=ioi_cache_dict_mlp0_dir, model=model)
            # ! (3B)
            # * This function (the `subtract_S1_attn_scores` argument) is where we subtract the S1-baseline from the linear map from queries -> attn scores))
            # * Obviously the same is true for the other 3 instances of the `attn_scores_as_linear_func_of_queries` function below
            linear_map_mlp0_dir, bias_term_mlp0_dir = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_mlp0_dir, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)

            # Overwrite the key-side vector with the bit that's perpendicular to the MLP0_output (plus the bias term)
            k_new = einops.einsum(resid_pre_in_mlp0_perpdir, W_K, "batch d_model, d_model d_head -> batch d_head") + b_K
            k_raw[range(batch_size), IO_seq_pos_indices, nnmh[1]] = k_new
            k_new_S1 = einops.einsum(resid_pre_in_mlp0_perpdir_S1, W_K, "batch d_model, d_model d_head -> batch d_head") + b_K
            k_raw[range(batch_size), S1_seq_pos_indices, nnmh[1]] = k_new_S1
            ioi_cache_dict_mlp0_perpdir = {**ioi_cache.cache_dict, **{k_name: k_raw.clone()}}
            ioi_cache_mlp0_perpdir = ActivationCache(cache_dict=ioi_cache_dict_mlp0_perpdir, model=model)
            linear_map_mlp0_perpdir, bias_term_mlp0_perpdir = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_mlp0_perpdir, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)
            
            linear_map_dict = {"MLP0_dir": (linear_map_mlp0_dir, bias_term_mlp0_dir), "MLP0_perp": (linear_map_mlp0_perpdir, bias_term_mlp0_perpdir)}
        
        # ! (2B)
        # * Get new linear function from queries -> attn scores, corresponding to subbing in MLP0_output as keyside vector
        elif intervene_on_key == "sub_MLP0":
            # Overwrite the key-side vector by replacing it with the (normalized) MLP0_output
            assert not(subtract_S1_attn_scores), "This will behave weirdly right now."
            k_new = einops.einsum(MLP0_output_scaled, W_K, "batch d_model, d_model d_head -> batch d_head") + b_K
            k_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = k_new
            ioi_cache_dict_mlp0_subbed = {**ioi_cache.cache_dict, **{k_name: k_raw.clone()}}
            ioi_cache_mlp0_subbed = ActivationCache(cache_dict=ioi_cache_dict_mlp0_subbed, model=model)
            linear_map_mlp0_subbed, bias_term_mlp0_subbed = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_mlp0_subbed, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)
            
            linear_map_dict = {"MLP0_sub": (linear_map_mlp0_subbed, bias_term_mlp0_subbed)}

        # * Get linear function from queries -> attn scores (no intervention on key)
        else:
            linear_map, bias_term = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)
            linear_map_dict = {"unchanged": (linear_map, bias_term)}


    t.cuda.empty_cache()

    contribution_to_attn_scores_list = []

    # * This is where we get the thing we're projecting keys onto if required (i.e. if we're decomposing by keys, and want to split into ||MLP0 and ⟂MLP0)
    if (intervene_on_key is not None) and (decompose_by == "keys"):
        assert intervene_on_key == "project_to_MLP0", "If you're decomposing by key component, then 'intervene_on_key' must be 'project_to_MLP0' or None."
        contribution_to_attn_scores_shape = (2, 1 + nnmh[0], model.cfg.n_heads + 1)
        contribution_to_attn_scores_names = ["MLP0_dir", "MLP0_perp"]

    # * This is where we get the thing we're projecting queries onto if required (i.e. if we're decomposing by queries, and want to split into ||W_U[IO] and ⟂W_U[IO])
    elif (intervene_on_query is not None) and (decompose_by == "queries"):
        assert intervene_on_query == "project_to_W_U_IO", "If you're decomposing by key component, then 'intervene_on_query' must be 'project_to_W_U_IO' or None."
        unembeddings = model.W_U.T[ioi_dataset.io_tokenIDs]
        contribution_to_attn_scores_shape = (2, 1 + nnmh[0], model.cfg.n_heads + 1)
        contribution_to_attn_scores_names = ["IO_dir", "IO_perp"]

    # * We're not projecting by anything when we get the decomposed bits
    else:
        contribution_to_attn_scores_shape = (1, 1 + nnmh[0], model.cfg.n_heads + 1)
        contribution_to_attn_scores_names = ["unchanged"]



    def get_decomposed_components(component_name, layer=None):
        '''
        This function does the following:
            > Get the value we want from the ioi_cache (at the appopriate sequence positions for the decomposition: either "IO" or "end")
            > If we need to project it in a direction, then apply that projection (this gives it an extra dim at the start)
            > If we need to subtract the mean of S1, do that too.
        '''
        assert component_name in ["result", "mlp_out", "embed", "pos_embed"]
        
        # Index from ioi cache
        component_output: Float[Tensor, "batch *n_heads d_model"] = ioi_cache[component_name, layer][range(batch_size), decomp_seq_pos_indices]

        # Apply scaling
        component_output_scaled = component_output / (ln_scale.unsqueeze(1) if (component_name == "result") else ln_scale)

        # Apply projections
        # ! (2A)
        # * This is where we decompose the query-side output of each component, by possibly projecting it onto the ||W_U[IO] and ⟂W_U[IO] directions
        if (decompose_by == "queries") and (intervene_on_query == "project_to_W_U_IO"):
            projection_dir = einops.repeat(unembeddings, "b d_m -> b heads d_m", heads=model.cfg.n_heads) if (component_name == "result") else unembeddings
            projection_dir_S1 = einops.repeat(unembeddings_S1, "b d_m -> b heads d_m", heads=model.cfg.n_heads) if (component_name == "result") else unembeddings_S1
            projection_dir = projection_dir if not(include_S1_in_unembed_projection) else [projection_dir, projection_dir_S1]
            component_output_scaled = t.stack(project(component_output_scaled, projection_dir))
        # ! (1B)
        # * This is where we decompose the key-side output of each component, by possibly projecting it onto the ||MLP0 and ⟂MLP0 directions
        elif (decompose_by == "keys") and (intervene_on_key == "project_to_MLP0"):
            projection_dir = einops.repeat(MLP0_output, "b d_m -> b heads d_m", heads=model.cfg.n_heads) if (component_name == "result") else MLP0_output
            component_output_scaled = t.stack(project(component_output_scaled, projection_dir))

        # ! (3A)
        # * This is where we subtract the keyside component baseline of S2 (if our decomposition is by-keys)
        # * This involves going through exactly the same process as above, except with S2 (I'll make the code shorter)
        if (decompose_by == "keys") and subtract_S1_attn_scores:
            # Calculate scaled baseline
            component_output_S1 = ioi_cache[component_name, layer][range(batch_size), S1_seq_pos_indices]
            component_output_scaled_S1 = component_output_S1 / (ln_scale_S1.unsqueeze(1) if (component_name == "result") else ln_scale_S1)
            # Apply projections, if required
            if (intervene_on_key == "project_to_MLP0"):
                projection_dir_S1 = einops.repeat(MLP0_output_S1, "b d_m -> b heads d_m", heads=model.cfg.n_heads) if (component_name == "result") else MLP0_output_S1
                component_output_scaled_S1 = t.stack(project(component_output_scaled_S1, projection_dir_S1))
            # Subtract baseline
            component_output_scaled = component_output_scaled - component_output_scaled_S1

        return component_output_scaled



    results_dict = {}

    for name, (linear_map, bias_term) in linear_map_dict.items():
        
        # Check linear map is valid
        assert linear_map.shape == (batch_size, model.cfg.d_model)
        assert bias_term.shape == (batch_size,)

        # Create tensor to store all the values for this facet plot (possibly 2 facet plots, if we're splitting by projecting our decomposed components)
        contribution_to_attn_scores = t.zeros(contribution_to_attn_scores_shape)

        # Get scale factor we'll be dividing all our components by
        ln_scale = ioi_cache["scale", nnmh[0], "ln1"][range(batch_size), decomp_seq_pos_indices, nnmh[1]]
        ln_scale_S1 = ioi_cache["scale", nnmh[0], "ln1"][range(batch_size), S1_seq_pos_indices, nnmh[1]]
        assert ln_scale.shape == ln_scale_S1.shape == (batch_size, 1)

        # We start with all the things before attn heads and MLPs
        embed_scaled = get_decomposed_components("embed")
        pos_embed_scaled = get_decomposed_components("pos_embed")
        # Add these to the results tensor. Note we use `:` because this covers cases where the first dim is 1 (no projection split) or 2 (projection split)
        contribution_to_attn_scores[:, 0, 0] = einops.einsum(embed_scaled, linear_map, "... batch d_model, batch d_model -> ... batch").mean(-1)
        contribution_to_attn_scores[:, 0, 1] = einops.einsum(pos_embed_scaled, linear_map, "... batch d_model, batch d_model -> ... batch").mean(-1)
        # Add the bias term (this is only ever added to the last term, because it's the perpendicular one)
        contribution_to_attn_scores[-1, 0, 2] = bias_term.mean()

        for layer in range(nnmh[0]):

            # Calculate output of each attention head, split by projecting onto MLP0 output if necessary, then add to our results tensor
            # z = ioi_cache["z", layer][range(batch_size), decomp_seq_pos_indices]
            # result = einops.einsum(z, model.W_O[layer], "batch n_heads d_head, n_heads d_head d_model -> batch n_heads d_model")
            results_scaled = get_decomposed_components("result", layer)
            contribution_to_attn_scores[:, 1 + layer, :model.cfg.n_heads] = einops.einsum(
                results_scaled, linear_map, 
                "... batch n_heads d_model, batch d_model -> ... n_heads batch"
            ).mean(-1)

            # Calculate output of the MLPs, split by projecting onto MLP0 output if necessary, then add to our results tensor
            mlp_out_scaled = get_decomposed_components("mlp_out", layer)
            contribution_to_attn_scores[:, 1 + layer, -1] = einops.einsum(
                mlp_out_scaled, linear_map,
                "... batch d_model, batch d_model -> ... batch"
            ).mean(-1)
        
        contribution_to_attn_scores_list.append(contribution_to_attn_scores.squeeze())

        for name2, contribution_to_attn_scores_slice in zip(contribution_to_attn_scores_names, contribution_to_attn_scores):
            names = tuple(sorted([name, name2]))
            results_dict[names] = contribution_to_attn_scores_slice.squeeze()

    if len(contribution_to_attn_scores_list) == 1:
        contribution_to_attn_scores = contribution_to_attn_scores_list[0]
    else:
        contribution_to_attn_scores = t.stack(contribution_to_attn_scores_list)

    if show_plot:
        plot_contribution_to_attn_scores(contribution_to_attn_scores, decompose_by, static=static)
    
    if len(results_dict) == 1:
        return results_dict[list(results_dict.keys())[0]]
    else:
        return results_dict




def plot_contribution_to_attn_scores(
    contribution_to_attn_scores: Float[Tensor, "... layer component"],
    decompose_by: str,
    facet_labels: Optional[List[str]] = None,
    animation_labels: Optional[List[str]] = None,
    facet_col_wrap: Optional[int] = None,
    title: Optional[str] = None,
    static: bool = False,
):
    text = [["W<sub>E</sub>", "W<sub>pos</sub>", "b<sub>K</sub>" if decompose_by == "keys" else "b<sub>Q</sub>"] + ["" for _ in range(10)]]
    for layer in range(0, 10):
        text.append([f"{layer}.{head}" for head in range(12)] + [f"MLP{layer}"])

    single_plot = contribution_to_attn_scores.ndim == 2
    facets = contribution_to_attn_scores.ndim == 3
    animation_and_facets = contribution_to_attn_scores.ndim == 4

    facet_col = None
    animation_frame = None
    if facets:
        facet_col = 0
    elif animation_and_facets:
        animation_frame = 0
        facet_col = 1

    if facet_col is not None:
        num_facets = contribution_to_attn_scores.shape[facet_col]
        if facet_col_wrap is None: facet_col_wrap = num_facets
        num_figs_width = facet_col_wrap
        num_figs_height = num_facets // facet_col_wrap
        width=1300 if (num_figs_width > 1) else 900
        height=(1100 if (num_figs_height > 1) else 600) + (100 if animation_frame is not None else 0)
    else:
        width = 900
        height = 600

    fig = imshow(
        contribution_to_attn_scores,
        facet_col = facet_col,
        facet_labels = facet_labels,
        facet_col_wrap = facet_col_wrap,
        animation_frame = animation_frame,
        animation_labels = animation_labels,
        title=f"Contribution to attention scores ({'key' if decompose_by in ['k', 'keys'] else 'query'}-side)" if title is None else title,
        labels={"x": "Component (attn heads & MLP)", "y": "Layer"},
        y=["misc"] + [str(i) for i in range(10)],
        x=[f"H{i}" for i in range(12)] + ["MLP"],
        border=True,
        width=width,
        height=height,
        return_fig=True,
    )
    for i in range(len(fig.data)):
        fig.data[i].update(
            text=text, 
            texttemplate="%{text}", 
            textfont={"size": 12}
        )
    config = {} if not(static) else {"staticPlot": True}
    fig.show(config=config)







def project(
    x: Float[Tensor, "... dim"],
    dir: Union[List[Float[Tensor, "... dim"]], Float[Tensor, "... dim"]],
    test: bool = False,
    return_type: Literal["projections", "coeffs", "both"] = "projections",
):
    '''
    x: 
        Shape (*batch_dims, d), or list of such shapes
        Batch of vectors
    
    dir:
        Shape (*batch_dims, d)
        Batch of vectors (which will be normalized)

    test:
        If true, runs a bunch of sanity-check-style tests, and prints out the output

    Returns:
        Two batches of vectors: x_dir and x_perp, such that:
            x_dir + x_perp = x
            x_dir is the component of x in the direction dir (or in the subspace
            spanned by the vectors in dir, if dir is a list).

    Notes:
        Make sure x and dir (or each element in dir) have the same shape, I don't want to
        mess up broadcasting by accident! Do einops.repeat on dir if you have to.
    '''
    assert return_type in ["projections", "coeffs", "both"]
    device = x.device
    if isinstance(dir, Tensor): dir = [dir]
    assert all([x.shape == dir_.shape for dir_ in dir])
    dir = t.stack(dir, dim=-1)

    # Get the SVD of the stack of matrices we're projecting in the direction of
    # So U tells us directions, and V tells us linear combinations (which we don't need)
    svd = t.svd(dir)
    if test:
        t.testing.assert_close(svd.U @ t.diag_embed(svd.S) @ svd.V.mH, dir)
        U_norms = svd.U.norm(dim=-2) # norm of columns
        t.testing.assert_close(U_norms, t.ones_like(U_norms))
        print("Running tests for projection function:")
        print("\tSVD tests passed")

    # Calculate the component of x along the different directions of svd.U
    x_coeffs = einops.einsum(
        x, svd.U,
        "... dim, ... dim directions -> ... directions"
    )
    if return_type == "coeffs":
        return x_coeffs

    # Project x onto these directions (summing over each of the directional projections)
    x_dir = einops.einsum(
        x_coeffs, svd.U,
        "... directions, ... dim directions -> ... dim"
    )

    if test:
        # First, test all the projections are orthogonal to each other
        x_dir_projections = einops.einsum(
            x_coeffs, svd.U,
            "... directions, ... dim directions -> ... dim directions"
        )
        x_dir_projections_normed = x_dir_projections / x_dir_projections.norm(dim=-2, keepdim=True)
        x_dir_cos_sims = einops.einsum(
            x_dir_projections_normed, x_dir_projections_normed,
            "... dim directions_left, ... dim directions_right -> ... directions_left directions_right"
        )
        
        x_dir_cos_sims_expected = t.eye(x_dir_cos_sims.shape[-1]).to(device)
        diff = t.where(x_dir_cos_sims_expected.bool(), t.tensor(0.0).to(device), x_dir_cos_sims - x_dir_cos_sims_expected).abs().max().item()
        assert diff < 1e-5
        print(f"\tCos sim test passed: max cos sim diff = {diff:.4e}")

        # Second, test that the sum of norms equals the original norm
        x_dir_norms = x_dir.norm(dim=-1).pow(2)
        x_dir_perp_norms = (x - x_dir).norm(dim=-1).pow(2)
        x_norms = x.norm(dim=-1).pow(2)
        diff = (x_dir_norms + x_dir_perp_norms - x_norms).abs().max().item()
        assert diff < 1e-5
        print(f"\tNorms test passed: max norm diff = {diff:.4e}")

    if return_type == "both":
        return x_dir, x - x_dir, x_coeffs
    elif return_type == "projections":
        return x_dir, x - x_dir

# def test_project(project: Callable):

#     # First test: 1D
#     x = t.tensor([1., 2.])
#     dir = t.tensor([1., 0.])
#     x_dir, x_perp = project(x, dir)
#     t.testing.assert_close(x_dir, t.tensor([1., 0.]))
#     t.testing.assert_close(x_perp, t.tensor([0., 2.]))

#     # Second test: 2D (with batch dim)
#     x = t.tensor([[1., 2.], [1., 2.]])
#     dir = t.tensor([[1., 0.], [0., 1.]])
#     x_dir, x_perp = project(x, dir)
#     t.testing.assert_close(x_dir, t.tensor([[1., 0.], [0., 2.]]))
#     t.testing.assert_close(x_perp, t.tensor([[0., 2.], [1., 0.]]))

#     # Third test: 2D (with batch dim), not normalized correctly
#     x = t.tensor([[1., 2.], [1., 2.]])
#     dir = t.tensor([[2., 0.], [0., -1.]])
#     x_dir, x_perp = project(x, dir)
#     t.testing.assert_close(x_dir, t.tensor([[1., 0.], [0., 2.]]))
#     t.testing.assert_close(x_perp, t.tensor([[0., 2.], [1., 0.]]))

#     print("All tests in `test_project` passed!")


# test_project(project)












def decompose_attn_scores_full(
    batch_size: int,
    seed: int,
    nnmh: Tuple[int, int],
    model: HookedTransformer,
    use_effective_embedding: bool = False,
    use_layer0_heads: bool = False,
    subtract_S1_attn_scores: bool = False,
    include_S1_in_unembed_projection: bool = False,
):
    '''
    Creates heatmaps of attention score decompositions.

    decompose_by:
        Can be "keys" or "queries", indicating what I want to take the decomposition over.
        For instance, if "keys", then we treat queries as fixed, and decompose attn score contributions by component writing on the key-side.

    intervene_on_query:
        If None, we leave the query vector unchanged.
        If "sub_W_U_IO", we substitute the query vector with the embedding of the IO token (because we think this is the "purest signal" for copy-suppression on query-side).
        If "project_to_W_U_IO", we get 2 different query vectors: the one from residual stream being projected onto the IO unembedding direction, and the residual.

    intervene_on_key:
        If None, we leave the key vector unchanged.
        If "sub_MLP0", we substitute the key vector with the output of the first MLP (because we think this is the "purest signal" for copy-suppression on key-side).
        If "project_to_MLP0", we get 2 different key vectors: the one from residual stream being projected onto the MLP0 direction, and the residual.

    
    Note that the `intervene_on_query` and `intervene_on_key` args do different things depending on the value of `decompose_by`.
    If `decompose_by == "keys"`, then:
        `intervene_on_query` is used to overwrite/decompose the query vector at head 10.7 (i.e. to define a new linear function)
            see (1A)
        `intervene_on_key` is used to overwrite/decompose the key vectors by component, before head 10.7
            see (1B)
    If `decompose_by == "queries"`, then the reverse is true (see (2A) and (2B)).

    Some other important arguments:

    use_effective_embedding:
        If True, we use the effective embedding (i.e. from fixing self-attn to be 1 in attn layer 0) rather than the actual output of MLP0. These shouldn't really be that 
        different (if our W_EE is principled), but unfortunately they are.

    use_layer0_heads:
        If True, then rather than using MLP0 output, we use MLP0 output plus the layer0 attention heads.

    subtract_S1_attn_scores:
        If "S1", we subtract the attention score from "END" to "S1"
        This seems like it might help clear up some annoying noise we see in the plots, and make the core pattern a bit cleaner.

        To be clear: 
            if decompose_by == "keys", then for each keyside component, we want to see if (END -> component_IO) is higher than (END -> component_S1)
                which means we'll need the component for IO and for S1, when we get to the ioi_cache indexing stage
                see (3A)
            if decompose_by == "queries", then for each queryside component, we want to see if (component_END -> IO) is higher than (component_END -> S1)
                which means we'll need to subtract the keyside linear map for S1 from the keyside linear map for IO
                see (3B)
    '''
    t.cuda.empty_cache()
    ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True, prepend_bos=True)

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

    contribution_to_attn_scores = t.zeros(
        4, # this is for the 4 options: (∥ / ⟂) to (unembed of IO on query side / MLP0 on key side)
        3 + (nnmh[0] * (1 + model.cfg.n_heads)), # this is for the query-side
        3 + (nnmh[0] * (1 + model.cfg.n_heads)), # this is for the key-side
    )

    keyside_components = []
    queryside_components = []

    # TODO - calculate product directly after filling these in, in case it's too large? Or maybe it's fine cause they are on CPU.
    keys_decomposed = t.zeros(2, 3 + (nnmh[0] * (1 + model.cfg.n_heads)), batch_size, model.cfg.d_head)
    queries_decomposed = t.zeros(2, 3 + (nnmh[0] * (1 + model.cfg.n_heads)), batch_size, model.cfg.d_head)

    def get_component(component_name, layer=None, keyside=False):
        '''
        Gets component (key or query side).

        If we need to subtract the baseline, it returns both the component for IO and the component for S1 (so we can project then subtract scores from each other).
        '''
        full_component = ioi_cache[component_name, layer]
        if keyside:
            component_IO = full_component[range(batch_size), IO_seq_pos_indices] / (ln_scale_IO.unsqueeze(1) if (component_name == "result") else ln_scale_IO)
            component_S1 = full_component[range(batch_size), S1_seq_pos_indices] / (ln_scale_S1.unsqueeze(1) if (component_name == "result") else ln_scale_S1)
            return (component_IO, component_S1) if subtract_S1_attn_scores else component_IO
        else:
            component_END = full_component[range(batch_size), end_seq_pos_indices] / (ln_scale_end.unsqueeze(1) if (component_name == "result") else ln_scale_end)
            return component_END

    b_K = model.b_K[nnmh[0], nnmh[1]]
    if subtract_S1_attn_scores: b_K *= 0
    b_K = einops.repeat(b_K, "d_head -> batch d_head", batch=batch_size)
    b_Q = einops.repeat(model.b_Q[nnmh[0], nnmh[1]], "d_head -> batch d_head", batch=batch_size)

    # First, get the biases and direct terms
    keyside_components.extend([
        b_K,
        get_component("embed", keyside=True),
        get_component("pos_embed", keyside=True),
    ])
    queryside_components.extend([
        b_Q,
        get_component("embed", keyside=False),
        get_component("pos_embed", keyside=False),
    ])

    # Next, get all the MLP terms
    for layer in range(nnmh[0]):
        keyside_components.append(get_component("mlp_out", layer=layer, keyside=True))
        queryside_components.append(get_component("mlp_out", layer=layer, keyside=False))

    # Lastly, all the heads
    for layer in range(nnmh[0]):
        keyside_heads = get_component("result", layer=layer, keyside=True)
        queryside_heads = get_component("result", layer=layer, keyside=False)
        for head in range(model.cfg.n_heads):
            if subtract_S1_attn_scores:
                keyside_components.append((keyside_heads[0][:, head, :], keyside_heads[1][:, head, :]))
            else:
                keyside_components.append(keyside_heads[:, head, :])
            queryside_components.append(queryside_heads[:, head, :])

    # Now, we do the projection thing...
    # ... for keys ....
    keys_decomposed[1, 0] = keyside_components[0]
    for i, keyside_component in enumerate(keyside_components[1:], 1):
        if subtract_S1_attn_scores:
            keyside_component_IO, keyside_component_S1 = keyside_component
            projections = project(keyside_component_IO, MLP0_output), project(keyside_component_S1, MLP0_output_S1)
            projections = t.stack([projections[0][0] - projections[1][0], projections[0][1] - projections[1][1]])
        else:
            projections = project(keyside_component, MLP0_output)
        keys_decomposed[:, i] = einops.einsum(projections.cpu(), model.W_K[nnmh[0], nnmh[1]].cpu(), "projection batch d_model, d_model d_head -> projection batch d_head")
    # ... and for queries ...
    queries_decomposed[1, 0] = queryside_components[0]
    for i, queryside_component in enumerate(queryside_components[1:], 1):
        projections = t.stack(project(queryside_component, unembeddings if not(include_S1_in_unembed_projection) else [unembeddings, unembeddings_S1]))
        queries_decomposed[:, i] = einops.einsum(projections.cpu(), model.W_Q[nnmh[0], nnmh[1]].cpu(), "projection batch d_model, d_model d_head -> projection batch d_head")
    

    # Finally, we do the outer product thing
    for (key_idx, keyside_component) in enumerate(keys_decomposed.unbind(dim=1)):
        for (query_idx, queryside_component) in enumerate(queries_decomposed.unbind(dim=1)):
            contribution_to_attn_scores[:, query_idx, key_idx] = einops.einsum(
                queryside_component,
                keyside_component,
                "q_projection batch d_head, k_projection batch d_head -> q_projection k_projection batch"
            ).mean(-1).flatten() / (model.cfg.d_head ** 0.5)


    return contribution_to_attn_scores





def create_fucking_massive_plot_1(contribution_to_attn_scores):

    full_labels = ["bias", "W<sub>E</sub>", "W<sub>pos</sub>"]
    full_labels += [f"MLP<sub>{L}</sub>" for L in range(10)]
    full_labels += [f"{L}.{H}" for L in range(10) for H in range(12)]

    projection_labels = [
        "q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>",
        "q ∥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>", 
        "q ⊥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>", 
        "q ⊥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>"
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

    full_labels = ["bias", "W<sub>E</sub>", "W<sub>pos</sub>"]
    full_labels += [f"MLP<sub>{L}</sub>" for L in range(10)]
    full_labels += [f"{L}.{H}" for L in range(10) for H in range(12)]

    projection_labels = [
        "q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>",
        "q ∥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>", 
        "q ⊥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>", 
        "q ⊥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>"
    ]

    contribution_to_attn_scores_normalized = contribution_to_attn_scores / contribution_to_attn_scores.sum()

    all_labels = [f"Q = {q}, K = {k}, {proj}" for proj in projection_labels for q in full_labels for k in full_labels]
    all_values = contribution_to_attn_scores_normalized.flatten()

    topk = all_values.abs().topk(40)

    top_labels = [all_labels[i] for i in topk.indices]
    top_values = [all_values[i] for i in topk.indices]

    expected_labels = [
        f'Q = {q}, K = MLP<sub>{L}</sub>, q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>'
        for q in ["9.6", "9.9", "8.10", "7.9"]
        for L in [0] # range(10)
    ]
    weird_but_kinda_expected_labels = [
        f'Q = {q}, K = MLP<sub>{L}</sub>, q ⊥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>'
        for q in ["9.6", "9.9", "8.10", "7.9"]
        for L in [0] # range(10)
    ]
    color_indices = [int(l in expected_labels) * 2 + int(l in weird_but_kinda_expected_labels) for l in top_labels[::-1]]

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

    newnames = ["Not expected", "Weird but kinda expected", "Expected"]
    fig.for_each_trace(lambda t: t.update(name = newnames.pop(0)))

    fig.show()
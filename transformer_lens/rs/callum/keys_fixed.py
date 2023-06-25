from torch import native_dropout
from transformer_lens.cautils.utils import *
from transformer_lens.rs.callum.generate_bag_of_words_quad_plot import get_effective_embedding



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
) -> Float[Tensor, "d_model"]:
    '''
    If you hold queries fixed, then attention scores are a linear function of the keys.

    I want to fix the queries of head 10.7, and get a linear function mapping keys -> attention scores.

    I can then see if (for example) the embedding vector for the IO token has a really big image in this linear fn.
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

    return linear_map, bias_term









def get_attn_scores_and_probs_as_linear_func_of_queries(
    NNMH: Tuple[int, int],
    num_batches: int,
    batch_size: int,
    model: HookedTransformer,
    name_tokens: List[int],
):

    attn_scores = {
        k: t.empty((0,)).to(device)
        for k in ["W_U[IO]", "NMH 9.9 output", "W_U[S]", "W_U[random name]", "W_U[random]", "No patching"]
    }
    attn_probs = {
        k: t.empty((0,)).to(device)
        for k in ["W_U[IO]", "NMH 9.9 output", "W_U[S]", "W_U[random name]", "W_U[random]", "No patching"]
    }

    for seed in tqdm(range(num_batches)):

        ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True)

        linear_map, bias_term = attn_scores_as_linear_func_of_queries(batch_idx=None, head=NNMH, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset)
        assert linear_map.shape == (batch_size, model.cfg.d_model)

        # Has to be manual, because apparently `apply_ln_to_stack` doesn't allow it to be applied at different sequence positions
        resid_vectors = {
            "W_U[IO]": model.W_U.T[t.tensor(ioi_dataset.io_tokenIDs)],
            "W_U[S]": model.W_U.T[t.tensor(ioi_dataset.s_tokenIDs)],
            "W_U[random]": model.W_U.T[t.randint(size=(batch_size,), low=0, high=model.cfg.d_vocab)],
            "W_U[random name]": model.W_U.T[np.random.choice(name_tokens, size=(batch_size,))],
            "NMH 9.9 output": einops.einsum(ioi_cache["z", 9][range(batch_size), ioi_dataset.word_idx["end"], 9], model.W_O[9, 9], "batch d_head, d_head d_model -> batch d_model"),
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




def get_attn_scores_and_probs_as_linear_func_of_keys(
    NNMH: Tuple[int, int],
    num_batches: int,
    batch_size: int,
    model: HookedTransformer,
    name_tokens: List[int],
):
    effective_embeddings = get_effective_embedding(model) 

    W_U = effective_embeddings["W_U (or W_E, no MLPs)"]
    W_EE = effective_embeddings["W_E (including MLPs)"]
    W_EE_subE = effective_embeddings["W_E (only MLPs)"]

    attn_scores = {
        k: t.empty((0,)).to(device)
        for k in ["W_E[IO]", "W_EE[IO]", "W_EE_subE[IO]", "No patching"]
    }
    attn_probs = {
        k: t.empty((0,)).to(device)
        for k in ["W_E[IO]", "W_EE[IO]", "W_EE_subE[IO]", "No patching"]
    }

    for seed in tqdm(range(num_batches)):

        ioi_dataset, ioi_cache = generate_data_and_caches(batch_size, model=model, seed=seed, only_ioi=True)

        linear_map, bias_term = attn_scores_as_linear_func_of_keys(batch_idx=None, head=NNMH, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset)
        assert linear_map.shape == (batch_size, model.cfg.d_model)

        # Has to be manual, because apparently `apply_ln_to_stack` doesn't allow it to be applied at different sequence positions
        # ! Note - I don't actually have to do this if I'm computing cosine similarity! Maybe I should be doing this instead?
        resid_vectors = {
            "W_E[IO]": W_U[t.tensor(ioi_dataset.io_tokenIDs)],
            "W_EE[IO]": W_EE[t.tensor(ioi_dataset.io_tokenIDs)],
            "W_EE_subE[IO]": W_EE_subE[t.tensor(ioi_dataset.io_tokenIDs)],
            "No patching": ioi_cache["resid_pre", NNMH[0]][range(batch_size), ioi_dataset.word_idx["IO"]]
        }

        normalized_resid_vectors = {
            name: (k_side_vector - k_side_vector.mean(dim=-1, keepdim=True)) / k_side_vector.var(dim=-1, keepdim=True).pow(0.5)
            for name, k_side_vector in resid_vectors.items()
        }

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
    subtract_S1_attn_scores: bool = False,
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
        If True, we use the effective embedding (only MLP output) rather than the actual output of MLP0. These shouldn't really be that different (if our W_EE is principled),
        but unfortunately they are.

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

    if decompose_by == "keys":

        assert intervene_on_query in [None, "sub_W_U_IO", "project_to_W_U_IO"]

        decomp_seq_pos_indices = ioi_dataset.word_idx["IO"]
        lin_map_seq_pos_indices = ioi_dataset.word_idx["end"]

        unembeddings = model.W_U.T[ioi_dataset.io_tokenIDs]
        unembeddings_scaled = (unembeddings - unembeddings.mean(-1, keepdim=True)) / unembeddings.var(dim=-1, keepdim=True).pow(0.5)

        resid_pre = ioi_cache["resid_pre", nnmh[0]]
        resid_pre_normalised = (resid_pre - resid_pre.mean(-1, keepdim=True)) / resid_pre.var(dim=-1, keepdim=True).pow(0.5)
        resid_pre_normalised_slice = resid_pre_normalised[range(batch_size), lin_map_seq_pos_indices]

        W_Q = model.W_Q[nnmh[0], nnmh[1]]
        b_Q = model.b_Q[nnmh[0], nnmh[1]]
        q_name = utils.get_act_name("q", nnmh[0])
        q_raw = ioi_cache[q_name].clone()
        
        # ! (1A)
        # * Get 2 linear functions from keys -> attn scores, corresponding to the 2 different components of query vectors: (∥ / ⟂) to W_U[IO]
        if intervene_on_query == "project_to_W_U_IO":
            resid_pre_in_io_dir, resid_pre_in_io_perpdir = project(resid_pre_normalised_slice, unembeddings)

            # Overwrite the query-side vector in the cache with the projection in the unembedding direction
            q_new = einops.einsum(resid_pre_in_io_dir, W_Q, "batch d_model, d_model d_head -> batch d_head")
            q_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = q_new
            ioi_cache_dict_io_dir = {**ioi_cache.cache_dict, **{q_name: q_raw.clone()}}
            ioi_cache_io_dir = ActivationCache(cache_dict=ioi_cache_dict_io_dir, model=model)
            linear_map_io_dir, bias_term_io_dir = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_io_dir, ioi_dataset=ioi_dataset)

            # Overwrite the query-side vector with the bit that's perpendicular to the IO unembedding (plus the bias term)
            q_new = einops.einsum(resid_pre_in_io_perpdir, W_Q, "batch d_model, d_model d_head -> batch d_head") + b_Q
            q_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = q_new
            ioi_cache_dict_io_perpdir = {**ioi_cache.cache_dict, **{q_name: q_raw.clone()}}
            ioi_cache_io_perpdir = ActivationCache(cache_dict=ioi_cache_dict_io_perpdir, model=model)
            linear_map_io_perpdir, bias_term_io_perpdir = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_io_perpdir, ioi_dataset=ioi_dataset)
            
            linear_map_dict = {"IO_dir": (linear_map_io_dir, bias_term_io_dir), "IO_perp": (linear_map_io_perpdir, bias_term_io_perpdir)}

        # ! (1A)
        # * Get new linear function from keys -> attn scores, corresponding to subbing in W_U[IO] as queryside vector
        elif intervene_on_query == "sub_W_U_IO":
            # Overwrite the query-side vector by replacing it with the (normalized) W_U[IO] unembeddings
            q_new = einops.einsum(unembeddings_scaled, W_Q, "batch d_model, d_model d_head -> batch d_head") + b_Q
            q_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = q_new
            ioi_cache_dict_io_subbed = {**ioi_cache.cache_dict, **{q_name: q_raw.clone()}}
            ioi_cache_io_subbed = ActivationCache(cache_dict=ioi_cache_dict_io_subbed, model=model)
            linear_map_io_subbed, bias_term_io_subbed = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_io_subbed, ioi_dataset=ioi_dataset)
            
            linear_map_dict = {"IO_sub": (linear_map_io_subbed, bias_term_io_subbed)}

        # * Get linear function from keys -> attn scores (no intervention on query)
        else:
            linear_map, bias_term = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset)
            linear_map_dict = {"unchanged": (linear_map, bias_term)}


    
    elif (decompose_by == "queries"):

        assert intervene_on_key in [None, "sub_MLP0", "project_to_MLP0"]

        decomp_seq_pos_indices = ioi_dataset.word_idx["end"]
        lin_map_seq_pos_indices = ioi_dataset.word_idx["IO"]

        if use_effective_embedding:
            effective_embeddings = get_effective_embedding(model) 
            # W_U = effective_embeddings["W_U (or W_E, no MLPs)"]
            # W_EE = effective_embeddings["W_E (including MLPs)"]
            W_EE_subE = effective_embeddings["W_E (only MLPs)"]
            MLP0_output = W_EE_subE[ioi_dataset.io_tokenIDs]
        else:
            MLP0_output = ioi_cache["mlp_out", 0][range(batch_size), lin_map_seq_pos_indices]
        MLP0_output_scaled = (MLP0_output - MLP0_output.mean(-1, keepdim=True)) / MLP0_output.var(dim=-1, keepdim=True).pow(0.5)

        resid_pre = ioi_cache["resid_pre", nnmh[0]]
        resid_pre_normalised = (resid_pre - resid_pre.mean(-1, keepdim=True)) / resid_pre.var(dim=-1, keepdim=True).pow(0.5)
        resid_pre_normalised_slice = resid_pre_normalised[range(batch_size), lin_map_seq_pos_indices]

        W_K = model.W_K[nnmh[0], nnmh[1]]
        b_K = model.b_K[nnmh[0], nnmh[1]]
        k_name = utils.get_act_name("k", nnmh[0])
        k_raw = ioi_cache[k_name].clone()
        
        # ! (2B)
        # * Get 2 linear functions from queries -> attn scores, corresponding to the 2 different components of key vectors: (∥ / ⟂) to MLP0_out
        if intervene_on_key == "project_to_MLP0":
            resid_pre_in_mlp0_dir, resid_pre_in_mlp0_perpdir = project(resid_pre_normalised_slice, MLP0_output)

            # Overwrite the key-side vector in the cache with the projection in the MLP0_output direction
            k_new = einops.einsum(resid_pre_in_mlp0_dir, W_K, "batch d_model, d_model d_head -> batch d_head")
            k_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = k_new
            ioi_cache_dict_mlp0_dir = {**ioi_cache.cache_dict, **{k_name: k_raw.clone()}}
            ioi_cache_mlp0_dir = ActivationCache(cache_dict=ioi_cache_dict_mlp0_dir, model=model)
            # ! (3B)
            # * This function (the `subtract_S1_attn_scores` argument) is where we subtract the S1-baseline from the linear map from queries -> attn scores))
            # * Obviously the same is true for the other 3 instances of the `attn_scores_as_linear_func_of_queries` function below
            linear_map_mlp0_dir, bias_term_mlp0_dir = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_mlp0_dir, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)

            # Overwrite the key-side vector with the bit that's perpendicular to the MLP0_output (plus the bias term)
            k_new = einops.einsum(resid_pre_in_mlp0_perpdir, W_K, "batch d_model, d_model d_head -> batch d_head") + b_K
            k_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = k_new
            ioi_cache_dict_mlp0_perpdir = {**ioi_cache.cache_dict, **{k_name: k_raw.clone()}}
            ioi_cache_mlp0_perpdir = ActivationCache(cache_dict=ioi_cache_dict_mlp0_perpdir, model=model)
            linear_map_mlp0_perpdir, bias_term_mlp0_perpdir = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache_mlp0_perpdir, ioi_dataset=ioi_dataset, subtract_S1_attn_scores=subtract_S1_attn_scores)
            
            linear_map_dict = {"MLP0_dir": (linear_map_mlp0_dir, bias_term_mlp0_dir), "MLP0_perp": (linear_map_mlp0_perpdir, bias_term_mlp0_perpdir)}
        
        # ! (2B)
        # * Get new linear function from queries -> attn scores, corresponding to subbing in MLP0_output as keyside vector
        elif intervene_on_key == "sub_MLP0":
            # Overwrite the key-side vector by replacing it with the (normalized) MLP0_output
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
        assert intervene_on_key == "project_to_MLP0", "If you're decomposing by key component, 'sub_MLP0' is invalid. Use 'project_to_MLP0' or None instead."
        # MLP0_output = get_effective_embedding(model)["W_E (only MLPs)"]
        # MLP0_output = MLP0_output[ioi_dataset.io_tokenIDs] # shape (batch, d_model)
        if use_effective_embedding:
            effective_embeddings = get_effective_embedding(model) 
            W_EE_subE = effective_embeddings["W_E (only MLPs)"]
            MLP0_output = W_EE_subE[ioi_dataset.io_tokenIDs]
        else:
            MLP0_output = ioi_cache["mlp_out", 0][range(batch_size), decomp_seq_pos_indices]
        contribution_to_attn_scores_shape = (2, 1 + nnmh[0], model.cfg.n_heads + 1)

    # * This is where we get the thing we're projecting queries onto if required (i.e. if we're decomposing by queries, and want to split into ||W_U[IO] and ⟂W_U[IO])
    elif (intervene_on_query is not None) and (decompose_by == "queries"):
        assert intervene_on_query == "project_to_W_U_IO", "If you're decomposing by key component, 'sub_W_U_IO' is invalid. Use 'project_to_W_U_IO' or None instead."
        unembeddings = model.W_U.T[ioi_dataset.io_tokenIDs]
        contribution_to_attn_scores_shape = (2, 1 + nnmh[0], model.cfg.n_heads + 1)

    # * We're not projecting by anything when we get the decomposed bits
    else:
        contribution_to_attn_scores_shape = (1, 1 + nnmh[0], model.cfg.n_heads + 1)



    def get_decomposed_components(component_name, layer=None):
        '''
        This function does the following:
            > Get the value we want from the ioi_cache (at the appopriate sequence positions for the decomposition: either "IO" or "end")
            > If we need to project it in a direction, then apply that projection (this gives it an extra dim at the start)
            > If we need to subtract the mean of S1, do that too.
        '''
        assert component_name in ["result", "mlp_out", "embed", "pos_embed"]
        assert isinstance(ln_scale, Float[Tensor, "batch 1"])
        
        # Index from ioi cache
        component_output: Float[Tensor, "batch *n_heads d_model"] = ioi_cache[component_name, layer][range(batch_size), decomp_seq_pos_indices]

        # Subtract baseline
        # ! (3A)
        # * This is where we subtract the keyside component baseline of S2 (if our decomposition is by-keys)
        if (decompose_by == "keys") and subtract_S1_attn_scores:
            component_output_S1: Float[Tensor, "batch *n_heads d_model"] = ioi_cache[component_name, layer][range(batch_size), S1_seq_pos_indices]
            component_output = component_output - component_output_S1

        # Apply scaling
        component_output_scaled = component_output / (ln_scale.unsqueeze(1) if (component_name == "result") else ln_scale)

        # Apply projections
        # ! (2A)
        # * This is where we decompose the query-side output of each component, by possibly projecting it onto the ||W_U[IO] and ⟂W_U[IO] directions
        if (decompose_by == "queries") and (intervene_on_query == "project_to_W_U_IO"):
            projection_dir = einops.repeat(unembeddings, "b d_m -> b heads d_m", heads=model.cfg.n_heads) if (component_name == "result") else unembeddings
            component_output_scaled = t.stack(project(component_output_scaled, projection_dir))
        # ! (1B)
        # * This is where we decompose the key-side output of each component, by possibly projecting it onto the ||MLP0 and ⟂MLP0 directions
        elif (decompose_by == "keys") and (intervene_on_key == "project_to_MLP0"):
            projection_dir = einops.repeat(MLP0_output, "b d_m -> b heads d_m", heads=model.cfg.n_heads) if (component_name == "result") else MLP0_output
            component_output_scaled = t.stack(project(component_output_scaled, projection_dir))

        return component_output_scaled



    for name, (linear_map, bias_term) in linear_map_dict.items():
        
        # Check linear map is valid
        assert linear_map.shape == (batch_size, model.cfg.d_model)
        assert bias_term.shape == (batch_size,)

        # Create tensor to store all the values for this facet plot (possibly 2 facet plots, if we're splitting by projecting our decomposed components)
        contribution_to_attn_scores = t.zeros(contribution_to_attn_scores_shape)

        # Get scale factor we'll be dividing all our components by
        ln_scale = ioi_cache["scale", nnmh[0], "ln1"][range(batch_size), decomp_seq_pos_indices, nnmh[1]]
        assert ln_scale.shape == (batch_size, 1)

        # We start with all the things before attn heads and MLPs
        embed_scaled = get_decomposed_components("embed")
        pos_embed_scaled = get_decomposed_components("pos_embed")
        # Add these to the results tensor. Note we use `:` because this covers cases where the first dim is 1 (no projection split) or 2 (projection split)
        contribution_to_attn_scores[:, 0, 0] = einops.einsum(embed_scaled, linear_map, "... batch d_model, batch d_model -> ... batch").mean(-1)
        contribution_to_attn_scores[:, 0, 1] = einops.einsum(pos_embed_scaled, linear_map, "... batch d_model, batch d_model -> ... batch").mean(-1)
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

    if len(contribution_to_attn_scores_list) == 1:
        contribution_to_attn_scores = contribution_to_attn_scores_list[0]
    else:
        contribution_to_attn_scores = t.stack(contribution_to_attn_scores_list)

    if show_plot:
        plot_contribution_to_attn_scores(contribution_to_attn_scores, decompose_by, static=static)
    
    return contribution_to_attn_scores





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
    x: Float[Tensor, "... d"],
    dir: Float[Tensor, "... d"],
):
    '''
    x: 
        Shape (*batch_dims, d)
        Batch of vectors
    
    dir:
        Shape (*batch_dims, d)
        Batch of vectors (which will be normalized)

    Returns:
        Two batches of vectors: x_dir and x_perp, such that:
            x_dir + x_perp = x
            x_dir is the component of x in the direction of dir

    Notes:
        Make sure x and dir have the same shape, I don't want to
        mess up broadcasting by accident! Do einops.repeat on dir
        if you have to.
    '''
    assert dir.shape == x.shape

    dir_normed = dir / dir.norm(dim=-1, keepdim=True)

    x_component = einops.einsum(
        x, dir_normed,
        "... d, ... d -> ..."
    )

    x_dir = einops.einsum(
        x_component, dir_normed,
        "..., ... d -> ... d"
    )

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
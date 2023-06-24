from transformer_lens.cautils.utils import *
from transformer_lens.rs.callum.generate_bag_of_words_quad_plot import get_effective_embedding



def attn_scores_as_linear_func_of_queries(
    batch_idx: Optional[Union[int, List[int], Int[Tensor, "batch"]]],
    head: Tuple[int, int],
    model: HookedTransformer,
    ioi_cache: ActivationCache,
    ioi_dataset: IOIDataset,
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

    keys = ioi_cache["k", layer][:, :, head_idx] # shape (all_batch, seq_K, d_head)
    keys_at_IO = keys[batch_idx, ioi_dataset.word_idx["IO"][batch_idx]] # shape (batch, d_head)
    
    W_Q = model.W_Q[layer, head_idx].clone() # shape (d_model, d_head)
    b_Q = model.b_Q[layer, head_idx].clone() # shape (d_head,)

    linear_map = einops.einsum(W_Q, keys_at_IO, "d_model d_head, batch d_head -> batch d_model") / (model.cfg.d_head ** 0.5)
    bias_term = einops.einsum(b_Q, keys_at_IO, "d_head, batch d_head -> batch") / (model.cfg.d_head ** 0.5)

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

        ioi_dataset, abc_dataset, ioi_cache, abc_cache, ioi_metric_noising = generate_data_and_caches(batch_size, model=model, seed=seed)

        linear_map, bias_term = attn_scores_as_linear_func_of_queries(batch_idx=None, head=NNMH, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset)
        assert linear_map.shape == (batch_size, model.cfg.d_model)

        # Has to be manual, because apparently `apply_ln_to_stack` doesn't allow it to be applied at different sequence positions
        # ! Note - I don't actually have to do this if I'm computing cosine similarity! Maybe I should be doing this instead?
        resid_vectors = {
            "W_U[IO]": model.W_U.T[t.tensor(ioi_dataset.io_tokenIDs)],
            "W_U[S]": model.W_U.T[t.tensor(ioi_dataset.s_tokenIDs)],
            "W_U[random]": model.W_U.T[t.randint(size=(batch_size,), low=0, high=model.cfg.d_vocab)],
            "W_U[random name]": model.W_U.T[np.random.choice(name_tokens, size=(batch_size,))],
            "NMH 9.9 output": einops.einsum(ioi_cache["z", 9][range(batch_size), ioi_dataset.word_idx["end"], 9], model.W_O[9, 9], "batch d_head, d_head d_model -> batch d_model"),
            "No patching": ioi_cache["resid_pre", NNMH[0]][range(batch_size), ioi_dataset.word_idx["end"]]
        }

        normalized_resid_vectors = {
            name: q_side_vector / q_side_vector.var(dim=-1, keepdim=True).pow(0.5)
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

        ioi_dataset, abc_dataset, ioi_cache, abc_cache, ioi_metric_noising = generate_data_and_caches(batch_size, model=model, seed=seed)

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
            name: k_side_vector / k_side_vector.var(dim=-1, keepdim=True).pow(0.5)
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
    as_function_of: str,
    show_plot: bool = False,
    sub_unembedding: bool = False, # substitute unembeddings into the query-side vector
    split_query_by_unembedding: bool = False, # get 2 plots, for unembedding projection on query-side, and orthogonal component
    split_key_by_MLP0: bool = False, # also get 2 plots, for the component in MLP0 direction vs perp to this
):
    t.cuda.empty_cache()
    ioi_dataset, abc_dataset, ioi_cache, abc_cache, ioi_metric_noising = generate_data_and_caches(batch_size, model=model, seed=seed)

    seq_pos_indices = ioi_dataset.word_idx["IO"] if as_function_of == "keys" else ioi_dataset.word_idx["end"]

    assert not(sub_unembedding and split_query_by_unembedding), "You can't do both at the same time."

    if sub_unembedding or split_query_by_unembedding:
        assert as_function_of == "keys", "You should only be substituting unembeddings into the query-side vector."

        unembeddings = model.W_U.T[ioi_dataset.io_tokenIDs]
        unembeddings_scaled = unembeddings / unembeddings.var(dim=-1, keepdim=True).pow(0.5)

        resid_pre = ioi_cache["resid_pre", nnmh[0]]
        resid_pre_normalised = resid_pre / resid_pre.var(dim=-1, keepdim=True).pow(0.5)
        resid_pre_normalised_slice = resid_pre_normalised[range(batch_size), ioi_dataset.word_idx["end"]]

        W_Q = model.W_Q[nnmh[0], nnmh[1]]
        b_Q = model.b_Q[nnmh[0], nnmh[1]]
        q_name = utils.get_act_name("q", nnmh[0])
        q_raw = ioi_cache[q_name].clone()
        
        if split_query_by_unembedding:
            resid_pre_in_io_dir, resid_pre_in_io_perpdir = project(resid_pre_normalised_slice, unembeddings)

            # Overwrite the query-side vector in the cache with the projection in the unembedding direction
            q_new = einops.einsum(resid_pre_in_io_dir, W_Q, "batch d_model, d_model d_head -> batch d_head")
            q_raw[range(batch_size), ioi_dataset.word_idx["end"], nnmh[1]] = q_new
            ioi_cache_dict_io_dir = {**ioi_cache.cache_dict, **{q_name: q_raw.clone()}}
            ioi_cache_io_dir = ActivationCache(cache_dict=ioi_cache_dict_io_dir, model=model)

            # Overwrite the query-side vector with the bit that's perpendicular to the IO unembedding (plus the bias term)
            q_new = einops.einsum(resid_pre_in_io_perpdir, W_Q, "batch d_model, d_model d_head -> batch d_head") + b_Q
            q_raw[range(batch_size), ioi_dataset.word_idx["end"], nnmh[1]] = q_new
            ioi_cache_dict_io_perpdir = {**ioi_cache.cache_dict, **{q_name: q_raw.clone()}}
            ioi_cache_io_perpdir = ActivationCache(cache_dict=ioi_cache_dict_io_perpdir, model=model)
            
            ioi_cache_list = [ioi_cache_io_dir, ioi_cache_io_perpdir]
        else:
            q_new = einops.einsum(unembeddings_scaled, W_Q, "batch d_model, d_model d_head -> batch d_head") + b_Q
            q_raw[range(batch_size), ioi_dataset.word_idx["end"], nnmh[1]] = q_new
            ioi_cache_dict_io_replaced = {**ioi_cache.cache_dict, **{q_name: q_raw.clone()}}
            ioi_cache_io_replaced = ActivationCache(cache_dict=ioi_cache_dict_io_replaced, model=model)
            
            ioi_cache_list = [ioi_cache_io_replaced]

    else:
        ioi_cache_list = [ioi_cache]

    t.cuda.empty_cache()

    contribution_to_attn_scores_list = []
    contribution_to_attn_scores_shape = (1, 1 + nnmh[0], model.cfg.n_heads + 1)

    if split_key_by_MLP0:
        # MLP0_output = get_effective_embedding(model)["W_E (only MLPs)"]
        # MLP0_output = MLP0_output[ioi_dataset.io_tokenIDs] # shape (batch, d_model)

        MLP0_output = ioi_cache["mlp_out", 0][range(batch_size), seq_pos_indices]

        contribution_to_attn_scores_shape = (2, 1 + nnmh[0], model.cfg.n_heads + 1)

    for ioi_cache in ioi_cache_list:

        if as_function_of == "keys":
            linear_map, bias_term = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset)
        elif as_function_of == "queries":
            linear_map, bias_term = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, ioi_cache=ioi_cache, ioi_dataset=ioi_dataset)
        else:
            raise Exception("as_function_of must be one of ['k', 'keys', 'q', 'queries']")

        assert linear_map.shape == (batch_size, model.cfg.d_model)
        assert bias_term.shape == (batch_size,)

        contribution_to_attn_scores = t.zeros(contribution_to_attn_scores_shape)

        ln_scale = ioi_cache["scale", nnmh[0], "ln1"][range(batch_size), seq_pos_indices, nnmh[1]]

        # bit hacky - having a zeroth layer for the embedding, and just putting it at zeroth column
        embed_scaled = ioi_cache["embed"][range(batch_size), seq_pos_indices] / ln_scale
        pos_embed_scaled = ioi_cache["pos_embed"][range(batch_size), seq_pos_indices] / ln_scale
        if split_key_by_MLP0:
            embed_scaled = t.stack(project(embed_scaled, MLP0_output))
            pos_embed_scaled = t.stack(project(pos_embed_scaled, MLP0_output))
        contribution_to_attn_scores[:, 0, 0] = einops.einsum(embed_scaled, linear_map, "... batch d_model, batch d_model -> ... batch").mean(-1)
        contribution_to_attn_scores[:, 0, 1] = einops.einsum(pos_embed_scaled, linear_map, "... batch d_model, batch d_model -> ... batch").mean(-1)
        contribution_to_attn_scores[-1, 0, 2] = bias_term.mean()

        for layer in range(nnmh[0]):

            # Calculate output of each attention head, split by projecting onto MLP0 output if necessary, then add to our results tensor
            z = ioi_cache["z", layer][range(batch_size), seq_pos_indices]
            result = einops.einsum(z, model.W_O[layer], "batch n_heads d_head, n_heads d_head d_model -> batch n_heads d_model")
            result_scaled = result / ln_scale.unsqueeze(1)

            if split_key_by_MLP0:
                MLP0_output_repeated = einops.repeat(MLP0_output, "batch d_model -> batch n_heads d_model", n_heads=model.cfg.n_heads)
                result_scaled = t.stack(project(result_scaled, MLP0_output_repeated))
            contribution_to_attn_scores[:, 1 + layer, :model.cfg.n_heads] = einops.einsum(
                result_scaled, linear_map, 
                "... batch n_heads d_model, batch d_model -> ... n_heads batch"
            ).mean(-1)

            # Calculate output of the MLPs, split by projecting onto MLP0 output if necessary, then add to our results tensor
            mlp_out = ioi_cache["mlp_out", layer][range(batch_size), seq_pos_indices]
            assert mlp_out.shape == (batch_size, model.cfg.d_model)
            mlp_out_scaled = mlp_out / ln_scale
            if split_key_by_MLP0:
                mlp_out_scaled = t.stack(project(mlp_out_scaled, MLP0_output))
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
        plot_contribution_to_attn_scores(contribution_to_attn_scores, as_function_of)
    
    return contribution_to_attn_scores



def plot_contribution_to_attn_scores(
    contribution_to_attn_scores: Float[Tensor, "... layer component"],
    as_function_of: str,
    facet_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    text = [["W<sub>E</sub>", "W<sub>pos</sub>", "b<sub>K</sub>" if as_function_of == "keys" else "b<sub>Q</sub>"] + ["" for _ in range(10)]]
    for layer in range(0, 10):
        text.append([f"{layer}.{head}" for head in range(12)] + [f"MLP{layer}"])

    has_3_dims = contribution_to_attn_scores.ndim >= 3
    has_4_dims = has_3_dims and contribution_to_attn_scores.shape[0] == 4
    fig = imshow(
        contribution_to_attn_scores,
        facet_col = 0 if has_3_dims else None,
        facet_labels = facet_labels if has_3_dims else None,
        facet_col_wrap = 2 if has_4_dims else None,
        title=f"Contribution to attention scores ({'key' if as_function_of in ['k', 'keys'] else 'query'}-side)" if title is None else title,
        labels={"x": "Component (attn heads & MLP)", "y": "Layer"},
        y=["misc"] + [str(i) for i in range(10)],
        x=[f"H{i}" for i in range(12)] + ["MLP"],
        border=True,
        width=1300 if has_3_dims else 900,
        height=1100 if has_4_dims else 600,
        return_fig=True,
    )
    for i in range(len(fig.data)):
        fig.data[i].update(
            text=text, 
            texttemplate="%{text}", 
            textfont={"size": 12}
        )
    fig.show()







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
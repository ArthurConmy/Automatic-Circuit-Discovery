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
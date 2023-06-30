from transformer_lens.cautils.utils import *



def lock_attn(
    attn_patterns: Float[t.Tensor, "batch head_idx dest_pos src_pos"],
    hook: HookPoint,
    ablate: bool = False,
) -> Float[t.Tensor, "batch head_idx dest_pos src_pos"]:
    
    assert isinstance(attn_patterns, Float[t.Tensor, "batch head_idx dest_pos src_pos"])
    assert hook.layer() == 0

    batch, n_heads, seq_len = attn_patterns.shape[:3]
    attn_new = einops.repeat(t.eye(seq_len), "dest src -> batch head_idx dest src", batch=batch, head_idx=n_heads).clone().to(attn_patterns.device)
    if ablate:
        attn_new = attn_new * 0
    return attn_new

def fwd_pass_lock_attn0_to_self(
    model: HookedTransformer,
    input: Union[List[str], Int[t.Tensor, "batch seq_pos"]],
    ablate: bool = False,
) -> Float[t.Tensor, "batch seq_pos d_vocab"]:

    model.reset_hooks()
    
    loss = model.run_with_hooks(
        input,
        return_type="loss",
        fwd_hooks=[(utils.get_act_name("pattern", 0), partial(lock_attn, ablate=ablate))],
    )

    return loss



def get_effective_embedding(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:

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
        "W_U (or W_E, no MLPs)": W_U.T,
        # "W_E (raw, no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE
    }


def get_EE_QK_circuit(
    layer_idx,
    head_idx,
    model: HookedTransformer,
    random_seeds: Optional[int] = 5,
    num_samples: Optional[int] = 500,
    bags_of_words: Optional[List[List[int]]] = None, # each List is a List of unique tokens
    mean_version: bool = True,
    show_plot: bool = False,
    W_E_query_side: Optional[t.Tensor] = None,
    query_side_bias: bool = False,
    W_E_key_side: Optional[t.Tensor] = None,
    key_side_bias: bool = False,
    apply_softmax: bool = True,
    apply_log_softmax: bool = False,
    norm = False,
    ten_x = False,
):
    assert (random_seeds is None and num_samples is None) != (bags_of_words is None), (random_seeds is None, num_samples is None, bags_of_words is None, "Must specify either random_seeds and num_samples or bag_of_words_version")

    if bags_of_words is not None:
        random_seeds = len(bags_of_words) # eh not quite random seeds but whatever
        assert all([len(bag_of_words) == len(bags_of_words[0])] for bag_of_words in bags_of_words), "Must have same number of words in each bag of words"
        num_samples = len(bags_of_words[0])

    W_Q_head = model.W_Q[layer_idx, head_idx]
    W_K_head = model.W_K[layer_idx, head_idx]

    assert W_E_query_side is not None
    assert W_E_key_side is not None
    W_E_Q_normed = W_E_query_side 
    W_E_K_normed = W_E_key_side
    if norm:
        if norm: 
            W_E_Q_normed /= W_E_query_side.var(dim=-1, keepdim=True).pow(0.5)
        if norm:
            W_E_K_normed /= W_E_key_side / W_E_key_side.var(dim=-1, keepdim=True).pow(0.5)

    if ten_x:
        W_E_Q_normed *= 10

    EE_QK_circuit = FactoredMatrix.FactoredMatrix(W_E_Q_normed @ W_Q_head, W_K_head.T @ W_E_K_normed.T)
    EE_QK_circuit_result = t.zeros((num_samples, num_samples))

    for random_seed in range(random_seeds):
        if bags_of_words is None:
            indices = t.randint(0, model.cfg.d_vocab, (num_samples,))
        else:
            indices = t.tensor(bags_of_words[random_seed])

        # assert False, "TODO: add Q and K and V biases???"
        EE_QK_circuit_sample = einops.einsum(
            EE_QK_circuit.A[indices, :] + (0.0 if query_side_bias else model.b_Q[layer_idx, head_idx]),
            EE_QK_circuit.B[:, indices] + (0.0 if key_side_bias else model.b_K[layer_idx, head_idx]),
            "num_query_samples d_head, d_head num_key_samples -> num_query_samples num_key_samples"
        ) / model.cfg.d_head ** 0.5

        if mean_version:
            # we're going to take a softmax so the constant factor is arbitrary 
            # and it's a good idea to centre all these results so adding them up is reasonable
            EE_QK_mean = EE_QK_circuit_sample.mean(dim=-1, keepdim=True)
            EE_QK_circuit_sample_centered = EE_QK_circuit_sample - EE_QK_mean 
            EE_QK_circuit_result += EE_QK_circuit_sample_centered.cpu()

        else:
            if apply_softmax or apply_log_softmax:
                EE_QK_softmax = t.nn.functional.softmax(EE_QK_circuit_sample, dim=-1)
                if apply_log_softmax:
                    EE_QK_softmax = t.log(EE_QK_softmax)
                EE_QK_circuit_result += EE_QK_softmax.cpu()
            else:
                EE_QK_circuit_result += EE_QK_circuit_sample.cpu()

    EE_QK_circuit_result /= random_seeds

    if show_plot:
        imshow(
            EE_QK_circuit_result,
            labels={"x": "Source/Key Token (embedding)", "y": "Destination/Query Token (unembedding)"},
            title=f"EE QK circuit for head {layer_idx}.{head_idx}",
            width=700,
        )

    return EE_QK_circuit_result
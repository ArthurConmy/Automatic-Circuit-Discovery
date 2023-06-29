#%% [markdown] [4]:

from transformer_lens.cautils.notebook import *

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
# model.set_use_split_qkv_input(True)

clear_output()

# full_data = get_webtext()
# TOTAL_OWT_SAMPLES = 100
# SEQ_LEN = 20
# data = full_data[:TOTAL_OWT_SAMPLES]

from transformer_lens import FactoredMatrix

# TODO legend groups

#%% [markdown] [5]:

def get_effective_embedding(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:

    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    # t.testing.assert_close(W_E[:10, :10], W_U[:10, :10].T)  NOT TRUE, because of the center unembed part!

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
    
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()

    return {
        "W_U (or W_E, no MLPs)": W_U.T,
        # "W_E (raw, no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE
    }

embeddings_dict = get_effective_embedding(model)

# $$
# W_E^Q W_Q W_K^T W_E^K
# $$

#%% [markdown]
# <p> Grab some webtext bag of words <p>

dataset = get_webtext()

#%%

bags_of_words = []
OUTER_LEN = 10
INNER_LEN = 10

dataset = list({
    " John": " John was reading a book when suddenly John heard a strange noise",
    " Maria": " Maria loves playing the piano and, moreover Maria also enjoys painting",
    " city": " The city was full of lights, making the city look like a sparkling diamond",
    " ball": " The ball rolled away, so the dog chased the ball all the way to the park",
    " Python": " Python is a popular language for programming. In fact, Python is known for its simplicity",
    " President": " The President announced new policies today. Many are waiting to see how the President's decisions will affect the economy",
    " Bitcoin": " Bitcoin's value has been increasing rapidly. Investors are closely watching Bitcoin's performance",
    " dog": " The dog wagged its tail happily. Seeing the dog so excited, the children started laughing",
    " cake": " The cake looked delicious. Everyone at the party was eager to taste the cake today",
    " book": " The book was so captivating, I couldn't put the book down",
    " house": " The house was quiet. Suddenly, a noise from the upstair s of the house startled everyone",
    " car": " The car pulled into the driveway. Everyone rushed out to see the new car today",
    " computer": " The computer screen flickered. She rebooted the computer hoping to resolve the issue",
    " key": " She lost the key to her apartment. She panicked when she realized she had misplaced the key today",
    " apple": " He took a bite of the apple. The apple was crisp and delicious",
    " phone": " The phone rang in the middle of the night. She picked up the phone with a groggy hello",
    " train": " The train was late. The passengers were annoyed because the train was delayed by an hour",
}.values())

idx = -1
while len(bags_of_words) < OUTER_LEN:
    idx+=1
    cur_tokens = model.to_tokens(dataset[idx]).tolist()[0][1:]

    cur_bag = []
    
    for i in range(len(cur_tokens)):
        if len(cur_bag) == INNER_LEN:
            break
        if cur_tokens[i] not in cur_bag:
            cur_bag.append(cur_tokens[i])

    if len(cur_bag) == INNER_LEN:
        bags_of_words.append(cur_bag)


#%% [markdown] [6]:

def plot_random_sample(
    embeddings_dict: Dict[str, Float[Tensor, "d_vocab d_model"]],
    model: HookedTransformer = model,
    sample_size: Optional[int] = None,
    bags_of_words: Optional[List[List[int]]] = None,
    num_batches: int = 1,
    head: Tuple[int, int] = (10, 7)
):
    """Specify sample size XOR bags_of_words"""
    assert (sample_size is None) != (bags_of_words is None), (sample_size, bags_of_words)

    if bags_of_words is not None:
        assert len(bags_of_words) == num_batches, (len(bags_of_words), num_batches)
        sample_size = len(bags_of_words[0])
        assert all([len(bag) == sample_size for bag in bags_of_words]), [len(bag) for bag in bags_of_words]

    results_for_each_batch = []        
    sorted_keys = sorted(embeddings_dict.keys())

    W_Q = model.W_Q[head[0], head[1]]
    W_K = model.W_K[head[0], head[1]]

    embeddings_dict_normalized = {k: v / (v.var(dim=-1, keepdim=True)+model.cfg.eps).pow(0.5) for k, v in embeddings_dict.items()}

    q_and_k_labels = [(q_name, k_name) for q_name in sorted_keys for k_name in sorted_keys]
    q_and_k_matrices = [(embeddings_dict_normalized[q_name], embeddings_dict_normalized[k_name]) for (q_name, k_name) in q_and_k_labels]

    # print("Doing a super slow ~30second assert (you may wish to comment out, but probably check this at least once...")
    # for q_matrix, k_matrix in q_and_k_matrices:
    #     assert list(q_matrix.shape) == [model.cfg.d_vocab, model.cfg.d_model], q_matrix.shape
    #     assert list(k_matrix.shape) == [model.cfg.d_vocab, model.cfg.d_model], k_matrix.shape
    #     for q_m in q_matrix:
    #         assert abs(q_m.norm().item() - model.cfg.d_model**0.5) < 1e-1, (q_m.shape, q_m.norm().item(), model.cfg.d_model**0.5)
    #     for k_m in k_matrix:
    #         assert abs(k_m.norm().item() - model.cfg.d_model**0.5) < 1e-1, (k_m.shape, k_m.norm().item(), model.cfg.d_model**0.5)
    # print("... assert done!")

    for batch_idx in range(num_batches):
        results = []
        if bags_of_words is not None:
            sample_indices = t.tensor(bags_of_words[batch_idx])
        else:
            sample_indices = t.randint(0, model.cfg.d_vocab, (sample_size,))
        for q_matrix, k_matrix in q_and_k_matrices:
            full_matrix = FactoredMatrix(q_matrix @ W_Q, W_K.T @ k_matrix.T)
            query_side_vector = full_matrix.A[sample_indices, :] + model.b_Q[head[0], head[1]] # TODO to do this addition maximally safe, assert some shapes and/or einops.repeat the bias
            key_side_vector = full_matrix.B[:, sample_indices] + model.b_K[head[0], head[1]].unsqueeze(-1)
            
            assert list(query_side_vector.shape) == [sample_size, model.cfg.d_head], query_side_vector.shape
            assert list(key_side_vector.shape) == [model.cfg.d_head, sample_size], key_side_vector.shape

            attention_scores = einops.einsum(
                query_side_vector,
                key_side_vector,
                "query d_head, d_head key -> query key",
            ) / (model.cfg.d_head ** 0.5)

            results.append(attention_scores)

            curt = einops.einsum

        results_for_each_batch.append(t.stack(results, dim=0))
    results = sum(results_for_each_batch) / len(results_for_each_batch) # oh right, sum over the list means adding tensors

    imshow(
        results,
        facet_col=0,
        facet_col_wrap=len(embeddings_dict),
        facet_labels=[f"Q = {q_name}<br>K = {k_name}" for (q_name, k_name) in q_and_k_labels],
        title=f"Sample of diagonal attention score for different matrices: head {head}",
        labels={"x": "Key", "y": "Query"},
        height=900, width=900
    )
    results_trace = results[:, range(sample_size), range(sample_size)].mean(-1).reshape((len(sorted_keys), len(sorted_keys)))

    # # this plot doesn't really work when we plot attention scores
    # imshow(
    #     1 / (1 - results_trace),
    #     x = sorted_keys,
    #     y = sorted_keys,
    #     title=f"1 / (1 - avg_trace) for {head} (to make close to one blow up!)",
    #     labels={"x": "Key", "y": "Query"},
    #     height=500, width=600,
    # )

# plot_random_sample(embeddings_dict, head = (10, 7), sample_size = 100, num_batches = 20)
plot_random_sample(
    bags_of_words=bags_of_words,
    embeddings_dict=embeddings_dict,
    head=(10, 7),
    num_batches=len(bags_of_words),
)


#%% [markdown] [27]:

def get_scores_for_all_heads(
    embeddings_dict: Dict[str, Float[Tensor, "d_vocab d_model"]],
    model: HookedTransformer = model,
    sample_size: int = 50,
    num_batches: int = 1,
    include_W_E_raw: bool = True,
    plot_probs: bool = False,
):
    raise NotImplementedError("Implementation wrong, need to include biases to queries and keys, and also / (self.cfg.d_model**0.5)")

    # results = []

    # sorted_keys = sorted(embeddings_dict.keys())

    # W_Qs = model.W_Q
    # W_Ks = model.W_K

    # W_U_Q = embeddings_dict["W_U (or W_E, no MLPs)"]
    # W_U_Q_normed = W_U_Q / W_U_Q.var(dim=-1, keepdim=True).pow(0.5)
    # if include_W_E_raw:
    #     W_E_K = embeddings_dict["W_E (including MLPs)"]
    # else:
    #     W_E_K = embeddings_dict["W_E (only MLPs)"]
    # W_E_K_normed = W_E_K / W_E_K.var(dim=-1, keepdim=True).pow(0.5)

    # W_Q_full = einops.einsum(W_U_Q_normed, W_Qs, "d_vocab d_model, layer head d_model d_head -> layer head d_vocab d_head")
    # W_K_full = einops.einsum(W_E_K_normed, W_Ks, "d_vocab d_model, layer head d_model d_head -> layer head d_vocab d_head")

    # W_QK_full = FactoredMatrix(W_Q_full, W_K_full.transpose(-1, -2))

    # for _ in range(num_batches):
    #     sample_indices = t.randint(0, model.cfg.d_vocab, (sample_size,))
    #     W_QK_sample = W_QK_full.A[..., sample_indices, :] @ W_QK_full.B[..., :, sample_indices]
    #     W_QK_sample = W_QK_sample - W_QK_sample.mean(dim=-1, keepdim=True)

    #     if plot_probs:
    #         W_QK_softmaxed = W_QK_sample.softmax(dim=-1)
    #         W_QK_avg_diag_prob = W_QK_softmaxed[..., range(sample_size), range(sample_size)].mean(-1)
    #         results.append(W_QK_avg_diag_prob)
    #     else:
    #         W_QK_diag_sum = W_QK_sample[..., range(sample_size), range(sample_size)].sum(-1)
    #         W_QK_offdiag_sum = W_QK_sample.sum(dim=(-1, -2)) - W_QK_diag_sum
    #         W_QK_avg_diag = W_QK_diag_sum / sample_size
    #         W_QK_avg_offdiag = W_QK_offdiag_sum / sample_size
    #         results.append(W_QK_avg_diag - W_QK_avg_offdiag)


    # results = sum(results) / len(results)
    # return results[1:]


results = get_scores_for_all_heads(embeddings_dict, sample_size = 250, num_batches = 40, plot_probs = True)

imshow(results, y=list(range(1, 12)), labels={"x": "Head", "y": "Layer"}, title="Prediction-attention scores (prob space, including MLP & W_E)", width=600)

#%% [markdown] [28]:

results = get_scores_for_all_heads(embeddings_dict, sample_size = 200, num_batches = 40, include_W_E_raw = False, plot_probs = True)

imshow(results, y=list(range(1, 12)), labels={"x": "Head", "y": "Layer"}, title="Prediction-attention scores (prob space, including MLP & not W_E)", width=600)

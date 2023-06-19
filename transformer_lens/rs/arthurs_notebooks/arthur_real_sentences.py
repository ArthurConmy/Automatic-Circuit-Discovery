#%% [markdown]
# <h1> Example notebook for the cautils import * statement </h1>

from transformer_lens.cautils.notebook import * # use from transformer_lens.cautils.utils import * instead to for the same effect without autoreload
DEVICE = "cuda"

#%%

# load a model
model = HookedTransformer.from_pretrained("gpt2-small").to(DEVICE)
data = get_webtext("test")

#%%

W_E = model.W_E.clone()
W_U = model.W_U.clone()

#%%

# Calculate W_{EE} edit
batch_size = 1000
nrows = model.cfg.d_vocab
W_EE = t.zeros((nrows, model.cfg.d_model)).to(DEVICE)

for i in tqdm(range(0, nrows + batch_size, batch_size)):
    cur_range = t.tensor(range(i, min(i + batch_size, nrows)))
    if len(cur_range)>0:
        embeds = W_E[cur_range].unsqueeze(0)
        pre_attention = model.blocks[0].ln1(embeds)
        post_attention = einops.einsum(
            pre_attention, 
            model.W_V[0],
            model.W_O[0],
            "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
        )
        normalized_resid_mid = model.blocks[0].ln2(post_attention + embeds)
        resid_post = model.blocks[0].mlp(normalized_resid_mid)
        W_EE[cur_range.to(DEVICE)] = resid_post

# %%

def prediction_attention_real_sentences(
    layer_idx,
    head_idx,
    tokens: Optional[List[List[int]]] = None, # each List is a List of unique tokens
    mean_version: bool = True,
    show_plot: bool = False,
):
# layer_idx = 10
# head_idx = 7
# # tokens = tokens
# mean_version = False
# show_plot = True

# if True:
    """Based off get_EE_QK_circuit from commit 4b32e53804764
    
    variable naming: maybe sentence_tokens should be called batch_tokens instead?"""

    random_seeds = len(tokens) # eh not quite random seeds but whatever
    assert all([len(sentence_tokens) == len(tokens[0])] for sentence_tokens in tokens), "Must have same number of tokens in each sentence"
    num_samples = len(tokens[0])

    W_Q_head = model.W_Q[layer_idx, head_idx]
    W_K_head = model.W_K[layer_idx, head_idx]

    EE_QK_circuit = transformer_lens.FactoredMatrix(W_U.T @ W_Q_head, W_K_head.T @ W_EE.T)
    EE_QK_circuit_result = t.zeros((num_samples, num_samples))

    
    n_layers, n_heads, d_model, d_head = model.W_Q.shape

    for random_seed in range(random_seeds):

        # Zero out the relevant things
        # (this deals with biases much more nicely too...)
        # Then get the attentions, then we're done
        # TODO speed this up by removing extra computation    

        sentence_tokens = t.tensor(tokens[random_seed])

        EE_QK_circuit_sample = einops.einsum(
            EE_QK_circuit.A[sentence_tokens, :],
            EE_QK_circuit.B[:, sentence_tokens],
            "num_query_samples d_head, d_head num_key_samples -> num_query_samples num_key_samples"
        ) / np.sqrt(d_head)

        if mean_version:
            # TODO check this is still reasonable...
            # we're going to take a softmax so the constant factor is arbitrary 
            # and it's a good idea to centre all these results so adding them up is reasonable
            EE_QK_mean = EE_QK_circuit_sample.mean(dim=1, keepdim=True)
            EE_QK_circuit_sample_centered = EE_QK_circuit_sample - EE_QK_mean 
            EE_QK_circuit_result += EE_QK_circuit_sample_centered.cpu()

        else:
            EE_QK_softmax = t.nn.functional.softmax(EE_QK_circuit_sample, dim=-1)
            EE_QK_circuit_result += EE_QK_softmax.cpu()

    EE_QK_circuit_result /= random_seeds

    if show_plot:
        imshow(
            EE_QK_circuit_result,
            labels={"x": "Source/Key Token (embedding)", "y": "Destination/Query Token (unembedding)"},
            title=f"EE QK circuit for head {layer_idx}.{head_idx}",
            width=700,
        )

    return EE_QK_circuit_result

#%%

tokens = [model.tokenizer.encode(data[i])[:256] for i in range(100) if len(model.tokenizer.encode(data[i])) >= 256]

result = prediction_attention_real_sentences(
    10, 
    7, 
    tokens=tokens,
    show_plot=True,
    mean_version=False,
)

# %%

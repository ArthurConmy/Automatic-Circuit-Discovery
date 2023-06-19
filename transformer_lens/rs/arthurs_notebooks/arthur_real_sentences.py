#%% [markdown]
# <h1> Example notebook for the cautils import * statement </h1>

from transformer_lens.cautils.notebook import * # use from transformer_lens.cautils.utils import * instead to for the same effect without autoreload
DEVICE = "cuda"

#%%

# load a model
model = HookedTransformer.from_pretrained("gpt2-small").to(DEVICE)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)

data = get_webtext()

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
    show_plot: bool = False,
    **kwargs,
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
    seq_len = len(tokens[0])
    n_layers, n_heads, d_model, d_head = model.W_Q.shape
    EE_QK_circuit_result = t.zeros((seq_len - 1, seq_len - 1))

    for random_seed in range(random_seeds):

        # Zero out the relevant things
        # (this deals with biases much more nicely too...)
        # Then get the attentions, then we're done
        # TODO speed this up by removing extra computation    

        sentence_tokens = t.tensor(tokens[random_seed])
        model.reset_hooks()

        for layer_to_zero_idx in range(1, layer_idx):
            for hook_name in [
                f"blocks.{layer_to_zero_idx}.attn.hook_result",
                f"blocks.{layer_to_zero_idx}.hook_mlp_out",
            ]:
                model.add_hook(
                    hook_name,
                    lambda z, hook: z * 0,
                )

        unembed = einops.rearrange(
            W_U[:, sentence_tokens[1:]], # shape [model_dim, seq_len-1]
            "d_model seq_len_minus_1 -> seq_len_minus_1 d_model",
        )

        def replace_with_unembedding(z, hook, unembedding_head_idx: int):
            assert len(z) == 1, z.shape
            assert z[0, :, unembedding_head_idx, :].shape == unembed.shape, (z.shape, unembed.shape)
            z[0, :, unembedding_head_idx] = unembed
            return z
        model.add_hook(
            f"blocks.{layer_idx}.hook_q_input",
            partial(replace_with_unembedding, unembedding_head_idx=head_idx),
        )

        # uh, so things will just attend to BOS unless we intervene ???
        def attn_score_hook(z, hook, attn_score_head):
            assert list(z.shape[2:]) == [seq_len - 1, seq_len - 1], (z.shape, (seq_len - 1, seq_len - 1))
            z[0, attn_score_head, 0, :] = -1e9 # ??? why fail
            return z
            
        model.add_hook(
            f"blocks.{layer_idx}.attn.hook_attn_scores",
            partial(attn_score_hook, attn_score_head=head_idx),
        )

        attn_pattern = t.zeros((seq_len - 1, seq_len - 1)) # used as a sort of cache
        def attn_pattern_hook(z, hook, attn_pattern_head):
            assert z.shape[2:] == attn_pattern.shape, (z.shape, attn_pattern.shape)
            attn_pattern[:] = z[0, attn_pattern_head].clone()
            return z
        model.add_hook(
            f"blocks.{layer_idx}.attn.hook_pattern",
            partial(attn_pattern_hook, attn_pattern_head=head_idx),
        )

        model(sentence_tokens[:-1]) # cut off the last embedding as we don't know the correct completion
        # TODO add mean functionality???

        EE_QK_circuit_result += attn_pattern.cpu()

    EE_QK_circuit_result /= random_seeds

    if show_plot:
        imshow(
            EE_QK_circuit_result[:, 1:],
            labels={"x": "Source/Key Token (embedding)", "y": "Destination/Query Token (unembedding)"},
            title=f"EE QK circuit for head {layer_idx}.{head_idx}",
            width=700,
            x=kwargs.get("x", None),
            y=kwargs.get("y", None),
        )

    return EE_QK_circuit_result

#%%

tokens = [model.tokenizer.encode(data[i])[:20] for i in range(100) if len(model.tokenizer.encode(data[i])) >= 256]
tokens = tokens[:1]
words = [model.tokenizer.decode(token) for token in tokens[0]]

result = prediction_attention_real_sentences(
    10, 
    7, 
    tokens=tokens,
    show_plot=True,
    # x=words[:-1],
    # y=words[:-1],
)

# %%

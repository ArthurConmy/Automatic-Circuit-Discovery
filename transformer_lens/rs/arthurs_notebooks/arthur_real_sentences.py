# %% [markdown]
# <h1> Example notebook for the cautils import * statement </h1>

from transformer_lens.cautils.notebook import * # use from transformer_lens.cautils.utils import * instead for the same effect without autoreload
from tqdm import tqdm
DEVICE = "cuda"

#%%

# load a model
model = HookedTransformer.from_pretrained("gpt2-small").to(DEVICE)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)

#%%
# <p> Load some data with unique sentences </p>

full_data = get_webtext()
TOTAL_OWT_SAMPLES = 100
SEQ_LEN = 20
data = full_data[:TOTAL_OWT_SAMPLES]

# %%

W_E = model.W_E.clone()
W_U = model.W_U.clone()

# %%

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
    tokens: Optional[List[List[int]]] = None,
    show_plot: bool = False,
    unembedding_indices: Optional[List[int]] = None,
    **kwargs,
):
    """Based off get_EE_QK_circuit from commit 4b32e53804764    
    Variable naming: maybe sentence_tokens should be called prompt_tokens instead?"""

    assert len(tokens) == 1, len(tokens)

    if unembedding_indices is None:
        print("Using the next tokens as the unembeddings")

    num_prompts = len(tokens) # eh not quite random seeds but whatever
    assert all([len(sentence_tokens) == len(tokens[0])] for sentence_tokens in tokens), "Must have same number of tokens in each sentence"
    
    seq_len = len(tokens[0])
    if unembedding_indices is None:
        seq_len -= 1

    n_layers, n_heads, d_model, d_head = model.W_Q.shape
    EE_QK_circuit_result = t.zeros((seq_len, seq_len))

    for prompt_idx in range(num_prompts):
        # Zero out the relevant things
        # (this deals with biases much more nicely too...)
        # Then get the attentions, then we're done
        # TODO speed this up by removing extra computation    

        sentence_tokens = t.tensor(tokens[prompt_idx])
        unembed = einops.rearrange(
            W_U[:, sentence_tokens[:-1]] if unembedding_indices is None else W_U[:, unembedding_indices[prompt_idx]], # shape [model_dim, seq_len-1]
            "d_model seq_len -> seq_len d_model",
        )

        model.reset_hooks()

        model.add_hook(
            "blocks.0.attn.hook_pattern",
            lock_attn, 
        )

        for layer_to_zero_idx in range(1, layer_idx):
            for hook_name in [
                f"blocks.{layer_to_zero_idx}.attn.hook_result",
                f"blocks.{layer_to_zero_idx}.hook_mlp_out",
            ]:
                model.add_hook(
                    hook_name,
                    lambda z, hook: z * 0,
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

        cached_attn_pattern = t.zeros((seq_len, seq_len))
        def attn_pattern_hook(z, hook, attn_pattern_head):
            assert z.shape[2:] == cached_attn_pattern.shape, (z.shape, cached_attn_pattern.shape)
            cached_attn_pattern[:] = z[0, attn_pattern_head].clone()
            return z
        model.add_hook(
            f"blocks.{layer_idx}.attn.hook_pattern",
            partial(attn_pattern_hook, attn_pattern_head=head_idx),
        )

        # if unembedding_indices None, cut off the last embedding as we don't know the correct completion
        input_tokens = sentence_tokens if unembedding_indices is not None else sentence_tokens[:-1]
        model(input_tokens)
        EE_QK_circuit_result += cached_attn_pattern.cpu()

    EE_QK_circuit_result /= num_prompts

    if show_plot:
        imshow(
            EE_QK_circuit_result,
            labels={"x": "Source/Key Token (embedding)", "y": "Destination/Query Token (unembedding)"},
            title=f"EE QK circuit for head {layer_idx}.{head_idx}, word {kwargs.get('title', None)}",
            width=700,
            x=kwargs.get("x", None),
            y=kwargs.get("y", None),
        )

    return EE_QK_circuit_result

# %% [markdown]

# Experiment setup:
# Feed in web text examples
# Measure attention batch to the unembedding token...

SEQ_LEN = 20
LAYER = 10
HEAD = 7

score = 0
score_denom = 0

all_rwords = []
vanilla_words = []

for prompt in data:
    tokens = model.to_tokens(prompt, prepend_bos=True)[0][:SEQ_LEN]
    words = [model.tokenizer.decode(token) for token in tokens]
    word_indices = {word: [] for word in set(words)}

    for idx, word in enumerate(words):
        word_indices[word].append(idx)

    word_attention = {word: 0.0 for word in set(words)}

    for word_idx, word in enumerate(set(words[1:])): # ignore BOS
        result = prediction_attention_real_sentences(
            LAYER,
            HEAD,
            tokens=[tokens],
            show_plot=False,
            unembedding_indices=[[model.tokenizer.encode(word)[0] for _ in range(len(tokens))]],
        )
        
        word_attention[word] = result[-1, word_indices[word]].sum()
    
    rwords = ""
    for word in words:
        if word_attention[word] > 0.8:
            rwords += f"[bold dark_orange u]{word}[/]"
        else:
            rwords += word
    all_rwords.append(rwords)
    vanilla_words.append("".join(words))
    if len(all_rwords)>20:
        break


#%%

print(vanilla_words[18])
rprint(all_rwords[18])

#%%

# now look at dataset statistics for words that occur frequently

for prompt in tqdm(full_data):
    tokens = model.to_tokens(prompt, prepend_bos=True)[0]

# %%

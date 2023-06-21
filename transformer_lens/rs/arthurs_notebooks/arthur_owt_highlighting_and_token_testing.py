# %% [markdown]
# <h1> Example notebook for the cautils import * statement </h1>

import token
from transformer_lens.cautils.notebook import * # use from transformer_lens.cautils.utils import * instead for the same effect without autoreload
from tqdm import tqdm
DEVICE = "cuda"

#%%

# load a model
# MODEL_NAME = "gpt2"
MODEL_NAME = "solu-10l"
model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
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

if "gpt" in model.cfg.model_name:
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

LAYER_IDX, HEAD_IDX = NEG_HEADS[model.cfg.model_name]
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
            LAYER_IDX,
            HEAD_IDX,
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

# WARNING: from here and below the method should probably be ignored, was quite a faff (though we did separate a/ the/ be etc from other proper nouns and things!!!)
# 
# now look at dataset statistics for words that occur frequently
# for prompt in tqdm(full_data):
#     tokens = model.to_tokens(prompt, prepend_bos=True)[0]

# %%

def get_word_ratios(
    data: List[str],
    window: int = 20
):
    ratios_tensor = t.zeros((model.cfg.d_vocab, 3)).int().to(device)
    # cols are (token appears, token is in last 20, token appears AND is in last 20)

    for prompt in tqdm(data):
        tokens = model.to_tokens(prompt, prepend_bos=True)[0]

        for idx, token in enumerate(tokens):
            ratios_tensor[token, 0] += 1
            last_window_tokens = tokens[max(0, idx-window): idx]
            ratios_tensor[list(set(last_window_tokens)), 1] += 1
            if token in last_window_tokens:
                ratios_tensor[token, 2] += 1
    
    return ratios_tensor.unbind(-1)


token_freq, token_freq_last_20, token_freq_and_last_20 = get_word_ratios(full_data[:100])

# %%

total_num_tokens = sum([len(model.to_str_tokens(i)) for i in full_data[:100]])

token_appears = token_freq > 10

tokens_which_appear = t.nonzero(token_appears).squeeze()
words_which_appear = model.to_str_tokens(tokens_which_appear)

nz_token_freq = token_freq[token_appears]
nz_token_freq_last_20 = token_freq_last_20[token_appears]
nz_token_freq_and_last_20 = token_freq_and_last_20[token_appears]


ratio = (total_num_tokens * nz_token_freq_and_last_20) / (nz_token_freq * nz_token_freq_last_20)

# %%

hist(t.log(ratio + 1), nbins=2500)

# %%

def sample_words(
    ratio: Tensor = t.log(ratio + 1), 
    words_which_appear: List[str] = words_which_appear,
    n: int = 10,
    cutoff_pt : float = 0.01
):
    indices_below = t.nonzero(ratio < cutoff_pt).squeeze().tolist()
    indices_above = t.nonzero(ratio >= cutoff_pt).squeeze().tolist()

    assert len(indices_below) >= n
    assert len(indices_above) >= n

    indices_below = np.random.choice(indices_below, size=n, replace=False).tolist()
    indices_above = np.random.choice(indices_above, size=n, replace=False).tolist()
    
    return indices_below, indices_above

indices_below, indices_above = sample_words()

print([words_which_appear[i] for i in indices_below])
print([words_which_appear[i] for i in indices_above])

# TODO - maybe return here and try and actually calculate the update ratio (rather than just intuiting it, which is what we'll try now)

# %%

non_update_word_lists = {
    " the": " the dog jumped over the fence.",
    " and": " she picked up the book and started reading.",
    " to": " he walked to the store.",
    " in": " she was in the kitchen.",
    " of": "He drank a glass of water.",
    " Thus": " Thus, we conclude our experiment with significant results.",
    # " Whence": " Whence did you arrive at this conclusion?",
    " Hence": " Hence, it is evident that the research hypothesis was incorrect.",
    # " Therefore": " The evidence was inconclusive; therefore, we cannot assert the suspect's guilt.",
    " Nonetheless": " Nonetheless, despite the initial setbacks, the project was a success.",
}

update_word_lists = {
    " John": " John was reading a book when suddenly, John heard a strange noise.",
    " Maria": " Maria loves playing the piano and, moreover, Maria also enjoys painting.",
    " city": " The city was full of lights, making the city look like a sparkling diamond.",
    " ball": " The ball rolled away, so the dog chased the ball all the way to the park.",
    " Python": " Python is a popular language for programming. In fact, Python is known for its simplicity.",
    " President": " The President announced new policies today. Many are waiting to see how the President's decisions will affect the economy.",
    " Bitcoin": " Bitcoin's value has been increasing rapidly. Investors are closely watching Bitcoin's performance.",
    " dog": " The dog wagged its tail happily. Seeing the dog so excited, the children started laughing.",
    " cake": " The cake looked delicious. Everyone at the party was eager to taste the cake.",
    " book": " The book was so captivating, I couldn't put the book down.",
    " house": " The house was quiet. Suddenly, a noise from the upstairs of the house startled everyone.",
    " car": " The car pulled into the driveway. Everyone rushed out to see the new car.",
    " computer": " The computer screen flickered. She rebooted the computer hoping to resolve the issue.",
    " key": " She lost the key to her apartment. She panicked when she realized she had misplaced the key.",
    " apple": " He took a bite of the apple. The apple was crisp and delicious.",
    " phone": " The phone rang in the middle of the night. She picked up the phone with a groggy hello.",
    " train": " The train was late. The passengers were annoyed because the train was delayed by an hour.",
}

LAYER = 10
HEAD = 7

score = 0
score_denom = 0

all_rwords = []
vanilla_words = []

all_word_data = {}

for word, prompt in {**non_update_word_lists, **update_word_lists}.items():

    tokens = model.to_tokens(prompt, prepend_bos=True)[0]
    word_token_idx = model.to_tokens(word, prepend_bos=False)[0]
    try:
        assert word_token_idx in tokens
    except:
        print(f"Word {word} not in prompt {prompt}")
        continue
    word_indices = [i for i, token in enumerate(tokens) if token == word_token_idx]

    result = prediction_attention_real_sentences(
        LAYER,
        HEAD,
        tokens=[tokens],
        show_plot=False,
        unembedding_indices=[[word_token_idx for _ in range(len(tokens))]],
    )
    
    all_word_data[word] = result[-1, word_indices].sum()

# %%

keys, values = zip(*all_word_data.items())

px.bar(y=[v.log().item() for v in values], x=keys, title="Log of in-the-wild prediction-attention scores for GPT-generated sentences", labels={"x": "Word", "y": "Log-attention"})

# %% [markdown]
# <p> Okay cool this method works well so let's just do this on random webtext </p>

SEQ_LEN = 20
attentions = defaultdict(list)

for prompt_idx, prompt in tqdm(enumerate(data[:100])):
    tokens = model.to_tokens(prompt, prepend_bos=True)[0]
    tokens = tokens[: (len(tokens) // SEQ_LEN) * SEQ_LEN]
    for window_start_idx in range(0, len(tokens), SEQ_LEN):
        window_tokens = tokens[window_start_idx : window_start_idx + SEQ_LEN]
        window_end_token = window_tokens[-1]
        word_indices = [i for i, token in enumerate(window_tokens) if token == window_end_token]

        result = prediction_attention_real_sentences(
            LAYER,
            HEAD,
            tokens=[window_tokens],
            show_plot=False,
            unembedding_indices=[[window_end_token for _ in range(len(window_tokens))]],
        )
        attentions[window_end_token.item()].append(result[-1, word_indices].sum().item())

print(attentions)

# %%

len_attentions = defaultdict(int)
for k, v in attentions.items():
    len_attentions[len(v)] += 1
# assert len(len_attentions) == 1 # ?????

hist(
    t.tensor(list(len_attentions.values())),
)
# %%


attentions_filtered = {
    k: v for (k, v) in attentions.items() if len(v) >= 1
}
mean_attn_values = [t.tensor(v).mean().item() for (k, v) in attentions_filtered.items()]
words = [model.to_str_tokens(t.tensor([k]), prepend_bos=False)[0] for k in attentions_mean.keys()]

tuples = list(zip(mean_attn_values, words))
tuples.sort(key=lambda x: -x[0])

mean_attn_values, words = zip(*tuples)

CUTOFF = 20
extreme_means = list(mean_attn_values[:CUTOFF]) + (t.tensor(mean_attn_values[-CUTOFF:]) - .1).tolist()
extreme_words = words[:CUTOFF] + words[-CUTOFF:]

fig = go.Figure()
fig.add_trace(go.Bar(y=extreme_means, name="count", text=extreme_words)) #, texttemplate="%{x}", textfont_size=20))

# %%

# %% [markdown]

from transformer_lens.cautils.notebook import *  # use from transformer_lens.cautils.utils import * instead for the same effect without autoreload
import gc

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%

MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)

# %%

# data = get_webtext()

update_word_lists = {
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
    " house": " The house was quiet. Suddenly, a noise from the upstairs of the house startled everyone",
    " car": " The car pulled into the driveway. Everyone rushed out to see the new car today",
    " computer": " The computer screen flickered. She rebooted the computer hoping to resolve the issue",
    " key": " She lost the key to her apartment. She panicked when she realized she had misplaced the key today",
    " apple": " He took a bite of the apple. The apple was crisp and delicious",
    " phone": " The phone rang in the middle of the night. She picked up the phone with a groggy hello",
    " train": " The train was late. The passengers were annoyed because the train was delayed by an hour",
}

update_tokens = {
    model.tokenizer.encode(k, return_tensors="pt").item(): model.to_tokens(v, prepend_bos=True, move_to_device=False).squeeze() for k, v in update_word_lists.items()
}

# %%

unembedding = model.W_U.clone()
LAYER_IDX, HEAD_IDX = NEG_HEADS[model.cfg.model_name]

#%%

attn_paids = []

for update_token_idx, (update_token, prompt_tokens) in enumerate(update_tokens.items()):
    update_token_positions = (prompt_tokens == update_token).nonzero().squeeze().tolist()
    prompt_words = [model.tokenizer.decode(token) for token in prompt_tokens]

    hook_pattern = f"blocks.{LAYER_IDX}.attn.hook_pattern"
    logits, cache = model.run_with_cache(
        prompt_tokens.to(DEVICE),
        names_filter=lambda name: name in [hook_pattern],
    )
    attn = cache[hook_pattern][0, HEAD_IDX, :, :].detach().cpu()
    attn_paid_to_update_token = attn[update_token_positions[-1]-1, update_token_positions].sum()
    attn_paids.append(attn_paid_to_update_token)

    if update_token_idx>7:
        imshow(
            attn,
            title=f"attn paid to {prompt_tokens[update_token_positions]}",
            x=[f"{idx}_{word}" for idx, word in enumerate(prompt_words)],
            y=[f"{idx}_{word}" for idx, word in enumerate(prompt_words)],
            labels={"x": "Key", "y": "Query"},
            font=dict(size=8),
        )

# %%

px.bar(
    x=list(update_word_lists.keys()),
    y=attn_paids,
) # TBH it seems somewhat lowish? Callum doesn't think so that much

# %% [markdown]

the_components = []
the_normalized_components = []
scales = []

for update_token_idx, (update_token, prompt_tokens) in enumerate(update_tokens.items()):
    update_token_positions = (prompt_tokens == update_token).nonzero().squeeze().tolist()
    prompt_words = [model.tokenizer.decode(token) for token in prompt_tokens]

    hook_pre = f"blocks.{LAYER_IDX}.hook_resid_pre"
    hook_scale = f"blocks.{LAYER_IDX}.ln1.hook_scale"
    logits, cache = model.run_with_cache(
        prompt_tokens.to(DEVICE),
        names_filter=lambda name: name in [hook_pre, hook_scale],
    )
    scales.append(cache[hook_scale])

    residual_stream_vector = cache[hook_pre][0, update_token_positions[-1]-1].clone()

    pre = cache[hook_pre]
    # fake layer norm
    pre = pre / pre.norm(dim=-1, keepdim=True)
    normalized_residual_stream_vector = pre[0, update_token_positions[-1]-1]
    
    unembedding_component = einops.einsum(
        residual_stream_vector,
        unembedding[:, update_token],
        "d, d ->",
    ).item()
    normalized_unembedding_component = einops.einsum(
        normalized_residual_stream_vector,
        unembedding[:, update_token],
        "d, d ->",
    ).item()

    the_normalized_components.append(normalized_unembedding_component)
    the_components.append(unembedding_component)

#%%

px.scatter(
    x=attn_paids,
    y=the_normalized_components,
    text=list(update_word_lists.keys()),
    labels = {"x": "Attention Paid", "y": "Unembedding Component"},
)

# %%

new_attn_paids = []
new_normalized_components = []
titles = []
colors = []
SCALE_FACTORS = [0.0, 0.5, 0.8, 0.9, 0.99, 1.0, 1.01, 1.1, 1.15, 1.2, 1.25, 1.5, 2.0]
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "black", "gray", "pink", "brown", "cyan", "magenta", "teal"]
assert len(SCALE_FACTORS) == len(COLORS), (len(SCALE_FACTORS), len(COLORS))

for scale_factor, color in tqdm(list(zip(SCALE_FACTORS, COLORS))):
    for update_token_idx, (update_token, prompt_tokens) in enumerate(update_tokens.items()):
        update_token_positions = (prompt_tokens == update_token).nonzero().squeeze().tolist()
        prompt_words = [model.tokenizer.decode(token) for token in prompt_tokens]
        unembedding_vector = unembedding[:, update_token]
        update_word = list(update_word_lists.keys())[update_token_idx]

        def increase_component(z, hook, mu):
            assert z[0, 0].shape == unembedding_vector.shape
            z[0, update_token_positions[-1]-1] += unembedding_vector * mu
            return z
        def decrease_component(z, hook, mu):
            z[0, update_token_positions[-1]-1] -= unembedding_vector * mu
            return z

        hook_pattern = f"blocks.{LAYER_IDX}.attn.hook_pattern"
        model.reset_hooks()
        mu = float(scale_factor * the_components[update_token_idx] / unembedding_vector.norm().item())
        model.add_hook(f"blocks.{LAYER_IDX}.hook_resid_pre", partial(increase_component, mu=mu))
        model.add_hook(f"blocks.{LAYER_IDX}.hook_resid_mid", partial(decrease_component, mu=mu))
        logits, cache = model.run_with_cache(
            prompt_tokens.to(DEVICE),
            names_filter=lambda name: name in [hook_pattern],
        )
        attn = cache[hook_pattern][0, HEAD_IDX, :, :].detach().cpu()
        attn_paid_to_update_token = attn[update_token_positions[-1]-1, update_token_positions].sum()
        new_attn_paids.append(attn_paid_to_update_token)

        # ugh surely a better way of making plots...
        titles.append(update_word + f" {scale_factor}")
        colors.append(color)

# %%

y_values = [c*scale_factor for scale_factor in SCALE_FACTORS for c in the_components]
px.scatter(
    x=new_attn_paids,
    y=y_values,
    text=titles,
    title="Clusterfuck version",
    color=colors,
    labels = {"x": "Attention Paid", "y": "Unembedding Component"},
)

# %%

# Prepare the figure
fig = go.Figure()
CUTOFF = 6

for update_token_idx, (update_token, prompt_tokens) in enumerate(list(update_tokens.items())[:CUTOFF]):
    # extract the data for this update_token
    x_values = [new_attn_paids[i] for i in range(update_token_idx, len(new_attn_paids), len(update_tokens))]
    y_values_current = [y_values[i] for i in range(update_token_idx, len(y_values), len(update_tokens))]
    color = COLORS[update_token_idx % len(COLORS)]  # using colors in a cyclic manner
    fig.add_trace(go.Scatter(x=x_values, y=y_values_current, mode='lines+markers', text=[titles[update_token_idx + len(update_tokens) * ii] for ii in range(len(x_values))], line=dict(color=color), name=titles[update_token_idx][:-4]))

# Update layout
fig.update_layout(title='Attention Paid vs Unembedding Component (Labels give how much we scaled the component by)',
                  xaxis_title='Attention Paid',
                  yaxis_title='Unembedding Component')

fig.show()

# %%

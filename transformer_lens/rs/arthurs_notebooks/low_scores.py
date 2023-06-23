from transformer_lens.cautils.notebook import *  # use from transformer_lens.cautils.utils import * instead for the same effect without autoreload

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_split_qkv_normalized_input(True)  # new flag who dis
USE_NAME_MOVER = False
MODE = "key"  # TODO implement value
assert MODE in ["query", "key"]
SHOW_LOADS = False
LOCK_QUERY_WU = True
# TODO implement runtime checking or whatever

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

if USE_NAME_MOVER:
    N = 20
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=True,
        seed=1,
        device=DEVICE,
    )
    update_word_lists = {" " + sent.split()[-1]: sent for sent in ioi_dataset.sentences}
    assert len(update_word_lists) == len(set(update_word_lists.keys())), "non unique1!"

update_tokens = {
    model.tokenizer.encode(k, return_tensors="pt")
    .item(): model.to_tokens(v, prepend_bos=True, move_to_device=False)
    .squeeze()
    for k, v in update_word_lists.items()
}

unembedding = model.W_U.clone()

# look at some components...

for update_token_idx, (update_token, prompt_tokens) in enumerate(update_tokens.items()):
    update_token_positions = (
        (prompt_tokens == update_token).nonzero().squeeze().tolist()
    )
    prompt_words = [model.tokenizer.decode(token) for token in prompt_tokens]
    unembedding_vector = unembedding[:, update_token]
    update_word = list(update_word_lists.keys())[update_token_idx]
    position = (
        update_token_positions[-1] - 1 if MODE == "query" else update_token_positions[0]
    )

    logits, cache = model.run_with_cache(
        prompt_tokens,
        names_filter=lambda name: name.endswith(("hook_result", "hook_mlp_out")),
    )

    res = t.zeros(12, 13)

    for LAYER_IDX in range(12):
        for HEAD_IDX in range(12):
            head_outp = cache[f"blocks.{LAYER_IDX}.attn.hook_result"][
                0, update_token_positions[-1] - 1, HEAD_IDX
            ]
            assert len(head_outp.shape) == 1, head_outp.shape
            comp = einops.einsum(
                head_outp,
                unembedding_vector,
                "i, i ->",
            )
            res[LAYER_IDX, HEAD_IDX] = comp.item()

        comp = einops.einsum(
            cache[f"blocks.{LAYER_IDX}.hook_mlp_out"][0, position],
            unembedding_vector,
            "i, i ->",
        )
        res[LAYER_IDX, -1] = comp.item()

    res[0, 12] = 30.0  # so things are roughly same scale

    if SHOW_LOADS:
        imshow(
            res,
            title="|".join(prompt_words),
            labels={"x": "Head", "y": "Layer"},
            # x=list(range(12)) + ["MLP"],
        )
    # if update_token_idx > 3:
    #     break

unembedding = model.W_U.clone()
LAYER_IDX, HEAD_IDX = (
    NEG_HEADS[model.cfg.model_name] if not USE_NAME_MOVER else (10, 10)
)

# Arthur makes a hook with tons of assertions so hopefully nothing goes wrong
# We only use it later, but declaring here makes syntax not defined show maybe?


def component_adjuster(
    z,
    hook,
    d_model,
    expected_component,
    unit_direction,
    mu,
    update_token_positions,
):
    position = (
        update_token_positions[-1] - 1 if MODE == "query" else update_token_positions[0]
    )

    assert z[0, position, HEAD_IDX].shape == unit_direction.shape
    assert abs(z[0, position, HEAD_IDX].norm().item() - d_model**0.5)
    component = einops.einsum(
        z[0, position, HEAD_IDX],
        unit_direction,
        "i, i ->",
    )
    assert abs(component.item() - expected_component) < 1e-4, (
        component.item(),
        expected_component,
    )

    # delete the current_unit_direction component
    z[0, position, HEAD_IDX] -= unit_direction * component
    orthogonal_component = z[0, position, HEAD_IDX].norm().item()

    assert abs(orthogonal_component) > 1e-5 # otherwise we're in trouble

    # rescale the orthogonal component
    j = (max(0,(d_model - mu**2 * component**2)) ** 0.5) / orthogonal_component # LOL python turns things into imaginary numbers with **0.5. Scary.
    z[0, position, HEAD_IDX] *= j

    # re-add the current_unit_direction component
    z[0, position, HEAD_IDX] += unit_direction * component * mu

    # we should now be variance 1 again
    assert abs(z[0, position, HEAD_IDX].norm().item() - d_model**0.5) < 3e-3, abs(z[0, position, HEAD_IDX].norm().item() - d_model**0.5)

    return z

# Let's cache some things to use as components

if MODE == "key":
    saved_unit_directions = {
        "blocks.0.hook_resid_pre": [],
        "blocks.0.hook_mlp_out": [],
        "blocks.1.hook_resid_pre": [],
        "unembedding": [],
    }

    for layer in range(2): # think about early layer contributions to residual stream
        saved_unit_directions[f"blocks.{layer}.hook_attn_out"] = []
        saved_unit_directions[f"blocks.{layer}.hook_mlp_out"] = []

else:
    saved_unit_directions = {
        "blocks.1.hook_resid_pre": [],
        "unembedding": [],
    }


def normalize(tens):
    assert len(list(tens.shape)) == 1
    return tens / tens.norm()


for update_token_idx, (update_token, prompt_tokens) in enumerate(update_tokens.items()):
    update_token_positions = (
        (prompt_tokens == update_token).nonzero().squeeze().tolist()
    )
    prompt_words = [model.tokenizer.decode(token) for token in prompt_tokens]

    logits, cache = model.run_with_cache(
        prompt_tokens.to(DEVICE),
        names_filter=lambda name: name in list(saved_unit_directions.keys()),
    )

    for name in list(saved_unit_directions.keys()):
        if name == "unembedding":
            continue

        position = (
            update_token_positions[-1] - 1
            if MODE == "query"
            else update_token_positions[0]
        )
        saved_unit_directions[name].append(
            normalize(cache[name][0, position]).detach().cpu().clone()
        )

    saved_unit_directions["unembedding"].append(
        normalize(unembedding[:, update_token]).detach().cpu().clone()
    )

# at first I'll just look at components... then do the full interpolation plot

component_data = {key: [] for key in saved_unit_directions.keys()}
INPUT_HOOK = f"blocks.{LAYER_IDX}.hook_{MODE.lower()[:1]}_normalized_input"

for update_token_idx, (update_token, prompt_tokens) in enumerate(update_tokens.items()):
    update_token_positions = (
        (prompt_tokens == update_token).nonzero().squeeze().tolist()
    )
    prompt_words = [model.tokenizer.decode(token) for token in prompt_tokens]
    unembedding_vector = unembedding[:, update_token]
    update_word = list(update_word_lists.keys())[update_token_idx]

    for unit_direction_string in saved_unit_directions.keys():
        current_unit_direction = saved_unit_directions[unit_direction_string][
            update_token_idx
        ].to(DEVICE)
        assert abs(current_unit_direction.norm().item() - 1) < 1e-4

        logits, cache = model.run_with_cache(
            prompt_tokens.to(DEVICE),
            names_filter=lambda name: name == INPUT_HOOK,
        )

        position = (
            update_token_positions[-1] - 1
            if MODE == "query"
            else update_token_positions[0]
        )
        normalized_residual_stream = cache[INPUT_HOOK][0, position, HEAD_IDX]
        assert (
            abs(normalized_residual_stream.norm().item() - (model.cfg.d_model) ** 0.5)
            < 1e-4
        )

        component_data[unit_direction_string].append(
            einops.einsum(
                normalized_residual_stream,
                current_unit_direction,
                "i, i ->",
            ).item()
        )

SCALE_FACTORS = (
    [None] + torch.arange(0, 2, 0.5).tolist()
    # [0.0, 0.5, 0.99, 1.0, 1.01, 1.1, 1.25, 1.5, 2.0]
    if not USE_NAME_MOVER
    else torch.arange(-2, 2, 1).tolist()
)

# if (MODE=="key"):
#     SCALE_FACTORS = torch.arange(-5, 20, 1).tolist()

ALL_COLORS = [
    "red",
    "orange",
    "green",
    "blue",
    "purple",
    "gray",
    "pink",
    "brown",
    "cyan",
    "magenta",
    "teal",
    "olive",
    "navy",
    "maroon",
    "lime",
    "black",
    "yellow",
]
assert len(ALL_COLORS) >= len(update_tokens), (len(ALL_COLORS), len(update_tokens))
ALL_COLORS = ALL_COLORS[: len(update_tokens)]

attention_scores: Dict[Tuple[str, str, float], float] = {
    # tuple of update_word, unit_direction_string, scale_factor to attention paid
}

for scale_factor in tqdm(SCALE_FACTORS):
    for update_token_idx, (update_token, prompt_tokens) in enumerate(
        update_tokens.items()
    ):
        update_token_positions = (
            (prompt_tokens == update_token).nonzero().squeeze().tolist()
        )
        prompt_words = [model.tokenizer.decode(token) for token in prompt_tokens]
        unembedding_vector = unembedding[:, update_token]
        update_word = list(update_word_lists.keys())[update_token_idx]
        position = (
            update_token_positions[-1] - 1
            if not (MODE == "key")
            else update_token_positions[0]
        )

        for unit_direction_string in saved_unit_directions.keys():
            current_unit_direction = saved_unit_directions[unit_direction_string][
                update_token_idx
            ].to(DEVICE)

            expected_component = component_data[unit_direction_string][update_token_idx]

            if scale_factor is None: 
                cur_scale_factor = (model.cfg.d_model)**0.5 / expected_component
            else:
                cur_scale_factor = scale_factor

            if (
                abs(expected_component) * abs(cur_scale_factor) > model.cfg.d_model**0.5
            ):  # rip
                continue

            model.reset_hooks()

            if LOCK_QUERY_WU:
                def set_to_unembedding(z, hook, direction, update_token_positions):
                    assert list(z.shape)[2:] == [
                        model.cfg.n_heads,
                        model.cfg.d_model,
                    ], list(z.shape)
                    z[
                        0, update_token_positions[-1] - 1, HEAD_IDX, :
                    ] = direction.clone()
                    return z

                model.add_hook(
                    f"blocks.{LAYER_IDX}.hook_q_normalized_input",
                    partial(
                        set_to_unembedding,
                        direction=saved_unit_directions["unembedding"][
                            update_token_idx
                        ],
                        update_token_positions=update_token_positions,
                    ),
                )

            model.add_hook(
                INPUT_HOOK,
                partial(
                    component_adjuster,
                    d_model=model.cfg.d_model,
                    expected_component=expected_component,
                    mu=cur_scale_factor,
                    unit_direction=current_unit_direction,
                    update_token_positions=update_token_positions,
                ),
                level=1,
            )

            hook_scores = f"blocks.{LAYER_IDX}.attn.hook_attn_scores"

            logits, cache = model.run_with_cache(
                prompt_tokens.to(DEVICE),
                names_filter=lambda name: name in [hook_scores],
            )
            attn_score = cache[hook_scores][0, HEAD_IDX, :, :].detach().cpu()
            assert len(update_token_positions) == 2
            attn_score_on_update_token = attn_score[
                update_token_positions[-1] - 1, update_token_positions[0]
            ].item()
            attention_scores[
                (
                    update_word,
                    unit_direction_string,
                    scale_factor,
                )
            ] = attn_score_on_update_token

# Prepare the figure
fig = go.Figure()
show = list(range(2, 4))

TEXTURES = {
    key: ["solid", "dot", "dash", "dashdot"][idx % 4]
    for idx, key in enumerate(saved_unit_directions.keys())
}

for unit_direction_string in list(saved_unit_directions.keys())[:4]:
    for update_token_idx, (update_token, prompt_tokens) in enumerate(
        list(update_tokens.items())
    ):
        update_word = list(update_word_lists.keys())[update_token_idx]

        y = []
        relevant_scale_factors = (
            []
        )  # maybe big and negative boys ae too big in magnitude
        for scale_factor in SCALE_FACTORS:
            if (update_word, unit_direction_string, scale_factor) in attention_scores:
                y.append(
                    attention_scores[(update_word, unit_direction_string, scale_factor)]
                )
                relevant_scale_factors.append(scale_factor)

        print(relevant_scale_factors, unit_direction_string, update_word)

        if update_token_idx not in show:
            continue

        fig.add_trace(
            go.Scatter(
                x = [
                    relevant_scale_factor * component_data[unit_direction_string][update_token_idx]
                    if relevant_scale_factor is not None
                    else (model.cfg.d_model)**0.5
                    for relevant_scale_factor in relevant_scale_factors
                ][: len(y)],
                y=y,
                mode="lines+markers",
                text=[
                    f"{update_word=} {unit_direction_string=} {relevant_scale_factor=}"
                    for relevant_scale_factor in relevant_scale_factors
                ][: len(y)],
                line=dict(
                    color=ALL_COLORS[update_token_idx],
                    dash=TEXTURES[unit_direction_string],
                ),
                name=f"{update_word=} {unit_direction_string=}",
            )
        )

# Add an x axis line at the max component, sqrt(d_model) and text
for sign in [-1.0, 1.0]:
    fig.add_trace(
        go.Scatter(
            x=[sign * model.cfg.d_model**0.5, sign * model.cfg.d_model**0.5],
            y=[0, 1.0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Max component",
        )
    )

# Update layout
fig.update_layout(
    title="Attention Scores vs Unembedding component when varying alpha tilde by 0 to +2 its original component",
    yaxis_title="Attention Scores",
    xaxis_title=f'Component in Normalized {LAYER_IDX}.{HEAD_IDX} {MODE} Input ("alpha tilde")',
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
)

fig.update_layout()
fig.show(
    config={
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ]
    }
)
# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthur_utils import dot_with_query

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_split_qkv_normalized_input(True)
DEVICE = "cuda"
LAYER_IDX, HEAD_IDX = NEG_HEADS[model.cfg.model_name]
INCLUDE_ORTHOGONAL = True
USE_ADD_ONE_TO_KEYS = False
USE_IOI = False

# %%

# Goal: see which key component is best
# Setup: do IOI

if USE_IOI:
    N = 200
    warnings.warn("Auto IOI")
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=True,
        seed=35795,
        device=DEVICE,
    )
    update_word_lists = {" " + sent.split()[-1]: sent for sent in ioi_dataset.sentences}
    assert len(update_word_lists) == len(set(update_word_lists.keys())), "Non-uniqueness!"

else:
    update_word_lists = {
        " John": "Today John was reading a book when suddenly John heard a strange noise",
        " Maria": "Today Maria loves playing the piano and, moreover Maria also enjoys painting",
        " city": " The city was full of lights, making the city look like a sparkling diamond",
        " ball": " The ball rolled away, so the dog chased the ball all the way to the park",
        " Python": "Currently Python is a popular language for programming. In fact, Python is known for its simplicity",
        # " President": "The President announced new policies today. Many are waiting to see how the President's decisions will affect the economy", # this one seemed bugged...
        " Bitcoin": "Lately Bitcoin's value has been increasing rapidly. Investors are closely watching Bitcoin's performance",
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

# %%

# (...continued)
#
# Let's edit the keys. Let's compare the components in the directions of
# and the normalized attention I guess?
#
# Baselines:
# - Full
# - MLP0
# - My `<BOS>The word` trick
# - Callum locked attentions
# ???

update_tokens = {
    model.tokenizer.encode(k, return_tensors="pt")
    .item(): model.to_tokens(v, prepend_bos=True, move_to_device=False)
    .squeeze()
    for k, v in update_word_lists.items()
}
N = len(update_tokens)
update_token_positions = torch.LongTensor([
    (prompt_tokens == update_token).nonzero().squeeze().tolist() for update_token, prompt_tokens in update_tokens.items()
])
update_token_positions[:, -1] -= 1

if USE_ADD_ONE_TO_KEYS: # this allows a better baseline comparison...
    update_token_positions[:, 0] += 1

assert list(update_token_positions.shape) == [len(update_tokens), 2], update_token_positions.shape
update_tokens_values = model.to_tokens([v for v in update_word_lists.values()], prepend_bos=True, move_to_device=False)
for update_tokens_element, update_tokens_values_element in list(zip(update_tokens.values(), update_tokens_values)):
    assert torch.allclose(
        update_tokens_element,
        update_tokens_values_element[: len(update_tokens_element)],
    )

# %%

# Cache The Inputs
_, cache = model.run_with_cache(
    update_tokens_values,
    names_filter=lambda name: name
    in [f"blocks.{LAYER_IDX}.hook_q_input", f"blocks.{LAYER_IDX}.hook_k_input", f"blocks.{LAYER_IDX}.attn.hook_attn_scores", "blocks.0.hook_mlp_out"],
)

# batch, pos, n_heads, d_model
cached_query_input = cache["blocks.{}.hook_q_input".format(LAYER_IDX)][
    torch.arange(len(update_tokens)), update_token_positions[:, -1], HEAD_IDX, :
]
cached_key_input = cache["blocks.{}.hook_k_input".format(LAYER_IDX)][
    torch.arange(len(update_tokens)), update_token_positions[:, 0], HEAD_IDX, :
]
cached_mlp0 = cache["blocks.0.hook_mlp_out"][torch.arange(N), update_token_positions[:, 0], :]
cached_attention_scores = cache["blocks.{}.attn.hook_attn_scores".format(LAYER_IDX)][torch.arange(N), HEAD_IDX, update_token_positions[:, -1], update_token_positions[:, 0]]

#%%

imshow(
    cache["blocks.{}.attn.hook_attn_scores".format(LAYER_IDX)][0, HEAD_IDX, :, :],
)

#%%

assert list(cached_attention_scores.shape) == [N], cached_attention_scores.shape
assert list(cached_query_input.shape) == [
    N,
    model.cfg.d_model,
], cached_query_input.shape

#%%

unnormalized_query_input = cached_query_input

# %%

results = dot_with_query(
    unnormalized_keys = cached_key_input,
    unnormalized_queries = unnormalized_query_input,
    model = model,
    layer_idx = LAYER_IDX,
    head_idx = HEAD_IDX,
)

assert torch.allclose(results.cpu(), cached_attention_scores.cpu(), atol=1e-2, rtol=1e-2), (results.norm().item(), cached_attention_scores.norm().item(), "dude the assertion is 1e-2 this sure should work!")

#%%

the_inputs = model.to_tokens(["The"+key for key in update_word_lists.keys()], prepend_bos=True, move_to_device=False)

assert torch.allclose(the_inputs[:, -1], update_tokens_values[torch.arange(N), update_token_positions[:, 0] - int(USE_ADD_ONE_TO_KEYS)])

if USE_ADD_ONE_TO_KEYS:
    the_inputs[:, -1] = update_tokens_values[torch.arange(N), update_token_positions[:, 0]]

assert list(the_inputs.shape) == [N, 3], the_inputs.shape # "<bos>The IO"
_, cache = model.run_with_cache(
    the_inputs,
    names_filter=lambda name: f"blocks.{LAYER_IDX}.hook_resid_pre" == name,
)
the_residuals = cache["blocks.{}.hook_resid_pre".format(LAYER_IDX)][:, -1]

_, cache = model.run_with_cache(
    update_tokens_values,
    names_filter = lambda name: name.endswith("attn.hook_pattern"),
) 

def attn_lock(z, hook, update_token_positions, scaled=False):
    old_z = z.clone()
    z[:] *= 0.0

    if scaled:
        for idx in range(N):
            att_io = old_z[idx, :, update_token_positions[idx, 0], update_token_positions[idx, 0]]
            att_bos = old_z[idx, :, update_token_positions[idx, 0], 0]

            z[idx, :, update_token_positions[idx, 0], update_token_positions[idx, 0]] = att_io/(att_io + att_bos)
            z[idx, :, update_token_positions[idx, 0], 0] = att_bos/(att_io + att_bos)

    else:
        # this thing is vectorized but difficult to extend
        all_key_positions = [[0 for _ in range(N)], update_token_positions[:, 0]]
        for key_positions in all_key_positions:
            z[torch.arange(N), :, update_token_positions[:, 0], key_positions] = old_z[torch.arange(N), :, update_token_positions[:, 0], key_positions] 

    return z

callums_baselines = []

for scaled in [False, True]:
    model.reset_hooks()
    for layer in range(LAYER_IDX):
        model.add_hook(
            "blocks.{}.attn.hook_pattern".format(layer),
            partial(attn_lock, scaled=scaled, update_token_positions=update_token_positions),
            level=1,
        )
    _, cache = model.run_with_cache(
        update_tokens_values,
        names_filter = lambda name: name==f"blocks.{LAYER_IDX}.hook_resid_pre",
    )
    model.reset_hooks()
    callums_baseline = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"][torch.arange(N), update_token_positions[:, 0]]
    callums_baselines.append(callums_baseline)

callums_baseline, callums_baseline_scaled = callums_baselines

#%%

# Baselines:
# - Full
# - MLP0
# - My `<BOS>The word` trick
# - Callum locked attentions
# ???

baselines = {
    "Actual key inputs": cached_key_input,
    "MLP0": cached_mlp0,
    "`The` baseline": the_residuals,
    "Callum's baseline": callums_baseline,
    # "Callum's baseline (scaled)": callums_baseline_scaled,
}

#%%

# orthogonal components 

OTHER=False

if INCLUDE_ORTHOGONAL:
    old_baseline_keys = list(baselines.keys())
    for baseline_name in old_baseline_keys:
        if baseline_name.startswith("Actual"):
            continue
        normalized_baseline = baselines[baseline_name] / baselines[baseline_name].norm(dim=-1, keepdim=True)
        normalized_keys = cached_key_input / cached_key_input.norm(dim=-1, keepdim=True)

        if OTHER:
            other_orthogonal_complement = normalized_keys - normalized_baseline
            baselines["Other complement to " + baseline_name] = other_orthogonal_complement

        else:
            orthogonal_complement = normalized_keys - einops.einsum(
                normalized_baseline,
                normalized_keys,
                "batch d_model, batch d_model -> batch",
            ).unsqueeze(-1) * normalized_baseline

            orthogonal_complement = orthogonal_complement / orthogonal_complement.norm(dim=-1, keepdim=True)

            assert einops.einsum(
                orthogonal_complement,
                normalized_baseline,
                "batch d_model, batch d_model -> batch",
            ).abs().max().item() < 1e-3, "Orthogonal complement is not orthogonal to the baseline"    

            assert torch.allclose(
                orthogonal_complement.norm(dim=-1),
                normalized_keys.norm(dim=-1),
                atol=1e-5,
                rtol=1e-5,
            ), "Orthogonal complement is not the same norm as the keys"

            baselines["Orthogonal complement of " + baseline_name] = orthogonal_complement

#%%

histone = {
    key: einops.einsum(
        cached_key_input / cached_key_input.norm(dim=-1, keepdim=True),
        value / value.norm(dim=-1, keepdim=True),
        "batch d_model, batch d_model -> batch",
    ) for key, value in baselines.items()
}

#%%

# generated from running with keys plussed by 1... (ADD_ONE_TO_KEYS=True)
my_saved_tensor = torch.tensor([[-0.7974, -0.5853, -0.9609, -0.6416, -1.2189,  1.2220, -3.5672, -1.4475,
         -1.6168,  0.9716, -1.7894,  0.6841,  0.3232, -1.0422,  1.6342,  0.4185],
        [-7.9158, -5.7688, -8.1737, -7.6218, -9.2938, -8.8542, -9.4700, -8.8647,
         -9.2955, -7.2335, -8.7087, -4.9248, -7.9534, -7.3958, -2.1351, -8.4300],
        [-1.5110, -2.0415, -1.8174, -1.4037, -2.2313, -1.9389, -2.3042, -2.2087,
         -2.9325, -1.1443, -1.9182,  0.8899,  0.0727, -1.7918,  0.5757, -0.4240],
        [-1.0657, -1.7476, -2.4428, -3.9636, -1.8691, -1.3279, -5.6341, -3.8707,
         -2.8990, -2.2111, -3.2645, -1.1164, -2.2040, -1.5478, -1.3107, -1.8053],
        [ 6.5099,  4.3223,  6.5926,  4.9765,  6.0188,  8.6397,  4.5854,  5.2127,
          6.2637,  7.7292,  4.6524,  5.0695,  6.9100,  5.3538,  3.8484,  7.9852],
        [ 0.8038,  1.2879,  1.0634,  0.8041,  0.7245,  4.2542, -3.0604,  0.6080,
          1.3173,  3.2422, -0.4191, -0.1362,  0.4467,  0.4173,  2.0429,  1.2919],
        [ 0.4434,  1.9573,  2.6945,  4.1407,  0.4045,  4.0924,  2.9677,  3.4216,
          1.9097,  5.7129,  1.3067,  3.4670,  3.6914,  0.5945,  4.5255,  3.6550]])

# generated from running with keys plussed by 2...
my_saved_tensor_two_back = torch.tensor([[ -0.8638,  -2.0370,  -2.6609,  -0.7906,  -1.3508,   0.7168,  -0.8655,
           0.1669,  -2.8414,  -0.4860,  -2.6967,  -4.6721,  -1.0419,  -0.6098,
          -1.7511,  -1.7714],
        [ -7.8829,  -7.6748,  -9.4408,  -7.9672,  -8.7748, -10.6031,  -7.1199,
          -3.8279,  -9.3967,  -5.0555,  -6.3869,  -9.9933,  -8.7495,  -6.6996,
          -6.9596,  -8.9850],
        [ -2.4180,  -3.1992,  -2.6070,  -1.2580,  -1.5044,  -1.2269,  -1.4923,
           1.2547,  -3.3147,  -0.4951,  -2.3064,  -3.8541,  -0.4694,  -2.0242,
          -2.6588,  -2.1211],
        [ -3.2770,  -3.7357,  -5.0653,   1.1981,  -1.9413,  -2.9573,  -3.6215,
           0.0261,  -4.6934,  -2.5263,  -4.4198,  -6.0736,  -2.7271,  -2.2481,
          -2.8648,  -4.3975],
        [  5.5429,   4.1920,   4.3383,   4.6313,   5.3504,   7.9819,   2.4739,
           3.6277,   3.8808,   3.8833,   1.7513,   3.9411,   6.1183,   5.4265,
           4.0928,   4.5705],
        [  1.7131,   0.6672,  -0.9323,   0.2165,  -0.3617,   3.1423,   0.1822,
          -1.3116,  -0.4674,   0.1345,  -1.4124,  -2.3481,  -0.9432,   1.5404,
           0.4323,  -0.3238],
        [  2.3017,   2.1383,   1.8457,  -1.3533,   0.1739,   4.7635,   1.9859,
           0.3683,   1.4773,   2.7086,   1.1542,   1.6280,   1.7707,   2.2659,
           1.1855,   2.2598]])

#%%

hist(
    list(histone.values()),
    title="Cosine sim of baselines with the actual keys",
    names=list(baselines.keys()),
    width=800,
    height=600,
    opacity=0.7,
    marginal="box",
    template="simple_white",
    nbins=50,
    # static=True,
)

hist(
    [x for x in (torch.stack([
        dot_with_query(
            baseline,
        ) for baseline in baselines.values()
    ], dim=0) - my_saved_tensor)], # this is a subtraction from a baseline that came from running with keys +1ed
    title="Attention Scores of Different Keys (these keys scaled to norm 1)",
    names=list(baselines.keys()),
    width=800,
    height=600,
    opacity=0.7,
    marginal="box",
    template="simple_white",
    # static=True,
)

# %%

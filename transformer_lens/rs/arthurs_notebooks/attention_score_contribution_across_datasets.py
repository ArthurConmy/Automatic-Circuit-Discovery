#%% [markdown] [1]:

"""
Hopefully this still makes the attention plot decomposition...
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import (
    project,
    get_effective_embedding_2,
)

from transformer_lens.rs.callum.orthogonal_query_investigation import (
    decompose_attn_scores_full,
    create_fucking_massive_plot_1,
    create_fucking_massive_plot_2,
    token_to_qperp_projection,
    FakeIOIDataset,
)

clear_output()
USE_IOI = False
#%% [markdown] [2]:

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
# model.set_use_split_qkv_normalized_input(True)
clear_output()

#%%

filtered_examples = Path("../arthur/json_data/filtered_for_high_attention.json")
with open(filtered_examples, "r") as f:
    filtered_examples = json.load(f)
max_act_fname = Path("../arthur/json_data/head_ten_point_seven_data_eight_max_importance_samples.json")
with open(max_act_fname, "r") as f:
    head_ten_point_seven_data_eight_max_importance_samples = json.load(f)
gpt_4_fname = Path("../arthur/json_data/gpt_4_update_words.json")
with open(gpt_4_fname, "r") as f:
    gpt_4_update_words = json.load(f)
totally_random_fname = Path("../arthur/json_data/totally_random_sentences_with_random_in_context_word_at_end.json")
with open(totally_random_fname, "r") as f:
    totally_random = json.load(f)
top5p_examples_fname = Path("../arthur/json_data/top5p_examples.json")
with open(top5p_examples_fname, "r") as f:
    top5p_examples = json.load(f)
more_top5p_examples = Path("../arthur/json_data/more_top5p_examples.json")
with open(more_top5p_examples, "r") as f:
    more_top5p_examples = json.load(f)

N = 60
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=True,
    seed=35795,
)

# TODO load the other examples

y_data = {}

# %%

for dataset_name, raw_dataset in [
    ("filtered_for_high_attention", filtered_examples),
    ("head_ten_point_seven_data_eight_max_importance_samples", head_ten_point_seven_data_eight_max_importance_samples),
    ("ioi_dataset", ioi_dataset),
    ("gpt_4_update_words", gpt_4_update_words),
    ("totally_random_sentences_with_random_in_context_word_at_end", totally_random),
    ("top5p_examples", top5p_examples),
    ("more_top5p_examples", more_top5p_examples),
]:
    print("Processing dataset: ", dataset_name, "...")

    if dataset_name in y_data:
        continue

    if isinstance(raw_dataset, dict): # need to make into FakeIOIDataset
        cur_res = {}
        for inc in tqdm([-2, -1, 1, 2]): # some selection of *different* keys
            dataset = FakeIOIDataset(
                sentences = list(raw_dataset.values()),
                io_tokens=list(raw_dataset.keys()),
                key_increment=inc,
                model=model,
            )
            cur_res[inc], ioi_cache = decompose_attn_scores_full(
                ioi_dataset=dataset,
                batch_size = dataset.N,
                seed = 0,
                nnmh = (10, 7),
                model = model,
                use_effective_embedding = False,
                use_layer0_heads = False,
                subtract_S1_attn_scores = True,
                include_S1_in_unembed_projection = False,
                project_onto_comms_space = "W_EE0A",
                return_cache=True,
            )
            ioi_cache = ioi_cache.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        data = sum(cur_res.values())
        assert len(data.shape) == 3
        data = data.sum(dim=(1, 2)) # first dim is the 6 projections
        y_data[dataset_name] = 100.0 * (data / data.sum()) # this is a percentage

    elif isinstance(raw_dataset, IOIDataset):
        full_results = decompose_attn_scores_full(
            ioi_dataset=raw_dataset,
            batch_size = raw_dataset.N,
            seed = 0,
            nnmh = (10, 7),
            model = model,
            use_effective_embedding = False,
            use_layer0_heads = False,
            subtract_S1_attn_scores = True,
            include_S1_in_unembed_projection = False,
            project_onto_comms_space = "W_EE0A",
        ).sum(dim=(1,2))
        y_data[dataset_name] = 100.0 * (full_results / full_results.sum()) # this is a percentage

    else:
        raise NotImplementedError()

    break

#%%

x_axis_points = [
    "q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>",
    "q ∥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>", 
    "q ⊥ W<sub>U</sub>[IO] & ∥ comms, k ∥ MLP<sub>0</sub>",
    "q ⊥ W<sub>U</sub>[IO] & ∥ comms, k ⊥ MLP<sub>0</sub>", 
    "q ⊥ W<sub>U</sub>[IO] & ⟂ comms, k ∥ MLP<sub>0</sub>", 
    "q ⊥ W<sub>U</sub>[IO] & ⟂ comms, k ⊥ MLP<sub>0</sub>"
]
fig = go.Figure(data=[
    go.Bar(name=dataset_name, x=x_axis_points, y=data)
    for dataset_name, data in y_data.items()
])
fig.update_layout(
    barmode='group',
    title="Percentage relative contribution to attention scores at copy positions.",
)

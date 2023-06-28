#%% [markdown] [1]:

"""
Mostly cribbed from transformer_lens/rs/callum/orthogonal_query_investigation_2.ipynb
(but I prefer .py investigations)
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

max_act_fname = Path("../arthur/json_data/head_ten_point_seven_data_eight_max_importance_samples.json")
with open(max_act_fname, "r") as f:
    head_ten_point_seven_data_eight_max_importance_samples = json.load(f)
gpt_4_fname = Path("../arthur/json_data/gpt_4_update_words.json")
with open(gpt_4_fname, "r") as f:
    gpt_4_update_words = json.load(f)
totally_random_fname = Path("../arthur/json_data/totally_random_sentences_with_random_in_context_word_at_end.json")
with open(totally_random_fname, "r") as f:
    totally_random = json.load(f)

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
    ("head_ten_point_seven_data_eight_max_importance_samples", head_ten_point_seven_data_eight_max_importance_samples),
    ("ioi_dataset", ioi_dataset),
    ("gpt_4_update_words", gpt_4_update_words),
    ("totally_random_sentences_with_random_in_context_word_at_end", totally_random),
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
            cur_res[inc]=decompose_attn_scores_full(
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
            )
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

#%%
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

update_word_lists = {
    " John": "Today John was reading a book when suddenly John heard a strange noise",
    " Maria": "Today Maria loves playing the piano and, moreover Maria also enjoys painting",
    " city": " The city was full of lights, making the city look like a sparkling diamond",
    " ball": " The ball rolled away, so the dog chased the ball all the way to the park",
    " Python": "Nowadays Python is a popular language for programming. In fact, Python is known for its simplicity",
    " President": " The President announced new policies today. Many are waiting to see how the President's decisions will affect the economy",
    " Bitcoin": "Recently, Bitcoin's value has been increasing rapidly. Investors are closely watching Bitcoin's performance",
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

ks = list(update_word_lists.keys())
for i, k in enumerate(ks):
    while not update_word_lists[k].endswith(k):
        update_word_lists[k] = update_word_lists[k][:-1]
    if i not in [3, 4, 6, 7, 8, 13, 15, 16]:
        update_word_lists.pop(k)

if USE_IOI:
    N = 60
    warnings.warn("Auto IOI")
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=True,
        seed=35795,
    )

for k, v in list(update_word_lists.items()):
    assert v.count(k)==2, (k, v)

#%% [markdown] [3]:

effective_embeddings = get_effective_embedding_2(model)
W_EE = effective_embeddings['W_E (including MLPs)']
W_EE0 = effective_embeddings['W_E (only MLPs)']
W_E = model.W_E
# Define an easier-to-use dict!
effective_embeddings = {"W_EE": W_EE, "W_EE0": W_EE0, "W_E": W_E}

#%%

res = []

#%%

res.append(decompose_attn_scores_full(
    ioi_dataset=ioi_dataset,
    batch_size = N,
    seed = 0,
    nnmh = (10, 7),
    model = model,
    use_effective_embedding = False,
    use_layer0_heads = False,
    subtract_S1_attn_scores = True,
    include_S1_in_unembed_projection = False,
    project_onto_comms_space = "W_EE0A",
))

create_fucking_massive_plot_1(res[0])

# %%

create_fucking_massive_plot_2(res[0])

# %%

ioi_fake = FakeIOIDataset(
    sentences = ioi_dataset.sentences,
    io_tokens = [" " + sent.split(" ")[-1] for sent in ioi_dataset.sentences],
    key_increment=0,
    model=model,
)
ioi_fake.s_tokenIDs = ioi_dataset.s_tokenIDs
ioi_fake.word_idx["S1"] = ioi_dataset.word_idx["S1"]

# %%

res2 = decompose_attn_scores_full(
    ioi_dataset=ioi_fake,
    batch_size = N,
    seed = 0,
    nnmh = (10, 7),
    model = model,
    use_effective_embedding = False,
    use_layer0_heads = False,
    subtract_S1_attn_scores = True,
    include_S1_in_unembed_projection = False,
    project_onto_comms_space = "W_EE0A",
)
# %%

# yey fake thing works

data1 = FakeIOIDataset(
    sentences = list(update_word_lists.values()),
    io_tokens=list(update_word_lists.keys()),
    key_increment=1,
    model=model,
)

# %%

res = {}

for inc in tqdm([1, 2]):

    data1 = FakeIOIDataset(
        sentences = list(update_word_lists.values()),
        io_tokens=list(update_word_lists.keys()),
        key_increment=inc,
        model=model,
    )

    res[inc]=decompose_attn_scores_full(
        ioi_dataset=data1,
        batch_size = data1.N,
        seed = 0,
        nnmh = (10, 7),
        model = model,
        use_effective_embedding = False,
        use_layer0_heads = False,
        subtract_S1_attn_scores = True,
        include_S1_in_unembed_projection = False,
        project_onto_comms_space = "W_EE0A",
    )

create_fucking_massive_plot_1(
    sum(list(res.values()))/(len(res)),
)

# %%

create_fucking_massive_plot_2(
    sum(list(res.values()))/(len(res)),
)

#%%

logits, cache = model.run_with_cache(
    data1.toks,
    names_filter=lambda name: name.endswith("pattern"),
)

#%%

imshow(
    cache["blocks.10.attn.hook_pattern"][torch.arange(data1.N), 7, data1.word_idx["end"], data1.word_idx["IO"]].unsqueeze(0)
)
# %%

# 3, 4, 6, 7, 8, 13, 15, 16
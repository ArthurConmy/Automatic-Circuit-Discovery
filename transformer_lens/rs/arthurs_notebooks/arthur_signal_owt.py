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
    token_to_qperp_projection
)

clear_output()

USE_IOI = True

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

#%%

head_9_9_max_activating = {
    # " James": ' Later I concluded that, because the Russians kept secret their developments in military weapons, they thought it improper to ask us about ours.\n\nJames F. Byrnes, Speaking Frankly (New York: Harper and Brothers, 1947) p. 263.\n\nSecretary of State', # weird cos James no space
    " du": """Jamaica Inn by Daphne du Maurier

Broken Rule(s): non-protagonist points of view and numerous redundancies.

Most writers compulsively check and double check for echoes and redundancies and remove them. Contrast that with Daphne""",
    " Sweeney": "Mandatory safety training was part of Sweeney's centerpiece bill, passed by both houses of the legislature last year but conditionally vetoed by the governor. The bill would have changed the way the state issues firearms licenses, made background checks instant and included private sales in the law. It also would have required proof of safety training prior to the issuance of a gun license. Training was among the elements altered by the governor's veto. After the conditional veto,",
    " Mark": """it will end up in a bright good place.

The Adventures of Huckleberry Finn by Mark Twain

Broken Rule: multiple regional dialects that are challenging to understand.

No discussion of writerly rule-breaking would be complete without mentioning""",
    " Lal": "Tyrone Troutman's girlfriend, Donna Lalor, told investigators that Joseph Troutman became enraged after his son grabbed a radio from the table and smashed it.\n\nJoseph Troutman then grabbed a butcher knife, came at his son and stabbed him,",
    " 1906": """ a Neapolitan street game, the "morra".[3][5] (In this game, two persons wave their hands simultaneously, while a crowd of surrounding gamblers guess, in chorus, at the total number of fingers exposed by the principal players.)[6] This activity was prohibited by the local government, and some people started making the players pay for being "protected" against the passing police.[3][7][8]

Camorristi in Naples, 1906 in Naples,""",
    " Mush": """ Omar Mushaweh, a Turkey-based leader of the Syrian Muslim Brotherhood, told IRIN in an online interview.

But, like other opposition sympathisers interviewed,""",
}

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
    update_word_lists = {" " + sent.split()[-1]: sent for sent in ioi_dataset.sentences}
    # lol will reduce in size cos of duplicate IOs
    old_update_word_lists=deepcopy(update_word_lists)
    for k, v in old_update_word_lists.items():
        assert v.endswith(k), (k, v)
        update_word_lists[k] = v[:-len(k)]
    assert len(update_word_lists) == len(set(update_word_lists.keys())), "Non-uniqueness!"
    head_9_9_max_activating = update_word_lists

for k, v in list(head_9_9_max_activating.items()):
    assert v.count(k)==1, k

tokens = model.to_tokens(list(head_9_9_max_activating.values()), truncate=False)
key_positions = []
query_positions = []
for i in range(len(tokens)):
    if tokens[i, -1].item()!=model.tokenizer.pad_token_id:
        query_positions.append(tokens[i].shape[-1]-1)
    else:
        for j in range(len(tokens[i])-1, -1, -1):
            if tokens[i, j].item()!=model.tokenizer.pad_token_id:
                query_positions.append(j)
                break

    key_token = model.to_tokens([list(head_9_9_max_activating.keys())[i]], prepend_bos=False).item()
    assert tokens[i].tolist().count(key_token)==1
    key_positions.append(tokens[i].tolist().index(key_token))

assert len(key_positions)==len(query_positions)==len(tokens), ("Missing things probably", len(key_positions), len(query_positions), len(tokens))

#%% [markdown] [3]:

effective_embeddings = get_effective_embedding_2(model)
W_EE = effective_embeddings['W_E (including MLPs)']
W_EE0 = effective_embeddings['W_E (only MLPs)']
W_E = model.W_E
# Define an easier-to-use dict!
effective_embeddings = {"W_EE": W_EE, "W_EE0": W_EE0, "W_E": W_E}

#%%

res = []

class FakeIOIDataset:
    """Used for normal webtext things where we imitate the IOI dataset methods"""

    def __init__(
        self,
        sentences,
        io_tokens,
        key_increment,
    ):
        self.N=len(sentences)
        sentences_trimmed = []
        update_word_lists = {} # different format used in the past        
        for k, v in list(zip(io_tokens, sentences, strict=True)):
            assert v.endswith(k), (k, v)
            sentences_trimmed.append(v[:-len(k)])
            assert sentences_trimmed[-1].count(k)==1

        self.toks = model.to_tokens(sentences_trimmed)
        self.word_idx={}
        self.word_idx["IO"] = []
        self.word_idx["end"] = []
        self.word_idx["S1"] = []

        for i in range(len(self.toks)):
            if self.toks[i, -1].item()!=model.tokenizer.pad_token_id:
                self.word_idx["end"].append(self.toks.shape[-1]-1)
            else:
                for j in range(len(self.toks[i])-1, -1, -1):
                    if self.toks[i, j].item()!=model.tokenizer.pad_token_id:
                        self.word_idx["end"].append(j)
                        break

            key_token = model.to_tokens([io_tokens[i]], prepend_bos=False).item()
            assert self.toks[i].tolist().count(key_token)==1
            self.word_idx["IO"].append(self.toks[i].tolist().index(key_token))

        self.io_tokenIDs = self.toks[torch.arange(self.N), self.word_idx["IO"]]

        self.word_idx["S1"] = (torch.LongTensor(self.word_idx["IO"]) + key_increment) + key_increment
        assert N==len(self.word_idx["IO"])==len(self.word_idx["S1"]), ("Missing things probably", len(self.word_idx["IO"]), len(self.word_idx["S1"]))

        assert 0 <= self.word_idx["S1"].min().item()
        assert self.toks.shape[1] > self.word_idx["S1"].max().item()
        self.word_idx["S1"] = self.word_idx["S1"].tolist()
        self.s_tokenIDs = self.toks[torch.arange(self.N), self.word_idx["S1"]]

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
    key_increment=0
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

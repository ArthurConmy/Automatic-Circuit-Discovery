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
    " station": """" including the Kibo and Columbus science modules, even seem to reflect the Earth's 
lovely bluish colors. The image also shows off large power generating solar arrays on the station's 90 meter long 
integrated truss structure Just put your cursor over the picture to identify some of the major parts of the""",
    " Roger": """Senators send letter to Roger GoodellSteve Delsohn reports the details surrounding the domestic 
violence charge against Panthers""",
    " overhead": """In addition, I attempted to lower the 
resolution of the shot to fit the other two better. Again, the first image in this series is the overhead,
the second two are from the Kim""",
    " exploitation": """Thus a husband
has a clear incentive to appropriate the wife��s future quasi-rents, by divorcing her unilaterally after having 
extracted most of his quasi-rent from the marriage. This is called quasi-rent destruction.While the example is 
provably the exception, it still is helpful in illustrating the concept. Clearly if a man was able to get away with
this he would be rewarded materially for betraying his wife. But divorce theft isn��t the only option available to 
the spouse which has the other one over a barrel. They could also use this change in fortunes to renegotiate the 
terms of the marriage in their favor under threat of divorce, which economists call exploitation:Brinig and Allen 
(2000) argue that there are two different types of quasi""",
    " site": """Fuqua Development has unveiled tentative plans for a major mixed-use development wedged against 
Memorial Drive and the Beltline corridor.A new marketing flyer released by the controversial developer elaborates 
on his planned project in Reynoldstown, which includes more than 100,000 square feet of retail and restaurant space
— including (potentially) a CineBistro and Sprouts — plus 120,000 square feet of offices, 600 apartments, and 100 
condos.Fuqua purchased the site at 905 Memorial Drive last year, which didn't thrill a lot of folks, given the 
developer's penchant for big-box projects. Fuqua's planned overhaul will replace a warehouse facility butted up 
against Interstate 20.According to this rendering, which might not represent the latest version of Fuqua's plans, 
retail spaces would front the Belt""",
    " use": """Synopsis EditProduction EditMusic EditMeaning EditReggio stated that the Qatsi films are intended to 
simply create an experience and that "it is up  the viewer to take for himself/herself what it is that  means." He 
also said that "these films have never been about the effect of technology, of industry on people. It's been that 
everyone: politics, education, things of the financial structure, the nation state structure, language, the 
culture, religion, all of that exists within the host of technology. So it's not the effect of, it's that 
everything exists within . It's not that we use technology, we live""",
    " oil": """ROME (Thomson Reuters Foundation) - Italian police said they have busted a crime ring exporting fake 
extra virgin olive oil to the United States, highlighting the mafia��s infiltration of Italy��s famed agriculture 
and food business.Twelve people with links to the ��Ndrangheta, the organized crime group based in the southern 
Calabria region, were arrested on Tuesday on a series of charges including mafia association and fraud, police said
in a statement.The gang shipped cheap olive p""",
#     " with": """ Park next month.��I am really looking forward to the inaugural SprintX race at Canadian Tire Motorsport Park.�� 
# Wittmer said.��Racing in my home country is always as nice thing, especially in front of family, friends, and 
# representing the BMW brand.��Wittmer, a BMW of North America factory driver, is a former IMSA GT Le Mans class 
# champion, and will bring a wealth of experience to the upstart team, which made its debut at Circuit of The 
# Americas last month.The Texas-based team, with the support BMW of North America, has shifted its focus entirely to 
# SprintX for the remainder of the year.��I��m honored to have the opportunity to participate with Kuno for this new 
# series,�� Mills said.��It��s a dream come true to be included in the BMW family, and we are excited to pursue a 
# championship with Kun""",
    " resolve": """ Each time TTC staff were forced to respond it took an average of 4 minutes 51 seconds to get things moving 
again.The TTC counts incidents that involve the police separately, labeling them "security incidents."Just behind 
passenger alerts was the TTC's own train problems. The rolling stock - a technical name for the trains - needed 
emergency repairs 1,323 times in 2012, causing 109 hours and 49 minutes of delays.Frustratingly, the problem that 
created the worst wait times is almost entirely preventable. 330 small fires, litter problems, and unauthorized 
people at track level stopped the subway for a total of 106 hours and 37 minutes in 2012. Each incident took an 
average of 19 minutes to resolve - longer than breakdowns and assistance alarms.In total there were 4,842 outages 
on the subway in 2012 that caused more than three weeks of delays (509 hours and 35 minutes to be precise.) The 
average delay across all types took 6 minutes and 31 seconds to clear""",
}

ks = list(update_word_lists.keys())
for i, k in enumerate(ks):

    # if i not in [3,5,7]: 
    #     update_word_lists.pop(k)
    #     continue

    # cur_sent = model.to_tokens([update_word_lists[k]])[0][-1].item()
    update_word_lists[k] = model.tokenizer.decode(model.to_tokens([update_word_lists[k]])[0][:-1])+k  # cut off last thing
    # while not update_word_lists[k].endswith(k):
    #     update_word_lists[k] = update_word_lists[k][:-1]

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
for inc in tqdm([-2, -1, 1, 2]):
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

#%%

px.bar( 
    x=[
        "q ∥ W<sub>U</sub>[IO], k ∥ MLP<sub>0</sub>",
        "q ∥ W<sub>U</sub>[IO], k ⊥ MLP<sub>0</sub>", 
        "q ⊥ W<sub>U</sub>[IO] & ∥ comms, k ∥ MLP<sub>0</sub>",
        "q ⊥ W<sub>U</sub>[IO] & ∥ comms, k ⊥ MLP<sub>0</sub>", 
        "q ⊥ W<sub>U</sub>[IO] & ⟂ comms, k ∥ MLP<sub>0</sub>", 
        "q ⊥ W<sub>U</sub>[IO] & ⟂ comms, k ⊥ MLP<sub>0</sub>"
    ],
    y=100*(sum(res.values()).sum(dim=(1, 2)) / sum(res.values()).sum()),
    title="Percentage contribution to attention scores, averaged over 'S1' positions which we take the mean over",
).show()

# %%

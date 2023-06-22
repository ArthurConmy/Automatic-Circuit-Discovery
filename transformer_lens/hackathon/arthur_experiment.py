# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
from typeguard import typechecked
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import torch as t
import torch
import einops
import itertools
import plotly.express as px
import numpy as np
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from jaxtyping import Float, Int, jaxtyped
from typing import Union, List, Dict, Tuple, Callable, Optional
from torch import Tensor
import gc
import transformer_lens
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens import utils
from transformer_lens.utils import to_numpy
t.set_grad_enabled(False)

# %%

def to_tensor(
    tensor,
):
    return t.from_numpy(to_numpy(tensor))

def imshow(
    tensor, 
    **kwargs,
):
    tensor = to_tensor(tensor)
    zmax = tensor.abs().max().item()

    if "zmin" not in kwargs:
        kwargs["zmin"] = -zmax
    if "zmax" not in kwargs:
        kwargs["zmax"] = zmax
    if "color_continuous_scale" not in kwargs:
        kwargs["color_continuous_scale"] = "RdBu"

    fig = px.imshow(
        to_numpy(tensor),
        **kwargs,
    )
    fig.show()

# %%

MODEL_NAME = "gpt2-small"
# MODEL_NAME = "solu-10l"
model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME)
from transformer_lens.hackathon.ioi_dataset import IOIDataset, NAMES

# %%

N = 100
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

ioi_dataset = IOIDataset(
    prompt_type="mixed" if model.cfg.tokenizer_name == "gpt2" else "BABA", # hacky fix for solu-10l IOi tokenization
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=True,
    nb_templates=None if model.cfg.tokenizer_name == "gpt2" else 1,
    seed=1,
    device=DEVICE,
)

#%%

model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)

# %%

# reproduce name mover heatmap

logits, cache = model.run_with_cache(
    ioi_dataset.toks,
    names_filter = lambda name: name.endswith("z"),
)

# %%

logit_attribution = t.zeros((model.cfg.n_layers, model.cfg.n_heads))
for i in range(model.cfg.n_layers):
    attn_result = einops.einsum(
        cache["z", i][t.arange(N), ioi_dataset.word_idx["end"]], # (batch, head_idx, d_head)
        model.W_O[i], # (head_idx, d_head, d_model)
        "batch head_idx d_head, head_idx d_head d_model -> batch head_idx d_model",
    )
    logit_dir = model.W_U.clone()[:, ioi_dataset.io_tokenIDs] - model.W_U.clone()[:, ioi_dataset.s_tokenIDs]

    layer_attribution_old = einops.einsum(
        attn_result,
        logit_dir,
        "batch head_idx d_model, d_model batch -> batch head_idx",
    )

    for j in range(model.cfg.n_heads):
        logit_attribution[i, j] = layer_attribution_old[:, j].mean()

# %%

imshow(
    logit_attribution,
    title="GPT-2 Small head direct logit attribution",
)

# %%

LAYER_IDX, HEAD_IDX = {
    "SoLU_10L1280W_C4_Code": (9, 18), # (9, 18) is somewhat cheaty
    "gpt2": (10, 7),
}[model.cfg.model_name]


W_U = model.W_U
W_Q_negative = model.W_Q[LAYER_IDX, HEAD_IDX]
W_K_negative = model.W_K[LAYER_IDX, HEAD_IDX]

W_E = model.W_E

# ! question - what's the approximation of GPT2-small's embedding?
# lock attn to 1 at current position
# lock attn to average
# don't include attention

#%%

from transformer_lens import FactoredMatrix

full_QK_circuit = FactoredMatrix(W_U.T @ W_Q_negative, W_K_negative.T @ W_E.T)

indices = t.randint(0, model.cfg.d_vocab, (250,))
full_QK_circuit_sample = full_QK_circuit.A[indices, :] @ full_QK_circuit.B[:, indices]

full_QK_circuit_sample_centered = full_QK_circuit_sample - full_QK_circuit_sample.mean(dim=1, keepdim=True)

imshow(
    full_QK_circuit_sample_centered,
    labels={"x": "Source / key token (embedding)", "y": "Destination / query token (unembedding)"},
    title="Full QK circuit for negative name mover head",
    width=700,
)

# %%

def top_1_acc_iteration(full_QK_circuit: FactoredMatrix, batch_size: int = 100) -> float: 
    '''
    This should take the argmax of each row (i.e. over dim=1) and return the fraction of the time that's equal to the correct logit
    '''
    # SOLUTION
    A, B = full_QK_circuit.A, full_QK_circuit.B
    nrows = full_QK_circuit.shape[0]
    nrows_max_on_diagonal = 0

    for i in tqdm(range(0, nrows + batch_size, batch_size)):
        rng = range(i, min(i + batch_size, nrows))
        if rng:
            submatrix = A[rng, :] @ B
            diag_indices = t.tensor(rng, device=submatrix.device)
            nrows_max_on_diagonal += (submatrix.argmax(-1) == diag_indices).float().sum().item()
    
    return nrows_max_on_diagonal / nrows

print(f"Top-1 accuracy of full QK circuit: {top_1_acc_iteration(full_QK_circuit):.2%}")

# %%

def top_5_acc_iteration(full_OV_circuit: FactoredMatrix, batch_size: int = 100) -> float:
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    A, B = full_OV_circuit.A, full_OV_circuit.B
    nrows = full_OV_circuit.shape[0]
    nrows_top5_on_diagonal = 0

    for i in tqdm(range(0, nrows + batch_size, batch_size)):
        rng = range(i, min(i + batch_size, nrows))
        if rng:
            submatrix = A[rng, :] @ B
            diag_indices = t.tensor(rng, device=submatrix.device).unsqueeze(-1)
            top5 = t.topk(submatrix, k=5).indices
            nrows_top5_on_diagonal += (diag_indices == top5).sum().item()

    return nrows_top5_on_diagonal / nrows

print(f"Top-5 accuracy of full QK circuit: {top_5_acc_iteration(full_QK_circuit):.2%}")

# %%

def lock_attn(
    attn_patterns: Float[t.Tensor, "batch head_idx dest_pos src_pos"],
    hook: HookPoint,
    ablate: bool = False,
) -> Float[t.Tensor, "batch head_idx dest_pos src_pos"]:
    
    assert isinstance(attn_patterns, Float[t.Tensor, "batch head_idx dest_pos src_pos"])
    assert hook.layer() == 0

    batch, n_heads, seq_len = attn_patterns.shape[:3]
    attn_new = einops.repeat(t.eye(seq_len), "dest src -> batch head_idx dest src", batch=batch, head_idx=n_heads).clone().to(attn_patterns.device)
    if ablate:
        attn_new = attn_new * 0
    return attn_new

def fwd_pass_lock_attn0_to_self(
    model: HookedTransformer,
    input: Union[List[str], Int[t.Tensor, "batch seq_pos"]],
    ablate: bool = False,
) -> Float[t.Tensor, "batch seq_pos d_vocab"]:

    model.reset_hooks()
    
    loss = model.run_with_hooks(
        input,
        return_type="loss",
        fwd_hooks=[(utils.get_act_name("pattern", 0), partial(lock_attn, ablate=ablate))],
    )

    return loss

# %%

raw_dataset = load_dataset("stas/openwebtext-10k")
train_dataset = raw_dataset["train"]
dataset = [train_dataset[i]["text"] for i in range(len(train_dataset))]

# %%

for i, s in enumerate(dataset):
    loss_hooked = fwd_pass_lock_attn0_to_self(model, s)
    print(f"Loss with attn locked to self: {loss_hooked:.2f}")
    loss_hooked_0 = fwd_pass_lock_attn0_to_self(model, s, ablate=True)
    print(f"Loss with attn locked to zero: {loss_hooked_0:.2f}")
    loss_orig = model(s, return_type="loss")
    print(f"Loss with attn free: {loss_orig:.2f}\n")

    # gc.collect()

    if i == 5:
        break

# %%

if "gpt" in model.cfg.model_name: # sigh, tied embeddings
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
            resid_post = model.blocks[0].mlp(normalized_resid_mid) # TODO not resid post!!!
            W_EE[cur_range.to(DEVICE)] = resid_post

else: 
    W_EE = W_E # untied embeddings so no need to calculate!

# %%

if "gpt" in model.cfg.model_name: # sigh, tied embeddings
    # sanity check this is the same 

    def remove_pos_embed(z, hook):
        return 0.0 * z

    # setup a forward pass that 
    model.reset_hooks()
    model.add_hook(
        name="hook_pos_embed",
        hook=remove_pos_embed,
        level=1, # ???
    ) 
    model.add_hook(
        name=utils.get_act_name("pattern", 0),
        hook=lock_attn,
    )
    logits, cache = model.run_with_cache(
        torch.arange(1000).to(DEVICE).unsqueeze(0),
        names_filter=lambda name: name=="blocks.1.hook_resid_pre",
        return_type="logits",
    )


    W_EE_test = cache["blocks.1.hook_resid_pre"].squeeze(0)
    W_EE_prefix = W_EE_test[:1000]

    assert torch.allclose(
        W_EE_prefix,
        W_EE_test,
        atol=1e-4,
        rtol=1e-4,
    )   

# %%

def get_EE_QK_circuit(
    layer_idx,
    head_idx,
    random_seeds: Optional[int] = 5,
    num_samples: Optional[int] = 500,
    bags_of_words: Optional[List[List[int]]] = None, # each List is a List of unique tokens
    mean_version: bool = True,
    show_plot: bool = False,
):
    assert (random_seeds is None and num_samples is None) != (bags_of_words is None), (random_seeds is None, num_samples is None, bags_of_words is None, "Must specify either random_seeds and num_samples or bag_of_words_version")

    if bags_of_words is not None:
        random_seeds = len(bags_of_words) # eh not quite random seeds but whatever
        assert all([len(bag_of_words) == len(bags_of_words[0])] for bag_of_words in bags_of_words), "Must have same number of words in each bag of words"
        num_samples = len(bags_of_words[0])

    W_Q_head = model.W_Q[layer_idx, head_idx]
    W_K_head = model.W_K[layer_idx, head_idx]

    EE_QK_circuit = FactoredMatrix(W_U.T @ W_Q_head, W_K_head.T @ W_EE.T)
    EE_QK_circuit_result = t.zeros((num_samples, num_samples))

    for random_seed in range(random_seeds):
        if bags_of_words is None:
            indices = t.randint(0, model.cfg.d_vocab, (num_samples,))
        else:
            indices = t.tensor(bags_of_words[random_seed])

        n_layers, n_heads, d_model, d_head = model.W_Q.shape

        # assert False, "TODO: add Q and K and V biases???"
        EE_QK_circuit_sample = einops.einsum(
            EE_QK_circuit.A[indices, :],
            EE_QK_circuit.B[:, indices],
            "num_query_samples d_head, d_head num_key_samples -> num_query_samples num_key_samples"
        ) / np.sqrt(d_head)

        if mean_version:
            # we're going to take a softmax so the constant factor is arbitrary 
            # and it's a good idea to centre all these results so adding them up is reasonable
            EE_QK_mean = EE_QK_circuit_sample.mean(dim=1, keepdim=True)
            EE_QK_circuit_sample_centered = EE_QK_circuit_sample - EE_QK_mean 
            EE_QK_circuit_result += EE_QK_circuit_sample_centered.cpu()

        else:
            EE_QK_softmax = t.nn.functional.softmax(EE_QK_circuit_sample, dim=-1)
            EE_QK_circuit_result += EE_QK_softmax.cpu()

    EE_QK_circuit_result /= random_seeds

    if show_plot:
        imshow(
            EE_QK_circuit_result,
            labels={"x": "Source/Key Token (embedding)", "y": "Destination/Query Token (unembedding)"},
            title=f"EE QK circuit for head {layer_idx}.{head_idx}",
            width=700,
        )

    return EE_QK_circuit_result

#%%

def get_single_example_plot(
    layer, 
    head,
    sentence="Tony Abbott under fire from Cabinet colleagues over decision",
):
    tokens = model.tokenizer.encode(sentence)
    pattern = get_EE_QK_circuit(
        layer,
        head,
        random_seeds=None,
        num_samples=None,
        show_plot=True,
        bags_of_words=[tokens],
        mean_version=False,
    )
    imshow(
        pattern, 
        x=sentence.split(" "), 
        y=sentence.split(" "),
        title=f"Unembedding Attention Score for Head {layer}.{head}",
        labels = {"y": "Query (W_U)", "x": "Key (W_EE)"},
    )

NAME_MOVERS = {
    "gpt2": [(9, 9), (10, 0), (9, 6)],
    "SoLU_10L1280W_C4_Code": [(7, 12), (5, 4), (8, 3)],
}[model.cfg.model_name]

NEGATIVE_NAME_MOVERS = {
    "gpt2": [(LAYER_IDX, HEAD_IDX), (11, 10)],
    "SoLU_10L1280W_C4_Code": [(LAYER_IDX, HEAD_IDX), (9, 15)], # second one on this one IOI prompt only...
}[model.cfg.model_name]

for layer, head in NAME_MOVERS + NEGATIVE_NAME_MOVERS:
    get_single_example_plot(layer, head)

# %%
        
# Prep some bags of words...
# OVERLY LONG because it really helps to have the bags of words the same length

bags_of_words = []

OUTER_LEN = 100
INNER_LEN = 10

idx = -1
while len(bags_of_words) < OUTER_LEN:
    idx+=1
    cur_tokens = model.tokenizer.encode(dataset[idx])
    cur_bag = []
    
    for i in range(len(cur_tokens)):
        if len(cur_bag) == INNER_LEN:
            break
        if cur_tokens[i] not in cur_bag:
            cur_bag.append(cur_tokens[i])

    if len(cur_bag) == INNER_LEN:
        bags_of_words.append(cur_bag)

#%%

for idx in range(OUTER_LEN):
    print(model.tokenizer.decode(bags_of_words[idx]), "ye")
    softmaxed_attn = get_EE_QK_circuit(
        LAYER_IDX,
        HEAD_IDX,
        show_plot=True,
        num_samples=None,
        random_seeds=None,
        bags_of_words=bags_of_words[idx:idx+1],
        mean_version=False,
    )

#%% [markdown]
# <p> Observe that a large value of num_samples gives better results </p>

# WARNING: ! below here is with random words

for num_samples, random_seeds in [
    (2**i, 2**(10-i)) for i in range(4, 11)
]:
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer, head in tqdm(list(itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):

        bags_of_words = None
        mean_version = False

        softmaxed_attn = get_EE_QK_circuit(
            layer,
            head,
            show_plot=False,
            num_samples=num_samples,
            random_seeds=random_seeds,
            bags_of_words=bags_of_words,
            mean_version=mean_version,
        )
        if mean_version:
            softmaxed_attn = t.nn.functional.softmax(softmaxed_attn, dim=-1)
        trace = einops.einsum(
            softmaxed_attn,
            "i i -> ",
        )
        results[layer, head] = trace / softmaxed_attn.shape[0] # average attention on "diagonal"
    
    imshow(results, title=f"num_samples={num_samples}, random_seeds={random_seeds}")

#%% [markdown]
# <p> Most of the experiments from here are Arthur's early experiments on 11.10 on the full distribution </p>

logits, cache = model.run_with_cache(
    ioi_dataset.toks,
    names_filter = lambda name: name.endswith("hook_result"),
)

# %%

unembedding = model.W_U.clone()

# %% [markdown]
# <p> 10.7 is a fascinating head because it REVERSES direction in the backup step</p>
# <p> What is it doing on the IOI distribution? </p>

#%%

LAYER_IDX = 10
HEAD_IDX = 7
HOOK_NAME = f"blocks.{LAYER_IDX}.attn.hook_result"

head_output = cache["result", LAYER_IDX][t.arange(N), ioi_dataset.word_idx["end"], HEAD_IDX, :]

head_logits = einops.einsum(
    head_output,
    unembedding,
    "b d, d V -> b V",
)

for b in range(10, 12):
    print("PROMPT:")
    print(ioi_dataset.tokenized_prompts[b])

    for outps_type, outps in [
        ("TOP TOKENS", t.topk(head_logits[b], 10).indices),
        ("BOTTOM_TOKENS", t.topk(-head_logits[b], 10).indices),
    ]:
        print(outps_type)
        for i in outps:
            print(ioi_dataset.tokenizer.decode(i))
    print()

print("So it seems like the bottom tokens (checked on more prompts, seems legit) are JUST the correct answer, and the top tokens are not interpretable")

# %% [markdown]
# <p> Okay let's look more generally at OWT...</p>

#%%

# Let's see some WEBTEXT
raw_dataset = load_dataset("stas/openwebtext-10k")
train_dataset = raw_dataset["train"]
dataset = [train_dataset[i]["text"] for i in range(len(train_dataset))]

dataset = [
    """Mormon Church Opens in Ozark as Mormon Faith Grows VideoOZARK, Mo. -- A new Mormon church has opened 
its doors in Ozark.It's the new home for the Ozark and Nixa congregations of The Church of Jesus Christ of the 
Latter-Day Saints.The growth of the Mormon faith has created a need for the new church in our area.The LDS church 
in Ozark has had its doors open for about two weeks now.Between the Nixa and Ozark wards, there are about 800 
people who have already joined."It's a worldwide congregation of about 15 million people and growing," said Ozark 
Ward Bishop Robert Guison.And the growth has been proven here in the Ozarks with a need for the brand new 
church."When i was young growing up in Monett, we had to come to Springfield," said Bishop Guison. "There are now 
seven stakes and each stake is approximately six to seven congregations-- so that's how much it's grown in my 
lifetime-- just in this area-- so there's been a tremendous amount of growth.""It's been exciting to see the growth
and people moving to the area," said Nixa Ward Bishop Michael Barker. "But also the natural growth from families --
and also those that choose to join the Mormon faith-- those that choose to investigate and join themselves to the 
church-- it's been fun."Before, those of the Mormon faith in Nixa and Ozark had to drive to Springfield."Then, as 
the need grew, we were too crowded in that building," said Bishop Guison. "So the next choice was what would be the
most benefit to members and the community? That's why this location was chosen.""It gives a foundation to our 
faith, a sense of permanence," said Bishop""",
    """Hire a senior Perl / Python programmer today; download my up-to-date resume (PDF)A scorpion giving 
birthThanks for the digg! Anyway, to answer some question I noticed in the comments: the babies do not come out of 
the mouth of the mother, but from an opening near the pectines (featherlike structures underneath the scorpion). 
The first photo on Scans of Diplocentrus species shows the pectines to the left and the right of a rectangular 
piece with a "half-round dent". The dent is where the opening is located.The mother might eat her babies, but this 
mostly happens when she is stressed too much. If you're interested in scorpions in general (or other arachnids like
tarantulas, vinegaroons, etc), this site has a lot of related photos, nature walks, etc. The easiest way to find 
something of interest is to use google, for example: site:johnbokma.com scorpions.RelatedToday, after Esme and I 
had returned from some shopping at Wal-Mart, I discovered that one scorpion was giving birth: a Diplocentrus 
species, probably Diplocentrus melici, which we had captured the 23rd of April, 2006. Later that day we captured 
another female. The latter gave birth some time ago, but shortly after died. My best guess is that it somehow got 
ill, and aborted the""",
    """*10 P.M. Update*Texas A&M's former mascot, Reveille VII, has died.The former First Lady of Aggieland 
died Thursday morning after undergoing emergency surgery at the Texas A&M Veterinary Hospital.Reveille VII served 
as Texas A&M's Mascot from 2001 to 2008.She died after complications from an ulcer and pneumonia.Now the former 
highest ranking member of the Corps of Cadets is being remembered for her contributions to the Aggie spirit.She 
came our way from Florida becoming Texas A&M's 7th mascot Reveille in May 2001.A precious puppy that would become 
the First Lady of the Corps of Cadets.James Mulvey was a member of the Corps when News 3 interviewed him in 2006 
and was proud to be her handler."I leave the dorm about 15 minutes early. I know somebody or a group of people is 
going to stop me," he said at the time.From chewing on Bevo along the sidelines during the Lone Star Showdown to 
even special birthday parties like her 10th in October 2010.Reveille posed for many pictures but was known to bark 
at strangers."That is her paw, her official paw print," said Tina Gardner as she looked through a scrapbook.Tina 
and Paul Gardner of College Station took care of her in her retirement years from 2008 till her passing.She became 
ill this week and Thursday morning died of complications from an ulcer.'I wouldn't say she was spoiled. She was 
definitely regal," said Paul Gardner."She was spoiled," laughed Tina.Recently Reveille began water treadmill 
treatment at the Texas A&M Vet School for arthritis.The Gardners were already planning her 13th birthday for this 
fall and were planning a "Bark mitzvah," as they are Jewish."Maybe she wanted the party part of the "Bark mitzvah,"
but she really didn't want to have to learn the Hebrew part and so she just chose to go to heaven early," added 
Tina Gardner.Funeral arrangements have not been set yet but are expected to happen in September once school resumes
with a final resting place in front of Kyle Field looking at the scoreboard.She would have turned 13 on October 
9th.The Gardners say the Reveille Cemetery will not be impacted by renovations at Kyle Field.Texas A&M Corps of 
Cadets StatementThe Office of the Commandant and the Corps of Cadets are deeply saddened to hear of the passing of 
Reveille VII, the former ��First Lady of Aggieland�� today in College Station. As Aggies we are all very proud of 
our mascot, and we have great respect for her and all the tradition that she represents. We in the Corps of Cadets 
are especially fond of Reveille, as she has been part of the Corps from the beginning, lives with the Corps every 
day as a member of Company E-2, marches with the Corps at all march-ins and parades, and is the highest ranking 
member of the Corps of Cadets. We will remember with great fondness all the joy that Reveille VII brought to all 
Aggies during her time as our mascot, and will remember her excited barks every time our football team scored a 
touchdown. We know that she will continue to do so in the future as she joins the other Reveilles in the North end 
of Kyle Field where she will always be able to see the scoreboard and bark for her team. Rest in Peace Miss 
Reveille. You will be missed but never forgotten…Reveille*Previous Story*The former mascot of Texas A&M, Reveille 
VII, has died, according to the university.Reveille was taken to the veterinary school on campus earlier in the 
week and eventually had to undergo emergency surgery. She died Thursday morning.The collie was born on October 9, 
2000. She was bred in Florida and located by the university after a nationwide search. She officially became A&M's 
seventh mascot by the Reveille name in May 2001.Reveille VII retired in the summer of 2008, and had been living 
with local residents Paul and Tina Gardner.Details on services for the dog are still being worked out.Known as the 
First Lady of Aggieland and the highest ranking member of the university's Corps of Cadets, the Reveille serving as
the mascot lives with a cadet on campus and goes to classes with the student.The""",
    """Senators send letter to Roger GoodellSteve Delsohn reports the details surrounding the domestic 
violence charge against Panthers DE Greg Hardy.Sixteen female U.S. senators have sent a letter to commissioner 
Roger Goodell calling for a "real zero-tolerance policy" against domestic violence in the NFL.The letter was sent 
to Goodell on Thursday. In it, the senators say they were "shocked and disgusted" by the video released Monday of 
former Baltimore Ravens running back Ray Rice striking his then-fiancée Janay Palmer in an Atlantic City, New 
Jersey, casino elevator and a subsequent report by The Associated Press that a league executive received the video 
from a law enforcement official in April."We are deeply concerned that the NFL's new policy, announced last month, 
would allow a player to commit a violent act against a woman and return after a short suspension," the letter 
reads. "If you violently assault a woman, you shouldn't get a second chance to play football in the NFL."The NFL's 
current policy sends a terrible message to players, fans and all Americans that even after committing a horrific 
act of violence, you can quickly be back on the field."The letter ends with a call for the NFL "to institute a real
zero-tolerance policy and send a strong message that the league will not tolerate violence against women by its 
players, who are role models for children across America."The letter was put together by Sen. Barbara Boxer, 
D-Calif., and was signed by 14 Democrats and two Republicans.President Barack Obama's press secretary released a 
statement earlier in the week, calling the issue of domestic violence "bigger than football.""The president is the 
father of two daughters. And like any American, he believes that domestic violence is contemptible and unacceptable
in a civilized society," the statement said. "Hitting a woman is not something a real man does, and that's true 
whether or not an act of violence happens in the public eye, or, far too often, behind closed doors. Stopping 
domestic violence is something that's bigger than football -- and all of us have a responsibility to put a stop to 
it."Ray Anderson, formerly the vice president of football operations under Goodell, expressed his displeasure with 
the turn of events inside league headquarters."I am personally very disappointed that the leadership at the NFL's 
New York office seems to be swirling around in chaos," Anderson told Arizona Sports 98.7 FM. "That's sad, because 
there are too many good people working there that don't deserve that and I'll just leave it at that."Anderson later
added, "In my time in the league, I thought there was an appropriate moral compass. I struggle now because I'm not 
sure I have as much faith that is occurring."The NFL has denied receiving the""",
    """"""
]

# %%

# In this cell I look at the sequence positions where the
# NORM of the 10.7 output (divided by the layer norm scale)
# is very large across several documents
# 
# we find that 
# i) just like in IOI, the top tokens are not interpretable and the bottom tokens repress certain tokens in prompt
# ii) unlike in IOI it seems that it is helpfully blocks the wrong tokens from prompt from being activated - example:
# 
# ' blacks', ' are', ' arrested', ' for', ' marijuana', ' possession', ' between', ' four', ' and', ' twelve', ' times', ' more', ' than'] -> 10.7 REPRESSES " blacks"

contributions = []

for i in tqdm(range(100)):
    tokens = model.tokenizer(
        dataset[i], 
        return_tensors="pt", 
        truncation=True, 
        padding=True
    )["input_ids"].to(DEVICE)
    
    # if tokens.shape[1] < 256: # lotsa short docs here
    #     print("SKIPPING short document", tokens.shape)
    #     continue

    tokens = tokens[0:1]

    model.reset_hooks()
    logits, cache = model.run_with_cache(
        tokens,
        names_filter = lambda name: name in [HOOK_NAME, "ln_final.hook_scale"],
    )
    output = cache[HOOK_NAME][0, :, HEAD_IDX] / cache["ln_final.hook_scale"][0, :, 0].unsqueeze(dim=-1) # account for layer norm scaling
    
    contribution = einops.einsum(
        output,
        unembedding,
        "s d, d V -> s V",
    )
    contributions.append(contribution.clone())

    for j in range(len(contribution)):
        if j == len(contribution)-1: # contribution[j].norm().item() > 80:
            print(model.to_str_tokens(tokens[0, j-30: j+1]))
            # print(model.tokenizer.decode(tokens[0, j+1]))
            print()

            top_tokens = t.topk(contribution[j], 10).indices
            bottom_tokens = t.topk(-contribution[j], 10).indices

            print("TOP TOKENS")
            for i in top_tokens:
                print(model.tokenizer.decode(i))
            print()
            print("BOTTOM TOKENS")
            for i in bottom_tokens:
                print(model.tokenizer.decode(i))

full_contributions = t.cat(contributions, dim=0)

# %%

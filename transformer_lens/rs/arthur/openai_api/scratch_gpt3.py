# %% [markdown] [4]:

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
from transformer_lens.rs.arthurs_notebooks.arthur_utils import get_metric_from_end_state
import openai
from torch.distributions.categorical import Categorical

#%%

model = HookedTransformer.from_pretrained("gpt2")

# %%

all_keys = []
while True:
    api_key = input("Enter API keys (enter string <10 chars to stop)>")
    if len(api_key) <= 10:
        break
    all_keys.append(api_key)
    openai.api_key = api_key
#%%

prompt = """"Greece's New Currency\n\nGreece is on its second bailout, with the possibility of a third bailout looming. The Guardian reports on a man in Greece who received groceries and tax services this month without spending a euro:\n\nIn return for his expert labour [as an electrician], Mavridis received a number of Local Alternative Units (known as tems in Greek) in his online network account. In return for the eggs, olive oil, tax advice and the rest, he transferred tems into other people’s accounts.\n\nTems is an alternative currency created by the Exchange and Solidarity Network. All transactions are recorded electronically via the network’s website. To avoid debt or hoarding, Tems has a debt floor of 300 units, and a ceiling of 1,200. Like time banks, it’s not a barter system as there’s a medium of exchange. However, the group does plan on opening a barter market in the following months.\n\nThe article implies that the Greek government is in support of such currency innovations:"""
str_tokens = model.to_str_tokens(prompt)
tokens = model.to_tokens(prompt).squeeze().tolist()
tensor_tokens = torch.LongTensor(tokens).unsqueeze(0).cuda()

#%%

response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=0,
    logprobs=10,
    echo=True,
    logit_bias={"38": -100},
)

# %%

model2 = HookedTransformer.from_pretrained("gpt2-large")

# %%

small_logits = model(tensor_tokens).squeeze()
large_logits = model2(tensor_tokens).squeeze()

#%%

# from torch.distributions.categorical import Categorical
# dist = Categorical(logits = logits)
# torch.distributions.kl.kl_divergence(p, q)
# entropy = dist.entropy()
# # Can also do things like...
# single_sample = dist.sample()

#%%

def get_labels_true_kls_est_kls(str_tokens, small_logits, large_logits, k=5, include_top_small=False, rescaling=False):
    labels = []
    true_kls = []
    est_kls = []

    for i in range(small_logits.shape[0]):
        small_dist = Categorical(logits=small_logits[i])
        large_dist = Categorical(logits=large_logits[i])
        true_kl = torch.distributions.kl.kl_divergence(large_dist, small_dist)
        top_large = torch.topk(large_dist.probs, k = k).indices.squeeze().tolist()

        if include_top_small:
            top_small = torch.topk(small_dist.probs, k = k).indices.squeeze().tolist()
            tops = list(set(top_large + top_small))
        else:
            tops = top_large

        kl_estimate = sum([
            (large_dist.probs[token_idx].item() * (large_dist.probs[token_idx] / small_dist.probs[token_idx]).log().item()) for token_idx in tops
        ])
        labels.append(str(i) + " |" + str_tokens[i] + "|" + ("<|NONE|>" if i == small_logits.shape[0]-1 else str_tokens[i+1]) + "|")
        true_kls.append(true_kl.item())
        est_kls.append(kl_estimate)

    return labels, true_kls, est_kls

# %%


def show_figure(k=5, include_top_small=False):
    labels, true_kls, est_kls = get_labels_true_kls_est_kls(str_tokens, small_logits, large_logits, k=k, include_top_small=include_top_small)
    fig = go.Figure()
    fig.add_scatter(
        x = true_kls,
        y = est_kls,
        mode = "markers",
        text = labels,
    )

    minx = min(min(true_kls), min(est_kls))
    maxx = max(max(true_kls), max(est_kls))

    fig.add_scatter(
        x = [minx, maxx],
        y = [minx, maxx],
        mode = "lines",
        line = dict(color = "black"),
    )

    fig.update_layout(
        xaxis_title = "True KL",
        yaxis_title = "Estimated KL",
        title = f"KL Divergence between GPT-2 Small and GPT-2-Large, with estimate from top {k} tokens from large {'and small output distributions' if include_top_small else ''}",
    )

    fig.show()

# %%

show_figure(k=5, include_top_small=False)

#%%

show_figure(k=100, include_top_small=False)

# %%

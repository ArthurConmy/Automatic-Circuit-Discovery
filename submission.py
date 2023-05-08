#%% 

# !pip install transformer_lens

#%% [markdown]
# <h1>Arthur's SERI MATS Application</h1>
# <p>This notebook contains my <10h project (about 8 hours stopwatch time, done over 12 hours. Neither of these numbers include writing time)</p>

# <h2>Title: How does factual information flow through language models?</h2>

# <p>Geva et. al (2023) refine the hypothesis in the ROME paper that factual information flows through models via early(-mid) layer MLPs and late-layer attention heads. Specifically, they find that “subject-enrichment” occurs at the final subject position by MLPs, and late-layer attention heads extract attributes (an operation more complex than mere copying).</p>

# <p>Link: https://arxiv.org/pdf/2304.14767.pdf</p>

# <p>However, Geva et. al’s analysis is limited. In Section 5 (in the “Attention Knockout” paragraph), they intervene on attention layers by overwriting attention scores. This neglects i) the effect of individual attention heads and ii) the effect of individual earlier attention heads or layers (rather than simply ALL earlier attention heads/layers). Similarly, Section 7.3 performs a patching experiment that involves setting all future layers’ attention inputs to a particular value. Again isolating individual layer-to-layer (or head-to-head) effects could localize behavior more.</p>

# <p> In this notebook we firstly refine the later layer model components that matter and then find the early layer components that matter.

# %% [markdown]
# <h2>Setup</h2>

import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2") 

from copy import deepcopy
import acdc
from acdc.graphics import show_pp
import IPython
from functools import partial
import torch
import json
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "colab"
from IPython import get_ipython
import transformer_lens
torch.autograd.set_grad_enabled(False)

def show_pp(
    m,
    xlabel="",
    ylabel="",
    title="",
    bartitle="",
    animate_axis=None,
    highlight_points=None,
    highlight_name="",
    return_fig=False,
    show_fig=True,
    **kwargs,
):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if animate_axis is None:
        fig = px.imshow(
            m,
            title=title if title else "",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    else:
        fig = px.imshow(
            einops.rearrange(m, "a b c -> a c b"),
            title=title if title else "",
            animation_frame=animate_axis,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=bartitle,
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=300,
            yanchor="top",
            y=1,
            ticks="outside",
        ),
    )

    if highlight_points is not None:
        fig.add_scatter(
            x=highlight_points[1],
            y=highlight_points[0],
            mode="markers",
            marker=dict(color="green", size=10, opacity=0.5),
            name=highlight_name,
        )

    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        xaxis_range=[-0.5, m.shape[1] - 0.5],
        showlegend=True,
        legend=dict(x=-0.1),
    )
    if highlight_points is not None:
        fig.update_yaxes(range=[m.shape[0] - 0.5, -0.5], autorange=False)
    if show_fig:
        fig.show()
    if return_fig:
        return fig

#%% [markdown]
# <h2>Select dataset and model</h2>

# <p> We chose GPT-2-XL as the model and used a dataset of prompts from counterfact that had a similar format: "The mother tongue of TOKEN5 TOKEN6 TOKEN7 TOKEN8 is". We chose this as it was able to reproduce Figure 2 from Geva et al (and so is likely representative of CounterFact as a whole).</p>

PATH_TO_COUNTERFACT = "/mnt/ssd-0/arthurworkspace/TransformerLens/dist/counterfact.json"
# download this from https://rome.baulab.info/data/dsets/counterfact.json
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-xl")
model.set_use_attn_result(True)

# some util functions
def show_tokens(tokens):
    # Prints the tokens as text, separated by |
    if type(tokens) == str:
        # If we input text, tokenize first
        tokens = model.to_tokens(tokens)
    text_tokens = [model.tokenizer.decode(t) for t in tokens.squeeze()]
    print("|".join(text_tokens))

def sample_next_token(
    model: transformer_lens.HookedTransformer, input_ids: torch.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0, cache=None
) -> torch.Tensor:
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    model.eval()
    with torch.inference_mode():
        all_logits = model(input_ids.unsqueeze(0))  # TODO: cache
    B, S, E = all_logits.shape
    logits = all_logits[0, -1]
    return logits

# sampling example
input = "The Eiffel Tower is in"
input_tokens = torch.tensor(model.tokenizer.encode(input))

logits = sample_next_token(model, input_tokens.long().to("cuda"))

values, indices = torch.topk(logits, k=20)
print(f"Model name: {model.cfg.model_name}")
print(f"Input: {input}")
print(f"token {'':<9} logits")
for i in range(20):
    print(f"{model.tokenizer.decode(indices[i]) :<15} {values[i].item()}")

import os
with open(os.path.expanduser(PATH_TO_COUNTERFACT), "rb") as f:
    counterfact = json.load(f)
ranks = []
prompts = [c["requested_rewrite"]["prompt"] for c in counterfact]
pdict = {}
for i, p in enumerate(prompts):
    if p not in pdict:
        pdict[p] = [i]
    pdict[p].append(i)

ids = pdict["The mother tongue of {} is"]

lens = {i : 0 for i in range(len(ids))}
ids2 = []

for i in ids:
    data = counterfact[i]
    rr = data["requested_rewrite"]
    cur = " "+rr["subject"]
    print(cur)
    tokens = model.tokenizer.encode(cur)
    lens[len(tokens)] += 1

    if len(tokens) == 4:
        ids2.append(i)

data = []
labels = []

for datapoint in [counterfact[i] for i in ids2]:
    rr = datapoint["requested_rewrite"]
    input = rr["prompt"].format(rr["subject"])
    target = " " + rr["target_true"]["str"]
    false_target = " " + rr["target_new"]["str"]
    input_tokens = model.to_tokens(input, prepend_bos=True, move_to_device=False)[0]
    target_tokens = model.to_tokens(target, prepend_bos=True, move_to_device=False)[0]
    false_target_tokens = model.to_tokens(false_target, prepend_bos=True)[0]
    logits = sample_next_token(model, input_tokens.long())
    top_token = torch.argmax(logits).item()

    if target_tokens[-1].item() == 4302:
        target_tokens[-1] = 13624 # Christian -> Christianity, this makes more sense

    data.append(input_tokens)
    labels.append(target_tokens[1:])

data = torch.stack(tuple(row for row in data)).long().to("cuda")
labels = torch.stack(tuple(row for row in labels)).long().to("cuda")
labels = labels.squeeze(-1) # can't see why you left an extra dim...
if get_ipython() is None:
    length = data.shape[0] // 2 # trim size cos memory
else:
    length = data.shape[0]
data = data[:length]
labels = labels[:length]

# Make a good baseline
patch_data = model.to_tokens("The obvious feature about that person over there is", prepend_bos=True)
patch_data = patch_data[0].long().to("cuda")
patch_data = patch_data.unsqueeze(0).repeat(data.shape[0], 1)
assert patch_data.shape == data.shape, (patch_data.shape, data.shape)

print("All the facts:")
for i in range(len(data)):
    print(model.tokenizer.decode(data[i]))
seq_len = data.shape[1]
N = data.shape[0]
old_data = deepcopy(data).cpu()
logits = model(data)

original_probs = torch.nn.functional.softmax(logits, dim=-1)

# correct_log_probs = log_probs[torch.arange(len(labels)).to(log_probs.device), -1, labels.to(log_probs.device)] 
device = original_probs.device
labels = labels.to(device).view(-1, 1, 1)

# Replace 1 with 0 in the gather() call to simulate the incorrect version
labels = labels.squeeze(-1).squeeze(-1)
new_correct_probs = original_probs[torch.arange(len(labels)), -1, labels]

# %%

# <h2>Which model components directly affect the output?</h2>

# <h3>Reproduce Figure 2 from the paper</h3>
    
relevant_positions = {
    " is": 9,
    " subject_end": 8,
    " subject_start": 8-4,
}
    
def mask_attention(z, hook, key_pos, head_no=None):
    # print(z.shape) # batch heads query (I think) key
    assert relevant_positions[" is"] == z.shape[2]-1, (relevant_positions, z.shape)

    if head_no is None:
        z[:, :, -1, key_pos] = 0
    else:
        z[:, head_no, -1, key_pos] = 0

answers = []
# heads = torch.max(matrix_answers, dim=-1).indices

for i in tqdm(range(4, model.cfg.n_layers-4)):
    # Reproduce Figure 2 from the paper?

    model.reset_hooks()

    for layer in range(i-4, i+5):
        # for pos in range(relevant_positions[" subject_start"], relevant_positions[" subject_end"]+1):
        for pos in [relevant_positions[" subject_end"]]:
            model.add_hook(
                f"blocks.{layer}.attn.hook_pattern",
                partial(mask_attention, key_pos=pos, head_no=None),
            )

    logits = model(data)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    correct_probs = probs[torch.arange(len(labels)).to(probs.device), -1, labels.to(probs.device)]
    assert len(list(correct_probs.shape))==1, probs.shape
    answers.append(correct_probs.sum().cpu()) 

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[i for i in range(4, model.cfg.n_layers-4)],
        y=[a.cpu() for a in answers],
    ))
# add title
fig.update_layout(
    title_text=f"Effect of masking attention to the end position of the subject",
    xaxis_title="Layer",
    yaxis_title="Sum of correct probs",
)
print("Baseline is", new_correct_probs.sum().cpu().item())
fig.show()

# %%

# Observation 1: the last few layers seem harmful for performance:

def patch_out(z, hook, positions=[]):
    for pos in positions:
        z[:, pos] = corrupted_cache[hook.name][:, pos]

def zero_out(z, hook, positions=[]):
    for pos in positions:
        z[:, pos] = 0.0

if get_ipython() is not None:
    answers = []

    for layer in tqdm(range(model.cfg.n_layers-1)):
        # ooh quite a lot like path patch
        model.reset_hooks()
        for layer_prime in range(layer+1, model.cfg.n_layers):
            for hook_name in [
                f"blocks.{layer_prime}.hook_attn_out",
                f"blocks.{layer_prime}.hook_mlp_out",
            ]:
                model.add_hook(hook_name, partial(zero_out, positions=[-1])) # actually should this be subject_end etc.?

        logits = model(data)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        correct_probs = probs[torch.arange(len(labels)).to(probs.device), -1, labels.to(probs.device)]
        # correct_log_probs = torch.log(correct_probs)
        answers.append(correct_probs.sum().cpu())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(model.cfg.n_layers-1)],
            y=[a.cpu() for a in answers],
    ))

    # add title
    fig.update_layout(
        title_text=f"Zeroing out all layers l_prime >= l",
        xaxis_title="Layer",
        yaxis_title="Sum of correct probs",
    )
    fig.show()

#%%
# <h2>Cache some valuable things</h2>

correct_direction = model.unembed.W_U[:, labels]
cache = {}
def cacher(z, hook):
    cache[hook.name] = z.clone()
    return z

model.reset_hooks()
for layer in range(model.cfg.n_layers):
    for name in [
        f"blocks.{layer}.attn.hook_result",
        f"blocks.{layer}.hook_mlp_out",
        f"blocks.{layer}.hook_resid_post",
        f"blocks.{layer}.attn.hook_pattern",    
    ]:
        model.add_hook(
            name,
            cacher,
        )
logits = model(data)
model.reset_hooks()

#%%

# Observation 2: there are a very sparse set of attention heads that direcly effect model output, mainly between Layers 30 and 43. Additionally, there is generally only one attention head per layer that is important.
# We use the logit lens per head, then extract the "probability" each head assigns to the correct answer this way (cf https://www.lesswrong.com/posts/6tHNM2s6SWzFHv3Wo/understanding-time-in-gpt-a-mechanistic-interpretability)

answers = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))

for layer in range(model.cfg.n_layers):

    results = torch.einsum(
        "bhd,dv->bhv",
        cache[f"blocks.{layer}.attn.hook_result"][:, -1],
        model.unembed.W_U,
        # correct_direction, # - incorrect_direction.unsqueeze(-1),
    )

    if True:
        results = torch.nn.functional.softmax(results, dim=-1) # [torch.arange(len(labels)), :, labels]
        correct_probs = results[torch.arange(len(labels)).to(results.device), :, labels.to(results.device)]

    for head in range(model.cfg.n_heads):
        answers[layer, head] = correct_probs[:, head].mean()

fig = show_pp(
    answers,
    return_fig=True,
)

# %%

# Observation 3: the MLPs that don't seem significant to this task for direct connection to the output

answers = torch.zeros((model.cfg.n_layers, 1))

for layer in range(model.cfg.n_layers):
    results = torch.einsum(
        "bd,dv->bv",
        cache[f"blocks.{layer}.hook_mlp_out"][:, -1],
        model.unembed.W_U,
        # correct_direction, # - incorrect_direction.unsqueeze(-1),
    )

    if True:
        results = torch.nn.functional.softmax(results, dim=-1)
        correct_probs = results[torch.arange(len(labels)).to(results.device), labels.to(results.device)]

    answers[layer] = correct_probs.mean()

fig = show_pp(
    answers,
    return_fig=True,
)

# %%


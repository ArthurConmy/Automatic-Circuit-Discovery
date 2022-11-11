# %%
from functools import *
import os
from collections import OrderedDict
import einops
import graphviz
import html
import itertools
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = "vscode"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# %%
from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
)

os.chdir("/home/ubuntu/unity")
from easy_transformer_utils import (
    show_tokens,
    sample_next_token,
    get_topk_completions,
    show_pp,
    show_attention_patterns,
    get_OV_circuit_output,
    get_bigram_freq_output,
)

os.chdir("/home/ubuntu/Easy-Transformer")

# %% Load model

model_name = "gpt2"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']
model = EasyTransformer.from_pretrained(model_name)  # , use_attn_result=True)
if torch.cuda.is_available():
    model.to("cuda")
model.set_use_attn_result(True)

# %%
webtext = load_dataset("stas/openwebtext-10k")

owt_toks = []
for seq in tqdm(webtext["train"]["text"]):
    owt_toks.append(model.to_tokens(seq)[0])
owt_toks = torch.cat(owt_toks, dim=0)
owt_toks = einops.rearrange(owt_toks[:-54], "(b n) -> b n", b=11253)
owt_toks.shape  # (11253, 1000)
# %%
toks = owt_toks[:10].to("cuda")
logits = model(toks)[:, :-1]
log_softmax = nn.LogSoftmax(dim=-1)
log_probs = log_softmax(logits)
loss = -torch.gather(log_probs, 2, toks[:, 1:].unsqueeze(2))

# %%
plt.hist(loss.cpu().numpy().flatten(), bins=100)
# %%
loss_f = loss.squeeze().cpu().flatten()
values, indices = torch.topk(loss_f, k=1000, largest=False)
# %%
for index in indices[540:550]:
    b = index // 999
    s = index % 999
    print("Text: ")
    show_tokens(model, toks[b, s - 10 : s + 10])
    print("")
    print("Low loss token: ", model.tokenizer.decode(toks[b, s]))
    print("")

# %%
from IPython.core.display import display, HTML

display(HTML("<h1>Hello, world!</h1>"))
# %%
def get_html(tokens: torch.Tensor) -> str:
    tokens = tokens.cuda()
    logits = model(tokens)[:, :-1]
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    loss = -torch.gather(log_probs, 2, tokens[:, 1:].unsqueeze(2))
    tokens = tokens[:, :-1].flatten()
    loss = loss.flatten()
    assert len(tokens) == len(loss), (len(tokens), len(loss))
    html_str = ""
    for i, tok in enumerate(tokens):
        frac = min(loss[i].item() ** 0.5 / 3, 1.0)
        word = repr(model.tokenizer.decode(tok))[1:-1]
        html_str += f'<span style="background-color: rgba({str(255 * frac)}, 0, 0, 1); font-size: 120%; white-space: pre-wrap">{html.escape(word)}</span>'
    return html_str


# %%
idx = np.random.choice(11252)
html_str = get_html(owt_toks[idx : idx + 1, :])
display(HTML(html_str))
# %%

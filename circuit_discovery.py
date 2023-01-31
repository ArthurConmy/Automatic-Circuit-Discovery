#%% [markdown]
# <p>This notebook covers the creation of an automatic circuit discovery experiment, and detailed explanation of how to generate automatic circuit pictures for the IOI task</p>

#%% [markdown]
# <h3>Sort out whether we're in a notebook or not</h3>

import os

try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
    os.system("pip install git+https://github.com/ArthurConmy/Easy-Transformer.git")

except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
# %% [markdown]
# <h2>Imports</h2>

from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from time import ctime
import einops
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import pickle
from subprocess import call
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
from easy_transformer import EasyTransformer
from easy_transformer.utils_circuit_discovery import (
    patch_all,
    direct_path_patching,
    logit_diff_io_s,
    Circuit,
    logit_diff_from_logits,
    get_datasets,
)
from easy_transformer.experiments import (
    get_act_hook,
)
from easy_transformer.ioi_utils import (
    show_pp,
)
from easy_transformer.ioi_dataset import IOIDataset
import os

file_prefix = "archive/" if os.path.exists("archive") else ""

#%% [markdown]
# <h2>Load in the model</h2>

model_name = "gpt2" # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']
model = EasyTransformer.from_pretrained(model_name)

#%% [markdown]
# <h2>Setup dataset</h2>
#
# Here, we define a dataset from IOI. We always use the same template: all sentences are of the pretty much the same form
# Don't worry about the IOIDataset object, it's just for convenience of some things that we do with IOI data.
#
# ACDC requires a dataset of pairs: default and patch datapoints. Nodes on the current hypothesis graph will receive default datapoints as input, while all other nodes will receive patch datapoints. As we test a child node for its importance, we'll swap the default datapoint for a patch datapoint, and see if the metric changes. In this implementation, we perform this test over all datapoint pairs, where the child node of interest is always receiving patch datapoints and the other children receive default datapoints.


N = 100
ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1,)

abc_Dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)

seq_len = ioi_dataset.toks.shape[1]
assert seq_len == 16, f"Well, I thought ABBA #1 was 16 not {seq_len} tokens long..."

print(
    "Example of prompts:\n\n",
    ioi_dataset.tokenized_prompts[0],
    "\n",
    ioi_dataset.tokenized_prompts[-1],
)

print(
    "\n... we also have some `ABC` data, that we use to scrub unimportant nodes:\n\n",
    abc_Dataset.tokenized_prompts[0],
    "\n",
    abc_Dataset.tokenized_prompts[-1],
)

# #%% [markdown]
# # <h2>Make the dataset</h2>

# template = "Last month it was {month} so this month it is"
# all_months = [
#     "January",
#     "February",
#     "March",
#     "April",
#     "May",
#     "June",
#     "July",
#     "August",
#     "September",
#     "October",
#     "November",
#     "December",
# ]
# sentences = []
# answers = []
# wrongs = []
# batch_size = 12
# for month_idx in range(batch_size):
#     cur_sentence = template.format(month=all_months[month_idx])
#     cur_ans = all_months[(month_idx + 1) % batch_size]
#     sentences.append(cur_sentence)
#     answers.append(cur_ans)
#     wrongs.append(all_months[month_idx])
# tokens = model.to_tokens(sentences, prepend_bos=True)
# answers = torch.tensor(model.tokenizer(answers)["input_ids"]).squeeze()
# wrongs = torch.tensor(model.tokenizer(wrongs)["input_ids"]).squeeze()

#%% [markdown]
# <h3>Make the positions labels (step 1)</h3>

# do positions, better! 

batch_size = N
positions = OrderedDict()
ones = torch.ones(size=(batch_size,)).long()
positions["IO"] = ones.clone() * 2
positions["S1"] = ones.clone() * 4
positions["S2"] = ones.clone() * 10
positions["END"] = ones.clone() * 14

#%%

def logit_diff_metric(model, dataset):
    logits = model(ioi_dataset.toks.long())

    corrects = ioi_dataset.toks.long()[:, positions["IO"][0].item()]
    wrongs = ioi_dataset.toks.long()[:, positions["S1"][0].item()]

    logits_on_correct = logits[torch.arange(batch_size), -2, corrects]
    logits_on_wrong = logits[torch.arange(batch_size), -2, wrongs]

    ans = torch.mean(logits_on_correct - logits_on_wrong)
    return ans.item()

#%% [markdown]
# Make the circuit object

h = Circuit(
    model,
    metric=logit_diff_metric,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_Dataset.toks.long(),
    threshold=0.25,
    orig_positions=positions,
    new_positions=positions, # in some datasets we might want to patch from different positions; not here
)
#%%
# <h2> Run path patching! </h2>
# <p> Only the first two lines of this cell matter; the rest are for saving images. This cell takes several minutes to run. If you cancel and then call h.show(), you can see intermediate representations of the circuit. </p>

while h.current_node is not None:
    h.step(show_graphics=False, verbose=True)

    a = h.show()
    # save digraph object
    with open(file_prefix + "hypothesis_tree.dot", "w") as f:
        f.write(a.source)

    # convert to png
    call(
        [
            "dot",
            "-Tpng",
            "hypothesis_tree.dot",
            "-o",
            file_prefix + f"gpt2_hypothesis_tree_{ctime()}.png",
            "-Gdpi=600",
        ]
    )
#%% [markdown]
# <h2> Show the circuit </h2>
h.show()
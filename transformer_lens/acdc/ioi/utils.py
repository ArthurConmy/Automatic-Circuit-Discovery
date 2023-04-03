#%%
# type: ignore
from copy import deepcopy
import torch.nn.functional as F
from typing import List
import click
import IPython
import torch
from transformer_lens.acdc.ioi.ioi_dataset import IOIDataset  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore
import wandb

with open(__file__, "r") as f:
    file_content = f.read()
    
USING_WANDB = True

#%% [markdown]
# <h2>Setup dataset</h2>
#
# Here, we define a dataset from IOI. We always use the same template: all sentences are of the pretty much the same form
# Don't worry about the IOIDataset object, it's just for convenience of some things that we do with IOI data.
#
# ACDC requires a dataset of pairs: default and patch datapoints. Nodes on the current hypothesis graph will receive default datapoints as input, while all other nodes will receive patch datapoints. As we test a child node for its importance, we'll swap the default datapoint for a patch datapoint, and see if the metric changes. In this implementation, we perform this test over all datapoint pairs, where the child node of interest is always receiving patch datapoints and the other children receive default datapoints.


N = 50
ioi_dataset = IOIDataset(
    prompt_type="ABBA",
    N=N,
    nb_templates=1,
    seed = 0,
)

abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=1)
    .gen_flipped_prompts(("S", "RAND"), seed=2)
    .gen_flipped_prompts(("S1", "RAND"), seed=3)
)

seq_len = ioi_dataset.toks.shape[1]
assert seq_len == 16, f"Well, I thought ABBA #1 was 16 not {seq_len} tokens long..."

default_data = ioi_dataset.toks.long()[:N, : seq_len - 1]
patch_data = abc_dataset.toks.long()[:N, : seq_len - 1]

# tokens_device_dtype = rc.TorchDeviceDtype("cuda:0", "int64")
# default_data = make_arr(
#     ioi_dataset.toks.long()[:N, : seq_len - 1],

# %%

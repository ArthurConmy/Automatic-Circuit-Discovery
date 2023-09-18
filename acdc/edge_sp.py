#%%

IN_COLAB = False
print("Running as a outside of colab")

import numpy # crucial to not get cursed error
import plotly

plotly.io.renderers.default = "vscode+colab"
import os # make images folder
if not os.path.exists("ims/"):
    os.mkdir("ims/")

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    print("Running as a notebook")
    ipython.run_line_magic("load_ext", "autoreload")  # type: ignore
    ipython.run_line_magic("autoreload", "2")  # type: ignore
else:
    print("Running as a script")

#%%

import wandb
import IPython
from IPython.display import Image, display
import torch
import gc
from tqdm import tqdm
import networkx as nx
import os
import torch
import huggingface_hub
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
import yaml
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
try:
    from acdc.tracr_task.utils import (
        get_all_tracr_things,
        get_tracr_model_input_and_tl_model,
    )
except Exception as e:
    print(f"Could not import `tracr` because {e}; the rest of the file should work but you cannot use the tracr tasks")
from acdc.docstring.utils import get_all_docstring_things
from acdc.acdc_utils import (
    make_nd_dict,
    reset_network,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment

from acdc.acdc_utils import (
    kl_divergence,
)
from acdc.ioi.utils import (
    get_all_ioi_things,
    get_gpt2_small,
)
from acdc.induction.utils import (
    get_all_induction_things,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.acdc_graphics import (
    build_colorscheme,
    show,
)
import argparse

#%%

# grab some induction things

num_examples = 10 if IN_COLAB else 50
seq_len = 300
# TODO initialize the `tl_model` with the right model
all_induction_things = get_all_induction_things(
    num_examples=num_examples, seq_len=seq_len, device="cuda", metric="kl_div",
)
tl_model, toks_int_values, toks_int_values_other, metric, mask_rep = (
    all_induction_things.tl_model,
    all_induction_things.validation_data,
    all_induction_things.validation_patch_data,
    all_induction_things.validation_metric,
    all_induction_things.validation_mask,
)

#%%

experiment = TLACDCExperiment(
    model=tl_model,
    threshold=0.0,
    ds=toks_int_values,
    ref_ds=None,  # This argument is the corrupted dataset from the ACDC paper. We're going to do zero ablation here so we omit this
    metric=metric,
    zero_ablation=True,
    hook_verbose=False,
)

#%%

experiment.model.reset_hooks()
experiment.setup_model_hooks(
    add_sender_hooks=True,
    add_receiver_hooks=True,
    doing_acdc_runs=False,
)

# %%

def regularizer(
    exp: TLACDCExperiment,
    gamma: float = -0.1,
    zeta: float = 1.1,
    beta: float = 2 / 3,
):
    def regularization_term(mask: torch.nn.Parameter) -> torch.Tensor:
        return torch.sigmoid(mask - beta * np.log(-gamma / zeta)).mean()

    relevant_masks = list(set(
        [e.mask] for _, e in exp.all_edges()
    ))

    return torch.mean(torch.stack(relevant_masks))

#%%


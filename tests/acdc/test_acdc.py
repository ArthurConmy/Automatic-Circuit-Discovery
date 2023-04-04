from copy import deepcopy
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Union,
    Callable,
    TypeVar,
    Iterable,
    Set,
)
import wandb
import IPython
import torch

# from easy_transformer.ioi_dataset import IOIDataset  # type: ignore
from tqdm import tqdm
import random
from functools import *
import json
import pathlib
import warnings
import time
import networkx as nx
import os
import torch
import huggingface_hub
import graphviz
from enum import Enum
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
from transformer_lens.acdc.utils import (
    make_nd_dict,
    shuffle_tensor,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from transformer_lens.acdc.TLACDCCorrespondence import TLACDCCorrespondence
from transformer_lens.acdc.TLACDCInterpNode import TLACDCInterpNode
from transformer_lens.acdc.TLACDCExperiment import TLACDCExperiment

from collections import defaultdict, deque, OrderedDict
from transformer_lens.acdc.induction.utils import (
    get_all_induction_things,
    get_model,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from transformer_lens.acdc.tracr.utils import get_tracr_model_input_and_tl_model
from transformer_lens.acdc.graphics import (
    build_colorscheme,
    show,
)

def test_induction_step_one(): # sad, currently failings. Bin search from this commit and uh test addition?
    num_examples = 400
    seq_len = 30
    device = "cuda" # CPU not supported : (

    tl_model, toks_int_values, toks_int_values_other, metric = get_all_induction_things(num_examples=num_examples, seq_len=seq_len, device=device)
    tl_model.global_cache.clear()
    tl_model.reset_hooks()

    exp = TLACDCExperiment(
        model=tl_model,
        threshold=0.3,
        using_wandb=False,
        zero_ablation=False,
        ds=toks_int_values,
        ref_ds=toks_int_values_other,
        metric=metric,
        verbose=True,
    )

    exp.step()

    present_edges = []
    edge_strengths = []

    for child_name in exp.corr.edges["blocks.1.hook_resid_post"][TorchIndex([None])]:
        for child_index in exp.corr.edges["blocks.1.hook_resid_post"][TorchIndex([None])][child_name]:
            edge_strengths.append(exp.corr.edges["blocks.1.hook_resid_post"][TorchIndex([None])][child_name][child_index].effect_size)
            if exp.corr.edges["blocks.1.hook_resid_post"][TorchIndex([None])][child_name][child_index].present:
                present_edges.append((child_name, child_index))

    assert present_edges == [
        ('blocks.1.attn.hook_result', TorchIndex([None, None, 6])),
        ('blocks.1.attn.hook_result', TorchIndex([None, None, 5])),
        ('blocks.0.hook_resid_pre', TorchIndex([None])),
    ], present_edges

    edge_strengths = torch.tensor(edge_strengths)
    expected_strengths = torch.tensor([0.019034162163734436, 0.6195546388626099, 0.841758131980896, 0.030694350600242615, 0.04168818145990372, 0.028957054018974304, 0.02783321961760521, 0.03458511456847191, 0.11434471607208252, 0.07611888647079468, 0.2855337858200073, 0.18742594122886658, 0.11805811524391174, 0.0844639241695404, 0.07739323377609253, 0.02535034716129303, 1.1878613233566284])

    assert torch.allclose(
        edge_strengths,
        expected_strengths,
    ), (edge_strengths.norm().item(), expected_strengths.norm().item(), (edge_strengths - expected_strengths).norm().item())

def test_tracr_port():
    expected_im = torch.tensor([[5.3888778e-07, 5.3888778e-07, 5.3888778e-07, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        3.8146973e-06, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
        0.0000000e+00],
       [2.9040084e-13, 2.9040084e-13, 9.9999940e-01, 1.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 2.5000000e-01, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00],
       [2.9040084e-13, 9.9999940e-01, 2.9040084e-13, 0.0000000e+00,
        1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 2.5000000e-01, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00],
       [9.9999940e-01, 2.9040084e-13, 2.9040084e-13, 0.0000000e+00,
        0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 2.5000000e-01, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
        0.0000000e+00]])

    im = torch.tensor(get_tracr_model_input_and_tl_model(task="reverse", return_im = True))

    assert torch.allclose(
        im,
        expected_im,
    ), (im.norm().item(), expected_im.norm().item(), (im - expected_im).norm().item())
#%%
import gc
import json
import os
import pathlib
import random
import time
import warnings
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from enum import Enum
from functools import *
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union

import einops
import huggingface_hub
import IPython
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from plotly.subplots import make_subplots
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import HookedTransformer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from acdc.acdc_graphics import build_colorscheme, show
from acdc.acdc_utils import ct, make_nd_dict, shuffle_tensor
from acdc.docstring.utils import get_all_docstring_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.induction.utils import (
    get_all_induction_things,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
    get_validation_data,
)
from acdc.ioi.utils import get_all_ioi_things
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import Edge, EdgeType, TorchIndex
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.tracr_task.utils import get_all_tracr_things, get_tracr_model_input_and_tl_model

# these introduce several important classes !!!


@pytest.mark.slow
@pytest.mark.skip(reason="TODO fix")
def test_induction_several_steps():
    # get induction task stuff
    num_examples = 400
    seq_len = 30
    # TODO initialize the `tl_model` with the right model
    all_induction_things = get_all_induction_things(
        num_examples=num_examples, seq_len=seq_len, device="cpu"
    )  # removed some randomize seq_len thing - hopefully unimportant
    tl_model, toks_int_values, toks_int_values_other, metric = (
        all_induction_things.tl_model,
        all_induction_things.validation_data,
        all_induction_things.validation_patch_data,
        all_induction_things.validation_metric,
    )

    gc.collect()
    torch.cuda.empty_cache()

    # initialise object
    exp = TLACDCExperiment(
        model=tl_model,
        threshold=0.1,
        using_wandb=False,
        zero_ablation=False,
        ds=toks_int_values,
        ref_ds=toks_int_values_other,
        metric=metric,
        second_metric=None,
        verbose=True,
        indices_mode="reverse",
        names_mode="normal",
        corrupted_cache_cpu=True,
        online_cache_cpu=True,
        add_sender_hooks=True,  # attempting to be efficient...
        add_receiver_hooks=False,
        remove_redundant=True,
    )

    for STEP_IDX in range(10):
        exp.step()

    edges_to_consider = {
        edge_tuple: edge for edge_tuple, edge in exp.corr.all_edges().items() if edge.effect_size is not None
    }

    EDGE_EFFECTS = OrderedDict(
        [
            (
                (
                    "blocks.1.hook_resid_post",
                    TorchIndex([None]),
                    "blocks.1.attn.hook_result",
                    TorchIndex([None, None, 6]),
                ),
                0.6195546984672546,
            ),
            (
                (
                    "blocks.1.hook_resid_post",
                    TorchIndex([None]),
                    "blocks.1.attn.hook_result",
                    TorchIndex([None, None, 5]),
                ),
                0.8417580723762512,
            ),
            (
                (
                    "blocks.1.hook_resid_post",
                    TorchIndex([None]),
                    "blocks.0.attn.hook_result",
                    TorchIndex([None, None, 5]),
                ),
                0.1795809268951416,
            ),
            (
                (
                    "blocks.1.hook_resid_post",
                    TorchIndex([None]),
                    "blocks.0.attn.hook_result",
                    TorchIndex([None, None, 4]),
                ),
                0.15076303482055664,
            ),
            (
                (
                    "blocks.1.hook_resid_post",
                    TorchIndex([None]),
                    "blocks.0.attn.hook_result",
                    TorchIndex([None, None, 3]),
                ),
                0.11805805563926697,
            ),
            (
                ("blocks.1.hook_resid_post", TorchIndex([None]), "blocks.0.hook_resid_pre", TorchIndex([None])),
                0.6345541179180145,
            ),
            (
                (
                    "blocks.1.attn.hook_q",
                    TorchIndex([None, None, 6]),
                    "blocks.1.hook_q_input",
                    TorchIndex([None, None, 6]),
                ),
                1.4423644244670868,
            ),
            (
                (
                    "blocks.1.attn.hook_q",
                    TorchIndex([None, None, 5]),
                    "blocks.1.hook_q_input",
                    TorchIndex([None, None, 5]),
                ),
                1.2416923940181732,
            ),
            (
                (
                    "blocks.1.attn.hook_k",
                    TorchIndex([None, None, 6]),
                    "blocks.1.hook_k_input",
                    TorchIndex([None, None, 6]),
                ),
                1.4157390296459198,
            ),
            (
                (
                    "blocks.1.attn.hook_k",
                    TorchIndex([None, None, 5]),
                    "blocks.1.hook_k_input",
                    TorchIndex([None, None, 5]),
                ),
                1.270191639661789,
            ),
            (
                (
                    "blocks.1.attn.hook_v",
                    TorchIndex([None, None, 6]),
                    "blocks.1.hook_v_input",
                    TorchIndex([None, None, 6]),
                ),
                2.9806662499904633,
            ),
            (
                (
                    "blocks.1.attn.hook_v",
                    TorchIndex([None, None, 5]),
                    "blocks.1.hook_v_input",
                    TorchIndex([None, None, 5]),
                ),
                2.7053256928920746,
            ),
            (
                (
                    "blocks.1.hook_v_input",
                    TorchIndex([None, None, 6]),
                    "blocks.0.attn.hook_result",
                    TorchIndex([None, None, 2]),
                ),
                0.12778228521347046,
            ),
            (
                ("blocks.1.hook_v_input", TorchIndex([None, None, 6]), "blocks.0.hook_resid_pre", TorchIndex([None])),
                1.8775241374969482,
            ),
        ]
    )

    assert set(edges_to_consider.keys()) == set(EDGE_EFFECTS.keys()), (
        set(edges_to_consider.keys()) - set(EDGE_EFFECTS.keys()),
        set(EDGE_EFFECTS.keys()) - set(edges_to_consider.keys()),
        EDGE_EFFECTS.keys(),
    )

    for edge_tuple, edge in edges_to_consider.items():
        assert abs(edge.effect_size - EDGE_EFFECTS[edge_tuple]) < 1e-5, (
            edge_tuple,
            edge.effect_size,
            EDGE_EFFECTS[edge_tuple],
        )


@pytest.mark.slow
def test_main_script():
    import subprocess

    main_path = Path(__file__).resolve().parent.parent.parent / "acdc" / "main.py"

    for task in ["induction", "ioi", "tracr", "docstring"]:
        subprocess.check_call(
            ["python", str(main_path), f"--task={task}", "--threshold=1234", "--single-step", "--device=cpu"]
        )


def test_editing_edges_notebook():
    import notebooks.editing_edges


@pytest.mark.parametrize("task", ["tracr-proportion", "tracr-reverse", "docstring", "induction", "ioi", "greaterthan"])
@pytest.mark.parametrize("zero_ablation", [False, True])
def test_full_correspondence_zero_kl(
    task, zero_ablation, device="cpu", metric_name="kl_div", num_examples=4, seq_len=10
):
    if task == "tracr-proportion":
        things = get_all_tracr_things(task="proportion", num_examples=num_examples, device=device, metric_name="l2")
    elif task == "tracr-reverse":
        things = get_all_tracr_things(task="reverse", num_examples=6, device=device, metric_name="l2")
    elif task == "induction":
        things = get_all_induction_things(num_examples=100, seq_len=20, device=device, metric=metric_name)
    elif task == "ioi":
        things = get_all_ioi_things(num_examples=num_examples, device=device, metric_name=metric_name)
    elif task == "docstring":
        things = get_all_docstring_things(
            num_examples=num_examples,
            seq_len=seq_len,
            device=device,
            metric_name=metric_name,
            correct_incorrect_wandb=False,
        )
    elif task == "greaterthan":
        things = get_all_greaterthan_things(num_examples=num_examples, metric_name=metric_name, device=device)
    else:
        raise ValueError(task)

    exp = TLACDCExperiment(
        model=things.tl_model,
        threshold=100_000,
        early_exit=False,
        using_wandb=False,
        zero_ablation=zero_ablation,
        ds=things.test_data,
        ref_ds=things.test_patch_data,
        metric=things.validation_metric,
        second_metric=None,
        verbose=True,
        use_pos_embed=False,  # In the case that this is True, the KL should not be zero.
        online_cache_cpu=True,
        corrupted_cache_cpu=True,
    )
    exp.setup_corrupted_cache()

    corr = deepcopy(exp.corr)
    for e in corr.all_edges().values():
        e.present = True

    with torch.no_grad():
        out = exp.call_metric_with_corr(corr, things.test_metrics["kl_div"], things.test_data)
    assert abs(out) < 1e-6, f"{out} should be abs(out) < 1e-6"

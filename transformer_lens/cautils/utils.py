"""
Main set of utils to import at the start of all scripts and notebooks
"""

import warnings

import torch as t
import torch
warnings.warn("Setting grad enabled false...")
t.set_grad_enabled(False)

import einops
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from functools import partial
from tqdm import tqdm
from jaxtyping import Float, Int, jaxtyped
from typing import Union, List, Dict, Tuple, Callable, Optional, Any, Sequence, Iterable, Mapping, TypeVar, Generic, NamedTuple
import transformer_lens
from transformer_lens import *
from transformer_lens.utils import *

def to_tensor(
    tensor,
):
    return t.from_numpy(to_numpy(tensor))

def old_imshow(
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

# TODO add Callum's nice hist functions
# TODO add Callum's path patching function
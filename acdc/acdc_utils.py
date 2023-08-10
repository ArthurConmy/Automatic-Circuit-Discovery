import ast
from collections import OrderedDict
import re
import sys
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Optional, Tuple, Union, List
from huggingface_hub import hf_hub_download

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from transformer_lens.HookedTransformer import HookedTransformer

from acdc.TLACDCEdge import (
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!

class OrderedDefaultdict(defaultdict):
    def __init__(self, *args, **kwargs):
        if sys.version_info < (3, 7):
            raise Exception("You need Python >= 3.7 so dict is ordered by default. You could revert to the old unmantained implementation https://github.com/ArthurConmy/Automatic-Circuit-Discovery/commit/65301ec57c31534bd34383c243c782e3ccb7ed82")
        super().__init__(*args, **kwargs)

# -------------------------
# Some ACDC metric utils
# -------------------------

def kl_divergence(
    logits: torch.Tensor,
    base_model_logprobs: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    last_seq_element_only: bool = True,
    base_model_probs_last_seq_element_only: bool = False,
    return_one_element: bool = True,
) -> torch.Tensor:
    # Note: we want base_model_probs_last_seq_element_only to remain False by default, because when the Docstring
    # circuit uses this, it already takes the last position before passing it in.

    if last_seq_element_only:
        logits = logits[:, -1, :]

    if base_model_probs_last_seq_element_only:
        base_model_logprobs = base_model_logprobs[:, -1, :]

    logprobs = F.log_softmax(logits, dim=-1)
    kl_div = F.kl_div(logprobs, base_model_logprobs, log_target=True, reduction="none").sum(dim=-1)

    if mask_repeat_candidates is not None:
        assert kl_div.shape == mask_repeat_candidates.shape, (kl_div.shape, mask_repeat_candidates.shape)
        answer = kl_div[mask_repeat_candidates]
    elif not last_seq_element_only:
        assert kl_div.ndim == 2, kl_div.shape
        answer = kl_div.view(-1)
    else:
        answer = kl_div

    if return_one_element:
        return answer.mean()

    return answer


def negative_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    baseline: Union[float, torch.Tensor] = 0.0,
    last_seq_element_only: bool = True,
    return_one_element: bool=True,
) -> torch.Tensor:
    logprobs = F.log_softmax(logits, dim=-1)

    if last_seq_element_only:
        logprobs = logprobs[:, -1, :]

    # Subtract a baseline for each element -- which could be 0 or the NLL of the base_model_logprobs
    nll_all = (
        F.nll_loss(logprobs.view(-1, logprobs.size(-1)), labels.view(-1), reduction="none").view(logprobs.size()[:-1])
        - baseline
    )

    if mask_repeat_candidates is not None:
        assert nll_all.shape == mask_repeat_candidates.shape, (
            nll_all.shape,
            mask_repeat_candidates.shape,
        )
        answer = nll_all[mask_repeat_candidates]
    elif not last_seq_element_only:
        assert nll_all.ndim == 2, nll_all.shape
        answer = nll_all.view(-1)
    else:
        answer = nll_all

    if return_one_element:
        return answer.mean()

    return answer


class MatchNLLMetric:
    def __init__(
        self,
        labels: torch.Tensor,
        base_model_logprobs: torch.Tensor,
        mask_repeat_candidates: Optional[torch.Tensor] = None,
        last_seq_element_only: bool = True,
        return_one_element: bool = True,
    ):
        self.labels = labels
        self.mask_repeat_candidates = mask_repeat_candidates
        self.last_seq_element_only = last_seq_element_only

        logprobs = base_model_logprobs
        if last_seq_element_only:
            assert logprobs.ndim == 2
        else:
            assert logprobs.ndim == 3

        self.base_nll_unreduced = F.nll_loss(
            logprobs.view(-1, logprobs.size(-1)), labels.view(-1), reduction="none"
        ).view(logprobs.size()[:-1])
        if mask_repeat_candidates is not None:
            assert self.base_nll_unreduced.shape == mask_repeat_candidates.shape

        self.return_one_element = return_one_element

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return negative_log_probs(
            logits,
            self.labels,
            mask_repeat_candidates=self.mask_repeat_candidates,
            baseline=self.base_nll_unreduced,
            last_seq_element_only=self.last_seq_element_only,
            return_one_element=self.return_one_element,
        )

def logit_diff_metric(logits, correct_labels, wrong_labels, return_one_element: bool=True) -> torch.Tensor:
    range = torch.arange(len(logits))
    correct_logits = logits[range, -1, correct_labels]
    incorrect_logits = logits[range, -1, wrong_labels]

    # Note: negative sign so we minimize
    # TODO de-duplicate with docstring/utils.py `raw_docstring_metric`
    if return_one_element:
        return -(correct_logits.mean() - incorrect_logits.mean())
    else:
        return -(correct_logits - incorrect_logits).view(-1)

def frac_correct_metric(logits, correct_labels, wrong_labels, return_one_element: bool=True) -> torch.Tensor:
    range = torch.arange(len(logits))
    correct_logits = logits[range, -1, correct_labels]
    incorrect_logits = logits[range, -1, wrong_labels]

    # Neg so we maximize correct
    if return_one_element:
        return -(correct_logits > incorrect_logits).float().mean()
    else:
        return -(correct_logits > incorrect_logits).float().view(-1)

# -----------
# Utils of secondary importance
# -----------

def next_key(ordered_dict: OrderedDict, current_key):
    key_iterator = iter(ordered_dict)
    next((key for key in key_iterator if key == current_key), None)
    return next(key_iterator, None)

def make_nd_dict(end_type, n = 3) -> Any:
    """Make biiig default dicts : ) : )"""

    if n not in [3, 4]:
        raise NotImplementedError("Only implemented for 3/4")
        
    if n == 3:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(end_type)))

    if n == 4:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(end_type))))

def cleanup():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def shuffle_tensor(tens, seed=42):
    """Shuffle tensor along first dimension"""
    torch.random.manual_seed(seed)
    return tens[torch.randperm(tens.shape[0])]

def ct():
    return time.ctime().replace(" ", "_").replace(":", "_").replace("__", "_")

# ----------------------------------
# Random helpers for scraping
# ----------------------------------


def extract_info(string):
    """Thanks GPT-4 for writing all this..."""

    # Regex patterns
    parent_pattern = r"cur_parent=TLACDCInterpNode\((.*?), \[(.*?)\]\)"
    current_pattern = r"self.current_node=TLACDCInterpNode\((.*?), \[(.*?)\]\)"

    # Extract parent info
    parent_match = re.search(parent_pattern, string)
    parent_name = parent_match.group(1) if parent_match else None
    parent_list_str = parent_match.group(2) if parent_match else None
    parent_list = None
    if parent_list_str:
        parent_list_items = parent_list_str.split(", ")
        parent_list = [ast.literal_eval(item if item != "COL" else "None") for item in parent_list_items]

    # Extract current node info
    current_match = re.search(current_pattern, string)
    current_name = current_match.group(1) if current_match else None
    current_list_str = current_match.group(2) if current_match else None
    current_list = None
    if current_list_str:
        current_list_items = current_list_str.split(", ")
        current_list = [ast.literal_eval(item if item != "COL" else "None") for item in current_list_items]

    return parent_name.replace("hook_resid_mid", "hook_mlp_in"), parent_list, current_name.replace("hook_resid_mid", "hook_mlp_in"), current_list

# ----------------------------------
# Precision and recall etc metrics
# ----------------------------------

def get_present_nodes(graph) -> tuple[set[tuple[str, TorchIndex]], set[tuple[str, TorchIndex]]]:
    present_nodes = set()
    all_nodes = set()

    for t, e in graph.all_edges().items():
        all_nodes.add((t[0], t[1]))
        all_nodes.add((t[2], t[3]))

        if e.present and e.edge_type != EdgeType.PLACEHOLDER:
            present_nodes.add((t[0], t[1]))
            present_nodes.add((t[2], t[3]))

    return present_nodes, all_nodes

def filter_nodes(nodes: set[tuple[str, TorchIndex]]) -> set[tuple[str, TorchIndex]]:
    all_nodes = nodes.copy()

    # combine MLP things
    for node in nodes:
        if "resid_mid" in node[0] or "mlp_in" in node[0]: # new and old names
            try:
                all_nodes.add((f"blocks.{node[0].split()[1]}.hook_mlp_out", node[1])) # assume that we're not doing any neuron or positional stuff
            except:
                a = 1
            all_nodes.remove(node)
        for letter in "qkv":
            hook_name = f"hook_{letter}_input"
            if hook_name in node[0]:
                all_nodes.add((node[0].replace(hook_name, f"hook_{letter}"), node[1]))
                all_nodes.remove(node)
    return all_nodes


def get_node_stats(ground_truth, recovered) -> dict[str, int]:
    assert set(ground_truth.all_edges().keys()) == set(recovered.all_edges().keys()), "There is a mismatch between the keys we're cmparing here"

    ground_truth_nodes, all_nodes = get_present_nodes(ground_truth)

    recovered_nodes, all_rec_nodes = get_present_nodes(recovered)
    assert all_nodes == all_rec_nodes
    del all_rec_nodes

    # filter
    all_nodes = filter_nodes(all_nodes)
    ground_truth_nodes = filter_nodes(ground_truth_nodes)
    recovered_nodes = filter_nodes(recovered_nodes)

    counts = {
        "true positive": 0,
        "false positive": 0,
        "true negative": 0,
        "false negative": 0,
    }

    for node in all_nodes:
        if node in ground_truth_nodes:
            if node in recovered_nodes:
                counts["true positive"] += 1
            else:
                counts["false negative"] += 1
        else:
            if node in recovered_nodes:
                counts["false positive"] += 1
            else:
                counts["true negative"] += 1

    counts["all"] = len(all_nodes)
    counts["ground truth"] = len(ground_truth_nodes)
    counts["recovered"] = len(recovered_nodes)


    assert counts["all"] == counts["true positive"] + counts["false positive"] + counts["true negative"] + counts["false negative"]
    assert counts["ground truth"] == counts["true positive"] + counts["false negative"]
    assert counts["recovered"] == counts["true positive"] + counts["false positive"]

    # Idk if this one is any constraint
    assert counts["all"] == counts["ground truth"] + counts["recovered"] - counts["true positive"] + counts["true negative"]

    return counts

def get_edge_stats(ground_truth, recovered):    
    assert set(ground_truth.all_edges().keys()) == set(recovered.all_edges().keys()), "There is a mismatch between the keys we're comparing here"

    ground_truth_all_edges = ground_truth.all_edges()
    recovered_all_edges = recovered.all_edges()

    counts = {
        "true positive": 0,
        "false positive": 0,
        "true negative": 0,
        "false negative": 0,
    }

    for tupl, edge in ground_truth_all_edges.items():
        if edge.edge_type == EdgeType.PLACEHOLDER:
            continue
        if recovered_all_edges[tupl].present:
            if edge.present:
                counts["true positive"] += 1
            else:
                counts["false positive"] += 1
        else:
            if edge.present:
                counts["false negative"] += 1
            else:
                counts["true negative"] += 1
            

    counts["all"] = len([e for e in ground_truth_all_edges.values() if e.edge_type != EdgeType.PLACEHOLDER])
    counts["ground truth"] = len([e for e in ground_truth_all_edges.values() if e.edge_type != EdgeType.PLACEHOLDER and e.present])
    counts["recovered"] = len([e for e in recovered_all_edges.values() if e.edge_type != EdgeType.PLACEHOLDER and e.present])


    assert counts["all"] == counts["true positive"] + counts["false positive"] + counts["true negative"] + counts["false negative"]
    assert counts["ground truth"] == counts["true positive"] + counts["false negative"]
    assert counts["recovered"] == counts["true positive"] + counts["false positive"]

    # Idk if this one is any constraint
    assert counts["all"] == counts["ground truth"] + counts["recovered"] - counts["true positive"] + counts["true negative"]

    return counts
            

def false_positive_rate(ground_truth, recovered, verbose=False):
    return get_stat(ground_truth, recovered, mode="false positive", verbose=verbose)

def false_negative_rate(ground_truth, recovered, verbose=False):
    return get_stat(ground_truth, recovered, mode="false negative", verbose=verbose)

def true_positive_stat(ground_truth, recovered, verbose=False):
    return get_stat(ground_truth, recovered, mode="true positive", verbose=verbose)

# ----------------------------------
# Resetting networks; Appendix
# ----------------------------------


def reset_network(task: str, device, model: torch.nn.Module) -> None:
    filename = {
        "ioi": "ioi_reset_heads_neurons.pt",
        "tracr-reverse": "tracr_reverse_reset_heads_neurons.pt",
        "tracr-proportion": "tracr_proportion_reset_heads_neurons.pt",
        "induction": "induction_reset_heads_neurons.pt",
        "docstring": "docstring_reset_heads_neurons.pt",
        "greaterthan": "greaterthan_reset_heads_neurons.pt",
    }[task]
    random_model_file = hf_hub_download(repo_id="agaralon/acdc_reset_models", filename=filename)
    reset_state_dict = torch.load(random_model_file, map_location=device)
    model.load_state_dict(reset_state_dict, strict=False)

# ----------------------------------
# Munging utils
# ----------------------------------

def get_col_from_df(df, col_name):
    return df[col_name].values

def df_to_np(df):
    return df.values

def get_time_diff(run_name):
    """Get the difference between first log and last log of a WANBB run"""
    api = wandb.Api()    
    run = api.run(run_name)
    df = run.history()["_timestamp"]
    arr = df_to_np(df)
    n = len(arr)
    for i in range(n-1):
        assert arr[i].item() < arr[i+1].item()
    print(arr[-1].item() - arr[0].item())

def get_nonan(arr, last=True):
    """Get last non nan by default (or first if last=False)"""
    
    indices = list(range(len(arr)-1, -1, -1)) if last else list(range(len(arr)))

    for i in indices: # range(len(arr)-1, -1, -1):
        if not np.isnan(arr[i]):
            return arr[i]

    return np.nan

def get_corresponding_element(
    df,
    col1_name,
    col1_value,
    col2_name, 
):
    """Get the corresponding element of col2_name for a given element of col1_name"""
    col1 = get_col_from_df(df, col1_name)
    col2 = get_col_from_df(df, col2_name)
    for i in range(len(col1)):
        if col1[i] == col1_value and not np.isnan(col2[i]):
            return col2[i]
    assert False, "No corresponding element found"

def get_first_element(
    df,
    col,
    last=False,
):
    col1 = get_col_from_df(df, "_step")
    col2 = get_col_from_df(df, col)

    cur_step = 1e30 if not last else -1e30
    cur_ans = None

    for i in range(len(col1)):
        if not last:
            if col1[i] < cur_step and not np.isnan(col2[i]):
                cur_step = col1[i]
                cur_ans = col2[i]
        else:
            if col1[i] > cur_step and not np.isnan(col2[i]):
                cur_step = col1[i]
                cur_ans = col2[i]

    assert cur_ans is not None
    return cur_ans

def get_longest_float(s, end_cutoff=None):
    ans = None
    if end_cutoff is None:
        end_cutoff = len(s)
    else:
        assert end_cutoff < 0, "Do -1 or -2 etc mate"

    for i in range(len(s)-1, -1, -1):
        try:
            ans = float(s[i:end_cutoff])
        except:
            pass
        else:
            ans = float(s[i:end_cutoff])
    assert ans is not None
    return ans

def get_threshold_zero(s, num=3, char="_"):
    return float(s.split(char)[num])

def process_nan(tens, reverse=False):
    # turn nans into -1s
    assert isinstance(tens, np.ndarray)
    assert len(tens.shape) == 1, tens.shape
    tens[np.isnan(tens)] = -1
    tens[0] = tens.max()
    
    # turn -1s into the minimum value
    tens[np.where(tens == -1)] = 1000

    if reverse:
        for i in range(len(tens)-2, -1, -1):
            tens[i] = min(tens[i], tens[i+1])
        
        for i in range(1, len(tens)):
            if tens[i] == 1000:
                tens[i] = tens[i-1]

    else:    
        for i in range(1, len(tens)):
            tens[i] = min(tens[i], tens[i-1])

        for i in range(1, len(tens)):
            if tens[i] == 1000:
                tens[i] = tens[i-1]

    return tens

if __name__ == "__main__":
    # some quick test
    string = "Node: cur_parent=TLACDCInterpNode(blocks.3.attn.hook_result, ['COL', 'COL', 1]) (self.current_node=TLACDCInterpNode(blocks.3.hook_resid_post, ['COL']))"
    parent_name, parent_list, current_name, current_list = extract_info(string)

    print(f"Parent Name: {parent_name}\nParent List: {parent_list}\nCurrent Name: {current_name}\nCurrent List: {current_list}")


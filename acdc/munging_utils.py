import wandb
import numpy as np

from typing import List, Tuple
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.acdc_utils import TorchIndex, EdgeType

def parse_interpnode(s: str) -> TLACDCInterpNode:

    try:
        name, idx = s.split("[")
        try:
            idx = int(idx[-3:-1])
        except:
            try: 
                idx = int(idx[-2])
            except:
                idx = None
        return TLACDCInterpNode(name, TorchIndex([None, None, idx]) if idx is not None else TorchIndex([None]), EdgeType.ADDITION)

    except Exception as e: 
        print(s, e)
        raise e

    return TLACDCInterpNode(name, TorchIndex([None, None, idx]), EdgeType.ADDITION)

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

    
def heads_to_nodes_to_mask(heads: List[Tuple[int, int]], return_dict=False):
    nodes_to_mask_strings = [
        f"blocks.{layer_idx}{'.attn' if not inputting else ''}.hook_{letter}{'_input' if inputting else ''}[COL, COL, {head_idx}]"
        # for layer_idx in range(model.cfg.n_layers)
        # for head_idx in range(model.cfg.n_heads)
        for layer_idx, head_idx in heads
        for letter in ["q", "k", "v"]
        for inputting in [True, False]
    ]
    nodes_to_mask_strings.extend([
        f"blocks.{layer_idx}.attn.hook_result[COL, COL, {head_idx}]"
        for layer_idx, head_idx in heads
    ])

    if return_dict:
        return {s: parse_interpnode(s) for s in nodes_to_mask_strings}

    else:
        return [parse_interpnode(s) for s in nodes_to_mask_strings]

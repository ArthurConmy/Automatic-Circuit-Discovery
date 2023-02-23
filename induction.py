# %% [markdown]

"""
Edited version of the hierarchy tutorial that runs a number of threshold runs, and then uses the results to plot a pareto frontier.
"""

# %%

import os
import time

# os.chdir(os.path.expanduser("~/mlab2/"))

import pandas as pd
# import interp.tools.optional as op
import numpy as np
import rust_circuit as rc
import torch
from rust_circuit.causal_scrubbing.experiment import (
    Experiment,
    ExperimentCheck,
    ExperimentEvalSettings,
    ScrubbedExperiment,
)
from rust_circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    CondSampler,
    ExactSampler,
    FuncSampler,
    InterpNode,
    UncondSampler,
    chain_excluding,
    corr_root_matcher,
)
import pandas as pd
from rust_circuit.model_rewrites import To, configure_transformer
from rust_circuit.module_library import load_model_id
# from interp.tools.indexer import TORCH_INDEXER as I
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
import wandb
import datetime

import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore
from typing import List
from tqdm import tqdm
from rust_circuit.py_utils import Indexer as I
from rust_circuit.causal_scrubbing.dataset import Dataset
from rust_circuit.causal_scrubbing.hypothesis import corr_root_matcher
from rust_circuit.model_rewrites import To, configure_transformer
from rust_circuit.module_library import load_transformer_model_string
from models.model_strings import attention_only_2_string
from rust_circuit.module_library import load_model_id
from utils import (
    ACDCCorrespondence,
    ACDCExperiment,
    ACDCInterpNode,
)
from utils import make_arr
import rust_circuit as rc
import torch
from rust_circuit.model_rewrites import To, configure_transformer
from rust_circuit.module_library import load_model_id
import pickle

MAIN = __name__ == "__main__"
# %%

try:
    with open(os.path.expanduser("~/induction/data/masks/mask_repeat_candidates.pkl"), "rb") as f:
        mask_repeat_candidates=pickle.load(f)
        # t[1] has 132, 136, 176 available...
        # if induction is AB...AB, B tokens are on 133(OK...), 137, 177
        # OK so this is where we punish losses
except:
    raise Exception("Have you cloned https://github.com/aryamanarora/induction ??? It is where all the masks are kept !!!")

DEVICE = "cuda:0"
SEQ_LEN = 300
NUM_EXAMPLES = 40
MODEL_ID = "attention_only_2"
PRINT_CIRCUITS = True
ACTUALLY_RUN = True
SLOW_EXPERIMENTS = True
DEFAULT_CHECKS: ExperimentCheck = True
EVAL_DEVICE = "cuda:0"
MAX_MEMORY = 20000000000
# BATCH_SIZE = 2000
USING_WANDB = True
MONOTONE_METRIC = "maximize"
START_TIME = datetime.datetime.now().strftime("%a-%d%b_%H%M%S")
PROJECT_NAME = f"induction_arthur"

# %%

(circ_dict, tokenizer, model_info) = load_transformer_model_string(attention_only_2_string) # load_model_id(MODEL_ID)
# %%

"""
Get toks and data
"""

# longer seq len is better, but short makes stuff a bit easier...
n_files = 1
reload_dataset = False
toks_int_values: rc.Array 

if reload_dataset:
    raise NotImplementedError("Can't use these functions on new rust library...")
    # dataset_toks = torch.tensor(get_val_seqs(n_files=n_files, files_start=0, max_size=SEQ_LEN + 1)).cuda()
    # NUM_EXAMPLES, _ = dataset_toks.shape
    # toks_int_values = rc.Array(dataset_toks.float(), name="toks_int_vals")
    # print(f'new dataset "{toks_int_values.repr()}"')
else:
    P = rc.Parser()
    toks_int_values_raw = P(f"'toks_int_vals' [104091,301] Array 3f36c4ca661798003df14994").cast_array()

assert toks_int_values_raw.shape[0] >= SEQ_LEN, f"toks_int_values_raw.shape[0] = {toks_int_values_raw.shape[0]} < {SEQ_LEN} - you could try increasing `n_files`"
assert toks_int_values_raw.shape[1] >= SEQ_LEN + 1, f"toks_int_values_raw.shape[1] = {toks_int_values_raw.shape[1]} < {SEQ_LEN + 1}"

tokens_device_dtype = rc.TorchDeviceDtype(device="cuda:0", dtype="int64")

toks_int_values = make_arr(
    toks_int_values_raw.value[:NUM_EXAMPLES, :SEQ_LEN],
    name="toks_int_vals",
    device_dtype=tokens_device_dtype,
)
mask_reshaped = mask_repeat_candidates[:NUM_EXAMPLES, :SEQ_LEN] # only used for loss
denom = mask_reshaped.int().sum().item()
print("We're going to study", denom, "examples...")
assert denom == 172, (denom, "was not expected")

toks_int_labels = make_arr(
    toks_int_values_raw.value[:NUM_EXAMPLES, 1:SEQ_LEN+1],
    name="toks_int_labels",
    device_dtype=tokens_device_dtype,
)
def shuffle_tensor(tens):
    """Shuffle tensor along first dimension"""
    return tens[torch.randperm(tens.shape[0])]

toks_int_values_other = make_arr(
    shuffle_tensor(toks_int_values.value[:NUM_EXAMPLES, :SEQ_LEN + 1]),
    name="toks_int_vals_other",
    device_dtype=tokens_device_dtype,
)

circ_dict = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device="cuda:0")) for s, c in circ_dict.items()}

orig_circuit = circ_dict["t.bind_w"]
tok_embeds = circ_dict["t.w.tok_embeds"]
pos_embeds = circ_dict["t.w.pos_embeds"]

default_input = toks_int_values.rename("tokens")
default_output = toks_int_labels.rename("labels")

print("\ntokens of input and output")
print(tokenizer.batch_decode(default_input.evaluate()[0, :10]))
print(tokenizer.batch_decode(default_output.evaluate()[0, :10]))

patch_input = toks_int_values_other.rename("tokens")
# make_arr(
#     toks_int_values_other[:NUM_EXAMPLES, :SEQ_LEN],
#     "tokens",
#     device_dtype=tokens_device_dtype,
# )
patch_output = default_output # oo cares..

default_ds = Dataset({"tokens": default_input, "labels": default_output})
patch_ds = Dataset({"tokens": patch_input, "labels": patch_output})

#%%

# from transformer_lens import HookedTransformer
# et_model = HookedTransformer.from_pretrained("gpt2-medium")

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer("Hello world")['input_ids']
[15496, 995]

# %%

"""
Create metric

As mentioned in the AF post, the UNSCRUBBED output = 0.179, 
Induction heads scrubbed = 0.24
"""

def negative_log_probs(
    dataset: Dataset,
    logits: torch.Tensor,
) -> float:
    """NOTE: this average over all sequence positions, I'm unsure why..."""
    labels = dataset.arrs["labels"].evaluate()
    probs = F.softmax(logits, dim=-1)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    log_probs = probs[torch.arange(NUM_EXAMPLES).unsqueeze(-1), torch.arange(SEQ_LEN).unsqueeze(0), labels].log()
    
    assert mask_reshaped.shape == log_probs.shape, (mask_reshaped.shape, log_probs.shape)

    masked_log_probs = log_probs * mask_reshaped.int()
    result = (-1.0 * float(masked_log_probs.sum())) / denom

    print('Result', result, denom)
    return result

# %%

"""
Create model
"""

assert model_info.causal_mask, "Should not apply causal mask if the transformer doesn't expect it!"

# TODO: may be an issue later!
causal_mask = rc.Array(
    (torch.arange(SEQ_LEN)[:, None] >= torch.arange(SEQ_LEN)[None, :]).to(tok_embeds.cast_array().value),
    f"t.a.c.causal_mask",
)
assert model_info.pos_enc_type == "shortformer"

assert False "I'm pretty sure you could implement this indexing with some cursed cast to pytorch and back hack, but I'm going to try and implement the indexer soon..."
pos_embeds = pos_embeds.index(I[:SEQ_LEN], name="t.w.pos_embeds_idxed")

tokens_arr = rc.cast_circuit(
    rc.Array(torch.zeros(SEQ_LEN).to(torch.long), name="tokens"),
    device_dtype=tokens_device_dtype.op(),
)
idxed_embeds = rc.GeneralFunction.gen_index(tok_embeds, tokens_arr, index_dim=0, name="idxed_embeds")

# CHECK
model = rc.module_new_bind(
    orig_circuit,
    ("t.input", idxed_embeds),
    ("a.mask", causal_mask),
    ("a.pos_input", pos_embeds),
    name="t.call",
)

# model = model_info.bind_to_input(
#     orig_circuit,
#     idxed_embeds,
#     pos_embeds,
#     causal_mask,
# )

# CHECK
model = model.update(
    "t.bind_w",
    lambda c: configure_transformer(
        c,
        To.ATTN_HEAD_MLP_NORM,
        split_by_head_config="full",
        use_pull_up_head_split=True,
        use_flatten_res=True,
        flatten_components=True,
    ),
)
model = model.cast_module().substitute()
model.print_html()
model = rc.Index(model, I[0]).rename("model")
model = rc.conform_all_modules(model)
model = model.update("t.call", lambda c: c.rename("logits"))
model = model.update("t.call", lambda c: c.rename("logits_with_bias"))
model = model.update(rc.Regex("[am]\\d(.h\\d)?$"), lambda c: c.rename(c.name + ".inner"))
model = model.update("t.inp_tok_pos", lambda c: c.rename("embeds"))
model = model.update("t.a.mask", lambda c: c.rename("padding_mask"))
for l in range(model_info.params.num_layers):
    for h in range(8):
        model = model.update(f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}"))
    next = "final" if l == model_info.params.num_layers - 1 else f"a{l + 1}"
    model = model.update(f"b{l}", lambda c: c.rename(f"{next}.input"))


def create_path_matcher(start_node: rc.MatcherIn, path: list[str], max_distance=10) -> rc.IterativeMatcher:
    """
    Creates a matcher that matches a path of nodes, given in a list of names, where the maximum distance between each node on the path is max_distance
    """

    initial_matcher = rc.IterativeMatcher(start_node)
    max_dis_path_matcher = lambda name: rc.restrict(rc.Matcher(name), end_depth=max_distance)
    chain_matcher = initial_matcher.chain(max_dis_path_matcher(path[0]))
    for i in range(1, len(path)):
        chain_matcher = chain_matcher.chain(max_dis_path_matcher(path[i]))
    return chain_matcher


q_path = [
    "a.comb_v",
    "a.attn_probs",
    "a.attn_scores",
    "a.attn_scores_raw",
    "a.q",
]
k_path = [
    "a.comb_v",
    "a.attn_probs",
    "a.attn_scores",
    "a.attn_scores_raw",
    "a.k",
]
v_path = ["a.comb_v", "a.v"]
qkv_paths = {"q": q_path, "k": k_path, "v": v_path}
attention_head_name = "a{layer}.h{head}"
qkv_node_name = "a{layer}.h{head}.{qkv}"
embed_name = "idxed_embeds"
root_name = "final.input"
no_layers = 2
no_heads = 8
new_circuit = model
for l in range(no_layers):
    for h in range(no_heads):
        for qkv in ["q", "k", "v"]:
            qkv_matcher = create_path_matcher(f"a{l}.h{h}", qkv_paths[qkv])
            new_circuit = new_circuit.update(qkv_matcher, lambda c: c.rename(f"a{l}.h{h}.{qkv}"))


printer = rc.PrintHtmlOptions(
    shape_only_when_necessary=False,
    traversal=rc.restrict(
        rc.IterativeMatcher("embeds", "padding_mask", "final.norm", rc.Regex("^[am]\\d(.h\\d)?$")),
        term_if_matches=True,
    ),
)
new_circuit = rc.substitute_all_modules(new_circuit)
new_circuit.get_unique("final.input").print_html()
new_circuit = new_circuit.get_unique("logits")

"""
Create correspondence
"""

# %%

all_names = (
    [embed_name, root_name]
    + [attention_head_name.format(layer=l, head=h) for l in range(no_layers) for h in range(no_heads)]
    + [
        qkv_node_name.format(layer=l, head=h, qkv=qkv)
        for l in range(no_layers)
        for h in range(no_heads)
        for qkv in ["q", "k", "v"]
    ]
)
all_names = set(all_names)
template_corr = ACDCCorrespondence(all_names=all_names)
root = ACDCInterpNode(root_name, is_root=True)
template_corr.add(root)

all_residual_stream_parents: List[ACDCInterpNode] = []
all_residual_stream_parents.append(root)
for layer in tqdm(range(no_layers - 1, -1, -1)):
    qkv_nodes_list = []
    for head in range(no_heads):
        head_node = ACDCInterpNode(attention_head_name.format(layer=layer, head=head))
        for node in all_residual_stream_parents:
            node.add_child(head_node)
            head_node.add_parent(node)
        template_corr.add(head_node)
        for qkv in ["q", "k", "v"]:
            qkv_node = ACDCInterpNode(f"a{layer}.h{head}.{qkv}")
            head_node.add_child(qkv_node)
            qkv_node.add_parent(head_node)
            template_corr.add(qkv_node)
            qkv_nodes_list.append(qkv_node)
    all_residual_stream_parents += qkv_nodes_list

# add embedding node
embed_node = ACDCInterpNode(embed_name)
for node in tqdm(all_residual_stream_parents):
    node.add_child(embed_node)
    embed_node.add_parent(node)
template_corr.add(embed_node)

template_corr.show()
template_corr.topologically_sort_corr()

#%%

def get_baselines():
# if True:  
    """
    WARNING: the baselines divide by ALL sequence positions... IDK why
    
    As in https://www.alignmentforum.org/posts/j6s9H9SHrEhEfuJnq/causal-scrubbing-results-on-induction-heads#Refined_hypothesis_4
    calculate the baseline scrubbed through the induction heads
    """

    exp = ACDCExperiment(
        circuit=new_circuit,
        ds=default_ds,
        ref_ds=patch_ds,
        template_corr=template_corr, # type: ignore
        metric=negative_log_probs, # lambda x, y: 0.0,
        random_seed=42,
        num_examples=NUM_EXAMPLES,
        check=None,
        threshold=0.05, # this run now picks up ... some things!
        verbose=False,
        monotone_metric="minimize",
        using_wandb=False,
    )

    unscrubbed = exp.cur_metric
    exp.expand_current_node()
    exp._nodes.show()

    for hindex in [5, 6]:
        exp.remove_connection(
            parent_node=exp._nodes.i_names["final.input"],
            child_node=exp._nodes.i_names[f"a1.h{hindex}"],
        )
    exp.update_cur_metric()
    return unscrubbed, exp.cur_metric

print(get_baselines(), "baselines")
# print("These should be 0.16 and 0.21 respectively, according to AF")

#%% [markdown]

# # Do some initial experiment setups

for threshold in [0.0075, 0.025, 0.2, 0.4, 0.75, 0.9]: # quite high, so that the experiment ends quickly
    start_time = time.time()

    def ct():
        """CTIME but with underscores"""
        return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = "arthur_fixed_edges_" + str(threshold)
    with open(__file__, "r") as f:
        file_content = f.read()
    wandb.init(project=PROJECT_NAME, name=f'acdc-run-{run_name}', notes=file_content)
    wandb.log({"start_time": start_time})
    
    exp = ACDCExperiment(
        circuit=new_circuit,
        ds=default_ds,
        ref_ds=patch_ds,
        template_corr=template_corr,
        metric=negative_log_probs,
        threshold=threshold,
        dynamic_threshold_metric="greedy_path",
        dynamic_threshold_method="off",
        # node_long_threshold=4.0*threshold,
        child_order_priority="late_layers",
        num_examples=len(default_ds),
        verbose=True,
        remove_redundant=True,
        parallel_hypotheses=1,
        expand_by_importance=False,
        using_wandb=True,
        wandb_project_name=PROJECT_NAME,
        monotone_metric='minimize',
        expensive_update_cur=False, # two new settings, both for perf improvements to IOI...
        connections_recursive=False,
    )
    while True:
            exp.step()
            if exp.current_node is None:
                break
            # assert exp.current_node is None
            # break
    wandb.log({"tot_time": time.time()-start_time})
    wandb.finish()

#%%


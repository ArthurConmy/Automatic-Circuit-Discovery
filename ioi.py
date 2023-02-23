#%%

import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")

import os
import warnings
from copy import deepcopy
from typing import List
from models.model_strings import gelu_12_tied_string
import click
import IPython
import rust_circuit as rc
import torch
from ioi_dataset import IOIDataset  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm

from rust_circuit.causal_scrubbing.dataset import Dataset
from rust_circuit.causal_scrubbing.hypothesis import corr_root_matcher
from rust_circuit.model_rewrites import To, configure_transformer
from rust_circuit.module_library import load_transformer_model_string
from utils import (
    ACDCTemplateCorrespondence,
    ACDCCorrespondence,
    ACDCExperiment,
    ACDCInterpNode,
    make_arr,
)
from rust_circuit.causal_scrubbing.hypothesis import Correspondence
from rust_circuit.causal_scrubbing.experiment import Experiment
import wandb

with open(__file__, "r") as f:
    file_content = f.read()
    
USING_WANDB = True
MONOTONE_METRIC = "maximize"

#%% [markdown]

# In this notebook we'll setup and run a basic version of Automatic Circuit Discovery Code (ACDC). The idea is to
# <ul>
#     <li>Define a DAG of all the nodes in a `rc.Circuit` object that could explain behavior.</li>
#     <li>Carry out a search over sub-DAGs to find a small DAG that explains most behavior.</li>
# </ul>
#
# In the next cell, there's a more detailed explanation of what ACDC is. We'll use causal scrubbing code, but generally CS objects are used in quite a cursed way, and I'm interested in improvements.

#%% [markdown]
# <h2>Formal definition of ACDC</h2>
# Consider the notation in <a href="https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing#2_Setup">the casual scrubbing post</a>. <b>In ACDC, we only ever consider interpretations I that are subgraphs of the computational graph G</b>. ACDC is a greedy algorithm for heuristically finding a subgraph that describes model behavior well. There are two steps to the algorithm:
# <ol>
#     <li>Expanding nodes: algebraically rewriting a leaf node to add its inputs. `as`</li>
#     <li>Removing an edge in I.</li>
# </ol>
#
# So it is similar to a Breadth-First Search.

#%% [markdown]
# Here's pseudocode for ACDC:
# However, note that for a fast implementation, we batch the child_node loop, see the next cell

print(
    """
# sort G so the a node is always processed before its inputs (so the OUTPUT node is G[0])
G.reverse_topological_sort()
# initialize I to be the graph with only the output node
I = Correspondence(G[0], G[0].matcher)
# this means we don't scrub anything, initially
metric = compute_metric(G, I)
# set some ACDC threshold
threshold = 0.1
# (larger = more nodes, but slower runtime)
node_queue = [G.root()]
while node_queue is not empty:
    node = node_queue.pop()
    I.expand(node) # add all of node's inputs to I
    for child_node in node.children:
        I.remove(parent_node=node, child_node=child_node)
        new_metric = compute_metric(G, I)
        if abs(new_metric - metric) < threshold:
            # child_node->node wasn't important
            metric = new_metric
        else:
            # child_node->node was important
            I.add(parent_node=node, child_node=child_node)
            if child_node not in node_queue:
                node_queue.append(child_node)
print(I) # print the final ACDC graph!
"""
)

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

tokens_device_dtype = rc.TorchDeviceDtype("cuda:0", "int64")
default_data = make_arr(
    ioi_dataset.toks.long()[:N, : seq_len - 1],
    "tokens",
    device_dtype=tokens_device_dtype,
)
patch_data = make_arr(
    abc_dataset.toks.long()[:N, : seq_len - 1],
    "tokens",
    device_dtype=tokens_device_dtype,
)
default_ds = Dataset({"tokens": default_data})
patch_ds = Dataset({"tokens": patch_data})

print(
    "Example of prompts:\n\n",
    ioi_dataset.tokenized_prompts[0],
    "\n",
    ioi_dataset.tokenized_prompts[-1],
)

print(
    "\n... we also have some `ABC` data, that we use to scrub unimportant nodes:\n\n",
    abc_dataset.tokenized_prompts[0],
    "\n",
    abc_dataset.tokenized_prompts[-1],
)
#%% [markdown]
# Set up the model
# This is slow the first time you run it, but then it's cached so should be faster


def get_model(
    seq_len,
):
    """See interp/demos/model_loading_with_modules!!!
    This is just copied, but we need to do it here because we need to set the seq_len"""
    
    # MODEL_ID = "gelu_12_tied"  # aka gpt2 small

    circ_dict, tokenizer, model_info = load_transformer_model_string(gelu_12_tied_string)
    keys = list(circ_dict.keys())
    for key in tqdm(keys):
        circ_dict[key] = rc.cast_circuit(circ_dict[key], device_dtype=rc.TorchDeviceDtype("cuda:0", "float32").op())
    unbound_circuit = circ_dict["t.bind_w"]
    assert not unbound_circuit.is_explicitly_computable
    tokens_device_dtype = rc.TorchDeviceDtype("cuda:0", "int64")
    tokens_arr = rc.cast_circuit(
        rc.Array(torch.zeros(seq_len).to(torch.long), name="tokens"),
        device_dtype=tokens_device_dtype.op(),
    )
    token_embeds = rc.GeneralFunction.gen_index(circ_dict["t.w.tok_embeds"], tokens_arr, 0, name="tok_embeds")
    bound_circuit = model_info.bind_to_input(unbound_circuit, token_embeds, circ_dict["t.w.pos_embeds"])
    assert bound_circuit.is_explicitly_computable
    transformed_circuit = bound_circuit.update(
        "t.bind_w",
        lambda c: configure_transformer(
            c,
            To.ATTN_HEAD_MLP_NORM,
            split_by_head_config="full",
            use_pull_up_head_split=True,
            use_flatten_res=True,
        ),
    )
    transformed_circuit = rc.conform_all_modules(transformed_circuit)
    subbed_circuit = transformed_circuit.substitute()
    subbed_circuit = subbed_circuit.rename("logits")
    renamed_circuit = subbed_circuit.update(rc.Regex(r"[am]\d(.h\d)?$"), lambda c: c.rename(c.name + ".inner"))
    renamed_circuit = renamed_circuit.update("t.inp_tok_pos", lambda c: c.rename("embeds"))
    for l in range(model_info.params.num_layers):
        next = "final" if l == model_info.params.num_layers - 1 else f"a{l+1}"
        renamed_circuit = renamed_circuit.update(f"b{l}", lambda c: c.rename(f"{next}.input"))
        renamed_circuit = renamed_circuit.update(f"b{l}.m", lambda c: c.rename(f"m{l}"))
        renamed_circuit = renamed_circuit.update(f"b{l}.m.p_bias", lambda c: c.rename(f"m{l}.p_bias"))
        renamed_circuit = renamed_circuit.update(f"b{l}.a", lambda c: c.rename(f"a{l}"))
        renamed_circuit = renamed_circuit.update(f"b{l}.a.p_bias", lambda c: c.rename(f"a{l}.p_bias"))
        for h in range(model_info.params.num_layers):
            renamed_circuit = renamed_circuit.update(f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}"))
    return renamed_circuit


warnings.warn("For some reason the last ~2 steps of this TQDM take ~30 seconds...")
model = get_model(seq_len - 1)
model = rc.substitute_all_modules(model)  # essential! 20x speedup
model.print_html()

#%% [markdown]
# Now, let's define a DAG! This DAG represents the **maximal hypothesis graph**: any node found in this DAG can later be added to our hypothesis graph, while nodes we do not add to this DAG will never be added to our hypothesis graph. For example, if we never specify MLPs in the DAG, then ACDC will not directly check for the influence of any MLP, and will never add an MLP to the hypothesis graph.
# For ACDC, each node in the DAG is named after the corresponding node in the rc.Circuit. **Importantly, ACDC requires that the nodes are named uniquely**. This means 'qkv nodes' like 'a.q' found on layer 3 and at head 2 will need to be renamed to a string which can uniquely identify it from other qkv nodes nodes, like 'a3.h2.q'.
#
# We first specify the `all_names` object which stores the names of every node we want to add to our maximal hypothesis graph. Then, we define our correspondence in `template_corr`, where the edges are implemented as .parents and .children of the ACDCInterpNode objects. Be careful to avoid specifying cycles here.
#
# NOTE: for a more useful decomposition, including into the Q and K and V paths into an attention head, see `remix_d5_acdc_hierarchy_tutorial.py`

attention_head_name = "a{layer}.h{head}"
mlp_name = "m{layer}"
embed_name = "tok_embeds"
root_name = "final.inp"
no_layers = 12
no_heads = 12
all_names = (
    [embed_name, root_name]
    + [mlp_name.format(layer=layer) for layer in range(no_layers)]
    + [attention_head_name.format(layer=layer, head=head) for layer in range(no_layers) for head in range(no_heads)]
)
all_names = set(all_names)

template_corr = ACDCTemplateCorrespondence(all_names=all_names)
root = ACDCInterpNode(root_name, is_root=True)
template_corr.add_with_auto_matcher(root)

#%% [markdown]
# This cell makes the DAG (just like the previous cell, you'll want to adapt it to your use case)
# TODO matcher creation gets slow with big graphs. Can it be sped up?

all_nodes: List[ACDCInterpNode] = []
all_nodes.append(root)

print("Constructing big Correspondence...")
for layer in tqdm(range(no_layers - 1, -1, -1)):
    # add MLP
    mlp_node = ACDCInterpNode(mlp_name.format(layer=layer))
    for node in all_nodes:
        node.add_child(mlp_node)
        mlp_node.add_parent(node)
    template_corr.add_with_auto_matcher(mlp_node)

    all_nodes.append(mlp_node)

    # add heads
    head_nodes = []
    for head in range(no_heads):
        head_node = ACDCInterpNode(attention_head_name.format(layer=layer, head=head))
        head_nodes.append(head_node)
        for node in all_nodes:
            node.add_child(head_node)
            head_node.add_parent(node)
    for node in head_nodes:
        template_corr.add_with_auto_matcher(node)

    for i, head_node in enumerate(head_nodes):
        all_nodes.append(head_node)
print("...done")

embed_node = ACDCInterpNode(embed_name)

for node in tqdm(all_nodes):
    node.add_child(embed_node)
    embed_node.add_parent(node)
template_corr.add_with_auto_matcher(embed_node)
# you can also template_corr.print(), though we won't as this takes several minutes.

#%% [markdown]
# For applications of ACDC to other models and cases etc., you'll want the following checks to pass:
template_corr.topologically_sort_corr()  # fails if there are cycles, or the DAG is disconnected
#%% [markdown]
# Skip this because it is too slow. Would be useful for smaller models.

if False:
    for node, matcher in tqdm(template_corr.corr.items()):  # can be skipped / stopped if timing out
        assert (
            len(matcher.get(model)) > 0
        ), f"Matcher {matcher} for node {node} does not match anything in the model"  # your matchers make sense

# TODO maybe also add a .match_all_paths() debugging explanation, too

#%% [markdown]
# When our circuit is evaluated, it just returns logits. So let's make a function that calculates loss.
# This only works for the IOIDataset that we used. You'll need to adapt this to your use case-you'll need to add more arrays to the dataset if you want to calculate loss differently on each datapoint.


def logit_diff_from_logits_dataset(
    dataset: Dataset,
    logits: torch.Tensor,
    mean=True,
):
    toks = dataset.arrs["tokens"].value
    assert toks.shape == (N, 15)
    # assert len(logits.shape) == 3, logits.shape

    io_labels = toks[:, 2]
    s_labels = toks[:, 4]

    io_logits = logits[torch.arange(N), -1, io_labels]
    s_logits = logits[torch.arange(N), -1, s_labels]

    logit_diff = io_logits - s_logits
    if mean:
        return logit_diff.mean().item()
    else:
        return logit_diff

# %%

# ACDC no longer requires the proper ACDCCorrespondence to be passed in, but it is left here for backwards compatibility
acdc_corr = ACDCCorrespondence(all_names)
new_root = ACDCInterpNode(root_name, is_root=True)
acdc_corr.add(new_root, corr_root_matcher)

# @click.command()
# @click.option("--t", help="Threshold(s) for ACDC", multiple=True, type=float)
# @click.option("--dtm", default="off", help="Dynamic threshold method")
# @click.option("--dts", default=1.0, help="Dynamic threshold scaling")
# @click.option("--use-extra-args", default=False, help="Use extra args")
# def main(t, dtm, dts, use_extra_args):

t = [0.2]
dtm = "off"
dts = 1.0
use_extra_args = False

if True:
    thresholds = list(t)
    dynamic_threshold_method = dtm
    dynamic_threshold_scaling = dts
    for threshold in thresholds:
            if USING_WANDB:
                wandb.init(project="acdc", name="acdc_ioi_test_" + str(thresholds)+ "_" + str(dynamic_threshold_method) + ("_real" if dtm=="geometric" else "") +"_use_extra_"+str(use_extra_args)+"_threshold_"+str(threshold)+"_scaling"+str(dts), notes=file_content)

            extra_args = {                              
                "parallel_hypotheses_long_threshold": 4*threshold, # TODO maybe this needs to be much much smaller ???
                "parallel_hypotheses_long_max_iters": 10,
                "node_long_threshold": 4*threshold,
                "node_long_max_iters": 10,
            }
            if not use_extra_args:
                extra_args = {}

            exp = ACDCExperiment(
                circuit=model,
                ds=default_ds,
                ref_ds=patch_ds,
                corr=acdc_corr,
                template_corr=template_corr,
                metric=logit_diff_from_logits_dataset,  # self._dataset
                random_seed=1234,
                num_examples=len(patch_ds), 
                check="fast",
                threshold=threshold,
                dynamic_threshold_method=dynamic_threshold_method,
                dynamic_threshold_scaling=dynamic_threshold_scaling,
                min_num_children=0,
                remove_redundant=True,
                verbose=True,
                parallel_hypotheses=40,
                expand_by_importance=True,
                using_wandb=USING_WANDB,
                monotone_metric="maximize",
                expensive_update_cur=False, # two new settings, both for perf improvements to IOI...
                connections_recursive=False,
                **extra_args,
            )
            exp._nodes.show()

            idx = 0
            for idx in range(100000):
                exp.step()
                exp._nodes.show(fname="acdc_plot_" + str(idx) + ".png")
                if exp.current_node is None:
                    print("Done")
                    break
        
            wandb.finish()

#%%
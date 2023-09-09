# In[1]:

"""
Arthur's .py version of acdcpp_ioi.ipynb for iteration 
"""

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import os
import sys
import re

import acdc
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import TorchIndex, EdgeType
import numpy as np
import torch as t
from torch import Tensor
import einops
import itertools

from transformer_lens import HookedTransformer, ActivationCache

import tqdm.notebook as tqdm
import plotly
from rich import print as rprint
from rich.table import Table

from jaxtyping import Float, Bool
from typing import Callable, Tuple, Union, Dict, Optional

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'Device: {device}')


# # Model Setup

# In[2]:


model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)


# # Dataset Setup

# In[4]:


from acdc.ioi_dataset import IOIDataset, format_prompt, make_table
N = 25
clean_dataset = IOIDataset(
    prompt_type='mixed',
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=device
)
corr_dataset = clean_dataset.gen_flipped_prompts('ABC->XYZ, BAB->XYZ')

make_table(
  colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
  cols = [
    map(format_prompt, clean_dataset.sentences),
    model.to_string(clean_dataset.s_tokenIDs).split(),
    model.to_string(clean_dataset.io_tokenIDs).split(),
    map(format_prompt, clean_dataset.sentences),
  ],
  title = "Sentences from IOI vs ABC distribution",
)


# # Metric Setup

# In[5]:


def ave_logit_diff(
    logits: Float[Tensor, 'batch seq d_vocab'],
    ioi_dataset: IOIDataset,
    per_prompt: bool = False
):
    '''
        Return average logit difference between correct and incorrect answers
    '''
    # Get logits for indirect objects
    io_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.io_tokenIDs]
    s_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.s_tokenIDs]
    # Get logits for subject
    logit_diff = io_logits - s_logits
    return logit_diff if per_prompt else logit_diff.mean()

with t.no_grad():
    clean_logits = model(clean_dataset.toks)
    corrupt_logits = model(corr_dataset.toks)
    clean_logit_diff = ave_logit_diff(clean_logits, clean_dataset).item()
    corrupt_logit_diff = ave_logit_diff(corrupt_logits, corr_dataset).item()

def ioi_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    corrupted_logit_diff: float = corrupt_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
    ioi_dataset: IOIDataset = clean_dataset
 ):
    patched_logit_diff = ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def negative_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -ioi_metric(logits)
    
# Get clean and corrupt logit differences
with t.no_grad():
    clean_metric = ioi_metric(clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset)
    corrupt_metric = ioi_metric(corrupt_logits, corrupt_logit_diff, clean_logit_diff, corr_dataset)

print(f'Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}')
print(f'Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}')


# # Helper Methods

# In[6]:


def remove_redundant_node(exp, node, safe=True, allow_fails=True):
        if safe:
            for parent_name in exp.corr.edges[node.name][node.index]:
                for parent_index in exp.corr.edges[node.name][node.index][parent_name]:
                    if exp.corr.edges[node.name][node.index][parent_name][parent_index].present:
                        raise Exception(f"You should not be removing a node that is still used by another node {node} {(parent_name, parent_index)}")

        bfs = [node]
        bfs_idx = 0

        while bfs_idx < len(bfs):
            cur_node = bfs[bfs_idx]
            bfs_idx += 1

            children = exp.corr.graph[cur_node.name][cur_node.index].children

            for child_node in children:
                if not cur_node.index in exp.corr.edges[child_node.name][child_node.index][cur_node.name]:
                    #print(f'\t CANT remove edge {cur_node.name}, {cur_node.index} <-> {child_node.name}, {child_node.index}')
                    continue
                    
                try:
                    #print(f'\t Removing edge {cur_node.name}, {cur_node.index} <-> {child_node.name}, {child_node.index}')
                    exp.corr.remove_edge(
                        child_node.name, child_node.index, cur_node.name, cur_node.index
                    )
                except KeyError as e:
                    print("Got an error", e)
                    if allow_fails:
                        continue
                    else:
                        raise e

                remove_this = True
                for parent_of_child_name in exp.corr.edges[child_node.name][child_node.index]:
                    for parent_of_child_index in exp.corr.edges[child_node.name][child_node.index][parent_of_child_name]:
                        if exp.corr.edges[child_node.name][child_node.index][parent_of_child_name][parent_of_child_index].present:
                            remove_this = False
                            break
                    if not remove_this:
                        break

                if remove_this and child_node not in bfs:
                    bfs.append(child_node)

def remove_node(exp, node):
    '''
        Method that removes node from model. Assumes children point towards
        the end of the residual stream and parents point towards the beginning.

        exp: A TLACDCExperiment object with a reverse top sorted graph
        node: A TLACDCInterpNode describing the node to remove
        root: Initally the first node in the graph
    '''
    #Removing all edges pointing to the node
    remove_edges = []
    for p_name in exp.corr.edges[node.name][node.index]:
        for p_idx in exp.corr.edges[node.name][node.index][p_name]:
            edge = exp.corr.edges[node.name][node.index][p_name][p_idx]
            remove_edges.append((node.name, node.index, p_name, p_idx))
            edge.present = False
    for n_name, n_idx, p_name, p_idx in remove_edges:
        #print(f'\t Removing edge {p_name}, {p_idx} <-> {n_name}, {n_idx}')
        exp.corr.remove_edge(
            n_name, n_idx, p_name, p_idx
        )
    # Removing all outgoing edges from the node using BFS
    remove_redundant_node(exp, node, safe=False)

def find_attn_node(exp, layer, head):
    return exp.corr.graph[f'blocks.{layer}.attn.hook_result'][TorchIndex([None, None, head])]

def find_attn_node_qkv(exp, layer, head):
    nodes = []
    for qkv in ['q', 'k', 'v']:
        nodes.append(exp.corr.graph[f'blocks.{layer}.attn.hook_{qkv}'][TorchIndex([None, None, head])])
        nodes.append(exp.corr.graph[f'blocks.{layer}.hook_{qkv}_input'][TorchIndex([None, None, head])])
    return nodes
    
def split_layers_and_heads(act: Tensor, model: HookedTransformer) -> Tensor:
    return einops.rearrange(act, '(layer head) batch seq d_model -> layer head batch seq d_model',
                            layer=model.cfg.n_layers,
                            head=model.cfg.n_heads)

hook_filter = lambda name: name.endswith("ln1.hook_normalized") or name.endswith("attn.hook_result")
def get_3_caches(model, clean_input, corrupted_input, metric):
    # cache the activations and gradients of the clean inputs
    model.reset_hooks()
    clean_cache = {}

    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach()

    model.add_hook(hook_filter, forward_cache_hook, "fwd")

    clean_grad_cache = {}

    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach()

    model.add_hook(hook_filter, backward_cache_hook, "bwd")

    value = metric(model(clean_input))
    value.backward()

    # cache the activations of the corrupted inputs
    model.reset_hooks()
    corrupted_cache = {}

    def forward_cache_hook(act, hook):
        corrupted_cache[hook.name] = act.detach()

    model.add_hook(hook_filter, forward_cache_hook, "fwd")
    model(corrupted_input)
    model.reset_hooks()

    clean_cache = ActivationCache(clean_cache, model)
    corrupted_cache = ActivationCache(corrupted_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, corrupted_cache, clean_grad_cache

def acdc_nodes(model: HookedTransformer,
              clean_input: Tensor,
              corrupted_input: Tensor,
              metric: Callable[[Tensor], Tensor],
              threshold: float,
              exp: TLACDCExperiment,
              attr_absolute_val: bool = False) -> Tuple[
                  HookedTransformer, Bool[Tensor, 'n_layer n_heads']]:
    '''
    Runs attribution-patching-based ACDC on the model, using the given metric and data.
    Returns the pruned model, and which heads were pruned.

    Arguments:
        model: the model to prune
        clean_input: the input to the model that contains should elicit the behavior we're looking for
        corrupted_input: the input to the model that should elicit random behavior
        metric: the metric to use to compare the model's performance on the clean and corrupted inputs
        threshold: the threshold below which to prune
        create_model: a function that returns a new model of the same type as the input model
        attr_absolute_val: whether to take the absolute value of the attribution before thresholding
    '''
    # get the 2 fwd and 1 bwd caches; cache "normalized" and "result" of attn layers
    clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(model, clean_input, corrupted_input, metric)

    # compute first-order Taylor approximation for each node to get the attribution
    clean_head_act = clean_cache.stack_head_results()
    corr_head_act = corrupted_cache.stack_head_results()
    clean_grad_act = clean_grad_cache.stack_head_results()

    # compute attributions of each node
    node_attr = (clean_head_act - corr_head_act) * clean_grad_act
    # separate layers and heads, sum over d_model (to complete the dot product), batch, and seq
    node_attr = split_layers_and_heads(node_attr, model).sum((2, 3, 4))

    if attr_absolute_val:
        node_attr = node_attr.abs()
    del clean_cache
    del clean_head_act
    del corrupted_cache
    del corr_head_act
    del clean_grad_cache
    del clean_grad_act
    t.cuda.empty_cache()
    # prune all nodes whose attribution is below the threshold
    should_prune = node_attr < threshold
    pruned_nodes_attr = {}
    for layer, head in itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)):
        if should_prune[layer, head]:
            # REMOVING NODE
            print(f'PRUNING L{layer}H{head} with attribution {node_attr[layer, head]}')
            # Find the corresponding node in computation graph
            node = find_attn_node(exp, layer, head)
            print(f'\tFound node {node.name}')
            # Prune node
            remove_node(exp, node)
            print(f'\tRemoved node {node.name}')
            pruned_nodes_attr[(layer, head)] = node_attr[layer, head]
            
            # REMOVING QKV
            qkv_nodes = find_attn_node_qkv(exp, layer, head)
            for node in qkv_nodes:
                remove_node(exp, node)
    return pruned_nodes_attr


# # Show resulting graph

# In[7]:


from acdc.TLACDCInterpNode import TLACDCInterpNode
import pygraphviz as pgv
from pathlib import Path

def get_node_name(node: TLACDCInterpNode, show_full_index=True):
    """Node name for use in pretty graphs"""

    if not show_full_index:
        name = ""
        qkv_substrings = [f"hook_{letter}" for letter in ["q", "k", "v"]]
        qkv_input_substrings = [f"hook_{letter}_input" for letter in ["q", "k", "v"]]

        # Handle embedz
        if "resid_pre" in node.name:
            assert "0" in node.name and not any([str(i) in node.name for i in range(1, 10)])
            name += "embed"
            if len(node.index.hashable_tuple) > 2:
                name += f"_[{node.index.hashable_tuple[2]}]"
            return name

        elif "embed" in node.name:
            name = "pos_embeds" if "pos" in node.name else "token_embeds"

        # Handle q_input and hook_q etc
        elif any([node.name.endswith(qkv_input_substring) for qkv_input_substring in qkv_input_substrings]):
            relevant_letter = None
            for letter, qkv_substring in zip(["q", "k", "v"], qkv_substrings):
                if qkv_substring in node.name:
                    assert relevant_letter is None
                    relevant_letter = letter
            name += "a" + node.name.split(".")[1] + "." + str(node.index.hashable_tuple[2]) + "_" + relevant_letter

        # Handle attention hook_result
        elif "hook_result" in node.name or any([qkv_substring in node.name for qkv_substring in qkv_substrings]):
            name = "a" + node.name.split(".")[1] + "." + str(node.index.hashable_tuple[2])

        # Handle MLPs
        elif node.name.endswith("resid_mid"):
            raise ValueError("We removed resid_mid annotations. Call these mlp_in now.")
        elif node.name.endswith("mlp_out") or node.name.endswith("mlp_in"):
            name = "m" + node.name.split(".")[1]

        # Handle resid_post
        elif "resid_post" in node.name:
            name += "resid_post"

        else:
            raise ValueError(f"Unrecognized node name {node.name}")

    else:
        
        name = node.name + str(node.index.graphviz_index(use_actual_colon=True))

    return "<" + name + ">"

def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """
    def rgb2hex(rgb):
        """
        https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
        """
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    return rgb2hex((np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))

def build_colorscheme(correspondence, colorscheme: str = "Pastel2", show_full_index=True) -> Dict[str, str]:
    colors = {}
    for node in correspondence.nodes():
        colors[get_node_name(node, show_full_index=show_full_index)] = generate_random_color(colorscheme)
    return colors

def show(
    correspondence: TLACDCInterpNode,
    fname=None,
    colorscheme: Union[Dict, str] = "Pastel2",
    minimum_penwidth: float = 0.3,
    show_full_index: bool = False,
    remove_self_loops: bool = True,
    remove_qkv: bool = True,
    layout: str="dot",
    edge_type_colouring: bool = False,
    show_placeholders: bool = False,
    seed: Optional[int] = None
):
    g = pgv.AGraph(directed=True, bgcolor="transparent", overlap="false", splines="true", layout=layout)

    if seed is not None:
        np.random.seed(seed)
    
    groups = {}
    if isinstance(colorscheme, str):
        colors = build_colorscheme(correspondence, colorscheme, show_full_index=show_full_index)
    else:
        colors = colorscheme
        for name, color in colors.items():
            if color not in groups:
                groups[color] = [name]
            else:
                groups[color].append(name)

    node_pos = {}
    if fname is not None:
        base_fname = ".".join(str(fname).split(".")[:-1])

        base_path = Path(base_fname)
        fpath = base_path / "layout.gv"
        if fpath.exists():
            g_pos = pgv.AGraph()
            g_pos.read(fpath)
            for node in g_pos.nodes():
                node_pos[node.name] = node.attr["pos"]
    
    for child_hook_name in correspondence.edges:
        for child_index in correspondence.edges[child_hook_name]:
            for parent_hook_name in correspondence.edges[child_hook_name][child_index]:
                for parent_index in correspondence.edges[child_hook_name][child_index][parent_hook_name]:
                    edge = correspondence.edges[child_hook_name][child_index][parent_hook_name][parent_index]

                    parent = correspondence.graph[parent_hook_name][parent_index]
                    child = correspondence.graph[child_hook_name][child_index]

                    parent_name = get_node_name(parent, show_full_index=show_full_index)
                    child_name = get_node_name(child, show_full_index=show_full_index)
                    
                    if remove_qkv:
                        if any(qkv in child_name or qkv in parent_name for qkv in ['_q_', '_k_', '_v_']):
                            continue
                        parent_name = parent_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")
                        child_name = child_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")

                    if remove_self_loops and parent_name == child_name:
                        # Important this go after the qkv removal
                        continue
                    
                    if edge.present and (edge.edge_type != EdgeType.PLACEHOLDER or show_placeholders):
                        #print(f'Edge from {parent_name=} to {child_name=}')
                        for node_name in [parent_name, child_name]:
                            maybe_pos = {}
                            if node_name in node_pos:
                                maybe_pos["pos"] = node_pos[node_name]
                            g.add_node(
                                node_name,
                                fillcolor=colors[node_name],
                                color="black",
                                style="filled, rounded",
                                shape="box",
                                fontname="Helvetica",
                                **maybe_pos,
                            )
                        
                        g.add_edge(
                            parent_name,
                            child_name,
                            penwidth=str(minimum_penwidth * 2),
                            color=colors[parent_name] if not edge_type_colouring else EDGE_TYPE_COLORS[edge.edge_type.value],
                        )
    if fname is not None:
        base_fname = ".".join(str(fname).split(".")[:-1])

        base_path = Path(base_fname)
        base_path.mkdir(exist_ok=True)
        for k, s in groups.items():
            g2 = pgv.AGraph(directed=True, bgcolor="transparent", overlap="false", splines="true", layout="neato")
            for node_name in s:
                g2.add_node(
                    node_name,
                    style="filled, rounded",
                    shape="box",
                )
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    g2.add_edge(s[i], s[j], style="invis", weight=200)
            g2.write(path=base_path / f"{k}.gv")

        g.write(path=base_fname + ".gv")

        if not fname.endswith(".gv"): # turn the .gv file into a .png file
            g.draw(path=fname, prog="dot")

    return g


# # Run Experiment

# In[8]:


def get_nodes(correspondence):
    nodes = set()
    for child_hook_name in correspondence.edges:
        for child_index in correspondence.edges[child_hook_name]:
            for parent_hook_name in correspondence.edges[child_hook_name][child_index]:
                for parent_index in correspondence.edges[child_hook_name][child_index][parent_hook_name]:
                    edge = correspondence.edges[child_hook_name][child_index][parent_hook_name][parent_index]

                    parent = correspondence.graph[parent_hook_name][parent_index]
                    child = correspondence.graph[child_hook_name][child_index]

                    parent_name = get_node_name(parent, show_full_index=False)
                    child_name = get_node_name(child, show_full_index=False)
                    
                    if any(qkv in child_name or qkv in parent_name for qkv in ['_q_', '_k_', '_v_']):
                        continue
                    parent_name = parent_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")
                    child_name = child_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")

                    if parent_name == child_name:
                        # Important this go after the qkv removal
                        continue
                    
                    if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
                        #print(f'Edge from {parent_name=} to {child_name=}')
                        for node_name in [parent_name, child_name]:
                            nodes.add(node_name)
    return nodes


# In[10]:


from time import time
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
run_name = 'ioi_thresh_run'
pruned_nodes_per_thresh = {}
num_forward_passes_per_thresh = {}
heads_per_thresh = {}
os.makedirs(f'ims/{run_name}', exist_ok=True)
for threshold in THRESHOLDS:
    start_thresh_time = time()
    # Set up model
    # Set up experiment
    exp = TLACDCExperiment(
        model=model,
        threshold=threshold,
        # run_name=run_name,
        ds=clean_dataset.toks,
        ref_ds=corr_dataset.toks,
        metric=negative_ioi_metric,
        zero_ablation=False,
        hook_verbose=False, 
        online_cache_cpu=False,
        corrupted_cache_cpu=False,
        verbose=True,
    )
    print('Setting up graph')
    # Set up computational graph
    exp.model.reset_hooks()
    exp.setup_model_hooks(
        add_sender_hooks=True,
        add_receiver_hooks=True,
        doing_acdc_runs=False,
    )
    exp_time = time()
    print(f'Time to set up exp: {exp_time - start_thresh_time}')
    for _ in range(10):
        pruned_nodes_attr = acdc_nodes(
            model=exp.model,
            clean_input=clean_dataset.toks,
            corrupted_input=corr_dataset.toks,
            metric=ioi_metric,
            threshold=threshold,
            exp=exp,
            attr_absolute_val=True,
        ) 
        t.cuda.empty_cache()
    acdcpp_time = time()
    print(f'ACDC++ time: {acdcpp_time - exp_time}')
    heads_per_thresh[threshold] = [get_nodes(exp.corr)]
    pruned_nodes_per_thresh[threshold] = pruned_nodes_attr
    show(exp.corr, fname=f'ims/{run_name}/thresh{threshold}_before_acdc.png')
    
    start_acdc_time = time()
    while "blocks.9" not in str(exp.current_node.name):
        exp.step(testing=False)

    # # We do not have Aaquib's changes yet
    # print(f'ACDC Time: {time() - start_acdc_time}, with steps {exp.num_steps}')
    # num_forward_passes_per_thresh[threshold] = exp.num_passes

    heads_per_thresh[threshold].append(get_nodes(exp.corr))
    show(exp.corr, fname=f'ims/{run_name}/thresh{threshold}_after_acdc.png')

    del exp
    t.cuda.empty_cache()

    break


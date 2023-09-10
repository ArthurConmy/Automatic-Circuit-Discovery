# In[1]:

"""
Arthur's .py version of acdcpp_ioi.ipynb to try and SCALE
"""

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
import os
import sys
import json
import warnings
import re
from time import time
from functools import partial
import acdc
from acdc.acdc_graphics import show, get_node_name
from acdc.TLACDCInterpNode import TLACDCInterpNode
import pygraphviz as pgv
from pathlib import Path
import plotly.express as px
from acdc.TLACDCExperiment import TLACDCExperiment
import torch
from acdc.acdc_utils import TorchIndex, EdgeType
import numpy as np
import torch as t
from torch import Tensor
import einops
import itertools
from acdc.ioi_dataset import IOIDataset, format_prompt, make_table
import gc
from transformer_lens import HookedTransformer, ActivationCache
import tqdm.notebook as tqdm
import plotly
from rich import print as rprint
from rich.table import Table
from jaxtyping import Float, Bool, Int
import cProfile
from typing import Callable, Tuple, Union, Dict, Optional, Literal

warnings.warn("Running on CPU for testing")
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
# device = "cpu"
print(f'Device: {device}')

MODE: Literal["ioi", "factual_recall"] = "factual_recall"

# In[2]:

if MODE == "ioi":
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

elif MODE == "factual_recall":
    warnings.warn("Loading GPT-J takes some time. 5.75 minutes on a runpod A100. And why? On colab this takes 3 minutes, including the downloading part!")

    def load_model():
        model = HookedTransformer.from_pretrained_no_processing( # Maybe this can speedup things more?
            # "gpt2",
            "gpt-j-6b", #  "gpt-j-6b", # Can smaller models be used so there is less waiting?
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False, # This is used as doing this processing is really slow
            # device=device, # CPU here makes things slower
        )
        return model

    cProfile.runctx('model = load_model()', globals(), locals())
    warnings.warn("If using GPT-J, ensure that you have this PR merged: https://github.com/neelnanda-io/TransformerLens/pull/380")

    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)

# In[4]:

if MODE == "ioi":
    N = 25
    clean_dataset = IOIDataset(
        prompt_type='mixed',
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=1,
        device=device,
    )
    clean_toks = clean_dataset.toks
    corr_dataset = clean_dataset.gen_flipped_prompts('ABC->XYZ, BAB->XYZ')
    corr_toks = corr_dataset.toks

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

elif MODE == "factual_recall":
    # Munge the data to find some cities that we can track...

    with open(os.path.expanduser("~/Automatic-Circuit-Discovery/acdc/factual_recall/city_data.json")) as f:
        raw_data = json.load(f) # Adjust fpath to yours...

    prompt_templates = raw_data["prompt_templates"] + raw_data["prompt_templates_zs"]

    filtered_data = []

    for sample in raw_data["samples"]:
        completion = model.to_tokens(" " + sample["object"], prepend_bos=False)
        subject = model.to_tokens(" " + sample["subject"], prepend_bos=False)
        if [completion.shape[-1], subject.shape[-1]] != [1, 1]:
            print(sample, "bad")
            continue
        else:
            print("Good")
            filtered_data.append(sample)

    filtered_data = list(reversed(filtered_data))

    losses = []
    stds = []

    for sample in filtered_data: # This helps as the model is confused by China->Shanghai!
        prompts = [
            template.format(sample["subject"]) for template in prompt_templates
        ]
        batched_tokens = model.to_tokens(prompts)
        completion = model.to_tokens(" " + sample["object"], prepend_bos=False).item()
        end_pos = [model.to_tokens(prompt).shape[-1]-1 for prompt in prompts]
        assert batched_tokens.shape[1] == max(end_pos) + 1
        logits = model(batched_tokens)[torch.arange(batched_tokens.shape[0]), end_pos]
        log_probs = t.log_softmax(logits, dim=-1)
        loss = - log_probs[torch.arange(batched_tokens.shape[0]), completion]
        losses.append(loss.mean().item())
        stds.append(loss.std().item())

        print(
            sample,
            "has loss",
            round(losses[-1], 4), 
            "+-",
            round(stds[-1], 4),
        )
        print(loss.tolist())

    # Q: What are the losses here?
    fig = px.bar(
        x=[sample["subject"] for sample in filtered_data],
        y=losses,
        # error_y=dict(
        #     type="data",
        #     array=stds,
        # ),
    ).update_layout(
        title="Average losses for factual recall",
    )
    if ipython is not None:
        fig.show()
    # A: they are quite small

    # Make the data

    gc.collect()
    t.cuda.empty_cache()

    BATCH_SIZE = 5 # Make this small so no OOM...
    CORR_MODE: Literal["here", "other_city"] = "here" # replace $city_name with $other_city_name or $here

    assert len(filtered_data) >= BATCH_SIZE
    all_subjects = [sample["subject"] for sample in filtered_data]

    torch.manual_seed(0)
    prompt_template_indices = torch.randint(len(prompt_templates), (BATCH_SIZE,))

    clean_sentences = [prompt_templates[prompt_idx].format(all_subjects[subject_idx]) for subject_idx, prompt_idx in enumerate(prompt_template_indices)]
    clean_toks = model.to_tokens([sentence for sentence in clean_sentences])
    clean_end_positions = [model.to_tokens(sentence).shape[-1]-1 for sentence in clean_sentences]
    clean_completions = [model.to_tokens(" " + filtered_data[i]["object"], prepend_bos=False).item() for i in range(BATCH_SIZE)]
    clean_completions = t.tensor(clean_completions, device=device)

    if CORR_MODE == "here":
        different_subjects = ["here" for _ in range(BATCH_SIZE)]

    elif CORR_MODE == "other_city":
        different_subjects = list(set(all_subjects) - set(all_subjects[:BATCH_SIZE]))

        # This city data was too small... sigh
        different_subjects = different_subjects + different_subjects

    assert len(different_subjects) >= BATCH_SIZE
    corr_subjects = different_subjects[:BATCH_SIZE]

    corr_sentences = [sentence.replace(all_subjects[i], corr_subjects[i]) for i, sentence in enumerate(clean_sentences)]
    corr_toks = model.to_tokens(corr_sentences)

    # Check that indeed losses are low
    logits = model(clean_toks).cpu()[torch.arange(clean_toks.shape[0]), clean_end_positions]
    logprobs = t.log_softmax(logits.cpu(), dim=-1)
    loss = - logprobs.cpu()[torch.arange(clean_toks.shape[0]), clean_completions.cpu()]

#%%

print(loss, "are the losses") # Most look reasonable. But 4???s
gc.collect()
t.cuda.empty_cache()

# In[5]:

if MODE == "ioi":
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
        ioi_dataset: IOIDataset = clean_dataset, 
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

elif MODE == "factual_recall":

    def ave_loss(
        logits: Float[Tensor, 'batch seq d_vocab'],
        end_positions: Int[Tensor, 'batch'],
        correct_tokens: Int[Tensor, 'batch'],
    ):
        '''
        Return average neglogprobs of correct tokens
        '''

        end_logits = logits[range(logits.size(0)), end_positions]
        logprobs = t.log_softmax(end_logits, dim=-1)
        loss = - logprobs[range(logits.size(0)), correct_tokens]
        return loss.mean()

    factual_recall_metric = partial(ave_loss, end_positions=clean_end_positions, correct_tokens=clean_completions)

    with t.no_grad():
        clean_logits = model(clean_toks)
        corrupt_logits = model(corr_toks)

    # Get clean and corrupt logit differences
    with t.no_grad():
        clean_metric = factual_recall_metric(clean_logits)
        corrupt_metric = factual_recall_metric(corrupt_logits)

# In[6]:

# # Helper Methods

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

def get_3_caches(model, clean_input, corrupted_input, metric, device=None):
    # cache the activations and gradients of the clean inputs
    model.reset_hooks()
    clean_cache = {}

    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach().to(device) # .to(None) is a no op

    model.add_hook(hook_filter, forward_cache_hook, "fwd")

    clean_grad_cache = {}

    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach().to(device)

    model.add_hook(hook_filter, backward_cache_hook, "bwd")

    value = metric(model(clean_input))
    value.backward()

    model.zero_grad()
    gc.collect()
    t.cuda.empty_cache()

    # cache the activations of the corrupted inputs
    model.reset_hooks()
    corrupted_cache = {}

    def forward_cache_hook(act, hook):
        corrupted_cache[hook.name] = act.detach().to(device)

    model.add_hook(hook_filter, forward_cache_hook, "fwd")
    with torch.no_grad():
        model(corrupted_input)
    model.reset_hooks()

    clean_cache = ActivationCache(clean_cache, model)
    corrupted_cache = ActivationCache(corrupted_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, corrupted_cache, clean_grad_cache

def acdc_nodes(
    model: HookedTransformer,
    clean_input: Tensor,
    corrupted_input: Tensor,
    metric: Callable[[Tensor], Tensor],
    threshold: float,
    exp: TLACDCExperiment,
    attr_absolute_val: bool = False,
    device = None, # TODO add types...
) -> Tuple[HookedTransformer, Bool[Tensor, 'n_layer n_heads']]:
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
    clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(model, clean_input, corrupted_input, metric, device=device)

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

# Show resulting graph

# In[7]:

# In[8]:

# # Run Experiment

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

run_name = 'ioi_thresh_run'
pruned_nodes_per_thresh = {}
num_forward_passes_per_thresh = {}
heads_per_thresh = {}
os.makedirs(f'ims/{run_name}', exist_ok=True)
threshold = 0.01
start_thresh_time = time()

# Set up experiment
# For GPT-J this takes >3 minutes if caches are on CPU. 30 seconds if not.

exp = TLACDCExperiment(
    model=model,
    threshold=threshold,
    # run_name=run_name, # TODO add this feature to main branch acdc
    ds=clean_toks,
    ref_ds=corr_toks,
    metric=negative_ioi_metric if MODE == "ioi" else factual_recall_metric,
    zero_ablation=False,
    hook_verbose=False, 
    online_cache_cpu=False, # Trialling this being bigger...
    corrupted_cache_cpu=False,
    verbose=True,
    add_sender_hooks=False,
)
print('Setting up graph')

#%%

# Set up computational graph
exp.model.reset_hooks()
exp.setup_model_hooks(
    add_sender_hooks=True,
    add_receiver_hooks=True,
    doing_acdc_runs=False,
)
exp_time = time()
print(f'Time to set up exp: {exp_time - start_thresh_time}')

N_TIMES = 1 # Number of times such that this does not OOM GPT-J...

for _ in range(N_TIMES):
    pruned_nodes_attr = acdc_nodes(
        model=exp.model,
        clean_input=clean_toks,
        corrupted_input=corr_toks,
        metric=ioi_metric if MODE == "ioi" else factual_recall_metric,
        threshold=threshold,
        exp=exp,
        attr_absolute_val=True,
    ) 
    t.cuda.empty_cache()
acdcpp_time = time()
print(f'ACDC++ time: {acdcpp_time - exp_time}')
gc.collect()
t.cuda.empty_cache()

heads_per_thresh[threshold] = [get_nodes(exp.corr)]
pruned_nodes_per_thresh[threshold] = pruned_nodes_attr

# # I think that this is too slow in general...
# show(exp.corr, fname=f'ims/{run_name}/thresh{threshold}_before_acdc.png', show_full_index=False)
    
#%%

start_acdc_time = time()
# Set up computational graph again
exp.model.reset_hooks()
exp.setup_model_hooks(
    add_sender_hooks=True,
    add_receiver_hooks=True,
    doing_acdc_runs=False,
)
# while "blocks.9" not in str(exp.current_node.name): # I used this while condition for profiling

used_layers = set()

while exp.current_node:
# if True: # Can we at least do one step?
    current_layer = exp.current_node.name.split(".")[1]
    if current_layer not in used_layers:
        show(exp.corr, fname=f'{current_layer}_thresh_{threshold}_in_acdc.png', show_full_index=False)
        used_layers.add(current_layer)

    exp.step(testing=False)

# # TODO We do not have Aaquib's changes yet so cannot run this
print(f'ACDC Time: {time() - start_acdc_time}, with steps {exp.num_steps}')

# num_forward_passes_per_thresh[threshold] = exp.num_passes

heads_per_thresh[threshold].append(get_nodes(exp.corr))
# # TODO add this back in 
show(exp.corr, fname=f'ims/{run_name}/thresh{threshold}_after_acdc.png')

#%%

del exp
gc.collect()
t.cuda.empty_cache()

# %%

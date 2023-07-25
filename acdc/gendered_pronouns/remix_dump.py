#%%
# type: ignore
from copy import deepcopy
from typing import List
import click
import IPython
import rust_circuit as rc
import torch
from ioi_dataset import (
    IOIDataset,
)  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm
from datetime import datetime
import einops

# from rust_circuit import SLICER as S
# from rust_circuit import TORCH_INDEXER as I
from interp.tools.indexer import SLICER as S
from interp.tools.indexer import TORCH_INDEXER as I

from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.causal_scrubbing.hypothesis import corr_root_matcher
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from remix_d5_acdc_utils import (
    ACDCTemplateCorrespondence,
    ACDCCorrespondence,
    ACDCExperiment,
    ACDCInterpNode,
)
from interp.circuit.causal_scrubbing.hypothesis import Correspondence
from interp.circuit.causal_scrubbing.experiment import Experiment
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import make_arr

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore
import wandb

# with open(__file__, "r") as f:
#     file_content = f.read()

USING_WANDB = True
MONOTONE_METRIC = "maximize"

#%% [markdown]
# ## Load the data


templates = [
    "So {name} is a really great friend, isn't",
    "So {name} is such a good cook, isn't",
    "So {name} is a very good athlete, isn't",
    "So {name} is a really nice person, isn't",
    "So {name} is such a funny person, isn't",
]

male_names = [
    "John",
    "David",
    "Mark",
    "Paul",
    "Ryan",
    "Gary",
    "Jack",
    "Sean",
    "Carl",
    "Joe",
]
female_names = [
    "Mary",
    "Lisa",
    "Anna",
    "Sarah",
    "Amy",
    "Carol",
    "Karen",
    "Susan",
    "Julie",
    "Judy",
]

sentences = []
answers = []
wrongs = []

responses = [" he", " she"]

count = 0

for name in male_names + female_names:
    for template in templates:
        cur_sentence = template.format(name=name)
        sentences.append(cur_sentence)

batch_size = len(sentences)

count = 0

for _ in range(batch_size):
    if count < (0.5 * len(sentences)):
        answers.append(responses[0])
        wrongs.append(responses[1])
        count += 1
    else:
        answers.append(responses[1])
        wrongs.append(responses[0])

print(sentences)


# %% [markdown]
# Set up the model
# This is slow the first time you run it, but then it's cached so should be faster


def get_model(
    seq_len,
):
    """See interp/demos/model_loading_with_modules!!!
    This is just copied, but we need to do it here because we need to set the seq_len"""
    MODEL_ID = "gelu_12_tied"  # aka gpt2 small
    circ_dict, tokenizer, model_info = load_model_id(MODEL_ID)
    keys = list(circ_dict.keys())
    for key in tqdm(keys):
        circ_dict[key] = rc.cast_circuit(
            circ_dict[key], device_dtype=rc.TorchDeviceDtype("cuda:0", "float32").op()
        )
    unbound_circuit = circ_dict["t.bind_w"]
    assert not unbound_circuit.is_explicitly_computable
    tokens_device_dtype = rc.TorchDeviceDtype("cuda:0", "int64")
    tokens_arr = rc.cast_circuit(
        rc.Array(torch.zeros(seq_len).to(torch.long), name="tokens"),
        device_dtype=tokens_device_dtype.op(),
    )
    token_embeds = rc.GeneralFunction.gen_index(
        circ_dict["t.w.tok_embeds"], tokens_arr, 0, name="tok_embeds"
    )
    bound_circuit = model_info.bind_to_input(
        unbound_circuit, token_embeds, circ_dict["t.w.pos_embeds"]
    )
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
    renamed_circuit = subbed_circuit.update(
        rc.Regex(r"[am]\d(.h\d)?$"), lambda c: c.rename(c.name + ".inner")
    )
    renamed_circuit = renamed_circuit.update(
        "t.inp_tok_pos", lambda c: c.rename("embeds")
    )
    for l in range(model_info.params.num_layers):
        next = "final" if l == model_info.params.num_layers - 1 else f"a{l+1}"
        renamed_circuit = renamed_circuit.update(
            f"b{l}", lambda c: c.rename(f"{next}.input")
        )
        renamed_circuit = renamed_circuit.update(f"b{l}.m", lambda c: c.rename(f"m{l}"))
        renamed_circuit = renamed_circuit.update(
            f"b{l}.m.p_bias", lambda c: c.rename(f"m{l}.p_bias")
        )
        renamed_circuit = renamed_circuit.update(f"b{l}.a", lambda c: c.rename(f"a{l}"))
        renamed_circuit = renamed_circuit.update(
            f"b{l}.a.p_bias", lambda c: c.rename(f"a{l}.p_bias")
        )
        for h in range(model_info.params.num_layers):
            renamed_circuit = renamed_circuit.update(
                f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}")
            )
    renamed_circuit = rc.substitute_all_modules(renamed_circuit)

    def create_path_matcher(
        start_node: rc.MatcherIn, path: list[str], max_distance=6
    ) -> rc.IterativeMatcher:
        """
        Creates a matcher that matches a path of nodes, given in a list of names, where the
        maximum distance between each node on the path is max_distance
        """

        initial_matcher = rc.IterativeMatcher(start_node)
        max_dis_path_matcher = lambda name: rc.restrict(
            rc.Matcher(name), end_depth=max_distance
        )
        chain_matcher = initial_matcher.chain(max_dis_path_matcher(path[0]))
        for i in range(1, len(path)):
            chain_matcher = chain_matcher.chain(max_dis_path_matcher(path[i]))
        return chain_matcher

    q_path = [
        "a.comb_v",
        "a.attn_probs",
        "a.attn_scores",
        "a.attn_scores_raw",
        "a.q_p_bias",
        "a.q",
    ]
    k_path = [
        "a.comb_v",
        "a.attn_probs",
        "a.attn_scores",
        "a.attn_scores_raw",
        "a.k_p_bias",
        "a.k",
    ]
    v_path = ["a.comb_v", "a.v_p_bias", "a.v"]
    qkv_paths = {"q": q_path, "k": k_path, "v": v_path}
    num_layers = model_info.params.num_layers
    num_heads = model_info.params.num_layers
    qkv_name = "a{layer}.h{head}.{qkv}"
    new_circuit = renamed_circuit
    for l in range(num_layers):
        for h in range(num_heads):
            for qkv in ["q", "k", "v"]:
                qkv_matcher = create_path_matcher(f"a{l}.h{h}", qkv_paths[qkv])
                new_circuit = new_circuit.update(
                    qkv_matcher, lambda c: c.rename(f"a{l}.h{h}.{qkv}")
                )

    # TODO: now we create new positional nodes
    for l in range(num_layers):
        # Split MLPs by position
        old_node = new_circuit.get_unique(rc.Matcher(f"m{l}"))
        positional_nodes_list = []
        for pos in range(seq_len):
            if pos == 1:
                pos_node = rc.Index(old_node, I[pos], name=f"m{l}_[name]")
            elif pos == 2:
                pos_node = rc.Index(old_node, I[pos], name=f"m{l}_[is]")
            elif pos == 6:
                pos_node = rc.Index(old_node, I[pos], name=f"m{l}_[person]")
            elif pos == 8:
                pos_node = rc.Index(old_node, I[pos], name=f"m{l}_[isn]")
            elif pos == 9:
                pos_node = rc.Index(old_node, I[pos], name=f"m{l}_['t]")
            else:
                pos_node = rc.Index(old_node, I[pos], name=f"m{l}_[{pos}]")
            positional_nodes_list.append(pos_node)
        new_node = rc.Concat.stack(*positional_nodes_list, axis=0, name=f"m{l}_")
        new_circuit = new_circuit.update(rc.Matcher(f"m{l}"), lambda c: new_node)

        # Now do attention heads
        for h in range(num_heads):
            for qkv in ["q", "k", "v"]:
                old_node = new_circuit.get_unique(rc.Matcher(f"a{l}.h{h}.{qkv}"))
                positional_nodes_list = []
                for pos in range(seq_len):
                    if pos == 1:
                        pos_node = rc.Index(
                            old_node, I[pos], name=f"a{l}.h{h}.{qkv}_[name]"
                        )
                    elif pos == 2:
                        pos_node = rc.Index(
                            old_node, I[pos], name=f"a{l}.h{h}.{qkv}_[is]"
                        )
                    elif pos == 6:
                        pos_node = rc.Index(
                            old_node, I[pos], name=f"a{l}.h{h}.{qkv}_[person]"
                        )
                    elif pos == 8:
                        pos_node = rc.Index(
                            old_node, I[pos], name=f"a{l}.h{h}.{qkv}_[isn]"
                        )
                    elif pos == 9:
                        pos_node = rc.Index(
                            old_node, I[pos], name=f"a{l}.h{h}.{qkv}_['t]"
                        )
                    else:
                        pos_node = rc.Index(
                            old_node, I[pos], name=f"a{l}.h{h}.{qkv}_[{pos}]"
                        )
                    positional_nodes_list.append(pos_node)
                new_node = rc.Concat.stack(
                    *positional_nodes_list, axis=0, name=f"a{l}.h{h}.{qkv}_"
                )
                new_circuit = new_circuit.update(
                    rc.Matcher(f"a{l}.h{h}.{qkv}"), lambda c: new_node
                )
    new_circuit = rc.substitute_all_modules(new_circuit)
    return new_circuit, tokenizer


# TODO: hardcode the seq_len
model, tokenizer = get_model(10)
model = rc.substitute_all_modules(model)  # essential! 20x speedup
model.print_html()

# %% [markdown]
# Get the tokens

# TODO: convert to tokens, and then a correct type input into a circuit
tokens = torch.tensor(tokenizer(sentences)["input_ids"])
answers_toks = torch.tensor(tokenizer(answers)["input_ids"]).squeeze()
wrongs_toks = torch.tensor(tokenizer(wrongs)["input_ids"]).squeeze()
seq_len = tokens.shape[1]

# %% [markdown]
# Make tokens into dataset
baseline_data = tokens.clone()
baseline_data[0] = torch.tensor(
    tokenizer("That person is a really great friend, isn't")["input_ids"]
)
baseline_data = einops.repeat(baseline_data[0], "seq -> batch seq", batch=batch_size)

tokens_device_dtype = rc.TorchDeviceDtype("cuda:0", "int64")
default_data = make_arr(
    tokens,
    "tokens",
    device_dtype=tokens_device_dtype,
)
patch_data = make_arr(
    baseline_data,
    "tokens",
    device_dtype=tokens_device_dtype,
)
default_ds = Dataset({"tokens": default_data})
patch_ds = Dataset({"tokens": patch_data})

# %% [markdown]

import transformer_lens
gpt2 = transformer_lens.HookedTransformer.from_pretrained("gpt2")

for n, p in gpt2.named_parameters():
    print(n, p.requires_grad)
    p.requires_grad=False

#%%

gpt2_logits = gpt2(torch.tensor(tokenizer(sentences)["input_ids"]))
assert len(list(gpt2_logits.shape)) == 3
gpt2_probs = torch.softmax(gpt2_logits, dim=-1)

#%%

# Get metric

def pronoun_metric(
    dataset: Dataset,
    logits: torch.Tensor,
):
    logits_on_correct = logits[torch.arange(batch_size), -1, answers_toks]
    logits_on_wrong = logits[torch.arange(batch_size), -1, wrongs_toks]
    result = torch.mean(logits_on_correct - logits_on_wrong)
    return result.item()

def kl_div(
    dataset: Dataset,
    logits: torch.Tensor,
):
    probs = torch.softmax(logits, dim=-1)
    kl = torch.nn.functional.kl_div(probs[:, -1].log(), gpt2_probs[:, -1], reduction="none")
    assert len(kl.shape) == 2, kl.shape
    kl = kl.sum(dim=-1)
    return kl.mean().item()

#%% [markdown]
# Now, let's define a DAG!

attention_head_name = "a{layer}.h{head}"
mlp_layer_name = "m{layer}_"
attention_pos_name = "a{layer}.h{head}.{qkv}_{token}"
mlp_pos_name = "m{layer}_{token}"
embed_name = "tok_embeds"
root_name = "final.inp"
no_layers = 12
no_heads = 12
pos_names = [
    "[name]",
    "[is]",
    "[person]",
    "[isn]",
    "['t]",
    "[0]",
    "[3]",
    "[4]",
    "[5]",
    "[7]",
]
# all_names = (
#     [embed_name, root_name]
#     + [mlp_name.format(layer=layer) for layer in range(no_layers)]
#     + [attention_head_name.format(layer=layer, head=head) for layer in range(no_layers) for head in range(no_heads)]
# )


#%% [markdown]
# This cell makes the DAG (just like the previous cell, you'll want to adapt it to your use case)
# TODO matcher creation gets slow with big graphs. Can it be sped up?
def make_dag(no_layers, no_heads):
    all_names = (
        [embed_name, root_name]
        + [mlp_layer_name.format(layer=layer) for layer in range(no_layers)]
        + [
            mlp_pos_name.format(layer=layer, token=token)
            for layer in range(no_layers)
            for token in pos_names
        ]
        + [
            attention_head_name.format(layer=layer, head=head)
            for layer in range(no_layers)
            for head in range(no_heads)
        ]
        + [
            attention_pos_name.format(layer=layer, head=head, qkv=qkv, token=token)
            for layer in range(no_layers)
            for head in range(no_heads)
            for qkv in ["q", "k", "v"]
            for token in pos_names
        ]
    )
    all_names = set(all_names)

    template_corr = ACDCTemplateCorrespondence(all_names=all_names)
    root = ACDCInterpNode(root_name, is_root=True)
    template_corr.add_with_auto_matcher(root)

    all_residual_stream_parents: List[ACDCInterpNode] = []
    all_residual_stream_parents.append(root)

    print("Takes XXXX")
    for layer in tqdm(range(no_layers - 1, -1, -1)):
        mlp_nodes_list = []
        mlp_layer_node = ACDCInterpNode(mlp_layer_name.format(layer=layer))
        for node in all_residual_stream_parents:
            node.add_child(mlp_layer_node)
            mlp_layer_node.add_parent(node)
        template_corr.add_with_auto_matcher(mlp_layer_node)
        for token in pos_names:
            # add MLP
            mlp_pos_node = ACDCInterpNode(mlp_pos_name.format(layer=layer, token=token))
            mlp_pos_node.add_parent(mlp_layer_node)
            mlp_layer_node.add_child(mlp_pos_node)
            template_corr.add_with_auto_matcher(mlp_pos_node)
            mlp_nodes_list.append(mlp_pos_node)
        all_residual_stream_parents += mlp_nodes_list

        attention_nodes_list = []
        for head in range(no_heads):
            attention_head_node = ACDCInterpNode(
                attention_head_name.format(layer=layer, head=head)
            )
            for node in all_residual_stream_parents:
                node.add_child(attention_head_node)
                attention_head_node.add_parent(node)
            template_corr.add_with_auto_matcher(attention_head_node)
            for token in pos_names:
                for qkv in ["q", "k", "v"]:
                    # add attention
                    attention_pos_node = ACDCInterpNode(
                        attention_pos_name.format(
                            layer=layer, head=head, qkv=qkv, token=token
                        )
                    )
                    attention_pos_node.add_parent(attention_head_node)
                    attention_head_node.add_child(attention_pos_node)
                    template_corr.add_with_auto_matcher(attention_pos_node)
                    attention_nodes_list.append(attention_pos_node)
        all_residual_stream_parents += attention_nodes_list

    embed_node = ACDCInterpNode(embed_name)
    for node in tqdm(all_residual_stream_parents):
        node.add_child(embed_node)
        embed_node.add_parent(node)
    template_corr.add_with_auto_matcher(embed_node)

    diff = list(set(all_names) - set([node.name for node in all_residual_stream_parents]))
    print(diff)
    expected_diff = ["tok_embeds"] + [
        "a{layer}.h{head}".format(layer=layer, head=head)
        for layer in range(no_layers)
        for head in range(no_heads)
    ] + ["m{layer}_".format(layer=layer) for layer in range(no_layers)]
    print(expected_diff)
    # diff_from_expected =  list(set(expected_diff)) - list(set(diff))
    # print(diff_from_expected)
    assert set(diff) == set(expected_diff)

    return template_corr, all_names, all_residual_stream_parents


template_corr, all_names, all_residual_stream_parents = make_dag(no_layers, no_heads)

print("len(all_residual_stream_parents)", len(all_residual_stream_parents))
print("len(all_names)", len(all_names))

print("...done")
# you can also template_corr.print(), though we won't as this takes several minutes.

# %%

dateandtime = datetime.now().strftime("$d$m_%H-%M-%S")
template_corr.topologically_sort_corr()  # fails if there are cycles, or the DAG is disconnected
# template_corr.show(f"acdc_experiments/{dateandtime}/comp_graph" + ".png")

USING_WANDB = True

if USING_WANDB:
    file_contents=""
    with open(__file__, "r") as f:
        file_contents = f.read()
    wandb.init(entity="remix_school-of-rock", project="gender bias", name="acdc_genderbias_test_not_" + str(0.015), notes=file_contents)


exp = ACDCExperiment(
    circuit=model,
    ds=default_ds,
    ref_ds=patch_ds,
    # corr=acdc_corr,
    template_corr=template_corr,
    metric=kl_div,  # self._dataset
    second_metric=pronoun_metric,
    random_seed=1234,
    num_examples=len(patch_ds),
    check="fast",
    threshold=0.0002,
    # dynamic_threshold_method=dynamic_threshold_method,
    # dynamic_threshold_scaling=dynamic_threshold_scaling,def chain_ex
    min_num_children=0,
    skip_edges=True,
    remove_redundant=True,
    verbose=True,
    parallel_hypotheses=40,
    expand_by_importance=False,
    using_wandb=USING_WANDB,
    monotone_metric="minimize",
    expensive_update_cur=False,
    connections_recursive=False,
    # **extra_args,
)

#%%

exp.step()

#%%

idx = 0
for idx in range(100000):
    exp.step()
    exp._nodes.show(
        fname=f"acdc_experiments/{dateandtime}/acdc_plot_" + str(idx) + ".png"
    )
    if USING_WANDB:
        wandb.log(
            {
                "acdc_plot": wandb.Image(
                    f"acdc_experiments/{dateandtime}/acdc_plot_" + str(idx) + ".png"
                ),
            }
        )
    if exp.current_node is None:
        print("Done")
        break

# %%
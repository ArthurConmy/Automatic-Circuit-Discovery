from copy import deepcopy
from functools import partial
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import torch
from easy_transformer import EasyTransformer
from easy_transformer.experiments import get_act_hook
from easy_transformer.ioi_dataset import (
    IOIDataset,
)
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from collections import OrderedDict
from easy_transformer.ioi_utils import show_pp
import graphviz  # need both pip install graphviz and sudo apt-get install graphviz


def get_comp_type(comp):
    if comp.endswith("_input"):
        return comp[-7]
    else:
        return "other"


def get_hook_tuple(layer, head_idx, comp=None, input=False, model_layers=12):
    """Very cursed"""
    """warning, built for 12 layer models"""

    if layer == -1:
        assert head_idx is None, head_idx
        assert comp is None, comp
        return ("blocks.0.hook_resid_pre", None)

    if comp is None:
        if head_idx is None:
            if layer < model_layers:
                if input:
                    return (f"blocks.{layer}.hook_resid_mid", None)
                else:
                    return (f"blocks.{layer}.hook_mlp_out", None)
            else:
                assert layer == model_layers
                return (f"blocks.{layer-1}.hook_resid_post", None)
        else:
            return (f"blocks.{layer}.attn.hook_result", head_idx)

    else:  # I think the QKV case here is quite different because this is INPUT to a component, not output
        assert comp in ["q", "k", "v"]
        assert head_idx is not None
        return (f"blocks.{layer}.attn.hook_{comp}_input", head_idx)


def patch_all(z, source_act, hook):
    z[:] = source_act  # make sure to slice! Otherwise objects get copied around
    return z


def patch_positions(z, source_act, hook, positions):
    assert isinstance(
        positions, torch.Tensor
    ), "Dropped support for everything that isn't a tensor of shape (batchsize,)"
    assert (
        source_act.shape[0] == positions.shape[0] == z.shape[0]
    ), f"Batch size mismatch {source_act.shape} {positions.shape} {z.shape}"
    batch_size = source_act.shape[0]

    z[torch.arange(batch_size), positions] = source_act[
        torch.arange(batch_size), positions
    ]
    return z


def get_datasets():
    """from unity"""
    batch_size = 1
    orig = "When John and Mary went to the store, John gave a bottle of milk to Mary"
    new = "When Alice and Bob went to the store, Charlie gave a bottle of milk to Mary"
    prompts_orig = [
        {"S": "John", "IO": "Mary", "TEMPLATE_IDX": -42, "text": orig}
    ]  # TODO make ET dataset construction not need TEMPLATE_IDX
    prompts_new = [{"S": "Alice", "IO": "Bob", "TEMPLATE_IDX": -42, "text": new}]
    prompts_new[0]["text"] = new
    dataset_orig = IOIDataset(
        N=batch_size, prompts=prompts_orig, prompt_type="mixed"
    )  # TODO make ET dataset construction not need prompt_type
    dataset_new = IOIDataset(
        N=batch_size,
        prompts=prompts_new,
        prompt_type="mixed",
        manual_word_idx=dataset_orig.word_idx,
    )
    return dataset_new, dataset_orig


def direct_path_patching(
    model: EasyTransformer,
    orig_data,
    new_data,
    receivers_to_senders: Dict[
        Tuple[str, Optional[int]], List[Tuple[str, Optional[int], str]]
    ],  # TODO support for pushing back to token embeddings?
    orig_positions,  # tensor of shape (batch_size,)
    new_positions,
    initial_receivers_to_senders: Optional[
        List[Tuple[Tuple[str, Optional[int]], Tuple[str, Optional[int], str]]]
    ] = None,  # these are the only edges where we patch from new_cache
    orig_cache=None,
    new_cache=None,
) -> EasyTransformer:
    """
    Generalisation of the path_patching from the paper, where we only consider direct effects, and never indirect follow through effects.

    `intial_receivers_to_sender` is a list of pairs representing the edges we patch the new_cache connection on.

    `receiver_to_senders`: dict of (hook_name, idx, pos) -> [(hook_name, head_idx, pos), ...]
    these define all of the edges in the graph

    NOTE: This relies on several changes to Neel's library (and RR/ET main, too)
    WARNING: this implementation is fairly cursed, mostly because it is in general hard to do these sorts of things with hooks
    """

    if initial_receivers_to_senders is None:
        initial_receivers_to_senders = []
        for receiver_hook, senders in receivers_to_senders.items():
            for sender_hook in senders:
                if (sender_hook[0], sender_hook[1]) not in receivers_to_senders:
                    initial_receivers_to_senders.append((receiver_hook, sender_hook))

    # caching...
    if orig_cache is None:
        # save activations from orig
        model.reset_hooks()
        orig_cache = {}
        model.cache_all(orig_cache)
        _ = model(orig_data, prepend_bos=False)
        model.reset_hooks()
    initial_sender_hook_names = [
        sender_hook[0] for _, sender_hook in initial_receivers_to_senders
    ]
    if new_cache is None:
        # save activations from new for senders
        model.reset_hooks()
        new_cache = {}
        model.cache_some(new_cache, lambda x: x in initial_sender_hook_names)
        _ = model(new_data, prepend_bos=False)
        model.reset_hooks()
    else:
        assert all(
            [x in new_cache for x in initial_sender_hook_names]
        ), f"Incomplete new_cache. Missing {set(initial_sender_hook_names) - set(new_cache.keys())}"
    model.reset_hooks()

    # setup a way for model components to dynamically see activations from the same forward pass
    for name, hp in model.hook_dict.items():
        assert (
            "model" not in hp.ctx or hp.ctx["model"] is model
        ), "Multiple models used as hook point references!"
        hp.ctx["model"] = model
        hp.ctx["hook_name"] = name
    model.cache = (
        {}
    )  # note this cache is quite different from other caches... it is populated and used on the same forward pass

    # for specifically editing the inputs from certain previous parts
    def input_activation_editor(
        z,
        hook,
        head_idx=None,
    ):
        """Probably too many asserts, ignore them"""
        new_z = z.clone()
        N = z.shape[0]
        hook_name = hook.ctx["hook_name"]
        assert (
            len(receivers_to_senders[(hook_name, head_idx)]) > 0
        ), f"No senders for {hook_name, head_idx}, this shouldn't be attached!"

        assert len(receivers_to_senders[(hook_name, head_idx)]) > 0, (
            receivers_to_senders,
            hook_name,
            head_idx,
        )
        for sender_hook_name, sender_hook_idx, sender_head_pos in receivers_to_senders[
            (hook_name, head_idx)
        ]:
            # setup the cache that the new_cache stuff will be replaced with
            if (
                (hook_name, head_idx),
                (sender_hook_name, sender_hook_idx, sender_head_pos),
            ) in initial_receivers_to_senders:  # hopefully fires > once
                cache_to_use = orig_cache
                positions_to_use = orig_positions

            else:
                cache_to_use = hook.ctx["model"].cache
                positions_to_use = orig_positions

            # we have to do both things casewise
            if sender_hook_idx is None:
                sender_value = (
                    cache_to_use[sender_hook_name][
                        torch.arange(N), positions_to_use[sender_head_pos]
                    ]
                    - new_cache[sender_hook_name][
                        torch.arange(N), positions_to_use[sender_head_pos]
                    ]
                )
            else:
                sender_value = (
                    cache_to_use[sender_hook_name][
                        torch.arange(N),
                        positions_to_use[sender_head_pos],
                        sender_hook_idx,
                    ]
                    - new_cache[sender_hook_name][
                        torch.arange(N),
                        positions_to_use[sender_head_pos],
                        sender_hook_idx,
                    ]
                )

            if head_idx is None:
                assert (
                    new_z[torch.arange(N), positions_to_use[sender_head_pos], :].shape
                    == sender_value.shape
                ), f"{new_z.shape} != {sender_value.shape}"
                new_z[
                    torch.arange(N), positions_to_use[sender_head_pos]
                ] += sender_value
            else:
                assert (
                    new_z[
                        torch.arange(N), positions_to_use[sender_head_pos], head_idx
                    ].shape
                    == sender_value.shape
                ), f"{new_z[:, positions_to_use[sender_head_pos], head_idx].shape} != {sender_value.shape}, {positions_to_use[sender_head_pos].shape}"
                new_z[
                    torch.arange(N), positions_to_use[sender_head_pos], head_idx
                ] += sender_value

        return new_z

    # for saving and then overwriting outputs of attention and MLP layers
    def layer_output_hook(z, hook):
        hook_name = hook.ctx["hook_name"]
        hook.ctx["model"].cache[hook_name] = z.clone()  # hmm maybe CPU if debugging OOM
        assert (
            z.shape == orig_cache[hook_name].shape
        ), f"Shape mismatch: {z.shape} vs {orig_cache[hook_name].shape}"
        if hook_name == "blocks.0.hook_resid_pre" and not torch.allclose(
            z, orig_cache["blocks.0.hook_resid_pre"]
        ):
            a = 1

        z[:] = new_cache[hook_name]        
        return z

    # save the embeddings! they will be useful
    model.add_hook(name="blocks.0.hook_resid_pre", hook=layer_output_hook)

    for layer_idx in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            # if this is a receiver, then compute the input activations carefully
            for letter in ["q", "k", "v"]:
                hook_name = f"blocks.{layer_idx}.attn.hook_{letter}_input"
                if (hook_name, head_idx) in receivers_to_senders:
                    model.add_hook(
                        name=hook_name,
                        hook=partial(input_activation_editor, head_idx=head_idx),
                    )
        hook_name = f"blocks.{layer_idx}.hook_resid_mid"
        if (hook_name, None) in receivers_to_senders:
            model.add_hook(name=hook_name, hook=input_activation_editor)

        # then add the hooks that save and edit outputs
        for hook_name in [
            f"blocks.{layer_idx}.attn.hook_result",
            f"blocks.{layer_idx}.hook_mlp_out",
        ]:
            model.add_hook(
                name=hook_name,
                hook=layer_output_hook,
            )

    # don't forget hook resid post (if missed, it would just be overwritten, which is pointless)
    model.add_hook(
        name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
        hook=input_activation_editor,
    )
    return model


def make_base_receiver_sender_objects(
    important_nodes,
    both=False,
):
    initial_receivers_to_senders = []
    base_receivers_to_senders = {}

    for receiver in important_nodes:
        hook = get_hook_tuple(receiver.layer, receiver.head, input=True)

        for sender_child, _, comp in receiver.children:

            if len(sender_child.children) == 0:
                sender_hook = get_hook_tuple(sender_child.layer, sender_child.head)
                initial_receivers_to_senders.append(
                    (hook, (sender_hook[0], sender_hook[1], sender_child.position))
                )

            if comp in ["v", "k", "q"]:
                qkv_hook = get_hook_tuple(receiver.layer, receiver.head, comp)
                if qkv_hook not in base_receivers_to_senders:
                    base_receivers_to_senders[qkv_hook] = []
                sender_hook = get_hook_tuple(sender_child.layer, sender_child.head)
                base_receivers_to_senders[qkv_hook].append(
                    (sender_hook[0], sender_hook[1], sender_child.position)
                )

            else:
                if hook not in base_receivers_to_senders:
                    base_receivers_to_senders[hook] = []
                sender_hook = get_hook_tuple(sender_child.layer, sender_child.head)
                base_receivers_to_senders[hook].append(
                    (sender_hook[0], sender_hook[1], sender_child.position)
                )

    if both:
        return base_receivers_to_senders, initial_receivers_to_senders
    else:
        return base_receivers_to_senders


def logit_diff_io_s(model: EasyTransformer, dataset: IOIDataset):
    N = dataset.N
    logits = model(dataset.toks.long())
    io_logits = logits[torch.arange(N), dataset.word_idx["end"], dataset.io_tokenIDs]
    s_logits = logits[torch.arange(N), dataset.word_idx["end"], dataset.s_tokenIDs]
    return (io_logits - s_logits).mean().item()


def logit_diff_from_logits(
    logits,
    ioi_dataset,
):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    assert len(logits.shape) == 3
    assert logits.shape[0] == len(ioi_dataset)

    IO_logits = logits[
        torch.arange(len(ioi_dataset)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.io_tokenIDs,
    ]
    S_logits = logits[
        torch.arange(len(ioi_dataset)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.s_tokenIDs,
    ]

    return IO_logits - S_logits


class Node:
    def __init__(self, layer: int, head: int, position: str, resid_out: bool = False):
        self.layer = layer
        self.head = head
        assert isinstance(
            position, str
        ), f"Position must be a string, not {type(position)}"
        self.position = position
        
        self.children = []
        self.parents = []
        
        self.resid_out = resid_out

    def __repr__(self):
        return f"Node({self.layer}, {self.head}, {self.position})"

    def repr_long(self):
        return f"Node({self.layer}, {self.head}, {self.position}) with children {[child.__repr__() for child in self.children]}"

    def display(self):
        if self.resid_out:
            return "resid out"
        elif self.layer == -1:
            return f"Embed\n{self.position}"
        elif self.head is None:
            return f"mlp{self.layer}\n{self.position}"
        else:
            return f"{self.layer}.{self.head}\n{self.position}"

    def add_child(self, child, comp, score):
        assert (child, comp, score) not in self.children
        self.children.append((child, comp, score))

    def add_parent(self, parent):
        assert parent not in self.parents
        self.parents.append(parent)

    def remove_child(self, child, comp, score):
        assert (child, comp, score) in self.children
        self.children.remove((child, comp, score))

    def remove_parent(self, parent):
        assert parent in self.parents
        self.parents.remove(parent)

class Circuit:
    def __init__(
        self,
        model: EasyTransformer,
        metric: Callable[[EasyTransformer, Any], float],
        orig_data,
        new_data,
        threshold: int,
        orig_positions: OrderedDict,
        new_positions: OrderedDict,
        use_caching: bool = True,
        dataset=None,
        verbose: bool = False,
    ):
        model.reset_hooks()
        self.model = model
        self.orig_positions = orig_positions
        self.new_positions = new_positions
        assert list(orig_positions.keys()) == list(
            new_positions.keys()
        ), "Number and order of keys should be the same ... for now"
        self.node_stack = OrderedDict()
        self.populate_node_stack()
        self.current_node = self.node_stack[
            next(reversed(self.node_stack))
        ]  # last element TODO make a method or something for this
        self.root_node = self.current_node
        self.metric = metric
        self.dataset = dataset
        self.orig_data = orig_data
        self.new_data = new_data
        self.threshold = threshold
        self.default_metric = self.metric(model, dataset)
        assert not torch.allclose(
            torch.tensor(self.default_metric),
            torch.zeros_like(torch.tensor(self.default_metric)),
        ), "Default metric should not be zero"
        self.orig_cache = None
        self.new_cache = None
        if use_caching:
            self.get_caches()
        self.important_nodes = []
        self.finished = False
        self.verbose = verbose

    def populate_node_stack(self):
        for pos in self.orig_positions:
            node = Node(-1, None, pos)  # represents the embedding
            self.node_stack[(-1, None, pos)] = node

        for layer in range(self.model.cfg.n_layers):
            for head in list(range(self.model.cfg.n_heads)) + [
                None
            ]:  # includes None for mlp
                for pos in self.orig_positions:
                    node = Node(layer, head, pos)
                    self.node_stack[(layer, head, pos)] = node
        layer = self.model.cfg.n_layers
        pos = next(
            reversed(self.orig_positions)
        )  # assume the last position specified is the one that we care about in the residual stream
        resid_post = Node(layer, None, pos, resid_out=True)
        self.node_stack[
            (layer, None, pos)
        ] = resid_post  # this represents blocks.{last}.hook_resid_post

    def get_caches(self):
        if "orig_cache" in self.__dict__.keys():
            warnings.warn("Caches already exist, overwriting")

        # save activations from orig
        self.orig_cache = {}
        self.model.reset_hooks()
        self.model.cache_all(self.orig_cache)
        _ = self.model(self.orig_data, prepend_bos=False)

        # save activations from new for senders
        self.new_cache = {}
        self.model.reset_hooks()
        self.model.cache_all(self.new_cache)
        _ = self.model(self.new_data, prepend_bos=False)

    def step(
        self,
        threshold: Union[float, None] = None,
        verbose: bool = False,
        show_graphics: bool = True,
        auto_threshold: float = 0.0,
    ):
        """See mlab2 repo docs for def step"""

        if threshold is None:
            threshold = self.threshold

        _, self.current_node = self.node_stack.popitem()
        self.important_nodes.append(self.current_node)
        print("Currently evaluating", self.current_node)

        current_node_position = self.current_node.position
        for pos in self.orig_positions:
            if (
                current_node_position != pos and self.current_node.head is None
            ):  # MLPs and the end state of the residual stream only care about the last position
                continue

            receiver_hooks = []
            if self.current_node.layer == -1:
                continue  # nothing before this
            elif self.current_node.layer == self.model.cfg.n_layers:
                receiver_hooks.append((f"blocks.{self.current_node.layer-1}.hook_resid_post", None))
            elif self.current_node.head is None:
                receiver_hooks.append((f"blocks.{self.current_node.layer}.hook_resid_mid", None))
            else:
                receiver_hooks.append(
                    (f"blocks.{self.current_node.layer}.attn.hook_v_input", self.current_node.head)
                )
                receiver_hooks.append(
                    (f"blocks.{self.current_node.layer}.attn.hook_k_input", self.current_node.head)
                )
                if pos == current_node_position:
                    receiver_hooks.append(
                        (f"blocks.{self.current_node.layer}.attn.hook_q_input", self.current_node.head)
                    )  # similar story to above, only care about the last position

            for receiver_hook in receiver_hooks:
                if verbose:
                    print(f"Working on pos {pos}, receiver hook {receiver_hook}")

                # dry run, that adds all hooks

                # add the embedding node
                self.node_stack[(-1, None, pos)].add_parent(self.current_node)
                self.current_node.add_child(
                    self.node_stack[(-1, None, pos)], None, None
                )

                max_layer = min(self.model.cfg.n_layers, self.current_node.layer + (1 if receiver_hook[1] is None else 0)) # this handles attn heads, MLPs and end-state-of-residual-stream
                for l in tqdm(range(max_layer)):
                    for h in range(self.model.cfg.n_heads):
                        self.node_stack[(l, h, pos)].add_parent(self.current_node) # TODO does this fuck up the registering of parent comp and score???
                        self.current_node.add_child(
                            self.node_stack[(l, h, pos)], None, None
                        )
                    # add the MLP
                    self.node_stack[(l, None, pos)].add_parent(self.current_node)
                    self.current_node.add_child(
                        self.node_stack[(l, None, pos)], None, None
                    )

                # TODO the online version
                for l in tqdm(range(max_layer-1, -1, -1)):
                    for h in range(self.model.cfg.n_heads):
                        cur_metric = self.evaluate_circuit(override_error=True, old_mode=False) # TODO keep updated
                        self.node_stack[(l, h, pos)].remove_parent(
                            self.current_node
                        )
                        self.current_node.remove_child(
                            self.node_stack[(l, h, pos)], None, None
                        )

                        new_metric = self.evaluate_circuit(override_error=True, old_mode=False)
                        print(cur_metric, new_metric)

                        self.model.reset_hooks()
                        default = self.metric(self.model, self.dataset)
                        print(f"{default=}")

                        if abs(new_metric - cur_metric) > threshold:
                            print(
                                "Found important head:",
                                (l, h),
                                "at position",
                                pos,
                            )
                            comp_type = get_comp_type(receiver_hook[0])

                            # add back connection 
                            self.node_stack[(l, h, pos)].add_parent(
                                self.current_node
                            )
                            self.current_node.add_child(
                                self.node_stack[(l, h, pos)], None, None
                            )

                        else:
                            if self.verbose:
                                print("Found unimportant head:", (l, h), "at position", pos)

                    if l < self.current_node.layer:  # don't look at MLP n -> MLP n effect : )
                        cur_metric = self.evaluate_circuit(override_error=True, old_mode=False) # TODO keep updated
                        self.node_stack[(l, None, pos)].remove_parent(
                            self.current_node
                        )
                        self.current_node.remove_child(
                            self.node_stack[(l, None, pos)], None, None
                        )

                        new_metric = self.evaluate_circuit(override_error=True, old_mode=False)
                        print(cur_metric, new_metric)                        

                        print("Found important MLP: layer", l, "position", pos)
                        # score = mlp_results[layer, 0]
                        # comp_type = get_comp_type(receiver_hook[0])
                        self.node_stack[
                            (l, None, pos)
                        ].parents.append(  # TODO fix the MLP thing with GPT-NEO
                            (self.current_node, None, None)
                        )
                        self.current_node.add_child(
                            self.node_stack[(l, None, pos)], None, None # TODO sort out the score and comp_type
                        )

                # # deal with the embedding layer tool
                # if abs(embed_results) > threshold:
                #     print("Found important embedding layer at position", pos)
                #     score = embed_results
                #     comp_type = get_comp_type(receiver_hook[0])
                #     self.node_stack[
                #         (-1, None, pos)
                #     ].parents.append(  # TODO fix the MLP thing with GPT-NEO
                #         (self.current_node, score, comp_type)
                #     )
                #     self.current_node.add_child(
                #         (self.node_stack[(-1, None, pos)], score, comp_type)
                #     )

            if current_node_position == pos:
                break

        # update self.current_node
        while (
            len(self.node_stack) > 0
            and len(self.node_stack[next(reversed(self.node_stack))].parents) == 0
        ):
            self.node_stack.popitem()
        if len(self.node_stack) > 0:
            self.current_node = self.node_stack[next(reversed(self.node_stack))]
        else:
            self.current_node = None

    def show(self, save_file: Optional[str] = None):
        g = graphviz.Digraph(format="png")
        g.attr("node", shape="box")
        color_dict = {
            "q": "red",
            "k": "green",
            "v": "blue",
            "other": "black",
        }
        # add each layer as a subgraph with rank=same
        for layer in range(-1, self.model.cfg.n_layers):
            with g.subgraph() as s:
                s.attr(rank="same")
                for node in self.important_nodes:
                    if node.layer == layer:
                        s.node(node.display())

        def scale(num: float):
            return 3 * min(1, abs(num) ** 0.4)

        for node in self.important_nodes:
            for child in node.children:
                g.edge(
                    child[0].display(),
                    node.display(),
                    color=color_dict[(child[2] or "other")],
                    penwidth=str(scale((child[1] or 1))),
                    arrowsize=str(scale((child[1] or 1))),
                )
        # add invisible edges to keep layers separate
        for i in range(len(self.important_nodes) - 1):
            node1 = self.important_nodes[i]
            node2 = self.important_nodes[i + 1]
            if node1.layer != node2.layer:
                g.edge(node2.display(), node1.display(), style="invis")
        return g

    def get_extracted_model(self, safe: bool = True) -> EasyTransformer:
        """Return the EasyTransformer model with the extracted subgraph"""
        if safe and self.current_node is not None:
            raise RuntimeError(
                "Cannot extract model while there are still nodes to explore"
            )

    def evaluate_circuit(self, override_error=True, old_mode=False):
        """Actually run a forward pass with the current graph object"""
        
        if self.current_node is not None and not override_error:
            raise NotImplementedError("Make circuit full")

        receivers_to_senders = make_base_receiver_sender_objects(self.important_nodes)

        # what we do here is make sure that the ONLY embed objects that are set to their values on 
        # the original dataset are the ones that are in the circuit
        initial_receivers_to_senders: List[
            Tuple[Tuple[str, Optional[int]], Tuple[str, Optional[int], str]]
        ] = []

        for node in self.important_nodes:
            for child, comp, score in node.children:
                if old_mode:
                    if child.layer == -1:
                        raise NotImplementedError(
                            "I don't understand what I'm doing here"
                        )
                        initial_receivers_to_senders.append(
                            (
                                ("blocks.0.hook_resid_pre", None),
                                ("blocks.0.hook_resid_pre", None, node.position),
                            )
                        )

        if not old_mode:
            (
                receivers_to_senders,
                initial_receivers_to_senders,
            ) = make_base_receiver_sender_objects(self.important_nodes, both=True)

        
        if len(initial_receivers_to_senders) > 0:
            warnings.warn("Need at least one embedding present!!!")

        initial_receivers_to_senders = list(set(initial_receivers_to_senders))

        for pos in self.orig_positions:
            assert torch.allclose(
                self.orig_positions[pos], self.new_positions[pos]
            ), "Data must be the same for all positions"


        model = direct_path_patching(
            model=self.model,
            orig_data=self.orig_data,  # TODO sort these being different these are different
            new_data=self.new_data,
            initial_receivers_to_senders=initial_receivers_to_senders,
            receivers_to_senders=receivers_to_senders,
            orig_positions=self.orig_positions,  # tensor of shape (batch_size,)
            new_positions=self.new_positions,
            orig_cache=self.orig_cache, # TODO also sort these different
            new_cache=self.new_cache,
        )
        return self.metric(model, self.dataset)

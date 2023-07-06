#!/usr/bin/env python3

import pygraphviz as pgv
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCEdge import EdgeType, TorchIndex
from acdc.acdc_graphics import show
import tempfile
import os

from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformer import HookedTransformer
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "code"))

from subnetwork_probing.train import iterative_correspondence_from_mask
import networkx as nx
from acdc.TLACDCInterpNode import parse_interpnode


def get_transformer_config():
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=256,
        n_ctx=2048,  # chekc pos embed size
        n_heads=8,
        d_head=32,
        # model_name : str = "custom"
        # d_mlp: Optional[int] = None
        # act_fn: Optional[str] = None
        d_vocab=50259,
        # eps: float = 1e-5
        use_attn_result=True,
        use_attn_scale=True,  # divide by sqrt(d_head)
        # use_local_attn: bool = False
        # original_architecture: Optional[str] = None
        # from_checkpoint: bool = False
        # checkpoint_index: Optional[int] = None
        # checkpoint_label_type: Optional[str] = None
        # checkpoint_value: Optional[int] = None
        # tokenizer_name: Optional[str] = None
        # window_size: Optional[int] = None
        # attn_types: Optional[List] = None
        # init_mode: str = "gpt2"
        # normalization_type: Optional[str] = "LN"
        # device: Optional[str] = None
        # attention_dir: str = "causal"
        attn_only=True,
        # seed: Optional[int] = None
        # initializer_range: float = -1.0
        # init_weights: bool = True
        # scale_attn_by_inverse_layer_idx: bool = False
        positional_embedding_type="shortformer",
        # final_rms: bool = False
        # d_vocab_out: int = -1
        # parallel_attn_mlp: bool = False
        # rotary_dim: Optional[int] = None
        # n_params: Optional[int] = None
        # use_hook_tokens: bool = False
    )
    return cfg

def delete_nested_dict(d: dict, keys: list):
    inner_dicts = [d]
    try:
        for k in keys[:-1]:
            inner_dicts.append(inner_dicts[-1][k])

        del inner_dicts[-1][keys[-1]]
    except KeyError:
        return
    assert len(inner_dicts) == len(keys)

    if len(inner_dicts[-1]) == 0:
        for k_to_delete, inner_dict in reversed(list(zip(keys, inner_dicts))):
            assert not isinstance(inner_dict, dict) or len(inner_dict) == 0
            try:
                del inner_dict[k_to_delete]
            except KeyError:
                return
            if len(inner_dict) > 0:
                break

def test_count_nodes():
    nodes_to_mask_str = [
        "blocks.0.attn.hook_q[COL, COL, 0]",
        "blocks.0.attn.hook_k[COL, COL, 0]",
        "blocks.0.attn.hook_q[COL, COL, 1]",
        "blocks.0.attn.hook_k[COL, COL, 1]",
        "blocks.0.attn.hook_v[COL, COL, 1]",
        "blocks.0.attn.hook_q[COL, COL, 2]",
        "blocks.0.attn.hook_k[COL, COL, 2]",
        "blocks.0.attn.hook_v[COL, COL, 2]",
        "blocks.0.attn.hook_q[COL, COL, 3]",
        "blocks.0.attn.hook_k[COL, COL, 3]",
        "blocks.0.attn.hook_v[COL, COL, 3]",
        "blocks.0.attn.hook_q[COL, COL, 4]",
        "blocks.0.attn.hook_k[COL, COL, 4]",
        "blocks.0.attn.hook_v[COL, COL, 4]",
        "blocks.0.attn.hook_q[COL, COL, 5]",
        "blocks.0.attn.hook_k[COL, COL, 5]",
        "blocks.0.attn.hook_q[COL, COL, 6]",
        "blocks.0.attn.hook_k[COL, COL, 6]",
        "blocks.0.attn.hook_q[COL, COL, 7]",
        "blocks.0.attn.hook_k[COL, COL, 7]",
        "blocks.1.attn.hook_q[COL, COL, 0]",
        "blocks.1.attn.hook_k[COL, COL, 0]",
        "blocks.1.attn.hook_v[COL, COL, 0]",
        "blocks.1.attn.hook_q[COL, COL, 1]",
        "blocks.1.attn.hook_k[COL, COL, 1]",
        "blocks.1.attn.hook_v[COL, COL, 1]",
        "blocks.1.attn.hook_q[COL, COL, 2]",
        "blocks.1.attn.hook_k[COL, COL, 2]",
        "blocks.1.attn.hook_v[COL, COL, 2]",
        "blocks.1.attn.hook_q[COL, COL, 3]",
        "blocks.1.attn.hook_k[COL, COL, 3]",
        "blocks.1.attn.hook_q[COL, COL, 4]",
        "blocks.1.attn.hook_k[COL, COL, 4]",
        "blocks.1.attn.hook_q[COL, COL, 5]",
        "blocks.1.attn.hook_k[COL, COL, 5]",
        "blocks.1.attn.hook_q[COL, COL, 6]",
        "blocks.1.attn.hook_k[COL, COL, 6]",
        "blocks.1.attn.hook_v[COL, COL, 6]",
        "blocks.1.attn.hook_q[COL, COL, 7]",
        "blocks.1.attn.hook_k[COL, COL, 7]",
    ]
    nodes_to_mask = [parse_interpnode(s) for s in nodes_to_mask_str]
    nodes_to_mask2 = [
        TLACDCInterpNode(
            n.name.replace(".attn", "") + "_input", n.index, EdgeType.ADDITION
        )
        for n in nodes_to_mask
    ]
    nodes_to_mask += nodes_to_mask2

    cfg = get_transformer_config()
    model = HookedTransformer(cfg, is_masked=True)

    corr = TLACDCCorrespondence.setup_from_model(model)
    for child_hook_name in corr.edges:
        for child_index in corr.edges[child_hook_name]:
            for parent_hook_name in corr.edges[child_hook_name][child_index]:
                for parent_index in corr.edges[child_hook_name][child_index][
                    parent_hook_name
                ]:
                    edge = corr.edges[child_hook_name][child_index][parent_hook_name][
                        parent_index
                    ]

                    if all(
                        (child_hook_name != n.name or child_index != n.index)
                        for n in nodes_to_mask
                    ) and all(
                        (parent_hook_name != n.name or parent_index != n.index)
                        for n in nodes_to_mask
                    ):
                        edge.effect_size = 1
                        
    with tempfile.TemporaryDirectory() as tmpdir:
        g = show(corr, os.path.join(tmpdir, "out.png"), show_full_index=False)
        assert isinstance(g, pgv.AGraph)
        path = Path(tmpdir) / "out.gv"
        assert path.exists()
        # In advance I predict that it should be 41
        g2 = nx.nx_agraph.read_dot(path)

    to_delete = []
    for n in g2.nodes:
        if not nx.has_path(g2, "embed", n) or not nx.has_path(g2, n, "<resid_post>"):
            to_delete.append(n)

    for n in to_delete:
        g2.remove_node(n)

    # Delete self-loops
    for n in g2.nodes:
        if g2.has_edge(n, n):
            g2.remove_edge(n, n)
    assert len(g2.edges) == 41

    corr, _ = iterative_correspondence_from_mask(model, nodes_to_mask)
    assert corr.count_no_edges() == 41

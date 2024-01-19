import argparse
from typing import List, Optional
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import collections
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.ioi.utils import get_all_ioi_things
import huggingface_hub
import gc

import networkx as nx
import numpy as np
from acdc.docstring.utils import AllDataThings, get_all_docstring_things
from acdc.logic_gates.utils import get_all_logic_gate_things
import pandas as pd
import torch
import torch.nn.functional as F
import subnetwork_probing.transformer_lens.transformer_lens.utils as utils
from acdc.tracr_task.utils import get_all_tracr_things
from acdc.acdc_utils import reset_network
from acdc.TLACDCEdge import (
    TorchIndex,
    Edge,
    EdgeType,
)
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.induction.utils import get_all_induction_things, get_mask_repeat_candidates
from tqdm import tqdm
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformer import HookedTransformer
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from subnetwork_probing.transformer_lens.transformer_lens.ioi_dataset import IOIDataset
import wandb


def iterative_correspondence_from_mask(
    model: HookedTransformer,
    nodes_to_mask: List[TLACDCInterpNode], # Can be empty
    use_pos_embed: bool = False,
    corr: Optional[TLACDCCorrespondence] = None,
    head_parents: Optional[List] = None,
) -> Tuple[TLACDCCorrespondence, List]:
    """Given corr has some nodes masked, also mask the nodes_to_mask"""

    assert (corr is None) == (head_parents is None), "Ensure we're either masking from scratch or we provide details on `head_parents`"

    if corr is None:
        corr = TLACDCCorrespondence.setup_from_model(model, use_pos_embed=use_pos_embed)
    if head_parents is None:
        head_parents = collections.defaultdict(lambda: 0)

    additional_nodes_to_mask = []

    for node in nodes_to_mask:
        additional_nodes_to_mask.append(
            TLACDCInterpNode(node.name.replace(".attn.", ".") + "_input", node.index, EdgeType.ADDITION)
        )

        if node.name.endswith("_q") or node.name.endswith("_k") or node.name.endswith("_v"):
            child_name = node.name.replace("_q", "_result").replace("_k", "_result").replace("_v", "_result")
            head_parents[(child_name, node.index)] += 1

            if head_parents[(child_name, node.index)] == 3:
                additional_nodes_to_mask.append(TLACDCInterpNode(child_name, node.index, EdgeType.PLACEHOLDER))

            # Forgot to add these in earlier versions of Subnetwork Probing, and so the edge counts were inflated
            additional_nodes_to_mask.append(TLACDCInterpNode(child_name + "_input", node.index, EdgeType.ADDITION))

        if node.name.endswith(("mlp_in", "resid_mid")):
            additional_nodes_to_mask.append(
                TLACDCInterpNode(
                    node.name.replace("resid_mid", "mlp_out").replace("mlp_in", "mlp_out"),
                    node.index,
                    EdgeType.DIRECT_COMPUTATION,
                )
            )

    assert all([v <= 3 for v in head_parents.values()]), "We should have at most three parents (Q, K and V, connected via placeholders)"

    for node in nodes_to_mask + additional_nodes_to_mask:
        # Mark edges where this is child as not present
        rest2 = corr.edges[node.name][node.index]
        for rest3 in rest2.values():
            for edge in rest3.values():
                edge.present = False

        # Mark edges where this is parent as not present
        for rest1 in corr.edges.values():
            for rest2 in rest1.values():
                if node.name in rest2 and node.index in rest2[node.name]:
                    rest2[node.name][node.index].present = False

    return corr, head_parents

def correspondence_from_mask(model: HookedTransformer, nodes_to_mask: list[TLACDCInterpNode], use_pos_embed: bool = False) -> TLACDCCorrespondence:
    corr = TLACDCCorrespondence.setup_from_model(model, use_pos_embed=use_pos_embed)

    additional_nodes_to_mask = []

    # If all of {qkv} is masked, also add its head child
    # to the list of nodes to mask
    head_parents = collections.defaultdict(lambda: 0)
    for node in nodes_to_mask:
        additional_nodes_to_mask.append(TLACDCInterpNode(node.name.replace(".attn.", ".") + "_input", node.index, EdgeType.ADDITION))

        if node.name.endswith("_q") or node.name.endswith("_k") or node.name.endswith("_v"):
            child_name = node.name.replace("_q", "_result").replace("_k", "_result").replace("_v", "_result")
            head_parents[(child_name, node.index)] += 1

            # Forgot to add these in earlier versions of Subnetwork Probing, and so the edge counts were inflated
            additional_nodes_to_mask.append(TLACDCInterpNode(child_name + "_input", node.index, EdgeType.ADDITION))
        
        if node.name.endswith(("mlp_in", "resid_mid")):
            additional_nodes_to_mask.append(TLACDCInterpNode(node.name.replace("resid_mid", "mlp_out").replace("mlp_in", "mlp_out"), node.index, EdgeType.DIRECT_COMPUTATION))

    # assert all([v <= 3 for v in head_parents.values()])

    for (child_name, child_index), count in head_parents.items():
        if count == 3:
            nodes_to_mask.append(TLACDCInterpNode(child_name, child_index, EdgeType.ADDITION))

    for node in nodes_to_mask + additional_nodes_to_mask:
        # Mark edges where this is child as not present
        rest2 = corr.edges[node.name][node.index]
        for rest3 in rest2.values():
            for edge in rest3.values():
                edge.present = False

        # Mark edges where this is parent as not present
        for rest1 in corr.edges.values():
            for rest2 in rest1.values():
                try:
                    rest2[node.name][node.index].present = False
                except KeyError as e:
                    print("Warning: key error in correspondence_from_mask", e)
                    pass
    return corr


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


def visualize_mask(model: HookedTransformer) -> tuple[int, list[TLACDCInterpNode]]:
    number_of_heads = model.cfg.n_heads
    number_of_layers = model.cfg.n_layers
    node_name_list = []
    mask_scores_for_names = []
    total_nodes = 0
    nodes_to_mask: list[TLACDCInterpNode] = []
    for layer_index, layer in enumerate(model.blocks):
        for head_index in range(number_of_heads):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                if q_k_v == "q":
                    mask_sample = layer.attn.hook_q.sample_mask()[head_index].cpu().item()
                elif q_k_v == "k":
                    mask_sample = layer.attn.hook_k.sample_mask()[head_index].cpu().item()
                elif q_k_v == "v":
                    mask_sample = layer.attn.hook_v.sample_mask()[head_index].cpu().item()
                else:
                    raise ValueError(f"{q_k_v=} must be q, k, or v")

                node_name = f"blocks.{layer_index}.attn.hook_{q_k_v}"
                node_name_with_index = f"{node_name}[{head_index}]"
                node_name_list.append(node_name_with_index)
                node = TLACDCInterpNode(
                    node_name, TorchIndex((None, None, head_index)), incoming_edge_type=EdgeType.ADDITION
                )

                mask_scores_for_names.append(mask_sample)
                if mask_sample < 0.5:
                    nodes_to_mask.append(node)

        # MLPs
        # This is actually fairly wrong for getting the exact nodes and edges we keep in the circuit but in the `filter_nodes` function
        # used in post-processing (in roc_plot_generator.py we process hook_resid_mid/mlp_in and mlp_out hooks together properly) we iron
        # these errors so that plots are correct
        for node_name, edge_type in [
            (f"blocks.{layer_index}.hook_mlp_out", EdgeType.PLACEHOLDER),
            (f"blocks.{layer_index}.hook_resid_mid", EdgeType.ADDITION),
        ]:
            node_name_list.append(node_name)
            node = TLACDCInterpNode(node_name, TorchIndex([None]), incoming_edge_type=edge_type)
            total_nodes += 1

        mask_sample = layer.hook_mlp_out.sample_mask().cpu().item()
        mask_scores_for_names.append(mask_sample)
        if mask_sample < 0.5:
            nodes_to_mask.append(node)

    # assert len(mask_scores_for_names) == 3 * number_of_heads * number_of_layers
    log_plotly_bar_chart(x=node_name_list, y=mask_scores_for_names)
    node_count = total_nodes - len(nodes_to_mask)
    return node_count, nodes_to_mask


def regularizer(
    model: HookedTransformer,
    gamma: float = -0.1,
    zeta: float = 1.1,
    beta: float = 2 / 3,
) -> torch.Tensor:
    # TODO: globally read hyperparams from config
    # need to also do this in the masked hook point so
    # the hyperparams are the same
    def regularization_term(mask: torch.nn.Parameter) -> torch.Tensor:
        return torch.sigmoid(mask - beta * np.log(-gamma / zeta)).mean()

    mask_scores = [regularization_term(p) for (n, p) in model.named_parameters() if "mask_scores" in n]
    return torch.mean(torch.stack(mask_scores))


def do_random_resample_caching(model: HookedTransformer, train_data: torch.Tensor) -> torch.Tensor:
    for layer in model.blocks:
        layer.attn.hook_q.is_caching = True
        layer.attn.hook_k.is_caching = True
        layer.attn.hook_v.is_caching = True
        layer.hook_mlp_out.is_caching = True

    with torch.no_grad():
        outs = model(train_data)

    for layer in model.blocks:
        layer.attn.hook_q.is_caching = False
        layer.attn.hook_k.is_caching = False
        layer.attn.hook_v.is_caching = False
        layer.hook_mlp_out.is_caching = False

    return outs


def do_zero_caching(model: HookedTransformer) -> None:
    for layer in model.blocks:
        layer.attn.hook_q.cache = None
        layer.attn.hook_k.cache = None
        layer.attn.hook_v.cache = None
        layer.hook_mlp_out.cache = None


def train_induction(
    args,
    induction_model: HookedTransformer,
    all_task_things: AllDataThings,
):
    epochs = args.epochs
    lambda_reg = args.lambda_reg


    print('lambda reg', lambda_reg)

    torch.manual_seed(args.seed)

    wandb.init(
        name=args.wandb_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config=args,
        dir=args.wandb_dir,
        mode=args.wandb_mode,
    )
    test_metric_fns = all_task_things.test_metrics

    print("Reset subject:", args.reset_subject)
    if args.reset_subject:
        reset_network(args.task, args.device, induction_model)
        gc.collect()
        torch.cuda.empty_cache()
        induction_model.freeze_weights()

        reset_logits = do_random_resample_caching(induction_model, all_task_things.validation_data)
        print("Reset validation metric: ", all_task_things.validation_metric(reset_logits))
        reset_logits = do_random_resample_caching(induction_model, all_task_things.test_data)
        print("Reset test metric: ", {k: v(reset_logits).item() for k, v in all_task_things.test_metrics.items()})

    # one parameter per thing that is masked
    mask_params = [p for n, p in induction_model.named_parameters() if "mask_scores" in n and p.requires_grad]
    # parameters for the probe (we don't use a probe)
    model_params = [p for n, p in induction_model.named_parameters() if "mask_scores" not in n and p.requires_grad]
    assert len(model_params) == 0, ("MODEL should be empty", model_params)
    trainer = torch.optim.Adam(mask_params, lr=args.lr)

    if args.zero_ablation:
        do_zero_caching(induction_model)
    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        if not args.zero_ablation:
            do_random_resample_caching(induction_model, all_task_things.validation_patch_data)
        induction_model.train()
        trainer.zero_grad()

        specific_metric_term = all_task_things.validation_metric(induction_model(all_task_things.validation_data))
        regularizer_term = regularizer(induction_model)
        loss = specific_metric_term + regularizer_term * lambda_reg
        loss.backward()

        trainer.step()

    number_of_nodes, nodes_to_mask = visualize_mask(induction_model)
    wandb.log(
        {
            "regularisation_loss": regularizer_term.item(),
            "specific_metric_loss": specific_metric_term.item(),
            "total_loss": loss.item(),
        }
    )

    with torch.no_grad():
        # The loss has a lot of variance so let's just average over a few runs with the same seed
        rng_state = torch.random.get_rng_state()

        # Final training loss
        specific_metric_term = 0.0
        for _ in range(args.n_loss_average_runs):
            if args.zero_ablation:
                do_zero_caching(induction_model)
            else:
                do_random_resample_caching(induction_model, all_task_things.validation_patch_data)
            specific_metric_term += all_task_things.validation_metric(
                induction_model(all_task_things.validation_data)
            ).item()
        print(f"Final train/validation metric: {specific_metric_term:.4f}")

        test_specific_metrics = {}
        for k, fn in test_metric_fns.items():
            torch.random.set_rng_state(rng_state)
            test_specific_metric_term = 0.0
            # Test loss
            for _ in range(args.n_loss_average_runs):
                if args.zero_ablation:
                    do_zero_caching(induction_model)
                else:
                    do_random_resample_caching(induction_model, all_task_things.test_patch_data)
                test_specific_metric_term += fn(induction_model(all_task_things.test_data)).item()
            test_specific_metrics[f"test_{k}"] = test_specific_metric_term

        print(f"Final test metric: {test_specific_metrics}")

        to_log_dict = dict(
            number_of_nodes=number_of_nodes,
            specific_metric=specific_metric_term,
            nodes_to_mask=nodes_to_mask,
            **test_specific_metrics,
        )

    return induction_model, to_log_dict


# check regularizer can set all the
def sanity_check_with_transformer_lens(mask_dict):
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1)
    train_data = ioi_dataset.toks.long()
    model = HookedTransformer.from_pretrained(is_masked=False, model_name="model")
    model.freeze_weights()
    logits = model(train_data)
    logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)

    fwd_hooks = make_forward_hooks(mask_dict)
    logits = model.run_with_hooks(train_data, return_type="logits", fwd_hooks=fwd_hooks)
    logit_diff_masked = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
    print("original logit diff", logit_diff)
    print("masked logit diff", logit_diff_masked)


def make_forward_hooks(mask_dict):
    number_of_heads = model.cfg.n_heads
    number_of_layers = model.cfg.n_layers
    forward_hooks = []
    for layer in range(number_of_layers):
        for head in range(number_of_heads):
            for qkv in ["q", "k", "v"]:
                mask_value = mask_dict[f"{layer}.{head}.{qkv}"]

                def head_ablation_hook(value, hook, head_idx, layer_idx, qkv_val, mask_value):
                    value[:, :, head_idx, :] *= mask_value
                    return value

                a_hook = (
                    utils.get_act_name(qkv, int(layer)),
                    partial(
                        head_ablation_hook,
                        head_idx=head,
                        layer_idx=layer,
                        qkv_val=qkv,
                        mask_value=mask_value,
                    ),
                )
                forward_hooks.append(a_hook)
    return forward_hooks


def log_percentage_binary(mask_val_dict: Dict) -> float:
    binary_count = 0
    total_count = 0
    for _, v in mask_val_dict.items():
        total_count += 1
        if v == 0 or v == 1:
            binary_count += 1
    return binary_count / total_count


def get_nodes_mask_dict(model: HookedTransformer):
    number_of_heads = model.cfg.n_heads
    number_of_layers = model.cfg.n_layers
    mask_value_dict = {}
    for layer_index, layer in enumerate(model.blocks):
        for head_index in range(number_of_heads):
            for q_k_v in ["q", "k", "v"]:
                # total_nodes += 1
                if q_k_v == "q":
                    mask_value = layer.attn.hook_q.sample_mask()[head_index].cpu().item()
                if q_k_v == "k":
                    mask_value = layer.attn.hook_k.sample_mask()[head_index].cpu().item()
                if q_k_v == "v":
                    mask_value = layer.attn.hook_v.sample_mask()[head_index].cpu().item()
                mask_value_dict[f"{layer_index}.{head_index}.{q_k_v}"] = mask_value
    return mask_value_dict


parser = argparse.ArgumentParser("train_induction")
parser.add_argument("--wandb-name", type=str, default="subnetwork-probing")
parser.add_argument("--wandb-project", type=str, default="subnetwork-probing")
parser.add_argument("--wandb-entity", type=str,  default="remix_school-of-rock")
parser.add_argument("--wandb-group", type=str,  default="subnetwork-probing")
parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
parser.add_argument("--wandb-mode", type=str, default="online")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--loss-type", type=str,  default='kl_div')
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--lambda_reg", type=float, default=50)
parser.add_argument("--zero-ablation", type=int, default=True)
parser.add_argument("--reset-subject", type=int, default=0)
parser.add_argument("--seed", type=int, default=random.randint(0, 2**31 - 1), help="Random seed (default: random)")
parser.add_argument("--num-examples", type=int, default=50)
parser.add_argument("--seq-len", type=int, default=300)
parser.add_argument("--n-loss-average-runs", type=int, default=20)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--torch-num-threads", type=int, default=0, help="How many threads to use for torch (0=all)")


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


if __name__ == "__main__":
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    torch.manual_seed(args.seed)

    if args.task == "ioi":
        all_task_things = get_all_ioi_things(
            num_examples=args.num_examples,
            device=torch.device(args.device),
            metric_name=args.loss_type,
        )
    elif args.task == "induction":
        all_task_things = get_all_induction_things(
            args.num_examples,
            args.seq_len,
            device=torch.device(args.device),
            metric=args.loss_type,
        )
    elif args.task == "tracr-reverse":
        all_task_things = get_all_tracr_things(
            task="reverse", metric_name=args.loss_type, num_examples=args.num_examples, device=torch.device(args.device)
        )
    elif args.task == "tracr-proportion":
        all_task_things = get_all_tracr_things(
            task="proportion",
            metric_name=args.loss_type,
            num_examples=args.num_examples,
            device=torch.device(args.device),
        )
    elif args.task == "docstring":
        all_task_things = get_all_docstring_things(
            num_examples=args.num_examples,
            seq_len=args.seq_len,
            device=torch.device(args.device),
            metric_name=args.loss_type,
            correct_incorrect_wandb=True,
        )
    elif args.task == "greaterthan":
        all_task_things = get_all_greaterthan_things(
            num_examples=args.num_examples,
            metric_name=args.loss_type,
            device=args.device,
        )
    elif args.task == "or_gate":
        num_examples = 1
        seq_len = 1

        all_task_things = get_all_logic_gate_things(
            mode="OR",
            num_examples=num_examples,
            seq_len=seq_len,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown task {args.task}")

    kwargs = dict(**all_task_things.tl_model.cfg.__dict__)
    for kwarg_string in [
        "use_split_qkv_input",
        "n_devices",
        "gated_mlp",
        "use_attn_in",
        "use_hook_mlp_in",
        "default_prepend_bos",
        "dtype",
        "add_special_tokens",
    ]:
        if kwarg_string in kwargs:
            del kwargs[kwarg_string]

    cfg = HookedTransformerConfig(**kwargs)
    model = HookedTransformer(cfg, is_masked=True)

    _acdc_model = all_task_things.tl_model
    model.load_state_dict(_acdc_model.state_dict(), strict=False)
    model = model.to(args.device)
    # Check that the model's outputs are the same
    torch.testing.assert_allclose(
        do_random_resample_caching(model, all_task_things.validation_data),
        _acdc_model(all_task_things.validation_data),
        atol=1e-3,
        rtol=1e-2,
    )
    del _acdc_model
    all_task_things.tl_model = None

    model.freeze_weights()
    print("Finding subnetwork...")
    model, to_log_dict = train_induction(
        args=args,
        induction_model=model,
        all_task_things=all_task_things,
    )

    corr, _ = iterative_correspondence_from_mask(model, to_log_dict["nodes_to_mask"])
    mask_val_dict = get_nodes_mask_dict(model)
    
    percentage_binary = log_percentage_binary(mask_val_dict)

    # Update dict with some different things
    to_log_dict["nodes_to_mask"] = list(map(str, to_log_dict["nodes_to_mask"]))
    to_log_dict["number_of_edges"] = corr.count_no_edges()
    to_log_dict["percentage_binary"] = percentage_binary

    wandb.log(to_log_dict)
    # sanity_check_with_transformer_lens(mask_val_dict)
    wandb.finish()

from collections import OrderedDict
from acdc.TLACDCEdge import (
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!
from acdc.TLACDCInterpNode import TLACDCInterpNode
import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
from acdc.acdc_utils import kl_divergence
import torch
from acdc.ioi.ioi_dataset import IOIDataset  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm
import wandb
from transformer_lens.HookedTransformer import HookedTransformer
import warnings
from functools import partial
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from acdc.acdc_utils import kl_divergence, TorchIndex
from acdc.docstring.utils import AllDataThings
from acdc.ioi.utils import get_gpt2_small
from collections import OrderedDict

NOUNS = [
    "abduction", "accord", "affair", "agreement", "appraisal",
    "assaults", "assessment", "attack", "attempts", "campaign", 
    "captivity", "case", "challenge", "chaos", "clash", 
    "collaboration", "coma", "competition", "confrontation", "consequence", 
    "conspiracy", "construction", "consultation", "contact",
    "contract", "convention", "cooperation", "custody", "deal", 
    "decline", "decrease", "demonstrations", "development", "disagreement", 
    "disorder", "dispute", "domination", "dynasty", "effect", 
    "effort", "employment", "endeavor", "engagement",
    "epidemic", "evaluation", "exchange", "existence", "expansion", 
    "expedition", "experiments", "fall", "fame", "flights",
    "friendship", "growth", "hardship", "hostility", "illness", 
    "impact", "imprisonment", "improvement", "incarceration",
    "increase", "insurgency", "invasion", "investigation", "journey", 
    "kingdom", "marriage", "modernization", "negotiation",
    "notoriety", "obstruction", "operation", "order", "outbreak", 
    "outcome", "overhaul", "patrols", "pilgrimage", "plague",
    "plan", "practice", "process", "program", "progress", 
    "project", "pursuit", "quest", "raids", "reforms", 
    "reign", "relationship",
    "retaliation", "riot", "rise", "rivalry", "romance", 
    "rule", "sanctions", "shift", "siege", "slump", 
    "stature", "stint", "strikes", "study",
    "test", "testing", "tests", "therapy", "tour", 
    "tradition", "treaty", "trial", "trip", "unemployment", 
    "voyage", "warfare", "work",
]

# %%
class GreaterThanConstants:
    YEARS: list[str]
    YEARS_BY_CENTURY: dict[str, list[str]]
    TOKENS: list[int]
    INV_TOKENS: dict[int, int]
    TOKENS_TENSOR: torch.Tensor
    INV_TOKENS_TENSOR: torch.Tensor

    _instance: ClassVar[Optional["GreaterThanConstants"]] = None

    @classmethod
    def get(cls: type["GreaterThanConstants"], device) -> "GreaterThanConstants":
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance

    def __init__(self, device):
        model = get_gpt2_small(device=device)
        _TOKENIZER = model.tokenizer
        del model

        self.YEARS = []
        self.YEARS_BY_CENTURY = {}

        for century in range(11, 18):
            all_success = []
            for year in range(century * 100 + 2, (century * 100) + 99):
                a = _TOKENIZER.encode(f" {year}")
                if a == [_TOKENIZER.encode(f" {str(year)[:2]}")[0], _TOKENIZER.encode(str(year)[2:])[0]]:
                    all_success.append(str(year))
                    continue
            self.YEARS.extend(all_success[1:-1])
            self.YEARS_BY_CENTURY[century] = all_success[1:-1]

        TOKENS = {
            i: _TOKENIZER.encode(f"{'0' if i<=9 else ''}{i}")[0] for i in range(0, 100)
        }
        self.INV_TOKENS = {v: k for k, v in TOKENS.items()}
        self.TOKENS = TOKENS

        TOKENS_TENSOR = torch.as_tensor([TOKENS[i] for i in range(0, 100)], dtype=torch.long)
        INV_TOKENS_TENSOR = torch.zeros(50290, dtype=torch.long)
        for i, v in enumerate(TOKENS_TENSOR):
            INV_TOKENS_TENSOR[v] = i

        self.TOKENS_TENSOR = TOKENS_TENSOR
        self.INV_TOKENS_TENSOR = INV_TOKENS_TENSOR

def greaterthan_metric_reference(logits, tokens):
    constants = GreaterThanConstants.get(logits.device)

    probs = F.softmax(logits[:, -1], dim=-1) # last elem???
    ans = 0.0
    for i in range(len(probs)):
        yearend = constants.INV_TOKENS[tokens[i][7].item()]
        for year_suff in range(yearend+1, 100):
            ans += probs[i, constants.TOKENS[year_suff]]
        for year_pref in range(0, yearend+1):
            ans -= probs[i, constants.TOKENS[year_pref]]
    return - float(ans / len(probs))

def greaterthan_metric(logits, tokens, return_one_element: bool=True):
    constants = GreaterThanConstants.get(logits.device)

    probs = F.softmax(logits[:, -1], dim=-1)
    csum = torch.cumsum(probs[:, constants.TOKENS_TENSOR], dim=-1)
    yearend = constants.INV_TOKENS_TENSOR[tokens[:, 7]].to(logits.device)

    # At or after: positive term
    range = torch.arange(len(yearend))
    positive = csum[:, -1]
    # Before: negative term
    negative = torch.where(yearend == 0, torch.zeros((), device=csum.device), csum[range, yearend])
    if return_one_element:
        return - (positive - 2*negative).mean()
    else:
        return - (positive - 2*negative)


def get_year_data(num_examples, model):
    constants = GreaterThanConstants.get(model.cfg.device)

    template = "The {noun} lasted from the year {year1} to "

    # set some random seed
    torch.random.manual_seed(54)
    nouns_perm = torch.randint(0, len(NOUNS), (num_examples,))
    years_perm = torch.randint(0, len(constants.YEARS), (num_examples,))

    prompts = []
    prompts_tokenized = []
    for i in range(num_examples):
        year = constants.YEARS[years_perm[i]]
        prompts.append(
            template.format(
                noun=NOUNS[nouns_perm[i]],
                year1=year,
            ) + year[:2]
        )
        prompts_tokenized.append(model.tokenizer.encode(prompts[-1], return_tensors="pt").to(model.cfg.device))
        assert prompts_tokenized[-1].shape == prompts_tokenized[0].shape, (prompts_tokenized[-1].shape, prompts_tokenized[0].shape)
    prompts_tokenized = torch.cat(prompts_tokenized, dim=0)
    assert len(prompts_tokenized.shape) == 2, prompts_tokenized.shape

    return prompts_tokenized, prompts

def get_all_greaterthan_things(num_examples, metric_name, device="cuda"):
    model = get_gpt2_small(device=device)
    data, prompts = get_year_data(num_examples*2, model)
    patch_data = data.clone()
    patch_data[:, 7] = 486  # replace with 01

    validation_data = data[:num_examples]
    validation_patch_data = patch_data[:num_examples]

    test_data = data[num_examples:]
    test_patch_data = patch_data[num_examples:]

    with torch.no_grad():
        base_logits = model(data)[:, -1, :]
        base_logprobs = F.log_softmax(base_logits, dim=-1)
        base_validation_logprobs = base_logprobs[:num_examples]
        base_test_logprobs = base_logprobs[num_examples:]

    if metric_name == "greaterthan":
        validation_metric = partial(greaterthan_metric, tokens=validation_data.cpu())
    elif metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            mask_repeat_candidates=None,
            last_seq_element_only=True,
        )
    else:
        raise ValueError(f"Unknown metric {metric_name}")

    test_metrics = {
        "greaterthan": partial(greaterthan_metric, tokens=test_data.cpu()),
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            mask_repeat_candidates=None,
            last_seq_element_only=True,
        ),
    }

    return AllDataThings(
        tl_model=model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=None,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=None,
        test_mask=None,
        test_patch_data=test_patch_data)


CIRCUIT = {
    # "input": [None], # special case input
    "0305": [(0, 3), (0, 5)],
    "01": [(0, 1)],
    "MEARLY": [(0, None), (1, None), (2, None), (3, None)],
    "AMID": [(5, 5), (6, 1), (6, 9), (7, 10), (8, 11), (9, 1)],
    "MLATE": [(8, None), (9, None), (10, None), (11, None)],
    # output special case
}

def get_greaterthan_true_edges(model):
    from subnetwork_probing.train import iterative_correspondence_from_mask

    corr, _ = iterative_correspondence_from_mask(
        model=model,
        nodes_to_mask = [],
    )
    for t, e in corr.all_edges().items():
        e.present = False

    connected_pairs = [
        ("01", "MEARLY"),
        ("01", "AMID"),
        ("0305", "AMID"),
        ("MEARLY", "AMID"),
        ("AMID", "MLATE"),
    ]

    def tuple_to_hooks(layer_idx, head_idx, outp=False):
        if outp:
            if head_idx is None:
                return [(f"blocks.{layer_idx}.hook_mlp_out", TorchIndex([None]))]
            else:
                return [(f"blocks.{layer_idx}.attn.hook_result", TorchIndex([None, None, head_idx]))]

        else:
            if head_idx is None:
                return [(f"blocks.{layer_idx}.hook_mlp_in", TorchIndex([None]))]
            else:
                ret = []
                for letter in "qkv":
                    ret.append((f"blocks.{layer_idx}.hook_{letter}_input", TorchIndex([None, None, head_idx])))
                return ret

    # attach input
    for GROUP in ["0305", "01", "MEARLY"]:
        for i, j in CIRCUIT[GROUP]:
            inps = tuple_to_hooks(i, j, outp=False)

            for hook_name, index in inps:
                corr.edges[hook_name][index]["blocks.0.hook_resid_pre"][TorchIndex([None])].present = True

    # attach output
    for GROUP in ["AMID", "MLATE"]:
        for i, j in CIRCUIT[GROUP]:
            outps = tuple_to_hooks(i, j, outp=True)
            for hook_name, index in outps:
                corr.edges["blocks.11.hook_resid_post"][TorchIndex([None])][hook_name][index].present = True

    # MLPs are interconnected
    for GROUP in CIRCUIT.keys():
        if CIRCUIT[GROUP][0][1] is not None: continue
        for i1, j1 in CIRCUIT[GROUP]:
            for i2, j2 in CIRCUIT[GROUP]:
                if i1 >= i2: continue
                corr.edges[f"blocks.{i2}.hook_mlp_in"][TorchIndex([None])][f"blocks.{i1}.hook_mlp_out"][TorchIndex([None])].present = True

    # connected pairs  
    for GROUP1, GROUP2 in connected_pairs:
        for i1, j1 in CIRCUIT[GROUP1]:
            for i2, j2 in CIRCUIT[GROUP2]:
                if i1 >= i2 and not (i1==i2 and j1 is not None and j2 is None):
                    continue
                for ii, jj in tuple_to_hooks(i1, j1, outp=True):
                    for iii, jjj in tuple_to_hooks(i2, j2, outp=False): # oh god I am so sorry poor code reade
                        corr.edges[iii][jjj][ii][jj].present = True

    # Connect qkv to heads
    for (layer, head) in sum(CIRCUIT.values(), start=[]):
        if head is None: continue
        for letter in "qkv":
            e = corr.edges[f"blocks.{layer}.attn.hook_{letter}"][TorchIndex([None, None, head])][f"blocks.{layer}.hook_{letter}_input"][TorchIndex([None, None, head])]
            e.present = True
            # print(e.edge_type)
            e = corr.edges[f"blocks.{layer}.attn.hook_result"][TorchIndex([None, None, head])][f"blocks.{layer}.attn.hook_{letter}"][TorchIndex([None, None, head])]
            e.present = True
            # print(e.edge_type)

    # Hanna et al have totally clean query inputs to AMID heads --- this is A LOT of edges so we add the MLP -> AMID Q edges

    MAX_AMID_LAYER = max([layer_idx for layer_idx, head_idx in CIRCUIT["AMID"]])
    # connect all MLPs before the AMID heads
    for mlp_sender_layer in range(0, MAX_AMID_LAYER):
        for mlp_receiver_layer in range(1+mlp_sender_layer, MAX_AMID_LAYER):
            corr.edges[f"blocks.{mlp_receiver_layer}.hook_mlp_in"][TorchIndex([None])][f"blocks.{mlp_sender_layer}.hook_mlp_out"][TorchIndex([None])].present = True
    
    # connect all early MLPs to AMID heads
    for layer_idx, head_idx in CIRCUIT["AMID"]:
        for mlp_sender_layer in range(0, layer_idx):
            corr.edges[f"blocks.{layer_idx}.hook_q_input"][TorchIndex([None, None, head_idx])][f"blocks.{mlp_sender_layer}.hook_mlp_out"][TorchIndex([None])].present = True

    ret =  OrderedDict({(t[0], t[1].hashable_tuple, t[2], t[3].hashable_tuple): e.present for t, e in corr.all_edges().items() if e.present})
    return ret


GROUP_COLORS = {
    "0305": "#d7f8ee",
    "01": "#e7f2da",
    "MEARLY": "#fee7d5",
    "AMID": "#ececf5",
    "MLATE": "#fff6db",
}
MLP_COLOR = "#f0f0f0"

def greaterthan_group_colorscheme():
    assert set(GROUP_COLORS.keys()) == set(CIRCUIT.keys())

    scheme = {
        "embed": "#cbd5e8",
        "<resid_post>": "#fff2ae",
    }

    for i in range(12):
        scheme[f"<m{i}>"] = MLP_COLOR

    for k, heads in CIRCUIT.items():
        for (layer, head) in heads:
            if head is None:
                scheme[f"<m{layer}>"] = GROUP_COLORS[k]
            else:
                for qkv in ["", "_q", "_k", "_v"]:
                    scheme[f"<a{layer}.{head}{qkv}>"] = GROUP_COLORS[k]
    return scheme

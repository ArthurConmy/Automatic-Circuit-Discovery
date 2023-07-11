# %%

from transformer_lens.cautils.notebook import *

from dataclasses import dataclass, field
from typing import Any
Head = Tuple[int, int]

def parse_str(s: str):
    doubles = "“”"
    singles = "‘’"
    for char in doubles: s = s.replace(char, '"')
    for char in singles: s = s.replace(char, "'")
    return s

def parse_str_tok_for_printing(s: str):
    s = s.replace("\n", "\\n")
    return s


# %% [markdown]

class HeadResults:
    data: Dict[Head, Tensor]
    def __init__(self, data=None):
        if data is None: # ! bad practice to have default arguments be dicts
            data = {}
        self.data = data

    def __getitem__(self, layer_and_head) -> Tensor:
        return self.data[layer_and_head].clone()
    
    def __setitem__(self, layer_and_head, value):
        self.data[layer_and_head] = value.clone()

@dataclass(frozen=False)
class LogitResults:
    zero_patched: HeadResults = HeadResults()
    mean_patched: HeadResults = HeadResults()
    zero_direct: HeadResults = HeadResults()
    mean_direct: HeadResults = HeadResults()

@dataclass(frozen=False)
class ModelResults:
    logits_orig: Tensor = t.empty(0)
    loss_orig: Tensor = t.empty(0)
    result: HeadResults = HeadResults()
    result_mean: HeadResults = HeadResults()
    pattern: HeadResults = HeadResults()
    v: HeadResults = HeadResults() # for value-weighted attn
    out_norm: HeadResults = HeadResults() # for value-weighted attn
    direct_effect: HeadResults = HeadResults()
    direct_effect_mean: HeadResults = HeadResults()
    scale: Tensor = t.empty(0)
    logits: LogitResults = LogitResults()
    loss: LogitResults = LogitResults()

    def clear(self):
        # Empties all intermediate results which we don't need
        self.result = HeadResults()
        self.result_mean = HeadResults()
        self.v = HeadResults()


def get_data_dict(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    negative_heads: List[Tuple[int, int]],
    use_cuda: bool = False,
):
    model.reset_hooks(including_permanent=True)
    t.cuda.empty_cache()

    device = str(model.cfg.device)
    if use_cuda: model = model.cuda()
    else: model = model.cpu()

    model_results = ModelResults()

    # Cache the head results and attention patterns, and final ln scale

    def cache_head_result(result: Float[Tensor, "batch seq n_heads d_model"], hook: HookPoint, head: int):
        model_results.result[hook.layer(), head] = result[:, :, head]
    
    def cache_head_pattern(pattern: Float[Tensor, "batch n_heads seq_Q seq_K"], hook: HookPoint, head: int):
        model_results.pattern[hook.layer(), head] = pattern[:, head]
    
    def cache_head_v(v: Float[Tensor, "batch seq n_heads d_head"], hook: HookPoint, head: int):
        model_results.v[hook.layer(), head] = v[:, :, head]
    
    def cache_scale(scale: Float[Tensor, "batch seq 1"], hook: HookPoint):
        model_results.scale = scale

    for layer, head in negative_heads:
        model.add_hook(utils.get_act_name("result", layer), partial(cache_head_result, head=head))
        model.add_hook(utils.get_act_name("v", layer), partial(cache_head_v, head=head))
        model.add_hook(utils.get_act_name("pattern", layer), partial(cache_head_pattern, head=head))
    model.add_hook(utils.get_act_name("scale"), cache_scale)

    # Run the forward pass, to cache all values (and get logits)

    model_results.logits_orig, model_results.loss_orig = model(toks, return_type="both", loss_per_token=True)

    # Get output norms for value-weighted attention

    for layer, head in negative_heads:
        out = einops.einsum(
            model_results.v[layer, head], model.W_O[layer, head],
            "batch seq d_head, d_head d_model -> batch seq d_model"
        )
        out_norm = einops.reduce(out.pow(2), "batch seq d_model -> batch seq", "sum").sqrt()
        model_results.out_norm[layer, head] = out_norm

    # Calculate the thing we'll be subbing in for mean ablation

    for layer, head in negative_heads:
        model_results.result_mean[layer, head] = einops.reduce(
            model_results.result[layer, head], 
            "batch seq d_model -> d_model", "mean"
        )

    # Now, use "result" to get the thing we'll eventually be adding to logits (i.e. scale it and map it through W_U)

    for layer, head in negative_heads:

        # TODO - is it more reasonable to patch in at the final value of residual stream instead of directly changing logits?
        model_results.direct_effect[layer, head] = einops.einsum(
            model_results.result[layer, head] / model_results.scale,
            model.W_U,
            "batch seq d_model, d_model d_vocab -> batch seq d_vocab"
        )
        model_results.direct_effect_mean[layer, head] = einops.reduce(
            model_results.direct_effect[layer, head],
            "batch seq d_vocab -> d_vocab",
            "mean"
        )

    # Two new forward passes: one with mean ablation, one with zero ablation. We only store logits from these

    def patch_head_result(
        result: Float[Tensor, "batch seq n_heads d_model"],
        hook: HookPoint,
        head: int,
        ablation_values: Optional[HeadResults] = None,
    ):
        if ablation_values is None:
            result[:, :, head] = t.zeros_like(result[:, :, head])
        else:
            result[:, :, head] = ablation_values[hook.layer(), head]
        return result

    for layer, head in negative_heads:
        model.add_hook(utils.get_act_name("result", layer), partial(patch_head_result, head=head))
        model_results.logits.zero_patched[layer, head] = model(toks, return_type="logits")
        model.add_hook(utils.get_act_name("result", layer), partial(patch_head_result, head=head, ablation_values=model_results.result_mean))
        model_results.logits.mean_patched[layer, head] = model(toks, return_type="logits")
    
    model_results.clear()

    # Now, the direct effects

    for layer, head in negative_heads:
        # Get the change in logits from removing the direct effect of the head
        model_results.logits.zero_direct[layer, head] = model_results.logits_orig - model_results.direct_effect[layer, head]
        # Get the change in logits from removing the direct effect of the head, and replacing with the mean effect
        model_results.logits.mean_direct[layer, head] = model_results.logits.zero_direct[layer, head] + model_results.direct_effect_mean[layer, head]

    # Calculate the loss for all of these
    for k in ["zero_patched", "mean_patched", "zero_direct", "mean_direct"]:
        setattr(model_results.loss, k, HeadResults({
            (layer, head): model.loss_fn(getattr(model_results.logits, k)[layer, head], toks, per_token=True)
            for layer, head in negative_heads
        }))

    model = model.to(device)
    return model_results


# %%

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = t.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()

# %%
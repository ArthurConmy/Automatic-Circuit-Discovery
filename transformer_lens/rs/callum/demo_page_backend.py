# %%

from transformer_lens.cautils.notebook import *

import pickle
from dataclasses import dataclass, field
Head = Tuple[int, int]

NEGATIVE_HEADS = [(10, 7), (11, 10)]

ST_HTML_PATH = "/home/ubuntu/Transformerlens/transformer_lens/rs/callum/streamlit/explore_prompts/"

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
        self.data = data or {} # ! bad practice to have default arguments be dicts

    def __getitem__(self, layer_and_head) -> Tensor:
        return self.data[layer_and_head].clone()
    
    def __setitem__(self, layer_and_head, value):
        self.data[layer_and_head] = value.clone()
    
    def size(self):
        if len(self.data) == 0: return 0
        element_size = next(iter(self.data.values())).element_size()
        numel = sum([v.numel() for v in self.data.values()])
        return element_size * numel


@dataclass(frozen=False)
class LogitResults:
    zero_patched: HeadResults = HeadResults()
    mean_patched: HeadResults = HeadResults()
    zero_direct: HeadResults = HeadResults()
    mean_direct: HeadResults = HeadResults()

    def size(self):
        return sum([v.size() for v in [self.zero_patched, self.mean_patched, self.zero_direct, self.mean_direct]])


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

    def save(self, filename: str):
        # Saves self as pickle file
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def items(self):
        return self.__dict__.items()

    def print_sizes(self):
        for k, v in self.items():
            if isinstance(v, Tensor): print(f"{k}: {v.element_size() * v.numel()}")
            else: print(f"{k}: {v.size()}")



def get_data_dict(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    negative_heads: List[Tuple[int, int]],
    use_cuda: bool = False,
):
    model.reset_hooks(including_permanent=True)
    t.cuda.empty_cache()

    current_device = str(next(iter(model.parameters())).device)
    if use_cuda and current_device == "cpu":
        model = model.cuda()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cpu()

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

    if use_cuda and current_device == "cpu":
        model = model.cpu()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cuda()

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

def generate_4_html_plots(
    model: HookedTransformer,
    data_toks: Float[Int, "batch seq_len"],
    data_str_toks_parsed: List[List[str]],
    negative_heads: List[Tuple[int, int]] = NEGATIVE_HEADS,
    save_files: bool = False,
    display_plot_for_batch_idx: Optional[int] = None
):
    def to_string(toks):
        s = model.to_string(toks)
        s = s.replace("\n", "\\n")
        return s

    BATCH_SIZE = data_toks.shape[0]

    MODEL_RESULTS = get_data_dict(model, data_toks, negative_heads = negative_heads)


    # ! (1) Calculate the loss diffs from ablating

    loss_diffs = t.stack([
        t.stack(list(MODEL_RESULTS.loss.mean_direct.data.values())),
        t.stack(list(MODEL_RESULTS.loss.zero_direct.data.values())),
        t.stack(list(MODEL_RESULTS.loss.mean_patched.data.values())),
        t.stack(list(MODEL_RESULTS.loss.zero_patched.data.values())),
    ]) - MODEL_RESULTS.loss_orig

    s_loss = ""

    for batch_idx in tqdm(range(BATCH_SIZE)):
        for head_idx, (layer, head) in enumerate(negative_heads):

            loss_diffs_padded = t.concat([loss_diffs[:, head_idx], t.zeros((4, BATCH_SIZE, 1))], dim=-1)
            loss_diffs_padded = einops.rearrange(
                loss_diffs_padded, 
                "direct_vs_patched_then_mean_vs_zero batch seq -> batch seq direct_vs_patched_then_mean_vs_zero",
            )

            for i, ablation_type in enumerate(["mean, direct", "zero, direct", "mean, patched", "zero, patched"]):
                html = cv.activations.text_neuron_activations(
                    tokens = data_str_toks_parsed[batch_idx],
                    activations = [loss_diffs_padded[batch_idx, :, [i], None]],
                    first_dimension_name = "Ablation (mean / zero)",
                    first_dimension_labels = [ablation_type.split(", ")[0]],
                    second_dimension_name = "Ablation (direct / patched)",
                    second_dimension_labels = [ablation_type.split(", ")[1]],
                )
                s_loss += str(html) + "\n</div>\n\n"

    if save_files:
        with open(ST_HTML_PATH + "loss_difference_from_ablating.html", "w") as file:
            file.write(s.strip())
    if display_plot_for_batch_idx is not None:
        loss_idx = 4 * len(negative_heads) * display_plot_for_batch_idx
        display(HTML(s_loss.strip().split("\n</div>\n\n")[loss_idx]))


    # ! (2) Calculate the logits & direct logit attributions

    s_orig = ""
    s_ablated = ""
    s_direct = ""

    for batch_idx in tqdm(range(BATCH_SIZE)):
        html_orig = cv.logits.token_log_probs(
            data_toks[batch_idx].cpu(),
            MODEL_RESULTS.logits_orig[batch_idx].log_softmax(-1),
            top_k = 5,
            to_string = to_string,
        )
        s_orig += str(html_orig) + "\n</div>\n\n"

        for (layer, head) in negative_heads:

            # Save original log probs
            head_name = f"{layer}{head}"

            # Save new log probs (post-ablation)
            for ablation_type in ["mean, direct", "zero, direct", "mean, patched", "zero, patched"]:
                html_ablated = cv.logits.token_log_probs(
                    data_toks[batch_idx].cpu(),
                    getattr(MODEL_RESULTS.logits, ablation_type.replace(", ", "_"))[layer, head][batch_idx].log_softmax(-1),
                    top_k = 5,
                    to_string = to_string,
                )
                s_ablated += str(html_ablated) + "\n</div>\n\n"
            
            # Save direct logit effect
            html_thishead = cv.logits.token_log_probs(
                data_toks[batch_idx].cpu(),
                MODEL_RESULTS.direct_effect[layer, head][batch_idx].log_softmax(-1),
                to_string = to_string,
                top_k = 5,
                negative = True,
                sub_mean = True,
            )
            s_direct += str(html_thishead) + "\n</div>\n\n"

    if save_files:
        for filename, s in zip(["orig", "ablated", "direct"], [s_orig, s_ablated, s_direct]):
            with open(ST_HTML_PATH + f"logits_{filename}.html", "w") as file:
                file.write(s.strip())
    if display_plot_for_batch_idx is not None:
        orig_idx = display_plot_for_batch_idx
        ablated_idx = len(negative_heads) * display_plot_for_batch_idx * 4
        direct_idx = len(negative_heads) * display_plot_for_batch_idx
        display(HTML(s_orig.strip().split("\n</div>\n\n")[orig_idx]))
        display(HTML(s_ablated.strip().split("\n</div>\n\n")[ablated_idx]))
        display(HTML(s_direct.strip().split("\n</div>\n\n")[direct_idx]))
    

    # ! (2) Calculate the attention probs

    s_attn = ""

    for batch_idx in tqdm(range(BATCH_SIZE)):

        for layer, head in negative_heads:
            head_name = f"{layer}.{head}"

            attn = MODEL_RESULTS.pattern[layer, head][batch_idx]
            weighted_attn = einops.einsum(
                MODEL_RESULTS.pattern[layer, head][batch_idx],
                MODEL_RESULTS.out_norm[layer, head][batch_idx] / MODEL_RESULTS.out_norm[layer, head][batch_idx].max(),
                "seqQ seqK, seqK -> seqQ seqK"
            )

            for vis_type in (cv.attention.attention_heads, cv.attention.attention_patterns):
                html = vis_type(
                    attention = attn.unsqueeze(0), # (heads=2, seqQ, seqK)
                    tokens = data_str_toks_parsed[batch_idx], # list of length seqQ
                    attention_head_names = [head_name],
                )
                s_attn += str(html) + "\n</div>\n\n"
                html_weighted = vis_type(
                    attention = weighted_attn.unsqueeze(0), # (heads=2, seqQ, seqK)
                    tokens = data_str_toks_parsed[batch_idx], # list of length seqQ
                    attention_head_names = [head_name],
                )
                s_attn += str(html_weighted) + "\n</div>\n\n"

    if save_files:
        with open(ST_HTML_PATH + "attn_patterns.html", "w") as file:
            file.write(s_attn)
    if display_plot_for_batch_idx is not None:
        num_plot_types = 2
        num_attn_types = 2
        attn_idx = display_plot_for_batch_idx * len(negative_heads) * (num_plot_types * num_attn_types) + 3
        display(HTML(s_attn.strip().split("\n</div>\n\n")[attn_idx]))


    # ! In this case, we're generating the plots from the "example" that someone types into Streamlit

    if not(save_files) and (display_plot_for_batch_idx is None):
        assert BATCH_SIZE == 1
        return (s_loss, s_orig, s_ablated, s_direct, s_attn)
        
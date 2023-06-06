import json
import torch
from pprint import pprint
import gc

from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.induction.utils import get_all_induction_things
from acdc.ioi.utils import get_all_ioi_things
from acdc.tracr_task.utils import get_all_tracr_things, get_tracr_model_input_and_tl_model
from acdc.docstring.utils import get_all_docstring_things, AllDataThings

torch.set_grad_enabled(False)

DEVICE = "cuda"

def scramble_sd(seed, things: AllDataThings, name, scramble_heads=True, scramble_mlps=True, scramble_head_outputs: bool=False):
    model = things.tl_model
    old_sd = model.state_dict()
    n_heads = model.cfg.n_heads
    n_neurons = model.cfg.d_mlp
    d_head = model.cfg.d_head
    torch.manual_seed(seed)

    if scramble_head_outputs and not scramble_heads:
        raise NotImplementedError

    sd = {}
    for k, v in old_sd.items():
        if scramble_heads and ("attn" in k and not (k.endswith("_O") or k.endswith("mask") or k.endswith("IGNORE"))):
            assert v.shape[0] == n_heads
            to_sd = v[torch.randperm(n_heads), ...].contiguous()
            if scramble_head_outputs:
                to_sd = to_sd[..., torch.randperm(d_head)].contiguous()
            sd[k] = to_sd
        elif scramble_mlps and ("mlp" in k):
            if k.endswith("W_in"):
                assert v.shape[1] == n_neurons
                sd[k] = v[:, torch.randperm(n_neurons)].contiguous()
            elif k.endswith("b_in"):
                assert v.shape[0] == n_neurons
                sd[k] = v[torch.randperm(n_neurons)].contiguous()
            else:
                # print(f"Leaving {k} intact")
                sd[k] = v
        else:
            # print(f"Leaving {k} intact")
            sd[k] = v
    assert sd.keys() == old_sd.keys()
    for k in sd.keys():
        assert sd[k].shape == old_sd[k].shape

    torch.save({k: v.cpu() for k, v in sd.items()}, name + ".pt")

    def all_metrics(logits):
        m = {}
        for k, v in things.test_metrics.items():
            m[k] = v(logits).item()
        return m

    metrics = {}
    metrics["trained_orig"] = all_metrics(model(things.test_data))
    metrics["trained_patched"] = all_metrics(model(things.test_patch_data))

    model.load_state_dict(sd)
    metrics["reset_orig"] = all_metrics(model(things.test_data))
    metrics["reset_patched"] = all_metrics(model(things.test_patch_data))

    with open(name + "_test_metrics.json", "w") as f:
        json.dump(metrics, f)
    pprint(metrics)


# %% IOI

things = get_all_ioi_things(num_examples=100, device=DEVICE, metric_name="kl_div")
scramble_sd(1504304416, things, "ioi_reset_heads_neurons")
del things
gc.collect()

# %% Tracr-reverse
things = get_all_tracr_things(task="reverse", metric_name="kl_div", num_examples=6, device=DEVICE)
scramble_sd(1207775456, things, "tracr_reverse_reset_heads_head_outputs_neurons", scramble_head_outputs=True)
gc.collect()

scramble_sd(1666927681, things, "tracr_reverse_reset_heads_neurons", scramble_head_outputs=False)
del things
gc.collect()

# %% Tracr-proportion

things = get_all_tracr_things(task="proportion", metric_name="kl_div", num_examples=50, device=DEVICE)
scramble_sd(2126292961, things, "tracr_proportion_reset_heads_head_outputs_neurons", scramble_head_outputs=True)
gc.collect()

scramble_sd(913070797, things, "tracr_proportion_reset_heads_neurons", scramble_head_outputs=False)
del things
gc.collect()

# %% Induction

things = get_all_induction_things(num_examples=50, seq_len=300, device=DEVICE, metric="kl_div")
scramble_sd(2016630123, things, "induction_reset_heads_neurons")
del things
gc.collect()

# %% Docstring

things = get_all_docstring_things(num_examples=50, seq_len=2, device=DEVICE, metric_name="kl_div", correct_incorrect_wandb=False)
scramble_sd(814220622, things, "docstring_reset_heads_neurons")
del things
gc.collect()


# %% GreaterThan

things = get_all_greaterthan_things(num_examples=100, device=DEVICE, metric_name="kl_div")
scramble_sd(1028419464, things, "greaterthan_reset_heads_neurons")
del things
gc.collect()

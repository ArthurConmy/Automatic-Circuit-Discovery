from functools import partial
import time
import torch
from typing import Literal
from transformer_lens.HookedTransformer import HookedTransformer, HookedTransformerConfig
from acdc.docstring.utils import AllDataThings
from acdc.tracr_task.utils import get_perm

MAX_LOGIC_GATE_SEQ_LEN = 100_000 # Can be increased further provided numerics and memory do not explode

def get_logic_gate_model(mode: Literal["OR", "AND"] = "OR", seq_len=1, device="cuda") -> HookedTransformer:

    assert 1 <= seq_len <= MAX_LOGIC_GATE_SEQ_LEN, "We need some bound on sequence length, but this can be increased if the "

    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 1, 
            "d_model": 3 if mode == "AND" else 2,
            "n_ctx": seq_len,
            "n_heads": 1,
            "d_head": 1,
            "act_fn": "relu",
            "d_vocab": 2,
            "d_mlp": 1,
            "d_vocab_out": 1,
            "normalization_type": None,
            # "attention_dir": "bidirectional",
            "attn_only": (mode == "OR"),

        }
    )

    model = HookedTransformer(cfg).to(device)
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in model.cfg.to_dict():
        model.set_use_hook_mlp_in(True)

    model = model.to(torch.double)

    # Turn off model gradient so we can edit weights
    # And also set all the weights to 0
    for param in model.parameters():
        param.requires_grad = False
        param[:] = 0.0

    if mode == "AND":

        # # Embed 1s as 1.0 in residual component 0
        model.embed.W_E[1, 0] = 1.0

        # No QK so uniform attention; this allows us to detect if everything is a 1 as the output into the channel 1 will be 1 not less than that

        # Output 1.0 into residual component 1 for all things present
        model.blocks[0].attn.W_V[0, 0, 0] = 1.0 # Shape [head_index d_model d_head]
        model.blocks[0].attn.W_O[0, 0, 1] = 1.0 # Shape [head_index d_head d_model]
        
        model.blocks[0].mlp.W_in[1, 0] = 1.0 # [d_model d_mlp]
        model.blocks[0].mlp.b_in[:] = -(MAX_LOGIC_GATE_SEQ_LEN-1)/MAX_LOGIC_GATE_SEQ_LEN # Unless everything in input is a 1, do not fire

        # Write the output to residual component 2
        # (TODO: I think we could get away with 2 components here?)
        model.blocks[0].mlp.W_out[0, 2] = MAX_LOGIC_GATE_SEQ_LEN # Shape [d_mlp d_model]

        model.unembed.W_U[2, 0] = 1.0 # Shape [d_model d_vocab_out]

    elif mode == "OR":
        model.embed.W_E[1, 0] = 1.0

        # If there are 1s present, attend to them only
        model.blocks[0].attn.W_Q[0, 0, 0] = 0.0
        model.blocks[0].attn.b_Q[0] = 1.0
        model.blocks[0].attn.W_K[0, 0, 0] = int(1e9)

        # Write 1 to residual component 1 if any 1s are present
        model.blocks[0].attn.W_V[0, 0, 0] = 1.0 # Shape [head_index d_model d_head]
        model.blocks[0].attn.W_O[0, 0, 1] = 1.0 # Shape [head_index d_head d_model]

        model.unembed.W_U[1, 0] = 1.0 # Shape [d_model d_vocab_out]

    else:
        raise ValueError(f"mode {mode} not recognized")

    return model

def test_logical_models():
    """
    Test that the OR and AND mod
    """
    
    seq_len=3
    and_model = get_logic_gate_model(mode="AND", seq_len=seq_len, device = "cpu")

    all_inputs = []
    for i in range(2**seq_len):
        input = torch.tensor([int(x) for x in f"{i:03b}"]).unsqueeze(0).long()
        all_inputs.append(input)
    input = torch.cat(all_inputs, dim=0)

    and_output = and_model(input)[:, -1, :]
    assert torch.equal(and_output[:2**seq_len - 1], torch.zeros(2**seq_len - 1, 1))
    torch.testing.assert_close(and_output[2**seq_len - 1], torch.ones(1))

    or_model = get_logic_gate_model(mode="OR", seq_len=seq_len, device = "cpu")
    or_output = or_model(input)[:, -1, :]

    torch.testing.assert_close(or_output[1:], torch.ones(2**seq_len - 1, 1))
    assert torch.equal(or_output[0], torch.zeros(1))

def get_all_logic_gate_things(mode: str = "AND", device=None, seq_len = 5, num_examples = 10):

    assert mode in ["AND", "OR"]
    # Create a set to hold our unique binary strings
    unique_binary_strings = set()

    assert num_examples <= 2**seq_len, "We can't have more examples than there are possible binary strings"

    start_time = time.time()

    # Keep adding binary strings until we have enough examples
    while len(unique_binary_strings) < num_examples and time.time() - start_time < 20:
        binary_string = torch.randint(0, 2, (seq_len,)).to(device)
        binary_str_representation = "".join(str(i) for i in binary_string.tolist())  # convert tensor to string
        unique_binary_strings.add(binary_str_representation)

    if len(unique_binary_strings) < num_examples:
        raise ValueError(f"Could not generate enough unique binary strings in {time.time() - start_time} seconds")

    # Convert the set of binary string back llto tensor
    data = []
    for bin_str in unique_binary_strings:
        data.append([int(ch) for ch in bin_str])
    data = torch.tensor(data).to(device)

    rand_perm = get_perm(num_examples).to(device)
    randomised_data = data[rand_perm]

    model = get_logic_gate_model(mode=mode, seq_len=seq_len, device = device)

    correct_answers = model(data)[:, -1, :]

    def validation_metric(output, correct):
        output = output[:, -1, :]
        assert output.shape == correct.shape
        return torch.mean((output - correct)**2)

    return AllDataThings(
        tl_model=model,
        validation_metric=partial(validation_metric, correct=correct_answers),
        validation_data=data,
        validation_labels=None,
        validation_mask=None,
        validation_patch_data=randomised_data,
        test_metrics=None,
        test_data=None,
        test_labels=None,
        test_mask=None,
        test_patch_data=None,
    )
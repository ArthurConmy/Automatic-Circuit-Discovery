#%%

from functools import partial
import time
import torch
from typing import Literal, Optional
from transformer_lens.HookedTransformer import HookedTransformer, HookedTransformerConfig
from acdc.docstring.utils import AllDataThings
from acdc.tracr_task.utils import get_perm

MAX_LOGIC_GATE_SEQ_LEN = 100_000 # Can be increased further provided numerics and memory do not explode

def get_logic_gate_model(mode: Literal["OR", "AND"] = "OR", seq_len: Optional[int]=None, device="cuda") -> HookedTransformer:

    if seq_len is None:
        assert 1 <= seq_len <= MAX_LOGIC_GATE_SEQ_LEN, "We need some bound on sequence length, but this can be increased if the variable at the top is increased"

    if mode == "OR":
        assert seq_len == 1
        cfg = HookedTransformerConfig.from_dict(
            {
                "n_layers": 1, 
                "d_model": 2,
                "n_ctx": 1,
                "n_heads": 2,
                "d_head": 1,
                "act_fn": "relu",
                "d_vocab": 1,
                "d_mlp": 1,
                "d_vocab_out": 1,
                "normalization_type": None,
                "attn_only": False,
            }
        )
    elif mode == "AND":
        cfg = HookedTransformerConfig.from_dict(
            {
                "n_layers": 1, 
                "d_model": 3,
                "n_ctx": seq_len,
                "n_heads": 1,
                "d_head": 1,
                "act_fn": "relu",
                "d_vocab": 2,
                "d_mlp": 1,
                "d_vocab_out": 1,
                "normalization_type": None,
            }
        )
    else:
        raise ValueError(f"mode {mode} not recognized")

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
        model.embed.W_E[0, 0] = 0.0 # ... ?

        # Both heads dump a 1 into the residual stream
        model.blocks[0].attn.b_V[:, 0] = 1.0 # [num_heads, d_head]
        model.blocks[0].attn.W_O[:, 0, 0] = 1.0 # [num_heads, d_head, d_model]

        model.blocks[0].mlp.W_in[0, 0] = -1.0 # [d_model d_mlp]
        model.blocks[0].mlp.b_in[:] = 1.0 # [d_mlp]
        
        model.blocks[0].mlp.W_out[0, 1] = -1.0 # 0 into -1 here
        model.blocks[0].mlp.b_out[:] = 1.0 # [d_model]

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
    torch.testing.assert_close(and_output[2**seq_len - 1], torch.ones(1).to(torch.double))

    for or_model in [get_logic_gate_model(mode="OR", seq_len=seq_len, device = "cpu"), get_or_with_trivial_attn_0(seq_len=seq_len, device = "cpu")]:
        or_output = or_model(input)[:, -1, :]

        torch.testing.assert_close(or_output[1:], torch.ones(2**seq_len - 1, 1).to(torch.double))
        assert torch.equal(or_output[0], torch.zeros(1)), f"or_output[0] = {or_output}"

#%%

def get_all_logic_gate_things(mode: str = "AND", device=None, seq_len: Optional[int] = 5, num_examples: Optional[int] = 10):

    assert mode == "OR" 

    model = get_logic_gate_model(mode=mode, seq_len=seq_len, device=device)
    # Convert the set of binary string back llto tensor
    data = torch.tensor([[0.0]]).long() # Input is actually meaningless, all that matters is Attention Heads 0 and 1
    correct_answers = data.clone().to(torch.double) + 1

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
        validation_patch_data=data.clone(), # We're doing zero ablation so irrelevant
        test_metrics=None,
        test_data=None,
        test_labels=None,
        test_mask=None,
        test_patch_data=None,
    )


# # # test_logical_models()
# # %%

# or_model = get_logic_gate_model(seq_len=1, device = "cpu")
# logits, cache = or_model.run_with_cache(
#     torch.tensor([[0]]).to(torch.long),
# )
# print(logits)

# # %%

# for key in cache.keys():
#     print(key)
#     print(cache[key].shape)
#     print(cache[key])
#     print("\n\n\n")
# # %%
# #batch pos head_index d_head for hook_q
# %%

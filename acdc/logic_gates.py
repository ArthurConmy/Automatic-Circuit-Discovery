#%%

import torch
from typing import Literal
from transformer_lens.HookedTransformer import HookedTransformer, HookedTransformerConfig
from acdc.docstring.utils import AllDataThings

MAX_LOGIC_GATE_SEQ_LEN = 100_000 # Can be increased further provided numerics and memory do not explode

def get_logic_gate_model(mode: Literal["OR", "AND"] = "OR", seq_len=1, device="cuda") -> HookedTransformer:

    assert 1 <= seq_len <= MAX_LOGIC_GATE_SEQ_LEN, "We need some bound on sequence length, but this can be increased if the "

    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 1, 
            "d_model": 3 if mode == "OR" else 2,
            "n_ctx": seq_len,
            "n_heads": 1,
            "d_head": 1,
            "act_fn": "relu",
            "d_vocab": 2,
            "d_mlp": 1,
            "d_vocab_out": 1,
            "normalization_type": None,
            "attention_dir": "bidirectional",
            "attn_only": (mode == "AND"),
        }
    )

    model = HookedTransformer(cfg).to(device)
    model = model.to(torch.double)

    # Turn off model gradient so we can edit weights
    # And also set all the weights to 0
    for param in model.parameters():
        param.requires_grad = False
        param[:] = 0.0

    if mode == "OR":

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

    elif mode == "AND":
        # Do: if we picked up on ANY 0 then kill everything. Else all good
        # Okay sure how do we convert to 1 and 0, though

        # Confusingly, embed the 0s as 1.0
        model.embed.W_E[0, 0] = 1.0

        # If there are 0s present, attend to them only
        model.blocks[0].attn.W_Q[0, 0, 0] = 0.0
        model.blocks[0].attn.b_Q[0] = 1.0
        model.blocks[0].attn.W_K[0, 0, 0] = int(1e9)

        # Write 0 to residual component 1 if 0s are present. Else 1
        model.blocks[0].attn.W_V[0, 0, 0] = 1.0 # Shape [head_index d_model d_head]
        model.blocks[0].attn.W_O[0, 0, 1] = -1.0 # Shape [head_index d_head d_model]
        model.blocks[0].attn.b_O[1] += 1

        model.unembed.W_U[1, 0] = 1.0 # Shape [d_model d_vocab_out]

    else:
        raise ValueError(f"mode {mode} not recognized")

    return model

# %%

model = get_logic_gate_model(mode="AND", seq_len=2, device="cpu")

# %%

model_out, cache = model.run_with_cache(torch.tensor([[1, 0]]))
print(model_out[:, 0, :])

# %%

for key in cache:
    print(key, "\n", cache[key].shape, "\n", cache[key])

# %%

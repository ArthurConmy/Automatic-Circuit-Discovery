# %% [markdown]
# <h1>General tutorial</h1>
# <p>This notebook gives a high-level overview of the main abstractions used in the ACDC codebase.</p>
# <p>If you are interested in models that are >10x the size of GPT-2 small, this library currently may be too slow and we would recommend you look at the path patching implementations in `TransformerLens` (forthcoming)</p>

# %% [markdown]
# <h2> Imports etc</h2>

from IPython import get_ipython

from acdc.acdc_utils import TorchIndex
if get_ipython() is not None:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

from acdc.HookedTransformer import HookedTransformer
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.induction.utils import get_all_induction_things

# %%

num_examples = 40
seq_len = 300

# load in a tl_model and grab some data
tl_model, toks_int_values, toks_int_values_other, metric, mask_rep = get_all_induction_things(
    num_examples=num_examples, seq_len=seq_len, device="cuda", randomize_data=False, return_mask_rep=True,
)

# You should read the get_model function from that file to see what the Redwood model is : ) 

#%%

# Let's see an example from the dataset:
# | separates tokens
print(
    "|".join(tl_model.to_str_tokens(toks_int_values[33, :36])),
) 
# This has several examples of induction! F -> #, mon -> ads 

# The `mask_rep` mask is a boolean mask of shape (num_examples, seq_len) that indicates where induction is present in the dataset
print()
for i in range(36):
    if mask_rep[33, i]:
        print(f"At position {i} there is induction")
        print(tl_model.to_str_tokens(toks_int_values[33:34, i:i+1]))


#%%

# Let's get the initial loss on the induction examples

def get_loss(model, data, mask):
    loss = model(
        data,
        return_type="loss",
        loss_per_token=True, 
    )
    return (loss * mask[:,:-1].int()).sum() / mask[:,:-1].int().sum()

print(f"Loss: {get_loss(tl_model, toks_int_values, mask_rep)}")

#%%

# We wrap ACDC things inside an `experiment`for further experiments
experiment = TLACDCExperiment(
    model=tl_model,
    threshold=0.0,
    ds=toks_int_values,
    ref_ds=None, # This is the corrupted dataset from the ACDC paper. We're going to do zero ablation here so we omit this
    metric=metric,
    zero_ablation=True,
)

# %%

# Usually, we efficiently add hooks to the model in order to do ACDC runs. For this tutorial, we'll add all the hooks so you can edit connections in the model as easily as possible

experiment.model.reset_hooks()
experiment.setup_model_hooks(
    add_sender_hooks=True,
    add_receiver_hooks=True,
    doing_acdc_runs=False,
)

# %%

# Let's take a look at the edges
for edge_indices, edge in experiment.corr.all_edges().items():
    
    # here's what's inside the edge
    receiver_name, receiver_index, sender_name, sender_index = edge_indices

    print(sender_name, sender_index, receiver_name, receiver_index, edge)

    # for now, all edges should be present
    assert edge.present, edge_indices

#%%

# Let's make a function that's able to turn off all the connections from the nodes to the output, excecpt the induction head (1.5 and 1.6)

# (we'll later turn on all connections EXCEPT the induction heads)

def change_direct_output_connections(exp, invert=False):
    residual_stream_end_name = "blocks.1.hook_resid_post"
    residual_stream_end_index = TorchIndex([None])
    induction_heads = [
        ("blocks.1.attn.hook_result", TorchIndex([None, None, 5])), 
        ("blocks.1.attn.hook_result", TorchIndex([None, None, 6])),
    ]

    inputs_to_residual_stream_end = exp.corr.edges[residual_stream_end_name][residual_stream_end_index]
    for sender_name in inputs_to_residual_stream_end:
        for sender_index in inputs_to_residual_stream_end[sender_name]:

            is_induction_head = (sender_name, sender_index) in induction_heads
            if is_induction_head:
                edge.present = invert 

            else:
                edge.present = not invert

            print(f"{['Adding', 'Rsemoving'][int(invert == is_induction_head)]} edge from {sender_name} {sender_index} to {residual_stream_end_name} {residual_stream_end_index}")

change_direct_output_connections(experiment)
print("Loss with only the induction head direct connections:", )
# %%


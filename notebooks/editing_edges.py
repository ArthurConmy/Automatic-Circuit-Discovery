# %% [markdown]
# <h1>ACDC Editing Edges Demo</h1>
#
# <p>This notebook gives a high-level overview of the main abstractions used in the ACDC codebase.</p>
#
# <p>If you are interested in models that are >=1B parameters, this library currently may be too slow and we would recommend you look at the path patching implementations in `TransformerLens` (for example, see <a href="https://colab.research.google.com/drive/15CJ1WAf8AWm6emI3t2nVfnO85-hxwyJU">this</a> notebook)</p>
#
# <h3>Setup</h2>
#
# <p>Janky code to do different setup when run in a Colab notebook vs VSCode (adapted from e.g <a href="https://github.com/neelnanda-io/TransformerLens/blob/5c89b7583e73ce96db5e46ef86a14b15f303dde6/demos/Activation_Patching_in_TL_Demo.ipynb">this notebook</a>)</p>
# 
# <p>You can ignore warnings that "packages were previously imported in this runtime"</p>

#%%

try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")

    import subprocess # to install graphviz dependencies
    command = ['apt-get', 'install', 'graphviz-dev']
    subprocess.run(command, check=True)

    from IPython import get_ipython
    ipython = get_ipython()

    ipython.run_line_magic( # install ACDC
        "pip",
        "install git+https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git@d89f7fa9cbd095202f3940c889cb7c6bf5a9b516",
    )

except Exception as e:
    IN_COLAB = False
    print("Running outside of Colab notebook")

    import numpy # crucial to not get cursed error
    import plotly

    plotly.io.renderers.default = "colab"  # added by Arthur so running as a .py notebook with #%% generates .ipynb notebooks that display in colab
    # disable this option when developing rather than generating notebook outputs

    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        print("Running as a notebook")
        ipython.run_line_magic("load_ext", "autoreload")  # type: ignore
        ipython.run_line_magic("autoreload", "2")  # type: ignore
    else:
        print("Running as a .py script")

# %% [markdown]
# <h2>Imports etc</h2>

#%%

from transformer_lens.HookedTransformer import HookedTransformer
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.induction.utils import get_all_induction_things
from acdc.acdc_utils import TorchIndex
import torch
import gc

# %% [markdown]
# <h2>Load in the model and data for the induction task

#%%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_examples = 40
seq_len = 50

# load in a tl_model and grab some data
all_induction_things = get_all_induction_things(
    num_examples=num_examples,
    seq_len=seq_len,
    device=DEVICE,
)

tl_model, toks_int_values, toks_int_values_other, metric, mask_rep = (
    all_induction_things.tl_model,
    all_induction_things.validation_data,
    all_induction_things.validation_patch_data,
    all_induction_things.validation_metric,
    all_induction_things.validation_mask,
)

# You should read the get_model function from that file to see what the tl_model is : )

# %% [markdown]
# <p>Ensure we stay under mem limit on small machines</p>

#%%
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# <p>Let's see an example from the dataset.</p>
# <p> `|` separates tokens </p>

#%%
EXAMPLE_NO = 33
EXAMPLE_LENGTH = 36

print(
    "|".join(tl_model.to_str_tokens(toks_int_values[EXAMPLE_NO, :EXAMPLE_LENGTH])),
)

#%% [markdown]
# <p>This dataset has several examples of induction! F -> #, mon -> ads</p>
# <p>The `mask_rep` mask is a boolean mask of shape `(num_examples, seq_len)` that indicates where induction is present in the dataset</p>
# <p> Let's see 

#%%
for i in range(EXAMPLE_LENGTH):
    if mask_rep[EXAMPLE_NO, i]:
        print(f"At position {i} there is induction")
        print(tl_model.to_str_tokens(toks_int_values[EXAMPLE_NO:EXAMPLE_NO+1, i : i + 1]))

# %% [markdown]
# <p>Let's get the initial loss on the induction examples</p>

#%%
def get_loss(model, data, mask):
    loss = model(
        data,
        return_type="loss",
        loss_per_token=True,
    )
    return (loss * mask[:, :-1].int()).sum() / mask[:, :-1].int().sum()


print(f"Loss: {get_loss(tl_model, toks_int_values, mask_rep)}")

#%% [markdown]
#<p>We will now wrap ACDC things inside an `experiment`for further experiments</p>
# <p>For more advanced usage of the `TLACDCExperiment` object (the main object in this codebase), see the README for links to the `main.py` and its demos</p>

#%%
experiment = TLACDCExperiment(
    model=tl_model,
    threshold=0.0,
    ds=toks_int_values,
    ref_ds=None,  # This argument is the corrupted dataset from the ACDC paper. We're going to do zero ablation here so we omit this
    metric=metric,
    zero_ablation=True,
    hook_verbose=False,
)

# %% [markdown]

# <p>Usually, the `TLACDCExperiment` efficiently add hooks to the model in order to do ACDC runs fast.</p>
# <p>For this tutorial, we'll add <b>ALL</b> the hooks so you can edit connections in the model as easily as possible.</p>

#%%
experiment.model.reset_hooks()
experiment.setup_model_hooks(
    add_sender_hooks=True,
    add_receiver_hooks=True,
    doing_acdc_runs=False,
)

# %% [markdown]
# Let's take a look at the edges

#%%
for edge_indices, edge in experiment.corr.all_edges().items():
    # here's what's inside the edge
    receiver_name, receiver_index, sender_name, sender_index = edge_indices

    # for now, all edges should be present
    assert edge.present, edge_indices

# %% [markdown]
# <p>Let's make a function that's able to turn off all the connections from the nodes to the output, except the induction head (1.5 and 1.6)</p>
# <p>(we'll later turn ON all connections EXCEPT the induction heads)</p>

#%%
def change_direct_output_connections(exp, invert=False):
    residual_stream_end_name = "blocks.1.hook_resid_post"
    residual_stream_end_index = TorchIndex([None])
    induction_heads = [
        ("blocks.1.attn.hook_result", TorchIndex([None, None, 5])),
        ("blocks.1.attn.hook_result", TorchIndex([None, None, 6])),
    ]

    inputs_to_residual_stream_end = exp.corr.edges[residual_stream_end_name][
        residual_stream_end_index
    ]
    for sender_name in inputs_to_residual_stream_end:
        for sender_index in inputs_to_residual_stream_end[sender_name]:
            edge = inputs_to_residual_stream_end[sender_name][sender_index]
            is_induction_head = (sender_name, sender_index) in induction_heads

            if is_induction_head:
                edge.present = not invert

            else:
                edge.present = invert

            print(
                f"{'Adding' if (invert == is_induction_head) else 'Removing'} edge from {sender_name} {sender_index} to {residual_stream_end_name} {residual_stream_end_index}"
            )


change_direct_output_connections(experiment)
print(
    "Loss with only the induction head direct connections:",
    get_loss(experiment.model, toks_int_values, mask_rep).item(),
)

# %% [markdown]
# <p>Let's turn ON all the connections EXCEPT the induction heads</p>

#%%
change_direct_output_connections(experiment, invert=True)
print(
    "Loss without the induction head direct connections:",
    get_loss(experiment.model, toks_int_values, mask_rep).item(),
)

#%% [markdown]
# <p>That's much larger!</p>
# <p>See acdc/main.py for how to run ACDC experiments; try `python acdc/main.py --help` or check the README for the links to this file</p>

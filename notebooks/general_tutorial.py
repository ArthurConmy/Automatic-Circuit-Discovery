# %% [markdown]
# <h1>General tutorial</h1>
# <p>This notebook gives a high-level overview of the main abstractions used in the ACDC codebase.</p>
# <p>If you are interested in models that are bigger than GPT-2 small, this library currently may be too slow and we would recommend you look at the path patching implementations in `TransformerLens` (forthcoming)</p>

# %% [markdown]
# <h2> Imports etc</h2>

from IPython import get_ipython

if get_ipython() is not None:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

from acdc.HookedTransformer import HookedTransformer
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.induction.utils import get_all_induction_things

# %%

num_examples = 400
seq_len = 30

# load in a tl_model and grab some data
tl_model, toks_int_values, toks_int_values_other, metric = get_all_induction_things(
    num_examples=num_examples, seq_len=seq_len, device="cuda", randomize_data=False
)

# you should read the get_model function from that file to see what the Redwood model is : ) 

# %%

# we wrap ACDC things inside an `experiment`

experiment = TLACDCExperiment(
    model=model,
    threshold=0.0,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=metric,
)

# %%

experiment.model.reset_hooks()
experiment.setup_model_hooks(
    add_sender_hooks=True,
    add_receiver_hooks=True,
)

# %%

# Let's take a look at the edges

for edge in experiment.corr.all_edges():
    
    # here's what's inside the edge
    receiver_name, receiver_index, sender_name, sender_index = edge

    # here's how to edit and access edges
    experiment.corr.edges[receiver_name][receiver_index][sender_name][sender_index].present = False

# %%

# If you add BACK in the edges from the previous token head to the two induction heads
# (And the edges from input to previous token, and induction to output)
# Then exp.tl_model(toks_int_values) should be reasonable (on the dataset examples of induction)
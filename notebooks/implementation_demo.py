# %% [markdown]
# <h1>ACDC Implementation Demo</h1>
#
# <p>This notebook gives a low-level explanation of how the forward passes and algorithm that iterates over the computational graph work in the ACDC codebase.</p>
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
        "install git+https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git@9d5844a",
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

#%% [markdown]
# 
# <h1> What are the two goals of ACDC? </h1>
#
# <p> In order to motivate the technical details on how we implement editable computational graphs, let's first state the the two goals the computational graph implementation has: </p>
# 
# <p> 1. 
# %%

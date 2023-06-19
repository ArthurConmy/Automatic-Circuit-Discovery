"""
Same as transformer_lens.cautils.utils, but with autoreload magic
"""


import warnings
from IPython import get_ipython
ipython = get_ipython()

if ipython is not None: # so this works as a script and in a notebook
    warnings.warn("Running load_ext autoreload...")
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

from transformer_lens.cautils.utils import *
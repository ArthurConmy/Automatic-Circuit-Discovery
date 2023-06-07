import os
from warnings import warn
import warnings
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')

    __file__ = os.path.join(get_ipython().run_line_magic('pwd', ''), "notebooks", "df_plots_data.py")

    from notebooks.emacs_plotly_render import set_plotly_renderer
    if "adria" in __file__:
        set_plotly_renderer("emacs")

import plotly
import numpy as np
import json
import wandb
from acdc.graphics import dict_merge, pessimistic_auc
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors as pc
from pathlib import Path
import plotly.express as px
import pandas as pd
import argparse

# %%

DATA_DIR = Path(__file__).resolve().parent.parent / "acdc" / "media" / "plots_data"
all_data = {}

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".json"):
        with open(DATA_DIR / fname, "r") as f:
            data = json.load(f)
        dict_merge(all_data, data)



# %% Possibly convert all this data to pandas dataframe

rows = []
for weights_type, v in all_data.items():
    for ablation_type, v2 in v.items():
        for task, v3 in v2.items():
            for metric, v4 in v3.items():
                for alg, v5 in v4.items():
                    for i in range(len(v5["score"])):
                        rows.append(pd.Series({
                            "weights_type": weights_type,
                            "ablation_type": ablation_type,
                            "task": task,
                            "metric": metric,
                            "alg": alg,
                            **{k: val[i] for k, val in v5.items()}}))

df = pd.DataFrame(rows)

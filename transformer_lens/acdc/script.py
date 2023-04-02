#%%

import pickle
import matplotlib.pyplot as plt
import numpy as np

fname = "/mnt/ssd-0/arthurworkspace/TransformerLens/transformer_lens/acdc/histories/Sun_Apr_2_09_22_56_2023.pkl"

with open(fname, "rb") as f:
    history = pickle.load(f)

#%%

import subprocess
import numpy as np 
from math import gcd

START = 0.0
STOP = 2.0

for it in range(3, int(1e6)):
    curspace = np.linspace(start=START, stop=STOP, num=it)
    for threshold_idx, threshold in list(enumerate(curspace))[1:-1]:
        if gcd(threshold_idx, it) != 1:
            continue
        subprocess.run(["python", "acdc.py", "--config", "../../configs_acdc/base_config.yaml", "--threshold", str(threshold), "--zero-ablation"])
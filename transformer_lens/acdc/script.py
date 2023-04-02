import subprocess
import numpy as np 
from math import gcd

START = -3
STOP = 1
div = 2

for it in range(3, int(1e6)):
    curspace = np.logspace(start=START, stop=STOP, num=it) / div
    for threshold_idx, threshold in list(enumerate(curspace))[1:-1]:
        subprocess.run(["python", "acdc.py", "--using-wandb", "--threshold", str(threshold), "--zero-ablation"])